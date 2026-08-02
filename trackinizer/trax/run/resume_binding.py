"""Bind an explicit provider resume to one pre-existing transcript."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Self
from uuid import UUID

import sqlite3

from trackinizer.trax.run.adapters.base import Adapter
from trackinizer.trax.run.adapters.claude import ClaudeAdapter
from trackinizer.trax.run.adapters.codex import CodexAdapter
from trackinizer.trax.run.transcript import (
    TranscriptClaim,
    TranscriptError,
    TranscriptReader,
    TranscriptSnapshot,
    snapshot_transcript,
)


__all__ = [
    "PreparedResume",
    "ResumeBindingError",
    "prepare_explicit_resume",
]


class ResumeBindingError(RuntimeError):
    """An explicit provider resume could not be bound without guessing."""


@dataclass(frozen=True, slots=True)
class _Selection:
    identity: str
    path: Path


class PreparedResume:
    """One pre-launch resume boundary, retained until its child is reaped."""

    def __init__(
        self,
        *,
        adapter: Adapter,
        cli_args: tuple[str, ...],
        identity: str,
        path: Path,
        snapshot: TranscriptSnapshot,
    ) -> None:
        self._adapter = adapter
        self._identity = identity
        self._path = path
        self._snapshot: TranscriptSnapshot | None = snapshot
        self._reader: TranscriptReader | None = None
        self._failure: Exception | None = None
        self._closed = False
        self.cli_args = cli_args

    @property
    def expected_cli_session_id(self) -> str:
        """The explicit native UUID proven before the provider starts."""
        return self._identity

    def poll(self) -> TranscriptReader:
        """Bind the retained pre-launch descriptor on first use."""
        if self._failure is not None:
            raise self._failure
        try:
            return self._poll()
        except Exception as error:
            self._failure = error
            raise

    def _poll(self) -> TranscriptReader:
        if self._closed:
            raise ResumeBindingError("resume binding is closed")
        if self._reader is not None:
            return self._reader
        snapshot = self._snapshot
        if snapshot is None:
            raise ResumeBindingError("resume transcript snapshot is unavailable")
        reader = TranscriptReader.open(
            self._adapter,
            TranscriptClaim(
                cli_session_id=self._identity,
                path=self._path,
                before=snapshot,
            ),
        )
        self._snapshot = None
        self._reader = reader
        return reader

    def finish(self) -> TranscriptReader:
        """Seal the provider's final EOF and return its exact reader."""
        if self._failure is not None:
            raise self._failure
        try:
            reader = self._poll()
            reader.seal()
            return reader
        except Exception as error:
            self._failure = error
            raise

    def close(self) -> None:
        """Release the retained transcript descriptor; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._reader is not None:
            self._reader.close()
        if self._snapshot is not None:
            self._snapshot.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def prepare_explicit_resume(
    adapter: Adapter,
    cli_args: Sequence[str],
    *,
    cwd: Path,
) -> PreparedResume | None:
    """Pin an explicit provider UUID to its current EOF before launch."""
    del cwd
    args = tuple(cli_args)
    if isinstance(adapter, CodexAdapter):
        identity = _select_codex_resume(args)
        if identity is None:
            return None
        selection = _Selection(identity, _resolve_codex(adapter, identity))
    elif isinstance(adapter, ClaudeAdapter):
        identity = _select_claude_resume(args)
        if identity is None:
            return None
        selection = _Selection(identity, _resolve_claude(adapter, identity))
    else:
        return None

    try:
        snapshot = snapshot_transcript(
            adapter,
            selection.path,
            selection.identity,
        )
    except TranscriptError as error:
        raise ResumeBindingError(str(error)) from error
    return PreparedResume(
        adapter=adapter,
        cli_args=args,
        identity=selection.identity,
        path=selection.path,
        snapshot=snapshot,
    )


_CLAUDE_HALT_OPTIONS = frozenset({"-h", "--help", "-v", "--version"})


def _select_claude_resume(args: tuple[str, ...]) -> str | None:
    """Return one explicit Claude resume UUID without changing its argv.

    Claude's ``--resume`` value is optional: a missing value or a non-UUID
    search term opens an interactive picker.  Neither supplies a pre-launch
    transcript identity, so fail closed instead of opening a duplicate
    AgentSession. ``--continue`` has the same ambiguity. ``--fork-session``
    deliberately creates a new native identity and therefore cannot reconcile
    the prior AgentSession.
    """
    identity: str | None = None
    fork_session = False
    continue_session = False
    options_enabled = True
    index = 0
    while index < len(args):
        argument = args[index]
        if options_enabled and argument == "--":
            options_enabled = False
            index += 1
            continue
        if not options_enabled:
            index += 1
            continue
        if argument in _CLAUDE_HALT_OPTIONS:
            return None
        if argument in {"-c", "--continue"}:
            continue_session = True
            index += 1
            continue
        if argument == "--fork-session":
            fork_session = True
            index += 1
            continue

        value: str | None = None
        if argument in {"-r", "--resume"}:
            if index + 1 >= len(args) or args[index + 1].startswith("-"):
                raise ResumeBindingError(
                    "Claude resume requires an explicit session UUID"
                )
            value = args[index + 1]
            index += 2
        elif argument.startswith("--resume="):
            value = argument.partition("=")[2]
            index += 1
        elif argument.startswith("-r") and len(argument) > 2:
            value = argument[2:].removeprefix("=")
            index += 1
        else:
            index += 1
            continue

        selected = _canonical_uuid(value, provider="Claude")
        if identity is not None and selected != identity:
            raise ResumeBindingError("Claude resume received multiple session UUIDs")
        identity = selected

    if identity is None:
        if continue_session:
            raise ResumeBindingError(
                "Claude --continue cannot be bound; use --resume with an explicit UUID"
            )
        return None
    if continue_session:
        raise ResumeBindingError("Claude resume cannot also use --continue")
    if fork_session:
        raise ResumeBindingError(
            "Claude --fork-session creates a new identity and cannot reconcile a resume"
        )
    return identity


def _resolve_claude(adapter: ClaudeAdapter, identity: str) -> Path:
    """Resolve Claude's UUID filename across its project transcript roots."""
    roots = tuple(adapter.session_dirs())
    if len(roots) != 1:
        raise ResumeBindingError("Claude transcript root is unavailable")
    projects = roots[0].absolute()
    try:
        matches = list(projects.glob(f"*/{identity}.jsonl"))
    except OSError as error:
        raise ResumeBindingError("Claude transcript lookup failed") from error
    if not matches:
        raise ResumeBindingError("Claude resume transcript was not found")
    if len(matches) > 1:
        raise ResumeBindingError(
            "multiple Claude transcripts claim the requested resume UUID"
        )
    return matches[0]


_CODEX_HALT_OPTIONS = frozenset({"-h", "--help", "-V", "--version"})
_CODEX_ROOT_VALUE_OPTIONS = frozenset(
    {
        "-a",
        "--add-dir",
        "--ask-for-approval",
        "-C",
        "--cd",
        "-c",
        "--config",
        "--disable",
        "--enable",
        "-i",
        "--image",
        "--local-provider",
        "-m",
        "--model",
        "-p",
        "--profile",
        "--remote",
        "--remote-auth-token-env",
        "-s",
        "--sandbox",
    }
)
_CODEX_ROOT_FLAG_OPTIONS = (
    frozenset(
        {
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
            "--full-auto",
            "--no-alt-screen",
            "--oss",
            "--search",
            "--strict-config",
        }
    )
    | _CODEX_HALT_OPTIONS
)
_CODEX_EXEC_VALUE_OPTIONS = frozenset(
    {
        "--add-dir",
        "-C",
        "--cd",
        "--color",
        "-c",
        "--config",
        "--disable",
        "--enable",
        "-i",
        "--image",
        "--local-provider",
        "-m",
        "--model",
        "-o",
        "--output-last-message",
        "--output-schema",
        "-p",
        "--profile",
        "-s",
        "--sandbox",
    }
)
_CODEX_EXEC_FLAG_OPTIONS = (
    frozenset(
        {
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
            "--ephemeral",
            "--full-auto",
            "--ignore-rules",
            "--ignore-user-config",
            "--json",
            "--oss",
            "--skip-git-repo-check",
            "--strict-config",
        }
    )
    | _CODEX_HALT_OPTIONS
)
_CODEX_RESUME_VALUE_OPTIONS = _CODEX_ROOT_VALUE_OPTIONS
_CODEX_RESUME_FLAG_OPTIONS = _CODEX_ROOT_FLAG_OPTIONS | {
    "--all",
    "--include-non-interactive",
}
_CODEX_EXEC_RESUME_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "--config",
        "--disable",
        "--enable",
        "-i",
        "--image",
        "-m",
        "--model",
        "-o",
        "--output-last-message",
        "--output-schema",
    }
)
_CODEX_EXEC_RESUME_FLAG_OPTIONS = (
    frozenset(
        {
            "--all",
            "--dangerously-bypass-approvals-and-sandbox",
            "--dangerously-bypass-hook-trust",
            "--ignore-rules",
            "--ignore-user-config",
            "--json",
            "--skip-git-repo-check",
            "--strict-config",
        }
    )
    | _CODEX_HALT_OPTIONS
)
_CODEX_ATTACHED_SHORT = frozenset({"-a", "-C", "-c", "-i", "-m", "-o", "-p", "-s"})


def _select_codex_resume(args: tuple[str, ...]) -> str | None:
    remote = False
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return None
        if argument in _CODEX_HALT_OPTIONS:
            return None
        if argument == "resume":
            if remote:
                raise ResumeBindingError(
                    "Codex remote resume has no local transcript to bind"
                )
            return _codex_resume_identity(args, index + 1, interactive=True)
        if argument in {"exec", "e"}:
            return _select_codex_exec_resume(args, index + 1, remote=remote)
        if argument == "fork":
            return None
        if not argument.startswith("-"):
            return None
        if argument == "--remote" or argument.startswith("--remote="):
            remote = True
        consumed = _consume_codex_option(
            args,
            index,
            value_options=_CODEX_ROOT_VALUE_OPTIONS,
            flag_options=_CODEX_ROOT_FLAG_OPTIONS,
            variadic_image=True,
        )
        if consumed is None:
            return _unknown_codex_option(args, index)
        index = consumed
    return None


def _select_codex_exec_resume(
    args: tuple[str, ...],
    index: int,
    *,
    remote: bool,
) -> str | None:
    ephemeral = False
    while index < len(args):
        argument = args[index]
        if argument == "--":
            return None
        if argument in _CODEX_HALT_OPTIONS:
            return None
        if argument == "resume":
            if remote:
                raise ResumeBindingError(
                    "Codex remote resume has no local transcript to bind"
                )
            if ephemeral:
                raise ResumeBindingError("Codex explicit resume cannot use --ephemeral")
            return _codex_resume_identity(args, index + 1, interactive=False)
        if argument == "review" or not argument.startswith("-"):
            return None
        if argument == "--ephemeral":
            ephemeral = True
            index += 1
            continue
        if argument == "--remote" or argument.startswith("--remote="):
            remote = True
        consumed = _consume_codex_option(
            args,
            index,
            value_options=_CODEX_EXEC_VALUE_OPTIONS,
            flag_options=_CODEX_EXEC_FLAG_OPTIONS,
            variadic_image=True,
        )
        if consumed is None:
            return _unknown_codex_option(args, index)
        index = consumed
    return None


def _codex_resume_identity(
    args: tuple[str, ...],
    index: int,
    *,
    interactive: bool,
) -> str | None:
    value_options = (
        _CODEX_RESUME_VALUE_OPTIONS if interactive else _CODEX_EXEC_RESUME_VALUE_OPTIONS
    )
    flag_options = (
        _CODEX_RESUME_FLAG_OPTIONS if interactive else _CODEX_EXEC_RESUME_FLAG_OPTIONS
    )
    identity: str | None = None
    prompt_seen = False
    options_enabled = True
    while index < len(args):
        argument = args[index]
        if options_enabled and argument == "--":
            options_enabled = False
            index += 1
            continue
        if options_enabled and argument in _CODEX_HALT_OPTIONS:
            return None
        if options_enabled and argument == "--last":
            raise ResumeBindingError(
                "Codex --last cannot be bound; use an explicit session UUID"
            )
        if options_enabled and argument == "--ephemeral":
            raise ResumeBindingError("Codex explicit resume cannot use --ephemeral")
        if options_enabled and (
            argument == "--remote" or argument.startswith("--remote=")
        ):
            raise ResumeBindingError(
                "Codex remote resume has no local transcript to bind"
            )
        if options_enabled and argument.startswith("-"):
            consumed = _consume_codex_option(
                args,
                index,
                value_options=value_options,
                flag_options=flag_options,
                variadic_image=interactive,
            )
            if consumed is None:
                raise ResumeBindingError(
                    f"cannot safely interpret Codex resume option {argument!r}"
                )
            index = consumed
            continue
        if identity is None:
            identity = _canonical_uuid(argument, provider="Codex")
        elif prompt_seen:
            raise ResumeBindingError("Codex resume received multiple prompt arguments")
        else:
            prompt_seen = True
        index += 1
    if identity is None:
        raise ResumeBindingError("Codex resume requires an explicit session UUID")
    return identity


def _consume_codex_images(args: tuple[str, ...], index: int) -> int | None:
    """Consume Codex's separate-token variadic image option."""
    if index + 1 >= len(args):
        return None
    index += 2
    while index < len(args) and not args[index].startswith("-"):
        index += 1
    return index


def _consume_codex_option(
    args: tuple[str, ...],
    index: int,
    *,
    value_options: frozenset[str],
    flag_options: frozenset[str],
    variadic_image: bool,
) -> int | None:
    argument = args[index]
    option, separator, _inline = argument.partition("=")
    if option in flag_options:
        return index + 1 if not separator else None
    if option not in value_options:
        attached = next(
            (
                short
                for short in _CODEX_ATTACHED_SHORT
                if short in value_options
                and argument.startswith(short)
                and len(argument) > len(short)
            ),
            None,
        )
        return index + 1 if attached is not None else None
    if separator:
        return index + 1
    if index + 1 >= len(args):
        return None
    if variadic_image and option in {"-i", "--image"}:
        # Codex declares this form variadic. A following ``resume`` is consumed
        # as another image rather than parsed as a subcommand.
        return _consume_codex_images(args, index)
    return index + 2


def _unknown_codex_option(args: tuple[str, ...], index: int) -> None:
    if "resume" in args[index + 1 :]:
        raise ResumeBindingError(
            f"cannot safely interpret Codex option {args[index]!r} before resume"
        )


def _resolve_codex(adapter: CodexAdapter, identity: str) -> Path:
    roots = tuple(adapter.session_dirs())
    if len(roots) != 1:
        raise ResumeBindingError("Codex transcript root is unavailable")
    sessions = roots[0].absolute()
    indexed = _codex_index_path(sessions.parent / "state_5.sqlite", identity)
    if indexed is not None:
        return indexed
    return _codex_bounded_fallback(sessions, identity)


def _codex_index_path(database: Path, identity: str) -> Path | None:
    try:
        connection = sqlite3.connect(
            f"{database.absolute().as_uri()}?mode=ro",
            uri=True,
            timeout=0.0,
        )
    except sqlite3.Error:
        return None
    try:
        rows = connection.execute(
            "SELECT rollout_path FROM threads WHERE id = ? LIMIT 2",
            (identity,),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if not rows:
        return None
    if len(rows) != 1 or not isinstance(rows[0][0], str):
        raise ResumeBindingError("Codex session index returned an ambiguous path")
    path = Path(rows[0][0])
    if not path.is_absolute():
        raise ResumeBindingError("Codex session index returned a relative path")
    return path


def _codex_bounded_fallback(sessions: Path, identity: str) -> Path:
    parsed = UUID(identity)
    if parsed.version != 7:
        raise ResumeBindingError(
            "Codex session index has no row and the UUID has no date shard"
        )
    milliseconds = parsed.int >> 80
    day = datetime.fromtimestamp(milliseconds / 1000, tz=UTC).date()
    matches: list[Path] = []
    for delta in (timedelta(), -timedelta(days=1), timedelta(days=1)):
        shard = sessions / f"{day + delta:%Y/%m/%d}"
        with suppress(OSError):
            matches.extend(shard.glob(f"rollout-*-{identity}.jsonl"))
    if not matches:
        raise ResumeBindingError("Codex resume transcript was not found")
    if len(matches) > 1:
        raise ResumeBindingError(
            "multiple Codex transcripts claim the requested resume UUID"
        )
    return matches[0]


def _canonical_uuid(value: str, *, provider: str) -> str:
    try:
        identity = str(UUID(value))
    except ValueError as error:
        raise ResumeBindingError(
            f"{provider} resume identity must be a canonical UUID"
        ) from error
    if identity != value:
        raise ResumeBindingError(f"{provider} resume identity must be a canonical UUID")
    return identity

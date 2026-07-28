"""Bind one Antigravity process to its exact native conversation transcript."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, cast
from uuid import UUID

import json
import os
import re
import stat
import tempfile

from trackinizer.lib.userdirs import state_dir
from trackinizer.trax.run.adapters.antigravity import AntigravityAdapter
from trackinizer.trax.run.transcript import (
    TranscriptClaim,
    TranscriptReader,
    TranscriptSnapshot,
    snapshot_transcript,
)


__all__ = [
    "AntigravityBindingError",
    "PreparedAntigravity",
    "prepare_antigravity",
]


class AntigravityBindingError(RuntimeError):
    """Antigravity did not prove one unambiguous native conversation."""


_MAX_BYTES = 1 << 20
_READ_BYTES = 64 << 10
_RESUME_MARKERS = frozenset(
    {"Resuming conversation", "Print mode: resuming conversation"}
)
_VALUE_OPTIONS = frozenset(
    {
        "--add-dir",
        "--agent",
        "--conversation",
        "--effort",
        "--log-file",
        "--mode",
        "--model",
        "--print-timeout",
        "--project",
    }
)
_FLAG_OPTIONS = frozenset(
    {
        "-c",
        "--continue",
        "--dangerously-skip-permissions",
        "-h",
        "--help",
        "-i",
        "--new-project",
        "-p",
        "--print",
        "--prompt",
        "--prompt-interactive",
        "--sandbox",
        "--version",
    }
)
_MARKER = re.compile(
    rb"^[IWEF]\d{4} \d{2}:\d{2}:\d{2}\.\d{6} +\d+ "
    rb"[^\]\r\n]+:\d+\] "
    rb"(Created conversation|Resuming conversation|"
    rb"Print mode: resuming conversation) "
    rb"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    rb"[0-9a-f]{4}-[0-9a-f]{12})$"
)


@dataclass(frozen=True, slots=True)
class _Selection:
    args: tuple[str, ...]
    expected_id: str | None


class _Diagnostic:
    """A private provider log retained by descriptor for one launch."""

    def __init__(self, *, fd: int, path: Path, directory: Path) -> None:
        self._fd = fd
        self.path = path
        self._directory = directory
        info = os.fstat(fd)
        self._device = info.st_dev
        self._inode = info.st_ino
        self._offset = 0
        self._buffer = bytearray()

    @classmethod
    def create(cls, root: Path) -> Self:
        """Create a unique 0700 directory and pre-opened 0600 log."""
        root = root.absolute()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise AntigravityBindingError("diagnostic root is not a real directory")
        directory = Path(tempfile.mkdtemp(prefix="agy-", dir=root))
        path = directory / "diagnostic.log"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW
            | os.O_NONBLOCK
        )
        fd = -1
        try:
            directory.chmod(0o700)
            fd = os.open(path, flags, 0o600)
            os.fchmod(fd, 0o600)
            _validate_owned_file(os.fstat(fd), private=True, label="diagnostic")
            return cls(fd=fd, path=path, directory=directory)
        except BaseException:
            if fd >= 0:
                os.close(fd)
            with suppress(FileNotFoundError):
                path.unlink()
            with suppress(OSError):
                directory.rmdir()
            raise

    def read_lines(self) -> tuple[bytes, ...]:
        """Return newly appended complete lines after validating ownership."""
        info = self._validated_info()
        if info.st_size < self._offset:
            raise AntigravityBindingError("owned diagnostic shrank")
        if self._offset < info.st_size:
            chunk = os.pread(
                self._fd,
                min(info.st_size - self._offset, _READ_BYTES),
                self._offset,
            )
            if not chunk:
                raise AntigravityBindingError("owned diagnostic changed while read")
            self._buffer.extend(chunk)
            self._offset += len(chunk)

        newline = self._buffer.rfind(b"\n")
        if newline < 0:
            if len(self._buffer) > _MAX_BYTES:
                raise AntigravityBindingError("diagnostic record exceeds 1 MiB")
            return ()
        complete = bytes(self._buffer[:newline])
        del self._buffer[: newline + 1]
        lines = complete.split(b"\n")
        if any(len(line) > _MAX_BYTES for line in lines):
            raise AntigravityBindingError("diagnostic record exceeds 1 MiB")
        return tuple(line[:-1] if line.endswith(b"\r") else line for line in lines)

    def caught_up(self) -> bool:
        """Whether the held descriptor has no unread bytes right now."""
        info = self._validated_info()
        if info.st_size < self._offset:
            raise AntigravityBindingError("owned diagnostic shrank")
        return info.st_size == self._offset

    def finish(self) -> None:
        """Reject a provider log that ends midway through a line."""
        if self._buffer:
            raise AntigravityBindingError("diagnostic ends with a partial line")

    def close(self) -> None:
        """Close and remove only the uniquely-owned diagnostic artifacts."""
        if self._fd < 0:
            return
        info = os.fstat(self._fd)
        os.close(self._fd)
        self._fd = -1
        try:
            visible = self.path.lstat()
        except FileNotFoundError:
            visible = None
        if visible is not None and (visible.st_dev, visible.st_ino) == (
            info.st_dev,
            info.st_ino,
        ):
            with suppress(OSError):
                self.path.unlink()
        with suppress(OSError):
            self._directory.rmdir()

    def _validated_info(self) -> os.stat_result:
        if self._fd < 0:
            raise AntigravityBindingError("diagnostic is closed")
        info = os.fstat(self._fd)
        if (info.st_dev, info.st_ino) != (self._device, self._inode):
            raise AntigravityBindingError("owned diagnostic changed identity")
        try:
            visible = self.path.lstat()
        except FileNotFoundError as err:
            raise AntigravityBindingError("owned diagnostic was replaced") from err
        if (visible.st_dev, visible.st_ino) != (self._device, self._inode):
            raise AntigravityBindingError("owned diagnostic was replaced")
        _validate_owned_file(info, private=True, label="diagnostic")
        return info


class PreparedAntigravity:
    """Pre-launch Antigravity arguments plus its exact transcript proof."""

    def __init__(
        self,
        *,
        adapter: AntigravityAdapter,
        args: tuple[str, ...],
        expected_id: str | None,
        diagnostic: _Diagnostic,
        snapshot: TranscriptSnapshot | None,
    ) -> None:
        self._adapter = adapter
        self._expected_id = expected_id
        self._diagnostic = diagnostic
        self._snapshot = snapshot
        self._proof: tuple[str, str] | None = None
        self._reader: TranscriptReader | None = None
        self._failure: Exception | None = None
        self._closed = False
        self.cli_args = ("--log-file", str(diagnostic.path), *args)

    @property
    def diagnostic_path(self) -> Path:
        """The unique log path passed to this Antigravity process."""
        return self._diagnostic.path

    @property
    def expected_cli_session_id(self) -> str | None:
        """The pre-launch native identity for a proven resume, if any."""
        return self._expected_id

    def poll(self) -> TranscriptReader | None:
        """Read owned proof and bind its exact transcript when it appears."""
        if self._failure is not None:
            raise self._failure
        try:
            return self._poll()
        except Exception as err:
            self._failure = err
            raise

    def _poll(self) -> TranscriptReader | None:
        if self._closed:
            raise AntigravityBindingError("Antigravity binding is closed")
        for line in self._diagnostic.read_lines():
            marker = _parse_marker(line)
            if marker is None:
                continue
            if self._proof is not None:
                raise AntigravityBindingError(
                    "Antigravity emitted ambiguous conversation proof"
                )
            self._validate_proof(marker)
            self._proof = marker
        if self._reader is not None:
            return self._reader
        if self._proof is None:
            return None

        _kind, identity = self._proof
        claim = TranscriptClaim(
            cli_session_id=identity,
            path=self._adapter.transcript_path(identity),
            before=self._snapshot,
        )
        if not claim.path.exists():
            return None
        try:
            reader = TranscriptReader.open(self._adapter, claim)
        except FileNotFoundError:
            return None
        self._snapshot = None
        self._reader = reader
        return reader

    def finish(self) -> TranscriptReader:
        """Perform the final proof read and require a bound transcript."""
        if self._failure is not None:
            raise self._failure
        try:
            reader = self._poll()
            while not self._diagnostic.caught_up():
                reader = self._poll()
            self._diagnostic.finish()
            return self._require_reader(reader)
        except Exception as err:
            self._failure = err
            raise

    def close(self) -> None:
        """Release the transcript claim and private diagnostic; idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._reader is not None:
            self._reader.close()
        if self._snapshot is not None:
            self._snapshot.close()
        self._diagnostic.close()

    def _validate_proof(self, proof: tuple[str, str]) -> None:
        kind, identity = proof
        if self._expected_id is None:
            if kind != "Created conversation":
                raise AntigravityBindingError(
                    "fresh Antigravity launch reported a resumed conversation"
                )
            return
        if identity != self._expected_id:
            raise AntigravityBindingError(
                "Antigravity resumed a conversation other than the requested one"
            )
        if kind not in _RESUME_MARKERS:
            raise AntigravityBindingError(
                "Antigravity did not report the requested conversation as resumed"
            )

    def _require_reader(self, reader: TranscriptReader | None) -> TranscriptReader:
        if reader is not None:
            return reader
        if self._proof is not None:
            raise AntigravityBindingError(
                "Antigravity conversation transcript never appeared"
            )
        raise AntigravityBindingError("no conversation proof from Antigravity")

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def prepare_antigravity(
    adapter: AntigravityAdapter,
    cli_args: Sequence[str],
    *,
    cwd: Path,
    runtime_root: Path | None = None,
) -> PreparedAntigravity:
    """Pin resume identity and create this launch's private proof channel."""
    selection = _select_conversation(adapter, tuple(cli_args), cwd)
    snapshot = (
        snapshot_transcript(
            adapter,
            adapter.transcript_path(selection.expected_id),
            selection.expected_id,
        )
        if selection.expected_id is not None
        else None
    )
    diagnostic: _Diagnostic | None = None
    try:
        diagnostic = _Diagnostic.create(
            runtime_root or state_dir("trax") / "run" / "binding"
        )
        return PreparedAntigravity(
            adapter=adapter,
            args=selection.args,
            expected_id=selection.expected_id,
            diagnostic=diagnostic,
            snapshot=snapshot,
        )
    except BaseException:
        if diagnostic is not None:
            diagnostic.close()
        if snapshot is not None:
            snapshot.close()
        raise


def _select_conversation(
    adapter: AntigravityAdapter, args: tuple[str, ...], cwd: Path
) -> _Selection:
    conversations: list[tuple[int, str]] = []
    continues: list[tuple[int, bool]] = []
    project_selected = False
    new_projects: list[bool] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--" or not argument.startswith("-"):
            break

        option, separator, inline_value = argument.partition("=")
        if option in _VALUE_OPTIONS:
            if option == "--log-file":
                raise AntigravityBindingError(
                    "trax owns Antigravity's --log-file for session proof"
                )
            if separator:
                value = inline_value
                consumed = 1
            else:
                if index + 1 == len(args):
                    if option == "--conversation":
                        raise AntigravityBindingError("--conversation requires a UUID")
                    break
                value = args[index + 1]
                consumed = 2
            if option == "--conversation":
                conversations.append((index, _canonical_id(value)))
            elif option == "--project":
                project_selected = True
            index += consumed
            continue

        if option in _FLAG_OPTIONS:
            enabled = _boolean_value(inline_value, option) if separator else True
            if option in {"-c", "--continue"}:
                continues.append((index, enabled))
            elif option == "--new-project":
                new_projects.append(enabled)
            index += 1
            continue

        raise AntigravityBindingError(
            f"cannot safely interpret Antigravity option {option!r}"
        )

    if len(conversations) > 1:
        raise AntigravityBindingError("conversation selector is repeated")
    continue_enabled = continues[-1][1] if continues else False
    project_selected |= bool(new_projects and new_projects[-1])
    if conversations and continue_enabled:
        raise AntigravityBindingError("cannot combine --conversation with --continue")
    if continue_enabled and project_selected:
        raise AntigravityBindingError(
            "cannot safely resolve --continue with Antigravity project selection"
        )
    if conversations:
        return _Selection(args=args, expected_id=conversations[0][1])
    if not continue_enabled:
        return _Selection(args=args, expected_id=None)

    expected = _cached_conversation(adapter, cwd)
    replacement = ("--conversation", expected) if expected is not None else ()
    positions = {position for position, _enabled in continues}
    last_position = continues[-1][0]
    rewritten: list[str] = []
    for position, argument in enumerate(args):
        if position == last_position:
            rewritten.extend(replacement)
        elif position not in positions:
            rewritten.append(argument)
    return _Selection(args=tuple(rewritten), expected_id=expected)


def _boolean_value(value: str, option: str) -> bool:
    """Parse the explicit Go-style boolean spellings Agy documents."""
    if value in {"1", "t", "T", "TRUE", "true", "True"}:
        return True
    if value in {"0", "f", "F", "FALSE", "false", "False"}:
        return False
    raise AntigravityBindingError(
        f"{option} requires a true or false value when written with '='"
    )


def _cached_conversation(adapter: AntigravityAdapter, cwd: Path) -> str | None:
    cache = adapter.brain_dir.parent / "cache" / "last_conversations.json"
    try:
        fd = os.open(
            cache,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except FileNotFoundError:
        return None
    except OSError as err:
        raise AntigravityBindingError(
            "cannot safely open Antigravity's conversation cache"
        ) from err
    try:
        before = os.fstat(fd)
        _validate_owned_file(before, private=False, label="conversation cache")
        if before.st_size > _MAX_BYTES:
            raise AntigravityBindingError(
                "Antigravity conversation cache exceeds 1 MiB"
            )
        raw = os.pread(fd, before.st_size + 1, 0)
        if len(raw) != before.st_size:
            raise AntigravityBindingError(
                "Antigravity conversation cache changed while read"
            )
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino, after.st_size) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ):
            raise AntigravityBindingError(
                "Antigravity conversation cache changed while read"
            )
        _verify_visible(cache, after, "Antigravity conversation cache")
    finally:
        os.close(fd)

    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise AntigravityBindingError(
            "Antigravity conversation cache is malformed"
        ) from err
    if not isinstance(parsed, Mapping):
        raise AntigravityBindingError("Antigravity conversation cache is not an object")
    mapping = cast(Mapping[object, object], parsed)
    try:
        logical = str(cwd.absolute())
        physical = str(cwd.resolve(strict=True))
    except OSError as err:
        raise AntigravityBindingError(
            "cannot resolve Antigravity working directory"
        ) from err
    identities = {
        _canonical_id(value)
        for key in {logical, physical}
        if (value := mapping.get(key)) is not None and isinstance(value, str)
    }
    for key in {logical, physical}:
        if key in mapping and not isinstance(mapping[key], str):
            raise AntigravityBindingError(
                "Antigravity conversation cache has an invalid identity"
            )
    if len(identities) > 1:
        raise AntigravityBindingError(
            "Antigravity conversation cache is ambiguous for this directory"
        )
    return next(iter(identities), None)


def _parse_marker(line: bytes) -> tuple[str, str] | None:
    match = _MARKER.fullmatch(line)
    if match is None:
        return None
    kind = match.group(1).decode("ascii")
    identity = match.group(2).decode("ascii")
    return (kind, _canonical_id(identity))


def _canonical_id(value: object) -> str:
    if not isinstance(value, str):
        raise AntigravityBindingError("conversation identity must be a canonical UUID")
    try:
        parsed = UUID(value)
    except ValueError as err:
        raise AntigravityBindingError(
            "conversation identity must be a canonical UUID"
        ) from err
    if str(parsed) != value:
        raise AntigravityBindingError("conversation identity must be a canonical UUID")
    return value


def _validate_owned_file(info: os.stat_result, *, private: bool, label: str) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise AntigravityBindingError(f"{label} is not one owned regular file")
    if info.st_uid != os.getuid():
        raise AntigravityBindingError(f"{label} is not owned by the current user")
    unsafe = 0o077 if private else 0o022
    if info.st_mode & unsafe:
        raise AntigravityBindingError(f"{label} is group/world writable")


def _verify_visible(path: Path, info: os.stat_result, label: str) -> None:
    try:
        visible = path.lstat()
    except FileNotFoundError as err:
        raise AntigravityBindingError(f"{label} was replaced") from err
    if (visible.st_dev, visible.st_ino) != (info.st_dev, info.st_ino):
        raise AntigravityBindingError(f"{label} was replaced")

"""Codex CLI adapter.

Sessions live at
``~/.codex/sessions/<YYYY>/<MM>/<DD>/rollout-<ISO>-<uuid-v7>.jsonl``,
append-only JSONL. Each line is ``{"timestamp", "type", "payload"}``, where
``type`` and the nested ``payload.type`` together name the record.

Codex logs each turn twice: a streamed ``event_msg`` and a canonical
``response_item``. ``parse`` keys off the **canonical** ``response_item``
records (the streamed ``event_msg`` duplicates are skipped) plus
``compacted``, so each line yields exactly one typed message.
Spawn codex with ``-c model_reasoning_summary=detailed`` so the ``reasoning``
item's ``summary[].text`` is populated. See
``docs/cli-scraping-investigation.md`` for the empirical layout.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final, cast
from uuid import UUID

import json
import os

from trackinizer.lib.custom_json import JSON, json_freeze
from trackinizer.trax.run.adapters.base import Event
from trackinizer.types.agent_session_events import (
    AssistantMessage,
    Compaction,
    Message,
    SystemMessage,
    ToolCall,
    ToolResult,
    UnknownMessage,
    UserMessage,
)


class CodexAdapter:
    """Reads the ``codex`` CLI's rollout JSONL files.

    Carries the last ``turn_context.model`` across :meth:`parse` calls so each
    turn's response items inherit it (codex logs the model once per turn, on a
    separate ``turn_context`` line). This per-run state is safe because the
    runner builds a fresh adapter per run (the registry holds the class as a
    factory) and drives it from a single drain thread, so no state leaks across
    runs or threads.
    """

    name: str = "codex"
    cli_binary: str = "codex"

    def __init__(self) -> None:
        # The most recent ``turn_context.model``, stamped onto subsequent
        # response_item Events until the next ``turn_context`` (R2R-020).
        self._last_model: str | None = None

    @property
    def _sessions_dir(self) -> Path:
        # Codex canonicalizes an explicit ``$CODEX_HOME`` (which may itself be
        # a symlink) and otherwise defaults to ``~/.codex``. Resolve per call,
        # not at import, so a switched home and tests see the same root as the
        # wrapped process.
        configured = os.environ.get("CODEX_HOME")
        codex_home = Path(configured) if configured else Path.home() / ".codex"
        return codex_home.resolve(strict=False) / "sessions"

    def session_dirs(self) -> Iterable[Path]:
        sessions = self._sessions_dir
        if not sessions.is_dir():
            return ()
        # Codex shards by Y/M/D; returning the root lets the runner glob
        # recursively, so older days still get captured if they keep growing.
        return (sessions,)

    def matches_session_file(self, path: Path) -> bool:
        canonical = path.resolve(strict=False)
        return (
            canonical.suffix == ".jsonl"
            and canonical.name.startswith("rollout-")
            and self._sessions_dir in canonical.parents
        )

    def is_root_session_file(self, path: Path) -> bool:
        """Whether a rollout belongs to the wrapped root Codex session.

        Current Codex releases write child-agent rollouts beside their root
        rollout, with the same cwd and creation time.  Directory/mtime scoping
        therefore cannot distinguish them.  A child identifies itself in its
        ``session_meta``: ``payload.id`` names the child file while
        ``payload.session_id`` names its root, and ``source.subagent`` records
        the child role.  Wait for a complete first record, then reject either
        positive child signal so one ``trax run`` can never bind to two native
        session identities.
        """
        try:
            with path.open("rb") as source:
                raw = source.readline()
        except OSError:
            return False
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(parsed, Mapping) or parsed.get("type") != "session_meta":
            return False
        payload = parsed.get("payload")
        if not isinstance(payload, Mapping):
            return False
        rollout_id = _canonical_uuid(payload.get("id"))
        root_id = _canonical_uuid(payload.get("session_id"))
        source = payload.get("source")
        if isinstance(source, Mapping) and "subagent" in source:
            return False
        return rollout_id is None or root_id is None or rollout_id == root_id

    def session_id_from_path(self, path: Path) -> str | None:
        """Return Codex's native rollout id, corroborating its two sources.

        Codex writes the same UUID in ``session_meta.payload.id`` and at the
        end of ``rollout-<timestamp>-<uuid>.jsonl``. Either source is enough
        while the file is being created, but two valid, differing UUIDs are
        ambiguous and therefore fail closed rather than correlating the run to
        the wrong AgentSession.
        """
        try:
            with path.open("rb") as source:
                first_record = source.readline()
        except OSError:
            first_record = None
        return self.session_id_from_transcript(path, first_record)

    def session_id_from_transcript(
        self, path: Path, first_record: bytes | None
    ) -> str | None:
        """Corroborate filename and metadata without reopening an owned path."""
        if path.suffix != ".jsonl" or not path.name.startswith("rollout-"):
            return None
        filename_id = _session_id_from_filename(path)
        metadata_id = _session_id_from_metadata(first_record)
        if (
            filename_id is not None
            and metadata_id is not None
            and filename_id != metadata_id
        ):
            return None
        return metadata_id or filename_id

    def parse(self, raw: bytes) -> Iterable[Event]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        if not isinstance(parsed, Mapping):
            return ()
        obj = json_freeze(cast(Mapping[str, object], parsed))
        if obj.get("type") == "turn_context":
            # Stash this turn's model (no event); it stamps the turn's items.
            model = _payload(obj).get("model")
            self._last_model = model if isinstance(model, str) and model else None
            return ()
        message = _to_message(obj)
        if message is None:
            return ()
        return (
            Event(message=message, timestamp=_timestamp(obj), model=self._last_model),
        )


def _session_id_from_filename(path: Path) -> str | None:
    """Extract the canonical trailing UUID from a Codex rollout filename."""
    stem = path.stem
    # A canonical UUID is 36 characters and Codex prefixes it with ``-``.
    if len(stem) <= 37 or stem[-37] != "-":
        return None
    return _canonical_uuid(stem[-36:])


def _session_id_from_metadata(raw: bytes | None) -> str | None:
    """Read the native id from a held transcript's first JSONL record."""
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(parsed, Mapping) or parsed.get("type") != "session_meta":
        return None
    payload = parsed.get("payload")
    if not isinstance(payload, Mapping):
        return None
    # ``payload.session_id`` may point at a parent/root rollout for subagents;
    # ``payload.id`` is this file's native identity and matches its filename.
    return _canonical_uuid(payload.get("id"))


def _canonical_uuid(value: object) -> str | None:
    """Return a UUID in canonical text form, or ``None`` for provider drift."""
    if not isinstance(value, str):
        return None
    try:
        return str(UUID(value))
    except ValueError:
        return None


def _to_message(obj: JSON) -> Message | None:
    """Normalize one rollout line to a typed message, or ``None`` to skip.

    Canonical ``response_item`` records carry the turn; ``compacted`` is a
    compaction. ``session_meta`` is the first line codex writes (at launch,
    before any turn) and carries the base system instructions, so it maps to a
    :class:`SystemMessage` -- this also opens the capture session at codex
    startup rather than on the first user turn. ``event_msg`` (streamed
    duplicates) and ``turn_context`` (per-turn metadata that lives on the
    AgentSession row) are skipped so each turn is counted once.
    """
    outer = obj.get("type")
    if outer == "session_meta":
        return _session_meta(obj)
    if outer == "compacted":
        return _compaction(obj)
    if outer != "response_item":
        return None
    payload = _payload(obj)
    inner = payload.get("type")
    if inner == "message":
        return _role_message(payload)
    if inner == "reasoning":
        return AssistantMessage(
            thinking=_reasoning_summary(payload),
            thinking_encrypted=_str(payload.get("encrypted_content")),
        )
    if inner == "function_call":
        return AssistantMessage(
            tool_calls=(
                ToolCall(
                    id=_str(payload.get("call_id")),
                    name=_str(payload.get("name")),
                    args=_json_args(payload.get("arguments")),
                ),
            ),
        )
    if inner == "custom_tool_call":
        return AssistantMessage(
            tool_calls=(
                ToolCall(
                    id=_str(payload.get("call_id")),
                    name=_str(payload.get("name")),
                    args=_custom_tool_args(payload.get("input")),
                ),
            ),
        )
    if inner in ("function_call_output", "custom_tool_call_output"):
        output = payload.get("output")
        return ToolResult(
            call_id=_str(payload.get("call_id")),
            content=_content_text(output),
        )
    return UnknownMessage(raw=obj)


def _session_meta(obj: JSON) -> Message:
    """The launch-time ``session_meta`` line: capture its base instructions.

    Codex writes this first, before any turn, with the model's base system
    prompt under ``payload.base_instructions.text``. Emitting it as a
    :class:`SystemMessage` both records that system prompt and opens the
    capture session at codex startup, not on the first user turn.
    """
    payload = _payload(obj)
    base = payload.get("base_instructions")
    text = _str(cast(JSON, base).get("text")) if isinstance(base, Mapping) else ""
    return SystemMessage(text=text, role="system")


# Codex injects environment context (the repo's AGENTS.md and similar) as a
# ``user``-role message the CLI never shows; it is recognizable by this
# prefix. It is primed context, not something the human typed, so it maps to
# a SystemMessage rather than a UserMessage.
_ENV_CONTEXT_PREFIXES: Final = (
    "# AGENTS.md instructions for ",
    "<INSTRUCTIONS>",
)


def _role_message(payload: JSON) -> Message:
    """A ``response_item`` of type ``message``, dispatched by wire role.

    ``user`` is human input -- except codex's auto-injected environment
    context (AGENTS.md), which is primed context the CLI hides, mapped to
    :class:`SystemMessage`. ``system`` / ``developer`` are provider-injected
    priming (a permissions preamble, a developer system prompt) -- captured as
    :class:`SystemMessage`, not as a model reply; ``assistant`` is the model
    turn. An unrecognized role (a future ``tool`` / ``function``) maps to
    :class:`UnknownMessage` rather than being mislabeled as an assistant reply
    (R2R-016).
    """
    text = _content_text(payload.get("content"))
    role = _str(payload.get("role"))
    if role == "user":
        if text.startswith(_ENV_CONTEXT_PREFIXES):
            return SystemMessage(text=text, role="user")
        return UserMessage(text=text)
    if role in ("system", "developer"):
        return SystemMessage(text=text, role=role)
    if role == "assistant":
        return AssistantMessage(text=text)
    return UnknownMessage(raw=payload)


def _compaction(obj: JSON) -> Message:
    payload = _payload(obj)
    return Compaction(text=_str(payload.get("summary")))


def _reasoning_summary(payload: JSON) -> str:
    """Join the ``summary[].text`` parts of a reasoning item."""
    summary = payload.get("summary")
    if not isinstance(summary, Sequence) or isinstance(summary, str):
        return ""
    return "".join(
        _str(cast(JSON, part).get("text"))
        for part in cast("Sequence[object]", summary)
        if isinstance(part, Mapping)
    )


def _content_text(value: object) -> str:
    """Join the ``text`` of a codex ``content`` block list."""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return "".join(
            _str(cast(JSON, b).get("text")) for b in value if isinstance(b, Mapping)
        )
    return ""


def _json_args(value: object) -> dict[str, object]:
    """Codex tool args arrive as a JSON-encoded string; decode to a dict."""
    if isinstance(value, Mapping):
        return dict(cast("Mapping[str, object]", value))
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, Mapping):
            return dict(cast("Mapping[str, object]", decoded))
    return {}


def _custom_tool_args(value: object) -> dict[str, object]:
    """Preserve Codex custom-tool input even when it is not JSON."""
    decoded = _json_args(value)
    if decoded:
        return decoded
    if isinstance(value, str):
        return {"input": value}
    if isinstance(value, Mapping):
        return dict(cast("Mapping[str, object]", value))
    return {}


def _payload(obj: JSON) -> JSON:
    payload = obj.get("payload")
    return cast(JSON, payload) if isinstance(payload, Mapping) else cast(JSON, {})


def _timestamp(obj: JSON) -> datetime | None:
    """Parse a rollout line's ``timestamp`` (CLI clock), or ``None`` if absent.

    Codex writes ISO-8601 with a trailing ``Z``, which ``fromisoformat``
    accepts directly (Python 3.11+).
    """
    raw = obj.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""

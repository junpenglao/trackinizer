"""Antigravity CLI adapter.

Antigravity 1.1.7 stores each conversation under::

    ~/.gemini/antigravity-cli/brain/<conversation-uuid>/
        .system_generated/logs/transcript_full.jsonl

The sibling ``transcript.jsonl`` is a truncated rendering of the same records,
so this adapter intentionally matches only ``transcript_full.jsonl``: matching
both would emit every turn twice and would discard parts of long thinking/tool
results. The conversation directory UUID is the same ID accepted by
Antigravity's ``--conversation`` flag and remains stable when that conversation
is resumed.

Each appended JSON object carries a provider ``step_index``, ``source``,
``type``, ``status``, and ``created_at``. Model planner records contain text,
thinking, and optionally one tool call; the following tool record uses the
next step index. We derive a stable call id from that provider step index so
the normalized :class:`ToolCall` and :class:`ToolResult` remain linked.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import cast
from uuid import UUID

import json

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


_SYSTEM_TYPES = frozenset(
    {"CONVERSATION_HISTORY", "EPHEMERAL_MESSAGE", "SYSTEM_MESSAGE"}
)
_TOOL_RESULT_TYPES = frozenset(
    {"CODE_ACTION", "LIST_DIRECTORY", "RUN_COMMAND", "VIEW_FILE"}
)
_ERROR_STATUSES = frozenset({"CANCELLED", "ERROR", "FAILED"})
_TERMINAL_STATUSES = _ERROR_STATUSES | {"DONE"}


class AntigravityAdapter:
    """Reads Antigravity's append-only full-conversation transcripts."""

    name: str = "agy"
    cli_binary: str = "agy"

    @property
    def brain_dir(self) -> Path:
        """Antigravity's conversation root under its retained state directory."""
        # Resolve ``$HOME`` per call, like every filesystem adapter: tests and
        # a run under a switched home must not inherit an import-time path.
        return Path.home() / ".gemini" / "antigravity-cli" / "brain"

    def session_dirs(self) -> Iterable[Path]:
        brain = self.brain_dir
        return (brain,) if brain.is_dir() else ()

    def transcript_path(self, cli_session_id: str) -> Path:
        """The canonical full-conversation transcript for one native UUID."""
        return (
            self.brain_dir
            / cli_session_id
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )

    def matches_session_file(self, path: Path) -> bool:
        return self._conversation_id_from_path(path) is not None

    def session_id_from_path(self, path: Path) -> str | None:
        """Return the exact native Antigravity conversation UUID in ``path``."""
        return self._conversation_id_from_path(path)

    def session_id_from_transcript(
        self, path: Path, first_record: bytes | None
    ) -> str | None:
        """Corroborate the held transcript; its UUID is in the directory path."""
        del first_record
        return self.session_id_from_path(path)

    def _conversation_id_from_path(self, path: Path) -> str | None:
        """Validate the full transcript path and extract its canonical UUID."""
        try:
            relative = path.relative_to(self.brain_dir)
        except ValueError:
            return None
        if len(relative.parts) != 4 or relative.parts[1:] != (
            ".system_generated",
            "logs",
            "transcript_full.jsonl",
        ):
            return None
        # Do not let a provider-shaped symlink escape the transcript store.
        # Exact binding later retains an opened descriptor; this lexical scan
        # still rejects every currently symlinked component before opening.
        provider_root = self.brain_dir.parent.parent
        components = (provider_root, provider_root / "antigravity-cli", self.brain_dir)
        current = self.brain_dir
        relative_components: list[Path] = []
        for part in relative.parts:
            current /= part
            relative_components.append(current)
        if any(
            component.is_symlink() for component in (*components, *relative_components)
        ):
            return None
        candidate = relative.parts[0]
        try:
            parsed = UUID(candidate)
        except ValueError:
            return None
        # Antigravity 1.1.7 writes canonical lowercase UUIDs. Reject alternate
        # spellings so a path cannot acquire a different identity after
        # normalization.
        return candidate if str(parsed) == candidate else None

    def parse(self, raw: bytes) -> Iterable[Event]:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return ()
        if not isinstance(parsed, Mapping):
            return ()
        obj = json_freeze(cast(Mapping[str, object], parsed))
        message = _to_message(obj)
        if message is None:
            return ()
        return (Event(message=message, timestamp=_timestamp(obj)),)


def _to_message(obj: JSON) -> Message | None:
    """Normalize one Antigravity record, or skip empty lifecycle markers."""
    source = _str(obj.get("source"))
    record_type = _str(obj.get("type"))

    if source == "USER_EXPLICIT" and record_type == "USER_INPUT":
        return UserMessage(text=_str(obj.get("content")))
    if source == "MODEL" and record_type == "PLANNER_RESPONSE":
        return _assistant_message(obj)
    if source == "MODEL" and record_type in _TOOL_RESULT_TYPES:
        return _tool_result(obj)
    if source == "SYSTEM" and record_type == "CHECKPOINT":
        return Compaction(text=_str(obj.get("content")))
    if source == "SYSTEM" and record_type in _SYSTEM_TYPES:
        content = _str(obj.get("content"))
        # Antigravity writes an empty CONVERSATION_HISTORY marker when it
        # initializes the context. It is lifecycle, not a message the model saw.
        return SystemMessage(text=content) if content else None
    return UnknownMessage(raw=obj)


def _assistant_message(obj: JSON) -> Message:
    """One Antigravity model response, including nested tool invocations."""
    raw_calls = _tool_calls(obj)
    if len(raw_calls) > 1:
        # Only zero/one-call planner records are evidenced in 1.1.7. Inventing
        # future step ids for a batch could link unrelated tool results.
        return UnknownMessage(raw=obj)
    calls: tuple[ToolCall, ...] = ()
    if raw_calls:
        step_index = _step_index(obj)
        if step_index is None:
            return UnknownMessage(raw=obj)
        result_step = step_index + 1
        raw_call = raw_calls[0]
        calls = (
            ToolCall(
                id=f"agy-step-{result_step}",
                name=_str(raw_call.get("name")),
                args=_mapping(raw_call.get("args")),
            ),
        )
    return AssistantMessage(
        text=_str(obj.get("content")),
        thinking=_str(obj.get("thinking")),
        tool_calls=calls,
    )


def _tool_calls(obj: JSON) -> tuple[JSON, ...]:
    calls = obj.get("tool_calls")
    if not isinstance(calls, Sequence) or isinstance(calls, str):
        return ()
    return tuple(
        cast(JSON, call)
        for call in cast("Sequence[object]", calls)
        if isinstance(call, Mapping)
    )


def _tool_result(obj: JSON) -> Message:
    step_index = _step_index(obj)
    status = _str(obj.get("status"))
    if step_index is None:
        return UnknownMessage(raw=obj)
    if status not in _TERMINAL_STATUSES:
        return UnknownMessage(raw=obj)
    return ToolResult(
        call_id=f"agy-step-{step_index}",
        content=_str(obj.get("content")),
        is_error=status in _ERROR_STATUSES,
    )


def _timestamp(obj: JSON) -> datetime | None:
    raw = obj.get("created_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _str(value: object) -> str:
    return value if isinstance(value, str) else ""


def _step_index(obj: JSON) -> int | None:
    value = obj.get("step_index")
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _mapping(value: object) -> dict[str, object]:
    return (
        dict(cast("Mapping[str, object]", value)) if isinstance(value, Mapping) else {}
    )

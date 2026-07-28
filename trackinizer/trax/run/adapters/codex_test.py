"""Tests for the codex adapter: rollout JSONL fixtures → typed messages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import json

from trackinizer.trax.run.adapters.base import Event
from trackinizer.trax.run.adapters.codex import CodexAdapter
from trackinizer.types.agent_session_events import (
    AssistantMessage,
    Compaction,
    SystemMessage,
    ToolResult,
    UnknownMessage,
    UserMessage,
)


if TYPE_CHECKING:
    import pytest


def _encode(obj: object) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _parse_one(raw: bytes) -> Event | None:
    """The single event for a one-line record, or ``None`` when skipped.

    A fresh ``CodexAdapter`` per call so the carried ``_last_model`` state never
    leaks between cases.
    """
    events = list(CodexAdapter().parse(raw, whole_file=False))
    assert len(events) <= 1, events
    return events[0] if events else None


class TestCodexSessionId:
    """Codex's rollout UUID is corroborated by the filename and session_meta."""

    _SESSION_ID = "019fa3b5-e77c-7503-8cca-369d0b3e304d"

    def _rollout(self, tmp_path: Path, session_id: str | None = None) -> Path:
        suffix = session_id or self._SESSION_ID
        return tmp_path / f"rollout-2026-07-27T15-13-55-{suffix}.jsonl"

    def test_matching_session_meta_and_filename_return_native_id(
        self, tmp_path: Path
    ) -> None:
        path = self._rollout(tmp_path)
        path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": self._SESSION_ID,
                        # Subagent rollouts point ``session_id`` at their root;
                        # ``id`` is the identity of this rollout file.
                        "session_id": "019f9fb7-96b9-7261-bc8a-4b82c6028b35",
                    },
                }
            )
            + "\n"
        )

        assert CodexAdapter().session_id_from_path(path) == self._SESSION_ID

    def test_filename_id_is_used_while_session_meta_is_not_yet_complete(
        self, tmp_path: Path
    ) -> None:
        path = self._rollout(tmp_path)
        path.write_text('{"type":"session_meta","payload":')

        assert CodexAdapter().session_id_from_path(path) == self._SESSION_ID

    def test_session_meta_id_is_used_when_filename_has_no_uuid(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "rollout-unexpected-name.jsonl"
        path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": self._SESSION_ID},
                }
            )
            + "\n"
        )

        assert CodexAdapter().session_id_from_path(path) == self._SESSION_ID

    def test_mismatched_session_meta_and_filename_fail_closed(
        self, tmp_path: Path
    ) -> None:
        path = self._rollout(tmp_path)
        other_id = "019fa3c1-d5de-7181-a4c6-90dd608fc015"
        path.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": other_id}}) + "\n"
        )

        assert CodexAdapter().session_id_from_path(path) is None

    def test_malformed_session_meta_id_falls_back_to_filename(
        self, tmp_path: Path
    ) -> None:
        path = self._rollout(tmp_path)
        path.write_text(
            json.dumps({"type": "session_meta", "payload": {"id": "not-a-uuid"}}) + "\n"
        )

        assert CodexAdapter().session_id_from_path(path) == self._SESSION_ID

    def test_path_without_rollout_shape_or_native_id_returns_none(
        self, tmp_path: Path
    ) -> None:
        non_rollout = tmp_path / f"notes-{self._SESSION_ID}.jsonl"
        non_rollout.write_text("")
        rollout_without_id = tmp_path / "rollout-2026-07-27T15-13-55.jsonl"
        rollout_without_id.write_text("")

        adapter = CodexAdapter()
        assert adapter.session_id_from_path(non_rollout) is None
        assert adapter.session_id_from_path(rollout_without_id) is None

    def test_custom_codex_home_and_symlinked_root_are_canonicalized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        actual_home = tmp_path / "actual-codex-home"
        sessions = actual_home / "sessions" / "2026" / "07" / "27"
        sessions.mkdir(parents=True)
        linked_home = tmp_path / "linked-codex-home"
        linked_home.symlink_to(actual_home, target_is_directory=True)
        monkeypatch.setenv("CODEX_HOME", str(linked_home))
        rollout = sessions / (f"rollout-2026-07-27T15-13-55-{self._SESSION_ID}.jsonl")
        rollout.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": self._SESSION_ID,
                        "session_id": self._SESSION_ID,
                    },
                }
            )
            + "\n"
        )

        adapter = CodexAdapter()

        assert tuple(adapter.session_dirs()) == (actual_home.resolve() / "sessions",)
        assert adapter.matches_session_file(rollout)
        assert adapter.session_id_from_path(rollout) == self._SESSION_ID


class TestCodexParseLine:
    """Codex logs each turn twice; only the canonical ``response_item`` (and
    ``compacted``) records yield a message. Streamed ``event_msg`` duplicates
    and lifecycle records (``session_meta`` / ``turn_context``) are skipped.
    """

    def test_session_meta_is_system_message_with_base_instructions(self) -> None:
        """``session_meta`` (codex's first line) captures the base system prompt
        and opens the session at startup, not on the first user turn.
        """
        line = _encode(
            {
                "timestamp": "2026-05-29T00:00:00Z",
                "type": "session_meta",
                "payload": {
                    "id": "abc",
                    "cwd": "/home/user/repo",
                    "base_instructions": {"text": "You are Codex."},
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, SystemMessage)
        assert event.message.text == "You are Codex."

    def test_line_timestamp_is_parsed_onto_event(self) -> None:
        """Codex's per-line ``timestamp`` carries onto the Event envelope.

        Each rollout line is ``{"timestamp", "type", "payload"}``; the
        timestamp is the CLI clock for the turn and the sink writes it as the
        event's ``timestamp``. Dropping it would stamp every turn with a
        default, losing the real ordering on the CLI clock.
        """
        line = _encode(
            {
                "timestamp": "2026-05-29T12:34:56.789Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"text": "hi"}],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert event.timestamp == datetime(2026, 5, 29, 12, 34, 56, 789_000, tzinfo=UTC)

    def test_env_context_user_message_is_system(self) -> None:
        """Codex's auto-injected AGENTS.md (role=user) is primed context, not
        a typed user turn -- it maps to SystemMessage so the UI can hide it.
        """
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "# AGENTS.md instructions for /repo\n\n<INSTRUCTIONS>",
                        }
                    ],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, SystemMessage)
        assert event.message.role == "user"

    def test_event_msg_is_skipped(self) -> None:
        line = _encode(
            {
                "type": "event_msg",
                "payload": {"type": "agent_message", "message": "hello"},
            }
        )
        assert _parse_one(line) is None

    def test_response_item_user_message(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, UserMessage)
        assert event.message.text == "hi"

    def test_response_item_assistant_message(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hello"}],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, AssistantMessage)
        assert event.message.text == "hello"

    def test_developer_role_is_system_message(self) -> None:
        """A ``developer``-role message is primed context, not a model reply.

        Regression: it used to fall into the assistant branch and render as a
        bogus model turn (the permissions/sandbox preamble the CLI hides).
        """
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions>"}],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, SystemMessage)
        assert event.message.text == "<permissions>"
        assert event.message.role == "developer"

    def test_system_role_is_system_message(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "you are codex"}],
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, SystemMessage)
        assert event.message.role == "system"

    def test_response_item_reasoning_is_assistant(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Clarifying"}],
                    "encrypted_content": "ENC",
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, AssistantMessage)
        assert event.message.thinking == "Clarifying"
        assert event.message.thinking_encrypted == "ENC"

    def test_response_item_function_call(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": "exec_command",
                    "arguments": '{"cmd":"ls"}',
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, AssistantMessage)
        assert len(event.message.tool_calls) == 1
        call = event.message.tool_calls[0]
        assert call.id == "c1"
        assert call.name == "exec_command"
        assert call.args == {"cmd": "ls"}

    def test_response_item_function_call_output(self) -> None:
        line = _encode(
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "x",
                    "output": "ok",
                },
            }
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, ToolResult)
        assert event.message.call_id == "x"
        assert event.message.content == "ok"

    def test_compacted_outer_type_is_compaction(self) -> None:
        line = _encode(
            {"type": "compacted", "payload": {"summary": "condensed history"}}
        )
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, Compaction)
        assert event.message.text == "condensed history"

    def test_turn_context_is_skipped(self) -> None:
        line = _encode({"type": "turn_context", "payload": {"model": "gpt-5.5"}})
        assert _parse_one(line) is None

    def test_turn_context_model_stamps_following_events(self) -> None:
        """A ``turn_context`` model carries onto subsequent message events.

        Codex writes a ``turn_context`` line (``payload.model``) before the
        turn's response items; the per-turn model belongs on every following
        Event until the next ``turn_context``. Dropping it leaves
        ``Event.model`` None and the agent_session_events.model column NULL.

        A fresh adapter is used because the model carry is per-instance state
        across ``parse`` calls; the shared singleton must not leak it into
        unrelated tests.
        """
        local = CodexAdapter()
        assert (
            list(
                local.parse(
                    _encode({"type": "turn_context", "payload": {"model": "gpt-5.5"}}),
                    whole_file=False,
                )
            )
            == []
        )
        events = list(
            local.parse(
                _encode(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hi"}],
                        },
                    }
                ),
                whole_file=False,
            )
        )
        assert len(events) == 1
        assert events[0].model == "gpt-5.5"

    def test_unknown_role_is_unknown_message(self) -> None:
        """An unrecognized message role maps to UnknownMessage, not Assistant.

        A future role (``tool`` / ``function``) must not be mislabeled as a
        model reply; it falls through to :class:`UnknownMessage` so the kind is
        honest and the raw record is preserved.
        """
        event = _parse_one(
            _encode(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "tool",
                        "content": [{"type": "output_text", "text": "x"}],
                    },
                }
            )
        )
        assert event is not None
        assert isinstance(event.message, UnknownMessage)

    def test_response_item_unknown_payload_is_unknown(self) -> None:
        line = _encode({"type": "response_item", "payload": {"type": "something_new"}})
        event = _parse_one(line)
        assert event is not None
        assert isinstance(event.message, UnknownMessage)

    def test_malformed_json_returns_none(self) -> None:
        assert _parse_one(b"{not json}") is None


if __name__ == "__main__":
    from trackinizer.lib.testing.main import test_main

    test_main(__file__)

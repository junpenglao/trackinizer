"""Tests for Antigravity: sanitized transcript fixtures to typed messages."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import json


if TYPE_CHECKING:
    import pytest

from trackinizer.trax.run.adapters.antigravity import AntigravityAdapter
from trackinizer.trax.run.adapters.base import Event
from trackinizer.types.agent_session_events import (
    AssistantMessage,
    Compaction,
    SystemMessage,
    ToolResult,
    UnknownMessage,
    UserMessage,
)


CONVERSATION_ID = "88bcf1db-0fa1-4092-9b24-f7ada0920617"


def _encode(obj: object) -> bytes:
    return (json.dumps(obj) + "\n").encode()


def _parse_one(raw: bytes) -> Event | None:
    events = list(AntigravityAdapter().parse(raw))
    assert len(events) <= 1, events
    return events[0] if events else None


def _record(
    *,
    step_index: object,
    source: str,
    record_type: str,
    status: str = "DONE",
    created_at: str = "2026-07-20T06:52:03Z",
    **payload: object,
) -> bytes:
    """A sanitized Antigravity 1.1.7 ``transcript_full.jsonl`` record."""
    return _encode(
        {
            "step_index": step_index,
            "source": source,
            "type": record_type,
            "status": status,
            "created_at": created_at,
            **payload,
        }
    )


class TestAntigravityTranscriptPath:
    def test_session_dirs_returns_global_brain_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        adapter = AntigravityAdapter()
        assert tuple(adapter.session_dirs()) == ()

        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        brain.mkdir(parents=True)
        assert tuple(adapter.session_dirs()) == (brain,)

    def test_matches_only_full_transcript_at_exact_conversation_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        adapter = AntigravityAdapter()
        logs = (
            tmp_path
            / ".gemini"
            / "antigravity-cli"
            / "brain"
            / CONVERSATION_ID
            / ".system_generated"
            / "logs"
        )

        assert adapter.matches_session_file(logs / "transcript_full.jsonl")
        # Antigravity writes this truncated mirror beside the full transcript.
        # Matching both would duplicate every turn.
        assert not adapter.matches_session_file(logs / "transcript.jsonl")
        assert not adapter.matches_session_file(logs / "other.jsonl")

    def test_rejects_wrong_root_shape_and_noncanonical_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        adapter = AntigravityAdapter()
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"

        assert not adapter.matches_session_file(
            brain
            / "not-a-conversation-id"
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )
        assert not adapter.matches_session_file(
            brain / CONVERSATION_ID / "logs" / "transcript_full.jsonl"
        )
        assert not adapter.matches_session_file(
            brain
            / CONVERSATION_ID.upper()
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )
        assert not adapter.matches_session_file(
            tmp_path
            / "elsewhere"
            / CONVERSATION_ID
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )

    def test_extracts_exact_native_conversation_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        adapter = AntigravityAdapter()
        path = (
            tmp_path
            / ".gemini"
            / "antigravity-cli"
            / "brain"
            / CONVERSATION_ID
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )
        assert adapter.session_id_from_path(path) == CONVERSATION_ID
        assert adapter.session_id_from_path(path.with_name("transcript.jsonl")) is None

    def test_rejects_symlinked_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        logs = (
            tmp_path
            / ".gemini"
            / "antigravity-cli"
            / "brain"
            / CONVERSATION_ID
            / ".system_generated"
            / "logs"
        )
        logs.mkdir(parents=True)
        outside = tmp_path / "outside.jsonl"
        outside.write_text("{}\n")
        transcript = logs / "transcript_full.jsonl"
        transcript.symlink_to(outside)

        adapter = AntigravityAdapter()
        assert not adapter.matches_session_file(transcript)
        assert adapter.session_id_from_path(transcript) is None

    def test_rejects_symlinked_conversation_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        brain = tmp_path / ".gemini" / "antigravity-cli" / "brain"
        brain.mkdir(parents=True)
        outside = tmp_path / "outside-conversation"
        logs = outside / ".system_generated" / "logs"
        logs.mkdir(parents=True)
        (logs / "transcript_full.jsonl").write_text("{}\n")
        (brain / CONVERSATION_ID).symlink_to(outside, target_is_directory=True)
        transcript = (
            brain
            / CONVERSATION_ID
            / ".system_generated"
            / "logs"
            / "transcript_full.jsonl"
        )

        adapter = AntigravityAdapter()
        assert not adapter.matches_session_file(transcript)
        assert adapter.session_id_from_path(transcript) is None


class TestAntigravityParseLine:
    def test_user_input_is_user_message_with_timestamp(self) -> None:
        event = _parse_one(
            _record(
                step_index=0,
                source="USER_EXPLICIT",
                record_type="USER_INPUT",
                content="Investigate the sampler.",
            )
        )
        assert event is not None
        assert isinstance(event.message, UserMessage)
        assert event.message.text == "Investigate the sampler."
        assert event.timestamp == datetime(2026, 7, 20, 6, 52, 3, tzinfo=UTC)

    def test_planner_response_is_one_assistant_turn(self) -> None:
        event = _parse_one(
            _record(
                step_index=3,
                source="MODEL",
                record_type="PLANNER_RESPONSE",
                content="I found the cause.",
                thinking="Compare both implementations.",
            )
        )
        assert event is not None
        assert isinstance(event.message, AssistantMessage)
        assert event.message.text == "I found the cause."
        assert event.message.thinking == "Compare both implementations."

    def test_generic_record_stays_unknown_despite_preceding_tool_call(self) -> None:
        adapter = AntigravityAdapter()
        call_events = list(
            adapter.parse(
                _record(
                    step_index=2,
                    source="MODEL",
                    record_type="PLANNER_RESPONSE",
                    tool_calls=[{"name": "list_permissions", "args": {}}],
                )
            )
        )
        result_events = list(
            adapter.parse(
                _record(
                    step_index=3,
                    source="MODEL",
                    record_type="GENERIC",
                    content="permission result",
                )
            )
        )

        assert len(call_events) == 1
        call_message = call_events[0].message
        assert isinstance(call_message, AssistantMessage)
        assert call_message.tool_calls[0].id == "agy-step-3"
        assert len(result_events) == 1
        result_message = result_events[0].message
        # GENERIC is result-shaped in the observed corpus, but has no stable
        # provider discriminator. Preserve it raw instead of sharing parser
        # context across independently drained transcript files.
        assert isinstance(result_message, UnknownMessage)

    def test_tool_call_and_result_share_provider_step_identity(self) -> None:
        call_event = _parse_one(
            _record(
                step_index=7,
                source="MODEL",
                record_type="PLANNER_RESPONSE",
                thinking="Run the focused test.",
                tool_calls=[
                    {
                        "name": "run_command",
                        "args": {
                            "CommandLine": "pytest -q sampler_test.py",
                            "Cwd": "/workspace",
                            "toolAction": "Run",
                            "toolSummary": "Focused tests",
                        },
                    }
                ],
            )
        )
        result_event = _parse_one(
            _record(
                step_index=8,
                source="MODEL",
                record_type="RUN_COMMAND",
                content="1 passed",
            )
        )

        assert call_event is not None
        assert isinstance(call_event.message, AssistantMessage)
        assert call_event.message.thinking == "Run the focused test."
        assert len(call_event.message.tool_calls) == 1
        call = call_event.message.tool_calls[0]
        assert call.id == "agy-step-8"
        assert call.name == "run_command"
        assert call.args["CommandLine"] == "pytest -q sampler_test.py"

        assert result_event is not None
        assert isinstance(result_event.message, ToolResult)
        assert result_event.message.call_id == call.id
        assert result_event.message.content == "1 passed"
        assert result_event.message.is_error is False

    def test_failed_tool_result_is_marked_as_error(self) -> None:
        event = _parse_one(
            _record(
                step_index=8,
                source="MODEL",
                record_type="RUN_COMMAND",
                status="FAILED",
                content="command failed",
            )
        )
        assert event is not None
        assert isinstance(event.message, ToolResult)
        assert event.message.is_error is True

    def test_running_tool_record_is_preserved_as_unknown(self) -> None:
        event = _parse_one(
            _record(
                step_index=8,
                source="MODEL",
                record_type="RUN_COMMAND",
                status="RUNNING",
                content="command still running",
            )
        )
        assert event is not None
        assert isinstance(event.message, UnknownMessage)

    def test_each_observed_tool_result_type_is_normalized(self) -> None:
        for record_type in (
            "CODE_ACTION",
            "LIST_DIRECTORY",
            "RUN_COMMAND",
            "VIEW_FILE",
        ):
            event = _parse_one(
                _record(
                    step_index=12,
                    source="MODEL",
                    record_type=record_type,
                    content=f"{record_type} result",
                )
            )
            assert event is not None
            assert isinstance(event.message, ToolResult), record_type
            assert event.message.call_id == "agy-step-12"

    def test_invalid_step_index_does_not_forge_tool_identity(self) -> None:
        for invalid in (None, True, "7", -1):
            event = _parse_one(
                _record(
                    step_index=invalid,
                    source="MODEL",
                    record_type="RUN_COMMAND",
                    content="result",
                )
            )
            assert event is not None
            assert isinstance(event.message, UnknownMessage), invalid

    def test_multiple_tool_calls_do_not_invent_future_step_ids(self) -> None:
        event = _parse_one(
            _record(
                step_index=7,
                source="MODEL",
                record_type="PLANNER_RESPONSE",
                tool_calls=[
                    {"name": "first", "args": {}},
                    {"name": "second", "args": {}},
                ],
            )
        )
        assert event is not None
        assert isinstance(event.message, UnknownMessage)

    def test_checkpoint_is_compaction(self) -> None:
        event = _parse_one(
            _record(
                step_index=4,
                source="SYSTEM",
                record_type="CHECKPOINT",
                content="Condensed conversation state.",
            )
        )
        assert event is not None
        assert isinstance(event.message, Compaction)
        assert event.message.text == "Condensed conversation state."

    def test_system_message_is_system_context(self) -> None:
        event = _parse_one(
            _record(
                step_index=5,
                source="SYSTEM",
                record_type="SYSTEM_MESSAGE",
                content="The background command is still running.",
            )
        )
        assert event is not None
        assert isinstance(event.message, SystemMessage)
        assert event.message.text == "The background command is still running."

    def test_empty_conversation_history_marker_is_skipped(self) -> None:
        assert (
            _parse_one(
                _record(
                    step_index=1,
                    source="SYSTEM",
                    record_type="CONVERSATION_HISTORY",
                )
            )
            is None
        )

    def test_unrecognized_record_is_unknown(self) -> None:
        event = _parse_one(
            _record(
                step_index=9,
                source="MODEL",
                record_type="FUTURE_STEP",
                content="new shape",
            )
        )
        assert event is not None
        assert isinstance(event.message, UnknownMessage)

    def test_malformed_json_returns_none(self) -> None:
        assert _parse_one(b"{not json}") is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_one(b"[]") is None

    def test_invalid_timestamp_is_ignored(self) -> None:
        event = _parse_one(
            _record(
                step_index=0,
                source="USER_EXPLICIT",
                record_type="USER_INPUT",
                created_at="not-a-timestamp",
                content="hello",
            )
        )
        assert event is not None
        assert event.timestamp is None


if __name__ == "__main__":
    from trackinizer.lib.testing.main import test_main

    test_main(__file__)

"""Tests for exact pre-launch binding of Codex resumes."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import NoReturn
from uuid import UUID

import json
import os
import shutil
import sqlite3
import subprocess

import pytest

from trackinizer.trax.run.adapters.claude import ClaudeAdapter
from trackinizer.trax.run.adapters.codex import CodexAdapter
from trackinizer.trax.run.resume_binding import (
    ResumeBindingError,
    prepare_explicit_resume,
)


CODEX_ID = "019fa8a4-d706-7133-816a-c30a919de837"
CLAUDE_ID = "4f8a5fcc-8f12-4bec-9e77-1e31ff57e510"
_CODEX_BINARY = shutil.which("codex")


def _codex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[CodexAdapter, Path, Path]:
    home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(home))
    transcript = (
        home
        / "sessions"
        / "2026"
        / "07"
        / "28"
        / f"rollout-2026-07-28T14-13-23-{CODEX_ID}.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(
        (
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {"id": CODEX_ID, "session_id": CODEX_ID},
                }
            )
            + "\n"
        ).encode()
    )
    database = home / "state_5.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO threads (id, rollout_path) VALUES (?, ?)",
            (CODEX_ID, str(transcript)),
        )
    return CodexAdapter(), transcript, database


def _append(path: Path, content: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(content)


def _claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ClaudeAdapter, Path]:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    transcript = (
        home / ".claude" / "projects" / "-workspace" / f"{CLAUDE_ID}.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_bytes(b'{"type":"user","message":{"content":"old"}}\n')
    return ClaudeAdapter(), transcript


@pytest.mark.parametrize(
    "args",
    [
        ("--resume", CLAUDE_ID),
        ("-r", CLAUDE_ID),
        (f"--resume={CLAUDE_ID}",),
        (f"-r{CLAUDE_ID}",),
        ("--model", "sonnet", "--resume", CLAUDE_ID),
        ("--resume", CLAUDE_ID, "continue this"),
    ],
)
def test_claude_explicit_resume_selectors_preserve_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    adapter, _transcript = _claude(tmp_path, monkeypatch)
    prepared = prepare_explicit_resume(adapter, args, cwd=tmp_path)

    assert prepared is not None
    with prepared:
        assert prepared.cli_args == args
        assert prepared.expected_cli_session_id == CLAUDE_ID


@pytest.mark.parametrize(
    "args",
    [
        (),
        ("hello",),
        ("--from-pr", "42"),
        ("--", "--resume", CLAUDE_ID),
    ],
)
def test_fresh_claude_command_does_not_claim_a_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    adapter, _transcript = _claude(tmp_path, monkeypatch)

    assert prepare_explicit_resume(adapter, args, cwd=tmp_path) is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--resume",), "explicit"),
        (("--resume", "conversation-name"), "canonical UUID"),
        (("--resume", "--model", "sonnet"), "explicit"),
        (("--continue",), "explicit UUID"),
        (("-c",), "explicit UUID"),
        (("--resume", CLAUDE_ID, "--continue"), "also use --continue"),
        (("--resume", CLAUDE_ID, "--fork-session"), "new identity"),
        (("--fork-session", "--resume", CLAUDE_ID), "new identity"),
    ],
)
def test_ambiguous_claude_resume_selectors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
    message: str,
) -> None:
    adapter, _transcript = _claude(tmp_path, monkeypatch)

    with pytest.raises(ResumeBindingError, match=message):
        prepare_explicit_resume(adapter, args, cwd=tmp_path)


def test_claude_resume_snapshot_emits_only_the_post_launch_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, transcript = _claude(tmp_path, monkeypatch)
    prepared = prepare_explicit_resume(
        adapter, ("--resume", CLAUDE_ID), cwd=tmp_path
    )

    assert prepared is not None
    with prepared:
        _append(transcript, b'{"type":"assistant","message":{"content":"new"}}\n')

        reader = prepared.poll()
        assert reader.cli_session_id == CLAUDE_ID
        assert reader.read_lines() == (
            b'{"type":"assistant","message":{"content":"new"}}',
        )
        assert reader.read_lines() == ()
        assert prepared.finish() is reader
        reader.finish()


def test_duplicate_claude_resume_transcripts_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, transcript = _claude(tmp_path, monkeypatch)
    duplicate = transcript.parent.parent / "-other" / transcript.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(transcript.read_bytes())

    with pytest.raises(ResumeBindingError, match="multiple Claude"):
        prepare_explicit_resume(
            adapter, ("--resume", CLAUDE_ID), cwd=tmp_path
        )


def _forbid_enumeration(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("exact resume resolution must not enumerate archives")


@pytest.mark.parametrize(
    "args",
    [
        ("resume", CODEX_ID),
        ("exec", "resume", CODEX_ID),
        ("e", "resume", CODEX_ID),
        ("-m", "resume", "resume", CODEX_ID),
        ("--image=/tmp/prompt.png", "resume", CODEX_ID),
        ("resume", "--model", "resume", CODEX_ID),
        (
            "resume",
            "-i",
            "prompt.png",
            "--model",
            "gpt-5",
            CODEX_ID,
        ),
        ("resume", "--", CODEX_ID, "continue this"),
        (
            "exec",
            "--json",
            "--color",
            "never",
            "--sandbox",
            "read-only",
            "-c",
            "model_reasoning_summary=detailed",
            "resume",
            CODEX_ID,
            "continue this",
        ),
    ],
)
def test_codex_explicit_resume_selectors_preserve_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    adapter, _transcript, _database = _codex(tmp_path, monkeypatch)
    prepared = prepare_explicit_resume(adapter, args, cwd=tmp_path)

    assert prepared is not None
    with prepared:
        assert prepared.cli_args == args
        assert prepared.expected_cli_session_id == CODEX_ID


@pytest.mark.parametrize(
    "args",
    [
        ("exec", "--json", "hello"),
        ("--model", "resume", "hello"),
        ("exec", "--model", "resume", "hello"),
        ("-i", "prompt.png", "resume", CODEX_ID),
        ("exec", "-i", "prompt.png", "resume", CODEX_ID),
        ("--", "resume", CODEX_ID),
        ("exec", "--", "resume", CODEX_ID),
        ("review", "resume", CODEX_ID),
        ("fork", "--last"),
        ("fork", CODEX_ID),
        ("--help", "resume", CODEX_ID),
        ("resume", CODEX_ID, "--help"),
        ("exec", "resume", CODEX_ID, "--help"),
    ],
)
def test_fresh_codex_command_does_not_claim_a_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
) -> None:
    adapter, _transcript, _database = _codex(tmp_path, monkeypatch)

    assert prepare_explicit_resume(adapter, args, cwd=tmp_path) is None


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("resume",), "explicit"),
        (("resume", "conversation-name"), "UUID"),
        (("resume", "--last"), "explicit"),
        (("exec", "--ephemeral", "resume", CODEX_ID), "ephemeral"),
        (("exec", "resume", "--ephemeral", CODEX_ID), "ephemeral"),
        (("exec", "resume", CODEX_ID, "--ephemeral"), "ephemeral"),
        (("resume", CODEX_ID, "--last"), "explicit"),
        (("resume", CODEX_ID, "--remote", "ws://localhost:4500"), "remote"),
        (("--remote", "ws://localhost:4500", "resume", CODEX_ID), "remote"),
        (
            ("exec", "--remote", "ws://localhost:4500", "resume", CODEX_ID),
            "remote",
        ),
        (("resume", "-i", "prompt.png", CODEX_ID), "explicit"),
        (("--future-option", "resume", CODEX_ID), "interpret"),
        (("--json", "resume", CODEX_ID), "interpret"),
        (
            ("exec", "resume", "--ask-for-approval", "never", CODEX_ID),
            "interpret",
        ),
    ],
)
def test_ambiguous_or_unstable_codex_selectors_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
    message: str,
) -> None:
    adapter, _transcript, _database = _codex(tmp_path, monkeypatch)

    with pytest.raises(ResumeBindingError, match=message):
        prepare_explicit_resume(adapter, args, cwd=tmp_path)


def test_resume_snapshot_emits_only_the_post_launch_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, transcript, _database = _codex(tmp_path, monkeypatch)
    args = ("exec", "resume", CODEX_ID)
    prepared = prepare_explicit_resume(adapter, args, cwd=tmp_path)

    assert prepared is not None
    with prepared:
        _append(transcript, b'{"type":"user","message":{"content":"new"}}\n')

        reader = prepared.poll()
        assert reader is not None
        assert reader.cli_session_id == CODEX_ID
        assert reader.read_lines() == (b'{"type":"user","message":{"content":"new"}}',)
        assert reader.read_lines() == ()
        assert prepared.finish() is reader
        reader.finish()


def test_codex_index_path_outside_provider_root_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _transcript, database = _codex(tmp_path, monkeypatch)
    outside = tmp_path / f"rollout-2026-07-28T14-13-23-{CODEX_ID}.jsonl"
    outside.write_bytes(
        (
            json.dumps({"type": "session_meta", "payload": {"id": CODEX_ID}}) + "\n"
        ).encode()
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE threads SET rollout_path = ? WHERE id = ?",
            (str(outside), CODEX_ID),
        )

    with pytest.raises(RuntimeError, match=r"outside|provider root"):
        prepare_explicit_resume(
            adapter,
            ("exec", "resume", CODEX_ID),
            cwd=tmp_path,
        )


def test_codex_exact_lookup_does_not_scan_1000_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter, _transcript, database = _codex(tmp_path, monkeypatch)
    with sqlite3.connect(database) as connection:
        connection.executemany(
            "INSERT INTO threads (id, rollout_path) VALUES (?, ?)",
            (
                (
                    str(UUID(int=index)),
                    str(tmp_path / "unrelated" / f"{index}.jsonl"),
                )
                for index in range(1, 1001)
            ),
        )

    for owner, name in (
        (Path, "glob"),
        (Path, "rglob"),
        (Path, "iterdir"),
        (os, "walk"),
        (os, "scandir"),
        (os, "listdir"),
    ):
        monkeypatch.setattr(owner, name, _forbid_enumeration)

    started = perf_counter()
    prepared = prepare_explicit_resume(
        adapter,
        ("exec", "resume", CODEX_ID),
        cwd=tmp_path,
    )
    elapsed = perf_counter() - started

    assert prepared is not None
    with prepared:
        assert prepared.expected_cli_session_id == CODEX_ID

    assert elapsed <= 0.005


@pytest.mark.skipif(_CODEX_BINARY is None, reason="Codex is not installed")
@pytest.mark.parametrize(
    "args",
    [
        ("--help", "resume", CODEX_ID),
        ("resume", CODEX_ID, "--help"),
        ("exec", "resume", CODEX_ID, "--help"),
        ("fork", "--help"),
    ],
)
def test_codex_help_grammar_oracle_never_starts_a_session(
    args: tuple[str, ...],
) -> None:
    assert _CODEX_BINARY is not None
    completed = subprocess.run(  # noqa: S603 - installed Codex, fixed help argv
        [_CODEX_BINARY, *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=5.0,
    )

    assert completed.returncode == 0

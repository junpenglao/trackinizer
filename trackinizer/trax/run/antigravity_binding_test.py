"""Tests for Antigravity's provider-owned transcript proof."""

from __future__ import annotations

from pathlib import Path

import json
import os
import time

import pytest

from trackinizer.trax.run import antigravity_binding as binding_mod
from trackinizer.trax.run.adapters.antigravity import AntigravityAdapter
from trackinizer.trax.run.antigravity_binding import (
    AntigravityBindingError,
    prepare_antigravity,
)
from trackinizer.trax.run.transcript import TranscriptError


ID_A = "88bcf1db-0fa1-4092-9b24-f7ada0920617"
ID_B = "99bcf1db-0fa1-4092-9b24-f7ada0920618"


def _adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AntigravityAdapter:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return AntigravityAdapter()


def _transcript(
    adapter: AntigravityAdapter, identity: str, content: bytes = b""
) -> Path:
    path = adapter.transcript_path(identity)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _cache(adapter: AntigravityAdapter, mapping: dict[str, object]) -> Path:
    path = adapter.brain_dir.parent / "cache" / "last_conversations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping))
    path.chmod(0o600)
    return path


def _marker(kind: str, identity: str) -> bytes:
    return (f"I0728 09:13:40.123456 123 main.go:42] {kind} {identity}\n").encode()


def _append(path: Path, content: bytes) -> None:
    with path.open("ab") as stream:
        stream.write(content)


class TestPrepareAntigravity:
    def test_explicit_resume_pins_prelaunch_eof(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        transcript = _transcript(adapter, ID_A, b"old\n")

        with prepare_antigravity(
            adapter,
            (f"--conversation={ID_A}", "--model", "future-model"),
            cwd=tmp_path,
            runtime_root=tmp_path / "runtime",
        ) as prepared:
            assert prepared.expected_cli_session_id == ID_A
            assert prepared.cli_args == (
                "--log-file",
                str(prepared.diagnostic_path),
                f"--conversation={ID_A}",
                "--model",
                "future-model",
            )
            _append(transcript, b"new\n")
            _append(prepared.diagnostic_path, _marker("Resuming conversation", ID_A))

            reader = prepared.poll()
            assert reader is not None
            assert reader.cli_session_id == ID_A
            assert reader.read_lines() == (b"new",)

    @pytest.mark.parametrize(
        ("selectors", "rewritten"),
        [
            (("-c",), ("--conversation", ID_A)),
            (("--continue=true",), ("--conversation", ID_A)),
            (
                ("--continue=false", "--continue=true"),
                ("--conversation", ID_A),
            ),
            (
                ("--new-project=true", "--new-project=false", "--continue"),
                (
                    "--new-project=true",
                    "--new-project=false",
                    "--conversation",
                    ID_A,
                ),
            ),
        ],
    )
    def test_continue_is_pinned_to_cached_conversation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        selectors: tuple[str, ...],
        rewritten: tuple[str, ...],
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        cwd = tmp_path / "project"
        cwd.mkdir()
        transcript = _transcript(adapter, ID_A, b"old\n")
        _cache(adapter, {str(cwd): ID_A})

        with prepare_antigravity(
            adapter,
            ("--model", "gemini-3-pro", *selectors, "--print"),
            cwd=cwd,
            runtime_root=tmp_path / "runtime",
        ) as prepared:
            assert prepared.expected_cli_session_id == ID_A
            assert prepared.cli_args[2:] == (
                "--model",
                "gemini-3-pro",
                *rewritten,
                "--print",
            )
            _append(transcript, b"continued\n")
            _append(
                prepared.diagnostic_path,
                _marker("Print mode: resuming conversation", ID_A),
            )

            reader = prepared.poll()
            assert reader is not None
            assert reader.read_lines() == (b"continued",)

    def test_continue_without_cache_entry_becomes_confirmed_fresh(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        cwd = tmp_path / "new-project"
        cwd.mkdir()
        _cache(adapter, {})

        with prepare_antigravity(
            adapter,
            ("--continue",),
            cwd=cwd,
            runtime_root=tmp_path / "runtime",
        ) as prepared:
            assert prepared.expected_cli_session_id is None
            assert "--continue" not in prepared.cli_args
            assert "--conversation" not in prepared.cli_args
            transcript = _transcript(adapter, ID_A, b"system\nuser\n")
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))

            reader = prepared.poll()
            assert reader is not None
            assert reader.read_lines() == (b"system", b"user")
            assert transcript == reader.path

    def test_concurrent_fresh_runs_bind_only_their_owned_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        runtime = tmp_path / "runtime"

        with (
            prepare_antigravity(
                adapter, (), cwd=tmp_path, runtime_root=runtime
            ) as first,
            prepare_antigravity(
                adapter, (), cwd=tmp_path, runtime_root=runtime
            ) as second,
        ):
            assert first.diagnostic_path != second.diagnostic_path
            assert first.diagnostic_path.stat().st_mode & 0o777 == 0o600
            first_path = first.diagnostic_path
            second_path = second.diagnostic_path
            _transcript(adapter, ID_A, b"a\n")
            _transcript(adapter, ID_B, b"b\n")
            _append(first.diagnostic_path, _marker("Created conversation", ID_A))
            _append(second.diagnostic_path, _marker("Created conversation", ID_B))

            first_reader = first.poll()
            second_reader = second.poll()
            assert first_reader is not None
            assert second_reader is not None
            assert first_reader.cli_session_id == ID_A
            assert second_reader.cli_session_id == ID_B
            assert first_reader.read_lines() == (b"a",)
            assert second_reader.read_lines() == (b"b",)
        assert not first_path.exists()
        assert not second_path.exists()

    def test_marker_can_arrive_before_fresh_transcript(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))
            assert prepared.poll() is None

            _transcript(adapter, ID_A, b"first\n")
            reader = prepared.poll()
            assert reader is not None
            assert reader.read_lines() == (b"first",)

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (("--conversation",), "requires"),
            (("--conversation", "not-a-uuid"), "canonical"),
            (("--conversation", ID_A, "--conversation", ID_A), "repeated"),
            (("--conversation", ID_A, "--continue"), "combine"),
            (("--log-file", "user.log"), "log-file"),
            (("--log-file=user.log",), "log-file"),
            (("--future-option", "value"), "safely interpret"),
            (("--continue=tRuE",), "true or false"),
        ],
    )
    def test_ambiguous_or_unowned_arguments_fail_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        args: tuple[str, ...],
        message: str,
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with pytest.raises(AntigravityBindingError, match=message):
            prepare_antigravity(
                adapter, args, cwd=tmp_path, runtime_root=tmp_path / "runtime"
            )

    @pytest.mark.parametrize(
        "args",
        [
            ("--add-dir", "--continue", "help"),
            ("--project", "--continue", "help"),
            ("help", "--continue"),
            ("--continue=false",),
            ("--continue=true", "--continue=false"),
        ],
    )
    def test_selector_like_values_and_positionals_are_untouched(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        args: tuple[str, ...],
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, args, cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            assert prepared.expected_cli_session_id is None
            assert prepared.cli_args[2:] == args

    @pytest.mark.parametrize("selector", ["-c", "--continue"])
    @pytest.mark.parametrize("project", ["--project", "--new-project"])
    def test_continue_with_project_selection_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        selector: str,
        project: str,
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        args = (
            (selector, project)
            if project == "--new-project"
            else (
                selector,
                project,
                "project-id",
            )
        )
        with pytest.raises(AntigravityBindingError, match="project"):
            prepare_antigravity(
                adapter, args, cwd=tmp_path, runtime_root=tmp_path / "runtime"
            )

    def test_same_explicit_conversation_cannot_be_prepared_twice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"old\n")

        with (
            prepare_antigravity(
                adapter,
                ("--conversation", ID_A),
                cwd=tmp_path,
                runtime_root=tmp_path / "runtime",
            ),
            pytest.raises(TranscriptError, match="already captured"),
        ):
            prepare_antigravity(
                adapter,
                ("--conversation", ID_A),
                cwd=tmp_path,
                runtime_root=tmp_path / "runtime",
            )


class TestAntigravityProof:
    def test_resume_marker_must_match_requested_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"old\n")
        with prepare_antigravity(
            adapter,
            ("--conversation", ID_A),
            cwd=tmp_path,
            runtime_root=tmp_path / "runtime",
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Resuming conversation", ID_B))
            with pytest.raises(AntigravityBindingError, match="requested"):
                prepared.poll()

    def test_fresh_launch_rejects_resume_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Resuming conversation", ID_A))
            with pytest.raises(AntigravityBindingError, match="fresh"):
                prepared.poll()

    def test_malformed_marker_is_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(
                prepared.diagnostic_path,
                f"Created conversation {ID_A}\n".encode(),
            )
            assert prepared.poll() is None

    def test_partial_marker_is_buffered_until_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"first\n")
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            marker = _marker("Created conversation", ID_A)
            _append(prepared.diagnostic_path, marker[:-1])
            assert prepared.poll() is None

            _append(prepared.diagnostic_path, marker[-1:])
            reader = prepared.poll()
            assert reader is not None
            assert reader.cli_session_id == ID_A

    def test_later_proof_marker_is_ambiguous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"first\n")
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))
            assert prepared.poll() is not None
            _append(
                prepared.diagnostic_path,
                _marker("Created conversation", ID_B),
            )
            with pytest.raises(AntigravityBindingError, match="ambiguous"):
                prepared.poll()

    def test_invalid_proof_permanently_poisons_binding(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Resuming conversation", ID_A))
            with pytest.raises(AntigravityBindingError, match="fresh"):
                prepared.poll()

            _transcript(adapter, ID_B, b"must not bind\n")
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_B))
            with pytest.raises(AntigravityBindingError, match="fresh"):
                prepared.poll()

    def test_large_diagnostic_backlog_is_bounded_per_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"first\n")
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, b"unrelated diagnostic\n" * 50_000)
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))

            started = time.perf_counter()
            assert prepared.poll() is None
            assert time.perf_counter() - started <= 0.005
            assert prepared.finish().cli_session_id == ID_A

    def test_finish_seals_transcript_before_return(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        transcript = _transcript(adapter, ID_A, b"captured\n")
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))
            reader = prepared.finish()
            _append(transcript, b"later\n")

            assert reader.read_lines() == (b"captured",)
            assert reader.caught_up()
            reader.finish()

    def test_finish_seals_diagnostic_before_final_poll(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        _transcript(adapter, ID_A, b"captured\n")
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, _marker("Created conversation", ID_A))
            diagnostic_type = (  # pyright: ignore[reportPrivateUsage]
                binding_mod._Diagnostic
            )
            read_lines = diagnostic_type.read_lines
            appended = False

            def read_then_append(
                diagnostic: binding_mod._Diagnostic,  # pyright: ignore[reportPrivateUsage]
            ) -> tuple[bytes, ...]:
                nonlocal appended
                lines = read_lines(diagnostic)
                if not appended:
                    appended = True
                    _append(
                        diagnostic.path,
                        _marker("Created conversation", ID_B),
                    )
                return lines

            monkeypatch.setattr(diagnostic_type, "read_lines", read_then_append)
            assert prepared.finish().cli_session_id == ID_A

    @pytest.mark.parametrize(
        ("content", "message"),
        [(b"", "no conversation"), (b"I0728 partial", "partial")],
    )
    def test_finish_rejects_missing_or_partial_proof(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        content: bytes,
        message: str,
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            _append(prepared.diagnostic_path, content)
            with pytest.raises(AntigravityBindingError, match=message):
                prepared.finish()

    def test_replaced_diagnostic_is_rejected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        with prepare_antigravity(
            adapter, (), cwd=tmp_path, runtime_root=tmp_path / "runtime"
        ) as prepared:
            replacement = prepared.diagnostic_path.with_suffix(".replacement")
            replacement.write_bytes(_marker("Created conversation", ID_A))
            replacement.replace(prepared.diagnostic_path)
            with pytest.raises(AntigravityBindingError, match="replaced"):
                prepared.poll()


def test_diagnostic_seal_ignores_later_appends(tmp_path: Path) -> None:
    diagnostic = binding_mod._Diagnostic.create(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "runtime"
    )
    try:
        with pytest.raises(AntigravityBindingError, match="sealed"):
            diagnostic.finish()
        _append(diagnostic.path, b"captured\n")
        diagnostic.seal()
        _append(diagnostic.path, b"later\n")
        diagnostic.seal()

        assert diagnostic.read_lines() == (b"captured",)
        assert diagnostic.caught_up()
        diagnostic.finish()
    finally:
        diagnostic.close()


def test_diagnostic_sealed_read_rejects_zero_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diagnostic = binding_mod._Diagnostic.create(  # pyright: ignore[reportPrivateUsage]
        tmp_path / "runtime"
    )
    try:
        _append(diagnostic.path, b"x" * 65 + b"\n")
        diagnostic.seal()
        real_pread = os.pread

        def stalled_pread(fd: int, length: int, offset: int) -> bytes:
            return b"" if offset == 0 else real_pread(fd, length, offset)

        monkeypatch.setattr(os, "pread", stalled_pread)
        with pytest.raises(AntigravityBindingError, match="changed while read"):
            diagnostic.read_lines()
    finally:
        diagnostic.close()


class TestContinueCache:
    def test_unsafe_cache_permissions_fail_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        cwd = tmp_path / "project"
        cwd.mkdir()
        cache = _cache(adapter, {str(cwd): ID_A})
        cache.chmod(0o660)

        with pytest.raises(AntigravityBindingError, match="writable"):
            prepare_antigravity(
                adapter,
                ("--continue",),
                cwd=cwd,
                runtime_root=tmp_path / "runtime",
            )

    def test_conflicting_logical_and_physical_cache_entries_fail_closed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(tmp_path, monkeypatch)
        physical = tmp_path / "physical"
        physical.mkdir()
        logical = tmp_path / "logical"
        logical.symlink_to(physical, target_is_directory=True)
        _cache(adapter, {str(logical): ID_A, str(physical): ID_B})

        with pytest.raises(AntigravityBindingError, match="ambiguous"):
            prepare_antigravity(
                adapter,
                ("--continue",),
                cwd=logical,
                runtime_root=tmp_path / "runtime",
            )

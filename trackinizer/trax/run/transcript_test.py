"""Tests for exact, single-file transcript ownership."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import NoReturn, cast, override

import os
import subprocess
import sys

import pytest

from trackinizer.trax.run.adapters.base import Adapter, Event
from trackinizer.trax.run.transcript import (
    TranscriptClaim,
    TranscriptError,
    TranscriptReader,
    TranscriptSnapshot,
    snapshot_transcript,
)


class _Adapter:
    """A minimal append-only adapter whose filename is its native identity."""

    name: str = "test"
    cli_binary: str = "test"

    def __init__(self, root: Path) -> None:
        self.root = root

    def session_dirs(self) -> Iterable[Path]:
        return (self.root,)

    def matches_session_file(self, path: Path) -> bool:
        return path.suffix == ".jsonl" and self.root in path.parents

    def session_id_from_path(self, path: Path) -> str | None:
        return path.stem if self.matches_session_file(path) else None

    def session_id_from_transcript(
        self, path: Path, first_record: bytes | None
    ) -> str | None:
        del first_record
        return self.session_id_from_path(path)

    def parse(self, raw: bytes) -> Iterable[Event]:
        del raw
        return ()


class _HeldRecordAdapter(_Adapter):
    """Uses only the first record supplied from the retained descriptor."""

    @override
    def session_id_from_path(self, path: Path) -> str | None:
        raise AssertionError(f"must not reopen transcript path {path}")

    @override
    def session_id_from_transcript(
        self, path: Path, first_record: bytes | None
    ) -> str | None:
        del path
        return first_record.decode() if first_record is not None else None


def _reader(
    adapter: _Adapter,
    path: Path,
    *,
    cli_session_id: str | None = None,
    before: TranscriptSnapshot | None = None,
) -> TranscriptReader:
    return TranscriptReader.open(
        cast(Adapter, adapter),
        TranscriptClaim(
            cli_session_id=path.stem if cli_session_id is None else cli_session_id,
            path=path,
            before=before,
        ),
    )


def _forbid_enumeration(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("exact transcript ownership must never enumerate archives")


class TestTranscriptReader:
    def test_resume_reads_only_bytes_after_prelaunch_eof(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"old turn\n")
        adapter = _Adapter(root)
        before = snapshot_transcript(cast(Adapter, adapter), path, "native-id")

        with path.open("ab") as stream:
            stream.write(b"new turn\n")

        with _reader(adapter, path, before=before) as reader:
            assert reader.read_lines() == (b"new turn",)
            assert reader.read_lines() == ()

    def test_fresh_capture_starts_at_zero(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"system\nuser\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"system", b"user")

    def test_fragmented_line_is_emitted_once(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.touch()

        with _reader(_Adapter(root), path) as reader:
            with path.open("ab") as stream:
                stream.write(b"frag")
            assert reader.read_lines() == ()

            with path.open("ab") as stream:
                stream.write(b"mented\n")
            assert reader.read_lines() == (b"fragmented",)
            assert reader.read_lines() == ()

    def test_unrelated_transcript_is_never_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        mine = root / "mine.jsonl"
        other = root / "other.jsonl"
        mine.write_bytes(b"mine\n")
        other.write_bytes(b"private\n")
        monkeypatch.setattr(Path, "rglob", _forbid_enumeration)

        with _reader(_Adapter(root), mine) as reader:
            assert reader.read_lines() == (b"mine",)
            with other.open("ab") as stream:
                stream.write(b"still private\n")
            assert reader.read_lines() == ()

    def test_rejects_wrong_native_identity(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "actual.jsonl"
        path.touch()

        with pytest.raises(TranscriptError, match="identity"):
            _reader(_Adapter(root), path, cli_session_id="claimed")

    def test_rejects_empty_native_identity(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "actual.jsonl"
        path.touch()

        with pytest.raises(TranscriptError, match="empty"):
            _reader(_Adapter(root), path, cli_session_id="")

    def test_identity_is_read_from_held_descriptor(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "provider-shaped.jsonl"
        path.write_bytes(b"held-native-id\n")

        with _reader(
            _HeldRecordAdapter(root), path, cli_session_id="held-native-id"
        ) as reader:
            assert reader.cli_session_id == "held-native-id"

    def test_rejects_path_outside_provider_root(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        outside = tmp_path / "outside.jsonl"
        outside.touch()

        with pytest.raises(TranscriptError, match="provider root"):
            _reader(_Adapter(root), outside)

    def test_rejects_symlinked_file(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        target = root / "target.jsonl"
        target.touch()
        link = root / "linked.jsonl"
        link.symlink_to(target)

        with pytest.raises(TranscriptError, match="open transcript"):
            _reader(_Adapter(root), link)

    def test_rejects_symlinked_parent(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        real = root / "real"
        real.mkdir(parents=True)
        (root / "linked").symlink_to(real, target_is_directory=True)
        path = root / "linked" / "native-id.jsonl"
        path.touch()

        with pytest.raises(TranscriptError, match="open transcript"):
            _reader(_Adapter(root), path)

    def test_rejects_fifo_without_blocking(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        os.mkfifo(path)

        with pytest.raises(TranscriptError, match="regular file"):
            _reader(_Adapter(root), path)

    def test_rejects_hard_linked_file(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        original = root / "original.jsonl"
        original.touch()
        path = root / "native-id.jsonl"
        path.hardlink_to(original)

        with pytest.raises(TranscriptError, match="link"):
            _reader(_Adapter(root), path)

    def test_rejects_group_writable_file(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.touch()
        path.chmod(0o660)

        with pytest.raises(TranscriptError, match="writable"):
            _reader(_Adapter(root), path)

    def test_rejects_replaced_resume_file(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"before\n")
        adapter = _Adapter(root)
        before = snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        path.unlink()
        path.write_bytes(b"replacement\n")

        with pytest.raises(TranscriptError, match="replaced"):
            _reader(adapter, path, before=before)

    def test_rejects_rewritten_resume_boundary(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"before\n")
        adapter = _Adapter(root)
        before = snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        path.write_bytes(b"mutate\n")

        with pytest.raises(TranscriptError, match="changed"):
            _reader(adapter, path, before=before)

    def test_rejects_truncate_and_regrow_past_resume_boundary(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"old\n")
        adapter = _Adapter(root)
        before = snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        path.write_bytes(b"rewritten-prefix\nnew\n")

        with pytest.raises(TranscriptError, match="changed"):
            _reader(adapter, path, before=before)

    def test_rejects_partial_prelaunch_line(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"incomplete")
        adapter = _Adapter(root)

        with pytest.raises(TranscriptError, match="partial"):
            snapshot_transcript(cast(Adapter, adapter), path, "native-id")

    def test_rejects_truncation_after_binding(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"first\nsecond\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"first", b"second")
            path.write_bytes(b"short\n")
            with pytest.raises(TranscriptError, match="shrunk"):
                reader.read_lines()

    def test_rejects_path_replacement_after_binding(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"first\n")

        with _reader(_Adapter(root), path) as reader:
            replacement = root / "replacement"
            replacement.write_bytes(b"second\n")
            replacement.replace(path)
            with pytest.raises(TranscriptError, match="replaced"):
                reader.read_lines()

    def test_rejects_same_size_rewrite_after_binding(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"first\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"first",)
            path.write_bytes(b"other\n")
            with pytest.raises(TranscriptError, match="changed"):
                reader.read_lines()

    def test_rejects_truncate_and_regrow_past_consumed_offset(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"old\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"old",)
            path.write_bytes(b"rewritten-prefix\nnew\n")
            with pytest.raises(TranscriptError, match="changed"):
                reader.read_lines()

    def test_second_snapshot_cannot_own_same_transcript(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.touch()
        adapter = _Adapter(root)

        first = snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        try:
            with pytest.raises(TranscriptError, match="already captured"):
                snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        finally:
            first.close()

    def test_lock_excludes_another_process_and_releases_on_close(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.touch()
        adapter = _Adapter(root)
        script = (
            "import fcntl, os, sys; "
            "fd=os.open(sys.argv[1], os.O_RDONLY); "
            "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)"
        )

        snapshot = snapshot_transcript(cast(Adapter, adapter), path, "native-id")
        try:
            blocked = subprocess.run(  # noqa: S603 - fixed interpreter and script
                [sys.executable, "-c", script, str(path)],
                check=False,
                capture_output=True,
            )
            assert blocked.returncode != 0
        finally:
            snapshot.close()
        acquired = subprocess.run(  # noqa: S603 - fixed interpreter and script
            [sys.executable, "-c", script, str(path)],
            check=False,
            capture_output=True,
        )
        assert acquired.returncode == 0

    def test_hard_link_added_after_binding_is_rejected(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"one\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"one",)
            (root / "second-link").hardlink_to(path)
            with pytest.raises(TranscriptError, match="link"):
                reader.read_lines()

    def test_mixed_complete_and_fragmented_lines(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"one\ntwo-")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"one",)
            with path.open("ab") as stream:
                stream.write(b"part\nthree\n")
            assert reader.read_lines() == (b"two-part", b"three")

    def test_finish_rejects_unterminated_final_record(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b'{"type":')

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == ()
            with pytest.raises(TranscriptError, match="partial"):
                reader.finish()

    def test_finish_requires_every_complete_record_to_be_drained(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        first = b"a" * 700_000
        second = b"b" * 700_000
        path.write_bytes(first + b"\n" + second + b"\n")

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (first,)
            with pytest.raises(TranscriptError, match="unread"):
                reader.finish()

            assert not reader.caught_up()
            assert reader.read_lines() == (second,)
            assert reader.caught_up()
            reader.finish()

    def test_reads_large_suffix_in_bounded_chunks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"identity\n" + b"x" * (2 << 20))
        requested: list[int] = []
        real_pread = os.pread

        def bounded_pread(fd: int, length: int, offset: int) -> bytes:
            requested.append(length)
            return real_pread(fd, length, offset)

        monkeypatch.setattr(os, "pread", bounded_pread)
        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"identity",)
        assert max(requested) <= (1 << 20) + 1

    def test_rejects_oversized_unterminated_record(self, tmp_path: Path) -> None:
        root = tmp_path / "sessions"
        root.mkdir()
        path = root / "native-id.jsonl"
        path.write_bytes(b"identity\n" + b"x" * (17 << 20))

        with _reader(_Adapter(root), path) as reader:
            assert reader.read_lines() == (b"identity",)

            def exhaust_record_limit() -> None:
                for _ in range(17):
                    reader.read_lines()

            with pytest.raises(TranscriptError, match="16 MiB"):
                exhaust_record_limit()


def test_exact_reader_speed_does_not_scale_with_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One owned descriptor stays fast with 1,000 archived conversations."""
    root = tmp_path / "sessions"
    root.mkdir()
    for index in range(1_000):
        archive = root / f"archive-{index}"
        archive.mkdir()
        for item in range(3):
            (archive / f"ignored-{item}.jsonl").touch()
    active_dir = root / "active"
    active_dir.mkdir()
    active = active_dir / "native-id.jsonl"
    active.write_bytes(b"ready\n")
    adapter = _Adapter(root)

    monkeypatch.setattr(Path, "rglob", _forbid_enumeration)
    monkeypatch.setattr(Path, "glob", _forbid_enumeration)
    monkeypatch.setattr(Path, "iterdir", _forbid_enumeration)
    monkeypatch.setattr(os, "walk", _forbid_enumeration)
    monkeypatch.setattr(os, "scandir", _forbid_enumeration)
    monkeypatch.setattr(os, "listdir", _forbid_enumeration)
    started = perf_counter()
    reader = _reader(adapter, active)
    cold_seconds = perf_counter() - started
    assert cold_seconds <= 0.050
    try:
        assert reader.read_lines() == (b"ready",)
        real_fstat = os.fstat
        real_stat = Path.stat
        metadata_calls = 0

        def counted_fstat(fd: int) -> os.stat_result:
            nonlocal metadata_calls
            metadata_calls += 1
            return real_fstat(fd)

        def counted_stat(
            path: Path,
            *,
            follow_symlinks: bool = True,
        ) -> os.stat_result:
            nonlocal metadata_calls
            metadata_calls += 1
            return real_stat(path, follow_symlinks=follow_symlinks)

        monkeypatch.setattr(os, "fstat", counted_fstat)
        monkeypatch.setattr(Path, "stat", counted_stat)
        samples: list[float] = []
        per_tick_metadata: list[int] = []
        for _ in range(100):
            before = metadata_calls
            started = perf_counter()
            assert reader.read_lines() == ()
            samples.append(perf_counter() - started)
            per_tick_metadata.append(metadata_calls - before)
        assert sorted(samples)[94] <= 0.005
        assert max(per_tick_metadata) <= 4
    finally:
        reader.close()

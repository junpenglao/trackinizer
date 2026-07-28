"""Exact Antigravity binding at the session-runner boundary."""

from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn, cast, override
from uuid import UUID, uuid4

import os
import shutil
import sys
import threading
import time

import pytest

from trackinizer.client.client import Client
from trackinizer.trax.run import session as session_mod
from trackinizer.trax.run.adapters.antigravity import AntigravityAdapter
from trackinizer.trax.run.adapters.base import Adapter, Event
from trackinizer.trax.run.adapters.codex import CodexAdapter
from trackinizer.trax.run.antigravity_binding import PreparedAntigravity
from trackinizer.trax.run.pty_pump import PtyPump
from trackinizer.trax.run.resume_binding import PreparedResume
from trackinizer.trax.run.session import RunConfig, _Stats
from trackinizer.types.agent_session_events import SlashCommand, UserMessage


_ID = "88bcf1db-0fa1-4092-9b24-f7ada0920617"
_CONFIG = RunConfig(
    cli_name="agy",
    cli_args=("--conversation", _ID),
    sync=False,
    quiesce_seconds=0.0,
)


class _Adapter(AntigravityAdapter):
    @override
    def session_dirs(self) -> Iterable[Path]:
        raise AssertionError("exact capture must not enumerate archives")

    @override
    def parse(self, raw: bytes) -> Iterable[Event]:
        return (Event(message=UserMessage(text=raw.decode())),)


class _Codex(CodexAdapter):
    @override
    def session_dirs(self) -> Iterable[Path]:
        raise AssertionError("explicit resume must not scan provider archives")

    @override
    def parse(self, raw: bytes) -> Iterable[Event]:
        return (Event(message=UserMessage(text=raw.decode())),)


def _forbid_enumeration(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("exact capture must not enumerate provider archives")


class _Reader:
    def __init__(self, line: bytes, *, reads: int = 1) -> None:
        self.cli_session_id = _ID
        self.line = line
        self.remaining = reads
        self.reads = 0
        self.sealed = False
        self.finished = False

    def read_lines(self) -> tuple[bytes, ...]:
        self.reads += 1
        if self.remaining == 0:
            return ()
        self.remaining -= 1
        return (self.line,) if self.remaining == 0 and self.line else ()

    def caught_up(self) -> bool:
        return self.remaining == 0

    def seal(self) -> None:
        self.sealed = True

    def finish(self) -> None:
        assert self.sealed
        assert self.caught_up()
        self.finished = True


class _Sink:
    def __init__(self, *, session_id: UUID | None = None) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.emitted = threading.Event()
        self.user_emitted = threading.Event()
        self._session_id = session_id

    @property
    def session_id(self) -> UUID | None:
        return self._session_id

    def open(self) -> str:
        self.calls.append(("open", None))
        return "agent"

    def set_cli_session_id(self, cli_session_id: str) -> None:
        self.calls.append(("set", cli_session_id))

    def emit(self, adapter_name: str, event: Event) -> None:
        del adapter_name
        if isinstance(event.message, UserMessage):
            rendered = event.message.text
            is_user = True
        else:
            assert isinstance(event.message, SlashCommand)
            rendered = f"/{event.message.command}"
            if event.message.args:
                rendered += f" {event.message.args}"
            is_user = False
        self.calls.append(("emit", rendered))
        if is_user:
            self.user_emitted.set()
        self.emitted.set()

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _Prepared:
    def __init__(
        self,
        reader: _Reader | None,
        *,
        expected_id: str | None,
        error: Exception | None = None,
        poll_returns_reader: bool = True,
        poll_gate: threading.Event | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self.reader = reader
        self.expected_cli_session_id = expected_id
        self.cli_args: tuple[str, ...] = (
            "--log-file",
            "owned.log",
            "--conversation",
            _ID,
        )
        self.error = error
        self.poll_returns_reader = poll_returns_reader
        self.poll_gate = poll_gate
        self.on_close = on_close
        self.poll_entered = threading.Event()
        self.polls = 0
        self.finish_calls = 0
        self.closed = False

    def poll(self) -> _Reader | None:
        assert not self.closed
        self.polls += 1
        self.poll_entered.set()
        if self.poll_gate is not None:
            assert self.poll_gate.wait(1.0)
        if self.error is not None:
            raise self.error
        return self.reader if self.poll_returns_reader else None

    def finish(self) -> _Reader:
        assert not self.closed
        self.finish_calls += 1
        assert self.reader is not None
        self.reader.seal()
        return self.reader

    def close(self) -> None:
        if self.closed:
            return
        assert self.reader is None or self.reader.finished
        if self.on_close is not None:
            self.on_close()
        self.closed = True


class _InboundClient:
    def __init__(self) -> None:
        self.calls = 0

    def drain_inbound(
        self, _session_id: UUID
    ) -> list[tuple[str, str | None, str | None]]:
        self.calls += 1
        return []


class _ObservedBinding:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self._ready = threading.Event()

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.set()
        return self._ready.wait(timeout)

    def set(self) -> None:
        self._ready.set()


def _sync_config(client: _InboundClient) -> RunConfig:
    return RunConfig(
        cli_name=_CONFIG.cli_name,
        cli_args=_CONFIG.cli_args,
        sync=True,
        client=cast(Client, client),
        quiesce_seconds=0.0,
    )


def _install(
    monkeypatch: pytest.MonkeyPatch,
    prepared: _Prepared,
    wait_for_live_result: Callable[[], bool],
) -> list[str]:
    captured_argv: list[str] = []

    class _Pump:
        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            captured_argv.extend(argv)

        def run(self, *, on_started: Callable[[], None] | None = None) -> int:
            assert prepared.polls == 0, "drain worker started before PTY callback"
            assert on_started is not None
            on_started()
            assert wait_for_live_result(), "live worker result arrived after PTY exit"
            return 23

    def prepare(
        adapter: AntigravityAdapter,
        args: Sequence[str],
        *,
        cwd: Path,
    ) -> PreparedAntigravity:
        assert isinstance(adapter, _Adapter)
        assert tuple(args) == _CONFIG.cli_args
        assert cwd == Path.cwd()
        return cast(PreparedAntigravity, prepared)

    monkeypatch.setattr(session_mod, "prepare_antigravity", prepare)
    monkeypatch.setattr(session_mod, "PtyPump", _Pump)

    def found_binary(_binary: str) -> str:
        return "/bin/agy"

    monkeypatch.setattr(shutil, "which", found_binary)
    for owner, name in (
        (Path, "glob"),
        (Path, "rglob"),
        (Path, "iterdir"),
        (os, "walk"),
        (os, "scandir"),
        (os, "listdir"),
    ):
        monkeypatch.setattr(owner, name, _forbid_enumeration)
    return captured_argv


def _run(
    sink: _Sink,
    adapter: Adapter | None = None,
    config: RunConfig = _CONFIG,
) -> int:
    return session_mod._spawn_and_drain(config, adapter or _Adapter(), sink, _Stats())


def test_resume_prebinds_identity_and_drains_owned_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(b"new")
    sink = _Sink()
    prepared = _Prepared(reader, expected_id=_ID)
    argv = _install(monkeypatch, prepared, lambda: sink.emitted.wait(1.0))

    assert _run(sink) == 23
    assert sink.calls == [("set", _ID), ("open", None), ("emit", "new")]
    assert argv == ["agy", *prepared.cli_args]
    assert prepared.finish_calls == 1
    assert prepared.closed


def test_codex_resume_dispatches_to_exact_prebound_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _Codex()
    args = ("resume", _ID)
    config = RunConfig(
        cli_name="codex",
        cli_args=args,
        sync=False,
        quiesce_seconds=0.0,
    )
    reader = _Reader(b"new")
    prepared = _Prepared(reader, expected_id=_ID)
    prepared.cli_args = args
    sink = _Sink()
    captured_argv: list[str] = []

    def prepare(
        actual_adapter: Adapter,
        actual_args: Sequence[str],
        *,
        cwd: Path,
    ) -> PreparedResume:
        assert actual_adapter is adapter
        assert tuple(actual_args) == args
        assert cwd == Path.cwd()
        return cast(PreparedResume, prepared)

    class _Pump:
        def __init__(self, argv: list[str], **_kwargs: object) -> None:
            captured_argv.extend(argv)

        def run(self, *, on_started: Callable[[], None] | None = None) -> int:
            assert on_started is not None
            on_started()
            assert sink.emitted.wait(1.0)
            return 23

    monkeypatch.setattr(session_mod, "prepare_explicit_resume", prepare)
    monkeypatch.setattr(session_mod, "_existing_session_files", _forbid_enumeration)
    monkeypatch.setattr(session_mod, "PtyPump", _Pump)

    def found_binary(_binary: str) -> str:
        return "/bin/codex"

    monkeypatch.setattr(shutil, "which", found_binary)

    assert _run(sink, adapter, config) == 23
    assert sink.calls == [("set", _ID), ("open", None), ("emit", "new")]
    assert captured_argv == ["codex", *args]
    assert prepared.closed


def test_fresh_proof_sets_identity_before_first_emit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(b"first")
    sink = _Sink()
    prepared = _Prepared(reader, expected_id=None)
    _install(monkeypatch, prepared, lambda: sink.emitted.wait(1.0))

    assert _run(sink) == 23
    assert sink.calls == [("open", None), ("set", _ID), ("emit", "first")]
    assert prepared.finish_calls == 1
    assert prepared.closed


def test_inbound_poll_waits_for_binding_and_obeys_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def poll(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(session_mod, "_inbound_poll_loop", poll)
    client = cast(Client, _InboundClient())
    sink = _Sink()
    pump = PtyPump(["unused"])

    binding = _ObservedBinding()
    stop = threading.Event()
    worker = threading.Thread(
        target=session_mod._poll_inbound_after_binding,
        args=(client, sink, pump, stop),
        kwargs={"binding_ready": cast(threading.Event, binding)},
    )
    worker.start()
    assert binding.entered.wait(1.0)
    assert calls == 0
    binding.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert calls == 1

    binding = _ObservedBinding()
    stop = threading.Event()
    worker = threading.Thread(
        target=session_mod._poll_inbound_after_binding,
        args=(client, sink, pump, stop),
        kwargs={"binding_ready": cast(threading.Event, binding)},
    )
    worker.start()
    assert binding.entered.wait(1.0)
    stop.set()
    binding.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert calls == 1


def test_final_only_proof_never_starts_inbound_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _Reader(b"final")
    prepared = _Prepared(
        reader,
        expected_id=None,
        poll_returns_reader=False,
    )
    client = _InboundClient()
    sink = _Sink(session_id=uuid4())
    _install(monkeypatch, prepared, lambda: prepared.poll_entered.wait(1.0))

    assert _run(sink, config=_sync_config(client)) == 23
    assert prepared.finish_calls == 1
    assert client.calls == 0
    assert sink.calls == [("open", None), ("set", _ID), ("emit", "final")]


def test_live_slash_waits_for_fresh_proof_and_identity() -> None:
    release_proof = threading.Event()
    reader = _Reader(b"live")
    prepared = _Prepared(
        reader,
        expected_id=None,
        poll_gate=release_proof,
    )
    sink = _Sink()
    stop = threading.Event()
    binding_ready = threading.Event()
    slash_queue = deque(
        [
            (
                SlashCommand(command="model", args="frontier"),
                datetime(2026, 7, 28, tzinfo=UTC),
            )
        ]
    )
    worker = threading.Thread(
        target=session_mod._drain_exact_loop,
        args=(
            _Adapter(),
            cast(PreparedAntigravity, prepared),
            sink,
            _Stats(),
            _CONFIG,
        ),
        kwargs={
            "stop": stop,
            "slash_queue": slash_queue,
            "binding_ready": binding_ready,
        },
    )
    worker.start()
    assert prepared.poll_entered.wait(1.0)
    assert not binding_ready.is_set()
    assert sink.calls == []

    release_proof.set()
    assert sink.user_emitted.wait(1.0)
    stop.set()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert binding_ready.is_set()
    assert not slash_queue
    assert sink.calls == [
        ("set", _ID),
        ("emit", "/model frontier"),
        ("emit", "live"),
    ]
    prepared.close()


def test_binding_failure_terminates_live_child_and_reaches_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("proof failed")
    adapter = _Adapter()
    adapter.cli_binary = sys.executable
    pumps: list[PtyPump] = []
    close_child_pids: list[int] = []
    prepared = _Prepared(
        None,
        expected_id=None,
        error=error,
        on_close=lambda: close_child_pids.append(pumps[0]._pid),
    )
    prepared.cli_args = ("-c", "import time; time.sleep(3)")
    client = _InboundClient()
    sink = _Sink(session_id=uuid4())

    def capture_pump(
        argv: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        on_input: Callable[[bytes], None] | None = None,
    ) -> PtyPump:
        pump = PtyPump(argv, env=env, on_input=on_input)
        pumps.append(pump)
        return pump

    def prepare(
        _adapter: AntigravityAdapter,
        args: Sequence[str],
        *,
        cwd: Path,
    ) -> PreparedAntigravity:
        assert tuple(args) == _CONFIG.cli_args
        assert cwd == Path.cwd()
        return cast(PreparedAntigravity, prepared)

    monkeypatch.setattr(session_mod, "prepare_antigravity", prepare)
    monkeypatch.setattr(session_mod, "PtyPump", capture_pump)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="proof failed") as caught:
        _run(sink, adapter, _sync_config(client))

    assert caught.value is error
    assert time.monotonic() - started < 1.0
    assert len(pumps) == 1
    assert pumps[0]._pid == -1
    assert close_child_pids == [-1]
    assert client.calls == 0
    assert prepared.closed


def test_finalization_drains_four_chunks_before_finish() -> None:
    reader = _Reader(b"tail", reads=4)
    sink = _Sink()
    prepared = _Prepared(reader, expected_id=None, poll_returns_reader=False)
    stop = threading.Event()
    stop.set()
    slash_queue = deque(
        [(SlashCommand(command="exit"), datetime(2026, 7, 28, tzinfo=UTC))]
    )

    session_mod._drain_exact_loop(
        _Adapter(),
        cast(PreparedAntigravity, prepared),
        sink,
        _Stats(),
        _CONFIG,
        stop=stop,
        slash_queue=slash_queue,
    )

    assert prepared.finish_calls == 1
    assert reader.reads == 4
    assert reader.finished
    assert not slash_queue
    assert sink.calls == [("set", _ID), ("emit", "/exit"), ("emit", "tail")]

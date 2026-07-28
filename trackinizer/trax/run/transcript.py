"""Safe, constant-cost reads from one provider-confirmed transcript."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import fcntl
import os
import stat

from trackinizer.trax.run.adapters.base import Adapter


__all__ = [
    "FileBoundary",
    "TranscriptClaim",
    "TranscriptError",
    "TranscriptReader",
    "TranscriptSnapshot",
    "snapshot_transcript",
]


class TranscriptError(RuntimeError):
    """A claimed transcript could not be owned without guessing."""


_BOUNDARY_BYTES = 64
_READ_BYTES = 1 << 20
_MAX_RECORD_BYTES = 16 << 20


@dataclass(frozen=True, slots=True, kw_only=True)
class FileBoundary:
    """The exact transcript inode and EOF observed before provider launch."""

    size: int
    device: int
    inode: int
    tail: bytes


@dataclass(frozen=True, slots=True, kw_only=True)
class TranscriptClaim:
    """A provider-confirmed native identity and its one transcript path."""

    cli_session_id: str
    path: Path
    before: TranscriptSnapshot | None


class TranscriptSnapshot:
    """A pre-launch resume boundary that keeps the original inode open."""

    def __init__(
        self,
        *,
        fd: int,
        path: Path,
        cli_session_id: str,
        boundary: FileBoundary,
    ) -> None:
        self._fd = fd
        self.path = path
        self.cli_session_id = cli_session_id
        self.boundary = boundary

    def consume(self) -> int:
        """Transfer the held descriptor to exactly one transcript reader."""
        if self._fd < 0:
            raise TranscriptError("resume transcript snapshot is closed")
        fd = self._fd
        self._fd = -1
        return fd

    def close(self) -> None:
        """Release the pre-launch descriptor if no reader consumed it."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


class TranscriptReader:
    """Own one transcript descriptor and return each appended line once."""

    def __init__(
        self,
        *,
        fd: int,
        path: Path,
        cli_session_id: str,
        offset: int,
        device: int,
        inode: int,
        guard: bytes,
    ) -> None:
        self._fd = fd
        self._path = path
        self._cli_session_id = cli_session_id
        self._offset = offset
        self._device = device
        self._inode = inode
        self._guard = guard
        self._buffer = bytearray()

    @classmethod
    def open(cls, adapter: Adapter, claim: TranscriptClaim) -> Self:
        """Validate ``claim`` and retain its exact transcript descriptor."""
        if not claim.cli_session_id:
            raise TranscriptError("transcript identity is empty")
        if claim.before is None:
            fd, path, info = _open_validated(adapter, claim.path, claim.cli_session_id)
            offset = 0
            guard = b""
        else:
            snapshot = claim.before
            path = claim.path.absolute()
            if snapshot.cli_session_id != claim.cli_session_id or snapshot.path != path:
                raise TranscriptError("resume snapshot identity does not match claim")
            fd = snapshot.consume()
            try:
                info = _validate_open_fd(adapter, fd, path, claim.cli_session_id)
            except BaseException:
                os.close(fd)
                raise
            before = snapshot.boundary
            if (info.st_dev, info.st_ino) != (before.device, before.inode):
                os.close(fd)
                raise TranscriptError("resume transcript was replaced before binding")
            if info.st_size < before.size:
                os.close(fd)
                raise TranscriptError("resume transcript shrunk before binding")
            _verify_boundary(fd, before)
            offset = before.size
            guard = before.tail
        try:
            return cls(
                fd=fd,
                path=path,
                cli_session_id=claim.cli_session_id,
                offset=offset,
                device=info.st_dev,
                inode=info.st_ino,
                guard=guard,
            )
        except BaseException:
            os.close(fd)
            raise

    @property
    def cli_session_id(self) -> str:
        """The provider's immutable native session identity."""
        return self._cli_session_id

    @property
    def path(self) -> Path:
        """The provider-confirmed transcript path."""
        return self._path

    def read_lines(self) -> tuple[bytes, ...]:
        """Read the current suffix and return complete, unreplayed records."""
        if self._fd < 0:
            raise TranscriptError("transcript reader is closed")
        info = os.fstat(self._fd)
        if (info.st_dev, info.st_ino) != (self._device, self._inode):
            raise TranscriptError("owned transcript descriptor changed identity")
        _verify_path_identity(self._path, info)
        _validate_file(info)
        if info.st_size < self._offset:
            raise TranscriptError("owned transcript shrunk after binding")
        _verify_tail(self._fd, self._offset, self._guard)
        if info.st_size > self._offset:
            read_size = min(info.st_size - self._offset, _READ_BYTES)
            chunk = os.pread(self._fd, read_size, self._offset)
            self._buffer.extend(chunk)
            self._offset += len(chunk)
            self._guard = (self._guard + chunk)[-_BOUNDARY_BYTES:]

        lines: list[bytes] = []
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if len(self._buffer) > _MAX_RECORD_BYTES:
                    raise TranscriptError("transcript record exceeds 16 MiB")
                return tuple(lines)
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            if line.strip():
                lines.append(line)

    def finish(self) -> None:
        """Fail visibly if the provider exits midway through a JSONL record."""
        if self._buffer.strip():
            raise TranscriptError("transcript ends with a partial JSONL record")
        self._buffer.clear()

    def close(self) -> None:
        """Release the owned transcript descriptor; idempotent."""
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()


def snapshot_transcript(
    adapter: Adapter, path: Path, cli_session_id: str
) -> TranscriptSnapshot:
    """Capture a safe pre-launch EOF boundary for an exact resume."""
    fd, _path, info = _open_validated(adapter, path, cli_session_id)
    try:
        boundary = _snapshot_boundary(fd, info)
        return TranscriptSnapshot(
            fd=fd,
            path=path.absolute(),
            cli_session_id=cli_session_id,
            boundary=boundary,
        )
    except BaseException:
        os.close(fd)
        raise


def _open_validated(
    adapter: Adapter, path: Path, cli_session_id: str
) -> tuple[int, Path, os.stat_result]:
    """Open ``path`` beneath an adapter root without following child symlinks."""
    candidate = path.absolute()
    root_and_relative = _provider_root(adapter, candidate)
    if root_and_relative is None:
        raise TranscriptError(f"transcript is outside the {adapter.name} provider root")
    root, relative = root_and_relative
    if not adapter.matches_session_file(candidate):
        raise TranscriptError(f"path is not a {adapter.name} session transcript")

    fd = _open_beneath(root, relative)
    try:
        _lock(fd)
        info = _validate_open_fd(adapter, fd, candidate, cli_session_id)
        return fd, candidate, info
    except BaseException:
        os.close(fd)
        raise


def _validate_open_fd(
    adapter: Adapter, fd: int, path: Path, cli_session_id: str
) -> os.stat_result:
    """Corroborate a held descriptor against its provider path and identity."""
    if _provider_root(adapter, path) is None:
        raise TranscriptError(f"transcript is outside the {adapter.name} provider root")
    if not adapter.matches_session_file(path):
        raise TranscriptError(f"path is not a {adapter.name} session transcript")
    info = os.fstat(fd)
    _verify_path_identity(path, info)
    _validate_file(info)
    observed_id = adapter.session_id_from_transcript(
        path, _read_first_record(fd, info.st_size)
    )
    _verify_path_identity(path, info)
    if observed_id != cli_session_id:
        raise TranscriptError(
            "claimed transcript identity does not match its native record"
        )
    return info


def _provider_root(adapter: Adapter, path: Path) -> tuple[Path, Path] | None:
    """Return the most-specific allowed root and lexical relative path."""
    matches: list[tuple[Path, Path]] = []
    for configured in adapter.session_dirs():
        root = configured.absolute()
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative.parts and ".." not in relative.parts:
            matches.append((root, relative))
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0].parts))


def _open_beneath(root: Path, relative: Path) -> int:
    """Open one regular file below ``root`` with no symlink traversal."""
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_fd = -1
    try:
        # The configured provider root is trusted but may itself be a benign
        # symlink. Resolve it once, then reject symlinks in every child
        # component via dirfd-relative O_NOFOLLOW opens.
        directory_fd = os.open(root.resolve(strict=True), directory_flags)
        for component in relative.parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return os.open(relative.name, file_flags, dir_fd=directory_fd)
    except OSError as error:
        raise TranscriptError("could not safely open transcript") from error
    finally:
        if directory_fd >= 0:
            os.close(directory_fd)


def _validate_file(info: os.stat_result) -> None:
    """Require a private, current-user-owned regular transcript file."""
    if not stat.S_ISREG(info.st_mode):
        raise TranscriptError("transcript is not a regular file")
    if info.st_uid != os.geteuid():
        raise TranscriptError("transcript is not owned by the current user")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise TranscriptError("transcript is group- or world-writable")
    if info.st_nlink != 1:
        raise TranscriptError("transcript must have exactly one filesystem link")


def _verify_path_identity(path: Path, info: os.stat_result) -> None:
    """Ensure the visible path still names the descriptor's inode."""
    try:
        visible = path.stat(follow_symlinks=False)
    except OSError as error:
        raise TranscriptError("owned transcript path disappeared") from error
    if (visible.st_dev, visible.st_ino) != (info.st_dev, info.st_ino):
        raise TranscriptError("owned transcript path was replaced")


def _lock(fd: int) -> None:
    """Exclude a second local Trackinizer owner before provider launch."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise TranscriptError("native transcript is already captured") from error
    except OSError as error:
        raise TranscriptError("could not lock native transcript") from error


def _read_first_record(fd: int, size: int) -> bytes | None:
    """Read a bounded first record from the already-owned descriptor."""
    if size == 0:
        return None
    raw = os.pread(fd, min(size, _READ_BYTES + 1), 0)
    newline = raw.find(b"\n")
    if newline >= 0:
        return raw[:newline]
    if size > _READ_BYTES:
        raise TranscriptError("transcript identity record exceeds 1 MiB")
    return raw


def _read_tail(fd: int, offset: int) -> bytes:
    """Read the small append-only sentinel ending at ``offset``."""
    size = min(offset, _BOUNDARY_BYTES)
    return os.pread(fd, size, offset - size) if size else b""


def _snapshot_boundary(fd: int, info: os.stat_result) -> FileBoundary:
    """Build a resumable EOF boundary or reject a partial pre-launch record."""
    if info.st_size and os.pread(fd, 1, info.st_size - 1) != b"\n":
        raise TranscriptError("resume boundary ends with a partial transcript line")
    return FileBoundary(
        size=info.st_size,
        device=info.st_dev,
        inode=info.st_ino,
        tail=_read_tail(fd, info.st_size),
    )


def _verify_boundary(fd: int, boundary: FileBoundary) -> None:
    """Reject same-inode truncate/regrow before accepting a resume."""
    if _read_tail(fd, boundary.size) != boundary.tail:
        raise TranscriptError("resume transcript prefix changed before binding")


def _verify_tail(fd: int, offset: int, expected: bytes) -> None:
    """Reject an in-place rewrite of the suffix already consumed."""
    if expected and _read_tail(fd, offset) != expected:
        raise TranscriptError("owned transcript prefix changed after binding")

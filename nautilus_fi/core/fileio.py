"""Files that several processes write at once: atomic replacement and advisory locks.

Research agents run in parallel against one ledger and one cache directory. A file is written
whole to a temporary name and renamed into place (readers see the old or the new file, never half
of one); read-modify-write sequences hold an exclusive ``flock`` on a side file (POSIX; a no-op
where unavailable).
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path


@contextlib.contextmanager
def file_lock(path: str | Path):
    """Exclusive advisory lock on ``path`` for the duration of the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as fh:
        try:
            import fcntl
        except ImportError:                          # not POSIX: no advisory locks
            yield
            return
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except OSError:                              # filesystem without flock support
            yield
            return
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def tmp_path(path: str | Path) -> Path:
    """A temporary sibling of ``path`` (same directory, so ``os.replace`` is atomic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


__all__ = ["atomic_write_bytes", "atomic_write_text", "file_lock", "tmp_path"]

"""Run-isolation and provenance stamps shared by the VixStory stages.

``OutputNamespaceLock``: one live run per output namespace, enforced with ``flock``.

``git_sha`` / ``now_iso``: the code revision and UTC timestamp the VixStory run manifests
record for provenance.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import os
import subprocess
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def git_sha(repo: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------- output namespace lock


class LockHeld(RuntimeError):
    """Another live process owns the output namespace lock."""


class OutputNamespaceLock:
    """Exclusive cross-process lock over one output namespace, backed by ``flock``.

    ``acquire()`` raises :class:`LockHeld` while another live holder exists; the namespace
    stays locked until ``release()`` or until the holding process exits (the kernel drops
    the lock, even after a kill). Use as a context manager to release on any exit.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise LockHeld(f"lock already held for {self.path}: {exc}") from exc
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None

    def __enter__(self) -> "OutputNamespaceLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()

"""Stable hashing helpers.

Python's builtin ``hash()`` is randomized per process and must never be used for
anything that is persisted or compared across runs. Everything here is SHA256
based and therefore stable across processes and machines.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization used for hashing structured values."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed derived from arbitrary parts."""
    digest = sha256_text("|".join(str(p) for p in parts))
    return int(digest[:16], 16) % (2**63)

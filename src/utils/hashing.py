"""Stable hashing helpers.

Python's builtin ``hash()`` is randomized per process and must never be used for
anything that is persisted or compared across runs. Everything here is SHA256
based and therefore stable across processes and machines.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

_CHUNK = 1 << 20


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace. Diacritics are preserved."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def normalize_title(title: str) -> str:
    """Group key for splitting: NFKC, casefold, collapse whitespace, strip punctuation noise."""
    title = unicodedata.normalize("NFKC", title)
    title = title.casefold().strip()
    cleaned = []
    for ch in title:
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith("M"):
            cleaned.append(ch)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def stable_seed(*parts: object) -> int:
    """Deterministic 63-bit seed derived from arbitrary parts."""
    digest = sha256_text("|".join(str(p) for p in parts))
    return int(digest[:16], 16) % (2**63)


def content_doc_id(text: str) -> str:
    """doc_id from canonical (NFC, whitespace-collapsed) content."""
    return sha256_text(normalize_text(text))


def hash_list(items: Iterable[str]) -> str:
    return sha256_text("\n".join(items))


def file_manifest(paths: Iterable[str | Path], root: str | Path | None = None) -> dict:
    """{relative_path: {sha256, bytes}} for a set of files."""
    out: dict[str, dict] = {}
    root = Path(root) if root is not None else None
    for p in sorted(Path(x) for x in paths):
        key = str(p.relative_to(root)) if root is not None else str(p)
        out[key] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return out

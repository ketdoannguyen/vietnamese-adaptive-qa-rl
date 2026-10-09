"""Parquet-backed cache of Laya action probabilities, keyed ``(qid, state_hash)``.

The RL stage asks the same frozen checkpoint about the same states many times across
rollouts and updates. This cache answers those repeated questions from disk:

    cache = LayaProbCache(Path("outputs/models/laya_probs_cache/rlcd"))
    probs = cache.get_or_compute(adapter, [(qid, state_hash(text), text), ...])

Staleness is the one thing this module must never get wrong. Rows are stamped with the
adapter's ``model_hash``; a lookup under a different checkpoint invalidates the stored rows
before anything is returned, so a probability computed from an old checkpoint is never
served for a new one. A caller that wants to hear about the mismatch instead of silently
recomputing calls :meth:`require_model_hash` first (``cache_laya`` does). A cache built
without the model bound (no meta sidecar yet) answers :meth:`lookup` with ``None`` until
:meth:`get_or_compute` binds a model.

The rows also carry ``state_text_version``: a change in the state layout invalidates them,
because a ``state_hash`` from the old layout no longer describes the text the model saw.

Column order in the parquet is the canonical ``ACTION_ORDER``:
``p_answer, p_search_bm25, p_search_dense, p_abstain``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from src.decision.schema import ACTION_ORDER
from src.decision.state_text import STATE_TEXT_VERSION
from src.utils.io import atomic_write_json, read_json, write_parquet

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, never executed at runtime
    from src.decision.laya_adapter import LayaAdapter

CACHE_VERSION = "v1"


class ModelHashMismatch(RuntimeError):
    """The cache was built from a different checkpoint than the caller offers."""

#: Derived from ``ACTION_ORDER`` so the column set can never drift from the action set.
PROB_COLUMNS = tuple(f"p_{action.lower()}" for action in ACTION_ORDER)

#: A row is ``(qid, state_hash)`` plus one column per action plus provenance.
COLUMNS = ("qid", "state_hash") + PROB_COLUMNS + ("model_hash", "cache_version",
                                                  "state_text_version")

Key = tuple[str, str]


class LayaProbCache:
    """Rows live in ``<cache_dir>/laya_probs.parquet`` with a ``.meta.json`` sidecar."""

    def __init__(self, cache_dir: str | Path, *, cache_version: str = CACHE_VERSION,
                 state_text_version: str = STATE_TEXT_VERSION) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_version = cache_version
        self.state_text_version = state_text_version
        self.parquet_path = self.cache_dir / "laya_probs.parquet"
        self.meta_path = self.cache_dir / "laya_probs.meta.json"
        self.model_hash: str | None = None
        self.invalidated_rows = 0
        #: Request-level counters, for the ``cache_laya`` summary.
        self.hits = 0
        self.misses = 0
        self._rows: dict[Key, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        """Read stored rows; anything not matching these versions is discarded, not served."""
        if not self.parquet_path.is_file():
            return
        import pyarrow.parquet as pq

        meta: Mapping[str, Any] = read_json(self.meta_path) if self.meta_path.is_file() else {}
        stored = (meta.get("cache_version"), meta.get("state_text_version"))
        if stored != (self.cache_version, self.state_text_version):
            self.invalidated_rows += pq.read_table(str(self.parquet_path)).num_rows
            return
        self.model_hash = meta.get("model_hash")
        for row in pq.read_table(str(self.parquet_path)).to_pylist():
            if (row.get("cache_version") != self.cache_version
                    or row.get("state_text_version") != self.state_text_version):
                self.invalidated_rows += 1
                continue
            self._rows[(row["qid"], row["state_hash"])] = row

    def _bind(self, model_hash: str) -> None:
        """Adopt ``model_hash``; a different checkpoint invalidates every stored row."""
        if self.model_hash == model_hash:
            return
        if self.model_hash is not None and self._rows:
            self.invalidated_rows += len(self._rows)
            # Drop the stale rows from disk too, so a crash right here cannot leave a file
            # that a later process would load and (wrongly) believe.
            self.parquet_path.unlink(missing_ok=True)
            self.meta_path.unlink(missing_ok=True)
        self._rows.clear()
        self.model_hash = model_hash

    def require_model_hash(self, model_hash: str) -> None:
        """Fail fast when the cache belongs to another checkpoint.

        ``get_or_compute`` would silently invalidate and recompute; a caller that wants to
        know it is about to throw a cache away (``cache_laya``) asks this first.
        """
        if self.model_hash is not None and self.model_hash != model_hash:
            raise ModelHashMismatch(
                f"{self.cache_dir} holds probabilities from checkpoint {self.model_hash[:16]}…, "
                f"but the adapter is {model_hash[:16]}…; rebuild the cache for the frozen "
                "checkpoint instead of mixing priors")

    # ------------------------------------------------------------------ reading
    def lookup(self, qid: str, state_hash: str) -> np.ndarray | None:
        """Probabilities for a stored state, or ``None`` on a miss or a bound mismatch."""
        row = self._rows.get((qid, state_hash))
        if row is None or self.model_hash is None or row.get("model_hash") != self.model_hash:
            return None
        return np.array([float(row[column]) for column in PROB_COLUMNS], dtype=np.float64)

    def __len__(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------ writing
    def get_or_compute(self, adapter: "LayaAdapter",
                       requests: Sequence[tuple[str, str, str]]) -> dict[Key, np.ndarray]:
        """Probabilities for ``(qid, state_hash, state_text)`` requests, computing misses.

        Hits come from disk, misses from ``adapter`` in one batched call. Computed rows are
        appended and the parquet is rewritten once, so a crash mid-call leaves the previous
        cache intact. Repeated keys in ``requests`` are computed once.
        """
        self._bind(adapter.model_hash())

        out: dict[Key, np.ndarray] = {}
        missing: list[tuple[str, str, str]] = []
        seen: set[Key] = set()
        for qid, state_hash, state_text in requests:
            key = (qid, state_hash)
            if key in seen:
                continue  # a repeated request is answered once, not computed twice
            seen.add(key)
            hit = self.lookup(qid, state_hash)
            if hit is not None:
                out[key] = hit
                self.hits += 1
            else:
                missing.append((qid, state_hash, state_text))
                self.misses += 1

        if missing:
            probs = adapter.predict_probs([text for _, _, text in missing])
            if probs.shape != (len(missing), len(ACTION_ORDER)):
                raise ValueError(f"adapter returned {probs.shape} for {len(missing)} requests")
            for (qid, state_hash, _), row_probs in zip(missing, probs):
                row = {column: float(value) for column, value in zip(PROB_COLUMNS, row_probs)}
                row.update({"qid": qid, "state_hash": state_hash,
                            "model_hash": self.model_hash, "cache_version": self.cache_version,
                            "state_text_version": self.state_text_version})
                self._rows[(qid, state_hash)] = row
                out[(qid, state_hash)] = np.array(row_probs, dtype=np.float64)
            self._flush()
        return out

    def _flush(self) -> None:
        rows = [self._rows[key] for key in sorted(self._rows)]
        written = write_parquet(self.parquet_path, [{column: row.get(column) for column in COLUMNS}
                                                    for row in rows])
        meta = {"cache_version": self.cache_version,
                "state_text_version": self.state_text_version,
                "model_hash": self.model_hash,
                "columns": list(COLUMNS), "n_rows": written["rows"],
                "parquet_sha256": written["sha256"]}
        atomic_write_json(self.meta_path, meta)

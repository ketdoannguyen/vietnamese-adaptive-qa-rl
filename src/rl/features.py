"""Observation -> fixed-width feature vector (`02_IMPLEMENTATION/05_SEQUENTIAL_REINFORCE_TRAINING.md` §2).

Rules this module obeys:
- Only *visible* information becomes a feature. No gold answer, no answerability flag,
  no teacher Q-value, no reward. ``tests/test_policy_gradient.py`` asserts this.
- Features are **raw** here. The scaler is fitted on **train** only, persisted with the
  policy (``scaler.json``), and applied exactly once — at the network input, via
  ``TorchScaler`` (``src/rl/policy.py``). Never a second time in the numpy path.
- The reader's ``null_margin`` and ``non_null_score`` are treated as uncalibrated scores,
  never as a probability the answer is correct.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from src.qa.metrics import tokenize

N_ACTIONS = 4


def _entropy(probs: Sequence[float]) -> float:
    total = 0.0
    for p in probs:
        if p > 0:
            total -= p * math.log(p)
    return total


@dataclass
class FeatureScaler:
    """Standardiser persisted as JSON (no pickle, so no untrusted-object loading)."""

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    clip: float = 10.0

    @classmethod
    def fit(cls, matrix: np.ndarray, *, clip: float = 10.0) -> "FeatureScaler":
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("scaler requires a non-empty 2-D matrix")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.tolist(), std=std.tolist(), clip=clip)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        out = (matrix - mean) / std
        return np.clip(out, -self.clip, self.clip)

    def to_json(self) -> dict:
        return {"mean": self.mean, "std": self.std, "clip": self.clip}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def feature_names() -> list[str]:
    names = ["laya_p_answer", "laya_p_bm25", "laya_p_dense", "laya_p_abstain",
             "laya_entropy", "laya_missing"]
    names += ["reader_non_null_score", "reader_null_margin", "reader_has_candidate",
              "candidate_len_tokens", "candidate_norm_len",
              "reader_error_flag"]
    names += ["bm25_top_score", "bm25_top_gap", "dense_top_score", "dense_top_gap"]
    names += ["seen_count", "seen_tokens", "optional_used", "remaining_optional",
              "searches_used", "timestep"]
    names += [f"last_action_{name}" for name in ("ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN")]
    names += ["mask_answer", "mask_bm25", "mask_dense", "mask_abstain"]
    names += ["question_len_tokens", "question_norm_len"]
    names += ["preview_enabled"]
    return names


class FeatureExtractor:
    """Turns an observation (plus frozen Laya probabilities) into a raw float32 vector.

    Standardisation is deliberately *not* applied here: every torch consumer builds a
    ``TorchScaler`` from the persisted ``FeatureScaler`` and applies it once at the network
    input, so there is exactly one standardisation point in every path.
    """

    def __init__(self) -> None:
        self.names = feature_names()

    @property
    def dim(self) -> int:
        return len(self.names)

    def vector(self, obs: dict, laya_probs: Sequence[float] | None = None) -> np.ndarray:
        candidate = obs.get("candidate", {})
        retrieval = obs.get("retrieval", {})
        mask = np.asarray(obs.get("action_mask"), dtype=bool)

        if laya_probs is None:
            p = [0.0, 0.0, 0.0, 0.0]
            missing = 1.0
            ent = 0.0
        else:
            p = [float(x) for x in laya_probs]
            missing = 0.0
            ent = _entropy(p)

        answer_text = candidate.get("text")
        cand_tokens = tokenize(answer_text) if answer_text else []

        last_action = int(obs.get("last_action", -1))
        last_onehot = [1.0 if last_action == i else 0.0 for i in range(N_ACTIONS)]

        def finite(value: float, fallback: float = 0.0) -> float:
            value = float(value)
            return value if math.isfinite(value) else fallback

        q = str(obs.get("question", ""))
        q_tokens = tokenize(q)

        vec = [
            p[0], p[1], p[2], p[3], ent, missing,
            finite(candidate.get("non_null_score"), -50.0),
            finite(candidate.get("null_margin"), 0.0),
            1.0 if answer_text else 0.0,
            float(len(cand_tokens)),
            float(len(answer_text)) if answer_text else 0.0,
            1.0 if candidate.get("provenance_ok") is False and answer_text is not None else 0.0,
            finite(retrieval.get("bm25_top_score")),
            finite(retrieval.get("bm25_top_gap")),
            finite(retrieval.get("dense_top_score")),
            finite(retrieval.get("dense_top_gap")),
            float(len(obs.get("seen_doc_ids", ()))),
            float(sum(len(tokenize(t)) for t in obs.get("seen_doc_texts", ()))),
            float(obs.get("optional_used", 0)),
            float(obs.get("remaining_optional_searches", 0)),
            float(obs.get("searches_used", 0)),
            float(obs.get("timestep", 0)),
            *last_onehot,
            *[1.0 if m else 0.0 for m in mask],
            float(len(q_tokens)),
            float(len(q)),
            1.0 if (retrieval.get("bm25_top_score") or retrieval.get("dense_top_score")) else 0.0,
        ]
        out = np.asarray(vec, dtype=np.float32)
        if out.shape[0] != self.dim:
            raise AssertionError(f"feature width {out.shape[0]} != declared {self.dim}")
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def matrix(self, observations: list[dict],
               laya_probs: list[Sequence[float]] | None = None) -> np.ndarray:
        rows = [self.vector(o, None if laya_probs is None else laya_probs[i])
                for i, o in enumerate(observations)]
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)


def mask_logits(logits: np.ndarray, mask: np.ndarray, *, fill: float = -1e9) -> np.ndarray:
    """Masked logits for Categorical sampling. Invalid actions get no probability mass."""
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != logits.shape:
        raise ValueError(f"mask shape {mask.shape} != logits shape {logits.shape}")
    if not mask.any():
        raise ValueError("every action is masked; ABSTAIN must always be legal")
    return np.where(mask, logits, fill)

"""V2 decision schema: the three evidence-retrieval actions and their Laya record shape.

V2 (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §3) drops the V1
reader: an episode is a free BM25 bootstrap passage plus at most two optional SEARCH
actions (``SEARCH_BM25`` / ``SEARCH_DENSE``) and a ``STOP`` that keeps the ordered evidence
set. There is no ANSWER/ABSTAIN, no candidate answer and no reader score in the loop.

This module owns the cross-module contract every other V2 module imports:

* :data:`ACTION_ORDER` / :data:`N_ACTIONS` — the one action ordering shared by the
  environment, the teacher, the policy head and the reports.
* :data:`STATE_TEXT_VERSION` — the V2 state-serializer tag. V1 already used the version
  string ``"v2"``, so the evidence-first tag is spelled out in full.
* :func:`build_questions_field` / :func:`laya_record` — the Laya record shape, with soft
  teacher targets only.
* :func:`assert_public_state_is_clean` — the recursive structural scan that keeps gold
  labels, answers, rewards, coverage and oracle values out of every public state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from src.data.schemas import FORBIDDEN_IN_OBSERVATION

ACTION_ORDER = ("STOP", "SEARCH_BM25", "SEARCH_DENSE")
N_ACTIONS = len(ACTION_ORDER)
STATE_TEXT_VERSION = "evidence-retrieval-v2-len8192"

ACTION_CRITERIA = {
    "STOP": "Dừng và giữ tập bằng chứng đã thu thập",
    "SEARCH_BM25": "Lấy thêm đoạn chưa xem bằng khớp từ khóa",
    "SEARCH_DENSE": "Lấy thêm đoạn chưa xem bằng tìm kiếm ngữ nghĩa",
}
INSTRUCTIONS = ("Chọn hành động tiếp theo để thu đủ đoạn ngữ cảnh gold "
                "và tiết kiệm lượt truy hồi.")

#: Keys that must never appear in a public state at any depth: the V1 label fields plus the
#: V2 retrieval-quality vocabulary. Matching is structural (dict keys only) — evidence text
#: is untrusted dataset content and is never scanned for words.
FORBIDDEN_PUBLIC_KEYS = frozenset(FORBIDDEN_IN_OBSERVATION | {
    "answer", "context_ids", "gold_contexts", "gold_context_ids", "gold_doc_ids",
    "gold_ids", "plausible_texts", "coverage", "gold_coverage", "gold_recall",
    "full_success", "precision", "retrieval_return", "q_values", "oracle_return",
    "matched_gold_ids", "missing_gold_ids",
})


def _scan_forbidden(state: Any, path: str = "") -> list[str]:
    """Recursive structural scan for label-bearing keys in dicts/lists/tuples."""
    hits: list[str] = []
    if isinstance(state, Mapping):
        for key, value in state.items():
            key_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_PUBLIC_KEYS:
                hits.append(key_path)
            hits.extend(_scan_forbidden(value, key_path))
    elif isinstance(state, (list, tuple)):
        for index, value in enumerate(state):
            hits.extend(_scan_forbidden(value, f"{path}[{index}]"))
    return hits


def assert_public_state_is_clean(public_state: Any) -> None:
    """Reject a public state that carries a label-bearing *field*, at any depth.

    The guarantee is structural: gold labels, answers, rewards, coverage and oracle values
    never appear as dict keys in anything a policy reads. Public evidence keys
    (``seen_doc_ids`` / ``seen_doc_texts``) are allowed anywhere.
    """
    hits = _scan_forbidden(public_state)
    if hits:
        raise ValueError(f"public state leaks label-bearing field(s): {hits}")


def build_questions_field(instruction_text: str = INSTRUCTIONS) -> dict[str, Any]:
    """A fresh Laya questions field over exactly the three V2 actions."""
    return {"next_action": {"type": "choice", "instructions": instruction_text,
                            "criteria": dict(ACTION_CRITERIA)}}


def laya_record(state_text: str, probs: Mapping[str, float] | Sequence[float],
                meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One V2 Laya record: state text, the three-action question field, soft target.

    ``probs`` is either a mapping keyed by :data:`ACTION_ORDER` or a sequence of three
    non-negative finite numbers summing to 1. ``meta`` carries private bookkeeping (qid,
    history, ...) and is copied so the caller cannot mutate the record through it.
    """
    if not isinstance(state_text, str) or not state_text.strip():
        raise ValueError("state_text must be a non-empty string")
    names: dict[str, float] = {}
    if isinstance(probs, Mapping):
        if set(probs) != set(ACTION_ORDER):
            raise ValueError(f"probabilities must be keyed by exactly {ACTION_ORDER}; "
                             f"got {sorted(probs)}")
        for name in ACTION_ORDER:
            names[name] = _as_probability(probs[name])
    elif isinstance(probs, Sequence) and not isinstance(probs, (str, bytes)):
        if len(probs) != N_ACTIONS:
            raise ValueError(f"probability sequence must have {N_ACTIONS} entries")
        for name, value in zip(ACTION_ORDER, probs):
            names[name] = _as_probability(value)
    else:
        raise ValueError("probabilities must be a mapping or a sequence of numbers; "
                         f"got {type(probs).__name__}")
    total = math.fsum(names.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {total!r}")
    record: dict[str, Any] = {"state": state_text,
                              "questions": build_questions_field(),
                              "gold": {"next_action": {"probabilities": names}}}
    if meta is not None:
        if not isinstance(meta, Mapping):
            raise ValueError(f"meta must be a mapping; got {type(meta).__name__}")
        record["meta"] = deepcopy(dict(meta))
    return record


def _as_probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"probability must be a real number; got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"probability must be finite and non-negative; got {value!r}")
    return number

"""Canonical visible-state text for the Laya decision model (`02_IMPL/04` §2).

The Laya model only ever sees text. This module renders one environment observation into
that text, deterministically: the same environment state always yields byte-identical text,
so :func:`state_hash` is a stable JSONL ``state`` value and a stable cache key for the RL
stage. The text is a pure function of the observation — nothing else — so a rollout in
:mod:`src.rl.env` reconstructs exactly the same text the fine-tuning data carried.

Public information only. The observation is checked structurally before rendering and the
rendered text is checked for literal label strings, so gold answers, ``is_impossible``,
rewards, plausible answers and teacher/oracle values cannot reach the model input — the
non-negotiable rule of `00_READ_FIRST/02_NONNEGOTIABLE_RULES.md`.

Reader answer spans are not renderable: ``_observation()`` in :mod:`src.rl.env` exposes the
candidate's text, doc id, scores and provenance flag but no character offsets, so the text
carries the score/null-margin summary instead.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.data.schemas import find_forbidden_keys
from src.decision.schema import ACTION_ORDER
from src.utils.hashing import sha256_text

#: Bumped whenever the layout below changes; dataset manifests and cache keys record it.
STATE_TEXT_VERSION = "v2"

#: Document ids are truncated to keep tokens for the evidence text itself.
DOC_ID_CHARS = 16

#: Every observation key the layout reads. A missing one means the caller handed us
#: something that is not an ``AdaptiveQAEnv`` observation; fail with that, not a KeyError.
_REQUIRED_OBS_KEYS = (
    "question", "seen_doc_ids", "seen_doc_texts", "candidate", "retrieval",
    "history", "optional_used", "remaining_optional_searches", "action_mask",
)

#: Literal substrings that must never reach a model input (lowercased compare). Kept to
#: tokens that cannot occur in Vietnamese evidence prose, so the check cannot false-positive
#: on a passage (``src/decision/schema.py`` documents that risk for substring scanning).
_FORBIDDEN_LITERALS = (
    "gold_answer",
    "is_impossible",
    "plausible_answers",
    "expected_action",
    "teacher_action",
    "oracle_q",
    "gold_distribution",
)


def _fmt(value: Any) -> str:
    """Fixed 3-decimal formatting so a float never renders differently between runs."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.3f}"


def assert_state_is_public(obs: Mapping[str, Any]) -> None:
    """Raise if the observation carries a forbidden key anywhere (recursive)."""
    hits = find_forbidden_keys(obs)
    if hits:
        raise ValueError(f"observation carries forbidden keys: {sorted(set(hits))}")


def serialize_state(obs: Mapping[str, Any], *, max_snippet_chars: int = 700) -> str:
    """Render one observation as the canonical state text (deterministic).

    Ordering follows the observation itself: seen documents appear in reveal order and the
    action history in the order taken. Nothing is re-sorted here — the environment's
    observation is already canonical for a state, so sorting would only mask a
    non-deterministic environment.

    This is the text the fine-tuned Laya saw (``make_teacher``) and the text an RL rollout
    feeds it, so it is rendered from exactly those observation keys the environment exposes.
    """
    assert_state_is_public(obs)

    missing = [key for key in _REQUIRED_OBS_KEYS if key not in obs]
    if missing:
        raise KeyError(f"observation is missing {missing}; not an AdaptiveQAEnv observation")

    lines = [f"Câu hỏi: {' '.join(str(obs['question']).split())}"]

    seen_doc_ids = list(obs["seen_doc_ids"])
    seen_texts = list(obs["seen_doc_texts"])
    lines.append(f"Bằng chứng đã xem: {len(seen_doc_ids)} đoạn")
    for index, (doc_id, text) in enumerate(zip(seen_doc_ids, seen_texts), start=1):
        snippet = " ".join(str(text).split())
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "…"
        lines.append(f"[{index}] doc={str(doc_id)[:DOC_ID_CHARS]} {snippet}")

    candidate = obs["candidate"]
    answer = candidate.get("text")
    lines.append(f"Ứng viên trả lời: {answer if answer else '(không có)'}")
    provenance = "có" if candidate.get("provenance_ok") else "không"
    lines.append(f"Nguồn ứng viên: doc={str(candidate.get('doc_id'))[:DOC_ID_CHARS]} | hợp lệ: {provenance}")
    lines.append("Điểm đọc: non_null={} | null={} | chênh lệch null={}".format(
        _fmt(candidate.get("non_null_score")), _fmt(candidate.get("null_score")),
        _fmt(candidate.get("null_margin"))))

    summary = obs["retrieval"]
    lines.append("Truy hồi: BM25 top={} (khoảng cách {}) | Dense top={} (khoảng cách {})".format(
        _fmt(summary.get("bm25_top_score")), _fmt(summary.get("bm25_top_gap")),
        _fmt(summary.get("dense_top_score")), _fmt(summary.get("dense_top_gap"))))

    history = [ACTION_ORDER[int(action)] for action in obs["history"]]
    lines.append(f"Hành động đã thực hiện: {' > '.join(history) if history else '(chưa có)'}")
    lines.append("Số lượt tìm đã dùng: {} | còn lại: {}".format(
        int(obs["optional_used"]), int(obs["remaining_optional_searches"])))

    valid = [ACTION_ORDER[index] for index, ok in enumerate(obs["action_mask"]) if bool(ok)]
    lines.append(f"Hành động hợp lệ: {', '.join(valid)}")

    text = "\n".join(lines)
    lowered = text.lower()
    leaked = [needle for needle in _FORBIDDEN_LITERALS if needle in lowered]
    if leaked:
        raise ValueError(f"rendered state text leaks forbidden content: {leaked}")
    return text


def state_hash(text: str) -> str:
    """Cache key for a rendered state: a pure function of the text, nothing else."""
    return sha256_text(text)

"""Canonical serialization shared by the Laya state string and the teacher targets.

**The single rule of this module:** the public serialization of a state contains only
what the agent has already paid to see. Gold answers, answerability and oracle Q-values
live on the private side and are asserted absent by tests.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from src.data.schemas import find_forbidden_keys

STATE_SERIALIZER_VERSION = "v1"

ACTION_CRITERIA = {
    "ANSWER": "Kết thúc và trả lời bằng chứng cứ có sẵn",
    "SEARCH_BM25": "Lấy thêm đoạn chưa đọc bằng khớp từ khóa",
    "SEARCH_DENSE": "Lấy thêm đoạn chưa đọc bằng tìm kiếm ngữ nghĩa",
    "ABSTAIN": "Dừng vì không có cơ sở trả lời đáng tin cậy",
}
ACTION_ORDER = ("ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN")
INSTRUCTIONS = ("Chọn hành động tiếp theo để tối đa hóa chất lượng câu trả lời "
                "và tiết kiệm truy hồi.")

_MAX_SNIPPET_CHARS = 700


def render_state(obs: dict, *, max_snippet_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Human-readable Vietnamese state string handed to the decision model.

    Contains: the question, the evidence the agent has already opened, the current
    candidate answer with reader scores, retrieval summaries, and the remaining budget.
    """
    lines = [f"Câu hỏi: {obs.get('question', '').strip()}"]
    doc_ids: Sequence[str] = obs.get("seen_doc_ids", ())
    texts: Sequence[str] = obs.get("seen_doc_texts", ())
    lines.append("Bằng chứng đã xem:")
    if doc_ids:
        for i, (doc_id, text) in enumerate(zip(doc_ids, texts), start=1):
            snippet = " ".join(str(text).split())
            if len(snippet) > max_snippet_chars:
                snippet = snippet[:max_snippet_chars] + "…"
            lines.append(f"[{i}] doc={doc_id[:12]} {snippet}")
    else:
        lines.append("(chưa có)")

    candidate = obs.get("candidate", {})
    answer = candidate.get("text")
    lines.append(f"Ứng viên trả lời: {answer if answer else '(không có)'}")
    lines.append(f"Điểm đọc: {_fmt(candidate.get('non_null_score'))} "
                 f"| chênh lệch null: {_fmt(candidate.get('null_margin'))}")

    retrieval = obs.get("retrieval", {})
    lines.append(f"BM25 score: {_fmt(retrieval.get('bm25_top_score'))} "
                 f"| Dense score: {_fmt(retrieval.get('dense_top_score'))}")

    mask = obs.get("action_mask")
    lines.append(f"Số lượt tìm còn: {obs.get('remaining_optional_searches', 0)}")
    if mask is not None:
        allowed = ", ".join(ACTION_ORDER[i] for i, ok in enumerate(mask) if ok)
        lines.append(f"Hành động hợp lệ: {allowed}")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if number != number or number in (float("inf"), float("-inf")):  # NaN / inf
        return "n/a"
    return f"{number:.3f}"


def build_questions_field(instruction_text: str = INSTRUCTIONS) -> dict:
    return {"next_action": {"type": "choice", "instructions": instruction_text,
                            "criteria": dict(ACTION_CRITERIA)}}


def laya_record(*, state: str, expected: str | None = None,
                gold_distribution: dict[str, float] | None = None,
                instruction_text: str = INSTRUCTIONS) -> dict:
    """One Laya training/inference record.

    Exactly one of ``expected`` (hard label) or ``gold_distribution`` (soft label) is
    normally supplied; the teacher manifest records which mode produced the file.
    """
    if expected is not None and expected not in ACTION_CRITERIA:
        raise ValueError(f"unknown action label {expected!r}")
    record: dict[str, Any] = {"state": state,
                              "questions": build_questions_field(instruction_text)}
    if expected is not None:
        record["expected"] = {"next_action": expected}
    if gold_distribution is not None:
        bad = set(gold_distribution) - set(ACTION_CRITERIA)
        if bad:
            raise ValueError(f"unknown action(s) in distribution: {sorted(bad)}")
        total = sum(gold_distribution.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"distribution must sum to 1, got {total!r}")
        if any(v < 0 for v in gold_distribution.values()):
            raise ValueError("distribution must be non-negative")
        record["gold"] = {"next_action": {"probabilities": dict(gold_distribution)}}
    return record


def assert_public_state_is_clean(obs: dict) -> None:
    """Reject an observation that carries a label-bearing *field*.

    Scanning the rendered string for words would produce false positives: evidence text
    is untrusted dataset content and may legitimately contain any word, including
    "reward". The guarantee we can actually enforce is structural — no gold field is
    present in the dict the string is rendered from.
    """
    hits = find_forbidden_keys(obs)
    if hits:
        raise ValueError(f"observation leaks label-bearing field(s): {hits}")


def dataset_manifest_entry(records: Sequence[dict]) -> dict:
    """Summary statistics required by `01_SPECIFICATIONS/07_...md` §7."""
    from collections import Counter

    action_counts: Counter[str] = Counter()
    option_counts: Counter[int] = Counter()
    for record in records:
        options = record["questions"]["next_action"]["criteria"]
        option_counts[len(options)] += 1
        hard = record.get("expected", {}).get("next_action")
        if hard:
            action_counts[hard] += 1
    return {"n_records": len(records),
            "options_per_record": dict(option_counts),
            "hard_label_histogram": dict(action_counts)}

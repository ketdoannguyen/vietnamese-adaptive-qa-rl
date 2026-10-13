"""Canonical four-action labels and Laya records for VixSTORY V1."""

from __future__ import annotations

from typing import Any

ACTION_CRITERIA = {
    "ANSWER": "Kết thúc và trả lời bằng chứng cứ có sẵn",
    "SEARCH_BM25": "Lấy thêm đoạn chưa đọc bằng khớp từ khóa",
    "SEARCH_DENSE": "Lấy thêm đoạn chưa đọc bằng tìm kiếm ngữ nghĩa",
    "ABSTAIN": "Dừng vì không có cơ sở trả lời đáng tin cậy",
}
ACTION_ORDER = ("ANSWER", "SEARCH_BM25", "SEARCH_DENSE", "ABSTAIN")
INSTRUCTIONS = ("Chọn hành động tiếp theo để tối đa hóa chất lượng câu trả lời "
                "và tiết kiệm truy hồi.")


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

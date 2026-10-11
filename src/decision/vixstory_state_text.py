"""Canonical public VixSTORY observation text, shared by teacher and Laya inference."""

from __future__ import annotations

from typing import Any, Mapping

from src.data.schemas import find_forbidden_keys
from src.decision.schema import ACTION_ORDER, build_questions_field
from src.decision.state_text import state_hash

STATE_TEXT_VERSION = "v2"
MAX_LEN = 768
HEAD_MAX_LEN = 192


def serialize_vixstory_state(obs: Mapping[str, Any], *, tokenizer: Any,
                              max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN) -> str:
    """Fit public decision fields and token-allocated evidence into Laya's real state room."""
    from laya.common import build_sequence, state_room
    from laya.train import to_internal

    forbidden = find_forbidden_keys(obs)
    if forbidden:
        raise ValueError(f"observation carries forbidden keys: {sorted(set(forbidden))}")
    required = ("question", "seen_doc_ids", "seen_doc_texts", "candidate", "history",
                "optional_used", "remaining_optional_searches", "action_mask")
    missing = [name for name in required if name not in obs]
    if missing:
        raise KeyError(f"VixSTORY observation is missing {missing}")
    ids, passages = obs["seen_doc_ids"], obs["seen_doc_texts"]
    if len(ids) != len(passages) or len(obs["action_mask"]) != len(ACTION_ORDER):
        raise ValueError("VixSTORY observation has inconsistent evidence or action mask")
    candidate = obs["candidate"]
    answer = candidate["text"]
    history = [ACTION_ORDER[int(action)] for action in obs["history"]]
    valid = [name for name, allowed in zip(ACTION_ORDER, obs["action_mask"]) if bool(allowed)]
    lines = [f"Câu hỏi: {' '.join(str(obs['question']).split())}",
             f"Ứng viên trả lời: {answer if answer else '(không có)'}",
             "Trích dẫn: {} | ID hợp lệ: {}".format(
                 ", ".join(candidate["cited_ids"]) or "(không có)",
                 "có" if candidate["citation_valid"] else "không"),
             f"Hành động đã thực hiện: {' > '.join(history) if history else '(chưa có)'}",
             "Số lượt tìm đã dùng: {} | còn lại: {}".format(
                 int(obs["optional_used"]), int(obs["remaining_optional_searches"])),
             f"Hành động hợp lệ: {', '.join(valid)}",
             f"Bằng chứng đã xem: {len(ids)} đoạn"]
    # Retain every visible ID even if its passage has no room; never lose the candidate/budget.
    labels = [f"[{index}] doc={doc_id}" for index, doc_id in enumerate(ids, start=1)]
    prefix = "\n".join(lines + labels)
    question = to_internal("next_action", build_questions_field()["next_action"])
    room = state_room(tokenizer, question, max_len, head_max_len)

    def length(text: str) -> int:
        return len(tokenizer(text.replace(tokenizer.mask_token, " "),
                             add_special_tokens=False)["input_ids"])

    if length(prefix) > room:
        raise ValueError(f"VixSTORY essential state fields exceed Laya's {room}-token room")
    result = "\n".join(lines)
    for index, (label, passage) in enumerate(zip(labels, passages)):
        words = str(passage).split()
        base = result + "\n" + label
        suffix = "".join("\n" + pending for pending in labels[index + 1:])
        # Reserve a share for each remaining passage; otherwise the first long passage
        # consumes the entire state budget and later evidence disappears again.
        share = (room - length(base + suffix)) // (len(labels) - index)
        limit = length(base + suffix) + max(0, share)
        low, high = 0, len(words)
        while low < high:
            mid = (low + high + 1) // 2
            snippet = " ".join(words[:mid])
            if length(base + " " + snippet + suffix) <= limit:
                low = mid
            else:
                high = mid - 1
        result = base + (" " + " ".join(words[:low]) if low else "")
    _, _, stats = build_sequence(tokenizer, result, question, max_len, head_max_len,
                                 return_truncation_stats=True)
    if stats["truncated"]:
        raise ValueError("VixSTORY Laya state would be silently truncated")
    return result


__all__ = ["STATE_TEXT_VERSION", "serialize_vixstory_state", "state_hash"]

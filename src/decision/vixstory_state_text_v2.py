"""Retrieval-only public Laya text and measurable passage-prefix retention."""

from __future__ import annotations

from typing import Any, Mapping

from src.decision.schema_v2 import ACTION_ORDER, assert_public_state_is_clean, build_questions_field
from src.decision.schema_v2 import STATE_TEXT_VERSION as STATE_TEXT_VERSION

MAX_LEN = 8192
HEAD_MAX_LEN = 192


def render_with_retention(obs: Mapping[str, Any], *, tokenizer: Any,
                          max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN) -> tuple[str, dict]:
    """Retain all passages when possible; allocate word prefixes only if over budget.

    Audit actual snippets against full passages: the final Laya clamp alone cannot
    detect evidence already removed by the serializer. No private labels are inputs.
    """
    from laya.common import build_sequence, state_room
    from laya.train import to_internal

    assert_public_state_is_clean(dict(obs))
    required = ("question", "seen_doc_ids", "seen_doc_texts", "history", "optional_used",
                "remaining_optional_searches", "action_mask")
    missing = [field for field in required if field not in obs]
    if missing:
        raise KeyError(f"V2 observation is missing {missing}")
    ids, passages, mask = obs["seen_doc_ids"], obs["seen_doc_texts"], obs["action_mask"]
    if len(ids) != len(passages) or not 1 <= len(ids) <= 5 or len(set(ids)) != len(ids):
        raise ValueError("inconsistent V2 evidence IDs/texts")
    if len(mask) != 3 or not mask[0]:
        raise ValueError("V2 requires a three-action mask with STOP legal")
    if any(not isinstance(p, str) or not p.strip() for p in passages):
        raise ValueError("empty passage content")
    if any(action not in ACTION_ORDER for action in obs["history"]):
        raise ValueError("unknown V2 history action")
    valid = [action for action, legal in zip(ACTION_ORDER, mask) if legal]
    lines = [f"Câu hỏi: {' '.join(obs['question'].split())}",
             f"Hành động đã thực hiện: {' > '.join(obs['history']) or '(chưa có)'}",
             f"Số lượt tìm đã dùng: {obs['optional_used']} | còn lại: {obs['remaining_optional_searches']}",
             f"Hành động hợp lệ: {', '.join(valid)}",
             f"Bằng chứng đã xem: {len(ids)} đoạn"]
    labels = [f"[{index}] doc={doc_id}" for index, doc_id in enumerate(ids, start=1)]
    question = to_internal("next_action", build_questions_field()["next_action"])
    room = state_room(tokenizer, question, max_len, head_max_len)

    def length(text: str) -> int:
        return len(tokenizer(text.replace(tokenizer.mask_token, " "),
                             add_special_tokens=False)["input_ids"])

    if length("\n".join(lines + labels)) > room:
        raise ValueError(f"V2 essential fields exceed Laya's {room}-token room")
    words_by_passage = [passage.split() for passage in passages]
    full_texts = [" ".join(words) for words in words_by_passage]
    full_state = "\n".join(lines + [f"{label} {text}" for label, text in zip(labels, full_texts)])
    fits_full_state = length(full_state) <= room
    result = "\n".join(lines)
    retained = []
    for index, (doc_id, label, words, full_text) in enumerate(
        zip(ids, labels, words_by_passage, full_texts)
    ):
        base = result + "\n" + label
        if fits_full_state:
            snippet = full_text
            cut = False
        else:
            suffix = "".join("\n" + pending for pending in labels[index + 1:])
            base_length = length(base + suffix)
            limit = base_length + max(0, (room - base_length) // (len(labels) - index))
            low, high = 0, len(words)
            while low < high:
                mid = (low + high + 1) // 2
                if length(base + " " + " ".join(words[:mid]) + suffix) <= limit:
                    low = mid
                else:
                    high = mid - 1
            snippet = " ".join(words[:low])
            cut = low < len(words)
        if not snippet or length(snippet) == 0:
            raise ValueError(f"V2 passage {doc_id} has no retained content")
        result = base + " " + snippet
        full_tokens, retained_tokens = length(full_text), length(snippet)
        retained.append({"doc_id": doc_id, "full_tokens": full_tokens,
                         "retained_tokens": retained_tokens,
                         "retention": retained_tokens / full_tokens,
                         "cut": cut, "snippet": snippet})
    _, markers, head_stats, stats = build_sequence(
        tokenizer, result, question, max_len, head_max_len,
        return_stats=True, return_truncation_stats=True)
    if stats["truncated"]:
        raise ValueError("V2 Laya state would be silently truncated")
    if len(markers) != 3 or head_stats["options_distinct"] != 3:
        raise ValueError("V2 essential three-action head would lose an option")
    return result, {**stats, "state_room": room, "passages": retained,
                    "head_options_distinct": head_stats["options_distinct"],
                    "essential_fields_present": all(line in result for line in lines + labels)}


def serialize_vixstory_state(obs: Mapping[str, Any], *, tokenizer: Any,
                              max_len: int = MAX_LEN, head_max_len: int = HEAD_MAX_LEN) -> str:
    return render_with_retention(obs, tokenizer=tokenizer, max_len=max_len,
                                 head_max_len=head_max_len)[0]

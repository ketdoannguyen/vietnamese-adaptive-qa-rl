"""State text is public-only, deterministic, and hashes stably (`02_IMPL/04` §2).

The observation is built explicitly here in the shape ``serialize_state`` requires
(``_REQUIRED_OBS_KEYS`` of ``src/decision/state_text.py``): the shared text contract is
pinned without the removed ViQuAD environment, and any drift in those keys or in the
rendered layout fails here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decision import state_text
from src.decision.schema import ACTION_ORDER
from src.utils.hashing import sha256_text

_BM25 = ACTION_ORDER.index("SEARCH_BM25")
_DENSE = ACTION_ORDER.index("SEARCH_DENSE")
_ANSWER = ACTION_ORDER.index("ANSWER")


def obs(**overrides) -> dict:
    """One observation shaped like the env contract the state text renders."""
    base = {
        "question": "Thủ đô của Việt Nam là gì?",
        "seen_doc_ids": ("doc_hanoi",),
        "seen_doc_texts": ("Hà Nội là thủ đô của Việt Nam.",),
        "candidate": {"text": "Hà Nội", "doc_id": "doc_hanoi", "provenance_ok": True,
                      "non_null_score": 4.5, "null_score": 1.25, "null_margin": -3.25},
        "retrieval": {"bm25_top_score": 7.5, "bm25_top_gap": 1.5,
                      "dense_top_score": 0.83, "dense_top_gap": 0.04},
        "history": (),
        "optional_used": 0,
        "remaining_optional_searches": 2,
        "action_mask": (True, True, True, True),
    }
    base.update(overrides)
    return base


def test_text_is_deterministic_and_carries_the_public_fields():
    root = obs()
    text = state_text.serialize_state(root)
    assert text == state_text.serialize_state(root)
    assert text.startswith("Câu hỏi:")
    assert "Bằng chứng đã xem: 1 đoạn" in text
    assert "Ứng viên trả lời:" in text
    assert "Điểm đọc:" in text and "Truy hồi:" in text
    assert "Hành động đã thực hiện: (chưa có)" in text
    assert "Số lượt tìm đã dùng: 0" in text
    assert f"Hành động hợp lệ: {', '.join(ACTION_ORDER)}" in text  # everything legal at t=0


def test_distinct_histories_render_distinct_text():
    seen = {}
    for history in ((), (_BM25,), (_DENSE,)):
        text = state_text.serialize_state(obs(history=history))
        assert text not in seen.values(), f"history {history} collided with {seen}"
        seen[history] = text
    assert "Hành động đã thực hiện: SEARCH_BM25" in seen[(_BM25,)]
    assert "Hành động đã thực hiện: SEARCH_DENSE" in seen[(_DENSE,)]


def test_state_hash_is_a_pure_function_of_the_text():
    text = state_text.serialize_state(obs())
    assert state_text.state_hash(text) == sha256_text(text)
    assert state_text.state_hash(text) == state_text.state_hash(text)
    assert state_text.state_hash(text) != state_text.state_hash(text + "x")


def test_forbidden_keys_are_rejected_before_rendering():
    for needle in ("gold_answer", "is_impossible", "expected_action", "plausible_answers"):
        bad = {**obs(), needle: "anything"}
        with pytest.raises(ValueError, match="forbidden"):
            state_text.serialize_state(bad)
        with pytest.raises(ValueError, match="forbidden"):
            state_text.assert_state_is_public(bad)


def test_nested_forbidden_keys_are_found():
    bad = obs()
    bad["candidate"]["plausible_answers"] = ["x"]
    with pytest.raises(ValueError, match="forbidden"):
        state_text.serialize_state(bad)


def test_a_missing_observation_key_is_named():
    bad = {key: value for key, value in obs().items() if key != "action_mask"}
    with pytest.raises(KeyError, match="action_mask"):
        state_text.serialize_state(bad)


def test_a_forbidden_literal_in_public_text_is_a_leak():
    bad = obs(question="gold_answer của câu hỏi này là gì?")
    with pytest.raises(ValueError, match="leaks"):
        state_text.serialize_state(bad)


def test_long_evidence_is_snippet_capped():
    obs_long = obs(seen_doc_ids=("doc:1",), seen_doc_texts=(" ".join(["từ"] * 500),))
    text = state_text.serialize_state(obs_long, max_snippet_chars=40)
    line = [ln for ln in text.splitlines() if ln.startswith("[1]")][0]
    assert line.endswith("…")
    assert len(line.split(" ", 2)[2]) == 41  # 40 chars + the ellipsis


def test_floats_render_with_fixed_precision_and_non_finite_as_na():
    bad = obs()
    bad["candidate"]["non_null_score"] = float("nan")
    bad["retrieval"]["bm25_top_gap"] = float("inf")
    text = state_text.serialize_state(bad)
    assert "non_null=n/a" in text and "khoảng cách n/a" in text
    bad["candidate"]["non_null_score"] = 1.0 / 3.0
    assert "non_null=0.333" in state_text.serialize_state(bad)


def test_masked_actions_are_absent_from_the_valid_list():
    spent = obs(history=(_BM25, _DENSE), optional_used=2,
                remaining_optional_searches=0,
                action_mask=(True, False, False, True))
    mask = np.asarray(spent["action_mask"], dtype=bool)
    assert not mask[1:3].any(), "the two optional searches are spent; both must be masked"
    text = state_text.serialize_state(spent)
    valid_line = text.split("Hành động hợp lệ: ", 1)[1].splitlines()[0]
    expected = ", ".join(name for index, name in enumerate(ACTION_ORDER) if mask[index])
    assert valid_line == expected
    assert "SEARCH_BM25" not in valid_line and "SEARCH_DENSE" not in valid_line


def test_answer_action_renders_as_history():
    text = state_text.serialize_state(obs(history=(_ANSWER,)))
    assert "Hành động đã thực hiện: ANSWER" in text

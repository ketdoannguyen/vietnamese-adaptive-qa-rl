"""Contract tests for the V2 decision schema (:mod:`src.decision.schema_v2`).

V2 (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md``) is evidence retrieval
only: three actions (STOP, SEARCH_BM25, SEARCH_DENSE), no ANSWER/ABSTAIN, no reader. These
tests pin the cross-module contract every V2 module imports: the action order, the state
text version tag, the Laya record shape and the recursive public-state privacy scanner.
"""

from __future__ import annotations

import json

import pytest

from src.data.schemas import FORBIDDEN_IN_OBSERVATION
from src.decision import schema_v2
from src.decision.schema_v2 import (
    ACTION_ORDER,
    N_ACTIONS,
    STATE_TEXT_VERSION,
    assert_public_state_is_clean,
    build_questions_field,
    laya_record,
)

PROBS = {"STOP": 0.25, "SEARCH_BM25": 0.5, "SEARCH_DENSE": 0.25}


def clean_obs() -> dict:
    """A canonical public V2 observation: public seen ids/texts are allowed everywhere."""
    return {
        "qid": "q1",
        "question": "Trường A thành lập năm nào?",
        "seen_doc_ids": ("d0", "d1"),
        "seen_doc_texts": ("Đoạn văn a.", "Đoạn văn b."),
        "history": ["SEARCH_BM25"],
        "optional_used": 1,
        "remaining_optional_searches": 1,
        "action_mask": [True, True, False],
    }


# --- constants and cross-module wiring -------------------------------------------------

def test_action_contract_is_exactly_three_actions():
    assert ACTION_ORDER == ("STOP", "SEARCH_BM25", "SEARCH_DENSE")
    assert N_ACTIONS == 3 == len(ACTION_ORDER)
    assert "ANSWER" not in ACTION_ORDER and "ABSTAIN" not in ACTION_ORDER


def test_state_text_version_is_the_evidence_retrieval_tag():
    from src.decision.vixstory_state_text_v2 import STATE_TEXT_VERSION as SERIALIZER_VERSION

    assert STATE_TEXT_VERSION == SERIALIZER_VERSION == "evidence-retrieval-v2-len8192"


def test_env_and_teacher_derive_from_the_schema_contract():
    from src.pipeline import vixstory_teacher_v2
    from src.rl import vixstory_env_v2

    assert vixstory_env_v2.ACTION_ORDER == ACTION_ORDER
    assert vixstory_env_v2.N_ACTIONS == N_ACTIONS
    assert (vixstory_env_v2.STOP, vixstory_env_v2.SEARCH_BM25,
            vixstory_env_v2.SEARCH_DENSE) == (0, 1, 2)
    assert vixstory_teacher_v2.SEARCH_COST == vixstory_env_v2.SEARCH_COST


# --- build_questions_field -------------------------------------------------------------

def test_questions_field_offers_exactly_the_three_v2_actions():
    field = build_questions_field()["next_action"]
    assert field["type"] == "choice"
    assert list(field["criteria"]) == list(ACTION_ORDER)
    assert all(isinstance(text, str) and text for text in field["criteria"].values())
    assert field["instructions"]


def test_questions_field_is_rebuilt_per_call_and_honours_custom_instructions():
    first, second = build_questions_field(), build_questions_field("Khác.")
    assert second["next_action"]["instructions"] == "Khác."
    first["next_action"]["criteria"]["STOP"] = "bị sửa"
    assert build_questions_field()["next_action"]["criteria"]["STOP"] != "bị sửa"


# --- laya_record -----------------------------------------------------------------------

def test_laya_record_carries_state_questions_and_soft_target():
    record = laya_record("CÂU HỎI VÀ BẰNG CHỨNG", PROBS)
    assert record["state"] == "CÂU HỎI VÀ BẰNG CHỨNG"
    assert list(record["questions"]["next_action"]["criteria"]) == list(ACTION_ORDER)
    assert record["gold"]["next_action"]["probabilities"] == PROBS
    assert "meta" not in record           # meta is optional
    assert "expected" not in record       # V2 teacher targets are soft only


def test_laya_record_accepts_the_teacher_probability_list():
    record = laya_record("state", [0.1, 0.2, 0.7])
    assert record["gold"]["next_action"]["probabilities"] == {
        "STOP": 0.1, "SEARCH_BM25": 0.2, "SEARCH_DENSE": 0.7}


def test_laya_record_attaches_private_meta_as_a_copy():
    meta = {"qid": "q1", "q_values": {"STOP": 0.0}, "history": []}
    record = laya_record("state", PROBS, meta=meta)
    assert record["meta"] == meta
    meta["qid"] = "đã đổi"
    meta["q_values"]["STOP"] = 1.0
    meta["history"].append("SEARCH_BM25")
    assert record["meta"] == {"qid": "q1", "q_values": {"STOP": 0.0}, "history": []}


def test_laya_record_is_json_serializable():
    json.dumps(laya_record("state", PROBS, meta={"qid": "q1"}), ensure_ascii=False)


@pytest.mark.parametrize("probs", [
    {"STOP": 0.5, "SEARCH_BM25": 0.5},                                      # missing action
    {"STOP": 0.5, "SEARCH_BM25": 0.5, "SEARCH_DENSE": 0.0, "ABSTAIN": 0.0},  # unknown action
    {"STOP": 0.5, "SEARCH_BM25": 0.3, "SEARCH_DENSE": 0.3},                  # sum != 1
    {"STOP": 1.2, "SEARCH_BM25": -0.2, "SEARCH_DENSE": 0.0},                 # negative mass
    {"STOP": 1.0, "SEARCH_BM25": 0.0, "SEARCH_DENSE": float("nan")},         # NaN
    {"STOP": 0.5, "SEARCH_BM25": 0.5, "SEARCH_DENSE": float("inf")},         # inf
    {"STOP": "x", "SEARCH_BM25": 0.0, "SEARCH_DENSE": 0.0},                  # non-numeric
    {"STOP": True, "SEARCH_BM25": 0.0, "SEARCH_DENSE": 0.0},                 # bool is not a probability
    [0.5, 0.5],                                                              # wrong length
    [0.5, 0.5, 0.5],                                                         # sum != 1
    "0.5 0.5 0",                                                             # a string is not a distribution
])
def test_laya_record_rejects_invalid_distributions(probs):
    with pytest.raises(ValueError):
        laya_record("state", probs)


@pytest.mark.parametrize("state_text", ["", "   ", None, 7])
def test_laya_record_rejects_an_empty_or_non_text_state(state_text):
    with pytest.raises(ValueError):
        laya_record(state_text, PROBS)


def test_laya_record_rejects_non_mapping_meta():
    with pytest.raises(ValueError):
        laya_record("state", PROBS, meta=["q1"])


# --- public-state privacy scanner ------------------------------------------------------

def test_clean_public_observation_passes():
    assert assert_public_state_is_clean(clean_obs()) is None


def test_scanner_covers_the_v1_forbidden_names_and_the_v2_contract_keywords():
    assert FORBIDDEN_IN_OBSERVATION <= schema_v2.FORBIDDEN_PUBLIC_KEYS
    required = {"gold_context_ids", "context_ids", "answers", "reward", "coverage",
                "oracle_q", "q_values", "expected_action", "full_success",
                "retrieval_return", "matched_gold_ids", "missing_gold_ids", "gold_answer",
                "precision", "gold_recall"}
    assert required <= schema_v2.FORBIDDEN_PUBLIC_KEYS
    assert {"seen_doc_ids", "seen_doc_texts"} <= set(clean_obs())


@pytest.mark.parametrize("key,value", [
    ("gold_context_ids", ["d1"]), ("context_ids", ["d1"]), ("gold_doc_ids", ["d1"]),
    ("answers", ["1995"]), ("answer", "1995"), ("gold_answer", "1995"),
    ("reward", 1.5), ("coverage", 0.5), ("gold_recall", 0.5), ("full_success", 1.0),
    ("precision", 0.5), ("retrieval_return", 1.4), ("matched_gold_ids", ["d1"]),
    ("missing_gold_ids", []), ("oracle_q", {"STOP": 1.0}), ("q_values", [0.0, 1.0, 1.0]),
    ("oracle_return", 1.45), ("expected_action", "STOP"), ("teacher_action", "STOP"),
    ("gold_distribution", {"STOP": 1.0}), ("is_impossible", False),
])
def test_forbidden_field_in_a_top_level_observation_is_rejected(key, value):
    obs = clean_obs()
    obs[key] = value
    with pytest.raises(ValueError, match=key):
        assert_public_state_is_clean(obs)


def test_forbidden_field_nested_anywhere_is_rejected():
    obs = clean_obs()
    obs["retrieval"] = {"tools": [{"name": "bm25", "coverage": 0.5}]}
    with pytest.raises(ValueError, match="coverage"):
        assert_public_state_is_clean(obs)


def test_public_seen_ids_and_texts_are_allowed_at_any_depth():
    obs = clean_obs()
    obs["nested"] = [{"seen_doc_ids": ("d0",), "seen_doc_texts": ("Đoạn.",)}]
    assert assert_public_state_is_clean(obs) is None

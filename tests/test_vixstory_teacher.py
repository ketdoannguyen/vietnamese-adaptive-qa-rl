"""Tests for the CPU VixSTORY teacher-row builder (:mod:`src.pipeline.vixstory_teacher`).

These exercise the module on a tiny in-memory world: no dataset, no GPU, no reader model.
"""

from __future__ import annotations

import pytest

from src.data.schemas import find_forbidden_keys
from src.decision.schema import ACTION_ORDER
class TokenizerStub:
    mask_token = "<mask>"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}
from src.pipeline.vixstory_teacher import (
    REWARD_TAU,
    answer_reward,
    SEARCH_COST,
    enumerate_states,
    expected_action,
    q_values,
    teacher_rows,
)
from src.retrieval.states import reachable_states

QID = "qa_toy"
# The gold reference is private: the reader's *candidate* ("alpha") echoes it by definition,
# but the reference text itself must never be rendered.
REFERENCE = "alpha beta"
CORRECT_CANDIDATE = "alpha"  # F1 vs the reference is 0.667 >= tau, so it pays.

DOCS = {
    "A": "đoạn văn a",
    "B": "đoạn văn b",
    "C": "đoạn văn c",
    "D": "đoạn văn d",
}
# bootstrap reveals A; BM25 then reveals B then C; Dense then reveals D then C.
RANKINGS = {"bm25": ["A", "B", "C"], "dense": ["D", "C", "A"]}


def _answers(ranking=("bm25", "dense")) -> dict:
    """One cached reader answer per reachable state; only (A, B) answers correctly."""
    rank = {"bm25": RANKINGS["bm25"], "dense": RANKINGS["dense"]}
    out = {}
    for state, _history in enumerate_states(rank):
        out[QID, state] = ((CORRECT_CANDIDATE, ("B",), True) if state == ("A", "B")
                           else ("khác", (state[0],), True))
    return out


def _rows(qids=(QID,), **overrides):
    kwargs = dict(
        questions={QID: "câu hỏi?"},
        references={QID: REFERENCE},
        rankings={QID: RANKINGS},
        docs=DOCS,
        reader_answers=_answers(),
        tokenizer=TokenizerStub(),
    )
    kwargs.update(overrides)
    return teacher_rows(list(qids), **kwargs)


# --- state enumeration -----------------------------------------------------------------

def test_enumerated_states_match_reachable_states():
    states = [state for state, _ in enumerate_states(RANKINGS)]
    assert states == reachable_states(RANKINGS)
    assert len(states) == 7  # initial, 2 at depth 1, 4 at depth 2


# --- reward / cost / legality ----------------------------------------------------------

def test_search_cost_is_applied_exactly_once_per_branch():
    rows = _rows()
    root = next(r for r in rows if r["meta"]["history"] == [])
    q = root["meta"]["q_values"]
    # Searches BM25 -> (A, B), whose value is the correct answer's reward 1.0, minus one cost.
    assert q["SEARCH_BM25"] == pytest.approx(1.0 - SEARCH_COST)
    # Dense's branch never improves on ABSTAIN, so it is just -search_cost, not doubled.
    assert q["SEARCH_DENSE"] == pytest.approx(-SEARCH_COST)
    assert q["ANSWER"] == 0.0 and q["ABSTAIN"] == 0.0
    assert root["expected"]["next_action"] == "SEARCH_BM25"


def test_branch_present_only_when_tool_has_unseen_and_budget_remains():
    ranking = {"bm25": ["A", "B", "C"], "dense": ["D"]}  # Dense exhausts after one reveal
    rows = _rows(rankings={QID: ranking})
    # State (A, D): remaining budget is 1, but Dense has nothing new -> no SEARCH_DENSE branch.
    ad = next(r for r in rows if r["meta"]["history"] == [2])
    q = ad["meta"]["q_values"]
    assert "SEARCH_DENSE" not in q and "SEARCH_BM25" in q
    # A depth-2 state has remaining 0 and therefore no SEARCH branch at all.
    leaf = next(r for r in rows if r["meta"]["depth"] == 2)
    assert not any(name.startswith("SEARCH_") for name in leaf["meta"]["q_values"])


def test_rendered_mask_and_soft_target_match_legal_branches():
    for row in _rows(rankings={QID: {"bm25": ["A", "B", "C"], "dense": ["D"]}}):
        legal = row["meta"]["valid_actions"]
        # The rendered "valid actions" line is exactly the legal branch set.
        line = next(text for text in row["state"].splitlines()
                    if text.startswith("Hành động hợp lệ:"))
        assert line.split(": ", 1)[1].split(", ") == legal
        assert row["expected"]["next_action"] in legal
        # Masked actions carry exactly zero mass in the private soft target.
        probs = row["gold"]["next_action"]["probabilities"]
        for name in ACTION_ORDER:
            if name not in legal:
                assert probs[name] == 0.0
        assert sum(probs.values()) == pytest.approx(1.0)


# --- public / private separation -------------------------------------------------------

def test_public_state_carries_no_gold_answer_or_gold_ids():
    gold_doc_ids = ["GOLD-DOC-UNIQUE"]
    for row in _rows():
        state = row["state"]
        # The gold reference and gold context ids never reach the public state. The reader's
        # own candidate text *does* (it is the model's answer), which is not a leak.
        assert REFERENCE not in state
        assert REFERENCE.title() not in state
        for gold_id in gold_doc_ids:
            assert gold_id not in state
        assert "non_null_score" not in state and "null_margin" not in state


def test_row_has_no_forbidden_label_keys_anywhere():
    for row in _rows():
        assert find_forbidden_keys(row) == []
    # The private Q target and the soft target are kept out of the state, in meta/gold.
    root = _rows()[0]
    assert "q_values" in root["meta"] and "state" not in root["meta"]
    assert "probabilities" not in root["state"]


def test_no_fabricated_reader_scores():
    """The VixSTORY reader reports no extractive scores or span provenance."""
    for row in _rows():
        assert "Điểm đọc" not in row["state"]
        assert "non_null" not in row["state"] and "null_margin" not in row["state"]
        assert "Trích dẫn:" in row["state"]
        assert set(row["gold"]["next_action"]["probabilities"]) == set(ACTION_ORDER)


# --- Laya JSONL shape ------------------------------------------------------------------

def test_rows_are_laya_records_with_soft_and_hard_targets():
    rows = _rows()
    assert rows, "expected rows for the caller's qids"
    for row in rows:
        assert isinstance(row["state"], str) and row["state"]
        assert len(row["questions"]["next_action"]["criteria"]) == 4
        assert row["expected"]["next_action"] in ACTION_ORDER
        probs = row["gold"]["next_action"]["probabilities"]
        assert sum(probs.values()) == pytest.approx(1.0)
        assert row["meta"]["qid"] == QID


def test_expected_action_breaks_ties_in_fixed_order():
    assert expected_action({"ANSWER": 1.0, "ABSTAIN": 1.0}) == "ANSWER"
    assert expected_action({"ABSTAIN": 1.0, "SEARCH_BM25": 1.0}) == "ABSTAIN"


# --- caller scope and missing-cache failure --------------------------------------------

def test_rows_returned_only_for_the_caller_qids():
    rankings = {QID: RANKINGS, "qa_other": RANKINGS}
    rows = teacher_rows(
        [QID],
        questions={QID: "câu hỏi?", "qa_other": "khác?"},
        references={QID: REFERENCE, "qa_other": REFERENCE},
        rankings=rankings,
        docs=DOCS,
        reader_answers=_answers(), tokenizer=TokenizerStub(),
    )
    assert {r["meta"]["qid"] for r in rows} == {QID}


def test_missing_cached_answer_is_an_error_not_a_fabricated_row():
    answers = _answers()
    answers.pop((QID, ("A",)))
    with pytest.raises(KeyError, match="no cached reader answer"):
        teacher_rows([QID], questions={QID: "câu hỏi?"}, references={QID: REFERENCE},
                     rankings={QID: RANKINGS}, docs=DOCS, reader_answers=answers,
                     tokenizer=TokenizerStub())


def test_identical_document_state_uses_its_own_question_answer():
    other = "qa_other"
    answers = _answers()
    answers.update({(other, state): ("khác", (state[0],), True)
                    for state, _history in enumerate_states(RANKINGS)})
    rows = _rows(qids=(QID, other),
                 questions={QID: "câu hỏi?", other: "câu hỏi khác?"},
                 references={QID: REFERENCE, other: REFERENCE},
                 rankings={QID: RANKINGS, other: RANKINGS}, reader_answers=answers)
    roots = {row["meta"]["qid"]: row for row in rows if row["meta"]["depth"] == 0}
    assert roots[QID]["meta"]["q_values"]["SEARCH_BM25"] == pytest.approx(0.95)
    assert roots[other]["meta"]["q_values"]["SEARCH_BM25"] == pytest.approx(-0.05)
    answers.pop((other, ("A",)))
    with pytest.raises(KeyError, match="qa_other: no cached reader answer"):
        _rows(qids=(QID, other), questions={QID: "?", other: "?"},
              references={QID: REFERENCE, other: REFERENCE},
              rankings={QID: RANKINGS, other: RANKINGS}, reader_answers=answers)


def test_q_values_terminal_answer_and_abstain():
    def terminal(state):
        return (1.0 if state == ("A",) else 0.0, True)

    value, q = q_values(("A",), RANKINGS, terminal, remaining=0, search_cost=SEARCH_COST)
    assert q == {"ANSWER": 1.0, "ABSTAIN": 0.0} and value == 1.0


@pytest.mark.parametrize("f1,expected", [(0.5 - 2e-12, 0.0),
                                              (0.5 - 5e-13, 1.0),
                                              (0.5, 1.0), (0.5 + 2e-12, 1.0)])
def test_teacher_reward_matches_environment_threshold(f1, expected):
    assert answer_reward(f1, True) == expected
    assert answer_reward(f1, False) == 0.0
    assert answer_reward(f1, False, gate_citation=False) == expected


def test_low_f1_does_not_pay_even_with_valid_citation():
    # (A,) answers incorrectly with a valid citation -> reward 0, so ABSTAIN ties it.
    root = next(r for r in _rows() if r["meta"]["history"] == [])
    assert root["meta"]["q_values"]["ANSWER"] == 0.0
    assert REWARD_TAU == 0.5

"""Dataset suitability diagnostics must not turn evidence annotations into retrieval."""

import pytest

from scripts.vixstory_suitability import (
    audit_record, controlled_initial, evidence_metrics, oracle_advantage,
    sample_controlled, sample_train, search_paths,
)


DOCS = {
    "a": {"context_id": "a", "school_id": "A", "text": "tuyển sinh A"},
    "b": {"context_id": "b", "school_id": "B", "text": "tuyển sinh B"},
    "c": {"context_id": "c", "school_id": "C", "text": "học phí C"},
}


def row(i, gold=("a", "b"), hop=2, kind="comparative"):
    return {"id": f"q{i}", "question": f"Câu hỏi {i}", "answer": "A khác B",
            "context_ids": list(gold), "hop_level": hop, "question_type": kind,
            "contexts": [{"context_id": d, "text": DOCS[d]["text"]} for d in gold]}


def test_audit_multigold_preserves_hop_vs_context_count():
    result = audit_record(row(1, hop=3), DOCS)
    assert result["n_gold_passages"] == 2
    assert result["hop_level"] == 3
    assert result["hop_mismatch"] is True
    assert result["answer_exact_span_in_gold"] is False
    invalid = row(2)
    invalid["context_ids"] = ["a", "z"]
    with pytest.raises(ValueError, match="missing gold passage"):
        audit_record(invalid, DOCS)


def test_train_sampling_is_stable_and_not_gold_conditioned():
    rows = [row(i, gold=("a",) if i % 2 else ("a", "b"), hop=(i % 3) + 1)
            for i in range(20)]
    one = sample_train(rows, n=12, seed=42)
    two = sample_train(list(reversed(rows)), n=12, seed=42)
    assert [r["id"] for r in one] == [r["id"] for r in two]
    assert len(one) == len({r["id"] for r in one}) == 12
    with pytest.raises(ValueError, match="not enough"):
        sample_train(rows, n=21, seed=42)


def test_evidence_reachability_requires_gold_in_same_state():
    rankings = {"bm25": ["a", "c", "b"], "dense": ["a", "b", "c"]}
    paths = search_paths(rankings, gold=("a", "b"), budget=2)
    assert paths["initial_full"] is False
    assert paths["can_complete_within"][1] is True
    assert paths["best_first_full_step"] == 1
    assert paths["initial_gold_count"] == 1
    assert evidence_metrics(("a", "b"), ("a",)) == {"full": False, "recall": 0.5}
    assert evidence_metrics(("a", "b"), ("a", "b"))["full"] is True


def test_union_of_different_branches_is_not_a_full_evidence_state():
    rankings = {"bm25": ["a", "b", "c"], "dense": ["a", "c", "b"]}
    result = search_paths(rankings, gold=("b", "c"), budget=1)
    assert result["initial_full"] is False
    assert result["can_complete_within"][1] is False
    assert result["best_first_full_step"] is None


def test_controlled_initial_is_diagnostic_and_never_adds_new_gold():
    ranks = {"bm25": ["a", "c", "b"], "dense": ["a", "b", "c"]}
    assert controlled_initial(ranks, ("a", "b"), mode="full") is None
    assert controlled_initial(ranks, ("a", "b"), mode="missing_one") == ("a",)
    assert controlled_initial(ranks, ("a", "b"), mode="missing_all") is None
    assert controlled_initial(ranks, ("a", "b"), mode="unreachable") is None
    # This is only a diagnostic starter: optional SEARCH must still walk real rankings.
    assert controlled_initial(ranks, ("a",), mode="full") == ("a",)


def test_controlled_sample_preserves_real_corpus_and_drops_undersized_strata():
    cases = [(f"q{i}", "full" if i < 4 else "first" if i < 7 else "unreachable")
             for i in range(10)]
    sample, counts = sample_controlled(cases, n=8, seed=42)
    assert len(sample) == len({qid for qid, _ in sample}) == 8
    assert counts == {"full": 3, "first": 3, "second": 0, "unreachable": 2}
    assert sample == sample_controlled(list(reversed(cases)), n=8, seed=42)[0]
    assert all(qid.startswith("q") for qid, _ in sample)


def test_oracle_uses_real_reader_returns_not_evidence_labels():
    rankings = {"bm25": ["a", "b", "c"], "dense": ["a", "c", "b"]}
    def terminal(state):
        # Search helps only when the deterministic reader actually improves.
        return (2.0 if "b" in state and "c" in state else -1.0, -1.0)
    outcome = oracle_advantage(rankings, terminal, search_costs={"bm25": 0.1, "dense": 0.25}, budget=2)
    assert outcome["advantage"] > 0
    assert outcome["optimal_action"] in {"SEARCH_BM25", "SEARCH_DENSE"}
    bad = oracle_advantage(rankings, lambda state: (-1.0, 0.8),
                           search_costs={"bm25": 0.1, "dense": 0.25}, budget=2)
    assert bad["advantage"] < 0
    assert bad["optimal_action"] == "ABSTAIN"

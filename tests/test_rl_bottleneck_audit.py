"""Offline retrieval/reader funnel; gold-side diagnostics never become policy inputs."""

from __future__ import annotations

from src.rl.env import EpisodeWorld
from src.rl.reward import EpisodePrivate
from scripts.audit_rl_bottlenecks import classify, summarize
from tests.toy_world import DOC_A, DOC_B, make_reader, span_prediction


def world(qid, *, gold_position=1, answer="1976", gold_answer="1976", impossible=False):
    docs = {"doc_a": DOC_A, "doc_b": DOC_B}
    order = ["doc_b", "doc_a"] if gold_position == 0 else ["doc_a", "doc_b"]
    predictions = {
        (order[0],): span_prediction(qid, order[0], docs[order[0]], "1976" if gold_position == 0 else "1010", 2.0),
        (order[0], order[1]): span_prediction(qid, "doc_b", DOC_B, answer, 3.0),
    }
    return EpisodeWorld(
        qid=qid, question="Năm nào?", split="train_strict", scenario_id=qid,
        corpus=docs, rankings={"bm25": order, "dense": order},
        private=EpisodePrivate(qid=qid, answer_texts=(gold_answer,), is_impossible=impossible,
                               gold_doc_id=None if impossible else "doc_b"),
        reader=make_reader(predictions, docs),
    )


def test_search_reveals_gold_and_reader_recovers():
    result = classify(world("search"), threshold=0.8, max_optional_search=2)
    assert result["gold_position"] == "first_search"
    assert result["bootstrap_reader"] == "wrong_zero_f1"
    assert result["best_with_gold_reader"] == "correct"
    assert result["search_recovers_reader"] is True


def test_bootstrap_gold_but_low_f1_is_reader_problem():
    result = classify(world("low", gold_position=0, answer="1976",
                            gold_answer="1976 và 1990"), threshold=0.8,
                      max_optional_search=2)
    assert result["gold_position"] == "bootstrap"
    assert result["bootstrap_reader"] == "partial_below_threshold"
    assert result["best_with_gold_reader"] == "partial_below_threshold"


def test_negative_and_unreachable_are_not_positive_recovery():
    negative = classify(world("negative", impossible=True), threshold=0.8, max_optional_search=2)
    assert negative["gold_position"] == "not_applicable"
    assert negative["search_recovers_reader"] is False
    assert negative["bootstrap_reader"] == "false_answer"
    docs = {"doc_a": DOC_A, "doc_b": DOC_B}
    unreachable = EpisodeWorld(
        qid="missing", question="Năm nào?", split="train_strict", scenario_id="missing",
        corpus=docs, rankings={"bm25": ["doc_a"], "dense": ["doc_a"]},
        private=EpisodePrivate(qid="missing", answer_texts=("1976",), is_impossible=False,
                               gold_doc_id="doc_b"),
        reader=make_reader({("doc_a",): span_prediction("missing", "doc_a", DOC_A, "1010", 2.0)}, docs))
    assert classify(unreachable, threshold=0.8, max_optional_search=2)["gold_position"] == "unreachable"


def test_summary_uses_distinct_answerable_and_negative_denominators():
    rows = [classify(world("search"), threshold=0.8, max_optional_search=2),
            classify(world("negative", impossible=True), threshold=0.8, max_optional_search=2)]
    summary = summarize(rows)
    assert summary["n"] == 2
    assert summary["gold_position"]["first_search"] == 1
    assert summary["gold_position"]["not_applicable"] == 1
    assert summary["search_recovers_reader"] == 1

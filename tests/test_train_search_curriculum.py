"""Only train-side strata can enter the frozen search-stress curriculum."""

from __future__ import annotations

import pytest

from scripts.select_train_search_scenarios import select


def test_curriculum_has_balanced_nonoverlapping_original_qids():
    rows = [
        {"qid": "recover", "is_impossible": False, "gold_position": "first_search",
         "bootstrap_reader": "no_candidate", "search_recovers_reader": True},
        {"qid": "easy", "is_impossible": False, "gold_position": "bootstrap",
         "bootstrap_reader": "correct", "search_recovers_reader": False},
        {"qid": "fail", "is_impossible": False, "gold_position": "first_search",
         "bootstrap_reader": "no_candidate", "search_recovers_reader": False},
        {"qid": "negative", "is_impossible": True, "gold_position": "not_applicable",
         "bootstrap_reader": "null", "search_recovers_reader": False},
        {"qid": "flagged", "is_impossible": True, "gold_position": "not_applicable",
         "bootstrap_reader": "false_answer", "search_recovers_reader": False},
    ]
    selected = select(rows, flagged={"flagged"}, per_bucket=1)
    assert selected["n"] == 4
    assert set().union(*(set(qids) for qids in selected["selected"].values())) == \
        {"recover", "easy", "fail", "negative"}
    with pytest.raises(ValueError, match="negative"):
        select(rows, flagged={"flagged", "negative"}, per_bucket=1)

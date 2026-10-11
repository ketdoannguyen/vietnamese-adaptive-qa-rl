"""The corrected reachability grid: one reachable path, no double count.

The superseded ``retrieval_budget_grid.json`` reported ``complete_within`` as
``sum(steps) + full_initial``. These tests pin the corrected contract (full evidence must fit
in a single reachable state, and ``complete_within`` is exactly ``sum(steps)``) and, when the
raw inputs are present offline, re-derive the five reader-eval diagnosis rows.
"""

import json

import pytest

from scripts.vixstory_reachability_grid import (
    QIDS_PATH,
    RANKINGS_PATH,
    REPORT_ROWS,
    SNAP,
    build_grid,
    cell_key,
    load_gold,
    reachability_cell,
    verify_report_cells,
)


def rankings(pairs):
    """pairs: (qid, bm25_ranking, dense_ranking)."""
    return {qid: {"bm25": bm25, "dense": dense} for qid, bm25, dense in pairs}


def cell(pairs, gold, **kwargs):
    defaults = {"bootstrap_passages": 1, "passage_per_search_action": 1, "max_optional_search": 1}
    return reachability_cell(rankings(pairs), gold, **{**defaults, **kwargs})


def test_cell_counts_only_evidence_held_in_one_state_not_a_union_of_branches():
    # Each branch reveals exactly one of the two gold passages; no single path holds both.
    result = cell([("q", ["a", "b", "c"], ["a", "c", "b"])], {"q": ("b", "c")})
    assert result["complete_within"] == 0
    assert result["steps"] == {}
    assert result["full_initial"] == 0
    assert result["max_states"] == 3


def test_steps_sum_to_complete_within_and_initial_bucket_is_not_double_counted():
    result = cell(
        [("q0", ["a", "b"], ["a", "b"]), ("q1", ["a", "c", "b"], ["a", "b", "c"])],
        {"q0": ("a",), "q1": ("a", "b")},
    )
    assert result["full_initial"] == 1
    assert result["steps"] == {"0": 1, "1": 1}
    assert result["complete_within"] == 2 == sum(result["steps"].values())
    # The superseded grid would have written sum(steps) + full_initial == 3 here.
    assert result["complete_within"] != sum(result["steps"].values()) + result["full_initial"]


def test_cell_reports_the_minimum_step_over_reachable_paths():
    # Only the bm25 -> bm25 path shows all of a, b, c, and it costs two extra passages.
    result = cell([("q", ["a", "b", "c"], ["a", "x", "y"])], {"q": ("a", "b", "c")},
                  max_optional_search=2)
    assert result["steps"] == {"2": 1}
    assert result["complete_within"] == 1
    assert result["max_states"] == 7


def test_cell_reports_average_and_max_reachable_states():
    # Divergent rankings keep the four depth-2 states distinct: 1 + 2 + 4 == 7.
    result = cell([("q0", ["a", "b", "d", "e"], ["a", "c", "f", "g"]),
                   ("q1", ["a", "b", "d", "e"], ["a", "c", "f", "g"])],
                  {"q0": ("a",), "q1": ("a",)}, max_optional_search=2)
    assert result["max_states"] == 7
    assert result["avg_states"] == 7.0


def test_build_grid_covers_the_axis_product_and_keeps_the_invariant():
    grid = build_grid(rankings([("q", ["a", "b"], ["a", "b"])]), {"q": ("a",)},
                      bootstraps=(1, 2), passages=(1, 3), budgets=(1, 2))
    assert set(grid) == {cell_key(boot, passage, budget)
                         for boot in (1, 2) for passage in (1, 3) for budget in (1, 2)}
    assert all(value["complete_within"] == sum(value["steps"].values()) for value in grid.values())


def test_verify_report_cells_accepts_expected_values_and_flags_drift():
    grid = {cell_key(boot, passage, budget): {"complete_within": expected}
            for boot, passage, budget, expected in REPORT_ROWS}
    verify_report_cells(grid)
    grid[cell_key(1, 1, 2)]["complete_within"] = 157
    with pytest.raises(AssertionError, match="drifted"):
        verify_report_cells(grid)


needs_raw_inputs = pytest.mark.skipif(
    not (RANKINGS_PATH.exists() and (SNAP / "data/train.parquet").exists()),
    reason="raw train500 rankings or the VixSTORY snapshot are not available offline",
)


@needs_raw_inputs
def test_real_grid_reproduces_the_five_diagnosis_rows():
    rankings_data = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    gold = load_gold(json.loads(QIDS_PATH.read_text(encoding="utf-8")))
    for boot, passage, budget, expected in REPORT_ROWS:
        result = reachability_cell(rankings_data, gold, bootstrap_passages=boot,
                                   passage_per_search_action=passage, max_optional_search=budget)
        assert result["complete_within"] == expected
        assert result["complete_within"] == sum(result["steps"].values())

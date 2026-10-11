"""Corrected, reproducible VixSTORY retrieval reachability grid.

The superseded ``outputs/experiments/2026-10-11-vixstory-diagnosis/retrieval_budget_grid.json``
reported ``complete_within = sum(steps) + full_initial`` for every cell, so the questions that
already held their full evidence at the bootstrap were counted twice. This generator recomputes
the same 3x3x3 grid (bootstrap passages x passages per SEARCH x SEARCH budget) from the raw
BM25/E5 rankings and the gold passage IDs, and reports ``complete_within == sum(steps)``.

Evidence rule (the environment's, not this script's): a question is complete only when *all* of
its gold passages are visible in a *single* reachable state. A union of two different search
branches is never a full-evidence state, so a question is never credited for evidence it cannot
actually hold on one path.

Output is a new versioned artifact; the superseded grid is never overwritten.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq

from src.retrieval.states import reachable_states
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json

REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
BOOTSTRAP_TOOL = "bm25"
TOOLS = ("bm25", "dense")
BOOTSTRAPS = (1, 2, 3)
PASSAGES_PER_SEARCH = (1, 2, 3)
BUDGETS = (1, 2, 3)

DIAGNOSIS = Path("outputs/experiments/2026-10-11-vixstory-diagnosis")
SUPERSEDED = DIAGNOSIS / "retrieval_budget_grid.json"
RANKINGS_PATH = DIAGNOSIS / "train500_rankings_top300.json"
QIDS_PATH = Path("outputs/experiments/2026-10-11-vixstory-suitability/train_sample_qids.json")
DEFAULT_OUT = DIAGNOSIS / "retrieval_budget_grid_v2.json"
SNAP = (Path.home() / ".cache/huggingface/hub/datasets--ura-hcmut--VixSTORY"
        / "snapshots" / REVISION)

# The five (bootstrap_passages, passage_per_search_action, max_optional_search, complete_within)
# rows of the reader-eval diagnosis table, independently recomputed from the raw rankings.
REPORT_ROWS = ((1, 1, 2, 108), (1, 1, 3, 119), (2, 2, 2, 131), (3, 3, 2, 151), (3, 3, 3, 166))


def cell_key(bootstrap_passages: int, passage_per_search_action: int, max_optional_search: int) -> str:
    return f"boot{bootstrap_passages}_p{passage_per_search_action}_b{max_optional_search}_dual"


def reachability_cell(rankings: Mapping[str, Mapping[str, Sequence[str]]],
                      gold: Mapping[str, Sequence[str]], *, bootstrap_passages: int,
                      passage_per_search_action: int, max_optional_search: int,
                      bootstrap_tool: str = BOOTSTRAP_TOOL, tools: Sequence[str] = TOOLS) -> dict:
    """Count questions whose full evidence fits in one reachable state, by extra passages.

    ``steps[s]`` counts questions whose earliest complete state reveals ``s`` passages beyond
    the bootstrap; ``complete_within`` is their sum, with no second term.
    """
    if not gold:
        raise ValueError("no questions to score")
    steps: Counter[int] = Counter()
    states_total = 0
    states_max = 0
    for qid, gold_ids in gold.items():
        ranking = rankings[qid]
        states = reachable_states({tool: ranking[tool] for tool in tools},
                                  bootstrap_tool=bootstrap_tool,
                                  bootstrap_passages=bootstrap_passages,
                                  passage_per_search_action=passage_per_search_action,
                                  max_optional_search=max_optional_search, tools=list(tools))
        first = states[0]
        target = set(gold_ids)
        best = min((len(state) - len(first) for state in states if target <= set(state)), default=None)
        states_total += len(states)
        states_max = max(states_max, len(states))
        if best is not None:
            steps[best] += 1
    return {"full_initial": steps.get(0, 0), "complete_within": sum(steps.values()),
            "steps": {str(step): steps[step] for step in sorted(steps)},
            "avg_states": round(states_total / len(gold), 1), "max_states": states_max}


def build_grid(rankings: Mapping[str, Mapping[str, Sequence[str]]],
               gold: Mapping[str, Sequence[str]], *, bootstraps: Sequence[int] = BOOTSTRAPS,
               passages: Sequence[int] = PASSAGES_PER_SEARCH, budgets: Sequence[int] = BUDGETS,
               bootstrap_tool: str = BOOTSTRAP_TOOL, tools: Sequence[str] = TOOLS) -> dict:
    return {cell_key(boot, passage, budget): reachability_cell(
        rankings, gold, bootstrap_passages=boot, passage_per_search_action=passage,
        max_optional_search=budget, bootstrap_tool=bootstrap_tool, tools=tools)
        for boot in bootstraps for passage in passages for budget in budgets}


def verify_report_cells(grid: Mapping[str, Mapping]) -> None:
    """Recomputation gate: the five reader-eval diagnosis rows must land exactly."""
    drift = {cell_key(boot, passage, budget): (grid[cell_key(boot, passage, budget)]["complete_within"],
                                               expected)
             for boot, passage, budget, expected in REPORT_ROWS
             if grid[cell_key(boot, passage, budget)]["complete_within"] != expected}
    if drift:
        raise AssertionError(f"reachability grid drifted from the recomputed rows: {drift}")


def load_gold(qids: Sequence[str], *, snapshot: Path = SNAP) -> dict[str, tuple[str, ...]]:
    train = {row["id"]: row for row in pq.read_table(snapshot / "data/train.parquet").to_pylist()}
    missing = [qid for qid in qids if qid not in train]
    if missing:
        raise ValueError(f"{len(missing)} sampled qids are absent from the train split")
    return {qid: tuple(train[qid]["context_ids"]) for qid in qids}


def run(*, out: Path = DEFAULT_OUT, rankings_path: Path = RANKINGS_PATH, qids_path: Path = QIDS_PATH,
        snapshot: Path = SNAP) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rankings = json.loads(Path(rankings_path).read_text(encoding="utf-8"))
    qids = json.loads(Path(qids_path).read_text(encoding="utf-8"))
    gold = load_gold(qids, snapshot=snapshot)
    grid = build_grid(rankings, gold)
    verify_report_cells(grid)
    artifact = {
        "artifact": "vixstory_retrieval_reachability_grid", "version": 2,
        "supersedes": SUPERSEDED.name,
        "correction": "the superseded grid counted complete_within = sum(steps) + full_initial; "
                      "here complete_within == sum(steps), so the bootstrap-only questions "
                      "are counted once.",
        "evidence_rule": "a question is complete only when every gold passage is visible in one "
                         "reachable state; a union of separate branches never counts.",
        "steps_rule": "steps[s] = questions whose earliest complete state reveals s passages beyond "
                      "the bootstrap; complete_within == sum(steps.values()).",
        "inputs": {"rankings_path": str(rankings_path), "rankings_sha256": sha256_file(rankings_path),
                   "qids_path": str(qids_path), "qids_sha256": sha256_file(qids_path),
                   "n_questions": len(qids), "bootstrap_tool": BOOTSTRAP_TOOL, "tools": list(TOOLS)},
        "grid": grid,
        "report_cells": {cell_key(boot, passage, budget): expected
                         for boot, passage, budget, expected in REPORT_ROWS},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rankings", type=Path, default=RANKINGS_PATH)
    parser.add_argument("--qids", type=Path, default=QIDS_PATH)
    args = parser.parse_args()
    artifact = run(out=args.out, rankings_path=args.rankings, qids_path=args.qids)
    print(json.dumps({"out": str(args.out), "report_cells": artifact["report_cells"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

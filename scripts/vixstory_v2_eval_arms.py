"""Evaluate the fixed V2 arms E00/E01/E02 plus a seeded random sanity baseline on validation.

Plan §7 scores E00/E01/E02 on the same 246 validation QIDs as the learned arms, choosing
K ∈ {1, 2} for E01/E02 with the shared selection rule. The random row answers the
"is the policy below random?" question and is not part of the E00-E07 matrix.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.pipeline.vixstory_arms_v2 import fixed_tool, stop
from src.pipeline.vixstory_eval_v2 import evaluate_policy, select_best
from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2
from src.pipeline.vixstory_worlds_v2 import load_worlds


def private_json_new(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def random_policy(seed: int) -> Callable[[Mapping[str, Any]], int]:
    """Uniform over the legal actions; deterministic for the seed."""
    rng = np.random.default_rng(seed)

    def _policy(obs: Mapping[str, Any]) -> int:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        return int(rng.choice(np.flatnonzero(mask)))

    return _policy


def run(run_dir: Path) -> dict:
    val = load_worlds(run_dir, "validation")
    if len(val) != 246 or len({w.qid for w in val}) != 246:
        raise RuntimeError("expected all 246 distinct official validation QIDs")
    out = run_dir / "baselines"
    out.mkdir(mode=0o700, exist_ok=True)
    policies: dict[str, Callable[[Mapping[str, Any]], int]] = {
        "E00": stop, "random-seed42": random_policy(42)}
    for arm, tool in (("E01", "bm25"), ("E02", "dense")):
        for k in (1, 2):
            policies[f"{arm}-K{k}"] = fixed_tool(tool, k)
    summaries = {}
    for name, policy in policies.items():
        evaluation = evaluate_policy(val, policy, name, teacher_factory=VixStoryTeacherV2)
        private_json_new(out / f"{name}-validation-private.json", evaluation)
        summary = evaluation["summary"]
        summaries[name] = summary
        print(json.dumps({key: summary[key] for key in
                          ("arm", "n", "mean_retrieval_return", "mean_full_gold_success",
                           "full_gold_success_n", "mean_gold_recall", "mean_gold_precision",
                           "mean_n_search")}), flush=True)
    selection = {}
    for arm in ("E01", "E02"):
        candidates = [{**summaries[f"{arm}-K{k}"], "index": k} for k in (1, 2)]
        selection[arm] = select_best(candidates)
    result = {"oracle": summaries["E00"]["oracle"], "selection": selection}
    private_json_new(out / "selection.json", result)
    selected = {arm: {"index": selection[arm]["index"],
                      "mean_retrieval_return": selection[arm]["mean_retrieval_return"]}
                for arm in selection}
    print(json.dumps({"oracle": result["oracle"], "selected": selected}), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()

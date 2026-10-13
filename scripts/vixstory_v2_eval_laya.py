"""Select real E04/E05 epoch checkpoints on all official V2 validation questions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.decision.laya_adapter_v2 import LayaAdapterV2
from src.pipeline.vixstory_arms_v2 import masked_greedy
from src.pipeline.vixstory_eval_v2 import evaluate_policy, select_best
from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2
from src.pipeline.vixstory_worlds_v2 import load_worlds


def private_json_new(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def run(run_dir: Path, arm: str) -> dict:
    if arm not in ("E04", "E05"):
        raise ValueError("expected E04 or E05")
    val = load_worlds(run_dir, "validation")
    if len(val) != 246 or len({w.qid for w in val}) != 246:
        raise RuntimeError("expected all 246 distinct official validation QIDs")
    arm_dir = run_dir / "laya" / arm
    candidates = []
    for epoch in (1, 2, 3):
        checkpoint = arm_dir / f"epoch-{epoch}"
        if not (checkpoint / "model.safetensors").is_file():
            raise FileNotFoundError(checkpoint)
        adapter = LayaAdapterV2.load(checkpoint, device="cuda")
        try:
            evaluation = evaluate_policy(val, masked_greedy(lambda obs: adapter.predict_probs([obs])[0]), arm,
                                         teacher_factory=VixStoryTeacherV2)
            summary = {**evaluation["summary"], "index": epoch,
                       "checkpoint_sha256": adapter.model_hash()}
            private_json_new(arm_dir / f"epoch-{epoch}-validation-private.json", evaluation)
            candidates.append(summary)
            print(json.dumps({"arm": arm, "epoch": epoch, "checkpoint_sha256": adapter.model_hash(),
                              "mean_retrieval_return": summary["mean_retrieval_return"],
                              "mean_full_gold_success": summary["mean_full_gold_success"],
                              "mean_n_search": summary["mean_n_search"], "n": summary["n"]}),
                  flush=True)
        finally:
            adapter.close()
    winner = select_best(candidates)
    result = {"arm": arm, "candidates": candidates, "selected_epoch": winner["index"],
              "frozen_checkpoint_sha256": winner["checkpoint_sha256"]}
    private_json_new(arm_dir / "selection.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("E04", "E05"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir, args.arm)), flush=True)


if __name__ == "__main__":
    main()

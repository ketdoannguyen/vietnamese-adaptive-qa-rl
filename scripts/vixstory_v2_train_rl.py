"""Train E06/E07 against frozen selected Laya priors, then select on validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from scripts.vixstory_v2_eval_laya import private_json_new
from src.decision.laya_adapter_v2 import LayaAdapterV2
from src.pipeline.vixstory_arms_v2 import masked_greedy
from src.pipeline.vixstory_eval_v2 import evaluate_policy, select_best
from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2
from src.pipeline.vixstory_worlds_v2 import load_worlds
from src.rl.vixstory_env_v2 import VixStoryEnvV2
from src.rl.vixstory_policy_v2 import (
    V2RLConfig,
    VixStoryV2ReinforceTrainer,
    policy_probs,
)
from src.utils.hashing import sha256_file

PRIOR_ARM = {"E06": "E05", "E07": "E04"}
EXPECTED_UPDATES = [*range(25, 241, 25), 240]


def selected_prior(run_dir: Path, arm: str) -> tuple[Path, str]:
    source = PRIOR_ARM[arm]
    selection = json.loads((run_dir / "laya" / source / "selection.json").read_text())
    if selection.get("arm") != source or len(selection.get("candidates", [])) != 3:
        raise ValueError(f"{source} selection is incomplete")
    epoch = selection.get("selected_epoch")
    if epoch not in (1, 2, 3):
        raise ValueError(f"{source} selected epoch is invalid")
    candidate = selection["candidates"][epoch - 1]
    if (candidate.get("index") != epoch or candidate.get("n") != 246
            or candidate.get("checkpoint_sha256") != selection.get("frozen_checkpoint_sha256")):
        raise ValueError(f"{source} selected checkpoint/validation does not match candidates")
    checkpoint = run_dir / "laya" / source / f"epoch-{epoch}"
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, selection["frozen_checkpoint_sha256"]


def run(run_dir: Path, arm: str, stage: str) -> dict:
    if arm not in PRIOR_ARM or stage not in ("train", "select"):
        raise ValueError("expected E06/E07 and train/select")
    arm_dir = run_dir / "rl" / arm
    checkpoint, expected_hash = selected_prior(run_dir, arm)
    if stage == "train" and arm_dir.exists():
        raise FileExistsError(arm_dir)
    if stage == "select" and (arm_dir / "selection.json").exists():
        raise FileExistsError(arm_dir / "selection.json")
    adapter = LayaAdapterV2.load(checkpoint, device="cuda")
    try:
        if adapter.model_hash() != expected_hash:
            raise ValueError("selected Laya checkpoint changed since validation")
        def prior(obs):
            return adapter.predict_probs([obs])[0]
        config = V2RLConfig()
        if stage == "train":
            worlds = load_worlds(run_dir, "train")
            if len(worlds) != 2553 or len({world.qid for world in worlds}) != 2553:
                raise ValueError("training requires 2553 distinct train QIDs")
            arm_dir.parent.mkdir(parents=True, exist_ok=True)
            original_umask = os.umask(0o077)
            try:
                trainer = VixStoryV2ReinforceTrainer(
                    VixStoryEnvV2(worlds), config, prior=prior, prior_hash=expected_hash,
                    run_dir=arm_dir,
                )
                result = trainer.train([world.qid for world in worlds])
            finally:
                os.umask(original_umask)
            if (result["n_episodes"] != 7659 or result["n_updates"] != 240
                    or result["episodes_per_qid"] != {"min": 3, "max": 3, "distinct_qids": 2553}):
                raise RuntimeError("RL schedule did not cover every train QID three times")
            return {key: result[key] for key in ("identity", "prior_hash", "schedule_sha256",
                                                   "n_episodes", "n_updates", "episodes_per_qid")}

        summary = json.loads((arm_dir / "summary.json").read_text())
        if (summary.get("prior_hash") != expected_hash or summary.get("n_episodes") != 7659
                or summary.get("n_updates") != 240
                or summary.get("episodes_per_qid") != {"min": 3, "max": 3, "distinct_qids": 2553}
                or summary.get("config") != vars(config)):
            raise ValueError("RL summary does not match pinned training protocol")
        checkpoints = arm_dir / "checkpoints"
        found = sorted(checkpoints.glob("update_*"))
        expected = [checkpoints / f"update_{update:04d}" for update in EXPECTED_UPDATES]
        if found != expected:
            raise ValueError("RL checkpoint cadence is incomplete")
        worlds = load_worlds(run_dir, "validation")
        if len(worlds) != 246 or len({world.qid for world in worlds}) != 246:
            raise ValueError("selection requires 246 distinct validation QIDs")
        candidates = []
        for index, (update, path) in enumerate(zip(EXPECTED_UPDATES, expected), start=1):
            metadata = json.loads((path / "checkpoint.json").read_text())
            if (metadata.get("update") != update or metadata.get("identity") != summary.get("identity")
                    or metadata.get("prior_hash") != expected_hash or metadata.get("n_actions") != 3
                    or metadata.get("config") != vars(config)):
                raise ValueError(f"RL checkpoint metadata mismatch: {path}")
            trainer = VixStoryV2ReinforceTrainer(VixStoryEnvV2(worlds), config,
                                                  prior=prior, prior_hash=expected_hash)
            trainer.load(path)
            evaluation = evaluate_policy(
                worlds, masked_greedy(policy_probs(trainer.net, trainer.extractor,
                                                   trainer.scaler_t, prior)), arm,
                teacher_factory=VixStoryTeacherV2,
            )
            policy_hash = sha256_file(path / "policy.safetensors")
            private_json_new(arm_dir / f"update_{update:04d}-validation-private.json", evaluation)
            candidate = {**evaluation["summary"], "index": index, "update": update,
                         "policy_sha256": policy_hash, "prior_hash": expected_hash}
            candidates.append(candidate)
            print(json.dumps({"arm": arm, "update": update, "n": candidate["n"],
                              "mean_retrieval_return": candidate["mean_retrieval_return"]}),
                  flush=True)
        winner = select_best(candidates)
        selection = {"arm": arm, "prior_arm": PRIOR_ARM[arm], "prior_hash": expected_hash,
                     "candidates": candidates, "selected_update": winner["update"],
                     "selected_policy_sha256": winner["policy_sha256"]}
        private_json_new(arm_dir / "selection.json", selection)
        return selection
    finally:
        adapter.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=tuple(PRIOR_ARM))
    parser.add_argument("--stage", required=True, choices=("train", "select"))
    args = parser.parse_args()
    print(json.dumps(run(args.run_dir, args.arm, args.stage)), flush=True)


if __name__ == "__main__":
    main()

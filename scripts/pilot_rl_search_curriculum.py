"""Train no-prior REINFORCE on balanced existing train episodes; compare val_tune.

Gold-aware curriculum selection happened offline on train only. The policy sees
observations, not oracle labels. No val_report/test gold is loaded. Historical models,
retrieval, reader and reward stay immutable; all new checkpoints live under ROOT.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_rl_bottlenecks import train_worlds
from scripts.audit_rl_search_value import PROTOCOL, verify_protocol, tune_worlds
from src.eval import reporting, splitting
from src.eval.policies import BootstrapThenAnswerPolicy, FixedToolPolicy, RLPolicy
from src.eval.rollout import run_policy
from src.pipeline import worlds
from src.rl.env import ABSTAIN, AdaptiveQAEnv
from src.rl.reinforce import RLConfig, ReinforceTrainer, evaluate
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import load_config
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import read_yaml

ROOT = Path("outputs/experiments/2026-10-11-rl-bottleneck/rl-curriculum-pilot")
CURRICULUM = ROOT.parent / "train_search_curriculum.json"


class AlwaysAbstain:
    id = "B_const_abstain"
    variant = "constant"

    def begin_episode(self, *, qid: str, seed: int) -> None:
        pass

    def act(self, obs: dict) -> int:
        return ABSTAIN


def validate_curriculum(selected: dict, train, flagged: set[str], tune: list[str], report: list[str]) -> list:
    qids = [qid for values in selected["selected"].values() for qid in values]
    if len(qids) != selected["n"] or len(set(qids)) != len(qids):
        raise ValueError("curriculum QIDs duplicate or missing")
    if sha256_json(sorted(qids)) != selected["qids_sha256"]:
        raise ValueError("curriculum hash changed")
    if set(qids) & (set(tune) | set(report) | flagged):
        raise ValueError("train curriculum overlaps reserved validation or excluded negatives")
    by_qid = {w.qid: w for w in train}
    if not set(qids) <= set(by_qid):
        raise ValueError("curriculum QIDs are not all train worlds")
    return [by_qid[qid] for qid in sorted(qids)]


def metric_rows(env: AdaptiveQAEnv, world_list: list, trainer: ReinforceTrainer,
                tune_groups: dict[str, str], ambiguous: frozenset[str]) -> dict:
    assert trainer.net is not None and trainer.scaler_t is not None
    policies = {
        "RL_no_prior": RLPolicy(net=trainer.net, extractor=trainer.extractor,
                               scaler_t=trainer.scaler_t, laya_fn=None, prior_regime=None),
        "E00_bootstrap_only": BootstrapThenAnswerPolicy(),
        "E01_fixed_bm25": FixedToolPolicy(tool="bm25"),
        "E02_fixed_dense": FixedToolPolicy(tool="dense"),
        "constant_abstain": AlwaysAbstain(),
    }
    outputs = {}
    for name, policy in policies.items():
        episodes = run_policy(env, policy, worlds=world_list, seed=42,
                              run_id="rl-search-curriculum-pilot", exp_id=name, dev=True)
        if len(episodes) != len(world_list) or {x.qid for x in episodes} != {w.qid for w in world_list}:
            raise ValueError("paired baseline did not evaluate exactly the same tune QIDs")
        metrics = reporting.aggregate(episodes, groups=tune_groups,
                                      ambiguous_qids=ambiguous, n_resamples=500)
        outputs[name] = {
            "metrics": {k: metrics[k] for k in (
                "n_episodes", "n_answerable", "n_unanswerable", "mean_episode_reward",
                "token_f1", "answerable_em", "coverage", "false_answer_rate_on_unanswerables",
                "avg_optional_retrievals", "action_count_bm25", "action_count_dense",
                "reward_ci_low", "reward_ci_high", "ci_groups")},
            "per_qid": {ep.qid: {"reward": ep.episode_return,
                                 "f1": ep.eval_correctness_f1,
                                 "searches": ep.optional_used}
                        for ep in episodes},
        }
    base = outputs["E00_bootstrap_only"]["per_qid"]
    rl = outputs["RL_no_prior"]["per_qid"]
    outputs["rl_minus_E00_reward_ci"] = splitting.paired_cluster_bootstrap_ci(
        {qid: rl[qid]["reward"] - base[qid]["reward"] for qid in base},
        tune_groups, n_resamples=500, seed=42)
    outputs["rl_minus_E00_f1_ci"] = splitting.paired_cluster_bootstrap_ci(
        {qid: rl[qid]["f1"] - base[qid]["f1"] for qid in base},
        tune_groups, n_resamples=500, seed=42)
    return outputs


def main() -> None:
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite RL pilot: {ROOT}")
    cfg = load_config("medium", extra=read_yaml("configs/rl_audit_medium.yaml"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    tune, groups = verify_protocol(cfg, protocol)
    train, flagged, train_groups = train_worlds(cfg)
    val = tune_worlds(cfg, tune)
    if set(train_groups.values()) & {groups[qid] for qid in tune + protocol["report_qids"]}:
        raise ValueError("strict train and validation title groups overlap")
    curriculum = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    if curriculum["diagnosis_sha256"] != sha256_file(CURRICULUM.parent / "diagnosis.json"):
        raise ValueError("train-only cohort diagnosis hash changed")
    chosen = validate_curriculum(curriculum, train, flagged, tune, protocol["report_qids"])
    reward = RewardCalculator(RewardConfig.from_dict(cfg))
    config = worlds.env_config(cfg)
    train_env = AdaptiveQAEnv(chosen, reward, config)
    val_env = AdaptiveQAEnv(val, reward, config)
    rl_cfg = RLConfig.from_dict(cfg)
    rl_cfg.max_updates = 24
    rl_cfg.eval_every = 4
    rl_cfg.early_stop_patience = 4
    ROOT.mkdir(parents=True, exist_ok=False)
    trainer = ReinforceTrainer(train_env, val_env, rl_cfg, laya_fn=None,
                               device="cpu", run_dir=ROOT, seed=42, val_qids=tune)
    trainer.fit_scaler()
    trainer.build()
    training = trainer.train()
    if training["best_update"] is None:
        raise RuntimeError("no checkpoint was selected on fixed tune QIDs")
    trainer.load_best(ROOT / "best")
    assert trainer.net is not None
    best = evaluate(val_env, trainer.net, trainer.extractor, trainer.scaler_t, None,
                    episodes=len(tune), qids=tune, device=trainer.device)
    chosen_val = [w for w in val if w.qid in set(tune)]
    outputs = metric_rows(val_env, chosen_val, trainer,
                          {qid: groups[qid] for qid in tune},
                          worlds.load_ambiguous_qids(cfg, "val_strict"))
    summary = {"protocol_sha256": sha256_file(PROTOCOL),
               "curriculum_sha256": sha256_file(CURRICULUM),
               "tune_qid_sha256": sha256_json(tune),
               "train_selected_qids_sha256": curriculum["qids_sha256"],
               "reader_state_meta_train_sha256": sha256_file(
                   worlds.state_cache_paths(cfg, "train_strict")[1]),
               "reader_state_meta_val_sha256": sha256_file(
                   worlds.state_cache_paths(cfg, "val_strict")[1]),
               "reward_spec_hash": reward.config.spec_hash(),
               "train_worlds": len(chosen), "val_worlds": len(val),
               "rl_config": rl_cfg.__dict__, "training": training,
               "best_remeasured": best, "comparison": outputs,
               "caveats": "Single-seed pilot; train cohort selected with train-only gold; "
                          "validation natural and historically exposed; no val_report/test scored."}
    with (ROOT / "comparison.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print("trained RL update", training["best_update"], "val reward", best["mean_reward"])
    for key, row in outputs.items():
        if "metrics" in row:
            m = row["metrics"]
            print(key, "reward", m["mean_episode_reward"], "F1", m["token_f1"],
                  "searches", m["avg_optional_retrievals"])
    print("paired RL-E00 reward CI:", outputs["rl_minus_E00_reward_ci"])


if __name__ == "__main__":
    main()

"""Create isolated paired hard-label fixtures; compare state predictions and QA rollouts.

Historical teacher and frozen reader/retrieval inputs are only read, never rewritten.
Usage: .venv/bin/python scripts/paired_pilot.py prepare|report
"""
from __future__ import annotations

import argparse
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.decision.schema import ACTION_ORDER
from src.decision.state_text import state_hash
from src.utils.config import REPO_ROOT
from src.utils.hashing import sha256_file
from src.utils.io import read_jsonl

ROOT = REPO_ROOT / "outputs/experiments/2026-10-09-gliner-laya"
SOURCE = REPO_ROOT / "data/processed/laya_train.jsonl"
TASK = "next_action"


def order(value):
    return hashlib.sha256(("paired-pilot-20261010:" + value).encode()).hexdigest()


def write_new(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare():
    targets = (ROOT / "data/processed/laya_train.jsonl", ROOT / "fixtures/val_states.jsonl",
               ROOT / "fixtures/manifest.json")
    if any(p.exists() for p in targets):
        raise FileExistsError("paired pilot fixture already exists; refusing overwrite")
    forbidden = set()
    for name in ("qids_test_strict.txt", "qids_val_strict.txt"):
        forbidden.update((REPO_ROOT / "data/processed" / name).read_text().splitlines())
    groups = defaultdict(list)
    seen = set()
    for row in read_jsonl(SOURCE):
        meta = row["meta"]
        qid = meta["qid"]
        label = row["expected"][TASK]
        group = meta["difficulty"]["source_doc_id"]
        key = (qid, meta["state_hash"])
        if (qid in forbidden or meta["split"] != "train_strict" or meta.get("privileged")
                or label not in ACTION_ORDER or label not in meta["valid_actions"]
                or meta.get("state_text_version") != "v2" or not group
                or key in seen or state_hash(row["state"]) != meta["state_hash"]):
            raise ValueError(f"invalid teacher row {key}")
        seen.add(key)
        groups[group].append(row)
    group_ids = sorted(groups, key=order)
    # Entire source documents are assigned to one side before selecting any state.
    train_groups = set(group_ids[:int(len(group_ids) * .8)])
    pool_train = [r for g in train_groups for r in groups[g]]
    pool_val = [r for g in group_ids if g not in train_groups for r in groups[g]]
    selected = []
    quotas = {"ANSWER": 120, "SEARCH_BM25": 30, "SEARCH_DENSE": 30, "ABSTAIN": 120}
    for label in ACTION_ORDER:
        candidates = sorted((r for r in pool_train if r["expected"][TASK] == label),
                            key=lambda r: order(r["meta"]["qid"] + r["meta"]["state_hash"]))
        if len(candidates) < quotas[label]:
            raise ValueError(f"insufficient {label}: {len(candidates)}")
        selected.extend(candidates[:quotas[label]])
    selected.sort(key=lambda r: order(r["meta"]["qid"] + r["meta"]["state_hash"]))
    # Unmodified distribution for diagnostic validation, selected by whole question.
    val_qids = sorted({r["meta"]["qid"] for r in pool_val}, key=order)[:30]
    val = [r for r in pool_val if r["meta"]["qid"] in val_qids]
    val.sort(key=lambda r: (r["meta"]["qid"], r["meta"]["state_hash"]))
    train_qids = {r["meta"]["qid"] for r in selected}
    if not val or train_qids & set(val_qids):
        raise ValueError("training/validation questions overlap or validation is empty")
    # Convert the existing soft teacher target to exactly the same hard action on both sides.
    hard = []
    for row in selected:
        new = dict(row)
        action = row["expected"][TASK]
        new["gold"] = {TASK: {"probabilities": {name: float(name == action)
                                                 for name in ACTION_ORDER}}}
        hard.append(new)
    write_new(targets[0], hard)
    write_new(targets[1], [{"qid": r["meta"]["qid"], "state_hash": r["meta"]["state_hash"],
                            "state": r["state"], "label": r["expected"][TASK],
                            "valid_actions": r["meta"]["valid_actions"]} for r in val])
    manifest = {
        "teacher_source": str(SOURCE), "teacher_sha256": sha256_file(SOURCE),
        "split": "train_strict grouped by source_doc_id (80/20 deterministic hash)",
        "train_selection": "train-only quotas: answer=120, bm25=30, dense=30, abstain=120",
        "validation_selection": "30 hashed qids in held-out groups; all states, natural class mix",
        "train_states": len(hard), "val_states": len(val),
        "train_qids": len(train_qids), "val_qids": len(val_qids),
        "train_labels": dict(Counter(r["expected"][TASK] for r in hard)),
        "val_labels": dict(Counter(r["expected"][TASK] for r in val)),
        "train_teacher_sha256": sha256_file(targets[0]), "val_sha256": sha256_file(targets[1]),
        "train_val_group_overlap": 0, "train_val_qid_overlap": 0,
        "strict_val_test_qid_overlap": 0,
    }
    targets[2].parent.mkdir(parents=True, exist_ok=True)
    with targets[2].open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False))


def episode_requests():
    from src.pipeline.worlds import build_worlds, env_config, reachable_state_requests
    from src.rl.reward import RewardCalculator, RewardConfig
    from src.utils.config import load_config

    output = ROOT / "fixtures/episode_requests.jsonl"
    if output.exists():
        raise FileExistsError(f"episode requests exist: {output}")
    historical = load_config("medium")
    bundle = build_worlds(historical, "val_strict", limit=100, laya_regime=None)
    reward = RewardCalculator(RewardConfig.from_dict(historical))
    requests = reachable_state_requests(bundle.worlds, reward, env_config(historical))
    write_new(output, [{"qid": qid, "state_hash": digest, "state": text}
                       for qid, digest, text in requests])
    print(json.dumps({"validation_worlds": len(bundle.worlds), "reachable_states": len(requests),
                      "sha256": sha256_file(output)}))


def score_metrics(actual, predicted):
    confusion = {name: dict.fromkeys(ACTION_ORDER, 0) for name in ACTION_ORDER}
    for a, p in zip(actual, predicted):
        confusion[a][p] += 1
    classes = {}
    for name in ACTION_ORDER:
        tp = confusion[name][name]
        support = sum(confusion[name].values())
        predicted_n = sum(confusion[a][name] for a in ACTION_ORDER)
        precision = tp / predicted_n if predicted_n else 0.0
        recall = tp / support if support else None
        f1 = 2 * precision * recall / (precision + recall) if recall and precision else 0.0
        classes[name] = {"support": support, "predicted": predicted_n,
                         "precision": precision, "recall": recall, "f1": f1 if support else None}
    defined = [c["f1"] for c in classes.values() if c["support"]]
    return {"n": len(actual), "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual),
            "macro_f1_present_classes": float(np.mean(defined)), "classes": classes,
            "confusion_true_by_pred": confusion}


def report():
    from src.eval.reporting import aggregate
    from src.eval.rollout import run_policy
    from src.pipeline.worlds import build_worlds, env_config, prior_lookup
    from src.rl.env import AdaptiveQAEnv
    from src.rl.reward import RewardCalculator, RewardConfig
    from src.utils.config import load_config
    from src.utils.io import read_json

    targets = [ROOT / "reports/paired_pilot.json", ROOT / "reports/paired_pilot.md"]
    if any(p.exists() for p in targets):
        raise FileExistsError("pilot report exists; refusing overwrite")
    val = list(read_jsonl(ROOT / "fixtures/val_states.jsonl"))
    cfg = load_config("medium")  # historical artifacts are read-only
    bundle = build_worlds(cfg, "val_strict", limit=100, laya_regime=None)
    reward = RewardCalculator(RewardConfig.from_dict(cfg))
    config = env_config(cfg)
    result = {"teacher_validation": {}, "val_strict_episode": {},
              "limitations": ["Single supervised seed; search classes are rare in natural validation.",
                              "Historical teacher/reader and historically exposed strict validation reused.",
                              "The same 30 randomly held-out Laya calibration rows were excluded from GLiNER weight training; GLiNER has no fitted calibration.",
                              "Laya full fine-tuning vs GLiNER LoRA is an applied configuration comparison.",
                              "The hard-label teacher action is not a gold QA outcome."]}
    env = AdaptiveQAEnv(bundle.worlds, reward, config)
    for backend in ("laya", "gliner"):
        rows = list(read_jsonl(ROOT / "outputs/diagnostics" / f"{backend}_val_states.jsonl"))
        if [(r["qid"], r["state_hash"]) for r in rows] != [
                (r["qid"], r["state_hash"]) for r in val]:
            raise ValueError(f"{backend}: validation state alignment failed")
        probs = np.array([r["probs"] for r in rows], dtype=float)
        masked = [ACTION_ORDER[int(np.argmax(np.where(
            np.array([a in v["valid_actions"] for a in ACTION_ORDER]), p, -np.inf)))]
                  for v, p in zip(val, probs)]
        m = score_metrics([r["label"] for r in val], masked)
        m["raw_argmax_illegal"] = sum(ACTION_ORDER[int(p.argmax())] not in v["valid_actions"]
                                          for v, p in zip(val, probs))
        m["nll"] = float(np.mean([-np.log(max(p[ACTION_ORDER.index(v["label"])], 1e-12))
                                      for v, p in zip(val, probs)]))
        result["teacher_validation"][backend] = m
        episode_rows = list(read_jsonl(ROOT / "outputs/diagnostics" / f"{backend}_val_episodes.jsonl"))
        requests = {(r["qid"], r["state_hash"]): np.asarray(r["probs"], dtype=float)
                    for r in episode_rows}
        from src.eval.policies import LayaArgmaxPolicy
        policy = LayaArgmaxPolicy(prior_lookup(requests), regime="ce")
        policy.id = f"pilot_{backend}"
        policy.variant = "masked_argmax"
        episodes = run_policy(env, policy, worlds=bundle.worlds, seed=42,
                              run_id="paired-pilot", exp_id=backend, prior_regime=backend)
        metrics = aggregate(episodes, n_resamples=50)
        result["val_strict_episode"][backend] = {
            k: metrics[k] for k in ("n_episodes", "mean_episode_reward", "token_f1", "coverage",
                "selective_risk", "avg_optional_retrievals", "avg_retrieval_cost",
                "action_count_answer", "action_count_bm25", "action_count_dense",
                "action_count_abstain", "false_answer_rate_on_unanswerables")}
        result["val_strict_episode"][backend]["qid_sha256"] = hashlib.sha256(
            "\n".join(sorted(e.qid for e in episodes)).encode()).hexdigest()
    result["fixtures"] = read_json(ROOT / "fixtures/manifest.json")
    targets[0].parent.mkdir(parents=True, exist_ok=True)
    with targets[0].open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    lines = ["# Paired pilot — GLiNER vs Laya", "", "Historical artifacts are read-only; no strict test used.", "",
             "| Backend | States: accuracy | Macro F1 | Episodes: reward | QA token F1 | Coverage | Retrieval cost |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for backend in ("laya", "gliner"):
        d, e = result["teacher_validation"][backend], result["val_strict_episode"][backend]
        lines.append(f"| {backend} | {d['accuracy']:.3f} | {d['macro_f1_present_classes']:.3f} | {e['mean_episode_reward']:.3f} | {e['token_f1']:.3f} | {e['coverage']:.3f} | {e['avg_retrieval_cost']:.3f} |")
    lines += ["", "## Limitations", *["- " + x for x in result["limitations"]],
              "", "See paired_pilot.json for confusion, per-class support and action counts."]
    with targets[1].open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "episode-requests", "report"])
    mode = parser.parse_args().mode
    {"prepare": prepare, "episode-requests": episode_requests, "report": report}[mode]()

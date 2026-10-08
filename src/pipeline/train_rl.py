"""P6 CLI: sequential REINFORCE training of the adaptive QA policy (B8 / E06).

    python -m src.pipeline.train_rl --config configs/pilot.yaml --seed 42
    python -m src.pipeline.train_rl --profile smoke --limit 20 --max-updates 2 --dry-run

What this stage is, exactly (`02_IMPLEMENTATION/05_SEQUENTIAL_REINFORCE_TRAINING.md`): the
policy is trained on episodes whose *dynamics* come from immutable caches — retrieval
rankings, the P3 reader state cache, frozen Laya probabilities — while **actions are always
sampled from the current policy**. No trajectory is ever replayed from a cache, so the
gradient is genuinely on-policy (spec 05 §8, `09_...md` §3).

Fairness (spec 05 §2, `03_EXPERIMENTS/00...` §1-2): worlds, masks, horizon, bootstrap and
retrieval budget all come from :mod:`src.pipeline.worlds`, the same builder the teacher and
the baselines use, so B8 differs from the other arms only by its policy.

Outputs, under ``outputs/models/rl_policy`` (plus plots and a report):

* ``policy.safetensors`` + ``scaler.json`` + ``rl_config.json`` — the best-by-validation policy
* ``config.json``               — seed, val reward, hashes, Laya/reader/retrieval provenance
* ``laya_checkpoint_ref.txt``   — the frozen prior checkpoint this policy was trained against
* ``trajectories.jsonl``        — bounded greedy traces of the final policy (no labels)
* ``train_log.jsonl``, ``best/``, ``last/`` — per-update log and checkpoint discipline
* ``reports/plots/reward_learning_curve.png``, ``reports/plots/action_histogram.png``
* ``reports/rl_training_report.json`` — the run summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.schema import ACTION_ORDER
from src.decision.state_text import STATE_TEXT_VERSION
from src.pipeline import build_retrieval, worlds
from src.pipeline.cache_reader import state_cache_paths
from src.pipeline.worlds import WorldBundle, build_worlds
from src.qa.state_cache import load_state_cache_meta
from src.rl.env import AdaptiveQAEnv, EpisodeWorld
from src.rl.policy import PolicyNet
from src.rl.reinforce import RLConfig, ReinforceTrainer, evaluate, rollout
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import add_profile_args, config_from_args, resolve, snapshot_config
from src.utils.io import atomic_write_json, write_jsonl
from src.utils.reproducibility import capture_library_versions, seed_everything
from src.utils.runs import RunRegistry, now_iso

STAGE = "P6_rl"
TRAIN_SPLIT, VAL_SPLIT = "train_strict", "val_strict"
DEFAULT_DEVICE = "cpu"

POLICY_DIR = "rl_policy"
CONFIG_FILE = "config.json"
PRIOR_REF_FILE = "laya_checkpoint_ref.txt"
TRAJECTORIES_FILE = "trajectories.jsonl"
TRAIN_LOG_FILE = "train_log.jsonl"
SUMMARY_FILE = "rl_training_report.json"
CURVE_FILE = "reward_learning_curve.png"
HISTOGRAM_FILE = "action_histogram.png"

#: Greedy traces kept for the package: bounded so the file stays inspectable.
TRAJECTORY_EPISODES = 32
#: Rolling window of the learning curve (updates), for the mean ± spread band.
CURVE_WINDOW = 10


def policy_dir(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg, "models") / POLICY_DIR


def make_env(cfg: Mapping[str, Any], bundle: WorldBundle) -> AdaptiveQAEnv:
    """The one environment definition from :func:`worlds.env_config`, wrapped in a scorer."""
    reward = RewardCalculator(RewardConfig.from_dict(cfg))
    return AdaptiveQAEnv(list(bundle.worlds), reward, worlds.env_config(cfg))


def split_ambiguous(train_bundle: WorldBundle) -> tuple[list[EpisodeWorld], list[str], float]:
    """Drop ambiguous negatives from the *train* split only; validation keeps them.

    These questions have no defensible gold label (the negative audit could not certify
    them), so a reward computed on them would train the policy on noise. They stay in
    validation because the val metric must reflect the real question distribution.
    """
    total = len(train_bundle.worlds)
    excluded = sorted(w.qid for w in train_bundle.worlds if w.qid in train_bundle.ambiguous_qids)
    kept = [w for w in train_bundle.worlds if w.qid not in train_bundle.ambiguous_qids]
    if not kept:
        raise ValueError(f"every {train_bundle.split} world is an ambiguous negative "
                         f"({total} worlds); nothing left to train on")
    return kept, excluded, (len(excluded) / total if total else 0.0)


def laya_reference(cfg: Mapping[str, Any], regime: str | None) -> dict[str, Any]:
    """Checkpoint path and behavioural hash of the frozen prior used by this run."""
    if regime is None:
        return {"regime": None, "checkpoint_ref": None, "model_hash": None,
                "state_text_version": STATE_TEXT_VERSION}
    checkpoint = worlds.laya_checkpoint_dir(cfg, regime)
    return {"regime": regime, "checkpoint_ref": str(checkpoint),
            "model_hash": checkpoint_model_hash(checkpoint),
            "state_text_version": STATE_TEXT_VERSION}


def combined_prior(bundles: Sequence[WorldBundle],
                   regime: str | None) -> Callable[[dict], np.ndarray] | None:
    """One prior callable over the pre-warmed bundles (train and val have disjoint qids)."""
    if regime is None:
        return None
    parts = [([world.qid for world in bundle.worlds], bundle.laya_fn)
             for bundle in bundles if bundle.laya_fn is not None]
    if not parts:
        raise RuntimeError(f"Laya regime {regime!r} was requested but no bundle pre-warmed a prior")
    return worlds.merge_laya_fns(parts)


def training_is_real(updates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Checks that the run actually trained (`05_...md` §11), not just executed.

    Gradients must be finite and nonzero after the first update and the parameters must have
    moved; NaN/Inf anywhere aborts the run. Reward direction is checked by the toy-MDP test.
    """
    if not updates:
        raise RuntimeError("no parameter update was performed; max_updates must be >= 1")
    first = updates[0]
    checks = {
        "grad_norm_finite": bool(np.isfinite(first["grad_norm"])),
        "grad_norm_nonzero": bool(abs(float(first["grad_norm"])) > 0.0),
        "params_changed": bool(float(first["param_delta"]) > 0.0),
        "train_reward_finite": bool(np.isfinite(first["train_mean_reward"])),
    }
    nonfinite = [int(u["update"]) for u in updates
                 if not (np.isfinite(u["grad_norm"]) and np.isfinite(u["param_delta"])
                         and np.isfinite(u["train_mean_reward"]))]
    checks["no_nonfinite_updates"] = not nonfinite
    checks["n_updates"] = len(updates)
    failed = sorted(k for k, ok in checks.items() if ok is False)
    if failed:
        raise RuntimeError(
            f"training did not happen as specified: {failed} failed after "
            f"{len(updates)} update(s) (first row: {dict(first)}); refusing to publish a "
            "model that was never trained")
    return checks


def write_trajectories(worlds_to_trace: Sequence[EpisodeWorld], env: AdaptiveQAEnv,
                       net: PolicyNet, extractor: Any, scaler_t: Any,
                       laya_fn: Callable[[dict], np.ndarray] | None, *, device: Any,
                       path: Path, limit: int = TRAJECTORY_EPISODES) -> int:
    """Bounded greedy traces of the final policy. Decisions and outcomes only — no labels."""
    rows: list[dict[str, Any]] = []
    for index, world in enumerate(sorted(worlds_to_trace, key=lambda w: w.qid)[:limit]):
        episode = rollout(env, net, extractor, scaler_t, laya_fn, seed=20_000 + index,
                          qid=world.qid, greedy=True, device=device)
        for t, transition in enumerate(episode.transitions):
            rows.append({
                "qid": episode.qid, "scenario_id": world.scenario_id, "t": t,
                "action": transition.action, "action_name": ACTION_ORDER[transition.action],
                "reward": transition.reward, "log_prob": transition.log_prob,
                "value": transition.value, "entropy": transition.entropy,
                "n_valid_actions": int(np.asarray(transition.mask, dtype=bool).sum()),
                "episode_return": episode.total_reward, "policy": "greedy_final",
            })
    write_jsonl(path, rows)
    return len(rows)


def _rolling(values: Sequence[float], window: int) -> tuple[np.ndarray, np.ndarray]:
    out_mean = np.zeros(len(values), dtype=np.float64)
    out_std = np.zeros(len(values), dtype=np.float64)
    for i in range(len(values)):
        chunk = np.asarray(values[max(0, i - window + 1):i + 1], dtype=np.float64)
        out_mean[i] = chunk.mean()
        out_std[i] = chunk.std()
    return out_mean, out_std


def write_plots(cfg: Mapping[str, Any], updates: Sequence[Mapping[str, Any]],
                final_val: Mapping[str, Any], *, best_update: int | None,
                window: int = CURVE_WINDOW) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = resolve(cfg, "plots")
    plots.mkdir(parents=True, exist_ok=True)
    curve_path = plots / CURVE_FILE
    histogram_path = plots / HISTOGRAM_FILE

    xs = [int(u["update"]) for u in updates]
    ys = [float(u["train_mean_reward"]) for u in updates]
    means, stds = _rolling(ys, window)
    val_x = [int(u["update"]) for u in updates if "val" in u]
    val_y = [float(u["val"]["mean_reward"]) for u in updates if "val" in u]
    val_err = [float(u["val"]["std_reward"]) for u in updates if "val" in u]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, ys, color="tab:blue", alpha=0.30, linewidth=1, label="train reward / update")
    ax.plot(xs, means, color="tab:blue", label=f"train reward, rolling mean (w={window})")
    ax.fill_between(xs, means - stds, means + stds, color="tab:blue", alpha=0.15,
                    label="train ±1 std (window)")
    if val_x:
        ax.errorbar(val_x, val_y, yerr=val_err, color="tab:orange", marker="o",
                    linestyle="none", capsize=3, label="validation (greedy)")
    if best_update:
        ax.axvline(best_update, color="black", linestyle=":", linewidth=1,
                   label=f"best update {best_update}")
    ax.set_xlabel("update")
    ax.set_ylabel("episode return")
    ax.set_title("P6 REINFORCE — reward learning curve")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    counts = [int(final_val["action_counts"][str(i)]) for i in range(len(ACTION_ORDER))]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(ACTION_ORDER), counts, color="tab:green")
    ax.set_ylabel("actions taken")
    ax.set_title("P6 greedy policy — action histogram")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(histogram_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"curve": curve_path, "histogram": histogram_path}


def run(*, cfg: Mapping[str, Any], profile: str, seed: int | None = None,
        limit: int | None = None, max_updates: int | None = None,
        laya_regime: str | None = None, dry_run: bool = False, device: str | None = None,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    """Train the B8 policy. ``laya_regime=None`` means "use cfg['laya']['frozen']"."""
    rl_config = RLConfig.from_dict(dict(cfg))
    if rl_config.algorithm != "reinforce":
        raise ValueError(f"rl.algorithm={rl_config.algorithm!r} is not implemented; "
                         "P6 trains 'reinforce' (spec 05 §4)")
    if seed is not None:
        rl_config.seed = int(seed)
    if max_updates is not None:
        rl_config.max_updates = int(max_updates)
    if rl_config.max_updates < 1 or rl_config.eval_every < 1:
        raise ValueError(f"rl.max_updates={rl_config.max_updates} and "
                         f"rl.eval_every={rl_config.eval_every} must both be >= 1")
    regime = worlds.resolve_laya_regime(cfg, worlds.DEFAULT_LAYA_SENTINEL
                                        if laya_regime is None else laya_regime)
    device = str(device or DEFAULT_DEVICE)
    out_dir = policy_dir(cfg)

    run_id = f"p6-rl-{profile}-{rl_config.seed}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile,
                    seed=rl_config.seed, limit=limit, max_updates=rl_config.max_updates,
                    laya_regime=regime, device=device, dry_run=dry_run)
    try:
        seed_everything(rl_config.seed)
        # 1. Worlds from the frozen caches; the Laya prior is pre-warmed inside build_worlds.
        train_bundle = build_worlds(cfg, TRAIN_SPLIT, limit=limit, laya_regime=regime)
        val_bundle = build_worlds(cfg, VAL_SPLIT, limit=limit, laya_regime=regime)
        train_worlds, excluded, excluded_rate = split_ambiguous(train_bundle)
        print(f"{TRAIN_SPLIT}: {len(train_worlds)} worlds to train on, "
              f"{len(excluded)} ambiguous negative(s) excluded "
              f"({excluded_rate:.1%} of {len(train_bundle.worlds)})")
        train_env = make_env(cfg, _replaced(train_bundle, train_worlds))
        val_env = make_env(cfg, val_bundle)
        laya_fn = combined_prior([train_bundle, val_bundle], regime)
        prior = laya_reference(cfg, regime)

        trainer = ReinforceTrainer(train_env, val_env, rl_config, laya_fn=laya_fn,
                                   device=device, run_dir=out_dir, seed=rl_config.seed)
        # 2. Scaler on TRAIN rollouts only; the validation split must not inform features.
        scaler = trainer.fit_scaler()
        net = trainer.build()

        if dry_run:
            probe = {name: evaluate(env, net, trainer.extractor, trainer.scaler_t,
                                    laya_fn, episodes=rl_config.eval_episodes,
                                    device=trainer.device)
                     for name, env in (("train", train_env), ("val", val_env))}
            summary = {
                "run_id": run_id, "stage": STAGE, "dry_run": True, "profile": profile,
                "seed": rl_config.seed, "device": device, "laya": prior,
                "splits": {"train": {"worlds": len(train_worlds),
                                     "ambiguous_total": len(train_bundle.ambiguous_qids)},
                           "val": {"worlds": len(val_bundle.worlds)}},
                "excluded_ambiguous": {"n": len(excluded), "rate": excluded_rate},
                "scaler_features": len(scaler.mean), "probe": probe,
            }
            registry.append(run_id=run_id, stage=STAGE, status="SUCCEEDED", profile=profile,
                            dry_run=True, probe=probe, note="no model or report written")
            return summary

        # 3. Train (on-policy rollouts, cached dynamics) on the train split, early-stop on val.
        result = trainer.train()
        checks = training_is_real(result["updates"])
        if result["best_update"] is None:
            raise RuntimeError(
                f"no validation evaluation ran in {rl_config.max_updates} update(s) "
                f"(rl.eval_every={rl_config.eval_every}); set eval_every <= max_updates so a "
                "checkpoint can be selected")
        # ``result["updates"]`` records every update that ran, including the one that
        # triggered early stopping.
        updates = result["updates"]
        # ``best/`` holds exactly the weights saved at the best validation update
        # (ReinforceTrainer writes it then and never overwrites it). Load it and measure
        # precisely what gets published, so val_reward describes the shipped artifact.
        trainer.load_best(out_dir / "best")
        net = trainer.net
        assert net is not None
        final_val = evaluate(val_env, net, trainer.extractor, trainer.scaler_t,
                             laya_fn, episodes=rl_config.eval_episodes,
                             device=trainer.device)
        published_update = len(updates)
        trainer.save(out_dir)  # canonical policy.safetensors + scaler.json + rl_config.json

        # 4. Provenance, traces, plots, report.
        reader = {split: _reader_meta(cfg, split) for split in (TRAIN_SPLIT, VAL_SPLIT)}
        config = {
            "stage": STAGE, "run_id": run_id, "profile": profile,
            "algorithm": rl_config.algorithm, "seed": rl_config.seed, "device": device,
            "best_update": int(result["best_update"]), "updates_ran": published_update,
            "stopped_early": bool(result["stopped_early"]),
            "val_reward": final_val["mean_reward"], "val_std_reward": final_val["std_reward"],
            "val_action_counts": final_val["action_counts"],
            "laya": prior, "reader": reader,
            "retrieval_config_sha": build_retrieval.retrieval_config_sha(dict(cfg)),
            "reward_spec_hash": RewardConfig.from_dict(dict(cfg)).spec_hash(),
            "worlds": {"train": len(train_worlds), "val": len(val_bundle.worlds),
                       "excluded_ambiguous": len(excluded), "excluded_rate": excluded_rate},
            "rl_config": asdict(rl_config), "env_config": asdict(worlds.env_config(cfg)),
            "created": now_iso(),
        }
        config_path = out_dir / CONFIG_FILE
        atomic_write_json(config_path, config)
        prior_path = out_dir / PRIOR_REF_FILE
        prior_path.write_text((prior["checkpoint_ref"] or "none") + "\n", encoding="utf-8")
        n_trace = write_trajectories(train_worlds, train_env, net, trainer.extractor,
                                     trainer.scaler_t, laya_fn, device=trainer.device,
                                     path=out_dir / TRAJECTORIES_FILE)
        plots = write_plots(cfg, updates, final_val,
                            best_update=int(result["best_update"]))

        artifacts = [out_dir / "policy.safetensors", out_dir / "scaler.json",
                     out_dir / "rl_config.json", config_path, prior_path,
                     out_dir / TRAJECTORIES_FILE, out_dir / TRAIN_LOG_FILE,
                     plots["curve"], plots["histogram"]]
        summary = {
            "run_id": run_id, "stage": STAGE, "profile": profile, "seed": rl_config.seed,
            "device": device, "algorithm": rl_config.algorithm,
            "splits": {"train": {"worlds": len(train_worlds),
                                 "ambiguous_total": len(train_bundle.ambiguous_qids)},
                       "val": {"worlds": len(val_bundle.worlds),
                               "ambiguous_kept": len(val_bundle.ambiguous_qids)}},
            "excluded_ambiguous": {"n": len(excluded), "rate": excluded_rate,
                                   "qids": excluded[:100]},
            "updates": published_update, "stopped_early": bool(result["stopped_early"]),
            "best_update": int(result["best_update"]),
            "best_val_reward": result["best_val_reward"], "final_val": final_val,
            "checkpoint_note": ("policy.safetensors is the trainer's best/ artifact — the "
                                "weights saved at validation update best_update, never "
                                "overwritten — and val_reward measures exactly them"),
            "training_is_real": checks, "trajectories": n_trace,
            "laya": prior, "reader": reader,
            "retrieval_config_sha": config["retrieval_config_sha"],
            "reward_spec_hash": config["reward_spec_hash"],
            "on_policy": {"actions": "sampled from the current policy at every step",
                          "dynamics": "cached rankings, P3 reader cache, pre-warmed Laya",
                          "cached_trajectories": False},
            "rl_config": asdict(rl_config), "env_config": config["env_config"],
            "artifacts": [str(path) for path in artifacts],
            "summary_path": str(resolve(cfg, "reports") / SUMMARY_FILE),
            "library_versions": capture_library_versions(),
        }
        atomic_write_json(resolve(cfg, "reports") / SUMMARY_FILE, summary)
        registry.record_artifact(
            run_id, STAGE, artifacts, profile=profile, seed=rl_config.seed,
            retrieval_config_sha=config["retrieval_config_sha"],
            best_update=config["best_update"], updates=summary["updates"],
            val_reward=final_val["mean_reward"], laya_model_hash=prior["model_hash"],
            reader_model_hash=reader[TRAIN_SPLIT]["reader_model_hash"])
        return summary
    except Exception as exc:
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise


def _replaced(bundle: WorldBundle, keep: Sequence[EpisodeWorld]) -> WorldBundle:
    """Same bundle with the ambiguous negatives removed from the world list."""
    return WorldBundle(split=bundle.split, worlds=list(keep), questions=bundle.questions,
                       docs=bundle.docs, laya_fn=bundle.laya_fn,
                       ambiguous_qids=bundle.ambiguous_qids)


def _reader_meta(cfg: Mapping[str, Any], split: str) -> dict[str, Any]:
    parquet, meta_path = state_cache_paths(cfg, split)
    meta = load_state_cache_meta(meta_path)
    return {"parquet": str(parquet), "meta": str(meta_path),
            "reader_model_hash": meta["reader_model_hash"],
            "tokenizer_sha": meta["tokenizer_sha"],
            "inference_config_sha": meta["inference_config_sha"],
            "retrieval_budget": meta.get("retrieval_budget")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.train_rl",
        description="P6: sequential REINFORCE training of the adaptive QA policy (B8)")
    add_profile_args(parser)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per split (must match the P3 cache build)")
    parser.add_argument("--max-updates", type=int, default=None,
                        help="override rl.max_updates")
    parser.add_argument("--laya-regime", choices=["ce", "rlcd", "none"], default=None,
                        help="frozen Laya prior (default: cfg['laya']['frozen'])")
    parser.add_argument("--dry-run", action="store_true",
                        help="build worlds, fit the scaler and probe both splits; write no model")
    parser.add_argument("--device", default=None, help="torch device (default: cpu)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    snapshot_config(cfg, resolve(cfg, "manifests") / f"P6_rl_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, seed=args.seed, limit=args.limit,
                  max_updates=args.max_updates, laya_regime=args.laya_regime,
                  dry_run=args.dry_run, device=args.device)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if summary.get("dry_run"):
        print(f"run_id={summary['run_id']} (dry run) "
              f"train={summary['splits']['train']['worlds']} worlds, "
              f"val={summary['splits']['val']['worlds']} worlds, "
              f"scaler_features={summary['scaler_features']}, "
              f"probe_val_reward={summary['probe']['val']['mean_reward']:.4f}")
        return 0
    print(f"run_id={summary['run_id']} updates={summary['updates']} "
          f"best_update={summary['best_update']} "
          f"val_reward={summary['final_val']['mean_reward']:.4f}")
    print(f"  policy: {resolve(cfg, 'models') / POLICY_DIR}")
    print(f"  report: {summary['summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

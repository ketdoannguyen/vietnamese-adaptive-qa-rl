"""P7 CLI: the baseline/ablation matrix on one shared set of worlds.

    python -m src.pipeline.eval_baselines --profile pilot --split val_strict --seed 42

Every row of `03_EXPERIMENTS/00_BASELINES_MATRIX_AND_BUDGET.md` §3 comes from the *same*
``WorldBundle`` (built once per run), the same environment config (mask, horizon, forced
bootstrap, optional budget) and the same reward calculator, and every row's episodes are
stepped online by :mod:`src.eval.rollout` — no policy ever replays a fixed action log and
none of them can see gold.

=============================================  ==========================================
row                                            policy (spec 00 §1)
=============================================  ==========================================
E00  rule bootstrap                            B1  bootstrap-then-answer/abstain
E01  fixed BM25                                B2
E02  fixed Dense                               B3
E03  threshold (calibrated p_correct)          B5
E04  Laya CE                                   B6  masked argmax, CE prior
E05  Laya RLCD                                 B7  masked argmax, RLCD prior
E06  Laya RLCD + REINFORCE                     B8  trained checkpoint, RLCD prior
E07  Laya CE + REINFORCE (optional arm)        B8  only if the checkpoint's own regime is CE
B0   random-valid sanity (``RANDOM_POLICY_SEEDS`` seeds, base seed + 0/1/2)
B4   search-all upper-compute (both tools, never above ``rl.max_optional_search``)
=============================================  ==========================================

Rows outside the E-matrix exist because spec 00 §1 defines B0 and B4 as policies while §3
does not give them an E code; they are labelled by policy code so no E number is invented.

Refusal and discipline
    ``--split test_strict`` is refused unless ``outputs/manifests/best_selection.json``
    exists (spec 00 §5: the test comparison happens once, after the freeze). ``--dev``
    bypasses the guard but stamps **every** produced row (CSV, parquet, registry) as dev.
    A missing checkpoint never fabricates a row: unavailable arms are written as SKIPPED
    with the reason, with ``N/A`` in every metric column.

B5's ``tau`` is ``--tau`` when given, else the estimator's own val-selected ``tau_answer``
(``threshold_selection.json``), else ``policies.DEFAULT_TAU`` — and the fallback is recorded
as a limitation in the report. Missing/NaN/out-of-range ``p_correct`` raises (spec 08 §6).

Artifacts: ``reports/tables/baseline_results_{val,test}.csv`` (all rows, FAILED/SKIPPED
included), ``outputs/predictions/{run_id}_{split}.parquet``, ``outputs/runs/trial_registry.jsonl``
(one entry per row), the three P7 plots, ``reports/baseline_report_{short}.json``, and
``outputs/runs/index.jsonl`` entries RUNNING -> SUCCEEDED/FAILED. The RL learning curve and
action histogram belong to the P6 stage and are not written here.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.eval import reporting, splitting
from src.eval.correctness import CorrectnessEstimator
from src.eval.policies import (
    DEFAULT_TAU, RANDOM_POLICY_SEEDS, BootstrapThenAnswerPolicy, ConfidenceThresholdPolicy,
    FixedToolPolicy, LayaArgmaxPolicy, RandomValidPolicy, SearchAllPolicy, load_rl_policy,
    resolve_policy_id, rl_checkpoint_status, short_code,
)
from src.eval.rollout import EpisodeRecord, episode_row, run_policy
from src.pipeline import build_retrieval, worlds
from src.pipeline.worlds import build_worlds
from src.rl.env import AdaptiveQAEnv
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import add_profile_args, config_from_args, resolve, snapshot_config
from src.utils.hashing import sha256_json
from src.utils.io import atomic_write_json, read_json, write_parquet
from src.utils.runs import RunRegistry, now_iso

STAGE = "P7_baselines"
DEFAULT_SPLIT = "val_strict"
SPLIT_SHORT = {"val_strict": "val", "test_strict": "test"}
#: Written by the P6 selection step; its presence is what unlocks the test split.
SELECTION_FREEZE = "best_selection.json"
#: The B5 estimator directory written by :mod:`src.pipeline.train_correctness`.
CORRECTNESS_DIR = "correctness_estimator"
CORRECTNESS_MODEL_FILE = "model.json"
THRESHOLD_FILE = "threshold_selection.json"
TRIAL_REGISTRY_FILE = "trial_registry.jsonl"


class TestSplitNotFrozen(RuntimeError):
    """``test_strict`` was requested before the model selection was frozen."""


@dataclass(frozen=True)
class RowSpec:
    """One row of the results matrix (``seeds`` > 1 expands to that many rows)."""

    exp_id: str
    policy: str
    prior: str | None = None
    seeds: int = 1
    note: str = ""


ROW_SPECS: tuple[RowSpec, ...] = (
    RowSpec("E00", "B1", note="rule bootstrap, zero adaptation"),
    RowSpec("E01", "B2", note="fixed BM25 to the optional budget"),
    RowSpec("E02", "B3", note="fixed Dense to the optional budget"),
    RowSpec("E03", "B5", note="calibrated p_correct threshold"),
    RowSpec("E04", "B6", prior="ce", note="Laya CE masked argmax"),
    RowSpec("E05", "B7", prior="rlcd", note="Laya RLCD masked argmax"),
    RowSpec("E06", "B8", prior="rlcd", note="REINFORCE trained on the RLCD prior"),
    RowSpec("E07", "B8", prior="ce", note="REINFORCE trained on the CE prior (optional)"),
    RowSpec("B0", "B0", seeds=RANDOM_POLICY_SEEDS, note="random-valid sanity, uniform legal"),
    RowSpec("B4", "B4", note="search-all upper-compute, both tools"),
)


# --------------------------------------------------------------------------- preconditions

def selection_freeze_path(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg, "manifests") / SELECTION_FREEZE


def require_frozen_selection(cfg: Mapping[str, Any], *, split: str, dev: bool = False
                             ) -> Path | None:
    """Refuse ``test_strict`` until the selection freeze exists (spec 00 §5)."""
    if split != "test_strict" or dev:
        return None
    path = selection_freeze_path(cfg)
    if not path.is_file():
        raise TestSplitNotFrozen(
            f"{split} is guarded: {path} does not exist, so no model selection has been "
            "frozen and a test comparison would be tuning-to-test. Run the val stage, train "
            "and select the checkpoint, write the freeze — or use --dev to bypass this "
            "guard (every row is then stamped dev).")
    return path


def select_specs(policies: Sequence[str] | None) -> list[RowSpec]:
    """Rows for the requested policy ids (``None`` = the whole matrix)."""
    if not policies:
        return list(ROW_SPECS)
    wanted = {resolve_policy_id(token) for token in policies}
    specs = [spec for spec in ROW_SPECS if resolve_policy_id(spec.policy) in wanted]
    if not specs:
        raise ValueError(f"no result row uses {sorted(policies)}; known policies: "
                         f"{sorted({spec.policy for spec in ROW_SPECS})}")
    return specs


def resolve_tau(cfg: Mapping[str, Any], requested: float | None) -> tuple[float, str]:
    """``(tau, source)`` for B5: the flag, else the estimator's val-selected value."""
    if requested is not None:
        return float(requested), "--tau"
    path = resolve(cfg, "models") / CORRECTNESS_DIR / THRESHOLD_FILE
    if path.is_file():
        selected = read_json(path).get("tau_answer")
        if selected is not None:
            return float(selected), str(path)
    return DEFAULT_TAU, "default"


def skip_reason(spec: RowSpec, cfg: Mapping[str, Any],
                rl_status: Mapping[str, Any]) -> str | None:
    """Why this row cannot run (SKIPPED), or ``None`` when its inputs all exist."""
    if spec.prior is not None:
        checkpoint = worlds.laya_checkpoint_dir(cfg, spec.prior)
        if not checkpoint.is_dir():
            return (f"frozen Laya prior {spec.prior!r} is missing at {checkpoint}; train the "
                    "P5 stage for that regime")
    if spec.policy == "B5":
        model = resolve(cfg, "models") / CORRECTNESS_DIR / CORRECTNESS_MODEL_FILE
        if not model.is_file():
            return (f"no B5 correctness estimator at {model.parent}; train it with "
                    "src.pipeline.train_correctness")
    if spec.policy == "B8":
        if rl_status["error"]:
            return (f"the B8 checkpoint at {rl_status['path']} is unreadable "
                    f"({rl_status['error']})")
        if not rl_status["exists"]:
            return (f"no B8 checkpoint at {rl_status['path']} "
                    f"(missing {rl_status['missing']}); train it with src.pipeline.train_rl")
        if rl_status["regime"] != spec.prior:
            return (f"the B8 checkpoint was trained with laya.regime={rl_status['regime']!r} "
                    f"while {spec.exp_id} needs {spec.prior!r}; one checkpoint is one arm, so "
                    "this row is SKIPPED rather than fabricated")
    return None


def regime_prior_fn(cfg: Mapping[str, Any], regime: str, world_list: Sequence[Any],
                    reward: RewardCalculator, config: Any) -> Callable[[dict], Any]:
    """Pre-warm one regime's frozen prior over every reachable state, then look up in memory.

    Identical to the prior ``build_worlds`` would attach for that regime; worlds are built
    without a prior so rows that do not need one cannot fail on a missing checkpoint.
    """
    checkpoint = worlds.laya_checkpoint_dir(cfg, regime)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Laya checkpoint for regime {regime!r} is missing: {checkpoint}")
    return worlds.prewarm_prior(cfg, checkpoint,
                                worlds.reachable_state_requests(world_list, reward, config))


def build_policy(spec: RowSpec, *, cfg: Mapping[str, Any], tau: float,
                 prior_fn: Callable[[str], Callable[[dict], Any]]) -> Any:
    code = short_code(resolve_policy_id(spec.policy))
    if code == "B0":
        return RandomValidPolicy()
    if code == "B1":
        return BootstrapThenAnswerPolicy()
    if code == "B2":
        return FixedToolPolicy(tool="bm25")
    if code == "B3":
        return FixedToolPolicy(tool="dense")
    if code == "B4":
        return SearchAllPolicy()
    if code == "B5":
        estimator = CorrectnessEstimator.load(resolve(cfg, "models") / CORRECTNESS_DIR)
        return ConfidenceThresholdPolicy(estimator, tau=tau)
    if code == "B6":
        return LayaArgmaxPolicy(prior_fn("ce"), regime="ce")
    if code == "B7":
        return LayaArgmaxPolicy(prior_fn("rlcd"), regime="rlcd")
    if code == "B8":
        assert spec.prior is not None
        return load_rl_policy(cfg, laya_fn=prior_fn(spec.prior), prior_regime=spec.prior)
    raise ValueError(f"no builder for policy {spec.policy!r}")


def title_groups_or_none(cfg: Mapping[str, Any], split: str
                         ) -> tuple[dict[str, str] | None, str | None]:
    """Title groups for the cluster CIs; a recorded limitation instead of a hard failure."""
    try:
        return splitting.title_groups(cfg, split), None
    except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
        return None, (f"title groups unavailable for {split} ({type(exc).__name__}: {exc}); "
                      "confidence intervals are N/A")


# ------------------------------------------------------------------------------------ run

def run(*, cfg: Mapping[str, Any], profile: str, split: str = DEFAULT_SPLIT, seed: int = 42,
        policies: Sequence[str] | None = None, limit: int | None = None, dry_run: bool = False,
        dev: bool = False, tau: float | None = None, registry: RunRegistry | None = None,
        trials: RunRegistry | None = None, n_resamples: int = 1000) -> dict[str, Any]:
    """Evaluate the requested rows. ``split`` is one of :data:`SPLIT_SHORT`."""
    if split not in SPLIT_SHORT:
        raise ValueError(f"unknown split {split!r}; known: {sorted(SPLIT_SHORT)}")
    short = SPLIT_SHORT[split]
    freeze = require_frozen_selection(cfg, split=split, dev=dev)
    specs = select_specs(policies)
    reward = RewardCalculator(RewardConfig.from_dict(dict(cfg)))
    config = worlds.env_config(cfg)
    tau_value, tau_source = resolve_tau(cfg, tau)

    run_id = f"p7-baselines-{profile}-{short}-{int(seed)}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    trials = trials or RunRegistry(resolve(cfg, "runs") / TRIAL_REGISTRY_FILE)
    registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile, split=split,
                    dev=bool(dev), seed=int(seed), limit=limit, dry_run=bool(dry_run),
                    policies=[spec.exp_id for spec in specs],
                    selection_freeze=str(freeze) if freeze else None)
    try:
        bundle = build_worlds(cfg, split, limit=limit, laya_regime=None)
        rl_status = rl_checkpoint_status(cfg)
        if dry_run:
            planned = []
            for spec in specs:
                reason = skip_reason(spec, cfg, rl_status)
                planned.append({"exp_id": spec.exp_id, "policy": spec.policy,
                                "prior": spec.prior, "seeds": spec.seeds,
                                "status": reporting.STATUS_SKIPPED if reason else "RUN",
                                "reason": reason or spec.note})
            summary = {"run_id": run_id, "stage": STAGE, "dry_run": True, "profile": profile,
                       "split": split, "dev": bool(dev), "seed": int(seed), "limit": limit,
                       "n_worlds": len(bundle.worlds),
                       "n_ambiguous_kept": len(bundle.ambiguous_qids),
                       "tau": tau_value, "tau_source": tau_source, "rows": planned}
            registry.append(run_id=run_id, stage=STAGE, status="SUCCEEDED", profile=profile,
                            split=split, dev=bool(dev), dry_run=True,
                            note="dry run: no artifact written")
            return summary

        groups, groups_limitation = title_groups_or_none(cfg, split)
        env = AdaptiveQAEnv(bundle.worlds, reward, config)
        priors: dict[str, Callable[[dict], Any]] = {}

        def prior_fn(regime: str) -> Callable[[dict], Any]:
            if regime not in priors:
                priors[regime] = regime_prior_fn(cfg, regime, bundle.worlds, reward, config)
            return priors[regime]

        rows: list[dict[str, Any]] = []
        episodes_by_row: dict[tuple[str, int], list[EpisodeRecord]] = {}
        parquet_rows: list[dict[str, Any]] = []
        for spec in specs:
            for offset in range(spec.seeds):
                row_seed = int(seed) + offset
                identity = {
                    "run_id": run_id, "exp_id": spec.exp_id,
                    "policy": resolve_policy_id(spec.policy), "policy_variant": "",
                    "prior_regime": spec.prior or "none", "split": split, "dev": bool(dev),
                    "seed": row_seed, "tau": tau_value if spec.policy == "B5" else None,
                }
                reason = skip_reason(spec, cfg, rl_status)
                if reason is not None:
                    row = {**identity, **reporting.empty_metrics(),
                           "status": reporting.STATUS_SKIPPED, "reason": reason}
                else:
                    try:
                        policy = build_policy(spec, cfg=cfg, tau=tau_value, prior_fn=prior_fn)
                        records = run_policy(env, policy, worlds=bundle.worlds, seed=row_seed,
                                             run_id=run_id, exp_id=spec.exp_id, dev=dev,
                                             prior_regime=spec.prior)
                        metrics = reporting.aggregate(
                            records, groups=groups, ambiguous_qids=bundle.ambiguous_qids,
                            calibrated=bool(getattr(policy, "calibrated", False)),
                            ci_seed=row_seed, n_resamples=n_resamples)
                        # A B5-RAW estimator is a documented fallback, not a calibrated score.
                        note = ""
                        if isinstance(policy, ConfidenceThresholdPolicy) and not policy.calibrated:
                            note = ("B5-RAW: uncalibrated reader margin used as the score; "
                                    "calibration metrics are N/A (spec 08 §5)")
                        row = {**identity, **metrics, "status": reporting.STATUS_SUCCEEDED,
                               "reason": note}
                        row["policy_variant"] = policy.variant
                        episodes_by_row[(spec.exp_id, row_seed)] = records
                        parquet_rows.extend(episode_row(record) for record in records)
                    except Exception as exc:  # one broken row must not lose the others
                        row = {**identity, **reporting.empty_metrics(),
                               "status": reporting.STATUS_FAILED,
                               "reason": f"{type(exc).__name__}: {exc}"}
                rows.append(row)
                trials.append(run_id=f"{run_id}:{spec.exp_id}:{row_seed}", stage=STAGE,
                              status=row["status"], exp_id=spec.exp_id, policy=spec.policy,
                              prior_regime=spec.prior, seed=row_seed, split=split, dev=bool(dev),
                              reason=row["reason"], note=spec.note)

        csv_path = reporting.write_results_csv(
            resolve(cfg, "tables") / f"baseline_results_{short}.csv", rows)
        parquet_path = parquet_meta = None
        if parquet_rows:
            parquet_path = resolve(cfg, "predictions") / f"{run_id}_{split}.parquet"
            parquet_meta = write_parquet(parquet_path, parquet_rows)
        recall = reporting.retrieval_recall(bundle.worlds)
        plots = reporting.write_plots(cfg, rows, episodes_by_row, split=split, recall=recall)

        limitations = []
        if groups_limitation:
            limitations.append(groups_limitation)
        if tau_source == "default" and any(spec.policy == "B5" for spec in specs):
            limitations.append(f"B5 used the fallback tau={DEFAULT_TAU} (no --tau and no "
                               f"{THRESHOLD_FILE}); spec 08 §5 requires a val-selected tau")
        if dev:
            limitations.append("dev-stamped run (test guard bypassed or --dev given): not a "
                               "final comparison")
        skipped = [row for row in rows if row["status"] != reporting.STATUS_SUCCEEDED]
        summary = {
            "run_id": run_id, "stage": STAGE, "profile": profile, "split": split,
            "dev": bool(dev), "seed": int(seed), "limit": limit,
            "n_worlds": len(bundle.worlds), "n_ambiguous_kept": len(bundle.ambiguous_qids),
            "selection_freeze": str(freeze) if freeze else None,
            "fairness": {
                "worlds": len(bundle.worlds),
                "qid_digest": sha256_json(sorted(world.qid for world in bundle.worlds)),
                "env_config": asdict(config),
                "reward": reward.config.to_dict(),
                "reward_spec_hash": reward.config.spec_hash(),
                "retrieval_config_sha": build_retrieval.retrieval_config_sha(dict(cfg)),
                "bootstrap": config.bootstrap_tool,
                "reader": "P3 state cache, one per strict split",
            },
            "tau": tau_value, "tau_source": tau_source,
            "rows": [{"exp_id": row["exp_id"], "policy": row["policy"], "seed": row["seed"],
                      "status": row["status"], "reason": row["reason"],
                      "mean_episode_reward": row["mean_episode_reward"],
                      "token_f1": row["token_f1"], "coverage": row["coverage"]}
                     for row in rows],
            "n_succeeded": len(rows) - len(skipped), "n_not_succeeded": len(skipped),
            "retrieval_recall": recall,
            "limitations": limitations,
            "results_csv": str(csv_path),
            "predictions": {"path": str(parquet_path), **(parquet_meta or {})},
            "plots": {name: str(path) for name, path in plots.items()},
        }
        report_path = resolve(cfg, "reports") / f"baseline_report_{short}.json"
        atomic_write_json(report_path, summary)
        artifacts = [csv_path, *([parquet_path] if parquet_path else []), *plots.values(),
                     report_path]
        registry.record_artifact(
            run_id, STAGE, artifacts, profile=profile, split=split, dev=bool(dev),
            seed=int(seed), limit=limit, n_rows=len(rows), n_succeeded=summary["n_succeeded"],
            n_not_succeeded=summary["n_not_succeeded"],
            selection_freeze=summary["selection_freeze"],
            retrieval_config_sha=summary["fairness"]["retrieval_config_sha"],
            reward_spec_hash=summary["fairness"]["reward_spec_hash"])
        return summary
    except Exception as exc:
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        split=split, dev=bool(dev),
                        error=f"{type(exc).__name__}: {exc}")
        raise


# ----------------------------------------------------------------------------------- main

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.eval_baselines",
        description="P7: the baseline/ablation matrix (B0-B8) on shared worlds")
    add_profile_args(parser)
    parser.add_argument("--split", choices=sorted(SPLIT_SHORT), default=DEFAULT_SPLIT,
                        help="evaluation split (test_strict needs a frozen selection)")
    parser.add_argument("--policy", default=None,
                        help="comma-separated policy ids (B0..B8 or full names); "
                             "default: every row of the matrix")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios (must match the P3 cache build)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the worlds and report which rows can run; write no artifact")
    parser.add_argument("--dev", action="store_true",
                        help="stamp every row as dev and allow test_strict before the freeze")
    parser.add_argument("--tau", type=float, default=None,
                        help="B5 answer threshold (default: the estimator's val-selected tau)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    policies = ([token.strip() for token in args.policy.split(",") if token.strip()]
                if args.policy else None)
    snapshot_config(cfg, resolve(cfg, "manifests") / f"P7_baselines_{profile}_config.yaml")
    try:
        summary = run(cfg=cfg, profile=profile, split=args.split, seed=int(cfg["seed"]),
                      policies=policies, limit=args.limit, dry_run=args.dry_run, dev=args.dev,
                      tau=args.tau)
    except (TestSplitNotFrozen, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if summary.get("dry_run"):
        print(f"run_id={summary['run_id']} (dry run) split={summary['split']} "
              f"worlds={summary['n_worlds']} dev={summary['dev']}")
        for row in summary["rows"]:
            print(f"  {row['exp_id']:>3} {row['policy']:<4} {row['status']:<9} {row['reason']}")
        return 0
    print(f"run_id={summary['run_id']} split={summary['split']} worlds={summary['n_worlds']} "
          f"succeeded={summary['n_succeeded']} not_succeeded={summary['n_not_succeeded']}")
    for row in summary["rows"]:
        if row["status"] != reporting.STATUS_SUCCEEDED:
            print(f"  {row['exp_id']:>3} {row['policy']:<4} {row['status']:<9} {row['reason']}")
    report_name = f"baseline_report_{SPLIT_SHORT[summary['split']]}.json"
    print(f"  table: {summary['results_csv']}")
    print(f"  report: {resolve(cfg, 'reports') / report_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

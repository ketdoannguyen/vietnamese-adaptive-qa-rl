"""P6 selection step: freeze the best-by-validation B8 checkpoint before any test access.

    python -m src.pipeline.select_rl --config configs/medium.yaml

Spec 02 §7 / 03 §5: one config is selected on validation (primary metric: **mean episodic
reward**) and its checkpoint hash is persisted in ``outputs/manifests/best_selection.json``
BEFORE the strict test split may be opened — :mod:`src.pipeline.eval_baselines` refuses
``--split test_strict`` while that file does not exist, and :mod:`src.pipeline.report` proves
the freeze-before-test ordering in ``docs/TEST_ACCESS_LOG.md``.

Multi-seed runs (seeds 42/43/44, spec 04 §5) each write the canonical
``outputs/models/rl_policy`` and an immutable archive under ``outputs/models/rl_runs/{run_id}/``
(:mod:`src.pipeline.train_rl`). This step scans the archives, applies the pre-registered
rule below, installs the winner into the canonical dir (byte-identical to its archive), and
writes the freeze LAST — a failure at any point leaves no new freeze.

The rule is fixed in code before any selection run (D016): primary = highest ``val_reward``;
tie-break = lowest seed. Eligibility requires the candidate to belong to the *current* chain:
its registry row is ``SUCCEEDED`` for this profile, its ``laya.regime`` equals
``laya.frozen``, and its retrieval / reader / reward / eval-episode fingerprints match what
this config would produce — an archive from a superseded chain must never be selectable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.eval.policies import rl_checkpoint_dir
from src.pipeline import build_retrieval, train_rl
from src.pipeline.eval_baselines import selection_freeze_path
from src.rl.reinforce import RLConfig
from src.rl.reward import RewardConfig
from src.utils.config import add_profile_args, config_from_args, resolve
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, read_json, read_jsonl
from src.utils.runs import RunRegistry, now_iso

STAGE = "P6_select"

# ------------------------------------------------------------------ the pre-registered rule
#: Primary selection metric: mean episodic reward on val_strict (greedy), as recorded by the
#: run's own post-selection re-measurement (``config.json.val_reward``), cross-checked
#: against the registry row. Numbers are never recomputed at select time.
PRIMARY_METRIC = "val_reward"
#: Deterministic tie-break; derivable from the frozen artifacts alone (D016).
TIE_BREAK = "lowest_seed"
#: Budgeted seeds (D015). Missing seeds are warned about and recorded, never hidden.
PLANNED_SEEDS = (42, 43, 44)
RULE = ("highest val_reward (mean episodic reward on val_strict, greedy) among eligible "
        "P6_rl archives; tie-break lowest seed; eligible = one SUCCEEDED P6_rl registry row "
        "for this profile with laya.regime == laya.frozen and retrieval/reader/reward/eval "
        "fingerprints matching the current chain")

#: The four artifacts that identify a B8 checkpoint (``policies.RL_REQUIRED_FILES`` + rl_config).
CORE_FILES = ("policy.safetensors", "scaler.json", "rl_config.json", train_rl.CONFIG_FILE)
#: The rest of the evidence set every archive must carry (see ``train_rl.archive_run``).
ARCHIVE_FILES = (train_rl.PRIOR_REF_FILE, train_rl.TRAJECTORIES_FILE, train_rl.TRAIN_LOG_FILE,
                 f"summary/{train_rl.SUMMARY_FILE}", f"summary/{train_rl.CURVE_FILE}",
                 f"summary/{train_rl.HISTOGRAM_FILE}")
TERMINAL = ("SUCCEEDED", "FAILED", "SKIPPED", "BLOCKED", "NOT_RUN", "INTERRUPTED")
STAGING_DIR = ".rl_policy.staging"

#: ``CORE_FILES`` name -> the ``<stem>_sha256`` key recorded for it in a candidate/winner.
_FILE_STEM = {"policy.safetensors": "policy", "scaler.json": "scaler",
              "rl_config.json": "rl_config", train_rl.CONFIG_FILE: "config"}


class SelectionRefused(RuntimeError):
    """The selection cannot be performed (nothing eligible, or a guard forbids it)."""


# ------------------------------------------------------------------------------- scanning

def _registry_row(rows: Sequence[Mapping[str, Any]], run_id: str, profile: str
                  ) -> tuple[dict | None, str | None]:
    """The single terminal ``P6_rl`` row for this run, or an exclusion reason."""
    mine = [r for r in rows
            if r.get("stage") == train_rl.STAGE and r.get("run_id") == run_id
            and r.get("status") in TERMINAL]
    if len(mine) != 1:
        return None, f"{len(mine)} terminal P6_rl registry row(s) for {run_id}; expected 1"
    row = mine[0]
    if row.get("status") != "SUCCEEDED":
        return None, f"registry row status {row.get('status')}"
    if row.get("profile") != profile:
        return None, f"registry row profile {row.get('profile')!r} != {profile!r}"
    return row, None


def scan_candidates(cfg: Mapping[str, Any], profile: str, rows: Sequence[Mapping[str, Any]]
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """``(candidates, excluded, corruption)`` over ``outputs/models/rl_runs/*/``.

    A candidate is selectable iff its registry row is a lone SUCCEEDED ``P6_rl`` row for this
    profile, its regime is the frozen one, and every chain fingerprint matches the current
    config. Anything else is *excluded with a reason*. A registry row that contradicts its own
    ``config.json`` (different seed or val_reward), or an archive missing part of its
    evidence set, is *corruption*: the archives are the evidence, so a disagreement means the
    evidence is broken — refusing beats selecting from it.
    """
    root = resolve(cfg, "models") / train_rl.RL_RUNS_DIR
    frozen = (cfg.get("laya") or {}).get("frozen")
    retrieval_sha = build_retrieval.retrieval_config_sha(dict(cfg))
    rl_config = RLConfig.from_dict(dict(cfg))
    reward_hash = RewardConfig.from_dict(dict(cfg)).spec_hash()
    candidates: list[dict] = []
    excluded: list[dict] = []
    corruption: list[dict] = []

    for archive in sorted(root.glob("*/")) if root.is_dir() else []:
        entry: dict[str, Any] = {"run_id": archive.name}
        config_path = archive / train_rl.CONFIG_FILE
        if not config_path.is_file():
            excluded.append({**entry, "reason": "no config.json in the archive"})
            continue
        config = read_json(config_path)
        entry.update(seed=config.get("seed"), regime=(config.get("laya") or {}).get("regime"))

        row, reason = _registry_row(rows, archive.name, profile)
        if row is None:
            excluded.append({**entry, "reason": reason or "no registry row"})
            continue
        if entry["regime"] != frozen:
            excluded.append({**entry, "reason": f"laya.regime {entry['regime']!r} "
                                                f"!= frozen {frozen!r}"})
            continue
        if config.get("retrieval_config_sha") != retrieval_sha:
            excluded.append({**entry,
                             "reason": "retrieval_config_sha differs from the current chain"})
            continue
        recorded_reader = config.get("reader") or {}
        mismatch = next(
            (f"reader {split} {key} differs from the current chain"
             for split in (train_rl.TRAIN_SPLIT, train_rl.VAL_SPLIT)
             for key in ("reader_model_hash", "tokenizer_sha", "inference_config_sha")
             if (recorded_reader.get(split) or {}).get(key)
             != train_rl.reader_meta(cfg, split).get(key)),
            None)
        if mismatch:
            excluded.append({**entry, "reason": mismatch})
            continue
        episodes = (config.get("rl_config") or {}).get("eval_episodes")
        if episodes != rl_config.eval_episodes:
            excluded.append({**entry, "reason": f"rl_config.eval_episodes {episodes!r} "
                                                f"!= {rl_config.eval_episodes!r} — different "
                                                "metric noise, not comparable"})
            continue
        if config.get("reward_spec_hash") != reward_hash:
            excluded.append({**entry,
                             "reason": "reward_spec_hash differs from the current chain"})
            continue

        missing = [name for name in (*CORE_FILES, *ARCHIVE_FILES)
                   if not (archive / name).is_file()]
        if missing:
            corruption.append({**entry, "reason": f"archive incomplete: missing {missing}"})
            continue
        disagreements = [f"registry {key}={row.get(key)!r} != config.json {key}={config.get(key)!r}"
                         for key in ("seed", "val_reward")
                         if row.get(key) is None or config.get(key) is None
                         or row[key] != config[key]]
        if disagreements:
            corruption.append({**entry, "reason": "; ".join(disagreements)})
            continue

        candidates.append({
            "seed": int(config["seed"]), "run_id": archive.name,
            "archive_dir": str(archive), "val_reward": config["val_reward"],
            "val_std_reward": config.get("val_std_reward"),
            "best_update": config.get("best_update"), "updates_ran": config.get("updates_ran"),
            "eval_episodes": episodes,
            "laya_model_hash": (config.get("laya") or {}).get("model_hash"),
            "retrieval_config_sha": config.get("retrieval_config_sha"),
            **{f"{_FILE_STEM[name]}_sha256": sha256_file(archive / name)
               for name in CORE_FILES},
        })
    return candidates, excluded, corruption


# ------------------------------------------------------------------ install + freeze

def install_winner(cfg: Mapping[str, Any], winner: Mapping[str, Any]) -> None:
    """Make ``outputs/models/rl_policy`` byte-identical to the winner's archive minus ``summary/``.

    Staged copy + sha256 verification *before* the swap: the canonical dir is only replaced
    once the staged files provably equal the winner's recorded hashes. The winner's
    ``summary/`` files then go back to the shared fixed paths, so every report citation
    describes the selected run and not whichever seed finished last.
    """
    canonical = rl_checkpoint_dir(cfg)
    archive = Path(winner["archive_dir"])
    staging = canonical.parent / STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(archive, staging, ignore=shutil.ignore_patterns("summary"))
    for name in CORE_FILES:
        digest = sha256_file(staging / name)
        expected = winner[f"{_FILE_STEM[name]}_sha256"]
        if digest != expected:
            raise RuntimeError(f"staged {name} sha256 {digest} != winner archive's {expected}")
    if canonical.exists():
        shutil.rmtree(canonical)
    staging.rename(canonical)
    for name, dest in ((train_rl.SUMMARY_FILE, resolve(cfg, "reports") / train_rl.SUMMARY_FILE),
                       (train_rl.CURVE_FILE, resolve(cfg, "plots") / train_rl.CURVE_FILE),
                       (train_rl.HISTOGRAM_FILE,
                        resolve(cfg, "plots") / train_rl.HISTOGRAM_FILE)):
        source = archive / "summary" / name
        if source.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)


def build_freeze(cfg: Mapping[str, Any], *, profile: str, now: str, run_id: str,
                 candidates: Sequence[Mapping[str, Any]], excluded: Sequence[Mapping[str, Any]],
                 winner: Mapping[str, Any], forced: bool) -> dict[str, Any]:
    """The ``best_selection.json`` payload; every number copied from the archives, none retyped."""
    canonical = rl_checkpoint_dir(cfg)
    values = [float(c["val_reward"]) for c in candidates]
    present = sorted(int(c["seed"]) for c in candidates)
    return {
        "stage": STAGE, "run_id": run_id, "profile": profile, "created_utc": now,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_desc": ("mean episodic reward on val_strict, greedy, "
                                f"eval_episodes={winner['eval_episodes']}"),
        "tie_break": TIE_BREAK, "rule": RULE,
        "frozen_prior_regime": (cfg.get("laya") or {}).get("frozen"),
        "eval_episodes": winner["eval_episodes"],
        "planned_seeds": list(PLANNED_SEEDS),
        "seeds_present": present,
        "seeds_missing": [seed for seed in PLANNED_SEEDS if seed not in present],
        "seed_summary": {"n": len(values), "val_reward_mean": sum(values) / len(values),
                         "val_reward_min": min(values), "val_reward_max": max(values)},
        "candidates": [dict(c) for c in candidates],
        "excluded": [dict(e) for e in excluded],
        "winner": dict(winner),
        "selected_update": winner.get("best_update"),
        "canonical_dir": str(canonical),
        "canonical_files": {name: {"sha256": sha256_file(canonical / name),
                                   "bytes": (canonical / name).stat().st_size}
                            for name in CORE_FILES},
        "forced": forced,
    }


# ----------------------------------------------------------------------------------- run

def run(*, cfg: Mapping[str, Any], profile: str, now: str | None = None, force: bool = False,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    now = now or now_iso()
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    rows = list(read_jsonl(registry.path)) if registry.path.is_file() else []
    run_id = f"p6-select-{profile}-{now.replace(':', '')}"
    freeze_path = selection_freeze_path(cfg)

    # The registry is append-only: a finished run keeps its historical RUNNING row, so a run
    # is "still training" only when its *latest* row is RUNNING.
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("stage") == train_rl.STAGE and row.get("profile") == profile:
            latest[str(row.get("run_id"))] = row
    running = [run for run, row in latest.items() if row.get("status") == "RUNNING"]
    if running:
        raise SelectionRefused(f"a P6_rl run is still RUNNING ({', '.join(running)}); "
                               "freeze only after training finished")
    tested = [r for r in rows if r.get("stage") == "P7_baselines"
              and r.get("split") == "test_strict" and r.get("status") == "SUCCEEDED"]
    if tested and not force:
        raise SelectionRefused(
            f"the strict test split was already evaluated ({tested[0].get('run_id')}); "
            "re-freezing now would invalidate the freeze-before-test ordering (spec 03 §5). "
            "Pass --force to record an explicit, disclosed deviation")

    candidates, excluded, corruption = scan_candidates(cfg, profile, rows)
    if corruption:
        registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile,
                        force=force)
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        error=f"archive corruption: {corruption}")
        raise RuntimeError(f"P6_rl archive corruption, refusing to select: {corruption}")
    if not candidates:
        raise SelectionRefused(
            f"no eligible P6_rl archives under {resolve(cfg, 'models') / train_rl.RL_RUNS_DIR}; "
            f"excluded={excluded}")

    winner = min(candidates, key=lambda c: (-float(c["val_reward"]), int(c["seed"])))
    missing = [seed for seed in PLANNED_SEEDS
               if seed not in {int(c["seed"]) for c in candidates}]
    if missing:
        print(f"WARNING: planned seed(s) {missing} have no eligible archive; the freeze "
              f"records them in seeds_missing (spec 04 §5: pilot seed first, widen if budget)")

    registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile,
                    force=force, winner_seed=winner["seed"])
    try:
        install_winner(cfg, winner)
        freeze = build_freeze(cfg, profile=profile, now=now, run_id=run_id,
                              candidates=candidates, excluded=excluded, winner=winner,
                              forced=bool(force))
        atomic_write_json(freeze_path, freeze)
        artifacts = [freeze_path, *(rl_checkpoint_dir(cfg) / name for name in CORE_FILES)]
        registry.record_artifact(run_id, STAGE, artifacts, profile=profile,
                                 winner_seed=winner["seed"], winner_run_id=winner["run_id"],
                                 policy_sha256=winner["policy_sha256"],
                                 val_reward=winner["val_reward"],
                                 seeds_present=freeze["seeds_present"],
                                 seeds_missing=freeze["seeds_missing"])
    except Exception as exc:
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise
    return {"run_id": run_id, "stage": STAGE, "profile": profile, "freeze": str(freeze_path),
            "canonical_dir": freeze["canonical_dir"], "winner": winner, "excluded": excluded,
            "seeds_present": freeze["seeds_present"], "seeds_missing": freeze["seeds_missing"],
            "seed_summary": freeze["seed_summary"], "forced": bool(force),
            "canonical_files": freeze["canonical_files"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.select_rl",
        description="P6: freeze the best-by-validation B8 checkpoint (best_selection.json)")
    add_profile_args(parser)
    parser.add_argument("--force", action="store_true",
                        help="allow re-freezing after a test row (recorded as forced=true)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    if args.seed is not None:
        print("note: --seed is ignored; selection is a deterministic rule over archived "
              "runs (the candidates' seeds are part of the rule)", file=sys.stderr)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    try:
        summary = run(cfg=cfg, profile=profile, force=args.force)
    except SelectionRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    winner = summary["winner"]
    print(f"run_id={summary['run_id']} winner_seed={winner['seed']} "
          f"val_reward={float(winner['val_reward']):.4f} "
          f"seeds_present={summary['seeds_present']}")
    print(f"  freeze: {summary['freeze']}")
    print(f"  canonical: {summary['canonical_dir']} policy.safetensors "
          f"sha256={summary['canonical_files']['policy.safetensors']['sha256'][:12]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

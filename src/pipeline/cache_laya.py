"""P5 CLI: pre-compute Laya action probabilities for every reachable state (`02_IMPL/04` §5).

    python -m src.pipeline.cache_laya --config configs/pilot.yaml --regime rlcd
    python -m src.pipeline.cache_laya --config configs/pilot.yaml --regime ce --force

P6 asks the frozen prior for probabilities at every step of every rollout. This stage answers
those questions once: it enumerates exactly the states a rollout can decide from
(:func:`src.pipeline.worlds.reachable_state_requests` replays the same environment the
rollout uses), computes the misses through :class:`~src.decision.laya_cache.LayaProbCache`
in batches, and leaves a parquet the rollouts read without touching the model.

**Fail-fast on checkpoint drift.** Rows are stamped with the adapter's ``model_hash``. If the
stored rows belong to another checkpoint the run stops with
:class:`~src.decision.laya_cache.ModelHashMismatch` *before* anything is recomputed — silently
mixing priors from two checkpoints would be a silent scientific error. ``--force`` is for the
legitimate replacement case only. Each regime owns its own cache directory
(``laya_probs_cache/<regime>``), so a forced run for one regime can only discard that regime's
rows — a P7 row for the other regime can never invalidate this run's artifact.

**Independent tree check.** Per split the run counts the states of the retrieval tree
(:func:`src.retrieval.states.reachable_states` over the same rankings the environment uses).
The replay must enumerate at least that many states; fewer means it missed a branch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.decision.laya_adapter import LayaAdapter
from src.decision.laya_cache import LayaProbCache
from src.pipeline import worlds
from src.retrieval.states import reachable_states
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import add_profile_args, config_from_args, profile_limit, resolve, \
    snapshot_config
from src.utils.io import atomic_write_json
from src.utils.runs import RunRegistry, now_iso

#: Splits P6 rolls out over (same pair ``cache_reader`` caches; test stays locked).
CACHE_SPLITS = ("train_strict", "val_strict")


def _cache_split(cfg: Mapping[str, Any], split: str, *, limit: int | None, batch: int,
                 cache: LayaProbCache, adapter: Any) -> dict[str, Any]:
    """Cache every reachable state of one split; returns that split's statistics."""
    bundle = worlds.build_worlds(cfg, split, limit=limit, laya_regime=None)
    config = worlds.env_config(cfg)
    reward = RewardCalculator(RewardConfig.from_dict(cfg))
    requests = worlds.reachable_state_requests(bundle.worlds, reward, config)
    if not requests:
        raise ValueError(f"{split}: the replay enumerated no states to cache")

    tree_states = sum(
        len(reachable_states(world.rankings, bootstrap_tool=config.bootstrap_tool,
                             bootstrap_passages=config.bootstrap_passages,
                             passage_per_search_action=config.passage_per_search_action,
                             max_optional_search=config.max_optional_search,
                             tools=worlds.TOOLS))
        for world in bundle.worlds)
    # Histories, not tuples: the rendered text carries the action history, so distinct
    # histories are distinct states. A decision state maps onto one retrieval-tree tuple and
    # every tuple comes from some history, so the replay can only ever enumerate more.
    if len(requests) < tree_states:
        raise ValueError(
            f"{split}: the replay enumerated {len(requests)} states but the retrieval tree "
            f"has {tree_states}; the replay and the environment disagree — rebuild the P3 "
            "cache (cache_reader) rather than caching a partial tree")

    started = time.monotonic()
    for start in range(0, len(requests), batch):
        probs = cache.get_or_compute(adapter, requests[start:start + batch])
        values = np.stack(list(probs.values()))
        if not np.all(np.isfinite(values)) or not np.allclose(values.sum(axis=1), 1.0,
                                                              atol=1e-6):
            raise ValueError(f"{split}: the adapter returned rows that are not probability "
                             "distributions; fix the checkpoint before caching its outputs")
    missing = [(qid, state_hash) for qid, state_hash, _ in requests
               if cache.lookup(qid, state_hash) is None]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} state(s) still missing after the run, "
                           f"e.g. {missing[0]}")

    return {"split": split, "questions": len(bundle.worlds), "states": len(requests),
            "tree_states": tree_states,
            "walltime_s": round(time.monotonic() - started, 1)}


def run(*, cfg: dict, profile: str, regime: str | None = None,
        splits: Sequence[str] = CACHE_SPLITS, limit: int | None = None,
        batch: int = worlds.PREWARM_BATCH, force: bool = False, adapter: Any | None = None,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    """Warm the Laya probability cache for ``splits``; ``adapter`` is a test seam."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    regime_name = worlds.resolve_laya_regime(cfg, worlds.DEFAULT_LAYA_SENTINEL
                                             if regime is None else regime)
    if regime_name is None:
        raise ValueError("cache_laya needs a frozen Laya checkpoint; pass --regime ce|rlcd")
    checkpoint = worlds.laya_checkpoint_dir(cfg, regime_name)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Laya checkpoint for regime {regime_name!r} is missing: "
                                f"{checkpoint} — train it with the P5 stage (train_laya)")
    cache = LayaProbCache(worlds.laya_cache_dir(cfg, regime_name))
    run_id = f"p5-cache-laya-{regime_name}-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P5_cache_laya", status="RUNNING", profile=profile,
                    regime=regime_name, splits=list(splits), force=force,
                    checkpoint=str(checkpoint))
    owns_adapter = adapter is None
    loaded = adapter if adapter is not None else LayaAdapter.load(checkpoint)
    try:
        model_hash = loaded.model_hash()
        if not force:
            cache.require_model_hash(model_hash)
        per_split = [_cache_split(cfg, split, limit=limit, batch=batch, cache=cache,
                                  adapter=loaded) for split in splits]
        summary = {
            "run_id": run_id, "stage": "P5_cache_laya", "profile": profile,
            "regime": regime_name, "checkpoint": str(checkpoint), "model_hash": model_hash,
            "cache_dir": str(cache.cache_dir), "batch": batch, "force": force,
            "splits": {stats["split"]: stats for stats in per_split},
            "states": sum(stats["states"] for stats in per_split),
            "tree_states": sum(stats["tree_states"] for stats in per_split),
            "hits": cache.hits, "misses": cache.misses,
            "invalidated_rows": cache.invalidated_rows,
            "rows_in_cache": len(cache),
            "created_utc": now_iso(),
        }
        summary_path = resolve(cfg, "manifests") / f"P5_cache_laya_{regime_name}_{profile}.json"
        atomic_write_json(summary_path, summary)
    except Exception as exc:
        registry.append(run_id=run_id, stage="P5_cache_laya", status="FAILED", profile=profile,
                        regime=regime_name, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if owns_adapter:
            loaded.close()

    registry.record_artifact(run_id, "P5_cache_laya",
                             [cache.parquet_path, cache.meta_path, summary_path],
                             profile=profile, regime=regime_name, model_hash=model_hash,
                             states=summary["states"], tree_states=summary["tree_states"],
                             cache_hits=cache.hits, cache_misses=cache.misses,
                             invalidated_rows=cache.invalidated_rows, force=force)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.cache_laya",
        description="P5: pre-compute Laya action probabilities for all reachable states")
    add_profile_args(parser)
    parser.add_argument("--regime", default=worlds.DEFAULT_LAYA_SENTINEL,
                        choices=(worlds.DEFAULT_LAYA_SENTINEL, *worlds.LAYA_REGIMES),
                        help="which fine-tuned checkpoint's probabilities to cache "
                             "(default: laya.frozen from the config)")
    parser.add_argument("--splits", default=",".join(CACHE_SPLITS),
                        help=f"comma-separated splits (default: {','.join(CACHE_SPLITS)})")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap the scenario list (default: the profile's cap)")
    parser.add_argument("--batch", type=int, default=worlds.PREWARM_BATCH,
                        help="states per get_or_compute call")
    parser.add_argument("--force", action="store_true",
                        help="replace a cache built from another checkpoint (discards it)")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    if not splits:
        parser.error("--splits must name at least one split")
    limit = args.limit if args.limit is not None else profile_limit(cfg, profile)
    regime = worlds.resolve_laya_regime(cfg, args.regime)
    if regime is None:
        parser.error("cache_laya needs a frozen Laya checkpoint; pass --regime ce|rlcd")
    snapshot_config(cfg, resolve(cfg, "manifests")
                    / f"P5_cache_laya_{regime}_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, regime=regime, splits=splits, limit=limit,
                  batch=args.batch, force=args.force)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"run_id={summary['run_id']} regime={summary['regime']} "
          f"checkpoint={summary['checkpoint']}")
    print(f"  model_hash={summary['model_hash'][:16]}… rows={summary['rows_in_cache']} "
          f"states={summary['states']} tree={summary['tree_states']}")
    print(f"  cache: hits={summary['hits']} misses={summary['misses']} "
          f"invalidated={summary['invalidated_rows']} dir={summary['cache_dir']}")
    for split, stats in summary["splits"].items():
        print(f"  {split}: questions={stats['questions']} states={stats['states']} "
              f"tree={stats['tree_states']} walltime={stats['walltime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

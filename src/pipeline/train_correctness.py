"""B5 CLI: answer-correctness estimator + calibrated ANSWER threshold (`08_...md`).

    python -m src.pipeline.train_correctness --config configs/pilot.yaml
    python -m src.pipeline.train_correctness --profile smoke --oof-folds 3 --dry-run

The stage trains the B5 baseline's ``P(correct)`` model and picks its ANSWER threshold.
The target is *answer correctness of the final candidate* (spec 08 §2), never
``P(action=ANSWER)`` and never the reader's raw span score:

    y_correct = 1  iff  the candidate matches gold under the reward's own criterion
                        (token-F1 >= candidate_f1_threshold and provenance valid)

Labels come from :mod:`src.rl.reward` via :func:`src.eval.correctness.candidate_label` — one
correctness definition for the whole project — while features are the observation-only vector
of :func:`src.eval.correctness.features_from_obs` (no gold, no label, no reward, no Laya).

Reader optimism (spec 08 §3) is handled by cross-fitting, never by hope:

* ``--oof-folds 1`` (default, budget option): one group-disjoint *title* holdout is removed
  from the reader training set; a reader is re-trained on the reduced set through the P3 code
  path (``train_reader.load_rows`` -> ``build_feature_set`` -> ``qa.train_reader.train`` -> the
  P3 val threshold sweep) and its predictions on the holdout become the estimator's training
  rows.
* ``--oof-folds 3``: full cross-fitting — fold ``i``'s features come from a reader trained on
  the other two folds. A fold whose val sweep meets neither of the §10 acceptance
  constraints (D010) has no usable reader: it is recorded as a FAILED fold and skipped, and
  the stage fails only when every fold fails.

The *deployed* reader is always the frozen P3 full-train reader (its state cache), so the
estimator never trains on predictions of the reader it will run with. Validation is split by
title group into ``val_calibration`` (fit the Platt/isotonic calibrator) and
``val_model_selection`` (choose ``tau_answer``); ``test_strict`` is never read.

If the calibration rows are too few / too imbalanced (``correctness.calibration_plan``), the
run deliberately degrades to **B5-RAW** — the reader ``null_margin`` threshold — and reports
``N/A`` for ECE/Brier instead of mislabeling a raw score as a probability (spec 08 §5, §8).

Artifacts (registered in ``outputs/runs/index.jsonl`` with SHA256):

* ``outputs/models/correctness_estimator/{model,scaler,calibrator}.json`` — the estimator
* ``outputs/models/correctness_estimator/provenance.json`` — hashes, criterion, fold manifest
* ``outputs/models/correctness_estimator/split_manifest.json`` — the locked qid partitions
* ``outputs/models/correctness_estimator/threshold_selection.json`` — the tau selection log
* ``reports/correctness_calibration.md``, ``reports/plots/correctness_reliability.png``
* the B5 row: appended to ``reports/results.csv`` when that file already exists, otherwise
  written to ``reports/results_b5_row.json`` with a note
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data import artifacts
from src.eval import splitting
from src.eval.correctness import (
    CORRECTNESS_FEATURES,
    Calibrator,
    CorrectnessEstimator,
    calibration_plan,
    candidate_label,
    features_from_obs,
    fit_calibrator,
    reliability_metrics,
    selection_metrics,
)
from src.pipeline import build_retrieval, train_reader, worlds
from src.pipeline.cache_reader import load_rankings, state_cache_paths
from src.qa import state_cache as state_cache_mod
from src.qa import train_reader as qa_train_reader
from src.qa.features import build_feature_set
from src.qa.infer import ReaderInference, inference_config_sha, use_fp16
from src.qa.postprocess import choose_best_threshold, sweep_threshold
from src.qa.reader import ReaderPrediction
from src.qa.reader_eval import evaluate_predictions, gold_index, threshold_grid
from src.rl.env import ABSTAIN, ANSWER, SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv, EpisodeWorld
from src.rl.features import FeatureScaler
from src.rl.reward import ACTIONS, RewardCalculator, RewardConfig
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, resolve, \
    snapshot_config
from src.utils.hashing import sha256_json
from src.utils.io import atomic_write_json, write_parquet
from src.utils.reproducibility import capture_library_versions, seed_everything
from src.utils.runs import RunRegistry, git_sha, now_iso

STAGE = "B5_correctness"
MODEL_DIR = "correctness_estimator"
OOF_READER_DIR = "oof_readers"
TRAIN_SPLIT = "train_strict"
VAL_SPLIT = "val_strict"

PROVENANCE_FILE = "provenance.json"
SPLIT_MANIFEST_FILE = "split_manifest.json"
THRESHOLD_FILE = "threshold_selection.json"
REPORT_FILE = "correctness_calibration.md"
PLOT_FILE = "correctness_reliability.png"
RESULTS_ROW_FILE = "results_b5_row.json"
RESULTS_CSV_FILE = "results.csv"

#: Optional overrides under ``cfg["correctness"]``; the defaults are the pre-registered ones.
DEFAULT_OOF_HOLDOUT_FRACTION = 0.25
DEFAULT_CALIBRATION_FRACTION = 0.2
DEFAULT_TAU_QUANTILES = 20
DEFAULT_BOOTSTRAP_RESAMPLES = 1000

#: A fold reader must be a real reader: a handful of questions cannot yield one.
MIN_FOLD_ROWS = 8
#: Reported risk at this coverage uses the calibrated ranking (spec 03 §1).
RISK_AT_COVERAGE = 0.8

#: **Pre-registered** threshold-selection criterion (spec 08 §5). Fixed here, in code, and
#: written verbatim into every artifact *by* the run — it is never chosen from the results.
PRE_REGISTERED_CRITERION = (
    "tau_answer = argmax over the candidate grid of mean_episode_reward on "
    "val_model_selection, replaying the deterministic B5 rule (ANSWER iff the action mask "
    "allows it and score >= tau; else SEARCH_BM25, then SEARCH_DENSE while budget remains; "
    "else ABSTAIN) through the shared environment/reward of src.pipeline.worlds; ties are "
    "broken toward the higher tau; the never-answer sentinel is flagged if it wins"
)


# ------------------------------------------------------------------------------- paths

def estimator_dir(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg, "models") / MODEL_DIR


def correctness_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("correctness") or {})


# ------------------------------------------------------------------------ reader sources

@dataclass
class ReaderSource:
    """Reader predictions the correctness features may be computed from.

    ``lookup`` raises on a miss: a substituted prediction would fabricate the evidence the
    estimator is supposed to be measuring (same rule as the RL environment).
    """

    name: str
    model_hash: str
    tokenizer_sha: str
    inference_config_sha: str
    threshold: float
    lookup: Callable[[str, tuple[str, ...]], ReaderPrediction]
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "reader_model_hash": self.model_hash,
                "tokenizer_sha": self.tokenizer_sha,
                "inference_config_sha": self.inference_config_sha,
                "null_score_diff_threshold": self.threshold, **self.meta}


def p3_reader_source(cfg: Mapping[str, Any], split: str) -> ReaderSource:
    """The frozen deployed reader: the exhaustive P3 state cache of ``split``."""
    parquet, meta_path = state_cache_paths(cfg, split)
    meta = state_cache_mod.load_state_cache_meta(meta_path)
    cache = state_cache_mod.load_state_cache(parquet)

    def lookup(qid: str, visible: tuple[str, ...]) -> ReaderPrediction:
        key = state_cache_mod.state_cache_key(meta, qid, visible)
        try:
            return cache.lookup(key)
        except KeyError as exc:
            raise KeyError(
                f"P3 state cache miss for qid={qid!r} visible={list(visible)}; rebuild the "
                "cache (cache_reader) instead of substituting a prediction") from exc

    return ReaderSource(name=f"p3_full_train[{split}]", model_hash=meta["reader_model_hash"],
                        tokenizer_sha=meta["tokenizer_sha"],
                        inference_config_sha=meta["inference_config_sha"],
                        threshold=float(meta.get("null_score_diff_threshold", 0.0)),
                        lookup=lookup, meta={"split": split, "parquet": str(parquet),
                                             "meta": str(meta_path)})


def train_fold_reader_source(cfg: Mapping[str, Any], *, profile: str, seed: int,
                             reader_train_qids: Sequence[str], feature_qids: Sequence[str],
                             out_dir: str | Path, fold_index: int, limit: int | None = None
                             ) -> tuple[ReaderSource, dict[str, Any]]:
    """Train an OOF reader on the reduced train set and cache its predictions.

    This is the P3 recipe reused, not a second trainer: the same row loader
    (:func:`src.pipeline.train_reader.load_rows`), the same feature builder, the same
    :func:`src.qa.train_reader.train` and the same val threshold sweep. Only the training
    *rows* differ (the group holdout is removed) and the output directory is the fold's.
    """
    reader_cfg = cfg["reader"]
    out_dir = Path(out_dir)
    rows, dropped, docs = train_reader.load_rows(cfg, TRAIN_SPLIT, profile=profile,
                                                 limit_override=limit)
    keep = set(reader_train_qids)
    subset = [row for row in rows if row["qid"] in keep]
    if len(subset) < MIN_FOLD_ROWS:
        raise ValueError(
            f"fold {fold_index}: only {len(subset)} reader row(s) after the title holdout "
            f"(< {MIN_FOLD_ROWS}); a reader trained on this cannot be an honest OOF source")
    seed_everything(seed)
    base_model, revision = reader_cfg["base_model"], reader_cfg.get("revision")
    tokenizer = qa_train_reader.load_tokenizer(base_model, revision)
    model = qa_train_reader.load_model(base_model, revision)
    features = build_feature_set(subset, docs, tokenizer, max_length=int(reader_cfg["max_length"]),
                                 doc_stride=int(reader_cfg["doc_stride"]))
    train_summary = qa_train_reader.train(
        model=model, tokenizer=tokenizer, rows=features.rows(), out_dir=out_dir,
        epochs=int(reader_cfg["epochs"]), lr=float(reader_cfg["lr"]),
        batch_size=int(reader_cfg["batch_size"]), grad_accum=int(reader_cfg["grad_accum"]),
        warmup_ratio=float(reader_cfg["warmup_ratio"]),
        weight_decay=float(reader_cfg["weight_decay"]), fp16=use_fp16(reader_cfg), seed=seed)
    model_hash = qa_train_reader.reader_model_hash(out_dir)
    tokenizer_sha = qa_train_reader.tokenizer_hash(tokenizer)
    qa_train_reader.save_model_card(out_dir, base_model=base_model, revision=revision,
                                    tokenizer_sha=tokenizer_sha, summary=train_summary)

    inference = ReaderInference.load(out_dir, max_length=int(reader_cfg["max_length"]),
                                     doc_stride=int(reader_cfg["doc_stride"]),
                                     max_answer_length=int(reader_cfg["max_answer_length"]),
                                     batch_size=int(reader_cfg["inference_batch_size"]),
                                     fp16=use_fp16(reader_cfg))
    sweep = tune_null_threshold(cfg, profile=profile, inference=inference, model_hash=model_hash)
    cache_summary = build_fold_state_cache(cfg, split=TRAIN_SPLIT, qids=feature_qids,
                                           inference=inference, model_hash=model_hash,
                                           tokenizer_sha=tokenizer_sha,
                                           threshold=float(sweep["threshold"]),
                                           out_dir=out_dir)
    meta = cache_summary["meta"]
    cache = state_cache_mod.load_state_cache(cache_summary["parquet"])

    def lookup(qid: str, visible: tuple[str, ...]) -> ReaderPrediction:
        key = state_cache_mod.state_cache_key(meta, qid, visible)
        try:
            return cache.lookup(key)
        except KeyError as exc:
            raise KeyError(
                f"fold {fold_index} state cache miss for qid={qid!r} visible={list(visible)}; "
                "the fold cache must cover every reachable state of its feature questions"
            ) from exc

    source = ReaderSource(
        name=f"oof_fold_{fold_index}",
        model_hash=model_hash, tokenizer_sha=tokenizer_sha,
        inference_config_sha=cache_summary["meta"]["inference_config_sha"],
        threshold=float(sweep["threshold"]), lookup=lookup,
        meta={"fold": fold_index, "reader_dir": str(out_dir),
              "fold_train_rows": len(subset), "fold_dropped_rows": len(dropped),
              "state_cache": str(cache_summary["parquet"]),
              "state_cache_states": int(cache_summary["stats"]["n_states"])})
    manifest = {
        "index": int(fold_index), "reader_dir": str(out_dir),
        "n_reader_train_questions": len(keep),
        "n_reader_rows": len(subset),
        "n_feature_questions": len(feature_qids),
        "reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
        "null_score_diff_threshold": float(sweep["threshold"]),
        "threshold_tuned_on": sweep["tuned_on"],
        "threshold_val_em_overall": sweep["val_metrics"].get("em_overall"),
        "threshold_val_f1_overall": sweep["val_metrics"].get("f1_overall"),
        "state_cache": str(cache_summary["parquet"]),
        "state_cache_stats": {k: v for k, v in cache_summary["stats"].items()
                              if not isinstance(v, list)},
    }
    return source, manifest


def tune_null_threshold(cfg: Mapping[str, Any], *, profile: str, inference: ReaderInference,
                        model_hash: str) -> dict[str, Any]:
    """The P3 threshold-tuning procedure, reused for a fold reader (oracle context on val)."""
    reader_cfg = cfg["reader"]
    val_rows, dropped, docs_val = train_reader.load_rows(cfg, VAL_SPLIT, profile=profile)
    if not val_rows:
        raise ValueError(f"no usable {VAL_SPLIT} rows to tune the fold reader's threshold")
    spans = inference.predict(train_reader.oracle_pairs(val_rows, docs_val))
    if not spans:
        raise ValueError("no val pair could be built for the threshold sweep")
    grid = reader_cfg["threshold_sweep"]
    sweep = sweep_threshold(spans, gold_index(val_rows),
                            threshold_grid(float(grid["start"]), float(grid["stop"]),
                                           float(grid["step"])),
                            reader_model_hash=model_hash, context_by_doc=dict(docs_val),
                            evaluate=evaluate_predictions)
    best = choose_best_threshold(sweep)
    return {"threshold": float(best["threshold"]), "val_metrics": best,
            "sweep_points": len(sweep), "tuned_on": VAL_SPLIT, "val_rows": len(val_rows),
            "val_dropped_rows": len(dropped)}


def build_fold_state_cache(cfg: Mapping[str, Any], *, split: str, qids: Sequence[str],
                           inference: ReaderInference, model_hash: str, tokenizer_sha: str,
                           threshold: float, out_dir: Path) -> dict[str, Any]:
    """Exhaustive state cache for ``qids`` under the fold reader (P3 builder, reused).

    Inputs are sourced exactly like :func:`src.pipeline.cache_reader.build_split_cache`
    (per-split docs/questions/rankings), so a fold row for a state the deployment also walks
    is byte-for-byte what P3 would have written for that state under the fold reader.
    """
    processed = resolve(cfg, "data_processed")
    wanted = set(qids)
    retrieval = cfg["retrieval"]
    config_check = worlds.env_config(cfg)
    if int(retrieval["optional_searches"]) < config_check.max_optional_search:
        raise ValueError(
            f"retrieval.optional_searches={retrieval['optional_searches']} < "
            f"rl.max_optional_search={config_check.max_optional_search}: the fold cache "
            "would not cover the states the environment walks")
    docs = artifacts.read_docs(processed, split)
    scenarios = sorted((scenario for scenario
                        in artifacts.read_scenarios(resolve(cfg, "data_scenarios"), split)
                        if scenario.qid in wanted), key=lambda s: s.scenario_id)
    if not scenarios:
        raise ValueError(f"no {split} scenario for the fold's feature questions")
    questions = artifacts.question_texts(processed, split)
    rankings = {qid: ranking for qid, ranking in load_rankings(cfg, split).items()
                if qid in wanted}
    config_sha = inference_config_sha(cfg["reader"], threshold=threshold, model_hash=model_hash)
    rows, stats = state_cache_mod.build_state_cache(
        scenarios=scenarios, questions=questions, docs=docs, rankings=rankings,
        inference=inference, threshold=threshold, reader_model_hash=model_hash,
        tokenizer_sha=tokenizer_sha, inference_config_sha=config_sha,
        retrieval=retrieval)
    out_dir = Path(out_dir)
    parquet_path = out_dir / f"qa_state_cache_{split}_fold.parquet"
    meta_path = out_dir / f"qa_state_cache_{split}_fold.json"
    parquet_meta = write_parquet(parquet_path, rows)
    meta = {
        "schema_version": state_cache_mod.SCHEMA_VERSION,
        "split": split, "fold_state_cache": True, "questions": sorted(wanted),
        "reader_model_hash": model_hash, "tokenizer_sha": tokenizer_sha,
        "inference_config_sha": config_sha, "null_score_diff_threshold": float(threshold),
        "retrieval_budget": {
            "bootstrap_tool": retrieval["initial_bootstrap"],
            "bootstrap_passages": int(retrieval["bootstrap_passages"]),
            "passage_per_search_action": int(retrieval["passage_per_search_action"]),
            "optional_searches": int(retrieval["optional_searches"])},
        "parquet": {"path": str(parquet_path), **parquet_meta},
    }
    atomic_write_json(meta_path, meta)
    return {"parquet": parquet_path, "meta_path": meta_path, "meta": meta, "stats": stats}


# ------------------------------------------------------------------------- state features

def _replay(env: AdaptiveQAEnv, qid: str, history: tuple[int, ...]) -> tuple[dict, bool]:
    obs, _ = env.reset(seed=0, options={"qid": qid})
    terminated = truncated = False
    for action in history:
        obs, _, terminated, truncated, _ = env.step(action)
    return obs, bool(terminated or truncated)


def with_reader(world: EpisodeWorld, source: ReaderSource) -> EpisodeWorld:
    """Same world, predictions served by ``source``.

    ``dataclasses.replace`` drops the per-tool score maps ``build_worlds`` attaches, so they
    are copied across explicitly: losing them would silently degrade the retrieval features to
    the rank proxy (``src.rl.env._retrieval_summary``).
    """
    def read(visible: tuple[str, ...], qid: str = world.qid) -> ReaderPrediction:
        return source.lookup(qid, tuple(visible))

    clone = replace(world, reader=read)
    for name in ("_scores_bm25", "_scores_dense"):
        if hasattr(world, name):
            setattr(clone, name, getattr(world, name))
    return clone


def world_set(cfg: Mapping[str, Any], bundle: worlds.WorldBundle, source: ReaderSource, *,
              qids: Sequence[str] | None = None, exclude: Sequence[str] = ()
              ) -> tuple[list[EpisodeWorld], AdaptiveQAEnv]:
    """The selected worlds (reader replaced by ``source``) and one environment over them."""
    wanted = None if qids is None else set(qids)
    blocked = set(exclude)
    selected = [with_reader(world, source) for world in bundle.worlds
                if (wanted is None or world.qid in wanted) and world.qid not in blocked]
    if not selected:
        raise ValueError("no world left after qid/exclusion filtering")
    reward = RewardCalculator(RewardConfig.from_dict(dict(cfg)))
    env = AdaptiveQAEnv(selected, reward, worlds.env_config(cfg))
    return selected, env


def reachable_states(env: AdaptiveQAEnv, world_list: Sequence[EpisodeWorld]
                     ) -> list[tuple[EpisodeWorld, dict, str]]:
    """``(world, obs, state_hash)`` for every state a rollout can decide from, deduplicated.

    The walk mirrors :func:`src.pipeline.worlds.reachable_state_requests` (same frontier, same
    replay), so the state set cannot drift from the one the RL rollouts visit. The state hash
    covers the action history, so every path the B5 rule can take is enumerated — and the
    environment is replayed to the popped history before hashing it, because ``state_hash``
    reads the *live* environment and the frontier walk leaves it at the last child.
    """
    out: list[tuple[EpisodeWorld, dict, str]] = []
    seen: set[str] = set()
    for world in world_list:
        frontier: list[tuple[tuple[int, ...], dict]] = [((), _replay(env, world.qid, ())[0])]
        while frontier:
            history, _ = frontier.pop()
            obs, _ = _replay(env, world.qid, history)
            state_hash = env.state_hash()
            if state_hash not in seen:
                seen.add(state_hash)
                out.append((world, obs, state_hash))
            mask = np.asarray(obs["action_mask"], dtype=bool)
            for action in (SEARCH_BM25, SEARCH_DENSE):
                if not mask[action]:
                    continue
                next_history = history + (action,)
                next_obs, done = _replay(env, world.qid, next_history)
                if not done:
                    frontier.append((next_history, next_obs))
    return out


def state_row(world: EpisodeWorld, obs: Mapping[str, Any], state_hash: str,
              reward: RewardCalculator) -> dict[str, Any] | None:
    """One supervised row, or ``None`` for a state without a candidate (spec 08 §2).

    The label is the reward calculator's own ANSWER criterion; the features are the
    observation-only vector. Gold is read *here and only here*, to label — never as a feature.
    """
    candidate = obs.get("candidate") or {}
    text = candidate.get("text")
    if text is None:
        return None
    seen = tuple(obs.get("seen_doc_ids", ()))
    provenance_ok = bool(candidate.get("provenance_ok")) and candidate.get("doc_id") in seen
    y, f1 = candidate_label(reward, candidate_text=text, provenance_ok=provenance_ok,
                            private=world.private)
    return {"qid": world.qid, "split": world.split, "state_hash": state_hash,
            "y": int(y), "f1": float(f1), "provenance_ok": provenance_ok,
            "is_impossible": bool(world.private.is_impossible),
            "features": features_from_obs(obs)}


def labeled_rows(states: Sequence[tuple[EpisodeWorld, dict, str]],
                 reward: RewardCalculator) -> list[dict[str, Any]]:
    rows = []
    for world, obs, state_hash in states:
        row = state_row(world, obs, state_hash, reward)
        if row is not None:
            rows.append(row)
    return rows


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        raise ValueError("no correctness rows to stack")
    return np.vstack([np.asarray(row["features"], dtype=np.float64) for row in rows])


def label_vector(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["y"]) for row in rows], dtype=np.int64)


def score_map(states: Sequence[tuple[EpisodeWorld, dict, str]],
              score_fn: Callable[[dict], float]) -> dict[str, float]:
    """``state_hash -> score`` for every reachable state, so a tau sweep costs no model calls."""
    out: dict[str, float] = {}
    for _, obs, state_hash in states:
        value = float(score_fn(obs))
        if not math.isfinite(value):
            raise ValueError(f"non-finite B5 score at state {state_hash[:16]}...: {value!r}")
        out[state_hash] = value
    return out


# ------------------------------------------------------------------ estimator + variant

def fit_estimator(rows: Sequence[Mapping[str, Any]], *, seed: int, variant: str,
                  meta: Mapping[str, Any]) -> CorrectnessEstimator:
    """Scaler + logistic regression on the (out-of-fold) train rows."""
    matrix = feature_matrix(rows)
    labels = label_vector(rows)
    classes = set(np.unique(labels).tolist())
    if classes == {0, 1}:
        return CorrectnessEstimator.fit(matrix, labels, split=TRAIN_SPLIT, seed=seed,
                                        variant=variant, meta=meta)
    if variant != "B5-RAW":
        raise ValueError(f"the correctness model needs both classes in the OOF rows, got "
                         f"{sorted(classes)} rows={len(rows)}")
    # A single-class OOF set cannot fit a logistic model. B5-RAW does not use one (its score
    # is the reader margin), so the estimator is stored with `fitted=False`: `predict` raises
    # instead of inventing probabilities.
    scaler = FeatureScaler.fit(matrix)
    return CorrectnessEstimator(
        feature_names=CORRECTNESS_FEATURES, scaler=scaler,
        coef=np.zeros(len(CORRECTNESS_FEATURES)), intercept=0.0,
        calibrator=Calibrator(mode="none"), variant="B5-RAW",
        meta={**dict(meta), "model_fit": "unfittable_single_class",
              "observed_classes": sorted(classes)}, fitted=False)


def with_calibrator(estimator: CorrectnessEstimator, calibrator: Calibrator
                    ) -> CorrectnessEstimator:
    return CorrectnessEstimator(feature_names=estimator.feature_names, scaler=estimator.scaler,
                                coef=estimator.coef, intercept=estimator.intercept,
                                calibrator=calibrator, variant=estimator.variant,
                                meta=estimator.meta, fitted=estimator.fitted)


def calibration_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pre-registered reliability decision on the val-calibration rows (spec 08 §4, §5)."""
    n = len(rows)
    positives = int(sum(int(row["y"]) for row in rows))
    plan = calibration_plan(n, positives, n - positives)
    return {"rows": n, "n_positive": positives, "n_negative": n - positives, "plan": plan,
            "calibrated": plan != "raw"}


# ------------------------------------------------------------------------------- B5 policy

def b5_action(obs: Mapping[str, Any], score: float, tau: float) -> int:
    """The deterministic B5 rule (spec 08 §5): no gold, no policy, just confidence + budget.

    ``mask[ANSWER]`` already encodes "candidate with valid provenance is visible", so the
    confidence test is exactly ``score >= tau`` on top of the legal-action mask.
    """
    mask = np.asarray(obs["action_mask"], dtype=bool)
    if mask[ANSWER] and score >= tau:
        return ANSWER
    if mask[SEARCH_BM25]:
        return SEARCH_BM25
    if mask[SEARCH_DENSE]:
        return SEARCH_DENSE
    return ABSTAIN


def simulate_episode(env: AdaptiveQAEnv, world: EpisodeWorld, scores: Mapping[str, float],
                     tau: float) -> dict[str, Any]:
    """One deterministic B5 episode over cached dynamics; returns the return and the trace."""
    obs, _ = env.reset(seed=0, options={"qid": world.qid})
    total = 0.0
    actions: list[str] = []
    outcome: dict[str, Any] = {"qid": world.qid, "reward": 0.0, "answered": False,
                               "terminal_action": None, "candidate_text": None,
                               "provenance_ok": False, "n_searches": 0}
    while True:
        state_hash = env.state_hash()
        if state_hash not in scores:
            raise KeyError(f"state {state_hash[:16]}... of qid={world.qid!r} was not scored; "
                           "the score table must cover every reachable state")
        action = b5_action(obs, scores[state_hash], tau)
        if action in (SEARCH_BM25, SEARCH_DENSE):
            outcome["n_searches"] += 1
        candidate = obs.get("candidate") or {}
        obs, reward, terminated, truncated, _ = env.step(action)
        actions.append(str(ACTIONS[action]))
        total += float(reward)
        if action == ANSWER:
            seen = tuple(obs.get("seen_doc_ids", ()))
            outcome.update({"answered": True, "candidate_text": candidate.get("text"),
                            "provenance_ok": bool(candidate.get("provenance_ok"))
                            and candidate.get("doc_id") in seen})
        if terminated or truncated:
            break
    outcome["reward"] = total
    outcome["terminal_action"] = actions[-1]
    outcome["actions"] = actions
    return outcome


def tau_candidates(scores: Sequence[float], *, n_quantiles: int = DEFAULT_TAU_QUANTILES
                   ) -> list[float]:
    """Gold-free candidate grid: score quantiles plus a never-answer sentinel."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot build a tau grid from no scores")
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_quantiles + 1)))
    sentinel = float(values.max()) + 1.0
    return [float(q) for q in quantiles] + [sentinel]


def select_tau(candidates: Sequence[float], evaluate: Callable[[float], dict[str, Any]]
               ) -> dict[str, Any]:
    """Apply the pre-registered criterion: max mean reward, ties to the higher tau."""
    log = []
    for tau in candidates:
        row = dict(evaluate(float(tau)))
        row["tau"] = float(tau)
        log.append(row)
    ordered = sorted(log, key=lambda row: (row["mean_episode_reward"], row["tau"]))
    best = ordered[-1]
    sentinel = max(float(tau) for tau in candidates)
    return {"criterion": PRE_REGISTERED_CRITERION, "selected_tau": float(best["tau"]),
            "selected": best, "log": log,
            "selected_is_never_answer": bool(abs(float(best["tau"]) - sentinel) < 1e-12)}


def evaluate_tau(env: AdaptiveQAEnv, world_list: Sequence[EpisodeWorld],
                 scores: Mapping[str, float], tau: float,
                 reward: RewardCalculator) -> dict[str, Any]:
    """Mean episode reward / coverage of B5 at ``tau`` on one val world set."""
    returns, answered, correct = [], 0, 0
    for world in world_list:
        outcome = simulate_episode(env, world, scores, tau)
        returns.append(outcome["reward"])
        if outcome["answered"]:
            answered += 1
            y, _ = candidate_label(reward, candidate_text=outcome["candidate_text"],
                                   provenance_ok=outcome["provenance_ok"], private=world.private)
            correct += int(y)
    n = len(world_list)
    return {"mean_episode_reward": float(np.mean(returns)) if returns else 0.0,
            "n_episodes": n, "n_answered": answered,
            "coverage": (answered / n) if n else 0.0,
            "selective_risk": (1.0 - correct / answered) if answered else None}


# ------------------------------------------------------------------------------ reporting

def risk_coverage_curve(scores: Sequence[float], labels: Sequence[int], *,
                        points: int = 20) -> list[dict[str, float]]:
    """Ranking-based selective prediction curve (states ranked by score, gold used to measure)."""
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    y = np.asarray(labels, dtype=np.float64)
    n = int(y.size)
    steps = sorted({max(1, int(round(k))) for k in np.linspace(1, n, points)})
    return [{"coverage": k / n, "risk": float(1.0 - y[order[:k]].mean())} for k in steps]


def subgroup_stats(rows: Sequence[Mapping[str, Any]],
                   scores: Mapping[str, float]) -> list[dict[str, Any]]:
    buckets = {"all": lambda row: True,
               "answerable": lambda row: not row["is_impossible"],
               "unanswerable": lambda row: row["is_impossible"],
               "provenance_ok": lambda row: bool(row["provenance_ok"]),
               "provenance_invalid": lambda row: not row["provenance_ok"]}
    out = []
    for name, predicate in buckets.items():
        kept = [row for row in rows if predicate(row)]
        if not kept:
            out.append({"subgroup": name, "n": 0, "n_positive": 0, "positive_rate": None,
                        "mean_score": None})
            continue
        values = [scores[row["state_hash"]] for row in kept]
        out.append({"subgroup": name, "n": len(kept),
                    "n_positive": int(sum(int(row["y"]) for row in kept)),
                    "positive_rate": float(np.mean([row["y"] for row in kept])),
                    "mean_score": float(np.mean(values))})
    return out


def render_report(*, summary: Mapping[str, Any], reliability: Mapping[str, Any] | None,
                  risk_at_coverage: Mapping[str, Any] | None, subgroups: Sequence[Mapping],
                  curve: Sequence[Mapping[str, float]]) -> str:
    def fmt(value: Any, digits: int = 4) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    variant = summary["variant"]
    lines = [
        f"# B5 correctness calibration — {variant}",
        "",
        f"- run: `{summary['run_id']}` (profile `{summary['profile']}`, seed {summary['seed']})",
        f"- variant: **{variant}** (calibrated probabilities: "
        f"{'yes' if summary['is_calibrated'] else 'NO — see the decision below'})",
        f"- deployed reader: `{summary['reader']['deployed']['reader_model_hash'][:16]}…` "
        "(P3 full-train reader; frozen for every baseline)",
        f"- OOF folds: {summary['training']['oof_folds']} "
        f"({summary['oof_reader']['n_folds']} fold reader(s) trained for this run)",
        f"- report generated: {summary['created']}",
        "",
        "## Pre-registered threshold criterion",
        "",
        f"> {PRE_REGISTERED_CRITERION}",
        "",
        f"Selected `tau_answer = {fmt(summary['tau_answer'])}` on `val_model_selection` "
        f"({summary['threshold_selection']['selected']['n_episodes']} episodes, "
        f"coverage {fmt(summary['threshold_selection']['selected']['coverage'])}, "
        f"mean episode reward "
        f"{fmt(summary['threshold_selection']['selected']['mean_episode_reward'])}).",
        "",
    ]
    if summary["threshold_selection"]["selected_is_never_answer"]:
        lines += ["> WARNING: the never-answer sentinel won the selection; the reported tau "
                  "makes B5 abstain on this validation split.", ""]
    lines += [
        "## Data",
        "",
        f"- training rows (out-of-fold, `train_strict` only): "
        f"{summary['training']['rows']} states with a non-empty candidate "
        f"({summary['training']['n_positive']} correct / "
        f"{summary['training']['n_negative']} incorrect)",
        f"- excluded ambiguous negatives: {summary['training']['excluded_ambiguous']} "
        "(no defensible gold label; excluded from the fit only)",
        f"- `val_calibration`: {summary['calibration']['rows']} states "
        f"({summary['calibration']['n_positive']} positive), "
        f"`val_model_selection`: {summary['selection_rows']} states",
        "- `test_strict` was not read at any point of this stage.",
        "",
        "## Calibration decision",
        "",
        f"`correctness.calibration_plan` on the val-calibration rows chose "
        f"**{summary['calibration']['plan']}** "
        f"(rows={summary['calibration']['rows']}, positives="
        f"{summary['calibration']['n_positive']}, negatives="
        f"{summary['calibration']['n_negative']}).",
        "",
    ]
    if summary["calibration"]["plan"] == "raw":
        lines += [
            f"Too few / too imbalanced calibration rows for a trustworthy calibrator, so this "
            f"run deploys **B5-RAW**: the ANSWER threshold is applied to the reader's "
            f"`null_margin`, which is *not* a probability. Brier/ECE are reported as `N/A` "
            f"instead of being computed on a raw score (spec 08 §5, §8).",
            "",
        ]
    else:
        lines += [
            f"Calibrator fitted on `val_calibration` only: mode "
            f"`{summary['calibration']['mode']}`. Reported below on the held-out "
            f"`val_model_selection` rows.",
            "",
        ]
    if reliability is None:
        lines += ["## Reliability (Brier / ECE)", "",
                  "- `N/A` — B5-RAW has no calibrated probability to score.", ""]
    else:
        lines += ["## Reliability (Brier / ECE) — val_model_selection", "",
                  f"- Brier: {fmt(reliability['brier'])}", f"- ECE (10 bins, L1): "
                  f"{fmt(reliability['ece'])}", f"- rows: {reliability['n']}", "",
                  "| bin | count | confidence | accuracy |", "| --- | --- | --- | --- |"]
        for b in reliability["bins"]:
            lines.append(f"| [{fmt(b['lo'], 2)}, {fmt(b['hi'], 2)}) | {b['count']} | "
                         f"{fmt(b['confidence'])} | {fmt(b['accuracy'])} |")
        lines.append("")
    lines += ["## Selective prediction", ""]
    if risk_at_coverage is not None:
        lines += [f"- risk at {int(RISK_AT_COVERAGE * 100)}% coverage (calibrated ranking, "
                  f"spec 03 §1): {fmt(risk_at_coverage['risk_at_coverage'])} over "
                  f"{risk_at_coverage['k']}/{risk_at_coverage['n']} states"]
    else:
        lines += [f"- risk at {int(RISK_AT_COVERAGE * 100)}% coverage: `N/A` — requires "
                  "calibrated confidence (B5-RAW ranks by a raw margin)"]
    selected = summary["threshold_selection"]["selected"]
    lines += [f"- B5 at `tau={fmt(summary['tau_answer'])}`: coverage "
              f"{fmt(selected['coverage'])}, selective risk "
              f"{fmt(selected['selective_risk'])} (answered {selected['n_answered']}/"
              f"{selected['n_episodes']} episodes)",
              f"- mean episode reward {fmt(selected['mean_episode_reward'])} "
              f"[{fmt(summary['selection_ci']['ci_low'])}; "
              f"{fmt(summary['selection_ci']['ci_high'])}] "
              f"(cluster bootstrap over {summary['selection_ci']['n_groups']} title groups)",
              "", "| coverage | risk |", "| --- | --- |"]
    for point in curve:
        lines.append(f"| {fmt(point['coverage'], 2)} | {fmt(point['risk'])} |")
    lines += ["", "## Subgroups (val_model_selection states)", "",
              "| subgroup | n | positives | positive rate | mean score |",
              "| --- | --- | --- | --- | --- |"]
    for row in subgroups:
        lines.append(f"| {row['subgroup']} | {row['n']} | {row['n_positive']} | "
                     f"{fmt(row['positive_rate'])} | {fmt(row['mean_score'])} |")
    lines += ["", "## Threshold selection log", "",
              "| tau | mean episode reward | coverage | answered | selective risk |",
              "| --- | --- | --- | --- | --- |"]
    for row in summary["threshold_selection"]["log"]:
        lines.append(f"| {fmt(row['tau'])} | {fmt(row['mean_episode_reward'])} | "
                     f"{fmt(row['coverage'])} | {row['n_answered']} | "
                     f"{fmt(row['selective_risk'])} |")
    lines += ["", "## Caveats", ""]
    lines += [f"- {note}" for note in summary["caveats"]]
    lines += ["", f"- results row: `{summary['results_row']['path']}` "
                  f"({summary['results_row']['mode']})", ""]
    return "\n".join(lines)


def write_plot(path: Path, *, variant: str, scores: Sequence[float], labels: Sequence[int],
               tau: float, reliability: Mapping[str, Any] | None,
               operating_point: Mapping[str, Any],
               curve: Sequence[Mapping[str, float]]) -> Path:
    """Reliability (or margin-accuracy) plus risk-coverage, one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))

    if reliability is not None:
        bins = [b for b in reliability["bins"] if b["count"]]
        conf = [b["confidence"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        left.plot([0, 1], [0, 1], linestyle=":", color="grey", label="perfect calibration")
        left.plot(conf, acc, marker="o", color="tab:blue", label="p_correct bins")
        left.set_xlabel("predicted P(correct)")
        left.set_ylabel("observed correctness")
        left.set_title(f"Reliability — B5 (ECE {reliability['ece']:.3f}, "
                       f"Brier {reliability['brier']:.3f})")
    else:
        order = np.argsort(scores)
        n = scores.size
        chunks = np.array_split(order, min(10, max(1, n)))
        xs = [float(scores[chunk].mean()) for chunk in chunks if chunk.size]
        ys = [float(y[chunk].mean()) for chunk in chunks if chunk.size]
        left.plot(xs, ys, marker="o", color="tab:red", label="reader null_margin bins")
        left.axvline(float(tau), color="black", linestyle="--",
                     label=f"tau_answer = {tau:.4g}")
        left.set_xlabel("reader null_margin (NOT a probability)")
        left.set_ylabel("observed correctness")
        left.set_title("B5-RAW: margin vs correctness (uncalibrated)")
    left.legend(fontsize=8)
    left.grid(alpha=0.25)

    xs = [point["coverage"] for point in curve]
    ys = [point["risk"] for point in curve]
    right.plot(xs, ys, color="tab:green", label="ranking by score (state level)")
    if operating_point.get("coverage") is not None:
        right.scatter([operating_point["coverage"]], [operating_point["risk"] or 0.0],
                      color="black", zorder=3, label="B5 at tau (episode level)")
    right.set_xlabel("coverage")
    right.set_ylabel("selective risk")
    right.set_title("Risk–coverage")
    right.legend(fontsize=8)
    right.grid(alpha=0.25)
    fig.suptitle(f"B5 correctness confidence — {variant}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def results_row(summary: Mapping[str, Any], reliability: Mapping[str, Any] | None,
                risk_at_coverage: Mapping[str, Any] | None) -> dict[str, Any]:
    """The canonical B5 result row (results.csv columns, spec 03 §1)."""
    selected = summary["threshold_selection"]["selected"]
    return {
        "baseline": "B5", "variant": summary["variant"], "run_id": summary["run_id"],
        "profile": summary["profile"], "seed": summary["seed"],
        "split": "val_model_selection", "n_episodes": selected["n_episodes"],
        "mean_episode_reward": selected["mean_episode_reward"],
        "reward_ci_low": summary["selection_ci"]["ci_low"],
        "reward_ci_high": summary["selection_ci"]["ci_high"],
        "coverage": selected["coverage"], "selective_risk": selected["selective_risk"],
        "risk_at_80pct_coverage": (risk_at_coverage or {}).get("risk_at_coverage"),
        "brier": (reliability or {}).get("brier"), "ece": (reliability or {}).get("ece"),
        "tau_answer": summary["tau_answer"], "score": summary["score"],
        "reader_model_hash": summary["reader"]["deployed"]["reader_model_hash"],
        "estimator_dir": summary["estimator_dir"], "report": summary["report"],
    }


def write_results_row(cfg: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Append to ``reports/results.csv`` when it exists, else write the standalone row JSON."""
    reports = resolve(cfg, "reports")
    csv_path = reports / RESULTS_CSV_FILE
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        if header:
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
                writer.writerow({key: _csv_value(row.get(key)) for key in header})
            return {"path": str(csv_path), "mode": "appended_to_results_csv"}
    json_path = reports / RESULTS_ROW_FILE
    atomic_write_json(json_path, {
        "note": ("reports/results.csv did not exist when this row was produced (the reporting "
                 "stage owns that file); this is the canonical B5 row to merge."),
        "row": dict(row), "columns": sorted(row)})
    return {"path": str(json_path), "mode": "results_b5_row_json"}


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return "" if value is None else value


# ----------------------------------------------------------------------------------- run

def title_groups_for(cfg: Mapping[str, Any], split: str, qids: Sequence[str]) -> dict[str, str]:
    """``qid -> title_group`` restricted to the qids this run actually uses."""
    groups = splitting.title_groups(cfg, split)
    missing = sorted(set(qids) - set(groups))
    if missing:
        raise KeyError(f"{len(missing)} {split} question(s) have no label row, e.g. "
                       f"{missing[:3]}; run the P1 audit first")
    return {str(qid): groups[str(qid)] for qid in qids}


def run(*, cfg: Mapping[str, Any], profile: str, oof_folds: int = 1, seed: int | None = None,
        limit: int | None = None, dry_run: bool = False,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    """Train the B5 correctness estimator and select ``tau_answer``."""
    overrides = correctness_cfg(cfg)
    oof_holdout_fraction = float(overrides.get("oof_holdout_fraction",
                                               DEFAULT_OOF_HOLDOUT_FRACTION))
    calibration_fraction = float(overrides.get("calibration_fraction",
                                               DEFAULT_CALIBRATION_FRACTION))
    seed = int(cfg["seed"] if seed is None else seed)
    if oof_folds < 1:
        raise ValueError(f"oof_folds must be >= 1, got {oof_folds}")
    out_dir = estimator_dir(cfg)
    run_id = f"b5-correctness-{profile}-{seed}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile, seed=seed,
                    oof_folds=oof_folds, limit=limit, dry_run=dry_run)
    try:
        cfg_sha = snapshot_config(dict(cfg),
                                  resolve(cfg, "manifests") / f"B5_correctness_{profile}_config.yaml")
        reward = RewardCalculator(RewardConfig.from_dict(dict(cfg)))
        train_bundle = worlds.build_worlds(cfg, TRAIN_SPLIT, limit=limit, laya_regime=None)
        val_bundle = worlds.build_worlds(cfg, VAL_SPLIT, limit=limit, laya_regime=None)
        train_groups = title_groups_for(cfg, TRAIN_SPLIT, [w.qid for w in train_bundle.worlds])
        val_groups = title_groups_for(cfg, VAL_SPLIT, [w.qid for w in val_bundle.worlds])
        specs = splitting.holdout_specs(train_groups, oof_folds, seed=seed,
                                        fraction=oof_holdout_fraction)

        if dry_run:
            probe = _dry_run_probe(cfg, train_bundle=train_bundle, val_bundle=val_bundle,
                                   train_groups=train_groups, val_groups=val_groups,
                                   specs=specs, reward=reward, seed=seed,
                                   calibration_fraction=calibration_fraction)
            summary = {"run_id": run_id, "stage": STAGE, "dry_run": True, "profile": profile,
                       "seed": seed, "oof_folds": oof_folds, "limit": limit,
                       "config_sha": cfg_sha, "probe": probe,
                       "note": "no estimator, report or plot was written"}
            registry.append(run_id=run_id, stage=STAGE, status="SUCCEEDED", profile=profile,
                            dry_run=True, probe=probe, note="no model or report written")
            return summary

        # 1. OOF features: a fold reader re-trained through the P3 code path predicts the
        #    questions the estimator is allowed to train on.
        oof_rows, fold_manifest, fold_sources = [], [], []
        failed_folds: list[str] = []
        for index, (reader_train_qids, feature_qids) in enumerate(specs):
            try:
                source, manifest = train_fold_reader_source(
                    cfg, profile=profile, seed=seed, reader_train_qids=reader_train_qids,
                    feature_qids=feature_qids, out_dir=out_dir / OOF_READER_DIR / f"fold_{index}",
                    fold_index=index, limit=limit)
            except ValueError as exc:
                # D010: a fold whose threshold sweep meets neither §10 acceptance constraint
                # (or that otherwise cannot produce a reader) has no usable predictions;
                # record it as a FAILED fold and let the remaining folds carry the fit.
                fold_manifest.append({"index": int(index), "status": "FAILED",
                                      "error": str(exc)})
                failed_folds.append(f"fold {index}: {exc}")
                continue
            selected, env = world_set(cfg, train_bundle, source, qids=feature_qids,
                                      exclude=sorted(train_bundle.ambiguous_qids))
            rows = labeled_rows(reachable_states(env, selected), reward)
            manifest["n_feature_rows"] = len(rows)
            fold_manifest.append(manifest)
            fold_sources.append(source.describe())
            oof_rows.extend(rows)
        if not oof_rows:
            if failed_folds:
                raise ValueError(
                    f"no fold reader could be produced ({len(failed_folds)} of {len(specs)} "
                    "fold(s) failed): " + "; ".join(failed_folds))
            raise ValueError(
                "no out-of-fold state had a non-empty candidate; there is nothing to train the "
                "correctness estimator on (check the fold reader's threshold)")
        excluded_ambiguous = len({w.qid for w in train_bundle.worlds
                                  if w.qid in train_bundle.ambiguous_qids})

        # 2. Deployed reader on validation; split val by title groups (spec 08 §3).
        val_source = p3_reader_source(cfg, VAL_SPLIT)
        calibration_qids, selection_qids = splitting.group_holdout(
            val_groups, fraction=calibration_fraction, seed=seed)
        cal_worlds, cal_env = world_set(cfg, val_bundle, val_source, qids=calibration_qids)
        sel_worlds, sel_env = world_set(cfg, val_bundle, val_source, qids=selection_qids)
        cal_rows = labeled_rows(reachable_states(cal_env, cal_worlds), reward)
        sel_states = reachable_states(sel_env, sel_worlds)
        sel_rows = labeled_rows(sel_states, reward)
        if not cal_rows or not sel_rows:
            raise ValueError(f"val split left an empty side (calibration={len(cal_rows)} rows, "
                             f"selection={len(sel_rows)} rows); adjust correctness."
                             "calibration_fraction or the profile scale")

        # 3. Fit scaler + model on OOF rows, decide, calibrate on val_calibration only.
        decision = calibration_decision(cal_rows)
        variant = "B5" if decision["calibrated"] else "B5-RAW"
        estimator = fit_estimator(oof_rows, seed=seed, variant=variant, meta={
            "oof_folds": oof_folds, "seed": seed,
            "train_rows": len(oof_rows), "train_questions": len({r["qid"] for r in oof_rows}),
            "calibration_plan": decision["plan"], "criterion": PRE_REGISTERED_CRITERION})
        calibrator = Calibrator(mode="none")
        if decision["calibrated"]:
            calibrator = fit_calibrator(
                scores=estimator.predict(feature_matrix(cal_rows)),
                labels=label_vector(cal_rows), split=VAL_SPLIT, plan=decision["plan"])
            estimator = with_calibrator(estimator, calibrator)
        decision["mode"] = estimator.calibrator.mode
        decision["variant"] = variant
        decision["estimator_fitted"] = bool(estimator.fitted)
        if not estimator.fitted:
            decision["reason"] = ("a single-class OOF set cannot fit a logistic model; B5-RAW "
                                  "uses the reader null_margin")
        elif decision["plan"] == "raw":
            decision["reason"] = (f"calibration needs >= {_min_cal_n()} rows with >= "
                                  f"{_min_cal_class()} per class; got {decision['rows']} "
                                  f"({decision['n_positive']}+/{decision['n_negative']}-)")

        # 4. tau_answer on val_model_selection by the pre-registered criterion.
        scores = score_map(sel_states, estimator.probability_from_obs)
        candidates = tau_candidates(list(scores.values()),
                                    n_quantiles=int(overrides.get("tau_quantiles",
                                                                  DEFAULT_TAU_QUANTILES)))
        selection = select_tau(candidates, lambda tau: evaluate_tau(sel_env, sel_worlds, scores,
                                                                   tau, reward))
        tau = float(selection["selected_tau"])
        per_qid = {world.qid: simulate_episode(sel_env, world, scores, tau)["reward"]
                   for world in sel_worlds}
        selection_ci = splitting.cluster_bootstrap_ci(
            per_qid, {qid: val_groups[qid] for qid in per_qid},
            n_resamples=int(overrides.get("bootstrap_resamples", DEFAULT_BOOTSTRAP_RESAMPLES)),
            seed=seed)

        # 5. Metrics that may only be computed on held-out selection rows.
        sel_scores = [scores[row["state_hash"]] for row in sel_rows]
        sel_labels = [int(row["y"]) for row in sel_rows]
        reliability = selection_metrics_at = None
        if estimator.is_calibrated:
            reliability = reliability_metrics(sel_scores, sel_labels)
            selection_metrics_at = selection_metrics(sel_scores, sel_labels,
                                                     coverage_fraction=RISK_AT_COVERAGE)
        curve = risk_coverage_curve(sel_scores, sel_labels)
        subgroups = subgroup_stats(sel_rows, scores)

        # 6. Artifacts.
        estimator.save(out_dir)
        report_path = resolve(cfg, "reports") / REPORT_FILE
        plot_path = resolve(cfg, "plots") / PLOT_FILE
        summary = {
            "run_id": run_id, "stage": STAGE, "profile": profile, "seed": seed,
            "variant": variant, "is_calibrated": estimator.is_calibrated,
            "score": "p_correct" if estimator.is_calibrated else "reader_null_margin",
            "oof_folds": oof_folds, "limit": limit, "created": now_iso(),
            "pre_registered_criterion": PRE_REGISTERED_CRITERION,
            "tau_answer": tau, "threshold_selection": selection,
            "selection_ci": selection_ci, "selection_rows": len(sel_rows),
            "calibration": decision,
            "training": {
                "rows": len(oof_rows), "n_positive": int(sum(r["y"] for r in oof_rows)),
                "n_negative": int(len(oof_rows) - sum(r["y"] for r in oof_rows)),
                "questions": len({r["qid"] for r in oof_rows}),
                "train_qid_digest": sha256_json(sorted({w.qid for w in train_bundle.worlds})),
                "feature_qid_digest": sha256_json(sorted({r["qid"] for r in oof_rows})),
                "state_hash_digest": sha256_json(sorted(r["state_hash"] for r in oof_rows)),
                "excluded_ambiguous": excluded_ambiguous, "oof_folds": oof_folds,
                "fold_manifest": fold_manifest,
            },
            "oof_reader": {"n_folds": len(fold_sources), "sources": fold_sources},
            "reader": {"deployed": val_source.describe(),
                       "train_cache": p3_reader_source(cfg, TRAIN_SPLIT).describe()},
            "splits": {"train_worlds": len(train_bundle.worlds),
                       "val_worlds": len(val_bundle.worlds),
                       "val_calibration_questions": len(calibration_qids),
                       "val_model_selection_questions": len(selection_qids),
                       "ambiguous_negatives_flagged": len(train_bundle.ambiguous_qids)},
            "features": list(CORRECTNESS_FEATURES),
            "coef": estimator.coef.tolist(), "intercept": estimator.intercept,
            "laya_regime": None, "config_sha": cfg_sha,
            "git_sha": git_sha(REPO_ROOT),
            "retrieval_config_sha": build_retrieval.retrieval_config_sha(dict(cfg)),
            "env_config": _env_config_dict(cfg),
            "estimator_dir": str(out_dir), "report": str(report_path),
            "plot": str(plot_path),
            "library_versions": capture_library_versions(),
            "caveats": _caveats(oof_folds, decision, estimator, train_bundle, val_bundle),
        }
        manifest = {"oof_folds": oof_folds, "seed": seed,
                    "oof_holdout_fraction": oof_holdout_fraction,
                    "calibration_fraction": calibration_fraction,
                    "grouping": "title_group (P1), group-disjoint by construction",
                    "folds": fold_manifest,
                    "val_calibration_qids": sorted(calibration_qids),
                    "val_model_selection_qids": sorted(selection_qids)}
        manifest_path = out_dir / SPLIT_MANIFEST_FILE
        atomic_write_json(manifest_path, manifest)
        provenance_path = out_dir / PROVENANCE_FILE
        atomic_write_json(provenance_path, summary)
        threshold_path = out_dir / THRESHOLD_FILE
        atomic_write_json(threshold_path, {
            "criterion": PRE_REGISTERED_CRITERION, "variant": variant,
            "score": summary["score"], "tau_answer": tau,
            "selected": selection["selected"], "log": selection["log"],
            "selected_is_never_answer": selection["selected_is_never_answer"],
            "split": "val_model_selection", "n_episodes": len(sel_worlds)})
        row = results_row(summary, reliability, selection_metrics_at)
        summary["results_row"] = write_results_row(cfg, row)
        report_path.write_text(render_report(summary=summary, reliability=reliability,
                                            risk_at_coverage=selection_metrics_at,
                                            subgroups=subgroups, curve=curve),
                               encoding="utf-8")
        write_plot(plot_path, variant=variant, scores=sel_scores, labels=sel_labels, tau=tau,
                   reliability=reliability,
                   operating_point={"coverage": selection["selected"]["coverage"],
                                    "risk": selection["selected"]["selective_risk"]},
                   curve=curve)
        summary.update({"results": row, "provenance": str(provenance_path),
                        "split_manifest": str(manifest_path),
                        "threshold_selection_file": str(threshold_path)})
        artifacts_written = [out_dir / "model.json", out_dir / "scaler.json",
                             out_dir / "calibrator.json", provenance_path, manifest_path,
                             threshold_path, report_path, plot_path,
                             Path(summary["results_row"]["path"]), out_dir / OOF_READER_DIR]
        registry.record_artifact(run_id, STAGE, artifacts_written, profile=profile, seed=seed,
                                 variant=variant, tau_answer=tau,
                                 oof_folds=oof_folds, is_calibrated=estimator.is_calibrated,
                                 mean_episode_reward=selection["selected"]["mean_episode_reward"],
                                 coverage=selection["selected"]["coverage"])
        return summary
    except Exception as exc:
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise


def _min_cal_n() -> int:
    from src.eval import correctness
    return int(correctness.MIN_CAL_N)


def _min_cal_class() -> int:
    from src.eval import correctness
    return int(correctness.MIN_CAL_CLASS)


def _env_config_dict(cfg: Mapping[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict
    return asdict(worlds.env_config(cfg))


def _caveats(oof_folds: int, decision: Mapping[str, Any], estimator: CorrectnessEstimator,
             train_bundle: worlds.WorldBundle, val_bundle: worlds.WorldBundle) -> list[str]:
    caveats = [
        f"Reader optimism: fitting rows come from readers trained without their feature "
        f"questions ({oof_folds} OOF fold(s)); the deployed reader is the P3 full-train "
        f"reader, so its predictions on those questions are never used as training labels.",
        "The deployed reader's null threshold was tuned by P3 on all of `val_strict`; "
        "`val_model_selection` therefore inherits that single scalar. Calibration and tau "
        "selection below are still fitted/evaluated only on their own val subsets.",
        f"{len(train_bundle.ambiguous_qids)} train question(s) are flagged ambiguous "
        "negatives; their states are excluded from the fit because their gold is not "
        "defensible (same rule as the P6 training reward), but they stay in validation.",
        "Subgroup rows use gold only for measurement; no gold enters any feature.",
    ]
    if decision["plan"] == "raw" or not estimator.fitted:
        caveats.append("B5-RAW deploys a reader-margin threshold; treat `tau_answer` as a "
                       "reader score, never as a probability.")
    if val_bundle.ambiguous_qids:
        caveats.append(f"{len(val_bundle.ambiguous_qids)} val question(s) are ambiguous "
                       "negatives and are kept in validation by design.")
    return caveats


def _dry_run_probe(cfg: Mapping[str, Any], *, train_bundle: worlds.WorldBundle,
                   val_bundle: worlds.WorldBundle, train_groups: Mapping[str, str],
                   val_groups: Mapping[str, str], specs: Sequence[tuple[list[str], list[str]]],
                   reward: RewardCalculator, seed: int,
                   calibration_fraction: float) -> dict[str, Any]:
    """Cheap wiring check: counts and label balance, no fold reader is trained or written."""
    source = p3_reader_source(cfg, TRAIN_SPLIT)
    train_worlds, train_env = world_set(cfg, train_bundle, source)
    rows = labeled_rows(reachable_states(train_env, train_worlds), reward)
    cal_qids, sel_qids = splitting.group_holdout(dict(val_groups), fraction=calibration_fraction,
                                                 seed=seed)
    return {
        "train_worlds": len(train_bundle.worlds), "val_worlds": len(val_bundle.worlds),
        "train_title_groups": len(set(train_groups.values())),
        "val_title_groups": len(set(val_groups.values())),
        "oof_folds": len(specs),
        "fold_feature_questions": [len(feature) for _, feature in specs],
        "fold_reader_train_questions": [len(reader) for reader, _ in specs],
        "val_calibration_questions": len(cal_qids),
        "val_model_selection_questions": len(sel_qids),
        "probe_states": len(rows),
        "probe_positives": int(sum(row["y"] for row in rows)),
        "probe_note": ("labels probed with the deployed P3 reader on its own training "
                       "questions, so they are optimistic; the real fit uses the OOF folds"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.train_correctness",
        description="B5: train the answer-correctness estimator and select tau_answer")
    add_profile_args(parser)
    parser.add_argument("--oof-folds", type=int, choices=(1, 3), default=1,
                        help="1 = one group-disjoint holdout reader (budget option, default); "
                             "3 = full cross-fitting")
    parser.add_argument("--limit", type=int, default=None,
                        help="cap scenarios per split (must match the P3 state cache build)")
    parser.add_argument("--dry-run", action="store_true",
                        help="check the wiring (worlds, splits, counts); write no estimator")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    summary = run(cfg=cfg, profile=profile, oof_folds=args.oof_folds, seed=args.seed,
                  limit=args.limit, dry_run=args.dry_run)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0
    if summary.get("dry_run"):
        probe = summary["probe"]
        print(f"run_id={summary['run_id']} (dry run) train={probe['train_worlds']} worlds, "
              f"val={probe['val_worlds']} worlds, oof_folds={probe['oof_folds']}, "
              f"fold_feature_questions={probe['fold_feature_questions']}, "
              f"probe_states={probe['probe_states']} ({probe['probe_positives']} positive)")
        return 0
    print(f"run_id={summary['run_id']} variant={summary['variant']} "
          f"tau_answer={summary['tau_answer']:.4f} "
          f"reward={summary['threshold_selection']['selected']['mean_episode_reward']:.4f} "
          f"coverage={summary['threshold_selection']['selected']['coverage']:.3f}")
    print(f"  estimator: {summary['estimator_dir']}")
    print(f"  report: {summary['report']}")
    if summary["threshold_selection"]["selected_is_never_answer"]:
        print("  WARNING: the never-answer sentinel won the threshold selection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

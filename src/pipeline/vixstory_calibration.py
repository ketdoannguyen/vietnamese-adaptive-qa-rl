"""Isolated VixSTORY E03 correctness-threshold policy and train-only calibrator.

Why a separate module
    The ViQuAD B5 stack (:mod:`src.eval.correctness` + :mod:`src.pipeline.train_correctness`)
    is bound to the extractive single-doc reader: its features read ``non_null_score`` and
    ``null_margin`` and its split guard hard-codes ``train_strict`` / ``val_strict``. VixSTORY
    has a free-form reader, multi-gold ids, its own four-action ``VixStoryEnv`` and a different
    protocol (plan ``.claude/plans/vixstory-pilot-300-e01-e07.md`` §3). Reusing that stack would
    silently change the measurement, so this module imports none of it. It touches only the
    isolated environment (:mod:`src.rl.vixstory_env`), the shared standardiser
    (:class:`src.rl.features.FeatureScaler`) and plain JSON I/O.

Protocol this module implements
    * **p(correct) is learned from train300 only.** Features are a *fixed* vector read from the
      observation alone (``candidate.{answer_present,citation_valid,text,cited_ids}``, the seen
      ids, the search budget, the step counter and the retrieval preview). No gold answer, no
      gold id, no hop and no oracle action ever enters a feature — and no extractive score is
      invented: the free-form reader produces no ``null_margin``, so none is fabricated.
    * **Labels come from the private outcome**, and only for fitting. A state whose reader
      produced no candidate is not a training row at all (ANSWER is illegal there).
    * **Cross-fitting on train300** gives honest out-of-fold ``p(correct)`` used to fit the
      calibrator; the deployed model is refit on all train rows. Where cross-fitting is not
      practical (too few questions, a single-class fold) it is skipped and the reason recorded.
    * **Threshold selection is on dev200 by real reward.** ``tau`` is chosen by replaying the
      deterministic E03 rule (ANSWER iff the mask allows it *and* ``p_correct >= tau``; else the
      next valid SEARCH; else ABSTAIN) through the environment and maximising the mean *episode
      return the environment actually pays* — the reward already charges SEARCH costs. Gold is
      used to *measure* correctness, never to choose an action: at inference the policy reads the
      observation, ``p_correct`` and ``tau`` only.
    * **The official validation split is never touched.** No function here accepts a
      ``validation`` argument; the pilot scores the 246-question split once, elsewhere, after the
      artefacts are frozen.
    * **Fitting is skipped, never faked.** With no rows, or with constant labels, no logistic
      model is fit; the run reports the blocker and (for a constant label) a constant probability
      flagged as such.

Artifacts
    :meth:`VixStoryCorrectnessEstimator.save` writes JSON only (no pickle, so no untrusted-object
    loading) and stamps an ``identity`` over the schema, the feature list and the reward
    criterion. :meth:`VixStoryCorrectnessEstimator.load` refuses an artifact whose feature list,
    schema or identity does not match, so a calibrator fit under one reward configuration cannot
    be silently reused under another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.qa.metrics import tokenize
from src.rl.features import FeatureScaler
from src.rl.vixstory_env import (
    ABSTAIN,
    ANSWER,
    N_ACTIONS,
    SEARCH_BM25,
    SEARCH_DENSE,
    VixStoryEnv,
    VixStoryEnvConfig,
)
from src.utils.hashing import sha256_json
from src.utils.io import atomic_write_json, read_json

#: Version tag written into every artifact; a mismatch refuses the load.
SCHEMA = "vixstory-e03-calibration/v1"

#: Observation-safe feature list, in column order. Reading any field the free-form reader does
#: not produce (a span score, a null margin) would fabricate calibration — none is listed.
VIXSTORY_CORRECTNESS_FEATURES: tuple[str, ...] = (
    "candidate_answer_present",
    "candidate_citation_valid",
    "candidate_n_cited",
    "candidate_token_len",
    "candidate_char_len",
    "seen_count",
    "optional_used",
    "remaining_optional",
    "timestep",
    "mask_bm25",
    "mask_dense",
    "bm25_top_score",
    "bm25_top_gap",
    "dense_top_score",
    "dense_top_gap",
    "question_token_len",
    "question_char_len",
)

#: Deterministic SEARCH fallback order of the E03 rule (pre-registered, never tuned).
SEARCH_FALLBACK: tuple[int, ...] = (SEARCH_BM25, SEARCH_DENSE)

DEFAULT_N_FOLDS = 3
DEFAULT_TAU_QUANTILES = 20
#: Cross-fit calibrator is only fit with enough out-of-fold rows and both classes present.
MIN_CAL_N = 40
MIN_CAL_CLASS = 8

#: Sentinels for legitimately missing observation values.
MISSING_COUNT = -1.0

#: Reward-relevant configuration the artifact identity is bound to. A stored calibrator is
#: refused unless the environment it was fit for carries exactly these values.
REWARD_CRITERION_KEYS: tuple[str, ...] = (
    "f1_threshold", "citation_gate", "search_cost", "max_optional_search",
    "bootstrap_tool", "bootstrap_passages", "passage_per_search_action",
)

#: **Pre-registered** threshold-selection criterion, written verbatim into every artifact.
PRE_REGISTERED_CRITERION = (
    "tau = argmax over the candidate grid of mean episode reward on dev200, replaying the "
    "deterministic E03 rule (ANSWER iff the action mask allows it and p_correct >= tau; else "
    "SEARCH_BM25, then SEARCH_DENSE while the budget remains; else ABSTAIN) through "
    "VixStoryEnv; ties are broken toward the higher tau; the never-answer sentinel is flagged "
    "if it wins. p_correct comes from the observation-only estimator; gold is used to measure "
    "correctness, never to choose an action"
)


class VixStoryCalibrationError(RuntimeError):
    """A calibration precondition failed (e.g. a scorer table missing a visited state)."""


# --------------------------------------------------------------------------- small helpers

def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def reward_criterion(config: VixStoryEnvConfig) -> dict[str, Any]:
    """The reward-defining subset of an environment config, used for the artifact identity."""
    return {key: getattr(config, key) for key in REWARD_CRITERION_KEYS}


def artifact_identity(criterion: Mapping[str, Any]) -> str:
    """Stable hash binding an artifact to the feature list, schema and reward criterion."""
    return sha256_json({"schema": SCHEMA, "feature_names": list(VIXSTORY_CORRECTNESS_FEATURES),
                        "reward_criterion": dict(criterion)})


def features_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """The estimator's input: visible state only, fixed width, always finite.

    A pure function of ``obs`` — mutating the hidden gold side of an episode cannot change a
    single value, which ``tests/test_vixstory_calibration.py`` asserts.
    """
    candidate = obs.get("candidate") or {}
    retrieval = obs.get("retrieval") or {}
    mask = np.asarray(obs.get("action_mask"), dtype=bool)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"observation action_mask must have shape ({N_ACTIONS},), got {mask.shape}")
    text = candidate.get("text")
    tokens = tokenize(text) if text else []
    question = str(obs.get("question", ""))
    vector = [
        1.0 if candidate.get("answer_present") else 0.0,
        1.0 if candidate.get("citation_valid") else 0.0,
        float(len(candidate.get("cited_ids") or ())),
        float(len(tokens)),
        float(len(text)) if text else 0.0,
        float(len(obs.get("seen_doc_ids", ()))),
        _finite(obs.get("optional_used"), MISSING_COUNT),
        _finite(obs.get("remaining_optional_searches"), MISSING_COUNT),
        _finite(obs.get("timestep"), MISSING_COUNT),
        1.0 if mask[SEARCH_BM25] else 0.0,
        1.0 if mask[SEARCH_DENSE] else 0.0,
        _finite(retrieval.get("bm25_top_score"), 0.0),
        _finite(retrieval.get("bm25_top_gap"), 0.0),
        _finite(retrieval.get("dense_top_score"), 0.0),
        _finite(retrieval.get("dense_top_gap"), 0.0),
        float(len(tokenize(question))),
        float(len(question)),
    ]
    out = np.asarray(vector, dtype=np.float64)
    if out.shape != (len(VIXSTORY_CORRECTNESS_FEATURES),):
        raise AssertionError(f"feature width {out.shape} != {len(VIXSTORY_CORRECTNESS_FEATURES)}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"non-finite correctness feature vector: {out.tolist()}")
    return out


# ------------------------------------------------------------------------------- state walk

@dataclass(frozen=True)
class StateRef:
    """One reachable state, identified by the search history that reaches it."""

    qid: str
    history: tuple[int, ...]
    state_hash: str


@dataclass
class CalibrationRow:
    """One supervised row: observation-only features plus a label from the private outcome."""

    qid: str
    state_hash: str
    y: int
    f1: float
    features: np.ndarray


def replay(env: VixStoryEnv, qid: str, history: Sequence[int]) -> dict[str, Any]:
    """Reset to ``qid`` and step through ``history``; return the resulting observation."""
    obs, _ = env.reset(seed=0, options={"qid": qid})
    for action in history:
        obs, _, terminated, truncated, _ = env.step(int(action))
        if terminated or truncated:
            raise VixStoryCalibrationError(
                f"replay of qid={qid!r} history={list(history)} hit a terminal state early")
    return obs


def enumerate_state_refs(env: VixStoryEnv, qids: Sequence[str]) -> list[StateRef]:
    """Every state a rollout can decide from, reached by BFS over the two SEARCH actions.

    The frontier is driven by the live action mask, so the state set cannot drift from the one
    the environment actually walks (it stops when both search actions are masked).
    """
    refs: list[StateRef] = []
    for qid in qids:
        seen: set[str] = set()
        stack: list[tuple[int, ...]] = [()]
        while stack:
            history = stack.pop()
            obs = replay(env, qid, history)
            state_hash = env.state_hash()
            if state_hash in seen:
                continue
            seen.add(state_hash)
            refs.append(StateRef(qid=qid, history=history, state_hash=state_hash))
            mask = np.asarray(obs["action_mask"], dtype=bool)
            for action in SEARCH_FALLBACK:
                if mask[action]:
                    stack.append(history + (action,))
    return refs


def label_rows(env: VixStoryEnv, refs: Sequence[StateRef]) -> list[CalibrationRow]:
    """Fit rows for every state with a candidate; the label is the env's own outcome.

    Reading ``private_outcome`` *here and only here* is permitted: this is a train-only fit
    label. A state without a candidate contributes no row — ANSWER is illegal there, so there is
    no decision to calibrate.
    """
    rows: list[CalibrationRow] = []
    for ref in refs:
        obs = replay(env, ref.qid, ref.history)
        if not obs["candidate"]["answer_present"]:
            continue
        features = features_from_obs(obs)
        env.step(ANSWER)  # legal whenever a candidate is present
        outcome = env.private_outcome()
        if outcome is None:
            raise VixStoryCalibrationError(f"no private outcome after ANSWER for qid={ref.qid!r}")
        rows.append(CalibrationRow(qid=ref.qid, state_hash=ref.state_hash,
                                   y=int(outcome["correct"]), f1=float(outcome["f1"]),
                                   features=features))
    return rows


def collect_rows(env: VixStoryEnv, qids: Sequence[str]) -> list[CalibrationRow]:
    """All labelled fit rows reachable from ``qids`` in ``env``."""
    return label_rows(env, enumerate_state_refs(env, qids))


def feature_matrix(rows: Sequence[CalibrationRow]) -> np.ndarray:
    if not rows:
        raise ValueError("no calibration rows to stack")
    return np.vstack([np.asarray(row.features, dtype=np.float64) for row in rows])


def label_vector(rows: Sequence[CalibrationRow]) -> np.ndarray:
    return np.asarray([int(row.y) for row in rows], dtype=np.int64)


# ------------------------------------------------------------------------------- calibrator

@dataclass
class VixStoryCalibrator:
    """Platt map from estimator score to ``P(correct)``, as plain JSON; ``none`` is identity."""

    mode: str = "none"  # "none" | "platt"
    params: dict[str, Any] = field(default_factory=dict)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self.mode == "none":
            return scores
        if self.mode == "platt":
            a, b = float(self.params["a"]), float(self.params["b"])
            return 1.0 / (1.0 + np.exp(-np.clip(a * scores + b, -60.0, 60.0)))
        raise ValueError(f"unknown calibrator mode {self.mode!r}")

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "params": dict(self.params)}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> VixStoryCalibrator:
        return cls(mode=str(data["mode"]), params=dict(data.get("params") or {}))


def fit_calibrator(scores: Sequence[float], labels: Sequence[int], *,
                   min_n: int = MIN_CAL_N, min_class: int = MIN_CAL_CLASS
                   ) -> tuple[VixStoryCalibrator, list[dict[str, Any]]]:
    """Platt calibrator on out-of-fold scores, or a recorded reason it was not fit.

    The calibrator is a training-side object: its input is the cross-fit ``p(correct)`` of the
    train questions, never a validation split. Too few rows or a single class means the mapping
    is undefined, so it is skipped and the reason returned instead of a fabricated curve.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        return VixStoryCalibrator(), [{"reason": "calibrator_needs_matching_non_empty_rows"}]
    if not np.all(np.isfinite(scores)):
        return VixStoryCalibrator(), [{"reason": "calibrator_received_non_finite_scores"}]
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if scores.size < min_n:
        return VixStoryCalibrator(), [{"reason": "too_few_oof_rows", "n": int(scores.size),
                                       "min_n": int(min_n)}]
    if min(n_pos, n_neg) < min_class:
        return VixStoryCalibrator(), [{"reason": "single_class_oof", "n_positive": n_pos,
                                       "n_negative": n_neg, "min_class": int(min_class)}]
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(max_iter=1000).fit(scores.reshape(-1, 1), labels)
    return VixStoryCalibrator(mode="platt", params={"a": float(lr.coef_[0, 0]),
                                                    "b": float(lr.intercept_[0])}), []


# -------------------------------------------------------------------------------- estimator

class VixStoryCorrectnessEstimator:
    """Standardised logistic regression (+ optional calibrator), persisted as JSON only.

    ``fitted=False`` records an estimator with no logistic model behind it (constant labels);
    its :meth:`predict` raises instead of returning invented probabilities, and
    :meth:`probability_from_obs` falls back to the stored constant rate, explicitly flagged.
    """

    def __init__(self, *, scaler: FeatureScaler, coef: Sequence[float], intercept: float,
                 calibrator: VixStoryCalibrator | None = None, fitted: bool = True,
                 constant: float | None = None, meta: Mapping[str, Any] | None = None) -> None:
        self.feature_names = list(VIXSTORY_CORRECTNESS_FEATURES)
        self.scaler = scaler
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept = float(intercept)
        self.calibrator = calibrator or VixStoryCalibrator()
        self.fitted = bool(fitted)
        self.constant = None if constant is None else float(constant)
        self.meta = dict(meta or {})
        if len(self.coef) != len(self.feature_names):
            raise ValueError(f"{len(self.coef)} coefficients for {len(self.feature_names)} features")
        if not self.fitted and self.constant is None:
            raise ValueError("an unfitted estimator must carry its constant base rate")

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, features: np.ndarray, labels: Sequence[int], *, seed: int = 42,
            calibrator: VixStoryCalibrator | None = None,
            meta: Mapping[str, Any] | None = None) -> VixStoryCorrectnessEstimator:
        """Scaler + logistic regression on the train rows; both classes are required."""
        from sklearn.linear_model import LogisticRegression

        matrix = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != labels.size or matrix.shape[0] == 0:
            raise ValueError(f"features {matrix.shape} and labels {labels.shape} disagree")
        if matrix.shape[1] != len(VIXSTORY_CORRECTNESS_FEATURES):
            raise ValueError(f"features have {matrix.shape[1]} columns, expected "
                             f"{len(VIXSTORY_CORRECTNESS_FEATURES)}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("correctness features contain non-finite values")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("the correctness model needs both classes in the training rows")
        scaler = FeatureScaler.fit(matrix)
        standardized = scaler.transform(matrix)
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(standardized, labels)
        return cls(scaler=scaler, coef=lr.coef_[0], intercept=float(lr.intercept_[0]),
                   calibrator=calibrator, fitted=True, meta=meta)

    @classmethod
    def constant(cls, *, rate: float, meta: Mapping[str, Any] | None = None
                 ) -> VixStoryCorrectnessEstimator:
        """An unfitted estimator for constant labels: no model, only a flagged base rate."""
        dim = len(VIXSTORY_CORRECTNESS_FEATURES)
        scaler = FeatureScaler(mean=[0.0] * dim, std=[1.0] * dim)
        return cls(scaler=scaler, coef=np.zeros(dim), intercept=0.0,
                   calibrator=VixStoryCalibrator(), fitted=False, constant=float(rate),
                   meta=meta)

    # ---------------------------------------------------------------- inference
    @property
    def calibrated(self) -> bool:
        return self.fitted and self.calibrator.mode == "platt"

    def score(self, features: np.ndarray) -> np.ndarray:
        """Standardised logistic score (before calibration); raises if unfitted."""
        if not self.fitted:
            raise ValueError("this estimator has no fitted logistic model (constant labels); "
                             "it has no score — use probability_from_obs, which returns the "
                             "constant base rate")
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return self.scaler.transform(matrix) @ self.coef + self.intercept

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``P(correct)`` in [0, 1]; raises if the pipeline produced anything else."""
        raw = np.asarray(self.score(features), dtype=np.float64)
        if self.calibrator.mode == "none":
            out = 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
        else:
            out = np.asarray(self.calibrator.transform(raw), dtype=np.float64)
        if not np.all(np.isfinite(out)) or out.min() < 0.0 or out.max() > 1.0:
            raise ValueError(f"p_correct outside [0, 1]: min={out.min()!r} max={out.max()!r}")
        return out

    def probability_from_obs(self, obs: Mapping[str, Any]) -> float:
        """The deployed confidence for one observation; never reads the gold side."""
        if not self.fitted:
            assert self.constant is not None
            return float(self.constant)
        return float(self.predict(features_from_obs(obs))[0])

    # -------------------------------------------------------------- persistence
    def to_json(self) -> dict[str, Any]:
        criterion = self.meta.get("reward_criterion")
        return {
            "schema": SCHEMA,
            "feature_names": self.feature_names,
            "scaler": self.scaler.to_json(),
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "calibrator": self.calibrator.to_json(),
            "fitted": self.fitted,
            "constant": self.constant,
            "calibrated": self.calibrated,
            "identity": artifact_identity(criterion) if criterion is not None else None,
            "meta": self.meta,
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory / "calibrator_model.json", self.to_json())
        return directory

    @classmethod
    def load(cls, directory: str | Path, *,
             expected_identity: str | None = None) -> VixStoryCorrectnessEstimator:
        """Load a persisted estimator, refusing any artifact this module cannot own."""
        data = read_json(Path(directory) / "calibrator_model.json")
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{directory}: schema {data.get('schema')!r} != {SCHEMA!r}")
        if list(data.get("feature_names", ())) != list(VIXSTORY_CORRECTNESS_FEATURES):
            raise ValueError(
                f"{directory}: artifact feature list {data.get('feature_names')} does not match "
                f"this module's {list(VIXSTORY_CORRECTNESS_FEATURES)}; retrain the estimator")
        criterion = (data.get("meta") or {}).get("reward_criterion")
        stored_identity = data.get("identity")
        if criterion is not None and stored_identity != artifact_identity(criterion):
            raise ValueError(f"{directory}: artifact identity does not match its reward criterion; "
                             "the file is corrupt or was edited")
        if expected_identity is not None and stored_identity != expected_identity:
            raise ValueError(f"{directory}: artifact identity {stored_identity!r} != expected "
                             f"{expected_identity!r}; it was fit for a different configuration")
        scaler = FeatureScaler(**data["scaler"])
        return cls(scaler=scaler, coef=data["coef"], intercept=data["intercept"],
                   calibrator=VixStoryCalibrator.from_json(data["calibrator"]),
                   fitted=bool(data["fitted"]),
                   constant=data.get("constant"), meta=data.get("meta"))


# --------------------------------------------------------------------------------- cross-fit

def group_folds(qids: Sequence[str], n_folds: int) -> dict[str, int]:
    """Assign each qid to a fold, round-robin over the sorted unique qids (group-disjoint)."""
    if n_folds < 2:
        raise ValueError(f"cross-fitting needs at least 2 folds, got {n_folds}")
    unique = sorted(set(str(qid) for qid in qids))
    if len(unique) < n_folds:
        raise ValueError(f"cannot split {len(unique)} question(s) into {n_folds} folds")
    return {qid: index % n_folds for index, qid in enumerate(unique)}


@dataclass
class CrossFit:
    """Out-of-fold scores plus the deployed estimator fit on every row."""

    estimator: VixStoryCorrectnessEstimator
    oof_scores: np.ndarray
    folds: dict[str, int]
    blocks: list[dict[str, Any]]
    calibrator_blocks: list[dict[str, Any]]


def cross_fit(rows: Sequence[CalibrationRow], *, n_folds: int = DEFAULT_N_FOLDS, seed: int = 42,
              calibrator: bool = True) -> CrossFit:
    """Cross-fit ``p(correct)`` on the train rows and refit the deployed estimator on all of them.

    Each fold's rows are scored by a model that never saw that fold's questions, so the
    out-of-fold scores are honest and the calibrator fit on them is not optimistic. A fold whose
    training side is single-class is skipped and recorded (its rows stay unscored, NaN) rather
    than forcing a fit.
    """
    folds = group_folds([row.qid for row in rows], n_folds)
    labels = label_vector(rows)
    oof = np.full(len(rows), np.nan, dtype=np.float64)
    blocks: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_idx = [i for i, row in enumerate(rows) if folds[row.qid] != fold]
        test_idx = [i for i, row in enumerate(rows) if folds[row.qid] == fold]
        if not train_idx or not test_idx:
            blocks.append({"fold": fold, "reason": "empty_fold"})
            continue
        train_labels = labels[train_idx]
        if set(np.unique(train_labels).tolist()) != {0, 1}:
            blocks.append({"fold": fold, "reason": "single_class_fold",
                           "observed_classes": sorted(set(np.unique(train_labels).tolist()))})
            continue
        fold_estimator = VixStoryCorrectnessEstimator.fit(
            feature_matrix([rows[i] for i in train_idx]), train_labels, seed=seed)
        # Out-of-fold *scores* (the pre-sigmoid logistic score), not probabilities: the
        # calibrator maps a score to P(correct) and is applied to the deployed model's score at
        # inference, so it must be fit on the same quantity it later transforms.
        oof[test_idx] = fold_estimator.score(feature_matrix([rows[i] for i in test_idx]))

    calibrator_blocks: list[dict[str, Any]] = []
    fitted_calibrator = VixStoryCalibrator()
    if calibrator:
        fitted_calibrator, calibrator_blocks = fit_calibrator(oof, labels)

    estimator = fit_deployed(rows, calibrator=fitted_calibrator, seed=seed)
    return CrossFit(estimator=estimator, oof_scores=oof, folds=folds, blocks=blocks,
                    calibrator_blocks=calibrator_blocks)


def fit_deployed(rows: Sequence[CalibrationRow], *, calibrator: VixStoryCalibrator,
                 seed: int = 42) -> VixStoryCorrectnessEstimator:
    """Fit the deployed estimator on every row, or return a flagged constant when labels are
    constant. Constant labels are never forced into a logistic fit."""
    labels = label_vector(rows)
    if set(np.unique(labels).tolist()) == {0, 1}:
        return VixStoryCorrectnessEstimator.fit(feature_matrix(rows), labels, seed=seed,
                                                calibrator=calibrator)
    return VixStoryCorrectnessEstimator.constant(rate=float(labels.mean()))


# ----------------------------------------------------------------------------- E03 policy

def e03_action(obs: Mapping[str, Any], p_correct: float, tau: float,
               fallback: Sequence[int] = SEARCH_FALLBACK) -> int:
    """The deterministic E03 rule: confidence gate on ANSWER, else a valid SEARCH, else ABSTAIN.

    Reads the observation mask, ``p_correct`` and ``tau`` only — no gold, no oracle.
    """
    mask = np.asarray(obs["action_mask"], dtype=bool)
    if mask[ANSWER] and float(p_correct) >= float(tau):
        return ANSWER
    for action in fallback:
        if mask[action]:
            return int(action)
    return ABSTAIN


def score_table(env: VixStoryEnv, estimator: VixStoryCorrectnessEstimator,
                qids: Sequence[str]) -> dict[str, float]:
    """``state_hash -> p_correct`` for every reachable state, so a tau sweep costs no model calls."""
    table: dict[str, float] = {}
    for ref in enumerate_state_refs(env, qids):
        obs = replay(env, ref.qid, ref.history)
        value = float(estimator.probability_from_obs(obs))
        if not math.isfinite(value):
            raise VixStoryCalibrationError(f"non-finite p_correct at state {ref.state_hash[:16]}...")
        table[ref.state_hash] = value
    return table


def tau_candidates(scores: Sequence[float], *,
                   n_quantiles: int = DEFAULT_TAU_QUANTILES) -> list[float]:
    """Gold-free candidate grid: score quantiles plus a never-answer sentinel."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot build a tau grid from no scores")
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_quantiles + 1)))
    sentinel = float(values.max()) + 1.0
    return [float(q) for q in quantiles] + [sentinel]


@dataclass
class EpisodeResult:
    """One simulated E03 episode. ``reward`` is the return the environment actually paid."""

    qid: str
    reward: float
    answered: bool
    correct: bool
    n_searches: int
    terminal_action: int


def simulate_episode(env: VixStoryEnv, qid: str, scores: Mapping[str, float],
                     tau: float) -> EpisodeResult:
    """Replay E03 on one question over the cached score table; no oracle at inference."""
    obs, _ = env.reset(seed=0, options={"qid": qid})
    total, n_searches, answered = 0.0, 0, False
    while True:
        state_hash = env.state_hash()
        if state_hash not in scores:
            raise VixStoryCalibrationError(
                f"state {state_hash[:16]}... of qid={qid!r} was not scored; the score table must "
                "cover every reachable state")
        action = e03_action(obs, scores[state_hash], tau)
        if action in (SEARCH_BM25, SEARCH_DENSE):
            n_searches += 1
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if action == ANSWER:
            answered = True
        if terminated or truncated:
            break
    outcome = env.private_outcome()
    if outcome is None:
        raise VixStoryCalibrationError(f"no private outcome after the episode of qid={qid!r}")
    return EpisodeResult(qid=qid, reward=total, answered=answered,
                         correct=bool(outcome["correct"]), n_searches=n_searches,
                         terminal_action=action)


def evaluate_tau(env: VixStoryEnv, qids: Sequence[str], scores: Mapping[str, float],
                 tau: float) -> dict[str, Any]:
    """Mean episode reward / coverage of E03 at ``tau`` on one question set."""
    results = [simulate_episode(env, qid, scores, tau) for qid in qids]
    returns = [r.reward for r in results]
    answered = [r for r in results if r.answered]
    correct = sum(r.correct for r in answered)
    n = len(qids)
    return {"mean_episode_reward": float(np.mean(returns)) if returns else 0.0,
            "n_episodes": n, "n_answered": len(answered),
            "coverage": (len(answered) / n) if n else 0.0,
            "selective_risk": (1.0 - correct / len(answered)) if answered else None,
            "mean_searches": float(np.mean([r.n_searches for r in results])) if results else 0.0}


def select_tau(candidates: Sequence[float], evaluate: Callable[[float], dict[str, Any]]
               ) -> dict[str, Any]:
    """Apply the pre-registered criterion: max mean reward, ties to the higher tau."""
    log = []
    for tau in candidates:
        row = dict(evaluate(float(tau)))
        row["tau"] = float(tau)
        log.append(row)
    best = sorted(log, key=lambda row: (row["mean_episode_reward"], row["tau"]))[-1]
    sentinel = max(float(tau) for tau in candidates)
    return {"criterion": PRE_REGISTERED_CRITERION, "selected_tau": float(best["tau"]),
            "selected": best, "log": log,
            "selected_is_never_answer": bool(abs(float(best["tau"]) - sentinel) < 1e-12)}


# ------------------------------------------------------------------------------------- run

def run_vixstory_calibration(env: VixStoryEnv, *, train_qids: Sequence[str],
                             dev_env: VixStoryEnv, dev_qids: Sequence[str],
                             n_folds: int = DEFAULT_N_FOLDS,
                             n_quantiles: int = DEFAULT_TAU_QUANTILES, seed: int = 42,
                             save_dir: str | Path | None = None) -> dict[str, Any]:
    """Fit p(correct) on ``train_qids`` (cross-fit) and pick ``tau`` on ``dev_qids`` by reward.

    ``env`` and ``dev_env`` are the train300 and dev200 environments; no validation split is
    accepted. The reward criterion (and the identity an artifact is bound to) is read from
    ``env.config`` — the reward the environment actually pays — never overridden here. Nothing
    is written unless ``save_dir`` is given, and blockers are returned rather than raised so a
    caller can see exactly which precondition failed.
    """
    criterion = reward_criterion(env.config)
    summary: dict[str, Any] = {
        "schema": SCHEMA, "seed": int(seed), "n_folds": int(n_folds),
        "reward_criterion": criterion, "identity": artifact_identity(criterion),
        "train_qids": [str(q) for q in train_qids], "dev_qids": [str(q) for q in dev_qids],
        "blockers": [], "notes": [], "tau": None, "selection": None, "artifact": None,
    }

    train_rows = collect_rows(env, train_qids)
    summary["train_rows"] = len(train_rows)
    summary["train_questions_with_rows"] = len({row.qid for row in train_rows})
    if not train_rows:
        summary["status"] = "BLOCKED"
        summary["blockers"].append(
            "no train state carries a candidate; the correctness estimator cannot be fit")
        return summary

    labels = label_vector(train_rows)
    classes = sorted(set(np.unique(labels).tolist()))
    if classes != [0, 1]:
        summary["blockers"].append(
            f"train labels are single-class {classes}; no logistic model is fit")
    try:
        cross = cross_fit(train_rows, n_folds=n_folds, seed=seed)
    except ValueError as exc:
        summary["status"] = "BLOCKED"
        summary["blockers"].append(f"cross-fit is not practical here: {exc}")
        return summary

    estimator = cross.estimator
    summary["estimator"] = {
        "fitted": estimator.fitted, "calibrated": estimator.calibrated,
        "calibrator_mode": estimator.calibrator.mode, "constant": estimator.constant,
        "coef": estimator.coef.tolist(), "intercept": estimator.intercept,
        "scaler_mean": list(estimator.scaler.mean), "scaler_std": list(estimator.scaler.std),
    }
    summary["folds"] = cross.folds
    summary["fold_blocks"] = cross.blocks
    summary["calibrator_blocks"] = cross.calibrator_blocks
    summary["oof_scored_rows"] = int(np.isfinite(cross.oof_scores).sum())
    if not estimator.fitted:
        summary["blockers"].append(
            "train labels were constant, so a constant p_correct was deployed instead of a "
            "fitted model; the threshold sweep over a constant score is degenerate")

    if not dev_qids:
        summary["status"] = "BLOCKED"
        summary["blockers"].append("no dev qids were provided; tau cannot be selected")
        return summary

    scores = score_table(dev_env, estimator, dev_qids)
    candidates = tau_candidates(list(scores.values()), n_quantiles=n_quantiles)
    selection = select_tau(candidates, lambda tau: evaluate_tau(dev_env, dev_qids, scores, tau))
    summary["tau"] = selection["selected_tau"]
    summary["selection"] = selection
    summary["dev_state_scores"] = scores
    if selection["selected_is_never_answer"]:
        summary["notes"].append(
            "the never-answer sentinel won the selection; on dev200 this tau makes E03 abstain "
            "on every episode")
    if selection["selected"]["n_answered"] == 0:
        summary["notes"].append("E03 answers no dev episode at the selected tau")

    if save_dir is not None:
        directory = Path(save_dir)
        estimator.meta.update({"reward_criterion": criterion, "seed": int(seed),
                               "train_qids": summary["train_qids"], "dev_qids": summary["dev_qids"],
                               "tau": summary["tau"], "criterion": PRE_REGISTERED_CRITERION})
        estimator.save(directory)
        atomic_write_json(directory / "threshold_selection.json", {
            "criterion": PRE_REGISTERED_CRITERION, "tau": summary["tau"],
            "selected": selection["selected"], "log": selection["log"],
            "selected_is_never_answer": selection["selected_is_never_answer"],
            "split": "dev200", "identity": summary["identity"]})
        summary["artifact"] = str(directory)

    summary["status"] = "DEGRADED" if summary["blockers"] else "SUCCEEDED"
    return summary

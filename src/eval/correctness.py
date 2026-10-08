"""Answer-correctness estimator and calibrated threshold rule (B5, `02_IMPLEMENTATION/08_...`).

The target is **answer correctness of the final candidate**, never ``P(action=ANSWER)`` and
never the reader's raw span score:

    y_correct = 1  iff  the final candidate matches gold under the reward's own criterion
                        (token-F1 >= candidate_f1_threshold **and** provenance valid)

Labels therefore come from :class:`~src.rl.reward.RewardCalculator` — the one definition of
"correct" in this codebase — while features and inference come from the *observation* only.
Gold text, labels, the reward value, the true no-answer flag and Laya probabilities are
forbidden as features (spec 08 §2).

Split discipline is enforced in code, not by convention: the model and its scaler may only
be fit on ``train_strict``, the calibrator only on the reserved ``val_strict`` calibration
subset. Passing the wrong split raises (spec 08 §7).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.qa.metrics import tokenize
from src.rl.features import FeatureScaler
from src.rl.reward import EpisodePrivate, RewardCalculator

#: Splits the correctness model may be trained on / calibrated on. Test is never allowed.
FIT_SPLIT = "train_strict"
CALIBRATION_SPLIT = "val_strict"

#: Sentinels for legitimately missing observation values (the env reports -inf for a null
#: candidate's "best" score). Documented constants beat silent NaN propagation.
MISSING_SCORE = -50.0
MISSING_COUNT = -1.0

#: Pre-registered calibration plan thresholds (spec 08 §4). Below ``MIN_CAL_N`` rows or
#: ``MIN_CAL_CLASS`` of either class the calibration is unreliable -> variant B5-RAW.
MIN_CAL_N = 100
MIN_CAL_CLASS = 20
ISOTONIC_MIN_N = 500

#: Observation-safe feature list. Adding a name here is a spec change; the order is the
#: column order of every stored matrix and the order of the persisted coefficients.
CORRECTNESS_FEATURES: tuple[str, ...] = (
    "reader_non_null_score",
    "reader_null_margin",
    "reader_has_candidate",
    "candidate_provenance_ok",
    "candidate_len_tokens",
    "candidate_norm_len",
    "seen_docs",
    "seen_tokens",
    "optional_used",
    "remaining_optional",
    "searches_used",
    "history_len",
    "bm25_top_score",
    "bm25_top_gap",
    "dense_top_score",
    "dense_top_gap",
    "retrieval_preview_missing",
    "question_len_tokens",
    "question_norm_len",
)


def _finite(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def features_from_obs(obs: Mapping[str, Any]) -> np.ndarray:
    """The estimator's input: visible state only, fixed width, always finite.

    Pure function of ``obs`` — mutating the hidden gold side of an episode cannot change a
    single value (enforced by ``tests/test_correctness.py``).
    """
    candidate = obs.get("candidate") or {}
    retrieval = obs.get("retrieval") or {}
    answer = candidate.get("text")
    tokens = tokenize(answer) if answer else []
    question = str(obs.get("question", ""))
    has_candidate = 1.0 if answer is not None else 0.0
    preview_missing = 1.0 if not (retrieval.get("bm25_top_score")
                                  or retrieval.get("dense_top_score")) else 0.0
    vector = [
        _finite(candidate.get("non_null_score"), MISSING_SCORE),
        _finite(candidate.get("null_margin"), MISSING_SCORE),
        has_candidate,
        1.0 if candidate.get("provenance_ok") else 0.0,
        float(len(tokens)),
        float(len(answer)) if answer else 0.0,
        float(len(obs.get("seen_doc_ids", ()))),
        float(sum(len(tokenize(text)) for text in obs.get("seen_doc_texts", ()))),
        float(_finite(obs.get("optional_used"), MISSING_COUNT)),
        float(_finite(obs.get("remaining_optional_searches"), MISSING_COUNT)),
        float(_finite(obs.get("searches_used"), MISSING_COUNT)),
        float(len(obs.get("history", ()))),
        _finite(retrieval.get("bm25_top_score"), 0.0),
        _finite(retrieval.get("bm25_top_gap"), 0.0),
        _finite(retrieval.get("dense_top_score"), 0.0),
        _finite(retrieval.get("dense_top_gap"), 0.0),
        preview_missing,
        float(len(tokenize(question))),
        float(len(question)),
    ]
    out = np.asarray(vector, dtype=np.float64)
    if out.shape != (len(CORRECTNESS_FEATURES),):
        raise AssertionError(f"feature width {out.shape} != {len(CORRECTNESS_FEATURES)}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"non-finite correctness feature vector: {out.tolist()}")
    return out


def candidate_label(reward: RewardCalculator, *, candidate_text: str | None,
                    provenance_ok: bool, private: EpisodePrivate) -> tuple[int, float]:
    """``(y_correct, f1)`` under the reward calculator's exact ANSWER criterion.

    Calling ``terminal_utility`` is deliberate: the estimator's target and the episode
    reward can then never drift apart, because there is only one correctness definition.
    """
    if reward.config.correct_answer == reward.config.wrong_answer:
        raise ValueError("reward config makes correct and wrong answers indistinguishable")
    breakdown = reward.terminal_utility(
        action="ANSWER", candidate_text=candidate_text, provenance_ok=provenance_ok,
        private=private, n_bm25=0, n_dense=0)
    correct = int(breakdown.terminal_utility == reward.config.correct_answer)
    return correct, float(breakdown.correctness_f1)


def _require_split(split: str, expected: str, what: str) -> None:
    if split != expected:
        raise ValueError(
            f"{what} may only be fit on {expected!r}, got {split!r}; test data is never "
            "used for fitting and val/train are not interchangeable (spec 08 §7)")


@dataclass
class Calibrator:
    """Platt or isotonic map from estimator score to ``P(correct)``, as plain JSON."""

    mode: str  # "platt" | "isotonic" | "none"
    params: dict[str, Any] = field(default_factory=dict)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if self.mode == "none":
            return scores
        if self.mode == "platt":
            a, b = float(self.params["a"]), float(self.params["b"])
            return 1.0 / (1.0 + np.exp(-(a * scores + b)))
        if self.mode == "isotonic":
            x = np.asarray(self.params["x"], dtype=np.float64)
            y = np.asarray(self.params["y"], dtype=np.float64)
            clipped = np.clip(scores, x[0], x[-1])
            return np.interp(clipped, x, y)
        raise ValueError(f"unknown calibrator mode {self.mode!r}")

    def to_json(self) -> dict[str, Any]:
        return {"mode": self.mode, "params": dict(self.params)}


def calibration_plan(n: int, n_positive: int, n_negative: int) -> str:
    """Pre-registered decision: ``"isotonic" | "platt" | "raw"`` (spec 08 §4)."""
    if min(n_positive, n_negative) < MIN_CAL_CLASS or n < MIN_CAL_N:
        return "raw"
    return "isotonic" if n >= ISOTONIC_MIN_N else "platt"


def fit_calibrator(scores: Sequence[float], labels: Sequence[int], *, split: str,
                   plan: str | None = None) -> Calibrator:
    """Fit the calibrator on the reserved val-calibration subset only."""
    _require_split(split, CALIBRATION_SPLIT, "the correctness calibrator")
    from sklearn.isotonic import IsotonicRegression

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError(f"calibrator needs matching non-empty arrays, got {scores.shape} "
                         f"and {labels.shape}")
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibrator received non-finite scores")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("calibrator needs both classes; calibration would be undefined")
    plan = plan or calibration_plan(int(scores.size), int(labels.sum()),
                                    int((1 - labels).sum()))
    if plan == "raw":
        raise ValueError("calibration_plan decided the calibration is unreliable; "
                         "deploy the B5-RAW variant instead of fitting a calibrator")
    if plan == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(max_iter=1000)
        lr.fit(scores.reshape(-1, 1), labels)
        return Calibrator(mode="platt",
                          params={"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])})
    if plan == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrated = iso.fit_transform(scores, labels)
        order = np.argsort(scores, kind="stable")
        return Calibrator(mode="isotonic",
                          params={"x": scores[order].tolist(),
                                  "y": np.asarray(calibrated)[order].tolist()})
    raise ValueError(f"unknown calibration plan {plan!r}")


class CorrectnessEstimator:
    """Standardized logistic regression + a calibration map, persisted as JSON only.

    ``fitted=False`` records an estimator that has no logistic model behind it (a B5-RAW
    run whose out-of-fold rows were single-class); its :meth:`predict` raises instead of
    returning invented probabilities, and only :meth:`probability_from_obs` is meaningful.
    """

    def __init__(self, *, feature_names: Sequence[str], scaler: FeatureScaler,
                 coef: Sequence[float], intercept: float, calibrator: Calibrator,
                 variant: str = "B5", meta: Mapping[str, Any] | None = None,
                 fitted: bool = True) -> None:
        self.feature_names = list(feature_names)
        self.scaler = scaler
        self.coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        self.intercept = float(intercept)
        self.calibrator = calibrator
        self.variant = variant
        self.meta = dict(meta or {})
        self.fitted = bool(fitted)
        if len(self.coef) != len(self.feature_names):
            raise ValueError(f"{len(self.coef)} coefficients for {len(self.feature_names)} features")

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(cls, features: np.ndarray, labels: Sequence[int], *, split: str,
            seed: int = 42, calibrator: Calibrator | None = None, variant: str = "B5",
            meta: Mapping[str, Any] | None = None) -> "CorrectnessEstimator":
        """Fit scaler + logistic regression. ``split`` must be ``train_strict``."""
        _require_split(split, FIT_SPLIT, "the correctness model and its scaler")
        from sklearn.linear_model import LogisticRegression

        matrix = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != labels.size or matrix.shape[0] == 0:
            raise ValueError(f"features {matrix.shape} and labels {labels.shape} disagree")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("correctness features contain non-finite values")
        if set(np.unique(labels).tolist()) != {0, 1}:
            raise ValueError("the correctness model needs both classes in the training rows")
        scaler = FeatureScaler.fit(matrix)
        standardized = scaler.transform(matrix)
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(standardized, labels)
        return cls(feature_names=CORRECTNESS_FEATURES, scaler=scaler,
                   coef=lr.coef_[0], intercept=float(lr.intercept_[0]),
                   calibrator=calibrator or Calibrator(mode="none"), variant=variant,
                   meta=meta)

    # ---------------------------------------------------------------- inference
    @property
    def is_calibrated(self) -> bool:
        return self.variant == "B5" and self.calibrator.mode in ("platt", "isotonic")

    def score(self, features: np.ndarray) -> np.ndarray:
        """Raw logistic score (before calibration)."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        standardized = self.scaler.transform(matrix)
        return standardized @ self.coef + self.intercept

    def predict(self, features: np.ndarray) -> np.ndarray:
        """``P(correct)`` in [0, 1]; raises if the pipeline produced anything else.

        With no calibrator (``mode="none"``) the logistic model's own output — the sigmoid of
        its score — is already a probability, so it is returned as such instead of the raw
        logit. Raises when the estimator was never fitted.
        """
        if not self.fitted:
            raise ValueError(
                "this estimator has no fitted logistic model (its training rows were "
                "single-class); it has no P(correct) — use probability_from_obs, which "
                "returns the reader null_margin for B5-RAW")
        raw = np.asarray(self.score(features), dtype=np.float64)
        if self.calibrator.mode == "none":
            out = 1.0 / (1.0 + np.exp(-np.clip(raw, -60.0, 60.0)))
        else:
            out = np.asarray(self.calibrator.transform(raw), dtype=np.float64)
        if not np.all(np.isfinite(out)) or out.min() < 0.0 or out.max() > 1.0:
            raise ValueError(f"p_correct outside [0, 1]: min={out.min()!r} max={out.max()!r} "
                             "— upstream scores are broken, refusing to clip silently")
        return out

    def probability_from_obs(self, obs: Mapping[str, Any]) -> float:
        """The deployed confidence for one observation.

        ``B5`` returns the calibrated ``P(correct)``. ``B5-RAW`` returns the reader's
        ``null_margin`` — explicitly *not* a probability and never reported as one — which is
        the fallback when calibration proved unreliable (spec 08 §5).
        """
        if self.variant == "B5-RAW":
            margin = _finite((obs.get("candidate") or {}).get("null_margin"), MISSING_SCORE)
            if not math.isfinite(margin):
                raise ValueError(f"B5-RAW null_margin is not finite: {margin!r}")
            return float(margin)
        return float(self.predict(features_from_obs(obs))[0])

    # -------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> Path:
        from src.utils.io import atomic_write_json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.scaler.save(directory / "scaler.json")
        atomic_write_json(directory / "calibrator.json", self.calibrator.to_json())
        atomic_write_json(directory / "model.json", {
            "feature_names": self.feature_names,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
            "variant": self.variant,
            "is_calibrated": self.is_calibrated,
            "fitted": self.fitted,
            "meta": self.meta,
        })
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "CorrectnessEstimator":
        from src.utils.io import read_json

        directory = Path(directory)
        model = read_json(directory / "model.json")
        scaler = FeatureScaler.load(directory / "scaler.json")
        calibrator = Calibrator(**read_json(directory / "calibrator.json"))
        expected = list(CORRECTNESS_FEATURES)
        if list(model["feature_names"]) != expected:
            raise ValueError(
                f"{directory / 'model.json'} was trained on a different feature list "
                f"({model['feature_names']} != {expected}); retrain the estimator")
        return cls(feature_names=model["feature_names"], scaler=scaler, coef=model["coef"],
                   intercept=model["intercept"], calibrator=calibrator,
                   variant=model.get("variant", "B5"), meta=model.get("meta"),
                   fitted=bool(model.get("fitted", True)))


def reliability_metrics(scores: Sequence[float], labels: Sequence[int], *,
                        n_bins: int = 10) -> dict[str, Any]:
    """Brier score, ECE and a reliability table for binary correctness (spec 01 §3).

    Scores must already be calibrated probabilities in [0, 1]; validating here is what keeps
    a raw reader score from being reported as a probability by accident.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError(f"reliability needs matching non-empty arrays, got {scores.shape}, "
                         f"{labels.shape}")
    if not np.all(np.isfinite(scores)) or scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError(f"scores outside [0, 1]: min={scores.min()!r} max={scores.max()!r}")
    if not np.all(np.isin(labels, (0.0, 1.0))):
        raise ValueError("labels must be binary")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": 0,
                         "confidence": None, "accuracy": None})
            continue
        confidence = float(scores[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += count / scores.size * abs(accuracy - confidence)
        bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "count": count,
                     "confidence": confidence, "accuracy": accuracy})
    return {"n": int(scores.size), "n_bins": int(n_bins),
            "brier": float(np.mean((scores - labels) ** 2)), "ece": float(ece), "bins": bins}


def selection_metrics(scores: Sequence[float], labels: Sequence[int], *,
                      coverage_fraction: float = 0.8) -> dict[str, Any]:
    """Risk at a target confidence-derived coverage (spec 01 §1, §6).

    Answers are ranked by calibrated confidence; the retained subset is the top
    ``coverage_fraction`` of them, and risk is ``1 - accuracy`` on that subset. Selection
    uses confidence only — gold labels are used to *measure*, never to choose.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if labels.size == 0:
        return {"n": 0, "coverage": 0.0, "risk_at_coverage": None, "reason": "N/A"}
    if not np.all(np.isfinite(scores)) or scores.min() < 0.0 or scores.max() > 1.0:
        raise ValueError("selection metrics need calibrated scores in [0, 1]")
    k = max(1, int(math.floor(coverage_fraction * labels.size)))
    order = np.argsort(-scores, kind="stable")[:k]
    return {"n": int(labels.size), "k": int(k), "coverage_fraction": float(coverage_fraction),
            "risk_at_coverage": float(1.0 - labels[order].mean())}

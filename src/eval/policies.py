"""Deterministic baseline policies B0-B8 (`03_EXPERIMENTS/00_BASELINES_MATRIX_AND_BUDGET.md` §1).

============================  =================================================================
id                            rule
============================  =================================================================
B0_random_valid               uniform sample over the legal actions (seeded per episode)
B1_bootstrap_then_answer      never searches again: ANSWER if a valid candidate exists else ABSTAIN
B2_bm25_fixed                 spends the whole optional budget on BM25, then decides
B3_dense_fixed                same budget on Dense (costs differ; the depth does not)
B4_search_all                 spends the budget across both tools, then decides (upper compute)
B5_confidence_threshold       ANSWER if valid candidate and p_correct >= tau; else SEARCH while
                              budget remains; else ABSTAIN
B6_laya_ce / B7_laya_rlcd     masked argmax of the frozen Laya prior (CE / RLCD regime)
B8_rl                         greedy masked argmax of the trained REINFORCE MLP
============================  =================================================================

Rules that hold for every policy here:

* **No gold.** Actions come from the observation, the frozen prior and the estimator only.
* **Legal only.** Every action is checked against the mask by :mod:`src.eval.rollout`; an
  illegal proposal raises instead of being tolerated.
* **Deterministic.** B0 is the only stochastic policy, and its generator is seeded from
  ``(seed, qid)``; everything else is a pure function of the observation.

Tool choice for the multi-tool policies (B4, B5) uses
:func:`choose_tool` in ``"alternate"`` mode — deterministic, no gold, no scores: pick the
tool used fewer times so far in this episode, ties (including the first search) go to BM25.
The ``"top_score"`` mode (higher retrieval preview score wins) exists as a documented
alternative and is never the default (spec 08 §5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from src.eval.correctness import CorrectnessEstimator
from src.rl.env import ANSWER, ABSTAIN, SEARCH_BM25, SEARCH_DENSE
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import PolicyNet, TorchScaler, greedy_action
from src.rl.reward import ACTIONS
from src.utils.config import resolve
from src.utils.hashing import stable_seed
from src.utils.io import read_json

POLICY_IDS: tuple[str, ...] = (
    "B0_random_valid", "B1_bootstrap_then_answer", "B2_bm25_fixed", "B3_dense_fixed",
    "B4_search_all", "B5_confidence_threshold", "B6_laya_ce", "B7_laya_rlcd", "B8_rl",
)
#: "B5" -> "B5_confidence_threshold"
POLICY_CODES: dict[str, str] = {policy_id.split("_")[0]: policy_id for policy_id in POLICY_IDS}

#: B0 runs at three seeds; the other policies are deterministic.
RANDOM_POLICY_SEEDS = 3

#: Fallback answer threshold when neither ``--tau`` nor the estimator's own selection log
#: supplies one (spec 08 §5: the threshold is chosen on validation, never on test).
DEFAULT_TAU = 0.5
TOOL_CHOICE_MODES = ("alternate", "top_score")

#: The B8 checkpoint layout written by :mod:`src.pipeline.train_rl`.
RL_POLICY_DIR = "rl_policy"
RL_REQUIRED_FILES = ("policy.safetensors", "scaler.json", "config.json")


def resolve_policy_id(name: str) -> str:
    """Accept ``"B5"`` or the full id; reject anything unknown loudly."""
    if name in POLICY_CODES:
        return POLICY_CODES[name]
    if name in POLICY_IDS:
        return name
    raise ValueError(f"unknown policy {name!r}; known: {list(POLICY_CODES)} or {list(POLICY_IDS)}")


def short_code(policy_id: str) -> str:
    return policy_id.split("_")[0]


def legal_actions(mask: Sequence[bool]) -> list[int]:
    """Indices of the legal actions in ``ACTIONS`` order. ABSTAIN is always legal."""
    actions = [i for i, ok in enumerate(np.asarray(mask, dtype=bool)) if ok]
    if not actions:
        raise ValueError("every action is masked; ABSTAIN must always be legal")
    return actions


def choose_tool(obs: Mapping[str, Any], *, mode: str = "alternate") -> int:
    """Deterministic, gold-free choice between SEARCH_BM25 and SEARCH_DENSE.

    ``alternate`` (default): the tool used fewer times in this episode's agent-chosen
    history; a tie — including the first optional search — goes to BM25. ``top_score``:
    the higher retrieval preview score, ties to BM25. Both fall back to the other tool when
    their favourite is illegal, and raise when no search is legal at all.
    """
    if mode not in TOOL_CHOICE_MODES:
        raise ValueError(f"unknown tool-choice mode {mode!r}; known: {list(TOOL_CHOICE_MODES)}")
    mask = np.asarray(obs["action_mask"], dtype=bool)
    legal = [action for action in (SEARCH_BM25, SEARCH_DENSE) if mask[action]]
    if not legal:
        raise ValueError("choose_tool called with no legal search action")
    if len(legal) == 1:
        return legal[0]
    if mode == "alternate":
        history = tuple(int(a) for a in obs.get("history", ()))
        prefer = SEARCH_BM25 if history.count(SEARCH_BM25) <= history.count(SEARCH_DENSE) \
            else SEARCH_DENSE
    else:  # top_score
        retrieval = obs.get("retrieval") or {}
        bm25 = float(retrieval.get("bm25_top_score") or 0.0)
        dense = float(retrieval.get("dense_top_score") or 0.0)
        prefer = SEARCH_BM25 if bm25 >= dense else SEARCH_DENSE
    return prefer if prefer in legal else legal[0]


class RulePolicy:
    """Shared plumbing: per-episode reset, the answer/abstain fallback, legality helpers."""

    id: str = ""
    variant: str = ""

    def __init__(self) -> None:
        self.last_confidence: float | None = None

    def begin_episode(self, *, qid: str, seed: int) -> None:
        self.last_confidence = None

    def act(self, obs: Mapping[str, Any]) -> int:
        raise NotImplementedError

    @staticmethod
    def _mask(obs: Mapping[str, Any]) -> np.ndarray:
        return np.asarray(obs["action_mask"], dtype=bool)

    def _decide(self, obs: Mapping[str, Any]) -> int:
        """B1's terminal rule: answer only with a valid candidate, otherwise abstain."""
        return ANSWER if self._mask(obs)[ANSWER] else ABSTAIN


class RandomValidPolicy(RulePolicy):
    """B0: uniform over legal actions; the generator is seeded from ``(seed, qid)``."""

    id = "B0_random_valid"
    variant = "uniform_legal"

    def begin_episode(self, *, qid: str, seed: int) -> None:
        super().begin_episode(qid=qid, seed=seed)
        self._rng = np.random.default_rng(stable_seed("B0", int(seed), str(qid)))

    def act(self, obs: Mapping[str, Any]) -> int:
        return int(self._rng.choice(legal_actions(self._mask(obs))))


class BootstrapThenAnswerPolicy(RulePolicy):
    """B1: zero-adaptation — the forced bootstrap is the only retrieval."""

    id = "B1_bootstrap_then_answer"
    variant = "bootstrap_only"

    def act(self, obs: Mapping[str, Any]) -> int:
        return self._decide(obs)


@dataclass
class FixedToolPolicy(RulePolicy):
    """B2/B3: spend the full optional budget on one tool, then decide."""

    tool: str = "bm25"

    def __post_init__(self) -> None:
        if self.tool not in ("bm25", "dense"):
            raise ValueError(f"unknown fixed tool {self.tool!r}")
        RulePolicy.__init__(self)
        self.id = "B2_bm25_fixed" if self.tool == "bm25" else "B3_dense_fixed"
        self.variant = self.tool

    @property
    def action(self) -> int:
        return SEARCH_BM25 if self.tool == "bm25" else SEARCH_DENSE

    def act(self, obs: Mapping[str, Any]) -> int:
        mask = self._mask(obs)
        if int(obs.get("remaining_optional_searches", 0)) > 0 and mask[self.action]:
            return self.action
        return self._decide(obs)


@dataclass
class SearchAllPolicy(RulePolicy):
    """B4: exhaust the budget across both tools, then decide. Never exceeds the budget."""

    tool_mode: str = "alternate"

    def __post_init__(self) -> None:
        RulePolicy.__init__(self)
        self.id = "B4_search_all"
        self.variant = f"search_all:{self.tool_mode}"

    def act(self, obs: Mapping[str, Any]) -> int:
        mask = self._mask(obs)
        budget_left = int(obs.get("remaining_optional_searches", 0)) > 0
        if budget_left and (mask[SEARCH_BM25] or mask[SEARCH_DENSE]):
            return choose_tool(obs, mode=self.tool_mode)
        return self._decide(obs)


class ConfidenceThresholdPolicy(RulePolicy):
    """B5: the calibrated-correctness threshold rule (spec 08 §5), exactly three branches.

        valid candidate and p_correct >= tau   -> ANSWER
        else, with optional budget left        -> fixed tool rule (default: alternate)
        else                                   -> ABSTAIN

    ``p_correct`` must be a finite probability for the calibrated variant; only the
    documented ``B5-RAW`` fallback may return a reader score instead, and its rows are
    never reported with calibration metrics (NaN / out-of-range values raise).
    """

    id = "B5_confidence_threshold"

    def __init__(self, estimator: CorrectnessEstimator, *, tau: float = DEFAULT_TAU,
                 tool_mode: str = "alternate") -> None:
        super().__init__()
        if tool_mode not in TOOL_CHOICE_MODES:
            raise ValueError(f"unknown tool-choice mode {tool_mode!r}")
        if not math.isfinite(float(tau)):
            raise ValueError(f"tau must be finite, got {tau!r}")
        self.estimator = estimator
        self.tau = float(tau)
        self.tool_mode = tool_mode
        self.variant = f"{estimator.variant}:{tool_mode}"

    @property
    def calibrated(self) -> bool:
        """True only when ``p_correct`` really is a calibrated probability (spec 08 §5)."""
        return self.estimator.variant != "B5-RAW" and self.estimator.is_calibrated

    def act(self, obs: Mapping[str, Any]) -> int:
        mask = self._mask(obs)
        probability = float(self.estimator.probability_from_obs(obs))
        self.last_confidence = probability
        if not math.isfinite(probability):
            raise ValueError(f"B5: p_correct is not finite ({probability!r}) for "
                             f"qid={obs.get('qid')}; refusing to decide on a broken estimate")
        if self.calibrated and not 0.0 <= probability <= 1.0:
            raise ValueError(f"B5: p_correct outside [0, 1] ({probability!r}) for "
                             f"qid={obs.get('qid')} (spec 08 §6)")
        if mask[ANSWER] and probability >= self.tau:
            return ANSWER
        budget_left = int(obs.get("remaining_optional_searches", 0)) > 0
        if budget_left and (mask[SEARCH_BM25] or mask[SEARCH_DENSE]):
            return choose_tool(obs, mode=self.tool_mode)
        return ABSTAIN


class LayaArgmaxPolicy(RulePolicy):
    """B6/B7: masked argmax over the frozen Laya prior pre-warmed by ``build_worlds``."""

    def __init__(self, laya_fn: Callable[[dict], np.ndarray], *, regime: str) -> None:
        super().__init__()
        if regime not in ("ce", "rlcd"):
            raise ValueError(f"unknown Laya regime {regime!r}")
        self.laya_fn = laya_fn
        self.regime = regime
        self.id = f"B{'6' if regime == 'ce' else '7'}_laya_{regime}"
        self.variant = f"masked_argmax:{regime}"

    def _probabilities(self, obs: Mapping[str, Any]) -> np.ndarray:
        try:
            probs = np.asarray(self.laya_fn(dict(obs)), dtype=np.float64).reshape(-1)
        except KeyError as exc:
            raise KeyError(f"{self.id}: {exc}") from exc
        if probs.shape != (len(ACTIONS),):
            raise ValueError(f"{self.id}: expected {len(ACTIONS)} action probabilities, got "
                             f"{probs.shape}")
        if not np.all(np.isfinite(probs)) or probs.min() < 0.0 or probs.sum() <= 0.0:
            raise ValueError(f"{self.id}: invalid probability row {probs.tolist()}")
        return probs / probs.sum()

    def act(self, obs: Mapping[str, Any]) -> int:
        probs = self._probabilities(obs)
        mask = self._mask(obs)
        return int(np.argmax(np.where(mask, probs, -np.inf)))


class RLPolicy(RulePolicy):
    """B8: greedy masked argmax of the trained MLP, mirroring ``reinforce.rollout(greedy=True)``.

    The input path replicates :meth:`src.rl.reinforce.ReinforceTrainer.load_best` exactly:
    raw features from the ``FeatureExtractor``, standardised once by the persisted
    ``TorchScaler`` — the single standardisation point every path uses — so the evaluated
    policy sees the same numbers it was trained and selected with.
    """

    id = "B8_rl"
    variant = "greedy"

    def __init__(self, *, net: PolicyNet, extractor: FeatureExtractor, scaler_t: TorchScaler,
                 laya_fn: Callable[[dict], np.ndarray] | None, prior_regime: str | None,
                 checkpoint: Path | None = None) -> None:
        super().__init__()
        self.net = net
        self.extractor = extractor
        self.scaler_t = scaler_t
        self.laya_fn = laya_fn
        self.prior_regime = prior_regime
        self.checkpoint = checkpoint

    def _prior(self, obs: Mapping[str, Any]) -> Sequence[float] | None:
        if self.laya_fn is None:
            return None
        try:
            return self.laya_fn(dict(obs))
        except KeyError as exc:
            raise KeyError(f"{self.id}: {exc}") from exc

    def act(self, obs: Mapping[str, Any]) -> int:
        raw = self.extractor.vector(obs, self._prior(obs))
        x = torch.as_tensor(raw, dtype=torch.float32).unsqueeze(0)
        if self.scaler_t is not None:
            x = self.scaler_t(x)
        mask = torch.as_tensor(self._mask(obs)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x)
            return int(greedy_action(logits, mask))


def rl_checkpoint_dir(cfg: Mapping[str, Any]) -> Path:
    """``outputs/models/rl_policy`` — the directory :mod:`src.pipeline.train_rl` writes."""
    return resolve(cfg, "models") / RL_POLICY_DIR


def rl_checkpoint_status(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a complete B8 checkpoint exists, and which frozen prior it was trained on.

    ``E06`` (RLCD-prior RL) and ``E07`` (CE-prior RL) are separate rows and only one of them
    can exist: the regime recorded in ``config.json`` says which one this checkpoint is.
    A checkpoint without ``config.json`` cannot be evaluated faithfully — the prior it was
    trained against is unknown — so it is reported as missing rather than guessed.
    """
    directory = rl_checkpoint_dir(cfg)
    missing = [name for name in RL_REQUIRED_FILES if not (directory / name).is_file()]
    status: dict[str, Any] = {"path": str(directory), "exists": not missing, "missing": missing,
                              "regime": None, "config_seed": None, "error": None}
    if missing:
        return status
    try:
        config = read_json(directory / "config.json")
        status["regime"] = (config.get("laya") or {}).get("regime")
        status["config_seed"] = config.get("seed")
        status["best_update"] = config.get("best_update")
    except Exception as exc:  # unreadable config.json = unusable checkpoint
        status["exists"] = False
        status["error"] = f"{type(exc).__name__}: {exc}"
    return status


def load_rl_policy(cfg: Mapping[str, Any], *, laya_fn: Callable[[dict], np.ndarray] | None,
                   prior_regime: str | None) -> RLPolicy:
    """Load the B8 checkpoint written by :mod:`src.pipeline.train_rl` (PolicyNet + scaler)."""
    directory = rl_checkpoint_dir(cfg)
    missing = [name for name in RL_REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"RL policy checkpoint is incomplete at {directory}: missing {missing}; train it "
            "with the P6 stage (src.pipeline.train_rl) before evaluating E06/E07")
    net = PolicyNet.load(directory / "policy.safetensors")
    scaler = FeatureScaler.load(directory / "scaler.json")
    extractor = FeatureExtractor()
    scaler_t = TorchScaler(np.asarray(scaler.mean, dtype=np.float32),
                           np.asarray(scaler.std, dtype=np.float32), scaler.clip)
    if int(net.d_in) != extractor.dim:
        raise ValueError(
            f"{directory / 'policy.safetensors'} expects {net.d_in} features but the current "
            f"FeatureExtractor produces {extractor.dim}; retrain the policy for this feature set")
    return RLPolicy(net=net, extractor=extractor, scaler_t=scaler_t, laya_fn=laya_fn,
                    prior_regime=prior_regime, checkpoint=directory)

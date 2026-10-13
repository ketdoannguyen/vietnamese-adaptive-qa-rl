"""V2 evidence-retrieval REINFORCE policy and trainer — three actions, no reader.

Scope (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §5): arms **E06/E07**
train a small MLP with REINFORCE against the V2 retrieval environment
(:mod:`src.rl.vixstory_env_v2`), conditioned on the public observation plus a **frozen
three-probability prior** (the selected E05/E04 checkpoint, handed in by the caller together
with its content hash). Everything here is V2-only and standalone on purpose:

* three actions everywhere — :data:`src.decision.schema_v2.ACTION_ORDER` is the only action
  ordering, the network is built/saved/loaded with ``n_actions=3``, and a four-action
  checkpoint is refused instead of silently truncated;
* no V1 import (``src.rl.reinforce`` / ``src.rl.features`` / ``src.rl.env`` / ``src.qa.*``)
  and no reader, network call or gold label: the extractor reads exactly the public
  observation keys plus the frozen prior's probabilities;
* the prior is *frozen*: it is called under ``torch.no_grad`` and its output is detached to
  plain floats before it enters the feature vector, so no gradient can reach it, and the
  caller-supplied ``prior_hash`` is part of every checkpoint's identity;
* on-policy: actions are always sampled from the network being trained (never replayed from
  a cache), which the tests re-verify by recomputing each transition's ``log_prob``;
* the QID schedule is fixed before training: three seed-42 permutations of the sorted train
  QIDs concatenated and chunked into batches of at most 32 that cross epoch boundaries, so
  every QID is seen exactly three times — 240 updates for the 2,553 train QIDs;
* the scaler is fitted on train episodes only — the trainer never receives a validation
  environment, so validation cannot enter the gradient, the scaler or the logs;
* checkpoints are written every ``checkpoint_every`` updates and at the end; the greedy
  validation evaluation and the checkpoint selection live outside this module
  (:func:`evaluate_greedy` / :func:`policy_probs` expose the pieces the evaluator needs).

Limits: this module does not run validation selection, does not implement safe resume, and
never touches the test split — by construction, not by omission.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.decision.schema_v2 import ACTION_ORDER
from src.rl.policy import PolicyNet, TorchScaler
from src.utils.hashing import sha256_json
from src.utils.io import append_jsonl, atomic_write_json

N_ACTIONS = len(ACTION_ORDER)
MASK_FILL = -1e9
PRIOR_SUM_TOLERANCE = 1e-6
CHECKPOINT_DIR = "checkpoints"
TRAIN_LOG_FILE = "train_log.jsonl"
SUMMARY_FILE = "summary.json"
CHECKPOINT_FILE = "checkpoint.json"
POLICY_FILE = "policy.safetensors"
SCALER_FILE = "scaler.json"

__all__ = [
    "N_ACTIONS",
    "Episode",
    "Transition",
    "V2FeatureScaler",
    "V2RLConfig",
    "VixStoryV2FeatureExtractor",
    "VixStoryV2ReinforceTrainer",
    "build_schedule",
    "discounted_returns",
    "evaluate_greedy",
    "policy_probs",
    "prior_probs",
    "reinforce_loss",
    "rollout",
    "split_batches",
    "vixstory_v2_feature_names",
]


def vixstory_v2_feature_names() -> list[str]:
    """The one V2 feature contract: 15 public features (no reader, no gold, no answer)."""
    return [
        "prior_p_stop", "prior_p_bm25", "prior_p_dense", "prior_entropy",
        "seen_count", "seen_tokens", "optional_used",
        "last_action_STOP", "last_action_SEARCH_BM25", "last_action_SEARCH_DENSE",
        "mask_stop", "mask_bm25", "mask_dense",
        "question_len_tokens", "question_len_chars",
    ]


def _validated_probs(raw: Any) -> tuple[float, float, float]:
    """Validate a three-probability prior output and detach it to plain floats."""
    if isinstance(raw, torch.Tensor):
        raw = raw.detach().cpu().numpy()
    probs = np.asarray(raw, dtype=np.float64)
    if probs.shape != (N_ACTIONS,):
        raise ValueError(f"prior must return {N_ACTIONS} probabilities; got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"prior probabilities must be finite; got {probs.tolist()}")
    if np.any(probs < 0.0):
        raise ValueError(f"prior probabilities must be non-negative; got {probs.tolist()}")
    total = float(probs.sum())
    if abs(total - 1.0) > PRIOR_SUM_TOLERANCE:
        raise ValueError(f"prior probabilities must sum to 1 (tolerance {PRIOR_SUM_TOLERANCE}); "
                         f"got {total!r}")
    return float(probs[0]), float(probs[1]), float(probs[2])


def prior_probs(prior: Callable[[Mapping[str, Any]], Any],
                obs: Mapping[str, Any]) -> tuple[float, float, float]:
    """Call the frozen prior under ``torch.no_grad`` and return validated plain floats.

    The prior never receives a gradient: it runs in inference mode and its output is
    detached to numpy floats before anything else touches it.
    """
    if not callable(prior):
        raise TypeError("prior must be a callable obs -> three probabilities")
    with torch.no_grad():
        raw = prior(obs)
    return _validated_probs(raw)


class VixStoryV2FeatureExtractor:
    """Public V2 observation (+ frozen prior probabilities) -> raw float32 vector.

    Reads only ``question``, ``seen_doc_ids``/``seen_doc_texts``, ``history``,
    ``optional_used`` and ``action_mask`` — the observation keys the V2 environment emits —
    plus the prior's three probabilities. No gold, no coverage, no reader, no reward.
    Standardisation is not applied here; it happens once at the network input.
    """

    def __init__(self) -> None:
        self.names = vixstory_v2_feature_names()

    @property
    def dim(self) -> int:
        return len(self.names)

    def vector(self, obs: Mapping[str, Any], probs: Sequence[float]) -> np.ndarray:
        mask = np.asarray(obs.get("action_mask"), dtype=bool)
        if mask.shape != (N_ACTIONS,):
            raise ValueError(f"action_mask must have shape ({N_ACTIONS},); got {mask.shape}")
        p = _validated_probs(probs)
        seen_ids = obs.get("seen_doc_ids", ())
        seen_texts = obs.get("seen_doc_texts", ())
        if len(seen_ids) != len(seen_texts):
            raise ValueError("seen_doc_ids and seen_doc_texts must be parallel sequences")
        history = list(obs.get("history", ()))
        last = history[-1] if history else None
        last_index = ACTION_ORDER.index(last) if last in ACTION_ORDER else -1
        question = str(obs.get("question", ""))
        entropy = -sum(value * math.log(value) for value in p if value > 0.0)
        values = [
            p[0], p[1], p[2], entropy,
            float(len(seen_ids)), float(sum(len(str(text).split()) for text in seen_texts)),
            float(obs.get("optional_used", 0)),
            *[1.0 if last_index == index else 0.0 for index in range(N_ACTIONS)],
            *[1.0 if legal else 0.0 for legal in mask],
            float(len(question.split())), float(len(question)),
        ]
        out = np.asarray(values, dtype=np.float32)
        if out.shape != (self.dim,):
            raise AssertionError(f"feature width {out.shape} != declared {(self.dim,)}")
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def matrix(self, observations: Sequence[Mapping[str, Any]],
               prob_rows: Sequence[Sequence[float]]) -> np.ndarray:
        rows = [self.vector(obs, prob_rows[index]) for index, obs in enumerate(observations)]
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack(rows)


@dataclass
class V2FeatureScaler:
    """The train-fitted standardiser, persisted as JSON (same shape as V1's scaler.json).

    A local copy on purpose: importing :mod:`src.rl.features` would pull the QA/reader
    package into the V2 route, which the plan forbids.
    """

    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    clip: float = 10.0

    @classmethod
    def fit(cls, matrix: np.ndarray, *, clip: float = 10.0) -> "V2FeatureScaler":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("scaler requires a non-empty 2-D matrix")
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.tolist(), std=std.tolist(), clip=clip)

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path) -> "V2FeatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(mean=list(payload["mean"]), std=list(payload["std"]),
                   clip=float(payload["clip"]))


@dataclass
class Transition:
    """One on-policy decision: the public features, the legal mask, the action and its reward."""

    features: np.ndarray
    mask: np.ndarray
    action: int
    log_prob: float
    value: float
    entropy: float
    reward: float


@dataclass
class Episode:
    qid: str
    transitions: list[Transition] = field(default_factory=list)
    total_reward: float = 0.0


def build_schedule(qids: Sequence[str], *, epochs: int = 3, seed: int = 42) -> list[str]:
    """``epochs`` seed-42 permutations of the sorted QIDs, concatenated into one schedule.

    The flat schedule is what lets batches cross epoch boundaries; every QID appears exactly
    ``epochs`` times and no QID is ever drawn with replacement.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int; got {seed!r}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError(f"epochs must be a positive int; got {epochs!r}")
    ordered = sorted(qids)
    if not ordered:
        raise ValueError("build_schedule requires at least one qid")
    if any(not isinstance(qid, str) or not qid for qid in ordered):
        raise ValueError("qids must be non-empty strings")
    if len(set(ordered)) != len(ordered):
        raise ValueError("build_schedule requires unique qids")
    rng = np.random.default_rng(seed)
    schedule: list[str] = []
    for _ in range(epochs):
        schedule.extend(ordered[int(index)] for index in rng.permutation(len(ordered)))
    return schedule


def split_batches(schedule: Sequence[str], batch_size: int) -> list[list[str]]:
    """Chunk a flat episode schedule into consecutive batches of at most ``batch_size``."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int; got {batch_size!r}")
    if not schedule:
        raise ValueError("split_batches requires a non-empty schedule")
    return [list(schedule[start:start + batch_size])
            for start in range(0, len(schedule), batch_size)]


def _scaled_tensor(features: np.ndarray, scaler_t: TorchScaler | None,
                   device: torch.device) -> torch.Tensor:
    x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=device).unsqueeze(0)
    return scaler_t(x) if scaler_t is not None else x


def _mask_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(mask, dtype=bool), device=device).unsqueeze(0)


def rollout(env: Any, net: PolicyNet, extractor: VixStoryV2FeatureExtractor,
            scaler_t: TorchScaler | None, prior: Callable[[Mapping[str, Any]], Any], *,
            qid: str, greedy: bool = False,
            device: str | torch.device = "cpu") -> Episode:
    """One episode for ``qid``; every action is sampled from ``net`` itself (on-policy)."""
    device = torch.device(device)
    obs, _ = env.reset(qid=qid)
    episode = Episode(qid=str(obs["qid"]))
    done = False
    while not done:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        features = extractor.vector(obs, prior_probs(prior, obs))
        mask_t = _mask_tensor(mask, device)
        with torch.no_grad():
            logits, value = net(_scaled_tensor(features, scaler_t, device))
            dist = torch.distributions.Categorical(
                logits=logits.masked_fill(~mask_t, MASK_FILL))
            if greedy:
                action = int(torch.argmax(dist.logits, dim=-1).item())
            else:
                action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=device)).item())
            entropy = float(dist.entropy().item())
        obs, reward, terminated, truncated, _ = env.step(action)
        episode.transitions.append(Transition(
            features=features, mask=mask, action=action, log_prob=log_prob,
            value=float(value.item()), entropy=entropy, reward=float(reward)))
        episode.total_reward += float(reward)
        done = terminated or truncated
    return episode


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    """``G_t = sum_{k>=0} gamma^k r_{t+k}``; gamma=1 makes it the suffix sum (plan §5)."""
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        out[index] = running
    return out


def reinforce_loss(episodes: Sequence[Episode], net: PolicyNet,
                   scaler_t: TorchScaler | None, config: V2RLConfig, *,
                   device: str | torch.device = "cpu") -> tuple[torch.Tensor, dict[str, Any]]:
    """Mean REINFORCE-with-baseline loss over every transition (plan §5).

    ``loss = mean[-(G_t - V(s_t).detach()) * log pi(a_t) + value_coef * (V(s_t) - G_t)^2
    - entropy_coef * H(pi)]``. Log-probabilities, values and entropies are recomputed from
    the current network — the stored rollout fields are diagnostics, not replay.
    """
    device = torch.device(device)
    policy_terms: list[torch.Tensor] = []
    value_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for episode in episodes:
        returns = discounted_returns([t.reward for t in episode.transitions], config.gamma)
        for index, transition in enumerate(episode.transitions):
            x = _scaled_tensor(transition.features, scaler_t, device)
            mask_t = _mask_tensor(transition.mask, device)
            logits, value = net(x)
            dist = torch.distributions.Categorical(
                logits=logits.masked_fill(~mask_t, MASK_FILL))
            action_t = torch.tensor([transition.action], device=device)
            log_prob = dist.log_prob(action_t)
            g_t = torch.tensor([returns[index]], dtype=torch.float32, device=device)
            advantage = g_t - value.detach()
            policy_terms.append(-(advantage * log_prob).squeeze(0))
            value_terms.append((value - g_t).pow(2).squeeze(0))
            entropy_terms.append(dist.entropy().squeeze(0))
    if not policy_terms:
        raise ValueError("no transitions collected; cannot compute a policy gradient")
    policy_loss = torch.stack(policy_terms).mean()
    value_loss = torch.stack(value_terms).mean()
    entropy = torch.stack(entropy_terms).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    stats = {"policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item()),
             "entropy": float(entropy.item()), "n_steps": len(policy_terms)}
    return loss, stats


def policy_probs(net: PolicyNet, extractor: VixStoryV2FeatureExtractor,
                 scaler_t: TorchScaler | None, prior: Callable[[Mapping[str, Any]], Any], *,
                 device: str | torch.device = "cpu") -> Callable[[Mapping[str, Any]], np.ndarray]:
    """Wrap the trained net as ``obs -> three masked-softmax probabilities``.

    This is the V2 arms bridge (:func:`src.pipeline.vixstory_arms_v2.masked_greedy`): the
    external evaluator/selector turns it into a greedy policy. Illegal actions receive
    exactly zero probability.
    """
    device = torch.device(device)

    def probs(obs: Mapping[str, Any]) -> np.ndarray:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        features = extractor.vector(obs, prior_probs(prior, obs))
        with torch.no_grad():
            logits, _ = net(_scaled_tensor(features, scaler_t, device))
            masked = logits.masked_fill(~_mask_tensor(mask, device), MASK_FILL)
            return torch.softmax(masked, dim=-1).squeeze(0).cpu().numpy().astype(np.float64)

    return probs


def evaluate_greedy(env: Any, net: PolicyNet, extractor: VixStoryV2FeatureExtractor,
                    scaler_t: TorchScaler | None,
                    prior: Callable[[Mapping[str, Any]], Any], qids: Sequence[str], *,
                    device: str | torch.device = "cpu") -> dict[str, Any]:
    """Greedy (deterministic argmax) evaluation, one episode per QID — the selection input.

    The mean episode return is exactly the mean retrieval return of plan §4: the V2 reward
    *is* the retrieval objective (``0.5*coverage + full_success - 0.05*n_search``), computed
    privately by the environment. Checkpoint selection itself stays external
    (:func:`src.pipeline.vixstory_eval_v2.select_best`).
    """
    qid_list = [str(qid) for qid in qids]
    if not qid_list or len(set(qid_list)) != len(qid_list):
        raise ValueError("evaluate_greedy requires a non-empty list of unique qids")
    device = torch.device(device)
    returns: list[float] = []
    searches: list[int] = []
    lengths: list[int] = []
    action_counts: Counter[str] = Counter()
    for qid in qid_list:
        episode = rollout(env, net, extractor, scaler_t, prior, qid=qid, greedy=True,
                          device=device)
        returns.append(episode.total_reward)
        lengths.append(len(episode.transitions))
        for transition in episode.transitions:
            action_counts[ACTION_ORDER[transition.action]] += 1
        searches.append(sum(1 for t in episode.transitions
                            if ACTION_ORDER[t.action] != "STOP"))
    return {
        "n": len(qid_list),
        "qids_sha256": sha256_json(sorted(qid_list)),
        "mean_retrieval_return": float(np.mean(returns)),
        "mean_n_search": float(np.mean(searches)),
        "mean_length": float(np.mean(lengths)),
        "action_counts": {name: int(action_counts.get(name, 0)) for name in ACTION_ORDER},
    }


@dataclass(frozen=True)
class V2RLConfig:
    """The pinned E06/E07 REINFORCE protocol (plan §5); the defaults are the protocol."""

    seed: int = 42
    gamma: float = 1.0
    hidden_dim: int = 128
    lr: float = 3e-4
    batch_size: int = 32
    updates: int = 240
    epochs: int = 3
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    grad_clip: float = 1.0
    checkpoint_every: int = 25

    def __post_init__(self) -> None:
        for name in ("hidden_dim", "batch_size", "updates", "epochs", "checkpoint_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive int; got {value!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"seed must be a non-negative int; got {self.seed!r}")
        for name in ("gamma", "lr", "entropy_coef", "value_coef", "grad_clip"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be a finite non-negative number; got {value!r}")
        if float(self.lr) == 0.0:
            raise ValueError("lr must be positive")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V2RLConfig":
        """Build from a config mapping, accepting either ``{"rl": {...}}`` or the flat keys."""
        raw = dict(payload.get("rl", payload))
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in raw.items() if key in known})


def _param_vector(net: PolicyNet) -> np.ndarray:
    return np.concatenate([parameter.detach().cpu().numpy().ravel()
                           for parameter in net.parameters()])


def _validate_qids(qids: Sequence[str]) -> tuple[str, ...]:
    out = tuple(qids)
    if not out:
        raise ValueError("train_qids must be non-empty")
    if any(not isinstance(qid, str) or not qid for qid in out):
        raise ValueError("train_qids must be non-empty strings")
    if len(set(out)) != len(out):
        raise ValueError("train_qids contains duplicates")
    return out


def _env_qids(env: Any) -> list[str] | None:
    worlds = getattr(env, "_worlds", None)
    if worlds is None:
        worlds = getattr(env, "worlds", None)
    if worlds is None:
        return None
    if isinstance(worlds, Mapping):
        return list(worlds)
    return [getattr(world, "qid", None) for world in worlds]


def _validate_train_env(env: Any, qids: Sequence[str]) -> None:
    """Prove the env holds exactly the manifest QIDs and can select each of them by qid."""
    env_qids = _env_qids(env)
    if env_qids is not None:
        if any(not isinstance(qid, str) for qid in env_qids) or set(env_qids) != set(qids):
            raise ValueError("train env does not match the QID manifest: "
                             f"env={sorted(map(str, env_qids))[:5]}... manifest={sorted(qids)[:5]}...")
    for qid in qids:
        obs, _ = env.reset(qid=qid)
        if obs.get("qid") != qid:
            raise ValueError(f"train env cannot select qid {qid!r} by qid")


class VixStoryV2ReinforceTrainer:
    """The E06/E07 trainer: fixed seed-42 schedule, frozen prior, checkpoints, no validation.

    The trainer only ever sees the train environment and the train QID manifest. Validation
    evaluation and checkpoint selection happen outside, greedily, so no validation state can
    reach the scaler, the gradients or the logged history.
    """

    def __init__(self, train_env: Any, config: V2RLConfig, *,
                 prior: Callable[[Mapping[str, Any]], Any], prior_hash: str,
                 run_dir: str | Path | None = None,
                 device: str | torch.device = "cpu") -> None:
        if not callable(prior):
            raise TypeError("prior must be a callable obs -> three probabilities")
        if not isinstance(prior_hash, str) or not prior_hash.strip():
            raise ValueError("prior_hash must be the caller's non-empty frozen-prior hash")
        self.train_env = train_env
        self.config = config
        self.prior = prior
        self.prior_hash = prior_hash
        self.device = torch.device(device)
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.extractor = VixStoryV2FeatureExtractor()
        self.net: PolicyNet | None = None
        self.scaler: V2FeatureScaler | None = None
        self.scaler_t: TorchScaler | None = None
        self.history: list[dict[str, Any]] = []
        self.checkpoints: list[Path] = []
        self.identity: str | None = None

    def build(self) -> PolicyNet:
        """A fresh three-action net, seeded with the pinned seed for reproducibility."""
        torch.manual_seed(self.config.seed)
        self.net = PolicyNet(self.extractor.dim, self.config.hidden_dim,
                             n_actions=N_ACTIONS).to(self.device)
        return self.net

    def fit_scaler(self, qids: Sequence[str], *, max_episodes: int = 64) -> V2FeatureScaler:
        """Fit the standardiser on TRAIN episodes only (random legal actions + frozen prior).

        The first ``max_episodes`` QIDs in sorted order are walked with a seeded RNG; the
        trainer holds no validation environment, so nothing else can enter the fit.
        """
        ordered = sorted(_validate_qids(qids))
        if isinstance(max_episodes, bool) or not isinstance(max_episodes, int) or max_episodes < 1:
            raise ValueError(f"max_episodes must be a positive int; got {max_episodes!r}")
        rng = np.random.default_rng(self.config.seed)
        observations: list[Mapping[str, Any]] = []
        prob_rows: list[tuple[float, float, float]] = []
        for qid in ordered[:max_episodes]:
            obs, _ = self.train_env.reset(qid=qid)
            done = False
            while not done:
                observations.append(obs)
                prob_rows.append(prior_probs(self.prior, obs))
                mask = np.asarray(obs["action_mask"], dtype=bool)
                action = int(rng.choice(np.flatnonzero(mask)))
                obs, _, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated
        self.scaler = V2FeatureScaler.fit(self.extractor.matrix(observations, prob_rows))
        self.scaler_t = TorchScaler(np.asarray(self.scaler.mean, dtype=np.float32),
                                    np.asarray(self.scaler.std, dtype=np.float32),
                                    self.scaler.clip).to(self.device)
        return self.scaler

    def train(self, train_qids: Sequence[str]) -> dict[str, Any]:
        """Run the pinned schedule to the end — no early stop, checkpoints on cadence and end.

        One call per trainer: the plan has a single fixed schedule per arm and no resume, so a
        second call (which would append to the same history under the same identity) is refused.
        """
        if self.history:
            raise RuntimeError("train() already ran on this trainer; the schedule is fixed "
                               "once per arm and this trainer has no resume")
        qids = _validate_qids(train_qids)
        _validate_train_env(self.train_env, qids)
        if self.run_dir is not None and self.run_dir.exists():
            raise FileExistsError(
                f"run directory already exists: {self.run_dir}; refusing to overwrite")
        schedule = build_schedule(qids, epochs=self.config.epochs, seed=self.config.seed)
        batches = split_batches(schedule, self.config.batch_size)
        if len(batches) != self.config.updates:
            raise ValueError(
                f"the pinned schedule yields {len(batches)} update(s) for {len(qids)} QIDs x "
                f"{self.config.epochs} epochs at batch size {self.config.batch_size}, but "
                f"config.updates={self.config.updates}; the protocol expects 240 updates for "
                "the 2,553 train QIDs")
        self.identity = self._identity(qids, schedule)
        if self.net is None:
            self.build()
        if self.scaler is None:
            self.fit_scaler(qids)
        assert self.net is not None and self.scaler is not None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        counts: Counter[str] = Counter()
        for update, batch in enumerate(batches, start=1):
            before = _param_vector(self.net)
            episodes = [rollout(self.train_env, self.net, self.extractor, self.scaler_t,
                                self.prior, qid=qid, device=self.device) for qid in batch]
            loss, stats = reinforce_loss(episodes, self.net, self.scaler_t, self.config,
                                         device=self.device)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
            optimizer.step()
            after = _param_vector(self.net)
            counts.update(episode.qid for episode in episodes)
            action_counts: Counter[str] = Counter(
                ACTION_ORDER[transition.action] for episode in episodes
                for transition in episode.transitions)
            row: dict[str, Any] = {
                "update": update,
                "train_mean_retrieval_return": float(np.mean(
                    [episode.total_reward for episode in episodes])),
                "mean_episode_length": float(np.mean(
                    [len(episode.transitions) for episode in episodes])),
                "unique_qids": len({episode.qid for episode in episodes}),
                "action_counts": {name: int(action_counts.get(name, 0))
                                  for name in ACTION_ORDER},
                "grad_norm": float(grad_norm),
                "param_delta": float(np.abs(after - before).mean()),
                **stats,
            }
            self.history.append(row)
            if self.run_dir is not None:
                append_jsonl(self.run_dir / TRAIN_LOG_FILE, row)
                if update % self.config.checkpoint_every == 0 or update == self.config.updates:
                    self.save(self.run_dir / CHECKPOINT_DIR / f"update_{update:04d}",
                              update=update)
        episode_counts = [counts[qid] for qid in qids]
        summary = {
            "module": "vixstory_policy_v2",
            "identity": self.identity,
            "prior_hash": self.prior_hash,
            "config": asdict(self.config),
            "features": self.extractor.names,
            "schedule_sha256": sha256_json(schedule),
            "n_episodes": len(schedule),
            "n_updates": len(self.history),
            "episodes_per_qid": {"min": min(episode_counts), "max": max(episode_counts),
                                 "distinct_qids": len(counts)},
            "checkpoints": [str(path) for path in self.checkpoints],
            "selection": ("external: greedy validation evaluation + select_best; this trainer "
                          "never sees validation"),
        }
        if self.run_dir is not None:
            atomic_write_json(self.run_dir / SUMMARY_FILE, summary)
        return {"identity": self.identity, "prior_hash": self.prior_hash,
                "schedule_sha256": summary["schedule_sha256"],
                "n_episodes": summary["n_episodes"], "n_updates": summary["n_updates"],
                "episodes_per_qid": summary["episodes_per_qid"],
                "checkpoints": self.checkpoints, "updates": list(self.history)}

    def save(self, path: str | Path, *, update: int | None = None) -> Path:
        """Write policy + train-fitted scaler + checkpoint identity into ``path``."""
        if self.net is None:
            raise RuntimeError("save requires a built network; call build() or train() first")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.net.save(path / POLICY_FILE)
        if self.scaler is not None:
            self.scaler.save(path / SCALER_FILE)
        atomic_write_json(path / CHECKPOINT_FILE, {
            "module": "vixstory_policy_v2",
            "update": update,
            "identity": self.identity,
            "prior_hash": self.prior_hash,
            "n_actions": N_ACTIONS,
            "features": self.extractor.names,
            "config": asdict(self.config),
        })
        if path not in self.checkpoints:
            self.checkpoints.append(path)
        return path

    def load(self, path: str | Path) -> None:
        """Restore policy + scaler; four-action or foreign-prior checkpoints are refused."""
        path = Path(path)
        policy_path = path / POLICY_FILE
        scaler_path = path / SCALER_FILE
        if not policy_path.is_file() or not scaler_path.is_file():
            raise ValueError(f"checkpoint {path} must hold {POLICY_FILE} and {SCALER_FILE}")
        net = PolicyNet.load(policy_path)
        if net.n_actions != N_ACTIONS:
            raise ValueError(f"checkpoint {path} is a {net.n_actions}-action policy; V2 has "
                             f"exactly {N_ACTIONS}")
        if net.d_in != self.extractor.dim:
            raise ValueError(f"checkpoint feature width {net.d_in} != V2 extractor width "
                             f"{self.extractor.dim}")
        checkpoint_path = path / CHECKPOINT_FILE
        if checkpoint_path.is_file():
            recorded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if recorded.get("prior_hash") not in (None, self.prior_hash):
                raise ValueError(
                    f"checkpoint {path} was trained with frozen prior "
                    f"{recorded.get('prior_hash')!r}, not {self.prior_hash!r}; refusing to "
                    "evaluate with a different prior")
        scaler = V2FeatureScaler.load(scaler_path)
        if len(scaler.mean) != self.extractor.dim or len(scaler.std) != self.extractor.dim:
            raise ValueError(f"checkpoint scaler width {len(scaler.mean)} != V2 extractor "
                             f"width {self.extractor.dim}")
        self.net = net.to(self.device)
        self.scaler = scaler
        self.scaler_t = TorchScaler(np.asarray(scaler.mean, dtype=np.float32),
                                    np.asarray(scaler.std, dtype=np.float32),
                                    scaler.clip).to(self.device)

    def _identity(self, qids: Sequence[str], schedule: Sequence[str]) -> str:
        return sha256_json({
            "module": "vixstory_policy_v2",
            "features": self.extractor.names,
            "config": asdict(self.config),
            "prior_hash": self.prior_hash,
            "train_qids_sha256": sha256_json(sorted(qids)),
            "schedule_sha256": sha256_json(schedule),
        })

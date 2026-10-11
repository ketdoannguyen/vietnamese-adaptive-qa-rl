"""Sequential REINFORCE on episode return.

    L = -1/N * sum_episodes sum_t (G_t - b(s_t)) log pi(a_t|s_t) - beta * H(pi)
        + value_coef * (V(s_t) - G_t)^2

Transitions (retrieval rankings, reader outputs) come from immutable caches and are
deterministic; **actions are always sampled from the current policy**, so the rollouts
are genuinely on-policy. Rule (spec 05 §8): caching full trajectories as a substitute
for sampling would not be REINFORCE, and is not done here.

Value-head advantages are detached for the policy term; gradients never flow into the
frozen Laya probabilities or into cached features.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from src.rl.env import AdaptiveQAEnv
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import N_ACTIONS, PolicyNet
from src.utils.io import append_jsonl


@dataclass
class RLConfig:
    algorithm: str = "reinforce"
    seed: int = 42
    gamma: float = 1.0
    hidden_dim: int = 128
    lr: float = 3e-4
    batch_episodes: int = 32
    max_updates: int = 500
    eval_every: int = 25
    eval_episodes: int = 64
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    grad_clip: float = 1.0
    early_stop_patience: int = 6
    checkpoint_every: int = 25

    @classmethod
    def from_dict(cls, cfg: dict) -> "RLConfig":
        raw = cfg.get("rl", cfg)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class Transition:
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


def discounted_returns(rewards: Sequence[float], gamma: float) -> np.ndarray:
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def rollout(env: AdaptiveQAEnv, net: PolicyNet, extractor: FeatureExtractor,
            scaler_t: Any | None, laya_fn: Callable[[dict], Sequence[float]] | None,
            *, seed: int | None = None, qid: str | None = None,
            greedy: bool = False, device: torch.device | None = None) -> Episode:
    """One episode. Actions always come from ``net`` — never from a cache."""
    device = device or torch.device("cpu")
    options = {"qid": qid} if qid else None
    obs, _ = env.reset(seed=seed, options=options)
    episode = Episode(qid=obs["qid"])
    done = False
    while not done:
        # The mask recorded on the transition must be the one the action was chosen
        # under — recomputing it after the step would train on a different state.
        action_mask = np.asarray(obs["action_mask"], dtype=bool)
        probs = laya_fn(obs) if laya_fn is not None else None
        features = extractor.vector(obs, probs)
        x = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        if scaler_t is not None:  # the one and only standardisation point
            x = scaler_t(x)
        mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(x)
            masked = logits.masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=masked)
            action = int(torch.argmax(masked, dim=-1).item()) if greedy else int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor([action], device=device)).item())
            entropy = float(dist.entropy().item())
        obs, reward, terminated, truncated, _ = env.step(action)
        episode.transitions.append(Transition(
            features=features, mask=action_mask, action=action, log_prob=log_prob,
            value=float(value.item()), entropy=entropy, reward=float(reward)))
        episode.total_reward += float(reward)
        done = terminated or truncated
    return episode


def reinforce_loss(episodes: list[Episode], net: PolicyNet, scaler_t: Any | None,
                   config: RLConfig, *, device: torch.device) -> tuple[torch.Tensor, dict]:
    """Policy-gradient loss with a value baseline. Sign checked by tests."""
    policy_terms, value_terms, entropies = [], [], []
    n_steps = 0
    for episode in episodes:
        rewards = [t.reward for t in episode.transitions]
        returns = discounted_returns(rewards, config.gamma)
        for t, transition in enumerate(episode.transitions):
            # ``transition.features`` is raw; standardise here, exactly as rollout did.
            x = torch.as_tensor(transition.features, dtype=torch.float32,
                                device=device).unsqueeze(0)
            if scaler_t is not None:
                x = scaler_t(x)
            mask_t = torch.as_tensor(transition.mask, dtype=torch.bool,
                                     device=device).unsqueeze(0)
            logits, value = net(x)
            masked = logits.masked_fill(~mask_t, -1e9)
            dist = torch.distributions.Categorical(logits=masked)
            action_t = torch.tensor([transition.action], device=device)
            log_prob = dist.log_prob(action_t)
            entropy = dist.entropy()
            g_t = torch.tensor([returns[t]], dtype=torch.float32, device=device)
            advantage = g_t - value.detach()          # detached: no gradient into baseline
            policy_terms.append(-(advantage * log_prob).squeeze(0))
            value_terms.append((value - g_t).pow(2).squeeze(0))
            entropies.append(entropy.squeeze(0))
            n_steps += 1
    if n_steps == 0:
        raise ValueError("no transitions collected; cannot compute a policy gradient")
    policy_loss = torch.stack(policy_terms).mean()
    value_loss = torch.stack(value_terms).mean()
    entropy = torch.stack(entropies).mean()
    loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
    stats = {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "entropy": float(entropy.item()),
        "n_steps": n_steps,
    }
    return loss, stats


def evaluate(env: AdaptiveQAEnv, net: PolicyNet, extractor: FeatureExtractor,
             scaler_t: Any | None, laya_fn, *, episodes: int, device: torch.device,
             seed_base: int = 10_000, qids: Sequence[str] | None = None) -> dict[str, Any]:
    """Deterministic greedy evaluation, optionally over fixed, unique question IDs."""
    if qids is not None and (len(qids) != episodes or len(set(qids)) != len(qids)):
        raise ValueError("fixed evaluation qids must match episodes and contain no duplicates")
    rewards, lengths, actions = [], [], np.zeros(N_ACTIONS, dtype=int)
    for i in range(episodes):
        ep = rollout(env, net, extractor, scaler_t, laya_fn, seed=seed_base + i,
                     qid=qids[i] if qids is not None else None, greedy=True, device=device)
        rewards.append(ep.total_reward)
        lengths.append(len(ep.transitions))
        for t in ep.transitions:
            actions[t.action] += 1
    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "episodes": episodes,
        "action_counts": {str(i): int(actions[i]) for i in range(N_ACTIONS)},
    }


class ReinforceTrainer:
    def __init__(self, train_env: AdaptiveQAEnv, val_env: AdaptiveQAEnv, config: RLConfig,
                 *, laya_fn=None, device: str = "cpu", run_dir: str | Path | None = None,
                 seed: int | None = None, val_qids: Sequence[str] | None = None) -> None:
        self.train_env = train_env
        self.val_env = val_env
        self.config = config
        self.laya_fn = laya_fn
        self.device = torch.device(device)
        self.seed = config.seed if seed is None else seed
        self.val_qids = tuple(val_qids) if val_qids is not None else None
        self.run_dir = Path(run_dir) if run_dir else None
        self.extractor = FeatureExtractor()
        self.net: PolicyNet | None = None
        self.scaler: FeatureScaler | None = None
        self.scaler_t = None
        self.history: list[dict] = []

    # -- setup ---------------------------------------------------------------
    def fit_scaler(self, *, episodes: int = 200) -> FeatureScaler:
        """Fit the standardiser on TRAIN rollouts only, with a neutral policy prior."""
        obs_rows, prob_rows = [], []
        rng = np.random.default_rng(self.seed)
        for i in range(episodes):
            obs, _ = self.train_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            done = False
            while not done:
                obs_rows.append(obs)
                probs = self.laya_fn(obs) if self.laya_fn is not None else None
                prob_rows.append(list(probs) if probs is not None else [0.25] * N_ACTIONS)
                mask = np.asarray(obs["action_mask"], dtype=bool)
                action = int(rng.choice(np.flatnonzero(mask)))
                obs, _, terminated, truncated, _ = self.train_env.step(action)
                done = terminated or truncated
        matrix = self.extractor.matrix(obs_rows, prob_rows)
        self.scaler = FeatureScaler.fit(matrix)
        from src.rl.policy import TorchScaler
        self.scaler_t = TorchScaler(np.asarray(self.scaler.mean, dtype=np.float32),
                                    np.asarray(self.scaler.std, dtype=np.float32),
                                    self.scaler.clip)
        return self.scaler

    def build(self, d_in: int | None = None) -> PolicyNet:
        d_in = d_in or self.extractor.dim
        torch.manual_seed(self.seed)
        self.net = PolicyNet(d_in, self.config.hidden_dim).to(self.device)
        return self.net

    # -- training ------------------------------------------------------------
    def train(self) -> dict[str, Any]:
        if self.net is None:
            self.build()
        if self.scaler is None:
            self.fit_scaler()
        assert self.net is not None and self.scaler is not None

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        rng = np.random.default_rng(self.seed)
        best_val = float("-inf")
        best_update = 0
        patience = 0
        result: dict[str, Any] = {"updates": [], "best_val_reward": None,
                                  "best_update": None, "stopped_early": False}

        for update in range(1, self.config.max_updates + 1):
            before = _param_vector(self.net)
            episodes: list[Episode] = []
            for _ in range(self.config.batch_episodes):
                seed = int(rng.integers(0, 2**31 - 1))
                episodes.append(rollout(self.train_env, self.net, self.extractor,
                                        self.scaler_t, self.laya_fn, seed=seed,
                                        device=self.device))
            loss, stats = reinforce_loss(episodes, self.net, self.scaler_t, self.config,
                                         device=self.device)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
            optimizer.step()

            after = _param_vector(self.net)
            delta = float(np.abs(after - before).mean())
            train_reward = float(np.mean([e.total_reward for e in episodes]))
            row = {
                "update": update,
                "train_mean_reward": train_reward,
                "grad_norm": float(grad_norm),
                "param_delta": delta,
                **stats,
            }

            if update % self.config.eval_every == 0:
                kwargs = {"episodes": self.config.eval_episodes, "device": self.device}
                if self.val_qids is not None:
                    kwargs.update(episodes=len(self.val_qids), qids=self.val_qids)
                val = evaluate(self.val_env, self.net, self.extractor, self.scaler_t,
                               self.laya_fn, **kwargs)
                row["val"] = val
                if val["mean_reward"] > best_val + 1e-6:
                    best_val = val["mean_reward"]
                    best_update = update
                    patience = 0
                    if self.run_dir:
                        self.save(self.run_dir / "best")
                else:
                    patience += 1
                    if patience >= self.config.early_stop_patience:
                        result["stopped_early"] = True
                        if self.run_dir:
                            self.save(self.run_dir / "last")
                            append_jsonl(self.run_dir / "train_log.jsonl", row)
                        self.history.append(row)
                        result["updates"].append(row)
                        break
            if self.run_dir:
                self.save(self.run_dir / "last")
                append_jsonl(self.run_dir / "train_log.jsonl", row)
            self.history.append(row)
            result["updates"].append(row)

        result["best_val_reward"] = best_val if best_val > float("-inf") else None
        result["best_update"] = best_update or None
        return result

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self.net is not None
        self.net.save(path / "policy.safetensors")
        if self.scaler is not None:
            self.scaler.save(path / "scaler.json")
        (path / "rl_config.json").write_text(
            json.dumps(self.config.__dict__, indent=2), encoding="utf-8")

    def load_best(self, path: str | Path) -> None:
        path = Path(path)
        self.net = PolicyNet.load(path / "policy.safetensors").to(self.device)
        scaler_path = path / "scaler.json"
        if scaler_path.exists():
            self.scaler = FeatureScaler.load(scaler_path)
            from src.rl.policy import TorchScaler
            self.scaler_t = TorchScaler(np.asarray(self.scaler.mean, dtype=np.float32),
                                        np.asarray(self.scaler.std, dtype=np.float32),
                                        self.scaler.clip)


def _param_vector(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])

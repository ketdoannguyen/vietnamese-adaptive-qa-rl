"""Finite-horizon privileged oracle — exact action-label generation for Laya.

Spec: `01_SPECIFICATIONS/07_TEACHER_ORACLE_COMPLETE_ALGORITHM.md`.

    Q(s, ANSWER/ABSTAIN) = terminal_utility(s, a)
    Q(s, SEARCH_*)       = -search_cost(a) + max_a' Q(next_state(s, a), a')
    V(s)                 = max over legal actions

The oracle may look at gold labels — that is precisely what makes it privileged. It is
**train-only** imitation data, never a deployable policy and never a test result. The
state handed to Laya is rendered exclusively from the public observation.

Because transitions are deterministic, a state is uniquely identified by the sequence of
actions taken since ``reset``; replaying that sequence through the real (already tested)
environment reconstructs it exactly. This keeps the oracle and the RL environment from
drifting apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from src.decision.schema import ACTION_ORDER, assert_public_state_is_clean, render_state
from src.rl.env import ABSTAIN, ANSWER, AdaptiveQAEnv, EnvConfig
from src.rl.reward import RewardCalculator
from src.utils.hashing import sha256_json

#: Deterministic tie-break order (spec 07 §5).
TIE_ORDER = ("ANSWER", "ABSTAIN", "SEARCH_BM25", "SEARCH_DENSE")
TIE_TOLERANCE = 1e-8


@dataclass
class StateTarget:
    qid: str
    history: tuple[int, ...]
    state_key: str
    state_text: str
    options: list[str]
    q_values: dict[str, float]
    expected: str
    soft: dict[str, float]
    valid_actions: list[str]
    tied: bool = False
    depth: int = 0


@dataclass
class OracleConfig:
    mode: str = "soft"                 # soft | hard
    temperature: float = 0.5
    max_states_per_question: int = 8
    tie_order: tuple[str, ...] = TIE_ORDER
    tie_tolerance: float = TIE_TOLERANCE

    def spec_hash(self) -> str:
        return sha256_json({"mode": self.mode, "temperature": self.temperature,
                            "tie_order": list(self.tie_order),
                            "tie_tolerance": self.tie_tolerance})


@dataclass
class OracleEpisode:
    qid: str
    targets: list[StateTarget] = field(default_factory=list)
    n_ties: int = 0
    max_depth_reached: int = 0


def _snapshot(env: AdaptiveQAEnv) -> tuple:
    return (list(env._seen), env._optional_used, env._n_bm25, env._n_dense,
            list(env._history), env._t, env._candidate)


def _restore(env: AdaptiveQAEnv, snap: tuple) -> None:
    (seen, opt, n_bm25, n_dense, history, t, candidate) = snap
    env._seen = list(seen)
    env._optional_used = opt
    env._n_bm25 = n_bm25
    env._n_dense = n_dense
    env._history = list(history)
    env._t = t
    env._candidate = candidate


def _q_values(env: AdaptiveQAEnv, depth: int) -> tuple[dict[str, float], bool]:
    """Q(s, ·) for the environment's current state. Returns (values, tied)."""
    mask = env._action_mask()
    values: dict[str, float] = {}
    for action in range(len(ACTION_ORDER)):
        if not mask[action]:
            continue
        snap = _snapshot(env)
        name = ACTION_ORDER[action]
        if action in (ANSWER, ABSTAIN):
            env.step(action)
            breakdown = env.private_outcome()
            values[name] = float(breakdown["terminal_utility"])
        else:
            _, step_reward, _, _, _ = env.step(action)
            if depth <= 0:
                # Budget exhausted inside the recursion: the only remaining reachable
                # value is whatever the resulting state can still yield.
                sub, _ = _q_values(env, 0)
            else:
                sub, _ = _q_values(env, depth - 1)
            values[name] = float(step_reward) + (max(sub.values()) if sub else 0.0)
        _restore(env, snap)
    if not values:
        raise AssertionError(f"no legal action at qid={env._world.qid}; ABSTAIN must be legal")
    top = max(values.values())
    tied = sum(1 for v in values.values() if abs(v - top) <= TIE_TOLERANCE) > 1
    return values, tied


def _soft_distribution(values: dict[str, float], config: OracleConfig) -> dict[str, float]:
    """Softmax over legal actions only; masked actions receive exactly zero (spec 07 §4)."""
    best = max(values.values())
    temperature = max(config.temperature, 1e-6)
    exps = {name: math.exp((value - best) / temperature) for name, value in values.items()}
    total = sum(exps.values())
    return {name: exps[name] / total for name in ACTION_ORDER if name in exps}


def _pick_expected(values: dict[str, float], tie_order: tuple[str, ...],
                   tolerance: float = TIE_TOLERANCE) -> str:
    top = max(values.values())
    for name in tie_order:
        if name in values and abs(values[name] - top) <= tolerance:
            return name
    raise AssertionError("no action matched the maximum")


def oracle_episode(env: AdaptiveQAEnv, qid: str, config: OracleConfig,
                   *, seed: int = 0) -> OracleEpisode:
    """Enumerate the reachable state tree for one question and label every state."""
    obs, _ = env.reset(seed=seed, options={"qid": qid})
    episode = OracleEpisode(qid=qid)

    def walk(history: tuple[int, ...], observed: dict, depth: int) -> None:
        assert_public_state_is_clean(observed)
        values, tied = _q_values(env, max(env.config.max_optional_search - env._optional_used, 0))
        expected = _pick_expected(values, config.tie_order, config.tie_tolerance)
        soft = _soft_distribution(values, config)
        state_text = render_state(observed)
        state_key = sha256_json({
            "qid": qid,
            "history": list(history),
            "reader_cache_sha": getattr(env._world, "reader_sha", ""),
            "state_serializer_version": "v1",
        })
        episode.targets.append(StateTarget(
            qid=qid, history=history, state_key=state_key, state_text=state_text,
            options=[n for n in ACTION_ORDER if n in values],
            q_values=values, expected=expected if config.mode == "hard" else expected,
            soft=soft, valid_actions=list(values), tied=tied, depth=depth))
        episode.n_ties += int(tied)
        episode.max_depth_reached = max(episode.max_depth_reached, depth)

        if len(episode.targets) >= config.max_states_per_question:
            return
        mask = env._action_mask()
        for action in (1, 2):  # SEARCH_BM25, SEARCH_DENSE
            if not mask[action]:
                continue
            snap = _snapshot(env)
            next_obs, _, terminated, truncated, _ = env.step(action)
            if not (terminated or truncated) and len(episode.targets) < config.max_states_per_question:
                walk(history + (action,), next_obs, depth + 1)
            _restore(env, snap)

    walk((), obs, 0)
    return episode


def iter_episodes(worlds, reward: RewardCalculator, env_config: EnvConfig,
                  oracle_config: OracleConfig, *, seed: int = 0) -> Iterator[OracleEpisode]:
    env = AdaptiveQAEnv(list(worlds), reward, env_config)
    for world in worlds:
        yield oracle_episode(env, world.qid, oracle_config, seed=seed)


def verify_against_hand_enumeration(episode: OracleEpisode,
                                    expected: dict[tuple[int, ...], dict[str, float]],
                                    *, tolerance: float = 1e-6) -> None:
    """Quality gate (spec 07 §6.1): oracle output must match a hand-computed answer."""
    by_history = {t.history: t for t in episode.targets}
    for history, expected_values in expected.items():
        target = by_history.get(history)
        if target is None:
            raise AssertionError(f"oracle never visited state {history}")
        for name, value in expected_values.items():
            if name not in target.q_values:
                raise AssertionError(f"state {history}: missing action {name}")
            got = target.q_values[name]
            if abs(got - value) > tolerance:
                raise AssertionError(
                    f"state {history} action {name}: oracle={got}, hand={value}")

    for target in episode.targets:
        assert target.q_values, "every state must have at least one legal action"
        for name, value in target.q_values.items():
            assert math.isfinite(value), f"non-finite Q at {target.history}/{name}"
        total = sum(target.soft.values())
        assert abs(total - 1.0) < 1e-6, f"soft labels sum to {total}, not 1"
        for name in ACTION_ORDER:
            if name not in target.valid_actions:
                assert target.soft.get(name, 0.0) == 0.0, \
                    f"masked action {name} received probability mass"

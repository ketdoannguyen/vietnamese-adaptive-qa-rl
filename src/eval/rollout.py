"""Shared episode runner for the P7 baseline policies (`03_EXPERIMENTS/00...` §1-§2).

One runner executes every non-MLP baseline so the fairness checklist holds by construction:
identical worlds, environment config (mask, horizon, forced bootstrap, optional budget) and
reward calculator for every row. Actions always come from the policy *online* — no fixed
action log is replayed, and the environment is stepped here rather than inside a policy.

The episode record separates two column groups and the split is enforced by tests:

* :data:`FEATURE_COLUMNS` — decision-side data only: actions, action masks, the reader
  candidate seen at each decision, retrieval preview scores, budget counters, cost
  arithmetic. Values here are a function of the visible state, so mutating the hidden gold
  side of an episode must not move a single one (``tests/test_baselines.py``).
* :data:`EVAL_COLUMNS` — gold-derived evaluation outputs, prefixed ``eval_``, written to
  prediction/result artifacts only and never readable by a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from src.rl.env import AdaptiveQAEnv, EpisodeWorld
from src.rl.reward import ACTIONS
from src.utils.hashing import stable_seed

#: Non-gold columns of ``outputs/predictions/{run_id}_{split}.parquet``.
FEATURE_COLUMNS: tuple[str, ...] = (
    "run_id", "exp_id", "policy", "policy_variant", "prior_regime", "split", "dev", "seed",
    "qid", "scenario_id", "steps", "action_trace", "action_name_trace", "action_mask_trace",
    "candidate_trace", "final_candidate", "final_candidate_doc_id",
    "final_candidate_provenance_ok", "optional_used", "bm25_calls", "dense_calls",
    "total_retrievals", "retrieval_cost", "terminal_action", "forced_terminal",
    "confidence_p_correct",
)

#: Gold-derived columns, prefix ``eval_``. Never a policy input, never a "feature column".
EVAL_COLUMNS: tuple[str, ...] = (
    "eval_step_rewards", "eval_episode_return", "eval_correct", "eval_correctness_f1",
    "eval_correctness_em", "eval_terminal_utility", "eval_is_impossible", "eval_gold_reachable",
)

#: Search actions (indices into ``ACTIONS``): everything else is terminal.
SEARCH_ACTIONS = (1, 2)


class Policy(Protocol):
    """What :func:`run_policy` needs from a policy; implemented by :mod:`src.eval.policies`."""

    id: str
    variant: str

    def begin_episode(self, *, qid: str, seed: int) -> None:
        """Reset per-episode state (random policies seed their generator here)."""

    def act(self, obs: Mapping[str, Any]) -> int:
        """Next action for a *visible* observation. Must pick a legal action."""


@dataclass
class StepRecord:
    """One decision point: the visible state the action was chosen from, then the outcome."""

    step: int
    action: int
    action_name: str
    reward: float
    mask: tuple[bool, ...]
    optional_used: int
    remaining_optional: int
    candidate_text: str | None
    candidate_doc_id: str | None
    provenance_ok: bool
    bm25_top_score: float
    dense_top_score: float


@dataclass
class EpisodeRecord:
    """One finished episode, decision side and gold side kept apart."""

    run_id: str
    exp_id: str
    policy: str
    policy_variant: str
    prior_regime: str
    split: str
    dev: bool
    seed: int
    qid: str
    scenario_id: str
    steps: list[StepRecord] = field(default_factory=list)
    episode_return: float = 0.0
    terminal_action: str = ""
    forced_terminal: bool = False
    confidence_p_correct: float | None = None
    # -- gold-derived (eval artifacts only) ------------------------------------------
    eval_correct: int = 0
    eval_correctness_f1: float = 0.0
    eval_correctness_em: float = 0.0
    eval_terminal_utility: float | None = None
    eval_is_impossible: bool = False
    eval_gold_reachable: bool | None = None
    retrieval_cost: float = 0.0
    bm25_calls: int = 0
    dense_calls: int = 0
    optional_used: int = 0

    @property
    def final_candidate(self) -> str | None:
        """The candidate the policy acted on at its last decision state."""
        return self.steps[-1].candidate_text if self.steps else None

    @property
    def final_candidate_doc_id(self) -> str | None:
        return self.steps[-1].candidate_doc_id if self.steps else None

    @property
    def final_candidate_provenance_ok(self) -> bool:
        return bool(self.steps and self.steps[-1].provenance_ok)

    @property
    def answered(self) -> bool:
        return self.terminal_action == ACTIONS[0]

    @property
    def abstained(self) -> bool:
        return self.terminal_action == ACTIONS[3]


def run_episode(env: AdaptiveQAEnv, policy: Policy, world: EpisodeWorld, *, seed: int, run_id: str,
                exp_id: str, dev: bool = False,
                prior_regime: str | None = None) -> EpisodeRecord:
    """Step one episode with ``policy``. Illegal actions raise instead of being tolerated."""
    obs, _ = env.reset(seed=stable_seed("p7", seed, world.qid), options={"qid": world.qid})
    policy.begin_episode(qid=world.qid, seed=seed)
    record = EpisodeRecord(
        run_id=run_id, exp_id=exp_id, policy=policy.id, policy_variant=policy.variant,
        prior_regime=prior_regime or "none", split=world.split, dev=bool(dev), seed=int(seed),
        qid=world.qid, scenario_id=world.scenario_id,
        eval_is_impossible=bool(world.private.is_impossible),
    )
    terminated = truncated = False
    while not (terminated or truncated):
        mask = np.asarray(obs["action_mask"], dtype=bool)
        action = int(policy.act(obs))
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"{policy.id}: action {action} outside [0, {len(ACTIONS)})")
        if not mask[action]:
            raise ValueError(
                f"{policy.id} proposed illegal action {ACTIONS[action]} for qid={world.qid} at "
                f"t={obs.get('timestep')} (valid: "
                f"{[ACTIONS[i] for i, ok in enumerate(mask) if ok]})")
        candidate = obs.get("candidate") or {}
        retrieval = obs.get("retrieval") or {}
        record.steps.append(StepRecord(
            step=int(obs.get("timestep", len(record.steps))), action=action,
            action_name=ACTIONS[action], reward=0.0, mask=tuple(bool(m) for m in mask),
            optional_used=int(obs.get("optional_used", 0)),
            remaining_optional=int(obs.get("remaining_optional_searches", 0)),
            candidate_text=candidate.get("text"), candidate_doc_id=candidate.get("doc_id"),
            provenance_ok=bool(candidate.get("provenance_ok")),
            bm25_top_score=float(retrieval.get("bm25_top_score", 0.0)),
            dense_top_score=float(retrieval.get("dense_top_score", 0.0)),
        ))
        obs, reward, terminated, truncated, _ = env.step(action)
        record.steps[-1].reward = float(reward)
        record.episode_return += float(reward)

    outcome = env.private_outcome() or {}
    record.optional_used = sum(1 for step in record.steps if step.action in SEARCH_ACTIONS)
    record.bm25_calls = int(outcome.get("n_bm25_searches", 0))
    record.dense_calls = int(outcome.get("n_dense_searches", 0))
    record.retrieval_cost = float(outcome.get("search_cost_total", 0.0))
    record.forced_terminal = bool(truncated)
    record.terminal_action = "TRUNCATED" if truncated else str(outcome.get("terminal_action", ""))
    record.confidence_p_correct = getattr(policy, "last_confidence", None)
    if not truncated:
        record.eval_terminal_utility = float(outcome.get("terminal_utility", 0.0))
    reward_config = env.reward.config
    record.eval_correct = int(
        not truncated and record.terminal_action == ACTIONS[0]
        and record.eval_terminal_utility == reward_config.correct_answer)
    record.eval_correctness_f1 = float(outcome.get("correctness_f1", 0.0))
    record.eval_correctness_em = float(outcome.get("correctness_em", 0.0))
    record.eval_gold_reachable = outcome.get("gold_reachable")
    return record


def run_policy(env: AdaptiveQAEnv, policy: Policy, *, worlds: Sequence[EpisodeWorld], seed: int,
               run_id: str, exp_id: str, dev: bool = False,
               prior_regime: str | None = None) -> list[EpisodeRecord]:
    """Every episode of one result row, in a deterministic (sorted by qid) order."""
    return [run_episode(env, policy, world, seed=seed, run_id=run_id, exp_id=exp_id, dev=dev,
                        prior_regime=prior_regime)
            for world in sorted(worlds, key=lambda w: w.qid)]


def episode_row(record: EpisodeRecord) -> dict[str, Any]:
    """The prediction-artifact row: :data:`FEATURE_COLUMNS` then :data:`EVAL_COLUMNS`."""
    row = {
        "run_id": record.run_id, "exp_id": record.exp_id, "policy": record.policy,
        "policy_variant": record.policy_variant, "prior_regime": record.prior_regime,
        "split": record.split, "dev": bool(record.dev), "seed": int(record.seed),
        "qid": record.qid, "scenario_id": record.scenario_id, "steps": len(record.steps),
        "action_trace": [step.action for step in record.steps],
        "action_name_trace": [step.action_name for step in record.steps],
        "action_mask_trace": [list(step.mask) for step in record.steps],
        "candidate_trace": [step.candidate_text for step in record.steps],
        "final_candidate": record.final_candidate,
        "final_candidate_doc_id": record.final_candidate_doc_id,
        "final_candidate_provenance_ok": record.final_candidate_provenance_ok,
        "optional_used": int(record.optional_used), "bm25_calls": int(record.bm25_calls),
        "dense_calls": int(record.dense_calls),
        "total_retrievals": int(record.bm25_calls + record.dense_calls),
        "retrieval_cost": float(record.retrieval_cost),
        "terminal_action": record.terminal_action, "forced_terminal": bool(record.forced_terminal),
        "confidence_p_correct": record.confidence_p_correct,
        "eval_step_rewards": [float(step.reward) for step in record.steps],
        "eval_episode_return": float(record.episode_return),
        "eval_correct": int(record.eval_correct),
        "eval_correctness_f1": float(record.eval_correctness_f1),
        "eval_correctness_em": float(record.eval_correctness_em),
        "eval_terminal_utility": record.eval_terminal_utility,
        "eval_is_impossible": bool(record.eval_is_impossible),
        "eval_gold_reachable": record.eval_gold_reachable,
    }
    return {column: row[column] for column in FEATURE_COLUMNS + EVAL_COLUMNS}

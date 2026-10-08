"""AdaptiveQAEnv — finite-horizon episodic QA decision environment.

Contract: `01_SPECIFICATIONS/05_MDP_ENVIRONMENT_CONTRACT.md`.

    action 0 ANSWER | 1 SEARCH_BM25 | 2 SEARCH_DENSE | 3 ABSTAIN

The environment holds the *private* side of the episode (gold answers, answerability,
the reward calculator). The policy only ever sees the observation returned by
``reset``/``step``. ``info`` carries no labels and no reward breakdown — diagnostic
detail is available through :meth:`AdaptiveQAEnv.private_outcome`, which training code
must not wire into features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from src.qa.reader import NULL_PREDICTION, ReaderPrediction
from src.rl.reward import ACTIONS, EpisodePrivate, RewardBreakdown, RewardCalculator
from src.utils.hashing import sha256_json

N_ACTIONS = 4
ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN = 0, 1, 2, 3


@dataclass(frozen=True)
class EnvConfig:
    max_optional_search: int = 2
    bootstrap_tool: str = "bm25"
    bootstrap_passages: int = 1
    passage_per_search_action: int = 1
    #: Expose the top retrieval score of the *next unseen* passage. This is a preview of
    #: information the agent has not paid for; it is applied identically to every policy,
    #: and `strict_no_preview` turns it off for the ablation required by spec 09 §6.
    preview_unseen_scores: bool = True
    #: Watchdog only. Reaching it produces truncated=True, never a fake terminal reward.
    max_steps: int = 16
    #: Dev/test guard: raise instead of tolerating an illegal action.
    strict_invalid_actions: bool = True


@dataclass
class EpisodeWorld:
    """Immutable per-episode inputs assembled from caches."""

    qid: str
    question: str
    split: str
    scenario_id: str
    corpus: Mapping[str, str]
    rankings: Mapping[str, Sequence[str]]
    private: EpisodePrivate
    reader: Callable[[tuple[str, ...]], ReaderPrediction]
    scenario_version: str = "v1"

    def __post_init__(self) -> None:
        for tool, ranking in self.rankings.items():
            unknown = [d for d in ranking if d not in self.corpus]
            if unknown:
                raise ValueError(
                    f"ranking for tool {tool!r} references {len(unknown)} doc(s) "
                    f"outside the episode corpus, e.g. {unknown[0]!r}")

    @property
    def corpus_sha(self) -> str:
        return sha256_json({"corpus": sorted(self.corpus.items()),
                            "scenario_id": self.scenario_id,
                            "scenario_version": self.scenario_version})


class AdaptiveQAEnv(gym.Env):
    """The RL environment. One instance per worker; ``reset`` selects the episode."""

    metadata = {"render_modes": []}

    def __init__(self, worlds: Sequence[EpisodeWorld], reward: RewardCalculator,
                 config: EnvConfig | None = None) -> None:
        super().__init__()
        if not worlds:
            raise ValueError("AdaptiveQAEnv requires at least one EpisodeWorld")
        self._worlds = list(worlds)
        self._by_qid = {w.qid: w for w in worlds}
        if len(self._by_qid) != len(self._worlds):
            raise ValueError("duplicate qid in episode worlds")
        self.reward = reward
        self.config = config or EnvConfig()
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Dict({
            "question": gym.spaces.Text(max_length=4096),
            "action_mask": gym.spaces.MultiBinary(N_ACTIONS),
        })
        self._world: EpisodeWorld | None = None
        self._seen: list[str] = []
        self._optional_used = 0
        self._n_bm25 = 0
        self._n_dense = 0
        self._history: list[int] = []
        self._candidate: ReaderPrediction = NULL_PREDICTION
        self._t = 0
        self._rng: np.random.Generator = np.random.default_rng(0)
        self._breakdown: RewardBreakdown | None = None
        self._terminal_reason: str | None = None

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        qid = options.get("qid")
        if qid is not None:
            if qid not in self._by_qid:
                raise KeyError(f"qid {qid!r} is not part of this environment's worlds")
            world = self._by_qid[qid]
        else:
            idx = int(self._rng.integers(0, len(self._worlds)))
            world = self._worlds[idx]
        self._world = world

        self._seen = []
        self._optional_used = 0
        self._n_dense = 0
        self._history = []
        self._candidate = NULL_PREDICTION
        self._t = 0
        self._breakdown = None
        self._terminal_reason = None

        # Forced bootstrap: one retrieval before the agent's first decision. Counted for
        # every policy (spec 05 §2), so it is charged against every episode's return.
        self._n_bm25 = 1
        boot = self._next_unseen(self.config.bootstrap_tool, self.config.bootstrap_passages)
        self._seen.extend(boot)
        self._refresh_candidate()

        obs = self._observation()
        info = self._safe_info()
        return obs, info

    # ----------------------------------------------------------------- step
    def step(self, action: int):
        if self._world is None:
            raise RuntimeError("step() called before reset()")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} outside [0, {N_ACTIONS})")

        mask = self._action_mask()
        if not mask[action]:
            if self.config.strict_invalid_actions:
                raise ValueError(
                    f"invalid action {ACTIONS[action]} at t={self._t} "
                    f"(mask={mask.astype(int).tolist()}); valid={self._valid_action_names()}")
            # Production tolerance path: documented, logged, never silent.
            self._history.append(action)
            self._t += 1
            obs = self._observation()
            info = self._safe_info()
            info["invalid_action_penalised"] = True
            return obs, self.reward.config.wrong_answer, False, False, info

        self._history.append(action)
        self._t += 1

        if action == ANSWER:
            return self._terminal(ANSWER)
        if action == ABSTAIN:
            return self._terminal(ABSTAIN)

        # Search actions: charge the cost now, reveal one previously unseen passage.
        tool = "bm25" if action == SEARCH_BM25 else "dense"
        if tool == "bm25":
            self._n_bm25 += 1
        else:
            self._n_dense += 1
        self._optional_used += 1
        new_docs = self._next_unseen(tool, self.config.passage_per_search_action)
        self._seen.extend(new_docs)
        self._refresh_candidate()

        step_reward = -self.reward.search_cost(ACTIONS[action])
        truncated = self._t >= self.config.max_steps
        info = self._safe_info()
        if truncated:
            self._terminal_reason = "watchdog_max_steps"
            info["truncated_reason"] = "watchdog_max_steps"
            self._breakdown = self._make_breakdown("ABSTAIN")
        return self._observation(), step_reward, False, truncated, info

    # ------------------------------------------------------------- internals
    def _terminal(self, action: int):
        assert self._world is not None
        name = ACTIONS[action]
        candidate_text = self._candidate.answer_text
        provenance_ok = bool(self._candidate.provenance_ok and candidate_text is not None
                             and self._candidate.doc_id in self._seen)
        self._breakdown = self.reward.terminal_utility(
            action=name,
            candidate_text=candidate_text,
            provenance_ok=provenance_ok,
            private=self._world.private,
            n_bm25=self._n_bm25,
            n_dense=self._n_dense,
        )
        # Search costs are already paid on their own steps, so the terminal step carries
        # the utility plus only the forced bootstrap cost — Gym's reset cannot return a
        # reward, so the bootstrap charge lands here (spec 05 §2). The episode return is
        # then exactly  terminal_utility - all search costs, as spec 06 §1 requires.
        # ``breakdown.total`` is already the full episode return
        # (terminal_utility - every search cost); this is only the terminal *step* reward.
        terminal_reward = self._breakdown.terminal_utility - self._bootstrap_cost()
        self._terminal_reason = name
        info = self._safe_info()
        info["terminal_action"] = name
        return self._observation(), terminal_reward, True, False, info

    def _make_breakdown(self, action: str) -> RewardBreakdown:
        assert self._world is not None
        return self.reward.terminal_utility(
            action=action, candidate_text=None, provenance_ok=False,
            private=self._world.private, n_bm25=self._n_bm25, n_dense=self._n_dense)

    def _bootstrap_cost(self) -> float:
        if self.config.bootstrap_tool == "dense":
            return self.reward.config.dense_cost
        return self.reward.config.bootstrap_bm25_cost

    def _next_unseen(self, tool: str, n: int) -> list[str]:
        assert self._world is not None
        ranking = self._world.rankings.get(tool, ())
        seen = set(self._seen)
        out = []
        for doc_id in ranking:
            if doc_id not in seen and doc_id in self._world.corpus:
                out.append(doc_id)
                seen.add(doc_id)
                if len(out) >= n:
                    break
        return out

    def _has_unseen(self, tool: str) -> bool:
        assert self._world is not None
        seen = set(self._seen)
        return any(d not in seen for d in self._world.rankings.get(tool, ()))

    def _refresh_candidate(self) -> None:
        assert self._world is not None
        self._candidate = self._world.reader(tuple(self._seen))

    def _action_mask(self) -> np.ndarray:
        assert self._world is not None
        searchable = self._optional_used < self.config.max_optional_search
        candidate_ok = (
            self._candidate.answer_text is not None
            and self._candidate.provenance_ok
            and self._candidate.doc_id in self._seen
        )
        mask = np.zeros(N_ACTIONS, dtype=bool)
        mask[ANSWER] = candidate_ok
        mask[SEARCH_BM25] = searchable and self._has_unseen("bm25")
        mask[SEARCH_DENSE] = searchable and self._has_unseen("dense")
        mask[ABSTAIN] = True  # always legal
        return mask

    def _valid_action_names(self) -> list[str]:
        return [ACTIONS[i] for i, ok in enumerate(self._action_mask()) if ok]

    def _retrieval_summary(self) -> dict[str, float]:
        assert self._world is not None

        def top(tool: str) -> tuple[float, float]:
            ranking = list(self._world.rankings.get(tool, ()))
            scores = getattr(self._world, f"_scores_{tool}", None)
            if not ranking:
                return 0.0, 0.0
            if scores:
                vals = [scores.get(d, 0.0) for d in ranking]
                gap = (vals[0] - vals[1]) if len(vals) > 1 else 0.0
                return float(vals[0]), float(gap)
            # Rank-only worlds (unit tests, toy fixtures): expose rank, not a fake score.
            seen = set(self._seen)
            next_unseen = next((i for i, d in enumerate(ranking) if d not in seen), None)
            return (float(-next_unseen) if next_unseen is not None else 0.0), 0.0

        if self.config.preview_unseen_scores:
            bm25_top, bm25_gap = top("bm25")
            dense_top, dense_gap = top("dense")
        else:
            bm25_top = bm25_gap = dense_top = dense_gap = 0.0
        return {"bm25_top_score": bm25_top, "bm25_top_gap": bm25_gap,
                "dense_top_score": dense_top, "dense_top_gap": dense_gap}

    def _observation(self) -> dict[str, Any]:
        assert self._world is not None
        cand = self._candidate
        return {
            "qid": self._world.qid,
            "question": self._world.question,
            "seen_doc_ids": tuple(self._seen),
            "seen_doc_texts": tuple(self._world.corpus[d] for d in self._seen),
            "candidate": {
                "text": cand.answer_text,
                "doc_id": cand.doc_id,
                "non_null_score": float(cand.non_null_score),
                "null_score": float(cand.null_score),
                "null_margin": float(cand.null_margin),
                "provenance_ok": bool(cand.provenance_ok),
            },
            "retrieval": self._retrieval_summary(),
            "searches_used": self._n_bm25 + self._n_dense,
            "optional_used": self._optional_used,
            "remaining_optional_searches": self.config.max_optional_search - self._optional_used,
            "last_action": int(self._history[-1]) if self._history else -1,
            "history": tuple(int(a) for a in self._history),
            "timestep": self._t,
            "action_mask": self._action_mask(),
        }

    def _safe_info(self) -> dict[str, Any]:
        """Training-visible metadata. Contains no labels, no rewards, no oracle actions."""
        assert self._world is not None
        return {
            "qid": self._world.qid,
            "split": self._world.split,
            "scenario_id": self._world.scenario_id,
            "timestep": self._t,
            "state_hash": self.state_hash(),
        }

    # --------------------------------------------------------------- public
    def state_hash(self) -> str:
        assert self._world is not None
        return sha256_json({
            "qid": self._world.qid,
            "scenario_id": self._world.scenario_id,
            "seen_doc_ids": self._seen,
            "optional_used": self._optional_used,
            "history": self._history,
            "state_version": self._world.scenario_version,
        })

    def private_outcome(self) -> dict[str, Any] | None:
        """Reward breakdown for diagnostics. **Never** wire this into policy features."""
        if self._breakdown is None:
            return None
        out = self._breakdown.to_dict()
        out["terminal_reason"] = self._terminal_reason
        out["gold_reachable"] = self._world.private.gold_reachable if self._world else None
        return out

    def trace_row(self) -> dict[str, Any]:
        """Canonical per-step trace row (spec 05 §8) — safe fields only."""
        assert self._world is not None
        return {
            "qid": self._world.qid,
            "scenario_id": self._world.scenario_id,
            "corpus_sha": self._world.corpus_sha,
            "timestep": self._t,
            "state_hash": self.state_hash(),
            "visible_doc_ids": list(self._seen),
            "candidate": self._candidate.to_dict(),
            "action_mask": self._action_mask().astype(int).tolist(),
            "last_action": int(self._history[-1]) if self._history else None,
        }

    @property
    def candidate(self) -> ReaderPrediction:
        return self._candidate

    @property
    def seen_doc_ids(self) -> tuple[str, ...]:
        return tuple(self._seen)

    @property
    def n_bm25(self) -> int:
        return self._n_bm25

    @property
    def n_dense(self) -> int:
        return self._n_dense

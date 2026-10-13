"""V2 gold teacher: exact Bellman over the at-most-two-SEARCH evidence tree.

The teacher (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §3) replays
:func:`~src.rl.vixstory_env_v2.transition` over the whole reachable tree of one world and
solves the retrieval MDP exactly::

    Q(s, STOP)        = reward_STOP(s) = 0.5 * coverage + 1.0 * I(G ⊆ S)
    Q(s, SEARCH_tool) = -0.05 + V(next_tool(s))       (illegal actions: -inf)
    V(s)              = max over legal actions        (pure Bellman max)

The memo is keyed by ``(ordered seen doc ids, remaining optional searches)`` so two
histories that collide on the same visible documents stay distinct states and the remaining
budget is part of the key. Ties resolve toward ``STOP``, then ``SEARCH_BM25``, then
``SEARCH_DENSE`` because values/probabilities are always aligned with
:data:`~src.decision.schema_v2.ACTION_ORDER`. Gold context IDs are read here and never
enter a public observation; :func:`retrieval_metrics` provides reference gold-only metrics
cross-checked against the evaluator's independent arithmetic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from src.decision.schema_v2 import N_ACTIONS
from src.rl.vixstory_env_v2 import (
    BOOTSTRAP_PASSAGES,
    COVERAGE_WEIGHT,
    FULL_SUCCESS_BONUS,
    MAX_OPTIONAL_SEARCHES,
    PASSAGES_PER_SEARCH,
    SEARCH_COST,
    STOP,
    VixStoryWorldV2,
    initial_state,
    stop_reward,
    transition,
    validate_obs,
)


class VixStoryTeacherV2:
    """Exact gold teacher for one world, at a fixed softmax temperature."""

    def __init__(self, world: VixStoryWorldV2, temperature: float = 0.5) -> None:
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(float(temperature)) or temperature <= 0.0):
            raise ValueError(f"temperature must be a positive finite number; "
                             f"got {temperature!r}")
        self.world = world
        self.temperature = float(temperature)
        #: (ordered seen ids, remaining optional searches) -> V(state).
        self._memo: dict[tuple[tuple[str, ...], int], float] = {}
        self._states: list[dict[str, Any]] | None = None

    def enumerate_states(self) -> list[dict[str, Any]]:
        """Every visible state of the search tree, root first (a defensive copy).

        States are identified by ``(ordered seen ids, history)``: two action orders that
        reveal the same documents in the same order remain two states because their
        rendered state text differs.
        """
        if self._states is None:
            root = initial_state(self.world)
            states = [root]
            identities = {(root["seen_doc_ids"], tuple(root["history"]))}
            frontier = [root]
            while frontier:
                new_frontier: list[dict[str, Any]] = []
                for obs in frontier:
                    for index in range(1, N_ACTIONS):          # SEARCH actions only
                        if not obs["action_mask"][index]:
                            continue
                        child = transition(self.world, obs, index)
                        identity = (child["seen_doc_ids"], tuple(child["history"]))
                        if identity in identities:
                            continue
                        identities.add(identity)
                        states.append(child)
                        new_frontier.append(child)
                frontier = new_frontier
            self._states = states
        return deepcopy(self._states)

    def q_values(self, obs: Mapping[str, Any]) -> list[float]:
        """Action values aligned with ``ACTION_ORDER``; illegal actions are ``-inf``.

        Callers that serialize these values for JSON must sanitize the infinities first;
        :meth:`probabilities` is the JSON-safe view.
        """
        validate_obs(self.world, obs)
        values: list[float] = []
        for index in range(N_ACTIONS):
            if not obs["action_mask"][index]:
                values.append(float("-inf"))
            elif index == STOP:
                values.append(stop_reward(self.world, obs))
            else:
                values.append(-SEARCH_COST + self._value(transition(self.world, obs, index)))
        key = (tuple(obs["seen_doc_ids"]), obs["remaining_optional_searches"])
        self._memo[key] = max(values)
        return values

    def _value(self, obs: Mapping[str, Any]) -> float:
        key = (tuple(obs["seen_doc_ids"]), obs["remaining_optional_searches"])
        if key not in self._memo:
            best = stop_reward(self.world, obs)
            for index in range(1, N_ACTIONS):
                if obs["action_mask"][index]:
                    best = max(best, -SEARCH_COST + self._value(transition(self.world, obs,
                                                                          index)))
            self._memo[key] = best
        return self._memo[key]

    def probabilities(self, obs: Mapping[str, Any]) -> list[float]:
        """Softmax over legal actions only; illegal actions get exactly 0.0 (JSON-safe)."""
        values = self.q_values(obs)
        finite = [value for value in values if math.isfinite(value)]
        top = max(finite)
        weights = [math.exp((value - top) / self.temperature) if math.isfinite(value) else 0.0
                   for value in values]
        total = math.fsum(weights)
        return [weight / total for weight in weights]

    def oracle_return(self) -> float:
        """``V(root)``: the optimal return within this world's rankings and budget."""
        return self._value(initial_state(self.world))

    def full_gold_reachable(self) -> bool:
        """Whether any reachable state's evidence already covers the full gold set."""
        gold = set(self.world.gold_context_ids)
        return any(gold <= set(obs["seen_doc_ids"]) for obs in self.enumerate_states())


def retrieval_metrics(seen_doc_ids: Sequence[str], gold_context_ids: Sequence[str],
                      n_search: int) -> dict[str, Any]:
    """End-of-episode retrieval metrics on the final evidence set (plan §4).

    ``seen_doc_ids`` is the ordered evidence list of the episode (repeated ids are
    deduplicated in first-occurrence order), ``gold_context_ids`` the unique gold set and
    ``n_search`` the number of optional SEARCH actions. Invalid input raises instead of
    scoring zero.
    """
    seen = _require_ids(seen_doc_ids, "seen_doc_ids", allow_repeats=True)
    gold = _require_ids(gold_context_ids, "gold_context_ids", allow_repeats=False)
    if (isinstance(n_search, bool) or not isinstance(n_search, int)
            or not 0 <= n_search <= MAX_OPTIONAL_SEARCHES):
        raise ValueError(f"n_search must be an integer from 0 to {MAX_OPTIONAL_SEARCHES}; "
                         f"got {n_search!r}")
    unique_seen = list(dict.fromkeys(seen))
    if len(unique_seen) > BOOTSTRAP_PASSAGES + PASSAGES_PER_SEARCH * n_search:
        raise ValueError(f"{len(unique_seen)} seen doc(s) are impossible within {n_search} "
                         "optional search(es)")
    seen_set = set(unique_seen)
    matched = [doc_id for doc_id in gold if doc_id in seen_set]
    missing = [doc_id for doc_id in gold if doc_id not in seen_set]
    coverage = len(matched) / len(gold)
    full_success = 1.0 if not missing else 0.0
    precision = len(matched) / len(unique_seen)
    return {
        "coverage": coverage,
        "full_success": full_success,
        "precision": precision,
        "retrieval_return": (COVERAGE_WEIGHT * coverage + FULL_SUCCESS_BONUS * full_success
                             - SEARCH_COST * n_search),
        "n_search": n_search,
        "n_seen": len(unique_seen),
        "matched_gold_ids": matched,
        "missing_gold_ids": missing,
    }


def _require_ids(values: Any, name: str, *, allow_repeats: bool) -> list[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of document ids")
    ids = list(values)
    if not ids:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
        raise ValueError(f"{name} must contain non-empty string doc ids")
    if not allow_repeats and len(set(ids)) != len(ids):
        raise ValueError(f"{name} must not repeat doc ids")
    return ids

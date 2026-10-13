"""V2 retrieval-only arms over the public observation — E00/E01/E02 and the prior bridge.

This module is the whole arm layer of the evidence-first V2 route
(``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §3, §5). The action order lives
in :mod:`src.decision.schema_v2` (``ACTION_ORDER``) and the constants (``STOP=0``,
``SEARCH_BM25=1``, ``SEARCH_DENSE=2``) in :mod:`src.rl.vixstory_env_v2`; every callable here
accepts exactly the public observation the environment emits and returns one integer action:

* ``stop`` — arm **E00**: end immediately after the free bootstrap.
* ``fixed_tool(tool, k)`` — arms **E01** (``"bm25"``) and **E02** (``"dense"``): SEARCH with
  that tool ``k`` times, then STOP. ``K ∈ {1, 2}`` is chosen on validation with the shared
  selection rule (:func:`src.pipeline.vixstory_eval_v2.select_best`); any other K is a
  protocol error — the pinned budget allows at most two optional SEARCH actions.
* ``masked_greedy(prior)`` — the single thin bridge for every learned arm. **E04/E05** plug in
  the Laya prior; **E06/E07** (REINFORCE with a frozen prior) integrate through the same
  shape, the parent wrapping its network as ``prior(obs) -> 3 probabilities``. The callable
  takes the masked argmax, and ``numpy.argmax``'s lowest-index tie rule reproduces the
  teacher's STOP > BM25 > Dense preference.

There is no fourth action, no fallback action and no threshold: a caller that hands in a
four-element prior or an out-of-protocol K fails loudly. Nothing here reads the gold side —
the observation is the only input, and any other source would be a wiring error the evaluator
exists to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from src.decision.schema_v2 import ACTION_ORDER
from src.rl.vixstory_env_v2 import MAX_OPTIONAL_SEARCHES, SEARCH_BM25, SEARCH_DENSE, STOP

N_ACTIONS = len(ACTION_ORDER)

_TOOL_ACTIONS: dict[str, int] = {"bm25": SEARCH_BM25, "dense": SEARCH_DENSE}


def stop(obs: Mapping[str, Any]) -> int:
    """Arm E00: STOP immediately — the observation is accepted but nothing is searched."""
    return STOP


def fixed_tool(tool: str, k: int) -> Callable[[Mapping[str, Any]], int]:
    """Arm E01/E02: SEARCH with ``tool`` up to ``k`` times, then STOP.

    Stateless: the SEARCH count is read from the public ``optional_used`` key, so the same
    callable behaves identically on every reset. A mask that forbids the tool (budget spent
    or ranking exhausted) also ends the arm with STOP.
    """
    if tool not in _TOOL_ACTIONS:
        raise ValueError(f"unknown tool {tool!r}; the V2 schema has {sorted(_TOOL_ACTIONS)}")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError(f"K must be an int, got {type(k).__name__}")
    if not 1 <= k <= MAX_OPTIONAL_SEARCHES:
        raise ValueError(
            f"K must lie in [1, {MAX_OPTIONAL_SEARCHES}] — the pinned optional-SEARCH budget; "
            f"got {k}")
    action = _TOOL_ACTIONS[tool]

    def _policy(obs: Mapping[str, Any]) -> int:
        mask = np.asarray(obs["action_mask"], dtype=bool)
        if mask.shape != (N_ACTIONS,):
            raise ValueError(f"action_mask must have shape ({N_ACTIONS},); got {mask.shape}")
        if int(obs["optional_used"]) < k and mask[action]:
            return action
        return STOP

    return _policy


def masked_greedy(prior: Callable[[Mapping[str, Any]], Sequence[float]]
                  ) -> Callable[[Mapping[str, Any]], int]:
    """The thin bridge every learned arm shares: ``prior(obs)`` -> masked argmax action.

    ``prior`` returns three probabilities (E04/E05: the Laya prior; E06/E07: the parent's
    frozen-prior REINFORCE policy wrapped to the same shape). The vector must be finite,
    non-negative and sum to a positive number — a four-element (schema-v1) vector is rejected
    rather than silently truncated — and the action is the argmax over legal actions only.
    """
    if not callable(prior):
        raise TypeError("prior must be callable obs -> three probabilities")

    def _policy(obs: Mapping[str, Any]) -> int:
        probs = np.asarray(prior(obs), dtype=np.float64)
        mask = np.asarray(obs["action_mask"], dtype=bool)
        if probs.shape != (N_ACTIONS,):
            raise ValueError(
                f"prior must return {N_ACTIONS} probabilities; got shape {probs.shape}")
        if mask.shape != (N_ACTIONS,):
            raise ValueError(f"action_mask must have shape ({N_ACTIONS},); got {mask.shape}")
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"prior probabilities must be finite; got {probs.tolist()}")
        if np.any(probs < 0) or probs.sum() <= 0:
            raise ValueError(
                f"prior probabilities must be non-negative with a positive sum; got "
                f"{probs.tolist()}")
        if not mask.any():
            raise ValueError("action_mask has no legal action; STOP is always legal in schema_v2")
        return int(np.argmax(np.where(mask, probs, -np.inf)))

    return _policy

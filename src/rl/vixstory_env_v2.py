"""VixSTORY V2 evidence-retrieval environment — three actions, no reader.

V2 (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §3) is retrieval-only:
an episode opens with one free BM25 bootstrap passage and then allows at most two optional
SEARCH actions, each revealing up to two passages of its tool that the agent has not seen.
``STOP`` ends the episode and keeps the ordered evidence list. There is no candidate
answer, no reader model and no network call anywhere in this module.

The state machine is :func:`initial_state` / :func:`transition` — two pure functions the
teacher replays to enumerate the whole at-most-two-SEARCH tree. :class:`VixStoryEnvV2` is
only the episode wrapper: world selection, rewards and private bookkeeping (the gold
context IDs leave the environment exclusively through
:meth:`VixStoryEnvV2.private_outcome`).
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.decision.schema_v2 import ACTION_ORDER, N_ACTIONS, assert_public_state_is_clean
from src.retrieval.states import next_unseen

STOP, SEARCH_BM25, SEARCH_DENSE = 0, 1, 2
TOOLS = ("bm25", "dense")
_TOOL_OF_ACTION = {SEARCH_BM25: TOOLS[0], SEARCH_DENSE: TOOLS[1]}
BOOTSTRAP_TOOL = TOOLS[0]

BOOTSTRAP_PASSAGES = 1          # free BM25 passage at reset
PASSAGES_PER_SEARCH = 2         # at most this many unseen passages per SEARCH
MAX_OPTIONAL_SEARCHES = 2
MAX_SEEN_DOCS = BOOTSTRAP_PASSAGES + PASSAGES_PER_SEARCH * MAX_OPTIONAL_SEARCHES
MAX_ACTIONS = MAX_OPTIONAL_SEARCHES + 1

SEARCH_COST = 0.05
COVERAGE_WEIGHT = 0.5
FULL_SUCCESS_BONUS = 1.0

PUBLIC_OBS_KEYS = ("qid", "question", "seen_doc_ids", "seen_doc_texts", "history",
                   "optional_used", "remaining_optional_searches", "action_mask")


@dataclass(frozen=True)
class VixStoryWorldV2:
    """One evidence-retrieval question.

    Public sides: ``qid``, ``question``, ``corpus`` (doc id -> passage text) and
    ``rankings`` (tool -> ordered doc ids). ``gold_context_ids`` is private data kept on
    this world; only the environment's ``private_outcome`` ever reads it.
    """

    qid: str
    question: str
    corpus: Mapping[str, str]
    rankings: Mapping[str, Sequence[str]]
    gold_context_ids: Sequence[str]

    def __post_init__(self) -> None:
        if not isinstance(self.qid, str) or not self.qid.strip():
            raise ValueError("world requires a non-empty qid")
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("world requires a non-empty question")
        corpus = self._validated_corpus()
        rankings = self._validated_rankings(corpus)
        gold = self._validated_gold(corpus)
        object.__setattr__(self, "corpus", corpus)
        object.__setattr__(self, "rankings", rankings)
        object.__setattr__(self, "gold_context_ids", gold)

    def _validated_corpus(self) -> dict[str, str]:
        corpus = self.corpus
        if not isinstance(corpus, Mapping) or not corpus:
            raise ValueError("corpus must be a non-empty mapping of doc id -> passage text")
        out: dict[str, str] = {}
        for doc_id, text in corpus.items():
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"corpus doc ids must be non-empty strings; got {doc_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"corpus passage {doc_id!r} must be non-empty text")
            out[doc_id] = text
        return out

    def _validated_rankings(self, corpus: Mapping[str, str]) -> dict[str, tuple[str, ...]]:
        rankings = self.rankings
        if not isinstance(rankings, Mapping) or set(rankings) != set(TOOLS):
            raise ValueError(f"rankings must hold exactly the tools {TOOLS}; got {rankings!r}")
        out: dict[str, tuple[str, ...]] = {}
        for tool in TOOLS:
            ranking = rankings[tool]
            if (isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence)
                    or not ranking):
                raise ValueError(f"ranking {tool!r} must be a non-empty sequence of doc ids")
            ids = tuple(ranking)
            if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
                raise ValueError(f"ranking {tool!r} must contain non-empty string doc ids")
            if len(set(ids)) != len(ids):
                raise ValueError(f"ranking {tool!r} must not repeat doc ids")
            missing = [doc_id for doc_id in ids if doc_id not in corpus]
            if missing:
                raise ValueError(f"ranking {tool!r} references doc ids outside the corpus: "
                                 f"{missing}")
            out[tool] = ids
        return out

    def _validated_gold(self, corpus: Mapping[str, str]) -> tuple[str, ...]:
        gold = self.gold_context_ids
        if isinstance(gold, (str, bytes)) or not isinstance(gold, Sequence) or not gold:
            raise ValueError("gold_context_ids must be a non-empty sequence of corpus doc ids")
        ids = tuple(gold)
        if any(not isinstance(doc_id, str) or not doc_id for doc_id in ids):
            raise ValueError("gold_context_ids must contain non-empty string doc ids")
        if len(set(ids)) != len(ids):
            raise ValueError("gold_context_ids must not repeat doc ids")
        missing = [doc_id for doc_id in ids if doc_id not in corpus]
        if missing:
            raise ValueError(f"gold context ids missing from the corpus: {missing}")
        return ids


def _canonical_mask(world: VixStoryWorldV2, seen: Sequence[str],
                    optional_used: int) -> list[bool]:
    """STOP is always legal; a SEARCH is legal only with budget *and* unseen passages."""
    mask = [True]
    for tool in TOOLS:
        legal = (optional_used < MAX_OPTIONAL_SEARCHES
                 and bool(next_unseen(world.rankings[tool], seen, 1)))
        mask.append(legal)
    return mask


def _public_obs(world: VixStoryWorldV2, seen: Sequence[str], history: Sequence[str],
                optional_used: int) -> dict[str, Any]:
    """Build (and scan) the public observation for the current evidence state."""
    seen = tuple(seen)
    obs: dict[str, Any] = {
        "qid": world.qid,
        "question": world.question,
        "seen_doc_ids": seen,
        "seen_doc_texts": tuple(world.corpus[doc_id] for doc_id in seen),
        "history": list(history),
        "optional_used": optional_used,
        "remaining_optional_searches": MAX_OPTIONAL_SEARCHES - optional_used,
        "action_mask": _canonical_mask(world, seen, optional_used),
    }
    assert_public_state_is_clean(obs)
    return obs


def initial_state(world: VixStoryWorldV2) -> dict[str, Any]:
    """The reset observation: the free BM25 bootstrap passage, empty history."""
    seen = next_unseen(world.rankings[BOOTSTRAP_TOOL], (), BOOTSTRAP_PASSAGES)
    return _public_obs(world, seen, [], 0)


def _action_index(action: Any) -> int:
    if isinstance(action, bool) or not isinstance(action, int):
        raise ValueError(f"action must be an int in [0, {N_ACTIONS}); got {action!r}")
    if not 0 <= action < N_ACTIONS:
        raise ValueError(f"action {action} outside [0, {N_ACTIONS})")
    return action


def stop_reward(world: VixStoryWorldV2, obs: Mapping[str, Any]) -> float:
    """``reward_STOP = 0.5 * coverage + 1.0 * I(G ⊆ S)`` with G = gold, S = seen (plan §3)."""
    gold = set(world.gold_context_ids)
    seen = set(obs["seen_doc_ids"])
    coverage = len(gold & seen) / len(gold)
    full_success = 1.0 if gold <= seen else 0.0
    return COVERAGE_WEIGHT * coverage + FULL_SUCCESS_BONUS * full_success


def transition(world: VixStoryWorldV2, obs: Mapping[str, Any], action: int) -> dict[str, Any]:
    """Pure transition: apply one legal action to a validated public observation."""
    validate_obs(world, obs)
    if ACTION_ORDER[STOP] in obs["history"]:
        raise ValueError("cannot transition an already terminated observation")
    index = _action_index(action)
    name = ACTION_ORDER[index]
    seen = list(obs["seen_doc_ids"])
    history = list(obs["history"])
    if index == STOP:
        history.append(name)
        return _public_obs(world, seen, history, obs["optional_used"])
    if not obs["action_mask"][index]:
        raise ValueError(f"action {name} is illegal in this state")
    history.append(name)
    seen.extend(next_unseen(world.rankings[_TOOL_OF_ACTION[index]], seen, PASSAGES_PER_SEARCH))
    return _public_obs(world, seen, history, obs["optional_used"] + 1)


def validate_obs(world: VixStoryWorldV2, obs: Mapping[str, Any]) -> None:
    """Validate public fields, budget and terminal history; env construction ensures reachability."""
    if not isinstance(obs, Mapping):
        raise ValueError(f"observation must be a mapping; got {type(obs).__name__}")
    assert_public_state_is_clean(obs)
    missing = sorted(set(PUBLIC_OBS_KEYS) - set(obs))
    extra = sorted(set(obs) - set(PUBLIC_OBS_KEYS))
    if missing or extra:
        raise ValueError(f"observation keys must be exactly {PUBLIC_OBS_KEYS}; "
                         f"missing={missing} extra={extra}")
    if obs["qid"] != world.qid:
        raise ValueError(f"observation qid {obs['qid']!r} does not belong to world "
                         f"{world.qid!r}")
    if obs["question"] != world.question:
        raise ValueError(f"observation question does not match world {world.qid!r}")
    seen = obs["seen_doc_ids"]
    texts = obs["seen_doc_texts"]
    if (not isinstance(seen, (list, tuple)) or not isinstance(texts, (list, tuple))
            or len(seen) != len(texts)):
        raise ValueError("seen_doc_ids and seen_doc_texts must be parallel sequences")
    if len(set(seen)) != len(seen) or not set(seen) <= set(world.corpus):
        raise ValueError(f"seen_doc_ids must be unique corpus ids; got {list(seen)!r}")
    for doc_id, text in zip(seen, texts):
        if text != world.corpus[doc_id]:
            raise ValueError("seen_doc_texts must be the corpus texts of seen_doc_ids "
                             f"(mismatch at {doc_id!r})")
    history = obs["history"]
    if (not isinstance(history, (list, tuple))
            or any(name not in ACTION_ORDER for name in history)):
        raise ValueError(f"history must be a sequence of action names from {ACTION_ORDER}; "
                         f"got {history!r}")
    if ACTION_ORDER[STOP] in history[:-1]:
        raise ValueError("STOP may appear only once, at the end of history")
    optional_used = obs["optional_used"]
    if (isinstance(optional_used, bool) or not isinstance(optional_used, int)
            or optional_used != sum(1 for name in history if name != ACTION_ORDER[STOP])
            or not 0 <= optional_used <= MAX_OPTIONAL_SEARCHES):
        raise ValueError(f"optional_used {optional_used!r} is inconsistent with history "
                         f"{history!r}")
    if not BOOTSTRAP_PASSAGES <= len(seen) <= BOOTSTRAP_PASSAGES + PASSAGES_PER_SEARCH * optional_used:
        raise ValueError("seen_doc_ids exceed the evidence budget or omit bootstrap evidence")
    if obs["remaining_optional_searches"] != MAX_OPTIONAL_SEARCHES - optional_used:
        raise ValueError("remaining_optional_searches must equal MAX_OPTIONAL_SEARCHES "
                         "minus optional_used")
    mask = obs["action_mask"]
    if not isinstance(mask, (list, tuple)) or len(mask) != N_ACTIONS:
        raise ValueError(f"action_mask must be a sequence of {N_ACTIONS} booleans")
    expected = _canonical_mask(world, tuple(seen), optional_used)
    if list(mask) != expected:
        raise ValueError(f"action_mask {list(mask)!r} does not match the legal mask "
                         f"{expected!r}")


class VixStoryEnvV2:
    """Three-action evidence-retrieval episode over one selected world."""

    def __init__(self, worlds: Sequence[VixStoryWorldV2]) -> None:
        if not worlds:
            raise ValueError("VixStoryEnvV2 requires at least one world")
        self._worlds = {world.qid: world for world in worlds}
        if len(self._worlds) != len(worlds):
            raise ValueError("duplicate qid in VixStoryEnvV2 worlds")
        self._rng = random.Random()
        self._world: VixStoryWorldV2 | None = None
        self._seen: list[str] = []
        self._history: list[str] = []
        self._optional_used = 0
        self._terminated = False

    def reset(self, seed: int | None = None, options: Mapping[str, Any] | None = None,
              qid: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Start an episode; select the world by ``qid``, ``options["qid"]`` or the seed."""
        if options is not None and not isinstance(options, Mapping):
            raise ValueError(f"options must be a mapping; got {type(options).__name__}")
        unknown = set(options or {}) - {"qid"}
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        option_qid = None if options is None else options.get("qid")
        if qid is not None and option_qid is not None and qid != option_qid:
            raise ValueError(f"conflicting qid arguments: {qid!r} != {option_qid!r}")
        select = qid if qid is not None else option_qid
        if select is None:
            rng = random.Random(seed) if seed is not None else self._rng
            select = rng.choice(sorted(self._worlds))
        world = self._worlds[select]
        self._world = world
        self._seen = next_unseen(world.rankings[BOOTSTRAP_TOOL], (), BOOTSTRAP_PASSAGES)
        self._history = []
        self._optional_used = 0
        self._terminated = False
        return _public_obs(world, self._seen, self._history, 0), {"qid": world.qid,
                                                                  "timestep": 0}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action; only ``STOP`` terminates and it never calls a reader."""
        if self._world is None or self._terminated:
            raise RuntimeError("step() requires an active episode; call reset() first")
        world = self._world
        index = _action_index(action)
        current = _public_obs(world, self._seen, self._history, self._optional_used)
        obs = transition(world, current, index)
        self._seen = list(obs["seen_doc_ids"])
        self._history = list(obs["history"])
        self._optional_used = obs["optional_used"]
        terminated = index == STOP
        self._terminated = terminated
        reward = stop_reward(world, obs) if terminated else -SEARCH_COST
        info = {"qid": world.qid, "timestep": len(self._history)}
        return obs, reward, terminated, False, info

    def private_outcome(self) -> dict[str, Any] | None:
        """Gold-side episode bookkeeping, meaningful only after the terminal ``STOP``."""
        if not self._terminated or self._world is None:
            return None
        world = self._world
        return {"qid": world.qid, "terminal_action": self._history[-1],
                "n_search": self._optional_used, "seen_doc_ids": list(self._seen),
                "gold_context_ids": list(world.gold_context_ids)}

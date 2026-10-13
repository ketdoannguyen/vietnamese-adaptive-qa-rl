"""Contract tests for the V2 public-observation arms (``src/pipeline/vixstory_arms_v2.py``).

Written before the module (plan ``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §5).
The arms are E00 ``stop``, E01/E02 ``fixed_tool("bm25"|"dense", K)`` and the single thin
``masked_greedy(prior)`` callable every learned arm (E04-E07) plugs into. Every callable must
act on the public observation alone, honor the environment's mask, and refuse anything outside
the pinned three-action schema — no fourth action, no fallback, no threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.schemas import find_forbidden_keys
from src.decision.schema_v2 import ACTION_ORDER
from src.pipeline.vixstory_arms_v2 import fixed_tool, masked_greedy, stop
from src.rl.vixstory_env_v2 import MAX_OPTIONAL_SEARCHES, SEARCH_BM25, SEARCH_DENSE, STOP


def public_obs(*, action_mask=(True, True, True), optional_used=0, **extra):
    """The public V2 observation shape: qid/question/evidence/history/budget/mask only."""
    obs = {"qid": "train-1", "question": "Trường A ở đâu?",
           "seen_doc_ids": ["d0"], "seen_doc_texts": ["Đoạn văn nguồn."],
           "history": [], "optional_used": optional_used,
           "remaining_optional_searches": 2 - optional_used,
           "action_mask": list(action_mask)}
    obs.update(extra)
    return obs


def test_schema_v2_action_order_is_pinned():
    assert ACTION_ORDER == ("STOP", "SEARCH_BM25", "SEARCH_DENSE")
    assert (STOP, SEARCH_BM25, SEARCH_DENSE) == (0, 1, 2)
    assert MAX_OPTIONAL_SEARCHES == 2


def test_stop_arm_returns_stop_on_any_public_observation():
    for obs in (public_obs(), public_obs(action_mask=(True, False, False), optional_used=2)):
        assert stop(obs) == STOP


def test_fixed_tool_requires_the_pinned_protocol():
    with pytest.raises(ValueError):
        fixed_tool("tfidf", 1)
    with pytest.raises(ValueError, match="K"):
        fixed_tool("bm25", 0)
    with pytest.raises(ValueError, match="K"):
        fixed_tool("bm25", MAX_OPTIONAL_SEARCHES + 1)
    with pytest.raises(ValueError, match="K"):
        fixed_tool("dense", -1)
    with pytest.raises(TypeError):
        fixed_tool("bm25", True)  # bool is not a K
    with pytest.raises(TypeError):
        fixed_tool("bm25", 1.5)
    for tool, k in (("bm25", 1), ("bm25", 2), ("dense", 1), ("dense", 2)):
        assert callable(fixed_tool(tool, k))


def test_fixed_bm25_searches_k_times_then_stops():
    policy = fixed_tool("bm25", 1)
    assert policy(public_obs(optional_used=0)) == SEARCH_BM25
    assert policy(public_obs(optional_used=1)) == STOP
    policy2 = fixed_tool("bm25", 2)
    assert policy2(public_obs(optional_used=0)) == SEARCH_BM25
    assert policy2(public_obs(optional_used=1)) == SEARCH_BM25
    assert policy2(public_obs(optional_used=2)) == STOP


def test_fixed_dense_searches_the_dense_tool():
    policy = fixed_tool("dense", 2)
    assert policy(public_obs(optional_used=0)) == SEARCH_DENSE
    assert policy(public_obs(optional_used=1)) == SEARCH_DENSE
    assert policy(public_obs(optional_used=2)) == STOP


def test_fixed_tool_stops_when_the_mask_forbids_the_tool():
    assert fixed_tool("bm25", 2)(public_obs(action_mask=(True, False, True))) == STOP
    assert fixed_tool("dense", 2)(public_obs(action_mask=(True, True, False))) == STOP
    # numpy mask, as the environment returns it
    mask = np.asarray([True, True, True])
    assert fixed_tool("bm25", 1)(public_obs(action_mask=mask)) == SEARCH_BM25


def test_masked_greedy_takes_the_masked_argmax():
    obs = public_obs()
    assert masked_greedy(lambda o: [0.1, 0.6, 0.3])(obs) == SEARCH_BM25
    assert masked_greedy(lambda o: [0.1, 0.6, 0.3])(
        public_obs(action_mask=(True, False, True))) == SEARCH_DENSE
    assert masked_greedy(lambda o: [0.7, 0.2, 0.1])(obs) == STOP


def test_masked_greedy_ties_follow_the_schema_order():
    # np.argmax picks the lowest action index on ties: STOP, then BM25, then Dense.
    assert masked_greedy(lambda o: [0.4, 0.4, 0.2])(public_obs()) == STOP
    assert masked_greedy(lambda o: [0.1, 0.5, 0.5])(public_obs()) == SEARCH_BM25


@pytest.mark.parametrize("bad", [
    [0.25, 0.25, 0.25, 0.25],  # four actions: the V2 schema has exactly three
    [0.5, 0.5],                # too short
    [0.5, 0.5, float("nan")],
    [0.5, 0.5, float("inf")],
    [-0.1, 0.6, 0.5],
    [0.0, 0.0, 0.0],
])
def test_masked_greedy_rejects_invalid_prior_vectors(bad):
    with pytest.raises(ValueError):
        masked_greedy(lambda o, p=bad: p)(public_obs())


def test_masked_greedy_prior_receives_exactly_the_public_observation():
    seen = {}

    def prior(obs):
        seen.update(obs)
        return [0.2, 0.3, 0.5]

    obs = public_obs()
    assert masked_greedy(prior)(obs) == SEARCH_DENSE
    assert seen == obs
    assert find_forbidden_keys(seen) == []


def test_masked_greedy_rejects_a_non_callable_prior():
    with pytest.raises(TypeError):
        masked_greedy([0.2, 0.3, 0.5])

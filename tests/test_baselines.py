"""Tool-selection and legality contracts for the shared policy helpers (``src/eval/policies``).

CPU-only and self-contained: the helpers read an observation mapping, so the fixtures are
plain in-memory dicts. What is under test is the contract the VixStory env relies on: every
returned action is legal under the mask, tool choice is deterministic and gold-free (ties go
to BM25), and an impossible request raises loudly instead of being guessed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decision.schema import ACTION_ORDER
from src.eval.policies import TOOL_CHOICE_MODES, choose_tool, legal_actions

#: Canonical indices the mask is written in (spec 08 §5).
ANSWER = ACTION_ORDER.index("ANSWER")
SEARCH_BM25 = ACTION_ORDER.index("SEARCH_BM25")
SEARCH_DENSE = ACTION_ORDER.index("SEARCH_DENSE")
ABSTAIN = ACTION_ORDER.index("ABSTAIN")


def _obs(mask, *, history=(), remaining=2):
    return {"action_mask": np.asarray(mask, dtype=bool), "history": tuple(history),
            "remaining_optional_searches": remaining, "qid": "q", "timestep": 0}


def test_legal_actions_lists_mask_indices_and_refuses_an_all_masked_state():
    # the helpers and the VixStory mask share the canonical schema: ANSWER first, ABSTAIN last
    assert (ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN) == (0, 1, 2, 3)
    assert legal_actions([True, False, True, True]) == [ANSWER, SEARCH_DENSE, ABSTAIN]
    assert legal_actions(np.array([False, True, False, False])) == [SEARCH_BM25]
    with pytest.raises(ValueError, match="ABSTAIN must always be legal"):
        legal_actions([False, False, False, False])


def test_tool_choice_modes_are_documented_and_validated():
    assert TOOL_CHOICE_MODES == ("alternate", "top_score")
    with pytest.raises(ValueError, match="unknown tool-choice mode"):
        choose_tool(_obs([True, True, True, True]), mode="random")


def test_tool_choice_helper_is_deterministic_and_documented():
    full = _obs([True, True, True, True])
    assert choose_tool(full) == SEARCH_BM25                    # tie -> BM25
    assert choose_tool({**full, "history": (SEARCH_BM25,)}) == SEARCH_DENSE  # fewer uses
    assert choose_tool({**full, "history": (SEARCH_BM25, SEARCH_DENSE)}) == SEARCH_BM25
    scored = {**full, "retrieval": {"bm25_top_score": 0.1, "dense_top_score": 0.9}}
    assert choose_tool(scored, mode="top_score") == SEARCH_DENSE
    # the preferred tool falls back to the other one when it is masked
    assert choose_tool({**full, "action_mask": np.array([True, False, True, True])}) \
        == SEARCH_DENSE
    with pytest.raises(ValueError, match="no legal search"):
        choose_tool({**full, "action_mask": np.array([True, False, False, True])})

"""Gold-free policy helpers shared with the VixStory environment contract (spec 08 §5).

``legal_actions`` and ``choose_tool`` are the two primitives the VixStory env's downstream
consumers rely on: mask legality in the canonical action order, and a deterministic tool
choice between the two search actions. Action indices come from
:data:`src.decision.schema.ACTION_ORDER` — the single canonical action schema — so the
indices here and in the env's ``action_mask`` are the same by construction.

Rules that hold for both helpers:

* **No gold.** They read the mask, the agent-chosen history and the retrieval preview
  scores only; nothing here can see a label.
* **Legal only.** A returned action is always legal under ``obs["action_mask"]``; an
  impossible request raises instead of being guessed.
* **Deterministic.** No sampling and no RNG: ties always resolve to SEARCH_BM25.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.decision.schema import ACTION_ORDER

#: Indices of the two search actions in the canonical action schema.
SEARCH_BM25 = ACTION_ORDER.index("SEARCH_BM25")
SEARCH_DENSE = ACTION_ORDER.index("SEARCH_DENSE")

#: The tool-choice modes accepted by :func:`choose_tool` (spec 08 §5).
TOOL_CHOICE_MODES = ("alternate", "top_score")


def legal_actions(mask: Sequence[bool]) -> list[int]:
    """Indices of the legal actions in ``ACTION_ORDER`` order. ABSTAIN is always legal."""
    actions = [i for i, ok in enumerate(np.asarray(mask, dtype=bool)) if ok]
    if not actions:
        raise ValueError("every action is masked; ABSTAIN must always be legal")
    return actions


def choose_tool(obs: Mapping[str, Any], *, mode: str = "alternate") -> int:
    """Deterministic, gold-free choice between SEARCH_BM25 and SEARCH_DENSE.

    ``alternate`` (default): the tool used fewer times in this episode's agent-chosen
    history; a tie — including the first optional search — goes to BM25. ``top_score``:
    the higher retrieval preview score, ties to BM25. Both fall back to the other tool when
    their favourite is illegal, and raise when no search is legal at all.
    """
    if mode not in TOOL_CHOICE_MODES:
        raise ValueError(f"unknown tool-choice mode {mode!r}; known: {list(TOOL_CHOICE_MODES)}")
    mask = np.asarray(obs["action_mask"], dtype=bool)
    legal = [action for action in (SEARCH_BM25, SEARCH_DENSE) if mask[action]]
    if not legal:
        raise ValueError("choose_tool called with no legal search action")
    if len(legal) == 1:
        return legal[0]
    if mode == "alternate":
        history = tuple(int(a) for a in obs.get("history", ()))
        prefer = SEARCH_BM25 if history.count(SEARCH_BM25) <= history.count(SEARCH_DENSE) \
            else SEARCH_DENSE
    else:  # top_score
        retrieval = obs.get("retrieval") or {}
        bm25 = float(retrieval.get("bm25_top_score") or 0.0)
        dense = float(retrieval.get("dense_top_score") or 0.0)
        prefer = SEARCH_BM25 if bm25 >= dense else SEARCH_DENSE
    return prefer if prefer in legal else legal[0]

"""Reachable visible-document states of the episodic retrieval tree (`03_...md` §6).

The episode opens with a forced bootstrap retrieval, then allows at most
``max_optional_search`` search actions; each search reveals ``passage_per_search_action``
passages of that tool that the agent has not seen yet. With the shipped config this tree has
at most 7 states — initial, 2 at depth 1, 4 at depth 2 — which is exactly why an exhaustive
reader cache over it is affordable.

A branch that cannot reveal anything new (the tool's ranking is exhausted) is dropped rather
than kept as a no-op state: the environment's own ``_next_unseen`` returns nothing there, so
such a state is unreachable by an actual episode.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

TOOLS = ("bm25", "dense")


def next_unseen(ranking: Sequence[str], seen: Iterable[str], n: int) -> list[str]:
    """First ``n`` documents of ``ranking`` not already in ``seen`` (cursor semantics)."""
    seen_set = set(seen)
    out: list[str] = []
    for doc_id in ranking:
        if doc_id not in seen_set:
            out.append(doc_id)
            seen_set.add(doc_id)
            if len(out) == n:
                break
    return out


def reachable_states(
    rankings: Mapping[str, Sequence[str]],
    *,
    bootstrap_tool: str = "bm25",
    bootstrap_passages: int = 1,
    passage_per_search_action: int = 1,
    max_optional_search: int = 2,
    tools: Sequence[str] = TOOLS,
) -> list[tuple[str, ...]]:
    """Every ordered visible-document tuple reachable within the search budget.

    Order is the order the documents were revealed in, which is what the state key and the
    environment's ``_seen`` list both carry. Two different tool orders therefore produce two
    states even when they end up showing the same set of documents.
    """
    if bootstrap_tool not in rankings:
        raise KeyError(f"no ranking for the bootstrap tool {bootstrap_tool!r}")
    for tool in tools:
        if tool not in rankings:
            raise KeyError(f"no ranking for tool {tool!r}")
    if bootstrap_passages < 0 or passage_per_search_action < 0 or max_optional_search < 0:
        raise ValueError("retrieval budget parameters must be non-negative")

    initial = tuple(next_unseen(rankings[bootstrap_tool], (), bootstrap_passages))
    states = [initial]
    frontier = [initial]
    for _ in range(max_optional_search):
        new_frontier: list[tuple[str, ...]] = []
        for state in frontier:
            for tool in tools:
                revealed = next_unseen(rankings[tool], state, passage_per_search_action)
                if not revealed:
                    continue
                candidate = state + tuple(revealed)
                if candidate not in states:
                    states.append(candidate)
                    new_frontier.append(candidate)
        frontier = new_frontier
    return states


def retrieved_doc_ids(states: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """Union of the documents any reachable state can show, in first-revealed order.

    This is the "retrieved context" of a question: everything the agent could actually
    retrieve within its budget, with no oracle passage mixed in.
    """
    out: list[str] = []
    for state in states:
        for doc_id in state:
            if doc_id not in out:
                out.append(doc_id)
    return tuple(out)

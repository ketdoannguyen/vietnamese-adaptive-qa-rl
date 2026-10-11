"""CPU teacher-row builder for the VixSTORY reader-driven decision task.

VixSTORY questions are free-form and multi-passage, so the ViQuAD oracle
(:mod:`src.decision.teacher`) does not apply: there is no extractive
``non_null_score``/``null_margin`` and no span provenance to render. This module
builds the *same* Laya JSONL row shape as :mod:`src.pipeline.make_teacher` from a
reader-driven reward instead of the privileged environment oracle:

    Q(s, ANSWER)   = answer_reward(F1(state), citation_valid; tau)
    Q(s, ABSTAIN)  = 0.0
    Q(s, SEARCH_t) = -search_cost + V(next_t(s))

where ``next_t(s)`` is the next state after revealing the first unseen document of
tool ``t`` and ``V`` is the max over legal actions. The reward is exactly the
pre-registered rule used by ``scripts/vixstory_reader_eval.py``: token-F1 of the
reader's answer against the reference, citation-gated. Search cost is applied
**once** per search edge (``future - search_cost``), never re-applied going down.

Everything here is CPU-only and deterministic. Inputs are supplied by the caller —
no dataset, train split or GPU work is loaded. A state whose reader answer is absent
from the cache is a hard error, never a fabricated row (the same "missing cache is
an error" rule ``make_teacher`` follows).

Private vs public
-----------------
The public ``state`` text uses :func:`src.decision.vixstory_state_text.serialize_vixstory_state`
for the same VixSTORY observation at teacher time and inference. The gold reference and raw
``Q`` values stay private, in the ``meta``/``gold`` blocks, never in ``state``. No extractive
reader scores are invented.

Trainer status
--------------
This module produces the rows; it does **not** train Laya. The consumer that reads these
rows into ``laya-train`` for the VixSTORY task is not implemented here — the caller must
wire :func:`teacher_rows` output to the trainer.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from src.decision.schema import ACTION_ORDER, laya_record
from src.decision.vixstory_state_text import STATE_TEXT_VERSION, serialize_vixstory_state, state_hash
from src.qa.metrics import f1_score
from src.retrieval.states import next_unseen

#: Action indices, derived from the canonical order so they can never drift from it.
ANSWER = ACTION_ORDER.index("ANSWER")
SEARCH_BM25 = ACTION_ORDER.index("SEARCH_BM25")
SEARCH_DENSE = ACTION_ORDER.index("SEARCH_DENSE")
ABSTAIN = ACTION_ORDER.index("ABSTAIN")

#: Retrieval tools in the order their SEARCH actions appear.
TOOL_ACTIONS = (("bm25", SEARCH_BM25), ("dense", SEARCH_DENSE))

#: Defaults mirror the locked reader-eval configuration.
SEARCH_COST = 0.05
REWARD_TAU = 0.5
TEMPERATURE = 0.5
BOOTSTRAP_PASSAGES = 1
PASSAGE_PER_SEARCH_ACTION = 1
MAX_OPTIONAL_SEARCH = 2

#: Deterministic tie-break order (matches :data:`src.decision.teacher.TIE_ORDER`).
TIE_ORDER = ("ANSWER", "ABSTAIN", "SEARCH_BM25", "SEARCH_DENSE")

#: A cached reader result: ``(answer_text, citation_valid)``. ``None`` text means the reader
#: produced no usable answer at that state.
ReaderAnswer = tuple[str | None, tuple[str, ...], bool]


def enumerate_states(
    rankings: Mapping[str, Sequence[str]],
    *,
    bootstrap_tool: str = "bm25",
    bootstrap_passages: int = BOOTSTRAP_PASSAGES,
    passage_per_search_action: int = PASSAGE_PER_SEARCH_ACTION,
    max_optional_search: int = MAX_OPTIONAL_SEARCH,
) -> list[tuple[tuple[str, ...], tuple[int, ...]]]:
    """Every reachable state as ``(seen_doc_tuple, action_history)``.

    Identical traversal and dedup to :func:`src.retrieval.states.reachable_states` (same
    bootstrap, same tool order, first-revealed wins), but each state also carries the action
    history that reaches it — the state text renderer needs the action sequence. The doc
    tuples are exactly ``reachable_states(...)``.
    """
    initial = tuple(next_unseen(rankings[bootstrap_tool], (), bootstrap_passages))
    out: list[tuple[tuple[str, ...], tuple[int, ...]]] = [(initial, ())]
    seen = {initial}
    frontier = [(initial, ())]
    for _ in range(max_optional_search):
        new_frontier: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
        for state, history in frontier:
            for tool, action in TOOL_ACTIONS:
                revealed = next_unseen(rankings[tool], state, passage_per_search_action)
                if not revealed:
                    continue
                candidate = state + tuple(revealed)
                if candidate in seen:
                    continue
                seen.add(candidate)
                entry = (candidate, history + (action,))
                out.append(entry)
                new_frontier.append(entry)
        frontier = new_frontier
    return out


def answer_reward(f1: float, citation_valid: bool, *, gate_citation: bool = True,
                  tau: float = REWARD_TAU) -> float:
    """Pre-registered ANSWER reward: token-F1 at/above ``tau``, citation-gated by default."""
    if gate_citation and not citation_valid:
        return 0.0
    return 1.0 if f1 >= tau - 1e-12 else 0.0


def q_values(
    state: tuple[str, ...],
    rankings: Mapping[str, Sequence[str]],
    terminal: Callable[[tuple[str, ...]], tuple[float, bool]],
    *,
    remaining: int,
    search_cost: float,
    passage_per_search_action: int = PASSAGE_PER_SEARCH_ACTION,
) -> tuple[float, dict[str, float]]:
    """Recursive state-action values over the reachable tree (CPU, pure).

    ``terminal(s) -> (answer_reward, answer_present)`` supplies the cached, citation-gated
    reward for the state's reader answer. Only *legal* actions enter the returned ``q`` dict:
    ``ANSWER`` iff an answer is present, ``ABSTAIN`` always, ``SEARCH_t`` iff ``remaining``
    permits and tool ``t`` can still reveal an unseen document. Search cost is subtracted once
    per branch edge, so a two-step path pays it twice total and never more.
    """
    answer_reward_value, answer_present = terminal(state)
    q: dict[str, float] = {}
    if answer_present:
        q["ANSWER"] = answer_reward_value
    q["ABSTAIN"] = 0.0
    if remaining:
        for tool, _action in TOOL_ACTIONS:
            addition = next_unseen(rankings[tool], state, passage_per_search_action)
            if not addition:
                continue
            future, _ = q_values(state + tuple(addition), rankings, terminal,
                                 remaining=remaining - 1, search_cost=search_cost,
                                 passage_per_search_action=passage_per_search_action)
            q[f"SEARCH_{tool.upper()}"] = future - search_cost
    if not q:
        raise AssertionError(f"no legal action at state {state!r}: ABSTAIN must be legal")
    return max(q.values()), q


def soft_distribution(q: Mapping[str, float], *,
                      temperature: float = TEMPERATURE) -> dict[str, float]:
    """Softmax over legal actions only; every masked action gets exactly zero mass."""
    best = max(q.values())
    temperature = max(temperature, 1e-6)
    exps = {name: math.exp((value - best) / temperature) for name, value in q.items()}
    total = sum(exps.values())
    return {name: (exps[name] / total if name in exps else 0.0) for name in ACTION_ORDER}


def expected_action(q: Mapping[str, float]) -> str:
    """Deterministic argmax, tie-broken by :data:`TIE_ORDER`."""
    top = max(q.values())
    for name in TIE_ORDER:
        if name in q and abs(q[name] - top) <= 1e-8:
            return name
    raise AssertionError("no action matched the maximum")


def _retrieval_summary() -> dict[str, float]:
    """Rankings here carry document ids only, so no score is invented (env default off)."""
    return {"bm25_top_score": 0.0, "bm25_top_gap": 0.0,
            "dense_top_score": 0.0, "dense_top_gap": 0.0}


def _observation(*, qid: str, question: str, state: tuple[str, ...],
                 docs: Mapping[str, str], answer: ReaderAnswer, history: tuple[int, ...],
                 remaining: int, mask: Sequence[bool]) -> dict[str, Any]:
    """Public VixSTORY observation — no gold answer, no gold ids, no fabricated scores."""
    text, cited_ids, citation_valid = answer
    return {
        "qid": qid,
        "question": question,
        "seen_doc_ids": tuple(state),
        "seen_doc_texts": tuple(docs[doc_id] for doc_id in state),
        "candidate": {"text": text, "cited_ids": tuple(cited_ids),
                      "citation_valid": bool(citation_valid),
                      "answer_present": text is not None and bool(text.strip())},
        "retrieval": _retrieval_summary(),
        "searches_used": len(history),
        "optional_used": len(history),
        "remaining_optional_searches": remaining,
        "last_action": int(history[-1]) if history else -1,
        "history": tuple(int(action) for action in history),
        "timestep": len(history),
        "action_mask": list(mask),
    }


def teacher_rows(
    qids: Sequence[str],
    *,
    questions: Mapping[str, str],
    references: Mapping[str, str],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    docs: Mapping[str, str],
    reader_answers: Mapping[tuple[str, tuple[str, ...]], ReaderAnswer],
    search_cost: float = SEARCH_COST,
    tau: float = REWARD_TAU,
    temperature: float = TEMPERATURE,
    gate_citation: bool = True,
    bootstrap_passages: int = BOOTSTRAP_PASSAGES,
    passage_per_search_action: int = PASSAGE_PER_SEARCH_ACTION,
    max_optional_search: int = MAX_OPTIONAL_SEARCH,
    tokenizer: Any,
    max_len: int = 768,
    head_max_len: int = 192,
) -> list[dict[str, Any]]:
    """Laya JSONL rows for exactly ``qids`` (the caller's train300 list).

    One row per reachable state. ``state`` is public; ``gold``/``meta.q_values`` are private
    teacher targets. Reader answers are keyed by (qid, ordered seen documents); a missing
    answer raises instead of fabricating an output or borrowing another question's result.
    """
    rows: list[dict[str, Any]] = []
    for qid in qids:
        ranking = {"bm25": rankings[qid]["bm25"], "dense": rankings[qid]["dense"]}
        reference = references[qid]

        def terminal(state: tuple[str, ...], _qid: str = qid) -> tuple[float, bool]:
            try:
                text, _cited_ids, citation_valid = reader_answers[_qid, state]
            except KeyError:
                raise KeyError(
                    f"{_qid}: no cached reader answer for state {state!r}; refusing to "
                    "fabricate a reward. Warm the reader cache for every reachable state.") from None
            if text is None or not text.strip():
                return 0.0, False
            reward = answer_reward(f1_score(text, [reference]), citation_valid,
                                   gate_citation=gate_citation, tau=tau)
            return reward, True

        for state, history in enumerate_states(
                ranking, bootstrap_passages=bootstrap_passages,
                passage_per_search_action=passage_per_search_action,
                max_optional_search=max_optional_search):
            remaining = max_optional_search - len(history)
            _value, q = q_values(state, ranking, terminal, remaining=remaining,
                                 search_cost=search_cost,
                                 passage_per_search_action=passage_per_search_action)
            mask = [name in q for name in ACTION_ORDER]
            obs = _observation(qid=qid, question=questions[qid], state=state, docs=docs,
                               answer=reader_answers[qid, state], history=history,
                               remaining=remaining, mask=mask)
            text = serialize_vixstory_state(obs, tokenizer=tokenizer, max_len=max_len,
                                             head_max_len=head_max_len)
            soft = soft_distribution(q, temperature=temperature)
            expected = expected_action(q)
            record = laya_record(state=text, expected=expected, gold_distribution=soft)
            record["meta"] = {
                "qid": qid,
                "state_hash": state_hash(text),
                "state_text_version": STATE_TEXT_VERSION,
                "history": list(history),
                "depth": len(history),
                "valid_actions": [name for name in ACTION_ORDER if name in q],
                "expected": expected,
                "q_values": {name: round(value, 6) for name, value in q.items()},
                "teacher_mode": "soft",
            }
            rows.append(record)
    return rows

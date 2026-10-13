"""V2 retrieval-only evaluator — exact gold-ID metrics, no Reader, no API.

Scope (``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §4): one policy runs one
episode per world; the final document set at STOP is scored against the world's gold context
IDs with exact set operations. Nothing here generates answers, imports a reader or an API
client, or reads a test split — the evaluator only drives
:class:`src.rl.vixstory_env_v2.VixStoryEnvV2` with the *public* observation and scores the
environment's own terminal state.

Guarantees this module exists to enforce:

* **Public observation only.** A policy receives exactly the observation ``reset``/``step``
  return. The gold side is read only by the evaluator, after termination, and every
  observation is checked for the required public keys and for label-bearing fields before it
  reaches a policy.
* **Exact sets, ordered evidence.** Gold recall/coverage, full-gold success and gold precision
  are exact ID-set operations on the final visible documents; the per-QID record keeps the
  ordered ``seen_doc_ids`` exactly as the environment revealed them (the reveal order is
  evidence; a set would erase it). ``matched_gold_ids``/``missing_gold_ids`` follow the
  world's gold order.
* **Fail closed.** Duplicate or empty QIDs, a non-callable policy, an out-of-schema or
  masked-out action, an episode that will not terminate within its own budget, a truncated
  episode, or a ``private_outcome()`` or reward stream that disagrees with the evaluator's own
  exact-set arithmetic are hard errors — never a silent zero.
* **Honest latency.** ``policy_latency_ms`` (policy calls only) and ``env_replay_latency_ms``
  (``reset`` + ``step``) are timed separately and reported as the environment's *replay* of
  the precomputed rankings: they are **not** live BM25/Dense index search latency, because the
  rankings were built offline and the environment only walks them with a cursor.

Also here: the shared checkpoint/K selection rule (:func:`select_best`) and the paired per-QID
difference with its deterministic QID-bootstrap CI (:func:`paired_difference`), both labeled
as development evaluation on the validation split (single seed 42; the CI reflects
question-level variation only, never stability across seeds).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from src.data.schemas import find_forbidden_keys
from src.decision.schema_v2 import ACTION_ORDER
from src.rl.vixstory_env_v2 import PUBLIC_OBS_KEYS, SEARCH_COST, STOP, VixStoryEnvV2
from src.utils.hashing import sha256_json

#: Version tag of the per-arm report dict.
SCHEMA = "vixstory-eval-v2"

N_ACTIONS = len(ACTION_ORDER)

LATENCY_NOTE = (
    "policy_latency_ms times only the policy calls; env_replay_latency_ms times the "
    "environment's retrieval replay (reset bootstrap + SEARCH steps) over the precomputed "
    "rankings. This is not live BM25/Dense index search latency: the rankings were built "
    "offline and the environment only walks them with a cursor."
)

#: Selection ties are judged within this tolerance on the mean retrieval return, because a
#: caller may aggregate the same per-QID returns in a different summation order and make the
#: last ulp differ. The n_search and index tie-breaks are compared exactly: a mean SEARCH
#: count is a sum of small integers divided once, so exact equality is meaningful there.
SELECTION_TOLERANCE = 1e-9

#: Every paired comparison produced here is a development evaluation: the validation split is
#: also used to choose configurations and checkpoints, so it is not an independent test, and
#: only seed 42 is run, so a QID bootstrap reflects question variation, not seed stability.
DEVELOPMENT_LABEL = "validation-development"
DEVELOPMENT_LABEL_NOTE = (
    "Validation/development evaluation: the validation split is also used to select "
    "configurations and checkpoints, so this is not an independent test result. Single seed "
    "(42): the QID-bootstrap interval reflects question-level variation only, not stability "
    "across seeds."
)

class EnvContractError(RuntimeError):
    """The V2 environment contract (public observation, budget, private outcome) was violated."""


def retrieval_return(gold_recall: float, full_gold_success: bool, n_search: int) -> float:
    """The pinned combined retrieval objective: ``0.5*coverage + I(G⊆S) - 0.05*n_search``.

    ``SEARCH_COST`` is the environment module's own constant, not a redefinition: the
    evaluator's arithmetic must charge exactly what the environment's rewards charge, and
    every episode's reward stream is cross-checked against it.
    """
    return 0.5 * float(gold_recall) + (1.0 if full_gold_success else 0.0) - SEARCH_COST * int(n_search)


# ------------------------------------------------------------------------------ episodes

def _check_public_observation(obs: Mapping[str, Any], qid: str) -> None:
    missing = [key for key in PUBLIC_OBS_KEYS if key not in obs]
    extra = [key for key in obs if key not in PUBLIC_OBS_KEYS]
    if missing or extra:
        raise EnvContractError(
            f"V2 observation for {qid!r} must carry exactly the public keys "
            f"{list(PUBLIC_OBS_KEYS)}; missing={missing} extra={extra}")
    leaks = find_forbidden_keys(dict(obs))
    if leaks:
        raise EnvContractError(
            f"V2 observation for {qid!r} carries label-bearing field(s) {leaks}")
    if str(obs["qid"]) != str(qid):
        raise EnvContractError(
            f"observation qid {obs['qid']!r} does not match the reset qid {qid!r}")
    ids, texts = obs["seen_doc_ids"], obs["seen_doc_texts"]
    if len(ids) != len(texts):
        raise EnvContractError(
            f"{qid!r}: seen_doc_ids/seen_doc_texts length mismatch "
            f"({len(ids)} vs {len(texts)})")
    mask = obs["action_mask"]
    if not isinstance(mask, (list, tuple)) or len(mask) != N_ACTIONS:
        raise EnvContractError(
            f"{qid!r}: action_mask must be a sequence of {N_ACTIONS} booleans; got {mask!r}")
    unknown = [name for name in obs["history"] if name not in ACTION_ORDER]
    if unknown:
        raise EnvContractError(f"{qid!r}: unknown history action name(s) {unknown}")


def _bounded_action(raw: Any, qid: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise EnvContractError(
            f"policy returned a non-integer action for {qid!r}: {raw!r}")
    action = int(raw)
    if not 0 <= action < N_ACTIONS:
        raise EnvContractError(
            f"policy returned action {action} outside [0, {N_ACTIONS}) for {qid!r}")
    return action


def _run_episode(env: Any, qid: str, policy: Callable[[Mapping[str, Any]], int]
                 ) -> tuple[Mapping[str, Any], Mapping[str, Any], list[float], float, float]:
    """One episode for ``qid``: public obs in, action out, then the private outcome.

    Returns ``(final_obs, outcome, rewards, policy_latency_ms, env_replay_latency_ms)``. The
    policy's action is validated against the mask *before* the environment sees it — a
    masked-out action is a policy contract violation. The watchdog is derived from the
    episode's own budget (``remaining_optional_searches`` at reset + the terminating STOP) —
    not an arbitrary constant — and a truncated episode is a contract error, because a legal
    mask path always ends in STOP well within that budget.
    """
    started = time.perf_counter()
    obs, _info = env.reset(qid=qid)
    env_ms = (time.perf_counter() - started) * 1000.0
    _check_public_observation(obs, qid)
    budget = int(obs["remaining_optional_searches"])
    if budget < 0:
        raise EnvContractError(f"{qid!r}: reset reports negative remaining_optional_searches")
    max_steps = budget + 1

    terminated = truncated = False
    steps = 0
    policy_ms = 0.0
    rewards: list[float] = []
    while not (terminated or truncated):
        if steps >= max_steps:
            raise EnvContractError(
                f"episode {qid!r} did not terminate within its own budget ({max_steps} steps: "
                "remaining optional searches + STOP)")
        started = time.perf_counter()
        raw = policy(obs)
        policy_ms += (time.perf_counter() - started) * 1000.0
        action = _bounded_action(raw, qid)
        if not obs["action_mask"][action]:
            raise EnvContractError(
                f"policy returned the masked-out action {ACTION_ORDER[action]} for {qid!r}")
        started = time.perf_counter()
        obs, reward, terminated, truncated, _info = env.step(action)
        env_ms += (time.perf_counter() - started) * 1000.0
        rewards.append(float(reward))
        steps += 1
        _check_public_observation(obs, qid)
    if truncated:
        raise EnvContractError(
            f"episode {qid!r} was truncated; the V2 budget has no legal truncation path")
    if len(obs["history"]) != steps:
        raise EnvContractError(
            f"{qid!r}: history carries {len(obs['history'])} action(s) after {steps} step(s)")
    outcome = env.private_outcome()
    if not isinstance(outcome, Mapping):
        raise EnvContractError(
            f"private_outcome() returned no mapping for {qid!r} after termination")
    return obs, outcome, rewards, policy_ms, env_ms


# ------------------------------------------------------------------------------- scoring

def _score(qid: str, arm: str, world: Any, obs: Mapping[str, Any], policy_ms: float,
           env_ms: float) -> dict[str, Any]:
    """Exact-set metrics of one terminated episode, from the final ordered document set."""
    seen = [str(doc_id) for doc_id in obs["seen_doc_ids"]]
    if len(set(seen)) != len(seen):
        raise EnvContractError(f"{qid!r}: duplicated document(s) in the final document set")
    if not seen:
        raise EnvContractError(
            f"{qid!r}: episode ended with no visible document; the free bootstrap must reveal one")
    gold = [str(doc_id) for doc_id in world.gold_context_ids]
    seen_set = set(seen)
    matched = [doc_id for doc_id in gold if doc_id in seen_set]
    missing = [doc_id for doc_id in gold if doc_id not in seen_set]
    gold_recall = len(matched) / len(gold)
    gold_precision = len(matched) / len(seen)
    full_gold_success = not missing
    n_search = int(obs["optional_used"])
    return {
        "qid": qid,
        "arm": arm,
        "actions": [str(name) for name in obs["history"]],
        "n_search": n_search,
        "seen_doc_ids": seen,
        "seen_count": len(seen),
        "gold_context_ids": gold,
        "matched_gold_ids": matched,
        "missing_gold_ids": missing,
        "gold_recall": float(gold_recall),
        "gold_precision": float(gold_precision),
        "full_gold_success": bool(full_gold_success),
        "retrieval_return": float(retrieval_return(gold_recall, full_gold_success, n_search)),
        "policy_latency_ms": float(policy_ms),
        "env_replay_latency_ms": float(env_ms),
    }


def _cross_check_outcome(outcome: Mapping[str, Any], record: Mapping[str, Any],
                         rewards: Sequence[float], qid: str) -> None:
    """The environment's private bookkeeping and rewards must match the evaluator's arithmetic.

    ``private_outcome()`` is bookkeeping (identity, the ordered seen ids, the gold ids, the
    SEARCH count), so the cross-check compares those values exactly against the evaluator's
    own record and then verifies the reward stream: the terminal reward must equal
    ``0.5*coverage + I(G⊆S)`` and the reward sum must equal the evaluator's retrieval return.
    Any disagreement is a contract error, never a scored zero.
    """
    problems = []
    required = ("qid", "terminal_action", "n_search", "seen_doc_ids", "gold_context_ids")
    missing = [key for key in required if key not in outcome]
    if missing:
        raise EnvContractError(
            f"private_outcome() has no {missing} key(s) for {qid!r}")
    if str(outcome["qid"]) != qid:
        problems.append(f"qid: evaluator {qid!r} != env {outcome['qid']!r}")
    if outcome["terminal_action"] != ACTION_ORDER[STOP]:
        problems.append(
            f"terminal_action: expected {ACTION_ORDER[STOP]!r}, env {outcome['terminal_action']!r}")
    if outcome["n_search"] != record["n_search"]:
        problems.append(f"n_search: evaluator {record['n_search']!r} != env {outcome['n_search']!r}")
    if [str(doc_id) for doc_id in outcome["seen_doc_ids"]] != record["seen_doc_ids"]:
        problems.append(
            f"seen_doc_ids: evaluator {record['seen_doc_ids']!r} != "
            f"env {list(outcome['seen_doc_ids'])!r}")
    if [str(doc_id) for doc_id in outcome["gold_context_ids"]] != record["gold_context_ids"]:
        problems.append(
            f"gold_context_ids: evaluator {record['gold_context_ids']!r} != "
            f"env {list(outcome['gold_context_ids'])!r}")
    stop_reward = 0.5 * record["gold_recall"] + (1.0 if record["full_gold_success"] else 0.0)
    if not rewards:
        problems.append("no rewards were recorded for the episode")
    else:
        if not math.isclose(rewards[-1], stop_reward, rel_tol=1e-9, abs_tol=1e-9):
            problems.append(
                f"terminal reward: evaluator {stop_reward!r} != env {rewards[-1]!r}")
        if not math.isclose(sum(rewards), record["retrieval_return"], rel_tol=1e-9,
                            abs_tol=1e-9):
            problems.append(
                f"reward sum: evaluator {record['retrieval_return']!r} != env {sum(rewards)!r}")
    if problems:
        raise EnvContractError(
            f"private_outcome() disagrees with the evaluator for {qid!r}: " + "; ".join(problems))


def _validate_worlds(worlds: Sequence[Any]) -> list[str]:
    qids: list[str] = []
    for world in worlds:
        qid = getattr(world, "qid", None)
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"every world must carry a non-empty string qid; got {qid!r}")
        gold = getattr(world, "gold_context_ids", None)
        if not gold:
            raise ValueError(f"world {qid!r} has no non-empty gold_context_ids")
        if len(set(map(str, gold))) != len(gold):
            raise ValueError(f"world {qid!r} gold_context_ids contain duplicates")
        qids.append(qid)
    duplicates = sorted({qid for qid in qids if qids.count(qid) > 1})
    if duplicates:
        raise ValueError(f"duplicate world qids are not allowed: {duplicates}")
    return qids


# --------------------------------------------------------------------------------- run

def evaluate_policy(worlds: Sequence[Any], policy: Callable[[Mapping[str, Any]], int],
                    arm: str, *, teacher_factory: Callable[[Any], Any] | None = None
                    ) -> dict[str, Any]:
    """Evaluate one arm over exactly ``worlds`` (each QID one episode, input order preserved).

    ``policy`` is any callable ``obs -> action`` (the arms in
    :mod:`src.pipeline.vixstory_arms_v2`, or a parent-supplied REINFORCE wrapper); it only
    ever receives the public observation. The returned dict carries the aggregate ``summary``
    and the private per-QID ``per_qid`` records.

    When ``teacher_factory`` is given, it is called once per world and must return an object
    with ``oracle_return()`` and ``full_gold_reachable()`` (the teacher's private interface);
    the summary then carries the oracle upper bound over the same rankings/budget. Without a
    teacher the oracle block is ``None`` — it is never fabricated.
    """
    world_list = list(worlds)
    if not world_list:
        raise ValueError("evaluate_policy requires at least one world")
    if not isinstance(arm, str) or not arm.strip():
        raise ValueError("arm must be a non-empty string")
    if not callable(policy):
        raise TypeError("policy must be a callable obs -> action")
    if teacher_factory is not None and not callable(teacher_factory):
        raise TypeError("teacher_factory must be a callable world -> teacher")
    qids = _validate_worlds(world_list)

    env = VixStoryEnvV2(world_list)
    per_qid: dict[str, dict[str, Any]] = {}
    for world in world_list:
        obs, outcome, rewards, policy_ms, env_ms = _run_episode(env, world.qid, policy)
        record = _score(world.qid, arm, world, obs, policy_ms, env_ms)
        _cross_check_outcome(outcome, record, rewards, world.qid)
        record["env_outcome_checked"] = True
        if teacher_factory is not None:
            teacher = teacher_factory(world)
            record["oracle_return"] = float(teacher.oracle_return())
            record["full_gold_reachable"] = bool(teacher.full_gold_reachable())
        per_qid[world.qid] = record

    summary: dict[str, Any] = {
        "arm": arm,
        "n": len(qids),
        "qids": qids,
        "qids_sha256": sha256_json(qids),
        "search_cost": SEARCH_COST,
        "mean_gold_recall": _mean(per_qid, "gold_recall"),
        "mean_gold_precision": _mean(per_qid, "gold_precision"),
        "mean_full_gold_success": _mean(per_qid, "full_gold_success"),
        "full_gold_success_n": sum(1 for r in per_qid.values() if r["full_gold_success"]),
        "mean_retrieval_return": _mean(per_qid, "retrieval_return"),
        "mean_n_search": _mean(per_qid, "n_search"),
        "total_n_search": int(sum(r["n_search"] for r in per_qid.values())),
        "mean_seen_count": _mean(per_qid, "seen_count"),
        "mean_policy_latency_ms": _mean(per_qid, "policy_latency_ms"),
        "total_policy_latency_ms": float(sum(r["policy_latency_ms"] for r in per_qid.values())),
        "mean_env_replay_latency_ms": _mean(per_qid, "env_replay_latency_ms"),
        "total_env_replay_latency_ms": float(sum(r["env_replay_latency_ms"]
                                                 for r in per_qid.values())),
        "latency_note": LATENCY_NOTE,
        "oracle": None,
    }
    if teacher_factory is not None:
        reachable = [bool(r["full_gold_reachable"]) for r in per_qid.values()]
        summary["oracle"] = {
            "mean_oracle_return": _mean(per_qid, "oracle_return"),
            "full_gold_reachable_rate": sum(reachable) / len(reachable),
            "full_gold_reachable_n": sum(1 for value in reachable if value),
        }
    return {"schema": SCHEMA, "arm": arm, "summary": summary, "per_qid": per_qid}


def _mean(per_qid: Mapping[str, Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(record[key]) for record in per_qid.values()]))


# ---------------------------------------------------------------------------- selection

def select_best(candidates: Sequence[Mapping[str, Any]],
                *, tolerance: float = SELECTION_TOLERANCE) -> Mapping[str, Any]:
    """The pre-registered shared selection rule (plan §5), fixed before any result is seen.

    Highest **mean retrieval return** wins; ties (within ``tolerance``, documented at
    :data:`SELECTION_TOLERANCE`) go to the lower **mean SEARCH count**; remaining ties to the
    lower **index** (checkpoint epoch / candidate order). Each candidate must carry the keys
    ``mean_retrieval_return``, ``mean_n_search`` and ``index`` — typically a
    :func:`evaluate_policy` summary with the checkpoint or K index added. Returns the winning
    mapping. Missing keys or an empty sequence raise.
    """
    items = list(candidates)
    if not items:
        raise ValueError("select_best requires at least one candidate")
    required = ("mean_retrieval_return", "mean_n_search", "index")
    for position, item in enumerate(items):
        missing = [key for key in required if key not in item]
        if missing:
            raise ValueError(f"candidate {position} is missing {missing}")
    best = items[0]
    for item in items[1:]:
        best = _preferred(best, item, tolerance)
    return best


def _preferred(a: Mapping[str, Any], b: Mapping[str, Any], tolerance: float
               ) -> Mapping[str, Any]:
    delta = float(a["mean_retrieval_return"]) - float(b["mean_retrieval_return"])
    if abs(delta) > tolerance:
        return a if delta > 0 else b
    searches_a, searches_b = float(a["mean_n_search"]), float(b["mean_n_search"])
    if searches_a != searches_b:
        return a if searches_a < searches_b else b
    return a if float(a["index"]) <= float(b["index"]) else b


# ------------------------------------------------------------------ paired differences

def paired_difference(per_qid_a: Mapping[str, Mapping[str, Any]],
                      per_qid_b: Mapping[str, Mapping[str, Any]], *, metric: str,
                      qids: Sequence[str] | None = None, bootstrap: bool = True,
                      n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05
                      ) -> dict[str, Any]:
    """Paired per-QID difference ``a - b`` on ``metric``, with an optional QID bootstrap CI.

    The two arms must have been evaluated on **identical QID sets** (a mismatch is an error,
    never a silent intersection) and duplicate QIDs are rejected. The bootstrap resamples
    QIDs with replacement, so the pairing is preserved; it is deterministic for the fixed
    seed (42) and independent of mapping order. The result is labeled
    :data:`DEVELOPMENT_LABEL`: validation-split development evaluation, single seed, CI over
    question variation only.
    """
    if not isinstance(per_qid_a, Mapping) or not isinstance(per_qid_b, Mapping):
        raise TypeError("paired_difference takes two per-QID mappings")
    keys_a, keys_b = set(per_qid_a), set(per_qid_b)
    if keys_a != keys_b:
        raise ValueError(
            f"QID sets do not match: only in left {sorted(keys_a - keys_b)}, "
            f"only in right {sorted(keys_b - keys_a)}")
    if qids is None:
        qid_list = sorted(keys_a)
    else:
        qid_list = [str(qid) for qid in qids]
        duplicates = sorted({qid for qid in qid_list if qid_list.count(qid) > 1})
        if duplicates:
            raise ValueError(f"duplicate qids are not allowed: {duplicates}")
        if set(qid_list) != keys_a:
            raise ValueError("qids must match the two arms' QID sets exactly")
    if not qid_list:
        raise ValueError("paired_difference needs at least one paired QID")

    differences: dict[str, float] = {}
    for qid in qid_list:
        for side, mapping in (("left", per_qid_a), ("right", per_qid_b)):
            if metric not in mapping[qid]:
                raise KeyError(f"{side} record for {qid!r} has no metric {metric!r}")
        differences[qid] = float(per_qid_a[qid][metric]) - float(per_qid_b[qid][metric])

    result: dict[str, Any] = {
        "metric": metric,
        "n": len(qid_list),
        "qids": qid_list,
        "mean_difference": float(np.mean([differences[qid] for qid in qid_list])),
        "bootstrap": None,
        "evaluation_label": DEVELOPMENT_LABEL,
        "evaluation_label_note": DEVELOPMENT_LABEL_NOTE,
    }
    if bootstrap:
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        values = np.asarray([differences[qid] for qid in sorted(differences)],
                            dtype=np.float64)
        rng = np.random.default_rng(int(seed))
        draws = values[rng.integers(0, values.size, size=(int(n_resamples), values.size))]
        low, high = np.percentile(draws.mean(axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])
        result["bootstrap"] = {
            "ci_low": float(low), "ci_high": float(high), "n": int(values.size),
            "n_resamples": int(n_resamples), "seed": int(seed), "alpha": float(alpha),
        }
    return result

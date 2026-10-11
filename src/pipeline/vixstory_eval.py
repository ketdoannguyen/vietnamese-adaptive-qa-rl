"""VixSTORY E00-E07 paired evaluation scaffolding — dependency-injected, no GPU, no data read.

Why a separate module
    The ViQuAD evaluator (:mod:`src.pipeline.eval_baselines` with :mod:`src.eval.rollout`) is
    bound to ``AdaptiveQAEnv``, the ViQuAD reward and the ViQuAD reader/reporting stack.
    VixSTORY has free-form answers, several gold context ids and its own frozen reader, so
    reusing that stack would silently change the measurement. This module is the paired
    evaluation *skeleton* only: it imports neither the environment, the reward, the reader nor
    the dataset, and it never reads the official validation split. Everything it runs is
    injected by the caller.

Arms
    ==================  =========================  ======================
    E00  no optional search  E04  Laya CE prior     ABSTAIN      always abstain
    E01  always BM25         E05  Laya RLCD prior   RANDOM_VALID uniform over legal
    E02  always Dense        E06  REINFORCE/RLCD
    E03  p_correct threshold E07  REINFORCE/CE
    ==================  =========================  ======================

Fairness by construction
    One zero-argument ``env_factory`` builds every arm's environment, and every arm resets it
    for the **same ordered qids** with the same per-episode seed, so the retrieval budget, the
    reader and the corpus are identical across arms. A policy only ever sees the observation
    that ``reset``/``step`` return; ``private_outcome`` is read only *after* an episode
    terminates and is used for scoring alone, never to choose an action.

Minimal environment contract (the *actual* ``src.rl.vixstory_env.VixStoryEnv`` outcome)
    ``reset(seed=..., options={"qid": qid}) -> (obs, info)``
    ``step(action) -> (obs, reward, terminated, truncated, info)``
    ``private_outcome() -> mapping | None`` (meaningful only after termination) exposing
    ``total`` (episode return = terminal utility minus every optional SEARCH cost), the
    free-form ``f1``/``em`` the environment computed against its private gold, the chosen
    ``terminal_action``, ``n_optional_searches`` and the tool counts
    ``n_bm25_searches``/``n_dense_searches``. The tool counts include the *free* bootstrap
    retrieval, so the optional SEARCH actions are read from ``n_optional_searches`` and the
    bootstrap is recovered as ``n_bm25_searches + n_dense_searches - n_optional_searches``.
    ``citation_valid`` (bool | None) is optional. This is the ViSTORY contract; the ViQuAD
    naming ``correctness_f1``/``correctness_em`` (see :mod:`src.eval.rollout`) does **not**
    apply and is never read here.

Statuses
    ``SUCCEEDED`` / ``FAILED`` / ``NOT_RUN``. A missing policy is ``NOT_RUN``, a raising arm is
    ``FAILED``, and neither fabricates a metric: every metric is ``None`` (N/A) when the arm
    did not run to completion. ``success`` is True only when every mandatory arm (E01-E07)
    succeeded; missing mandatory arms are named in ``missing_mandatory``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from src.utils.hashing import sha256_json, stable_seed

#: Version tag of the returned report dict.
SCHEMA = "vixstory-eval/v1"

STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_NOT_RUN = "NOT_RUN"

#: The full arm matrix, in report order.
ARM_IDS: tuple[str, ...] = (
    "E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "ABSTAIN", "RANDOM_VALID",
)
#: Arms that must all succeed for the evaluation to count as a final success.
MANDATORY_ARMS: tuple[str, ...] = ("E01", "E02", "E03", "E04", "E05", "E06", "E07")
#: The two arithmetic controls every adaptive arm is compared against.
CONTROL_ARMS: tuple[str, ...] = ("ABSTAIN", "RANDOM_VALID")
#: Arms a comparison block may use as its left-hand (reference) side.
REFERENCE_ARMS: tuple[str, ...] = ("E00", "ABSTAIN", "RANDOM_VALID")
#: Metrics compared per qid against each reference arm.
COMPARISON_METRICS: tuple[str, ...] = ("reward", "f1", "em")

#: Required keys of ``private_outcome()``; a missing key is a contract mismatch, not a zero.
#: These are the real ``VixStoryEnv`` fields — there is no ``correctness_f1``/``correctness_em``.
REQUIRED_OUTCOME_KEYS: tuple[str, ...] = (
    "total", "f1", "em", "terminal_action",
    "n_optional_searches", "n_bm25_searches", "n_dense_searches",
)

_NOT_RUN_CONTRACT = ("private_outcome() must expose total, f1, em, terminal_action, "
                     "n_optional_searches and the n_bm25_searches/n_dense_searches tool counts "
                     "after every terminated episode")


class EnvContractError(RuntimeError):
    """The injected environment did not honour the minimal reset/step/private_outcome contract."""


class Env(Protocol):
    """The only environment surface this module touches."""

    def reset(self, *, seed: int | None = ..., options: dict | None = ...) -> tuple[Any, Any]: ...

    def step(self, action: int) -> tuple[Any, float, bool, bool, Any]: ...

    def private_outcome(self) -> Mapping[str, Any] | None: ...


class Policy(Protocol):
    """An injected action policy. ``begin_episode`` is optional and only used when present."""

    def act(self, obs: Mapping[str, Any]) -> int: ...


# ------------------------------------------------------------------------------- statistics

def paired_bootstrap_ci(diff_by_qid: Mapping[str, float], *, n_resamples: int = 1000,
                        seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    """Percentile CI for a **paired** per-qid difference, resampling qids with replacement.

    The two sides of ``diff_by_qid`` are already paired (same qid), so resampling qid indices
    keeps the pairing intact. The generator is derived from ``seed`` alone, so the interval is
    reproducible for a fixed seed and independent of dict insertion order.
    """
    if not diff_by_qid:
        raise ValueError("paired bootstrap needs at least one paired observation")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    qids = sorted(diff_by_qid)
    values = np.asarray([float(diff_by_qid[qid]) for qid in qids], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    n = values.size
    draws = values[rng.integers(0, n, size=(int(n_resamples), n))].mean(axis=1)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "n": int(n), "n_resamples": int(n_resamples), "seed": int(seed),
            "alpha": float(alpha)}


# --------------------------------------------------------------------------- episode runner

def _run_episode(env: Env, policy: Policy, *, qid: str, seed: int, max_steps: int
                 ) -> Mapping[str, Any]:
    """One episode: actions come from ``obs`` only, then ``private_outcome`` is scored.

    ``private_outcome`` is read strictly *after* ``terminated``/``truncated``; a policy never
    sees it. A watchdog raises instead of looping if the episode will not end.
    """
    begin = getattr(policy, "begin_episode", None)
    if begin is not None:
        begin(qid=qid, seed=int(seed))
    obs, _info = env.reset(seed=stable_seed("vixstory", int(seed), qid), options={"qid": qid})
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        if steps >= max_steps:
            raise EnvContractError(f"episode {qid!r} did not terminate within {max_steps} steps")
        action = int(policy.act(obs))
        obs, _reward, terminated, truncated, _info = env.step(action)
        steps += 1
    outcome = env.private_outcome()
    if not isinstance(outcome, Mapping):
        raise EnvContractError(f"private_outcome() returned no mapping for {qid!r} "
                               f"after termination")
    missing = [key for key in REQUIRED_OUTCOME_KEYS if key not in outcome]
    if missing:
        raise EnvContractError(f"private_outcome() for {qid!r} is missing {missing}; "
                               f"{_NOT_RUN_CONTRACT}")
    return outcome


def _score(outcome: Mapping[str, Any], qid: str) -> tuple[dict[str, Any], str]:
    """Per-qid record from a terminated episode's private outcome, plus a contract note.

    ``reward`` is the environment's own episode return (``total``); ``f1``/``em`` are the
    free-form correctness the environment computed against its private gold (never
    ``correctness_f1``/``correctness_em``). The optional SEARCH actions are read from
    ``n_optional_searches`` and the free bootstrap is separated out from the tool counts, so
    ``optional_searches`` never hides the retrieval the environment forced at reset.
    """
    tool_calls = int(outcome["n_bm25_searches"]) + int(outcome["n_dense_searches"])
    optional = int(outcome["n_optional_searches"])
    bootstrap = tool_calls - optional
    if bootstrap < 0:
        raise EnvContractError(
            f"private_outcome() for {qid!r} reports {tool_calls} tool call(s) but {optional} "
            "optional search(es); the free bootstrap count cannot be negative")
    citation = outcome.get("citation_valid")
    success = outcome.get("correct")
    record = {
        "reward": float(outcome["total"]),
        "f1": float(outcome["f1"]),
        "em": float(outcome["em"]),
        "answered": str(outcome["terminal_action"]) == "ANSWER",
        "success": bool(success) if success is not None else None,
        "optional_searches": optional,
        "bootstrap_searches": bootstrap,
        "citation_valid": None if citation is None else bool(citation),
    }
    missing = [name for name in ("citation_valid", "correct") if name not in outcome]
    note = "" if not missing else (
        f"private_outcome() does not expose {missing}; those metrics are reported N/A")
    return record, note


def _run_arm(env_factory: Callable[[], Env], policy: Policy, *, arm: str,
             qids: Sequence[str], seed: int, max_steps: int
             ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Evaluate one arm over exactly ``qids``; any failure raises and fails the whole arm."""
    env = env_factory()
    per_qid: dict[str, dict[str, Any]] = {}
    notes: set[str] = set()
    for qid in qids:
        try:
            outcome = _run_episode(env, policy, qid=qid, seed=seed, max_steps=max_steps)
        except EnvContractError as exc:
            raise EnvContractError(f"arm {arm}: {exc}") from exc
        record, note = _score(outcome, qid)
        if note:
            notes.add(note)
        per_qid[qid] = record
    if set(per_qid) != set(qids):
        raise EnvContractError(f"arm {arm} did not evaluate exactly the injected qids")
    return per_qid, sorted(notes)


# ------------------------------------------------------------------------------ aggregation

def _blank_metrics() -> dict[str, Any]:
    return {"n": 0, "mean_reward": None, "mean_f1": None, "mean_em": None,
            "answer_rate": None, "success_rate": None, "success_n": 0,
            "mean_optional_searches": None,
            "total_optional_searches": None, "mean_bootstrap_searches": None,
            "total_bootstrap_searches": None,
            "citation_valid_rate": None, "citation_valid_n": 0}


def _aggregate(per_qid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    n = len(per_qid)
    if not n:
        return _blank_metrics()
    rows = list(per_qid.values())
    citations = [r["citation_valid"] for r in rows if r["citation_valid"] is not None]
    successes = [r["success"] for r in rows if r["success"] is not None]
    return {
        "n": n,
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_em": float(np.mean([r["em"] for r in rows])),
        "answer_rate": float(np.mean([r["answered"] for r in rows])),
        "success_rate": float(np.mean(successes)) if successes else None,
        "success_n": len(successes),
        "mean_optional_searches": float(np.mean([r["optional_searches"] for r in rows])),
        "total_optional_searches": int(sum(r["optional_searches"] for r in rows)),
        "mean_bootstrap_searches": float(np.mean([r["bootstrap_searches"] for r in rows])),
        "total_bootstrap_searches": int(sum(r["bootstrap_searches"] for r in rows)),
        "citation_valid_rate": float(np.mean(citations)) if citations else None,
        "citation_valid_n": len(citations),
    }


def _blank_arm(status: str, reason: str) -> dict[str, Any]:
    return {"status": status, "reason": reason, "n": 0, "metrics": _blank_metrics(),
            "per_qid": {}}


def _succeeded_arm(per_qid: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {"status": STATUS_SUCCEEDED, "reason": "", "n": len(per_qid),
            "metrics": _aggregate(per_qid), "per_qid": dict(per_qid)}


# ------------------------------------------------------------------------------------- run

def run_vixstory_eval(*, env_factory: Callable[[], Env], policies: Mapping[str, Policy],
                      qids: Sequence[str], seed: int = 42, ci_seed: int = 42,
                      n_resamples: int = 1000, alpha: float = 0.05, max_steps: int = 64
                      ) -> dict[str, Any]:
    """Run every injected arm over the same ``qids`` and return the paired report.

    ``policies`` maps arm id (see :data:`ARM_IDS`) to an injected policy. A missing arm is
    recorded ``NOT_RUN``; an arm that raises is recorded ``FAILED``. In both cases every
    metric stays ``None`` — no result is fabricated. ``env_factory`` is called once per arm and
    must be deterministic, so every arm sees the same retrieval/reader/budget configuration.
    """
    qid_list = [str(qid) for qid in qids]
    if not qid_list:
        raise ValueError("qids must be a non-empty sequence")
    duplicates = sorted({qid for qid in qid_list if qid_list.count(qid) > 1})
    if duplicates:
        raise ValueError(f"duplicate qids are not allowed: {duplicates}")
    unknown = sorted(set(policies) - set(ARM_IDS))
    if unknown:
        raise ValueError(f"unknown arm id(s) {unknown}; known arms: {list(ARM_IDS)}")

    arms: dict[str, dict[str, Any]] = {}
    limitations: list[str] = []
    for arm in ARM_IDS:
        if arm not in policies:
            arms[arm] = _blank_arm(
                STATUS_NOT_RUN, f"no policy was injected for arm {arm!r}")
            continue
        try:
            per_qid, notes = _run_arm(env_factory, policies[arm], arm=arm, qids=qid_list,
                                      seed=seed, max_steps=max_steps)
        except Exception as exc:  # one broken arm must not lose the others
            arms[arm] = _blank_arm(STATUS_FAILED, f"{type(exc).__name__}: {exc}")
            continue
        arms[arm] = _succeeded_arm(per_qid)
        limitations.extend(note for note in notes if note not in limitations)

    comparisons: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        if arms[arm]["status"] != STATUS_SUCCEEDED:
            continue
        for reference in REFERENCE_ARMS:
            if reference == arm or arms[reference]["status"] != STATUS_SUCCEEDED:
                continue
            left, right = arms[arm]["per_qid"], arms[reference]["per_qid"]
            block: dict[str, Any] = {}
            for metric in COMPARISON_METRICS:
                diff = {qid: left[qid][metric] - right[qid][metric] for qid in qid_list}
                block[metric] = paired_bootstrap_ci(
                    diff, n_resamples=n_resamples, alpha=alpha,
                    seed=stable_seed(ci_seed, "vixstory-paired", arm, reference))
            comparisons[f"{arm}_minus_{reference}"] = block

    for arm in ARM_IDS:
        if arms[arm]["status"] != STATUS_SUCCEEDED:
            limitations.append(f"arm {arm} is {arms[arm]['status']}: {arms[arm]['reason']}")
    for control in CONTROL_ARMS:
        if arms[control]["status"] != STATUS_SUCCEEDED:
            limitations.append(f"control {control} did not run; comparisons against it are absent")
    limitations.append("Confidence intervals resample qids (paired, seeded); they do not model "
                       "title-group clustering, so VixSTORY's shared corpus can make them "
                       "optimistic.")

    missing_mandatory = [arm for arm in MANDATORY_ARMS
                         if arms[arm]["status"] != STATUS_SUCCEEDED]
    return {
        "schema": SCHEMA,
        "seed": int(seed),
        "ci_seed": int(ci_seed),
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
        "max_steps": int(max_steps),
        "n_qids": len(qid_list),
        "qids_sha256": sha256_json(qid_list),
        "arms": arms,
        "comparisons": comparisons,
        "success": not missing_mandatory,
        "mandatory": list(MANDATORY_ARMS),
        "missing_mandatory": missing_mandatory,
        "limitations": limitations,
    }

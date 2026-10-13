"""Hand-computed retrieval-metric tests for the V2 evaluator (written before the module).

Every expected number is derived by hand from the pinned reward and the fixture rankings
(``.claude/plans/2026-10-12-vixstory-evidence-first-v2.md`` §3-§4), so the tests fail if the
evaluator drifts from the exact-set definitions: recall/precision/full-gold success are exact
ID-set operations on the final document set at STOP, the per-QID record preserves the ordered
``seen_doc_ids``, and nothing but the public observation ever reaches a policy.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.data.schemas import find_forbidden_keys
from src.pipeline.vixstory_arms_v2 import fixed_tool, masked_greedy, stop
from src.pipeline.vixstory_eval_v2 import (
    DEVELOPMENT_LABEL,
    LATENCY_NOTE,
    SCHEMA,
    EnvContractError,
    evaluate_policy,
    paired_difference,
    select_best,
)
from src.rl.vixstory_env_v2 import SEARCH_BM25, SEARCH_DENSE, STOP, VixStoryEnvV2, VixStoryWorldV2
from src.utils.hashing import sha256_json

ROOT = Path(__file__).resolve().parents[1]

CORPUS = {
    "d0": "VixSTORY: trường A toạ lạc tại Hà Nội.",
    "d1": "Trường A thành lập năm 1995.",
    "d2": "Trường A có 12 khoa.",
    "d3": "Trường A có 3 viện nghiên cứu.",
    "d4": "Đoạn nhiễu không liên quan.",
}
#: Bootstrap (BM25, 1 passage) -> d0. One SEARCH reveals the next two unseen ids of its tool:
#: bm25 K=1 -> d1, d2; bm25 K=2 -> then d4; dense (after bm25 K=1) -> d3.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d4"], "dense": ["d0", "d2", "d1", "d3"]}


def make_world(qid, gold, *, rankings=None, corpus=None):
    return VixStoryWorldV2(qid=qid, question=f"Câu hỏi cho {qid}?",
                           corpus=CORPUS if corpus is None else corpus,
                           rankings=RANKINGS if rankings is None else rankings,
                           gold_context_ids=tuple(gold))


# --------------------------------------------------------------------- exact-set scoring

def test_stop_arm_hand_computed_zero_coverage():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], stop, "E00")
    assert result["schema"] == SCHEMA
    assert result["arm"] == "E00"
    record = result["per_qid"]["q1"]
    assert record["arm"] == "E00"
    assert record["seen_doc_ids"] == ["d0"]
    assert record["actions"] == ["STOP"]
    assert record["n_search"] == 0
    assert record["gold_recall"] == 0.0
    assert record["gold_precision"] == 0.0
    assert record["full_gold_success"] is False
    assert record["retrieval_return"] == 0.0
    assert record["matched_gold_ids"] == []
    assert record["missing_gold_ids"] == ["d1", "d2"]
    summary = result["summary"]
    assert summary["arm"] == "E00"
    assert summary["n"] == 1
    assert summary["mean_retrieval_return"] == 0.0
    assert summary["mean_gold_recall"] == 0.0
    assert summary["mean_full_gold_success"] == 0.0
    assert summary["full_gold_success_n"] == 0
    assert summary["mean_n_search"] == 0.0
    assert summary["total_n_search"] == 0


def test_full_gold_from_the_free_bootstrap_pays_one_and_a_half():
    result = evaluate_policy([make_world("q1", ["d0"])], stop, "E00")
    record = result["per_qid"]["q1"]
    assert (record["gold_recall"], record["gold_precision"]) == (1.0, 1.0)
    assert record["full_gold_success"] is True
    assert record["retrieval_return"] == pytest.approx(0.5 + 1.0)


def test_fixed_bm25_k1_reaches_full_gold_with_one_search():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 1), "E01")
    record = result["per_qid"]["q1"]
    assert record["seen_doc_ids"] == ["d0", "d1", "d2"]
    assert record["actions"] == ["SEARCH_BM25", "STOP"]
    assert record["n_search"] == 1
    assert record["gold_recall"] == 1.0
    assert record["gold_precision"] == pytest.approx(2 / 3)
    assert record["full_gold_success"] is True
    assert record["retrieval_return"] == pytest.approx(1.5 - 0.05)
    assert record["matched_gold_ids"] == ["d1", "d2"]
    assert record["missing_gold_ids"] == []


def test_fixed_bm25_k2_second_search_costs_and_dilutes_precision():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 2), "E01")
    record = result["per_qid"]["q1"]
    assert record["seen_doc_ids"] == ["d0", "d1", "d2", "d4"]
    assert record["n_search"] == 2
    assert record["gold_recall"] == 1.0
    assert record["gold_precision"] == 0.5
    assert record["full_gold_success"] is True
    assert record["retrieval_return"] == pytest.approx(1.5 - 0.10)


def test_partial_coverage_is_not_full_success():
    result = evaluate_policy([make_world("q1", ["d2", "d3"])], fixed_tool("bm25", 1), "E01")
    record = result["per_qid"]["q1"]
    assert record["seen_doc_ids"] == ["d0", "d1", "d2"]
    assert record["gold_recall"] == 0.5
    assert record["gold_precision"] == pytest.approx(1 / 3)
    assert record["full_gold_success"] is False
    assert record["retrieval_return"] == pytest.approx(0.25 - 0.05)
    assert record["matched_gold_ids"] == ["d2"]
    assert record["missing_gold_ids"] == ["d3"]


def test_partial_then_dense_search_completes_the_gold_set():
    def policy(obs):
        if obs["optional_used"] == 0:
            return SEARCH_BM25
        if obs["optional_used"] == 1:
            return SEARCH_DENSE
        return STOP

    result = evaluate_policy([make_world("q1", ["d1", "d3"])], policy, "MIXED")
    record = result["per_qid"]["q1"]
    assert record["seen_doc_ids"] == ["d0", "d1", "d2", "d3"]
    assert record["actions"] == ["SEARCH_BM25", "SEARCH_DENSE", "STOP"]
    assert record["gold_recall"] == 1.0
    assert record["gold_precision"] == 0.5
    assert record["full_gold_success"] is True
    assert record["retrieval_return"] == pytest.approx(1.5 - 0.10)


# --------------------------------------------------------------------------- strictness

def test_every_world_is_evaluated_exactly_once_in_input_order():
    worlds = [make_world("qb", ["d1"]), make_world("qa", ["d2"])]
    result = evaluate_policy(worlds, stop, "E00")
    assert list(result["per_qid"]) == ["qb", "qa"]
    assert result["summary"]["n"] == 2
    assert result["summary"]["qids"] == ["qb", "qa"]
    assert result["summary"]["qids_sha256"] == sha256_json(["qb", "qa"])
    assert result["summary"]["mean_gold_recall"] == 0.0
    assert result["summary"]["mean_n_search"] == 0.0


def test_duplicate_worlds_are_rejected():
    world = make_world("q1", ["d0"])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_policy([world, make_world("q1", ["d0"])], stop, "E00")


def test_empty_worlds_and_bad_arm_and_bad_policy_are_rejected():
    with pytest.raises(ValueError):
        evaluate_policy([], stop, "E00")
    with pytest.raises(ValueError):
        evaluate_policy([make_world("q1", ["d0"])], stop, "")
    with pytest.raises(TypeError):
        evaluate_policy([make_world("q1", ["d0"])], "stop", "E00")


def test_worlds_without_gold_are_rejected_before_any_episode():
    with pytest.raises(ValueError, match="gold"):
        evaluate_policy([make_world("q1", [])], stop, "E00")


def test_policy_sees_only_the_public_observation_and_is_not_called_after_stop():
    calls = []

    def policy(obs):
        calls.append(dict(obs))
        return STOP

    result = evaluate_policy([make_world("q1", ["d1"])], policy, "E00")
    assert len(calls) == 1  # a single decision: the evaluator never calls the policy again
    obs = calls[0]
    assert set(obs) == {"qid", "question", "seen_doc_ids", "seen_doc_texts", "history",
                        "optional_used", "remaining_optional_searches", "action_mask"}
    assert find_forbidden_keys(obs) == []
    assert obs["qid"] == "q1"
    assert result["per_qid"]["q1"]["actions"] == ["STOP"]


def test_actions_outside_the_schema_fail_closed():
    for bad in (3, -1, True, "SEARCH_BM25", 1.0, None):
        with pytest.raises(EnvContractError):
            evaluate_policy([make_world("q1", ["d0"])], lambda obs, b=bad: b, "BAD")


def test_a_policy_that_never_stops_fails_closed():
    # Never STOP; the third SEARCH is masked out by the budget, so the episode cannot
    # terminate legally and the evaluator must raise instead of scoring a fabricated zero.
    with pytest.raises(EnvContractError, match="masked-out"):
        evaluate_policy([make_world("q1", ["d1"])], lambda obs: SEARCH_BM25, "BAD")


# ----------------------------------------------------------------------------- latencies

def test_policy_and_env_replay_latency_are_measured_separately():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 2), "E01")
    record = result["per_qid"]["q1"]
    for key in ("policy_latency_ms", "env_replay_latency_ms"):
        assert isinstance(record[key], float)
        assert record[key] >= 0.0
    summary = result["summary"]
    assert summary["mean_policy_latency_ms"] >= 0.0
    assert summary["mean_env_replay_latency_ms"] >= 0.0
    assert summary["total_policy_latency_ms"] >= summary["mean_policy_latency_ms"] - 1e-9
    assert summary["latency_note"] == LATENCY_NOTE
    assert "not live" in LATENCY_NOTE.lower()


def test_report_is_json_serializable():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 1), "E01")
    round_tripped = json.loads(json.dumps(result))
    assert round_tripped["per_qid"]["q1"]["n_search"] == 1
    assert round_tripped["per_qid"]["q1"]["full_gold_success"] is True


# ------------------------------------------------------------------ private outcome + oracle

def test_private_outcome_is_cross_checked_on_the_normal_path():
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 1), "E01")
    assert result["per_qid"]["q1"]["env_outcome_checked"] is True


@pytest.mark.parametrize("tamper", [
    lambda outcome: {**outcome, "n_search": outcome["n_search"] + 1},
    lambda outcome: {**outcome, "seen_doc_ids": list(reversed(outcome["seen_doc_ids"]))},
])
def test_private_outcome_disagreement_raises_instead_of_scoring(monkeypatch, tamper):
    import src.pipeline.vixstory_eval_v2 as eval_v2

    class Tampered:
        """Delegates to the real env but corrupts the private bookkeeping."""

        def __init__(self, worlds):
            self._inner = VixStoryEnvV2(worlds)

        def reset(self, **kwargs):
            return self._inner.reset(**kwargs)

        def step(self, action):
            return self._inner.step(action)

        def private_outcome(self):
            return tamper(dict(self._inner.private_outcome()))

    monkeypatch.setattr(eval_v2, "VixStoryEnvV2", Tampered)
    with pytest.raises(EnvContractError, match="disagrees"):
        evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 1), "E01")


def test_reward_disagreement_raises_instead_of_scoring(monkeypatch):
    import src.pipeline.vixstory_eval_v2 as eval_v2

    class TamperedReward:
        """Delegates to the real env but pays a bonus on every step."""

        def __init__(self, worlds):
            self._inner = VixStoryEnvV2(worlds)

        def reset(self, **kwargs):
            return self._inner.reset(**kwargs)

        def step(self, action):
            obs, reward, terminated, truncated, info = self._inner.step(action)
            return obs, reward + 0.5, terminated, truncated, info

        def private_outcome(self):
            return self._inner.private_outcome()

    monkeypatch.setattr(eval_v2, "VixStoryEnvV2", TamperedReward)
    with pytest.raises(EnvContractError, match="disagrees"):
        evaluate_policy([make_world("q1", ["d1", "d2"])], fixed_tool("bm25", 1), "E01")


class StubTeacher:
    def __init__(self, world, *, oracle=0.9, reachable=True):
        self.world = world
        self._oracle = oracle
        self._reachable = reachable

    def oracle_return(self):
        return self._oracle

    def full_gold_reachable(self):
        return self._reachable


def test_teacher_factory_adds_oracle_return_and_reachability():
    worlds = [make_world("q1", ["d1"]), make_world("q2", ["d1", "d2"])]
    result = evaluate_policy(worlds, stop, "E00", teacher_factory=StubTeacher)
    assert result["per_qid"]["q1"]["oracle_return"] == pytest.approx(0.9)
    assert result["per_qid"]["q1"]["full_gold_reachable"] is True
    assert result["summary"]["oracle"]["mean_oracle_return"] == pytest.approx(0.9)
    assert result["summary"]["oracle"]["full_gold_reachable_rate"] == 1.0
    assert result["summary"]["oracle"]["full_gold_reachable_n"] == 2


def test_without_a_teacher_the_oracle_block_is_absent_not_fabricated():
    result = evaluate_policy([make_world("q1", ["d1"])], stop, "E00")
    assert result["summary"]["oracle"] is None
    assert "oracle_return" not in result["per_qid"]["q1"]


# ------------------------------------------------------------------------------ selection

def candidate(index, mean_return, mean_n_search):
    return {"index": index, "mean_retrieval_return": mean_return, "mean_n_search": mean_n_search}


def test_selection_rule_mean_return_then_search_then_index():
    assert select_best([candidate(0, 1.40, 1.0), candidate(1, 1.45, 2.0)])["index"] == 1
    # equal return -> fewer mean SEARCH
    assert select_best([candidate(0, 1.45, 2.0), candidate(1, 1.45, 1.0)])["index"] == 1
    # equal return and SEARCH -> earlier checkpoint/candidate index
    assert select_best([candidate(2, 1.45, 1.0), candidate(1, 1.45, 1.0)])["index"] == 1


def test_selection_tolerance_is_documented_for_return_ties():
    tight = candidate(0, 1.45, 2.0)
    drift = candidate(1, 1.45 + 5e-10, 1.0)  # within the documented 1e-9 tolerance: a tie
    assert select_best([tight, drift])["index"] == 1  # tie -> fewer SEARCH
    better = candidate(1, 1.45 + 1e-6, 2.0)  # a real difference beats fewer SEARCH
    assert select_best([candidate(0, 1.45, 1.0), better])["index"] == 1


def test_selection_rejects_incomplete_or_empty_candidates():
    with pytest.raises(ValueError):
        select_best([])
    with pytest.raises(ValueError, match="mean_n_search"):
        select_best([{"index": 0, "mean_retrieval_return": 1.0}])


def test_selection_between_k1_and_k2_follows_mean_retrieval_return():
    worlds = [make_world("q1", ["d1", "d2"]), make_world("q2", ["d1", "d2"])]
    k1 = evaluate_policy(worlds, fixed_tool("bm25", 1), "E01")
    k2 = evaluate_policy(worlds, fixed_tool("bm25", 2), "E01")
    # per QID: K=1 -> 1.45; K=2 -> 1.40 (the extra SEARCH buys no additional gold)
    assert k1["summary"]["mean_retrieval_return"] == pytest.approx(1.45)
    assert k2["summary"]["mean_retrieval_return"] == pytest.approx(1.40)
    winner = select_best([{**k1["summary"], "index": 1},
                          {**k2["summary"], "index": 2}])
    assert winner["index"] == 1


# ----------------------------------------------------------------------- paired differences

def test_paired_difference_on_matching_qid_sets():
    left = {"qa": {"gold_recall": 1.0}, "qb": {"gold_recall": 0.5}}
    right = {"qa": {"gold_recall": 0.5}, "qb": {"gold_recall": 0.0}}
    result = paired_difference(left, right, metric="gold_recall")
    assert result["metric"] == "gold_recall"
    assert result["n"] == 2
    assert result["qids"] == ["qa", "qb"]
    assert result["mean_difference"] == pytest.approx(0.5)
    assert result["evaluation_label"] == DEVELOPMENT_LABEL
    assert "not an independent test" in result["evaluation_label_note"]
    assert "single seed" in result["evaluation_label_note"].lower()


def test_paired_difference_requires_identical_qid_sets():
    with pytest.raises(ValueError, match="QID sets do not match"):
        paired_difference({"a": {"gold_recall": 1.0}}, {"b": {"gold_recall": 1.0}},
                          metric="gold_recall")


def test_paired_difference_rejects_duplicate_qids():
    left = {"a": {"gold_recall": 1.0}, "b": {"gold_recall": 1.0}}
    right = {"a": {"gold_recall": 1.0}, "b": {"gold_recall": 1.0}}
    with pytest.raises(ValueError, match="duplicate"):
        paired_difference(left, right, metric="gold_recall", qids=["a", "a", "b"])


def test_paired_difference_requires_the_metric_on_both_sides():
    with pytest.raises((KeyError, ValueError)):
        paired_difference({"a": {"gold_recall": 1.0}}, {"a": {"gold_recall": 1.0}},
                          metric="missing_metric")


def test_paired_difference_bootstrap_is_deterministic_seed_42():
    left = {f"q{i}": {"gold_recall": 1.0} for i in range(10)}
    right = {f"q{i}": {"gold_recall": 0.5} for i in range(10)}
    first = paired_difference(left, right, metric="gold_recall")
    second = paired_difference(left, right, metric="gold_recall")
    assert first["bootstrap"] == second["bootstrap"]
    assert first["bootstrap"]["seed"] == 42
    # constant per-QID difference: the bootstrap distribution is degenerate at 0.5
    assert first["bootstrap"]["ci_low"] == pytest.approx(0.5)
    assert first["bootstrap"]["ci_high"] == pytest.approx(0.5)


def test_paired_difference_bootstrap_is_optional():
    left = {"a": {"gold_recall": 1.0}}
    right = {"a": {"gold_recall": 0.5}}
    result = paired_difference(left, right, metric="gold_recall", bootstrap=False)
    assert result["bootstrap"] is None
    assert result["mean_difference"] == pytest.approx(0.5)


def test_paired_difference_over_two_real_arms_hand_computed():
    worlds = [make_world("q1", ["d1", "d2"]), make_world("q2", ["d0"])]
    e00 = evaluate_policy(worlds, stop, "E00")
    e01 = evaluate_policy(worlds, fixed_tool("bm25", 1), "E01")
    diff = paired_difference(e01["per_qid"], e00["per_qid"], metric="retrieval_return")
    # q1: 1.45 - 0.0 = +1.45; q2: 1.45 - 1.5 = -0.05; mean = 0.7
    assert diff["mean_difference"] == pytest.approx(0.7)
    assert diff["n"] == 2


# ------------------------------------------------------------------ learned-arm integration

def test_masked_greedy_arm_drives_the_evaluator_with_a_stub_prior():
    prior = lambda obs: [0.7, 0.2, 0.1]  # noqa: E731 — a fixed stub prior
    result = evaluate_policy([make_world("q1", ["d1", "d2"])], masked_greedy(prior), "E04")
    record = result["per_qid"]["q1"]
    assert record["actions"] == ["STOP"]
    assert record["seen_doc_ids"] == ["d0"]
    assert record["retrieval_return"] == 0.0


def test_masked_greedy_follows_the_mask_when_bm25_is_exhausted():
    prior = lambda obs: [0.1, 0.8, 0.1]  # noqa: E731 — a fixed stub prior
    result = evaluate_policy([make_world("q2", ["d1", "d2"])], masked_greedy(prior), "E04")
    record = result["per_qid"]["q2"]
    # second BM25 SEARCH spends the budget; the third is masked, so the arm falls back to
    # the next legal action by the masked argmax: STOP.
    assert record["actions"] == ["SEARCH_BM25", "SEARCH_BM25", "STOP"]
    assert record["seen_doc_ids"] == ["d0", "d1", "d2", "d4"]
    assert record["retrieval_return"] == pytest.approx(1.40)


# ------------------------------------------------------------------------ no reader/API

FORBIDDEN_IMPORT_PREFIXES = ("src.qa", "requests", "httpx", "openai", "anthropic",
                             "aiohttp", "urllib", "google")


def test_v2_modules_import_no_reader_or_api_client():
    for name in ("vixstory_eval_v2.py", "vixstory_arms_v2.py"):
        path = ROOT / "src" / "pipeline" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        bad = [m for m in modules
               if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_IMPORT_PREFIXES)]
        assert bad == [], f"{name} imports forbidden module(s) {bad}"

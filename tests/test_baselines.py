"""P7 tests: the baseline matrix rules, artifacts, guards and fairness invariants.

CPU-only and self-contained: no P1/P2/P3 cache, no Laya checkpoint, no reader, no GPU. Two
pieces of machinery stand in for the real ones — a synthetic ``WorldBundle`` built from
``tests/toy_world`` episodes, and a tiny synthetic B8 checkpoint (a constant-logit policy)
written into ``tmp_path``. What is under test here is the matrix logic: which rows exist and
why, what a policy may do, that cost accounting and features never depend on gold or on the
policy, and that test_strict cannot be read before the selection freeze.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from src.eval.correctness import CORRECTNESS_FEATURES, Calibrator, CorrectnessEstimator
from src.eval.policies import (
    DEFAULT_TAU, RL_POLICY_DIR, BootstrapThenAnswerPolicy, ConfidenceThresholdPolicy,
    FixedToolPolicy, LayaArgmaxPolicy, RandomValidPolicy, SearchAllPolicy, choose_tool,
    load_rl_policy, resolve_policy_id, short_code,
)
from src.eval.rollout import (
    EVAL_COLUMNS, FEATURE_COLUMNS, episode_row, run_episode, run_policy,
)
from src.pipeline import eval_baselines as eb
from src.pipeline import worlds
from src.pipeline.worlds import WorldBundle
from src.rl.env import AdaptiveQAEnv, EnvConfig
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import PolicyNet
from src.rl.reward import RewardCalculator, RewardConfig
from src.utils.config import load_config
from src.utils.io import atomic_write_json, read_json, read_jsonl, write_parquet, write_yaml
from tests.toy_world import negative_world, positive_world

VAL, TEST = "val_strict", "test_strict"
QIDS = ("val:pos1", "val:pos2", "val:neg1", "val:neg2")
GROUPS = {"val:pos1": "g1", "val:pos2": "g1", "val:neg1": "g2", "val:neg2": "g2"}


# ----------------------------------------------------------------------------- fixtures

def _cfg(root: Path) -> dict:
    """A self-contained config: every path inside ``root``, so no test writes to the repo."""
    cfg = load_config(None)
    cfg["paths"] = {key: str(root / key) for key in cfg["paths"]}
    cfg["data"] = {**cfg["data"], "profile": "smoke"}
    cfg["laya"] = {**cfg["laya"], "frozen": "none"}
    cfg["rl"] = {**cfg["rl"], "max_optional_search": 2, "seed": 42}
    cfg["retrieval"] = {**cfg["retrieval"], "initial_bootstrap": "bm25",
                        "bootstrap_passages": 1, "passage_per_search_action": 1}
    return cfg


def _worlds(split: str) -> list:
    """Two answerable (gold only on BM25 #2) and two unanswerable toy episodes."""
    return [
        replace(positive_world(), qid=QIDS[0], scenario_id="s_pos1", split=split),
        replace(positive_world(), qid=QIDS[1], scenario_id="s_pos2", split=split),
        replace(negative_world(), qid=QIDS[2], scenario_id="s_neg1", split=split),
        replace(negative_world(), qid=QIDS[3], scenario_id="s_neg2", split=split),
    ]


def _write_questions(cfg: dict, split: str) -> None:
    """The public question rows ``splitting.title_groups`` reads (qid + title_group)."""
    rows = [{"qid": qid, "question": f"Câu hỏi {qid}?", "title_group": GROUPS[qid]}
            for qid in QIDS]
    write_parquet(Path(cfg["paths"]["data_processed"]) / f"questions_{split}.parquet", rows)


def _fake_prior(obs: dict) -> np.ndarray:
    """Uniform Laya probabilities: the masked argmax then follows the action order."""
    return np.full(4, 0.25)


def _b5_estimator() -> CorrectnessEstimator:
    """A real calibrated B5 estimator with p_correct = 0.5 (zero coef, platt a=1, b=0)."""
    return CorrectnessEstimator(
        feature_names=CORRECTNESS_FEATURES,
        scaler=FeatureScaler(mean=[0.0] * len(CORRECTNESS_FEATURES),
                             std=[1.0] * len(CORRECTNESS_FEATURES)),
        coef=np.zeros(len(CORRECTNESS_FEATURES)), intercept=0.0,
        calibrator=Calibrator(mode="platt", params={"a": 1.0, "b": 0.0}), variant="B5")


def _save_rl_checkpoint(cfg: dict, *, regime: str) -> Path:
    """A B8 checkpoint whose greedy action is 'first legal' (ANSWER > BM25 > DENSE > ABSTAIN)."""
    net = PolicyNet(d_in=FeatureExtractor().dim, d_hidden=4)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
        net.policy_logits.bias.copy_(torch.tensor([3.0, 2.0, 1.0, 0.0]))
    directory = Path(cfg["paths"]["models"]) / RL_POLICY_DIR
    net.save(directory / "policy.safetensors")
    dim = FeatureExtractor().dim
    FeatureScaler(mean=[0.0] * dim, std=[1.0] * dim).save(directory / "scaler.json")
    atomic_write_json(directory / "config.json", {"laya": {"regime": regime}, "seed": 42})
    return directory


@pytest.fixture()
def p7(tmp_path, monkeypatch):
    """Config + synthetic worlds in place of the frozen caches; returns (cfg, state, root)."""
    cfg = _cfg(tmp_path)
    for split in (VAL, TEST):
        _write_questions(cfg, split)
    state: dict = {"ambiguous": frozenset({QIDS[3]}), "calls": []}

    def fake_build_worlds(cfg_arg, split, *, limit=None, laya_regime=None):
        state["calls"].append({"split": split, "limit": limit, "laya_regime": laya_regime})
        ws = _worlds(split)
        if limit is not None:
            ws = ws[:limit]
        return WorldBundle(split=split, worlds=ws,
                           questions={w.qid: w.question for w in ws},
                           docs={key: text for w in ws for key, text in w.corpus.items()},
                           laya_fn=None, ambiguous_qids=frozenset(state["ambiguous"]))

    monkeypatch.setattr(eb, "build_worlds", fake_build_worlds)
    monkeypatch.setattr(eb, "regime_prior_fn",
                        lambda cfg_arg, regime, ws, reward, config: _fake_prior)
    return cfg, state, tmp_path


def _overlay(root: Path, cfg: dict) -> Path:
    path = root / "overlay.yaml"
    write_yaml(path, cfg)
    return path


def _read_csv_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _env(world_list, *, max_optional: int) -> AdaptiveQAEnv:
    config = EnvConfig(max_optional_search=max_optional, bootstrap_tool="bm25",
                       bootstrap_passages=1, passage_per_search_action=1)
    return AdaptiveQAEnv(list(world_list), RewardCalculator(RewardConfig()), config)


# ----------------------------------------------------------------- policy unit tests

def _obs(mask, *, history=(), remaining=2):
    return {"action_mask": np.asarray(mask, dtype=bool), "history": tuple(history),
            "remaining_optional_searches": remaining, "qid": "q", "timestep": 0}


def test_b0_samples_only_legal_actions_and_is_seeded_per_episode():
    policy = RandomValidPolicy()
    policy.begin_episode(qid="val:pos1", seed=42)
    assert {policy.act(_obs([False, True, False, True])) for _ in range(40)} <= {1, 3}

    def sequence():
        other = RandomValidPolicy()
        other.begin_episode(qid="val:pos1", seed=42)
        return [other.act(_obs([True, True, True, True])) for _ in range(5)]

    assert sequence() == sequence()  # same (seed, qid) -> identical stream


def test_b1_bootstrap_then_decide_never_searches_and_reads_the_mask():
    policy = BootstrapThenAnswerPolicy()
    policy.begin_episode(qid="q", seed=42)
    assert policy.act(_obs([True, True, True, True])) == 0   # ANSWER
    assert policy.act(_obs([False, True, True, True])) == 3  # ABSTAIN, never SEARCH

    ws = _worlds(VAL)
    record = run_episode(_env(ws, max_optional=2), policy, ws[0], seed=42, run_id="t",
                         exp_id="E00")
    assert record.optional_used == 0 and record.terminal_action == "ABSTAIN"


def test_b2_b3_spend_the_budget_on_one_tool():
    ws = _worlds(VAL)
    env = _env(ws, max_optional=2)
    bm25 = run_policy(env, FixedToolPolicy(tool="bm25"), worlds=ws, seed=42, run_id="t",
                      exp_id="E01")
    dense = run_policy(env, FixedToolPolicy(tool="dense"), worlds=ws, seed=42, run_id="t",
                       exp_id="E02")
    assert all(r.bm25_calls == 3 and r.dense_calls == 0 and r.optional_used == 2 for r in bm25)
    assert all(r.dense_calls == 2 and r.bm25_calls == 1 and r.optional_used == 2 for r in dense)
    # same depth, different tool: the dense rows pay the higher per-tool price
    assert bm25[0].retrieval_cost == pytest.approx(3 * 0.10)
    assert dense[0].retrieval_cost == pytest.approx(0.10 + 2 * 0.25)


def test_b4_searches_both_tools_without_exceeding_the_budget():
    for budget in (1, 2):
        ws = _worlds(VAL)
        records = run_policy(_env(ws, max_optional=budget), SearchAllPolicy(), worlds=ws,
                             seed=42, run_id="t", exp_id="B4")
        assert all(r.optional_used == budget for r in records)
        # +1 counts the forced bootstrap, which is a BM25 call for every policy
        assert all(r.bm25_calls == 2 and r.dense_calls == budget - 1 for r in records)


class _StubEstimator:
    """The only surface B5 uses; ``p_correct`` is fixed so each branch is testable."""

    variant = "B5"
    is_calibrated = True

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def probability_from_obs(self, obs) -> float:
        return self.probability


def test_b5_three_branches_high_confidence_low_confidence_and_no_budget():
    confident = ConfidenceThresholdPolicy(_StubEstimator(0.9), tau=0.5)
    confident.begin_episode(qid="q", seed=42)
    assert confident.act(_obs([True, True, True, True], remaining=2)) == 0  # ANSWER
    assert confident.last_confidence == 0.9

    unsure = ConfidenceThresholdPolicy(_StubEstimator(0.1), tau=0.5)
    unsure.begin_episode(qid="q", seed=42)
    assert unsure.act(_obs([True, True, True, True], remaining=2)) == 1  # SEARCH_BM25
    assert unsure.act(_obs([False, True, True, True], remaining=1, history=(1,))) == 2  # DENSE
    assert unsure.act(_obs([False, False, False, True], remaining=0)) == 3  # ABSTAIN
    # an available candidate never overrides a sub-threshold p_correct
    assert unsure.act(_obs([True, False, False, True], remaining=0)) == 3


def test_b5_refuses_a_broken_probability():
    broken = ConfidenceThresholdPolicy(_StubEstimator(float("nan")), tau=0.5)
    broken.begin_episode(qid="q", seed=42)
    with pytest.raises(ValueError, match="not finite"):
        broken.act(_obs([True, True, True, True]))
    out_of_range = ConfidenceThresholdPolicy(_StubEstimator(1.5), tau=0.5)
    out_of_range.begin_episode(qid="q", seed=42)
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        out_of_range.act(_obs([True, True, True, True]))


def test_b5_and_b8_agree_on_costs_corpus_and_decisions_for_the_same_qid(tmp_path):
    """With both policies reduced to the same rule ('answer if legal, else BM25 once', at
    budget 1), the decisions, the retrieved corpus and the cost bookkeeping must match qid by
    qid — costs and worlds belong to the episode, not to the policy."""
    cfg = _cfg(tmp_path)
    cfg["rl"] = {**cfg["rl"], "max_optional_search": 1}
    _save_rl_checkpoint(cfg, regime="rlcd")
    b8 = load_rl_policy(cfg, laya_fn=_fake_prior, prior_regime="rlcd")
    b5 = ConfidenceThresholdPolicy(_b5_estimator(), tau=0.5)  # p_correct = 0.5 >= tau

    ws = _worlds(VAL)
    env = _env(ws, max_optional=1)
    by_qid_b5 = {r.qid: r for r in run_policy(env, b5, worlds=ws, seed=42, run_id="t",
                                              exp_id="E03")}
    by_qid_b8 = {r.qid: r for r in run_policy(env, b8, worlds=ws, seed=42, run_id="t",
                                              exp_id="E06")}
    assert set(by_qid_b5) == set(by_qid_b8) == set(QIDS)
    for qid, left in by_qid_b5.items():
        right = by_qid_b8[qid]
        assert [step.action for step in left.steps] == [step.action for step in right.steps]
        assert [step.candidate_doc_id for step in left.steps] == \
            [step.candidate_doc_id for step in right.steps]
        assert (left.bm25_calls, left.dense_calls) == (right.bm25_calls, right.dense_calls)
        assert left.retrieval_cost == right.retrieval_cost
        assert left.episode_return == right.episode_return
    assert by_qid_b5[QIDS[0]].retrieval_cost == pytest.approx(0.20)  # bootstrap + BM25


def test_tool_choice_helper_is_deterministic_and_documented():
    full = _obs([True, True, True, True])
    assert choose_tool(full) == 1                              # tie -> BM25
    assert choose_tool({**full, "history": (1,)}) == 2         # fewer uses -> DENSE
    assert choose_tool({**full, "history": (1, 2)}) == 1
    scored = {**full, "retrieval": {"bm25_top_score": 0.1, "dense_top_score": 0.9}}
    assert choose_tool(scored, mode="top_score") == 2
    assert choose_tool({**full, "action_mask": np.array([True, False, True, True])}) == 2
    with pytest.raises(ValueError, match="no legal search"):
        choose_tool({**full, "action_mask": np.array([True, False, False, True])})


def test_laya_argmax_masks_and_fails_loudly_on_a_missing_prior():
    policy = LayaArgmaxPolicy(_fake_prior, regime="ce")
    assert policy.id == "B6_laya_ce"
    policy.begin_episode(qid="q", seed=42)
    assert policy.act(_obs([False, True, True, True])) == 1  # uniform -> first legal
    assert policy.act(_obs([True, True, True, True])) == 0

    def missing(obs):
        raise KeyError("Laya prior for qid='q' was never pre-warmed")

    with pytest.raises(KeyError, match="never pre-warmed"):
        LayaArgmaxPolicy(missing, regime="rlcd").act(_obs([True, True, True, True]))


# ----------------------------------------------------------- artifacts and guards (CLI)

def test_cli_writes_the_whole_matrix_including_skipped_rows(p7):
    cfg, _, root = p7
    # The RLCD arm is complete and both prior directories exist, but the one B8 checkpoint
    # was trained on RLCD: E07 (CE + REINFORCE) is SKIPPED with the regime reason.
    _b5_estimator().save(Path(cfg["paths"]["models"]) / eb.CORRECTNESS_DIR)
    _save_rl_checkpoint(cfg, regime="rlcd")
    for regime in ("ce", "rlcd"):
        worlds.laya_checkpoint_dir(cfg, regime).mkdir(parents=True, exist_ok=True)
    assert eb.main(["--config", str(_overlay(root, cfg)), "--profile", "smoke", "--json"]) == 0

    table = Path(cfg["paths"]["tables"]) / "baseline_results_val.csv"
    rows = _read_csv_rows(table)
    assert [row["exp_id"] for row in rows] == [
        "E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "B0", "B0", "B0", "B4"]
    by_row = {row["exp_id"]: row for row in rows}
    assert [row["exp_id"] for row in rows if row["status"] != "SUCCEEDED"] == ["E07"]
    assert "laya.regime='rlcd'" in by_row["E07"]["reason"]
    assert "fabricated" in by_row["E07"]["reason"]
    assert by_row["E07"]["token_f1"] == "N/A" and by_row["E07"]["coverage"] == "N/A"
    assert by_row["E07"]["f1_ci_low"] == "N/A"  # no episodes -> no interval, never a fake 0

    # B1 (zero adaptation) never answers these four: coverage 0, selective risk N/A
    assert by_row["E00"]["coverage"] == "0.000000"
    assert by_row["E00"]["selective_risk"] == "N/A"
    assert by_row["E00"]["token_f1"] == "0.000000"
    assert by_row["E00"]["avg_bm25_calls"] == "1.000000"  # the forced bootstrap only
    assert by_row["E00"]["action_count_bm25"] == "0"      # no agent-chosen search at all

    # B5 answers exactly the two answerable worlds, each after one BM25 search
    assert by_row["E03"]["coverage"] == "0.500000"
    assert by_row["E03"]["selective_risk"] == "0.000000"
    assert by_row["E03"]["token_f1"] == "0.500000"
    assert by_row["E03"]["action_count_bm25"] == "4"
    assert by_row["E03"]["action_count_dense"] == "2"
    assert by_row["E03"]["action_count_answer"] == "2"
    assert by_row["E03"]["action_count_abstain"] == "2"
    assert by_row["E03"]["brier"] != "N/A" and by_row["E03"]["ece"] != "N/A"
    assert by_row["E03"]["calibration"] == "calibrated_p_correct"
    assert by_row["E03"]["tau"] == f"{DEFAULT_TAU:.6f}"  # fallback, flagged in the report
    assert by_row["E03"]["ci_basis"] == "cluster_title_group"
    assert by_row["E03"]["ci_groups"] == "2"

    # B4 exhausts the budget across both tools; nothing is answered here, and that is what an
    # upper-compute row is for: its cost is real, its coverage 0 -> selective risk N/A
    assert by_row["B4"]["avg_optional_retrievals"] == "2.000000"
    assert by_row["B4"]["avg_bm25_calls"] == "2.000000"
    assert by_row["B4"]["avg_dense_calls"] == "1.000000"
    assert by_row["B4"]["coverage"] == "0.000000" and by_row["B4"]["selective_risk"] == "N/A"

    # B8 carries no calibrated p_correct, so no calibration metric is invented for it
    assert by_row["E06"]["brier"] == "N/A" and by_row["E06"]["calibration"] == "none"

    base_seed = int(cfg["seed"])
    assert sorted(row["seed"] for row in rows if row["exp_id"] == "B0") == [
        str(base_seed + offset) for offset in range(3)]
    assert all(row["dev"] == "false" and row["split"] == VAL for row in rows)

    parquet = next(Path(cfg["paths"]["predictions"]).glob("*.parquet"))
    predictions_table = pq.read_table(str(parquet))
    predictions = predictions_table.to_pylist()
    assert set(predictions_table.column_names) == set(FEATURE_COLUMNS + EVAL_COLUMNS)
    assert {row["exp_id"] for row in predictions} == {
        "E00", "E01", "E02", "E03", "E04", "E05", "E06", "B0", "B4"}
    assert {row["split"] for row in predictions} == {VAL}
    assert not any(row["dev"] for row in predictions)

    trial_rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "trial_registry.jsonl"))
    assert len(trial_rows) == len(rows) == 12
    assert {row["status"] for row in trial_rows} == {"SUCCEEDED", "SKIPPED"}
    index_rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl"))
    assert [row["status"] for row in index_rows] == ["RUNNING", "SUCCEEDED"]
    assert index_rows[-1]["artifacts"][str(table)]["sha256"]

    report = read_json(Path(cfg["paths"]["reports"]) / "baseline_report_val.json")
    assert report["n_succeeded"] == 11 and report["n_not_succeeded"] == 1
    assert report["fairness"]["worlds"] == 4
    assert report["fairness"]["env_config"]["max_optional_search"] == 2
    assert report["selection_freeze"] is None
    assert any("fallback tau" in note for note in report["limitations"])
    assert report["retrieval_recall"]["n_answerable"] == 2
    for name in ("answer_f1_vs_retrieval_cost", "risk_coverage", "retrieval_recall"):
        assert (Path(cfg["paths"]["plots"]) / f"{name}.png").is_file()
    assert not (Path(cfg["paths"]["plots"]) / "reward_learning_curve.png").exists()


def test_unavailable_arms_are_skipped_with_a_reason_and_all_metrics_na(p7, capsys):
    cfg, _, root = p7
    overlay = str(_overlay(root, cfg))
    # a dry run plans the rows, honours --limit and writes no table at all
    assert eb.main(["--config", overlay, "--profile", "smoke", "--dry-run", "--limit", "2",
                    "--json"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["dry_run"] is True and planned["n_worlds"] == 2
    assert {row["exp_id"]: row["status"] for row in planned["rows"]}["E00"] == "RUN"
    assert {row["exp_id"]: row["status"] for row in planned["rows"]}["E03"] == "SKIPPED"
    assert not (Path(cfg["paths"]["tables"]) / "baseline_results_val.csv").exists()

    assert eb.main(["--config", overlay, "--profile", "smoke"]) == 0
    by_row = {row["exp_id"]: row for row in
              _read_csv_rows(Path(cfg["paths"]["tables"]) / "baseline_results_val.csv")}
    assert [by_row[name]["status"] for name in ("E03", "E04", "E05", "E06", "E07")] == \
        ["SKIPPED"] * 5
    assert "correctness_estimator" in by_row["E03"]["reason"]
    assert "train_correctness" in by_row["E03"]["reason"]
    # every prior-dependent arm names the missing frozen prior; the prior gate comes first
    assert "Laya prior 'ce'" in by_row["E04"]["reason"]
    assert "Laya prior 'rlcd'" in by_row["E05"]["reason"]
    assert "Laya prior 'rlcd'" in by_row["E06"]["reason"]
    assert "Laya prior 'ce'" in by_row["E07"]["reason"]
    for name in ("E03", "E04", "E05", "E06", "E07"):
        assert "N/A" not in by_row[name]["reason"]
        for column in ("token_f1", "mean_episode_reward", "avg_retrieval_cost", "brier",
                       "selective_risk", "ci_basis"):
            assert by_row[name][column] == "N/A", (name, column)


def test_skip_reason_gates_the_b8_checkpoint_and_the_prior_regime(p7):
    """Each gate of :func:`skip_reason` fires on its own condition, and none of them fabricates
    a row: missing prior -> missing checkpoint -> regime mismatch -> runnable."""
    cfg, _, _ = p7
    specs = {spec.exp_id: spec for spec in eb.ROW_SPECS}

    assert "Laya prior 'rlcd'" in eb.skip_reason(specs["E06"], cfg, eb.rl_checkpoint_status(cfg))
    for regime in ("ce", "rlcd"):
        worlds.laya_checkpoint_dir(cfg, regime).mkdir(parents=True, exist_ok=True)
    reason = eb.skip_reason(specs["E06"], cfg, eb.rl_checkpoint_status(cfg))
    assert "rl_policy" in reason and "train_rl" in reason

    _save_rl_checkpoint(cfg, regime="ce")  # a CE checkpoint is the E07 arm, not the E06 arm
    status = eb.rl_checkpoint_status(cfg)
    assert status["exists"] and status["regime"] == "ce"
    assert eb.skip_reason(specs["E07"], cfg, status) is None
    mismatch = eb.skip_reason(specs["E06"], cfg, status)
    assert "laya.regime='ce'" in mismatch and "needs 'rlcd'" in mismatch
    assert "fabricated" in mismatch


def test_a_broken_row_is_failed_without_losing_the_others(p7, monkeypatch):
    cfg, _, root = p7
    for regime in ("ce", "rlcd"):
        worlds.laya_checkpoint_dir(cfg, regime).mkdir(parents=True, exist_ok=True)

    def broken(cfg_arg, regime, ws, reward, config):
        if regime == "ce":  # the RLCD arm keeps working; only the CE pre-warm dies
            raise FileNotFoundError("Laya checkpoint for regime 'ce' is missing")
        return _fake_prior

    monkeypatch.setattr(eb, "regime_prior_fn", broken)
    assert eb.main(["--config", str(_overlay(root, cfg)), "--profile", "smoke",
                    "--policy", "B1,B6,B7"]) == 0
    by_row = {row["exp_id"]: row for row in
              _read_csv_rows(Path(cfg["paths"]["tables"]) / "baseline_results_val.csv")}
    assert by_row["E04"]["status"] == "FAILED"
    assert "FileNotFoundError" in by_row["E04"]["reason"]
    assert by_row["E04"]["mean_episode_reward"] == "N/A"
    assert by_row["E00"]["status"] == "SUCCEEDED" and by_row["E05"]["status"] == "SUCCEEDED"
    report = read_json(Path(cfg["paths"]["reports"]) / "baseline_report_val.json")
    assert report["n_succeeded"] == 2 and report["n_not_succeeded"] == 1
    trial_rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "trial_registry.jsonl"))
    assert {row["exp_id"]: row["status"] for row in trial_rows} == {
        "E00": "SUCCEEDED", "E04": "FAILED", "E05": "SUCCEEDED"}


def test_test_split_is_refused_until_the_selection_is_frozen(p7, capsys):
    cfg, _, root = p7
    overlay = _overlay(root, cfg)
    assert eb.main(["--config", str(overlay), "--profile", "smoke",
                    "--split", "test_strict", "--policy", "B1"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not (Path(cfg["paths"]["tables"]) / "baseline_results_test.csv").exists()

    # --dev bypasses the guard and stamps every produced row as dev
    assert eb.main(["--config", str(overlay), "--profile", "smoke", "--split", "test_strict",
                    "--policy", "B1", "--dev", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["dev"] is True
    rows = _read_csv_rows(Path(cfg["paths"]["tables"]) / "baseline_results_test.csv")
    assert [row["status"] for row in rows] == ["SUCCEEDED"]
    assert rows[0]["dev"] == "true" and rows[0]["split"] == TEST
    parquet = next(Path(cfg["paths"]["predictions"]).glob("*test_strict.parquet"))
    predictions = pq.read_table(str(parquet)).to_pylist()
    assert {row["dev"] for row in predictions} == {True}
    assert {row["split"] for row in predictions} == {TEST}
    report = read_json(Path(cfg["paths"]["reports"]) / "baseline_report_test.json")
    assert report["dev"] is True
    assert any("dev-stamped" in note for note in report["limitations"])

    # the freeze unlocks the same split, and the row is no longer dev
    atomic_write_json(eb.selection_freeze_path(cfg), {"selected": "E06", "hash": "x" * 64})
    assert eb.main(["--config", str(overlay), "--profile", "smoke", "--split", "test_strict",
                    "--policy", "B1"]) == 0
    assert _read_csv_rows(Path(cfg["paths"]["tables"]) / "baseline_results_test.csv")[0]["dev"] \
        == "false"


def test_features_never_move_when_gold_moves(p7):
    """Mutating the hidden side of an episode must leave every feature column identical and
    move only the eval columns — the structural form of 'no gold in the features'."""
    world = replace(positive_world(gold_in_second_result=False), qid="val:gold", split=VAL)
    env = _env([world], max_optional=2)
    policy = BootstrapThenAnswerPolicy()
    before = episode_row(run_episode(env, policy, world, seed=42, run_id="t", exp_id="E00"))
    world.private.answer_texts = ("1802",)
    after = episode_row(run_episode(env, policy, world, seed=42, run_id="t", exp_id="E00"))

    for column in FEATURE_COLUMNS:
        assert before[column] == after[column], column
    assert before["eval_correctness_f1"] == 1.0 and after["eval_correctness_f1"] == 0.0
    assert before["eval_terminal_utility"] == 2.0 and after["eval_terminal_utility"] == -2.0
    assert before["eval_correct"] == 1 and after["eval_correct"] == 0
    assert not set(FEATURE_COLUMNS) & set(EVAL_COLUMNS)
    assert not [column for column in FEATURE_COLUMNS
                if "gold" in column or "answer" in column or column.startswith("eval_")]


# ------------------------------------------------------------------- matrix bookkeeping

def test_row_specs_select_policies_and_tau_resolution(tmp_path):
    assert len(eb.select_specs(None)) == len(eb.ROW_SPECS) == 10
    assert [spec.exp_id for spec in eb.select_specs(["B8"])] == ["E06", "E07"]
    assert [spec.exp_id for spec in eb.select_specs(["B0"])] == ["B0"]
    assert [spec.exp_id for spec in eb.select_specs(["B1", "B4"])] == ["E00", "B4"]
    with pytest.raises(ValueError, match="unknown policy"):
        eb.select_specs(["B9"])
    assert resolve_policy_id("B5") == "B5_confidence_threshold"
    assert short_code("B7_laya_rlcd") == "B7"
    with pytest.raises(ValueError, match="unknown policy"):
        resolve_policy_id("E00")  # E ids select rows, they are not policies

    cfg = _cfg(tmp_path)
    assert eb.resolve_tau(cfg, 0.7) == (0.7, "--tau")
    assert eb.resolve_tau(cfg, None) == (DEFAULT_TAU, "default")
    directory = Path(cfg["paths"]["models"]) / eb.CORRECTNESS_DIR
    atomic_write_json(directory / eb.THRESHOLD_FILE, {"tau_answer": 0.42})
    assert eb.resolve_tau(cfg, None) == (0.42, str(directory / eb.THRESHOLD_FILE))

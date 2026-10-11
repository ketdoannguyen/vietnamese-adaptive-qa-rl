"""P6 selection/freeze tests: the pre-registered rule, eligibility, corruption, the
canonical swap, and the freeze-before-test contract (D016).

Archives are synthetic by default — selection reads recorded numbers and file hashes, it
never loads a policy — with one integration test that trains two real toy-world runs so the
swap is exercised against ``load_rl_policy``/``rl_checkpoint_status`` themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.decision.laya_adapter import checkpoint_model_hash
from src.eval.policies import load_rl_policy, rl_checkpoint_dir, rl_checkpoint_status
from src.pipeline import build_retrieval, select_rl, train_rl
from src.pipeline.eval_baselines import selection_freeze_path
from src.rl.reward import RewardConfig
from src.utils.config import resolve
from src.utils.hashing import sha256_file
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.runs import RunRegistry
from tests.test_rl_pipeline import _fake_checkpoint, p6  # noqa: F401 - fixture reuse

READER_HASHES = {"reader_model_hash": "r" * 64, "tokenizer_sha": "t" * 64,
                 "inference_config_sha": "i" * 64}
SPLITS = (train_rl.TRAIN_SPLIT, train_rl.VAL_SPLIT)


def _archive(cfg: dict, *, seed: int, val_reward: float, regime: str = "rlcd",
             profile: str = "smoke", run_id: str | None = None,
             registry: bool = True, registry_seed: int | None = None,
             registry_val_reward: float | None = None, registry_status: str = "SUCCEEDED",
             retrieval_sha: str | None = None, reader: dict | None = None,
             eval_episodes: int | None = None, incomplete: bool = False,
             validation: dict | None = None) -> Path:
    """A fully synthetic P6_rl archive (+ registry rows) with controllable fingerprints."""
    run_id = run_id or f"p6-rl-{profile}-{seed}-2026-10-09T0000{seed:02d}"
    archive = train_rl.archive_dir(cfg, run_id)
    (archive / "summary").mkdir(parents=True)
    config = {
        "stage": train_rl.STAGE, "run_id": run_id, "profile": profile, "seed": seed,
        "val_reward": val_reward, "val_std_reward": 0.01, "best_update": 10, "updates_ran": 12,
        "laya": {"regime": regime,
                 "model_hash": checkpoint_model_hash(_fake_checkpoint(cfg, regime)),
                 "checkpoint_ref": "/tmp/fake"},
        "reader": {split: dict(READER_HASHES) for split in SPLITS},
        "retrieval_config_sha": retrieval_sha or build_retrieval.retrieval_config_sha(dict(cfg)),
        "reward_spec_hash": RewardConfig.from_dict(dict(cfg)).spec_hash(),
        "rl_config": {"eval_episodes": 4 if eval_episodes is None else eval_episodes},
        "created": "2026-10-09T00:00:00+00:00",
    }
    if reader is not None:
        config["reader"] = reader
    if validation is not None:
        config["validation"] = validation
    write_json(archive / train_rl.CONFIG_FILE, config)
    for name, payload in (("policy.safetensors", f"policy-{seed}"),
                          ("scaler.json", "{}"), ("rl_config.json", "{}"),
                          (train_rl.PRIOR_REF_FILE, "/tmp/fake\n"),
                          (train_rl.TRAJECTORIES_FILE, "{}\n"),
                          (train_rl.TRAIN_LOG_FILE, "{}\n"),
                          (f"summary/{train_rl.SUMMARY_FILE}", "{}\n"),
                          (f"summary/{train_rl.CURVE_FILE}", "png"),
                          (f"summary/{train_rl.HISTOGRAM_FILE}", "png")):
        (archive / name).write_text(payload, encoding="utf-8")
    if incomplete:
        (archive / train_rl.TRAJECTORIES_FILE).unlink()
    if registry:
        reg = RunRegistry(resolve(cfg, "runs") / "index.jsonl")
        reg.append(run_id=run_id, stage=train_rl.STAGE, status="RUNNING", profile=profile)
        reg.append(run_id=run_id, stage=train_rl.STAGE, status=registry_status, profile=profile,
                   seed=seed if registry_seed is None else registry_seed,
                   val_reward=val_reward if registry_val_reward is None else registry_val_reward)
    return archive


def _select_rows(cfg: dict) -> list[dict]:
    path = resolve(cfg, "runs") / "index.jsonl"
    return [r for r in read_jsonl(path) if r.get("stage") == select_rl.STAGE] \
        if path.is_file() else []


def _freeze_or_none(cfg: dict) -> dict | None:
    path = selection_freeze_path(cfg)
    return read_json(path) if path.is_file() else None


# --------------------------------------------------------------------------- the rule

def test_freezes_the_highest_val_reward_candidate(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    for seed, reward in ((42, 1.0), (43, 2.5), (44, 0.5)):
        _archive(cfg, seed=seed, val_reward=reward)

    summary = select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")

    freeze = _freeze_or_none(cfg)
    assert freeze is not None and freeze["created_utc"] == "2026-10-09T05:00:00+00:00"
    assert freeze["winner"]["seed"] == 43 and freeze["selected_update"] == 10
    assert freeze["primary_metric"] == select_rl.PRIMARY_METRIC
    assert freeze["tie_break"] == select_rl.TIE_BREAK
    assert freeze["planned_seeds"] == [42, 43, 44]
    assert freeze["seeds_present"] == [42, 43, 44] and freeze["seeds_missing"] == []
    assert freeze["seed_summary"] == {"n": 3, "val_reward_mean": pytest.approx(4.0 / 3),
                                      "val_reward_min": 0.5, "val_reward_max": 2.5}
    assert freeze["excluded"] == [] and summary["run_id"] == freeze["run_id"]
    # The canonical policy is exactly the winner archive's file (hash recomputed from disk).
    canonical = rl_checkpoint_dir(cfg)
    assert freeze["canonical_files"]["policy.safetensors"]["sha256"] == \
        sha256_file(canonical / "policy.safetensors") == freeze["winner"]["policy_sha256"]
    assert [r["status"] for r in _select_rows(cfg)] == ["RUNNING", "SUCCEEDED"]


def test_tie_break_is_lowest_seed_regardless_of_scan_order(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=44, val_reward=1.0)  # created first: scan order must not decide
    _archive(cfg, seed=42, val_reward=1.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg)["winner"]["seed"] == 42


def test_selected_update_is_best_update_not_updates_ran(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    archive = _archive(cfg, seed=42, val_reward=1.0)
    config = read_json(archive / train_rl.CONFIG_FILE)
    assert (config["best_update"], config["updates_ran"]) == (10, 12)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["selected_update"] == 10 and freeze["winner"]["updates_ran"] == 12


# --------------------------------------------------------------------------- eligibility

def test_non_frozen_regime_archives_are_excluded_with_a_reason(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=9.9, regime="ce")  # E07 arm: not budgeted, not frozen
    _archive(cfg, seed=43, val_reward=1.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["winner"]["seed"] == 43
    assert freeze["seeds_present"] == [43]
    assert any("laya.regime" in item["reason"] and item["seed"] == 42
               for item in freeze["excluded"])


def test_selection_refuses_mixed_validation_protocols(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    fixed = {"mode": "fixed_qids", "n_tune": 2, "tune_qid_sha256": "a" * 64,
             "n_report": 1, "report_qid_sha256": "b" * 64,
             "independent_report": True}
    _archive(cfg, seed=42, val_reward=1.0, validation=fixed)
    _archive(cfg, seed=43, val_reward=9.0)
    with pytest.raises(select_rl.SelectionRefused, match="validation protocol"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None


def test_stale_chain_archives_are_excluded(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=9.9, retrieval_sha="d" * 64)
    stale_reader = {split: {**READER_HASHES, "reader_model_hash": "x" * 64} for split in SPLITS}
    _archive(cfg, seed=43, val_reward=9.9, reader=stale_reader)
    _archive(cfg, seed=44, val_reward=1.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["winner"]["seed"] == 44
    reasons = {item["seed"]: item["reason"] for item in freeze["excluded"]}
    assert "retrieval_config_sha" in reasons[42]
    assert "reader train_strict reader_model_hash" in reasons[43]


def test_stale_laya_checkpoint_is_excluded_before_freeze(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=9.9)
    checkpoint = _fake_checkpoint(cfg, "rlcd")
    (checkpoint / "model.safetensors").write_bytes(b"new weights")
    with pytest.raises(select_rl.SelectionRefused, match="no eligible.*laya.model_hash"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None


def test_other_profile_or_unregistered_archives_are_excluded(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=9.9, profile="pilot")  # registered, other profile
    _archive(cfg, seed=43, val_reward=9.9, registry=False)   # archive without a registry row
    _archive(cfg, seed=44, val_reward=1.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["winner"]["seed"] == 44
    reasons = {item["seed"]: item["reason"] for item in freeze["excluded"]}
    assert "profile" in reasons[42] and "0 terminal" in reasons[43]


# --------------------------------------------------------------------------- corruption

def test_registry_and_config_disagreement_fails_loudly_and_freezes_nothing(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0, registry_val_reward=2.0)
    with pytest.raises(RuntimeError, match="corruption"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert [r["status"] for r in _select_rows(cfg)] == ["RUNNING", "FAILED"]
    assert _freeze_or_none(cfg) is None
    assert not (rl_checkpoint_dir(cfg) / "policy.safetensors").exists()


def test_an_incomplete_archive_is_corruption_not_a_candidate(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0, incomplete=True)
    with pytest.raises(RuntimeError, match="incomplete"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None


# --------------------------------------------------------------------------- refusals

def test_no_eligible_candidates_is_refused_without_a_freeze_or_a_row(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=9.9, regime="ce")  # present but ineligible
    with pytest.raises(select_rl.SelectionRefused, match="no eligible"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None and _select_rows(cfg) == []


def test_a_running_training_row_blocks_selection(p6):
    cfg, _, _ = p6
    reg = RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    reg.append(run_id="p6-rl-smoke-42-live", stage=train_rl.STAGE, status="RUNNING",
               profile="smoke")
    with pytest.raises(select_rl.SelectionRefused, match="RUNNING"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None and _select_rows(cfg) == []


def test_refuses_to_refreeze_after_a_test_row_without_force(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0)
    reg = RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    reg.append(run_id="p7-baselines-smoke-test", stage="P7_baselines", status="SUCCEEDED",
               profile="smoke", split="test_strict")
    with pytest.raises(select_rl.SelectionRefused, match="test split was already evaluated"):
        select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert _freeze_or_none(cfg) is None

    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T06:00:00+00:00", force=True)
    assert _freeze_or_none(cfg)["forced"] is True
    assert [r["status"] for r in _select_rows(cfg)] == ["RUNNING", "SUCCEEDED"]


# --------------------------------------------------------------------------- contract

def test_missing_seeds_are_warned_and_recorded_never_hidden(p6, capsys):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0)
    _archive(cfg, seed=43, val_reward=2.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["seeds_present"] == [42, 43] and freeze["seeds_missing"] == [44]
    assert freeze["seed_summary"] == {"n": 2, "val_reward_mean": 1.5,
                                      "val_reward_min": 1.0, "val_reward_max": 2.0}
    printed = capsys.readouterr().out
    assert "WARNING" in printed and "44" in printed
    assert freeze["winner"]["seed"] == 43


def test_freeze_meets_the_report_stage_contract(p6):
    """report.py's access log reads ``created_utc`` (registry-ts format) and the report tests
    pin ``selected_update`` — both keys must exist for the freeze-before-test proof."""
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    freeze = _freeze_or_none(cfg)
    assert freeze["created_utc"].endswith("+00:00") and "T" in freeze["created_utc"]
    assert "selected_update" in freeze


def test_same_inputs_and_now_produce_byte_identical_freeze(p6):
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _archive(cfg, seed=42, val_reward=1.0)
    _archive(cfg, seed=43, val_reward=2.0)
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    first = selection_freeze_path(cfg).read_bytes()
    select_rl.run(cfg=cfg, profile="smoke", now="2026-10-09T05:00:00+00:00")
    assert selection_freeze_path(cfg).read_bytes() == first


# --------------------------------------------------------- integration: the real swap

def test_canonical_dir_equals_the_winner_archive_and_loads(p6):
    """Two real toy-world runs, the loser's reward lowered consistently, then select:
    the canonical dir must be byte-identical to the winner's archive (minus ``summary/``)
    and loadable by the evaluator's own loader."""
    cfg, _, _ = p6
    cfg["laya"]["frozen"] = "rlcd"
    _fake_checkpoint(cfg, "rlcd")
    cfg["rl"] = {**cfg["rl"], "max_updates": 2}
    first = train_rl.run(cfg=cfg, profile="smoke", seed=5)
    second = train_rl.run(cfg=cfg, profile="smoke", seed=6)

    # Make the selection outcome deterministic: rewrite seed 6's recorded reward (config and
    # its registry row together, so the archives stay self-consistent).
    loser = train_rl.archive_dir(cfg, second["run_id"])
    config = read_json(loser / train_rl.CONFIG_FILE)
    config["val_reward"] = -1.0
    write_json(loser / train_rl.CONFIG_FILE, config)
    reg_path = resolve(cfg, "runs") / "index.jsonl"
    rows = list(read_jsonl(reg_path))
    for row in rows:
        if row.get("run_id") == second["run_id"] and row.get("status") == "SUCCEEDED":
            row["val_reward"] = -1.0
    write_jsonl(reg_path, rows)

    summary = select_rl.run(cfg=cfg, profile="smoke")
    assert summary["winner"]["run_id"] == first["run_id"]  # seed 5 keeps its real reward

    canonical = rl_checkpoint_dir(cfg)
    winner_archive = train_rl.archive_dir(cfg, summary["winner"]["run_id"])

    def files(root: Path) -> set[str]:
        return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}

    assert files(canonical) == files(winner_archive) - {
        f"summary/{train_rl.SUMMARY_FILE}", f"summary/{train_rl.CURVE_FILE}",
        f"summary/{train_rl.HISTOGRAM_FILE}"}
    assert read_json(canonical / train_rl.CONFIG_FILE)["seed"] == 5
    freeze = _freeze_or_none(cfg)
    assert freeze["canonical_files"]["policy.safetensors"]["sha256"] == \
        freeze["winner"]["policy_sha256"] == sha256_file(canonical / "policy.safetensors")
    assert not (canonical.parent / select_rl.STAGING_DIR).exists()  # no staging left behind

    status = rl_checkpoint_status(cfg)
    assert status["exists"] and status["regime"] == "rlcd" and status["config_seed"] == 5
    policy = load_rl_policy(cfg, laya_fn=None, prior_regime="rlcd")
    assert policy.checkpoint == canonical
    # The winner's summary/plots returned to the shared fixed paths (the loser's were there).
    stored = read_json(resolve(cfg, "reports") / train_rl.SUMMARY_FILE)
    assert stored["run_id"] == summary["winner"]["run_id"]

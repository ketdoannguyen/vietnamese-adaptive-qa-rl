"""The pilot preflight: verifies a manifest against pinned data and estimates feasibility.

Everything here runs on synthetic inputs so no GPU, dataset download or real checkpoint is
touched, and the assertions pin the honest-reporting contract: measured and estimated are
kept apart, the manifest is verified (not trusted) against the local split IDs, and the test
split is never loaded.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.vixstory_pilot_preflight as preflight
from scripts.vixstory_pilot_preflight import (
    CORE_DEPS,
    EXPECTED_ROWS,
    MODEL_REPOS,
    MODEL_REVISIONS,
    PARAM_EFFICIENT_DEPS,
    SPLIT_SIZES,
    build_report,
    check_dependencies,
    estimate_runtime,
    estimate_states,
    extract_split_qids,
    go_no_go,
    load_snapshot,
    model_snapshot,
    next_free_path,
    run,
    state_tree_size,
    text_sample,
    verify_splits,
)
from src.utils.hashing import sha256_file


# --- synthetic fixtures ---------------------------------------------------
def _qa_row(qid: str) -> dict:
    return {"id": qid, "question": "Câu hỏi?", "answer": "a", "hop_level": "1",
            "question_type": "factual", "context_ids": ["c0001"]}


_SNAPSHOT_FILES = {"corpus": "corpus/corpus.parquet", "train": "data/train.parquet",
                   "validation": "data/validation.parquet"}


def make_snapshot(root: Path) -> tuple[Path, list[str], list[str]]:
    """A tiny stand-in for the pinned snapshot with the real row counts."""
    (root / "corpus").mkdir(parents=True, exist_ok=True)
    (root / "data").mkdir(parents=True, exist_ok=True)
    corpus = [{"context_id": f"c{i:04d}", "school_id": "S", "text": "x" * 40}
              for i in range(EXPECTED_ROWS["corpus"])]
    train_ids = [f"qa_{i:06d}" for i in range(EXPECTED_ROWS["train"])]
    val_ids = [f"qa_v{i:06d}" for i in range(EXPECTED_ROWS["validation"])]
    pq.write_table(pa.Table.from_pylist(corpus), root / "corpus/corpus.parquet")
    pq.write_table(pa.Table.from_pylist([_qa_row(q) for q in train_ids]), root / "data/train.parquet")
    pq.write_table(pa.Table.from_pylist([_qa_row(q) for q in val_ids]), root / "data/validation.parquet")
    return root, train_ids, val_ids


def pin_synthetic_hashes(monkeypatch, root: Path) -> None:
    """Pin EXPECTED_SHA256 to the synthetic snapshot's real digests so hash checks can pass."""
    monkeypatch.setattr(preflight, "EXPECTED_SHA256",
                        {role: sha256_file(root / rel) for role, rel in _SNAPSHOT_FILES.items()})


def make_manifest(path: Path, train_ids: list[str], val_ids: list[str]) -> Path:
    splits = {"pilot_300": train_ids[:300], "train_dev_200": train_ids[300:500],
              "validation_246": val_ids}
    manifest = {"run_id": "test-run", "source": {"resolved_revision": "rev1", "files": {}},
                "splits": {name: {"n": len(qids), "qids": qids} for name, qids in splits.items()}}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def make_cache(root: Path, *, broken: bool = False) -> Path:
    """A cache whose snapshots are named by the pinned revisions, so the pin check passes."""
    for repo in MODEL_REPOS:
        snap = root / ("models--" + repo.replace("/", "--")) / "snapshots" / MODEL_REVISIONS[repo]
        snap.mkdir(parents=True)
        target = snap / "model.safetensors"
        if broken:
            target.symlink_to(snap / "missing-blob")
        else:
            target.write_bytes(b"\x00" * 2048)
    return root


def fake_resources(*, vram_free_mib: int = 22000, mem_available_mib: int = 16000,
                   mem_total_mib: int = 16000, mem_reclaimable_mib: int = 2000,
                   disk_free_gb: float = 50.0, gpu_present: bool = True) -> dict:
    return {"gpu": {"present": gpu_present, "vram_total_mib": 23034, "vram_free_mib": vram_free_mib,
                    "gpus": [], "torch": {"available": gpu_present}},
            "system": {"mem_total_mib": mem_total_mib, "mem_available_mib": mem_available_mib,
                       "mem_free_mib": 5000, "mem_reclaimable_mib": mem_reclaimable_mib},
            "disk": {"free_gb": disk_free_gb, "paths": {}}}


def fake_dependencies(*, core_missing: tuple[str, ...] = ()) -> dict:
    core = {name: name not in core_missing for name in CORE_DEPS}
    efficient = {name: False for name in PARAM_EFFICIENT_DEPS}
    return {"core": core, "core_ok": all(core.values()), "param_efficient": efficient,
            "param_efficient_ok": all(efficient.values())}


def ok_models(*, laya_ok: bool = True) -> dict:
    """Model snapshots as ``build_report`` records them: resolved at the pinned revisions."""
    return {"convaiinnovations/laya-multilingual": {
        "weights_ok": laya_ok, "total_weight_bytes": 1024,
        "snapshot_revision": MODEL_REVISIONS["convaiinnovations/laya-multilingual"],
        "resolved_weights": [], "unresolved_weights": [], "cached": True,
        "snapshot_entries": []}}


def corrupt_revision(repo: str) -> dict:
    """Model snapshots where ``repo`` resolved a revision other than the pin."""
    models = ok_models()
    models[repo]["snapshot_revision"] = "deadbeef"
    return models


# --- manifest handling ----------------------------------------------------
def test_extract_split_qids_maps_pilot_record_names():
    manifest = {"splits": {"pilot_300": {"qids": ["a"]}, "train_dev_200": {"qids": ["b"]},
                           "validation_246": {"qids": ["c"]}}}
    assert extract_split_qids(manifest) == {"train": ["a"], "dev": ["b"], "validation": ["c"]}


def test_extract_split_qids_accepts_generic_names_and_bare_mapping():
    assert extract_split_qids({"qids": {"train": ["a"], "dev": ["b"], "validation": ["c"]}}) == \
        {"train": ["a"], "dev": ["b"], "validation": ["c"]}
    with pytest.raises(ValueError, match="neither a 'splits'"):
        extract_split_qids({})
    with pytest.raises(ValueError, match="no dev split"):
        extract_split_qids({"splits": {"train": ["a"], "validation": ["c"]}})
    with pytest.raises(ValueError, match="list of string QIDs"):
        extract_split_qids({"splits": {"train": [1], "dev": ["b"], "validation": ["c"]}})


def test_verify_splits_flags_missing_duplicates_size_and_overlap():
    good = verify_splits({"train": ["t1", "t2"], "dev": ["d1"], "validation": ["v1"]},
                         train_ids={"t1", "t2", "d1"}, validation_ids={"v1"},
                         expected_sizes={"train": 2, "dev": 1, "validation": 1})
    assert good["ok"] and good["all_disjoint"]

    missing = verify_splits({"train": ["t1", "zz"], "dev": ["d1"], "validation": ["v1"]},
                            train_ids={"t1", "t2", "d1"}, validation_ids={"v1"},
                            expected_sizes={"train": 2, "dev": 1, "validation": 1})
    assert missing["splits"]["train"]["missing_examples"] == ["zz"] and not missing["ok"]

    overlap = verify_splits({"train": ["t1"], "dev": ["t1"], "validation": ["v1"]},
                            train_ids={"t1"}, validation_ids={"v1"},
                            expected_sizes={"train": 1, "dev": 1, "validation": 1})
    assert overlap["overlaps"]["train_dev"] == 1 and not overlap["all_disjoint"]

    dup = verify_splits({"train": ["t1", "t1"], "dev": ["d1"], "validation": ["v1"]},
                        train_ids={"t1", "d1"}, validation_ids={"v1"},
                        expected_sizes={"train": 2, "dev": 1, "validation": 1})
    assert dup["splits"]["train"]["duplicates"] == 1 and not dup["ok"]


# --- dataset loading ------------------------------------------------------
def test_load_snapshot_reports_counts_and_never_loads_test(tmp_path):
    root, _train_ids, _val_ids = make_snapshot(tmp_path)
    (root / "data/test.parquet").write_bytes(b"not a parquet, must never be read")
    report, docs, train, val = load_snapshot(root)
    assert report["counts"] == EXPECTED_ROWS
    assert all(report["counts_ok"].values())
    assert report["test_split_present_locally"] is True
    assert report["test_loaded"] is False
    assert len(docs) == EXPECTED_ROWS["corpus"] and len(train) == EXPECTED_ROWS["train"]
    assert len(val) == EXPECTED_ROWS["validation"]


def test_load_snapshot_refuses_a_missing_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError, match="incomplete"):
        load_snapshot(tmp_path / "absent")


# --- estimates ------------------------------------------------------------
def test_state_tree_size_is_config_dependent_and_bounded_by_divergence():
    ids = [f"c{i}" for i in range(20)]
    assert state_tree_size(ids, max_optional_search=0) == 1
    assert state_tree_size(ids, max_optional_search=1) == 3    # 1 + 2 divergent branches
    assert state_tree_size(ids, max_optional_search=2) == 7    # 1 + 2 + 4
    with pytest.raises(ValueError, match="at least 8"):
        state_tree_size(ids[:3])


def test_estimate_states_multiplies_the_upper_bound_over_the_splits():
    result = estimate_states([f"c{i}" for i in range(20)],
                             split_sizes={"train": 3, "dev": 2, "validation": 1})
    assert result["states_per_question_upper_bound"] == 7
    assert result["total_states_upper_bound"] == 7 * 6
    assert "upper bound" in result["basis"]


def test_text_sample_is_order_independent_and_applies_the_char_ratio():
    docs = {"c1": {"text": "x" * 100}, "c2": {"text": "x" * 200}}
    rows = [{"id": f"q{i}", "question": "abc", "context_ids": ["c1", "c2"]} for i in range(5)]
    first = text_sample(rows, docs, sample_n=3, seed=7, chars_per_token=2.0)
    second = text_sample(list(reversed(rows)), docs, sample_n=3, seed=7, chars_per_token=2.0)
    assert first == second
    assert first["gold_passage_chars"] == {"mean": 150.0, "max": 200}
    # 1 bootstrap + 1 per search x 2 searches = 3 passages of mean 150 chars, plus a 3-char question.
    assert first["max_state_text_chars"]["mean"] == 453.0
    assert first["max_state_text_tokens_estimate"]["mean"] == 226.5
    with pytest.raises(ValueError, match="no rows"):
        text_sample([], docs, sample_n=3, seed=7)


def test_estimate_runtime_is_unknown_and_keeps_a_labelled_projection():
    result = estimate_runtime(split_sizes={"train": 300, "dev": 200, "validation": 246})
    assert result["status"] == "UNKNOWN"
    assert result["timeout_promise"] is False
    # The ViQuAD scaling is retained, but only as an explicitly unsound cross-dataset estimate.
    assert result["projection"]["stages"]["laya_ce"]["seconds_estimate"] == round(15890.7 * 300 / 5000, 1)
    assert result["projection"]["total_gpu_hours_estimate"] > 0
    assert "unsound cross-dataset estimate" in result["basis"]
    # The only VixSTORY runtime evidence is a generation count, and no wall-clock was recorded.
    assert result["measured_reader_generation"]["wall_clock_seconds"] is None
    assert result["measured_reader_generation"]["search_boot2_generations"] == 1707


# --- models and dependencies ---------------------------------------------
def test_model_snapshot_resolves_weights_and_reports_a_broken_symlink(tmp_path):
    good = model_snapshot(make_cache(tmp_path / "good"), MODEL_REPOS[0], ("model.safetensors",))
    assert good["weights_ok"] and good["total_weight_bytes"] == 2048

    broken_root = tmp_path / "broken"
    for repo in MODEL_REPOS:
        snap = broken_root / ("models--" + repo.replace("/", "--")) / "snapshots" / "rev1"
        snap.mkdir(parents=True)
        (snap / "model.safetensors").symlink_to(snap / "missing")
    broken = model_snapshot(broken_root, MODEL_REPOS[0], ("model.safetensors",))
    assert broken["weights_ok"] is False
    assert broken["unresolved_weights"] == ["model.safetensors"]


def test_check_dependencies_probes_without_importing_and_reports_absence():
    result = check_dependencies(find_spec=lambda name: object() if name == "torch" else None)
    assert result["core"]["torch"] is True and result["core"]["numpy"] is False
    assert result["core_ok"] is False
    assert result["param_efficient"] == {name: False for name in PARAM_EFFICIENT_DEPS}

    def raising(_name):
        raise ImportError("boom")

    assert check_dependencies(find_spec=raising)["core_ok"] is False


# --- go / no-go -----------------------------------------------------------
def _dataset(**overrides) -> dict:
    base = {"counts_ok": dict.fromkeys(EXPECTED_ROWS, True),
            "sha256_matches_pinned": dict.fromkeys(EXPECTED_ROWS, True)}
    base.update(overrides)
    return base


def _decide(**overrides) -> dict:
    kwargs = {"resources": fake_resources(), "models": ok_models(), "dependencies": fake_dependencies(),
              "verification": {"ok": True}, "dataset": _dataset(),
              "verification_available": True, "runtime": {"status": "UNKNOWN"}}
    kwargs.update(overrides)
    return go_no_go(**kwargs)


def test_arm_a_is_go_when_resources_models_deps_and_manifest_are_all_fine():
    decisions = _decide()
    arm = decisions["cloud_reader_laya_two_rl_checkpoints"]
    assert arm["decision"] == "GO"
    assert arm["ram"]["classification"] == "ok"


@pytest.mark.parametrize("overrides,blocker", [
    ({"models": ok_models(laya_ok=False)}, "weights_unresolved:convaiinnovations/laya-multilingual"),
    ({"models": corrupt_revision("convaiinnovations/laya-multilingual")},
     "weights_revision_mismatch:convaiinnovations/laya-multilingual"),
    ({"verification": {"ok": False}}, "manifest_verification_failed"),
    ({"verification_available": False}, "manifest_unavailable"),
    ({"dependencies": fake_dependencies(core_missing=("torch",))}, "dependency_missing:torch"),
    ({"resources": fake_resources(gpu_present=False)}, "no_gpu_visible"),
    ({"resources": fake_resources(vram_free_mib=512)}, "vram_free_below_12288_mib"),
    ({"resources": fake_resources(mem_total_mib=4096, mem_available_mib=1024)},
     "ram_total_below_8192_mib"),
    ({"resources": fake_resources(disk_free_gb=1.0)}, "disk_free_below_20.0_gb"),
    ({"dataset": _dataset(counts_ok={"corpus": False, "train": True, "validation": True})},
     "dataset_counts_mismatch"),
    ({"dataset": _dataset(sha256_matches_pinned={"corpus": True, "train": False, "validation": True})},
     "dataset_sha256_mismatch"),
])
def test_arm_a_blocks_on_each_missing_condition(overrides, blocker):
    decisions = _decide(**overrides)
    arm = decisions["cloud_reader_laya_two_rl_checkpoints"]
    assert arm["decision"] == "NO-GO" and blocker in arm["blockers"]


def test_arm_a_ram_below_target_needs_measurement_not_no_go():
    # This is the real bug: low *available* RAM (page-cache pressure) must not be a NO-GO.
    decisions = _decide(resources=fake_resources(mem_available_mib=4096))
    arm = decisions["cloud_reader_laya_two_rl_checkpoints"]
    assert arm["decision"] == "NEEDS_MEASUREMENT"
    assert arm["blockers"] == []
    assert "ram_available_below_8192_mib_measure_stage_budget" in arm["measurements_required"]
    assert arm["ram"]["classification"] == "transient_cache_pressure"
    assert arm["stage_memory_budget"]["measured"] is False


def test_arm_a_ram_total_below_target_is_a_hard_deficit():
    decisions = _decide(resources=fake_resources(mem_total_mib=4096, mem_available_mib=2048))
    arm = decisions["cloud_reader_laya_two_rl_checkpoints"]
    assert arm["decision"] == "NO-GO"
    assert "ram_total_below_8192_mib" in arm["blockers"]
    assert arm["ram"]["classification"] == "hard_deficit"


# --- end to end -----------------------------------------------------------
def test_run_writes_preflight_only_once_the_manifest_exists(tmp_path, monkeypatch):
    snapshot, train_ids, val_ids = make_snapshot(tmp_path / "snap")
    pin_synthetic_hashes(monkeypatch, snapshot)
    cache = make_cache(tmp_path / "cache")
    out = tmp_path / "preflight.json"
    common = {"out": out, "snapshot_dir": snapshot, "cache_dir": cache,
              "resources": fake_resources(), "dependencies": fake_dependencies()}

    missing = run(manifest_path=tmp_path / "no-manifest.json", **common)
    assert missing["manifest"]["present"] is False
    assert not out.exists()  # nothing written before the manifest exists

    report = run(manifest_path=make_manifest(tmp_path / "manifest.json", train_ids, val_ids), **common)
    assert out.exists()
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["artifact"] == "vixstory_pilot_preflight"
    assert written["official_test_loaded"] is False
    assert written["mode"] == "CPU_ONLY"
    assert set(written["estimated"]) == {"states", "text", "runtime"}
    assert written["estimated"]["runtime"]["status"] == "UNKNOWN"
    assert written["manifest"]["verification"]["ok"] is True
    assert written["measured"]["dataset"]["sha256_matches_pinned"] == dict.fromkeys(EXPECTED_ROWS, True)
    assert all(m["revision_matches_pin"] for m in written["measured"]["models"].values())
    assert written["go_no_go"]["cloud_reader_laya_two_rl_checkpoints"]["decision"] == "GO"
    assert report["written"] == str(out)


def test_run_never_overwrites_an_existing_preflight(tmp_path, monkeypatch):
    snapshot, train_ids, val_ids = make_snapshot(tmp_path / "snap")
    pin_synthetic_hashes(monkeypatch, snapshot)
    out = tmp_path / "preflight.json"
    common = {"out": out, "snapshot_dir": snapshot, "cache_dir": make_cache(tmp_path / "cache"),
              "resources": fake_resources(), "dependencies": fake_dependencies()}
    manifest = make_manifest(tmp_path / "manifest.json", train_ids, val_ids)

    assert run(manifest_path=manifest, **common)["written"] == str(out)
    second = run(manifest_path=manifest, **common)
    versioned = out.with_name("preflight.v2.json")
    assert second["written"] == str(versioned)
    assert versioned.exists() and json.loads(versioned.read_text(encoding="utf-8"))["artifact"] == \
        "vixstory_pilot_preflight"


def test_next_free_path_versions_but_does_not_overwrite(tmp_path):
    base = tmp_path / "preflight.json"
    assert next_free_path(base) == base
    base.write_text("{}", encoding="utf-8")
    assert next_free_path(base) == base.with_name("preflight.v2.json")


def test_build_report_verifies_the_manifest_against_local_rows(tmp_path):
    snapshot, train_ids, val_ids = make_snapshot(tmp_path / "snap")
    manifest = make_manifest(tmp_path / "manifest.json", train_ids, val_ids)
    # Corrupt the dev split with a QID that is not in the local train file.
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["splits"]["train_dev_200"]["qids"][0] = "qa_missing"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    report = build_report(manifest_path=manifest, snapshot_dir=snapshot, cache_dir=make_cache(tmp_path / "cache"),
                          resources=fake_resources(), dependencies=fake_dependencies())
    assert report["manifest"]["verification"]["splits"]["dev"]["missing_from_split"] == 1
    assert report["go_no_go"]["cloud_reader_laya_two_rl_checkpoints"]["decision"] == "NO-GO"


def test_report_never_serialises_a_secret_environment_value(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_value_123")
    snapshot, train_ids, val_ids = make_snapshot(tmp_path / "snap")
    report = build_report(manifest_path=make_manifest(tmp_path / "manifest.json", train_ids, val_ids),
                          snapshot_dir=snapshot, cache_dir=make_cache(tmp_path / "cache"),
                          resources=fake_resources(), dependencies=fake_dependencies())
    assert "hf_super_secret_value_123" not in json.dumps(report, ensure_ascii=False)
    assert set(SPLIT_SIZES) == {"train", "dev", "validation"}

"""Contract tests for the staged VixSTORY pilot (synthetic, CPU-only)."""

import json
from pathlib import Path

import pytest

from scripts.vixstory_pilot import (
    collect_teacher_inputs,
    guard_pilot_splits,
    require_complete_eval,
    write_teacher,
)
from src.pipeline.vixstory_worlds import assemble_split, reader_state_prompts
from scripts.vixstory_reader_eval import CacheIdentity, GenerationCache
from src.rl.vixstory_env import VixStoryEnvConfig


CORPUS = {"d0": "Sai", "d1": "Hà Nội", "d2": "Huế", "d3": "Khác"}
RANKINGS = {"q1": {"bm25": ["d0", "d1", "d2", "d3"],
                   "dense": ["d0", "d2", "d1", "d3"]}}
ROWS = {"q1": {"id": "q1", "question": "Thành phố nào?", "answer": "Hà Nội",
               "context_ids": ["d1"], "hop_level": 1, "question_type": "factual"}}


def _bundle(tmp_path):
    identity = CacheIdentity(model="ds/deepseek-flash", model_revision="provider-managed-alias",
                             base_url="https://api.ai-box.vn/v1", prompt_version="v1",
                             decoding="temperature=0", max_new_tokens=384,
                             corpus_sha256="corpus", rankings_sha256="ranks",
                             message_version="chat-user-v1")
    cache = GenerationCache(tmp_path / "reader.json", identity)
    for sp in reader_state_prompts(qids=["q1"], rows=ROWS, corpus=CORPUS,
                                   rankings=RANKINGS):
        cache.put(sp.prompt, json.dumps({"answer": "Hà Nội" if "d1" in sp.state else "",
                                         "citations": ["d1"] if "d1" in sp.state else []}))
    return assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS,
                          rankings=RANKINGS, cache=cache, config=VixStoryEnvConfig())


def test_teacher_inputs_use_only_train_worlds_and_cached_predictions(tmp_path):
    bundle = _bundle(tmp_path)
    data = collect_teacher_inputs(bundle, VixStoryEnvConfig())
    assert data["questions"] == {"q1": ROWS["q1"]["question"]}
    assert data["references"] == {"q1": "Hà Nội"}
    assert data["reader_answers"]["q1", ("d0", "d1")][0] == "Hà Nội"
    assert len(data["reader_answers"]) == bundle.n_states
    path = tmp_path / "teacher.jsonl"
    from transformers import AutoTokenizer

    tok_dir = (Path(__file__).resolve().parents[1] /
               "outputs/experiments/vixstory-pilot-300-2026-10-11/laya/E04_ce/tokenizer")
    if not tok_dir.is_dir():
        pytest.skip("local Laya tokenizer not available")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    result = write_teacher(bundle, VixStoryEnvConfig(), path, tokenizer=tokenizer)
    assert result["n_qids"] == 1 and result["n_states"] == bundle.n_states
    assert result["tokenizer_contract_sha256"] and result["state_text_version"] == "v2"
    from src.decision import vixstory_state_text
    from src.utils.hashing import sha256_file
    assert result["state_text_sha256"] == sha256_file(Path(vixstory_state_text.__file__))
    assert len(path.read_text().splitlines()) == bundle.n_states
    with pytest.raises(FileExistsError):
        write_teacher(bundle, VixStoryEnvConfig(), path, tokenizer=tokenizer)


def test_pilot_requires_disjoint_exact_split_counts():
    assert guard_pilot_splits({"train": ("t",), "dev": ("d",),
                               "validation": ("v",)}, expected=(1, 1, 1)) is None
    with pytest.raises(ValueError, match="overlap"):
        guard_pilot_splits({"train": ("t",), "dev": ("t",),
                            "validation": ("v",)}, expected=(1, 1, 1))
    with pytest.raises(ValueError, match="expected"):
        guard_pilot_splits({"train": ("t",), "dev": ("d",),
                            "validation": ("v",)}, expected=(300, 200, 246))


def test_train_only_pilot_splits_are_valid_until_validation_stage():
    splits = {"train": ("t",), "dev": ("d",)}
    assert guard_pilot_splits(splits, expected=(1, 1, 1), require_validation=False) is None
    with pytest.raises(ValueError, match="validation"):
        guard_pilot_splits(splits, expected=(1, 1, 1), require_validation=True)


def test_validation_cannot_be_recorded_with_missing_arms():
    with pytest.raises(RuntimeError, match="E07"):
        require_complete_eval({"success": False, "missing_mandatory": ["E07"]})
    require_complete_eval({"success": True, "missing_mandatory": []})


def _identity_fixture(tmp_path):
    from src.utils.hashing import sha256_json

    pilot = tmp_path / "pilot"
    (pilot / "E03").mkdir(parents=True)
    (pilot / "laya").mkdir()
    for arm in ("E04_ce", "E05_rlcd"):
        (pilot / "laya" / arm).mkdir()
        for name in ("model.safetensors", "rl_agent_config.json", "questions.json",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                     "encoder/config.json"):
            path = pilot / "laya" / arm / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        (pilot / "laya" / f"{arm}.manifest.json").write_text("{}")
    for seed in (42, 43):
        for arm in ("E06_rl_rlcd", "E07_rl_ce"):
            directory = pilot / "reinforce" / f"seed-{seed}" / arm
            (directory / "best").mkdir(parents=True)
            for name in ("policy.safetensors", "policy.safetensors.config.json", "scaler.json",
                         "rl_config.json"):
                (directory / "best" / name).write_text("{}")
            for name in ("summary.json", "rl_config.json"):
                (directory / name).write_text("{}")
    for name in ("manifest.json", "rankings.json", "cache.json", "corpus.parquet",
                 "teacher/train300.jsonl", "teacher/train300.provenance.json",
                 "calibration.json",
                 "reinforce.json", "E03/threshold_selection.json", "E03/calibrator_model.json"):
        path = pilot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    splits = {name: {"n": count, "qids": qids, "qids_sha256": sha256_json(qids)}
              for name, prefix, count in (("pilot_300", "t", 300),
                                          ("train_dev_200", "d", 200),
                                          ("validation_246", "v", 246))
              for qids in [[f"{prefix}{i:04d}" for i in range(count)]]}
    (pilot / "manifest.json").write_text(json.dumps({"splits": splits}))
    return {"pilot": pilot, "manifest": pilot / "manifest.json",
            "rankings": pilot / "rankings.json", "cache": pilot / "cache.json",
            "corpus": pilot / "corpus.parquet"}


def test_teacher_stage_does_not_open_validation_when_manifest_is_train_only(
        tmp_path, monkeypatch):
    from scripts import vixstory_pilot as mod
    from src.utils.hashing import sha256_json

    paths = _identity_fixture(tmp_path)
    splits = json.loads(paths["manifest"].read_text("utf-8"))
    del splits["splits"]["validation_246"]
    paths["manifest"].write_text(json.dumps(splits), encoding="utf-8")
    (paths["pilot"] / "teacher/train300.jsonl").unlink()
    (paths["pilot"] / "teacher/train300.provenance.json").unlink()
    monkeypatch.setattr(mod, "_bundle", lambda role, *_args:
                        pytest.fail("validation bundle opened") if role == "validation"
                        else pytest.fail("train bundle reached"))
    tok_dir = (Path(__file__).resolve().parents[1] /
               "outputs/experiments/vixstory-pilot-300-2026-10-11/laya/E04_ce/tokenizer")
    if not tok_dir.is_dir():
        pytest.skip("local Laya tokenizer not available")
    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer").symlink_to(tok_dir, target_is_directory=True)
    with pytest.raises(pytest.fail.Exception, match="train bundle reached"):
        mod.main(["teacher", "--pilot-dir", str(paths["pilot"]),
                  "--manifest", str(paths["manifest"]), "--laya-base", str(base)])


def test_train_only_dev_identity_does_not_require_validation_qids(tmp_path):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    manifest = json.loads(paths["manifest"].read_text("utf-8"))
    del manifest["splits"]["validation_246"]
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    identity = mod.evaluation_identity(paths, VixStoryEnvConfig())
    assert set(identity["qids_sha256"]) == {"train", "dev"}
    with pytest.raises(ValueError, match="validation"):
        mod.main(["validation-eval", "--pilot-dir", str(paths["pilot"]),
                  "--manifest", str(paths["manifest"]),
                  "--corpus", str(paths["corpus"]), "--rankings", str(paths["rankings"])])


def test_validation_identity_tracks_both_seed_checkpoints(tmp_path):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    original = mod.evaluation_identity(paths, VixStoryEnvConfig())
    for seed in (42, 43):
        path = paths["pilot"] / "reinforce" / f"seed-{seed}" / "E06_rl_rlcd" / "best" / "policy.safetensors"
        path.write_bytes(b"different weights")
        changed = mod.evaluation_identity(paths, VixStoryEnvConfig())
        assert changed != original
        path.write_text("{}")


def test_validation_identity_tracks_policy_and_evaluation_source(tmp_path, monkeypatch):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    identity = mod.evaluation_identity(paths, VixStoryEnvConfig())
    for name in ("src/pipeline/vixstory_arms.py", "src/pipeline/vixstory_eval.py",
                 "src/rl/vixstory_policy.py", "src/decision/laya_adapter.py"):
        assert name in identity["source_files"]
    original = mod.sha256_file
    monkeypatch.setattr(mod, "sha256_file", lambda path: "changed" if str(path).endswith(
        "src/pipeline/vixstory_arms.py") else original(path))
    assert mod.evaluation_identity(paths, VixStoryEnvConfig()) != identity


def test_validation_identity_rejects_mutated_artifact_before_qa_read(tmp_path):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    pilot = paths["pilot"]
    first = mod.evaluation_identity(paths, VixStoryEnvConfig())
    (pilot / "reinforce" / "seed-42" / "E07_rl_ce" / "best" / "scaler.json").write_text("changed")
    assert first != mod.evaluation_identity(paths, VixStoryEnvConfig())
    assert first["files"]["reinforce/seed-42/E07_rl_ce/best/scaler.json"] != \
        mod.evaluation_identity(paths, VixStoryEnvConfig())["files"][
            "reinforce/seed-42/E07_rl_ce/best/scaler.json"]


@pytest.mark.parametrize("changed", ["reinforce/seed-43/E07_rl_ce/best/scaler.json",
                                   "laya/E04_ce/model.safetensors",
                                   "E03/threshold_selection.json", "rankings.json",
                                   "teacher/train300.jsonl"])
def test_validation_gate_fails_before_bundle_reads_qa(tmp_path, monkeypatch, changed):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    pilot = paths["pilot"]
    identity = mod.evaluation_identity(paths, VixStoryEnvConfig())
    (pilot / "dev-eval.json").write_text(json.dumps({"success": True,
                                                    "missing_mandatory": [],
                                                    "seeds": {str(seed): {
                                                        "seed": seed, "success": True,
                                                        "missing_mandatory": [], "n_qids": 200,
                                                        "qids_sha256": "dev"}
                                                        for seed in (42, 43)},
                                                    "evaluation_identity": identity}))
    (pilot / changed).write_text("changed")
    monkeypatch.setattr(mod, "_bundle", lambda *_args: pytest.fail("QA was read"))
    with pytest.raises(ValueError, match="identity"):
        mod.main(["validation-eval", "--pilot-dir", str(pilot),
                  "--manifest", str(paths["manifest"]), "--rankings", str(paths["rankings"]),
                  "--cache", str(paths["cache"]), "--corpus", str(paths["corpus"])])


def test_reinforce_stages_two_matched_seed_pairs_without_early_stop(tmp_path, monkeypatch):
    import torch
    from scripts import vixstory_pilot as mod
    from src.pipeline import vixstory_arms
    from src.rl import vixstory_policy

    (tmp_path / "calibration.json").write_text("{}", encoding="utf-8")
    for arm in ("E04_ce", "E05_rlcd"):
        ckpt = tmp_path / "laya" / arm / "model.safetensors"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"weights")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mod, "_bundle", lambda role, *_: type(
        "Bundle", (), {"worlds": [type("World", (), {"qid": f"{role}-1"})()],
                       "qids": (f"{role}-1",)})())

    class Prior:
        def __init__(self, path, *, device):
            self.path = str(path)

        def close(self):
            pass

    monkeypatch.setattr(vixstory_arms, "LayaPrior", Prior)
    calls = []

    def fake_train(*positional, **kwargs):
        calls.append((positional[2], kwargs))
        return {arm: type("Result", (), {"to_dict": lambda self, arm=arm: {"arm": arm}})()
                for arm in ("rlcd", "ce")}

    monkeypatch.setattr(vixstory_policy, "train_vixstory_policy", fake_train)
    args = type("Args", (), {"max_updates": 100, "batch_episodes": 32,
                            "eval_every": 10, "patience": 5})()
    result = mod._reinforce({"pilot": tmp_path}, VixStoryEnvConfig(), args)
    assert set(result) == {"42", "43"}
    assert [call["arm_seeds"] for _, call in calls] == [
        {"rlcd": 42, "ce": 42}, {"rlcd": 43, "ce": 43}]
    assert all(call["fixed_updates"] is True for _, call in calls)
    assert [call["run_root"] for _, call in calls] == [
        tmp_path / "reinforce" / "seed-42", tmp_path / "reinforce" / "seed-43"]
    assert all(call.get("resume", False) is False for _, call in calls)
    assert all(call["train_qids"] == ("train-1",) and call["dev_qids"] == ("dev-1",)
               for _, call in calls)
    assert all(config.max_updates == 100 and config.eval_every == 10 and
               config.batch_episodes == 32 for config, _ in calls)


def test_policy_loader_selects_and_requires_actual_seed_checkpoints(tmp_path, monkeypatch):
    from scripts import vixstory_pilot as mod
    from src.pipeline import vixstory_arms, vixstory_calibration

    paths = _identity_fixture(tmp_path)
    pilot = paths["pilot"]
    identity = vixstory_calibration.artifact_identity(
        vixstory_calibration.reward_criterion(VixStoryEnvConfig()))
    (pilot / "calibration.json").write_text(json.dumps({"status": "SUCCEEDED",
                                                       "identity": identity}))
    (pilot / "E03/threshold_selection.json").write_text(json.dumps({
        "identity": identity, "tau": 0.2}))
    (pilot / "reinforce.json").write_text(json.dumps({
        str(seed): {arm: {"seed": seed, "n_updates": 100, "best_update": 10,
                         "stopped_early": False} for arm in ("ce", "rlcd")}
        for seed in (42, 43)}))
    monkeypatch.setattr(vixstory_calibration.VixStoryCorrectnessEstimator, "load",
                        lambda *args, **kwargs: object())
    selected = []

    class Prior:
        def __init__(self, path, *, device):
            self.path = path

        def close(self):
            pass

    monkeypatch.setattr(vixstory_arms, "LayaPrior", Prior)
    monkeypatch.setattr(vixstory_arms, "ReinforcePolicy", lambda path, prior: (
        selected.append((path, prior.path)) or object()))
    for seed in (42, 43):
        policies, priors = mod._policies(paths, VixStoryEnvConfig(), seed)
        assert policies["E06"] is not policies["E07"]
        for prior in priors:
            prior.close()
    assert [str(path.relative_to(pilot)) for path, _ in selected] == [
        f"reinforce/seed-{seed}/{arm}/best"
        for seed in (42, 43) for arm in ("E06_rl_rlcd", "E07_rl_ce")]
    report = json.loads((pilot / "reinforce.json").read_text())
    report["43"]["ce"]["n_updates"] = 99
    (pilot / "reinforce.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="fixed-update"):
        mod._policies(paths, VixStoryEnvConfig(), 42)


def test_dev_evaluates_both_actual_seed_checkpoints(tmp_path, monkeypatch):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    paths["pilot"].joinpath("reinforce.json").write_text(json.dumps({
        str(seed): {arm: {"seed": seed, "n_updates": 100, "best_update": 10,
                         "stopped_early": False} for arm in ("ce", "rlcd")}
        for seed in (42, 43)}))
    loaded = []

    class Prior:
        def close(self):
            pass

    monkeypatch.setattr(mod, "_policies", lambda _paths, _config, seed: (
        loaded.append(seed) or {"E06": seed, "E07": seed}, (Prior(),)))
    monkeypatch.setattr(mod, "_bundle", lambda role, *_: type("Bundle", (), {
        "worlds": [role], "qids": ("d1",), "identity": type("ID", (), {
            "as_dict": lambda self: {"split": role}})()})())
    from src.pipeline import vixstory_eval
    evaluation_seeds = []

    def evaluate(**kwargs):
        evaluation_seeds.append(kwargs["seed"])
        return {"seed": kwargs["seed"], "success": True, "missing_mandatory": [],
                "arms": kwargs["policies"]}

    monkeypatch.setattr(vixstory_eval, "run_vixstory_eval", evaluate)
    out = paths["pilot"] / "dev-eval.json"
    result = mod._eval("dev", paths, VixStoryEnvConfig(), out)
    assert loaded == evaluation_seeds == [42, 43]
    assert set(result["seeds"]) == {"42", "43"}
    assert result["seeds"]["42"]["arms"]["E06"] == 42
    assert result["seeds"]["43"]["arms"]["E07"] == 43
    assert result["evaluation_identity"] == mod.evaluation_identity(paths, VixStoryEnvConfig())
    assert json.loads(out.read_text()) == result


def test_paired_eval_rejects_mislabeled_seed_and_incomplete_coverage():
    from scripts.vixstory_pilot import require_paired_eval

    report = {"success": True, "missing_mandatory": [], "seeds": {
        "42": {"seed": 42, "success": True, "missing_mandatory": [], "n_qids": 200,
               "qids_sha256": "dev", "arms": {"E06": {"status": "SUCCEEDED"}}},
        "43": {"seed": 43, "success": True, "missing_mandatory": [], "n_qids": 200,
               "qids_sha256": "dev", "arms": {"E06": {"status": "SUCCEEDED"}}}}}
    require_paired_eval(report)
    report["seeds"]["43"]["seed"] = 42
    with pytest.raises(RuntimeError, match="mislabeled"):
        require_paired_eval(report)
    report["seeds"]["43"]["seed"] = 43
    report["seeds"]["43"]["n_qids"] = 199
    with pytest.raises(RuntimeError, match="coverage"):
        require_paired_eval(report)
    report["seeds"]["43"]["n_qids"] = 200
    report["seeds"]["42"]["n_qids"] = 199
    report["seeds"]["43"]["n_qids"] = 199
    with pytest.raises(RuntimeError, match="200"):
        require_paired_eval(report)


def test_validation_requires_each_seed_success_before_qa_read(tmp_path, monkeypatch):
    from scripts import vixstory_pilot as mod

    paths = _identity_fixture(tmp_path)
    (paths["pilot"] / "dev-eval.json").write_text(json.dumps({
        "success": True, "missing_mandatory": [],
        "evaluation_identity": mod.evaluation_identity(paths, VixStoryEnvConfig()),
        "seeds": {"42": {"seed": 42, "success": True, "missing_mandatory": []},
                  "43": {"seed": 43, "success": False, "missing_mandatory": ["E07"]}}}))
    monkeypatch.setattr(mod, "_bundle", lambda *_args: pytest.fail("QA was read"))
    with pytest.raises(RuntimeError, match="E07"):
        mod.main(["validation-eval", "--pilot-dir", str(paths["pilot"]),
                  "--rankings", str(paths["rankings"]), "--cache", str(paths["cache"]),
                  "--corpus", str(paths["corpus"]),
                  "--corpus", str(paths["corpus"]), "--rankings", str(paths["rankings"])])


def test_reinforce_scaler_can_move_to_training_device():
    import numpy as np
    import torch
    from src.rl.policy import TorchScaler

    scaler = TorchScaler(np.array([1.0], dtype=np.float32),
                         np.array([2.0], dtype=np.float32)).to("cpu")
    assert scaler.mean.device == torch.device("cpu")
    assert scaler(torch.tensor([[3.0]])).item() == pytest.approx(1.0)

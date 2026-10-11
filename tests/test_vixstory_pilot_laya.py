"""The VixSTORY pilot Laya stage: TRAIN300-only scope, pilot outputs, offline base.

``laya.train`` is replaced by a stub module (same ``finetune`` / ``TrainConfig`` signatures)
and the base checkpoint is a local directory, so these run on CPU with no torch work, no
tokenizer, no network and no GPU. What they pin down is the stage's own contract: it refuses
non-pilot data, refuses preexisting checkpoints, never reaches for the Hub, and writes the
pilot arm directories plus their manifests.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.vixstory_pilot_laya import (
    LOSS_ARMS,
    guard_teacher_scope,
    main,
    pilot_split_qids,
    read_teacher_rows,
    refuse_preexisting,
    run,
)
from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.schema import laya_record
from src.pipeline import train_laya
from src.utils.hashing import sha256_file

PILOT_QIDS = [f"train:{i:04d}" for i in range(3)]
VALIDATION_QIDS = ["val:0001", "val:0002"]


@dataclasses.dataclass
class _TrainConfig:
    """Mirror of ``laya.train.TrainConfig`` 0.4.0 field set, constructible with no args."""

    epochs: int = 4
    micro_batch: int = 8
    grad_accum: int = 8
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    loss: str = "rlcd"
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    w_sph: float = 0.75
    w_rps: float = 1.0
    shuffle_options: tuple = ()
    option_layout: object = None
    freeze_encoder: bool = False
    calib_max: int = 400
    calib_frac: float = 0.1
    calib_seed: int = 20260922
    seed: int = 0
    max_len: object = None
    head_max_len: object = None
    amp: object = None
    gradient_checkpointing: object = None
    log_every: int = 100
    label_smoothing: float = 0.0
    text_column: str = "text"
    label_column: str = "label"
    question_id: str = "label"
    instructions: object = None
    eval_data: object = None
    target_error: float = 0.10
    min_abstain_n: int = 10

    def validate(self) -> None:
        if self.loss not in ("soft-ce", "rlcd"):
            raise ValueError(f"loss must be one of soft-ce, rlcd, got {self.loss!r}")


class _StubTrainModule:
    """The ``laya.train`` seam: records calls and writes a fake fine-tuned checkpoint."""

    TrainConfig = _TrainConfig

    def __init__(self) -> None:
        self.finetune_calls: list[dict] = []

    def calibration_records(self, model, tok, items, device, *args, **kwargs):
        return []

    def finetune(self, *, data, model_dir, output_dir, config, device):
        self.finetune_calls.append({"data": data, "model_dir": model_dir,
                                    "output_dir": output_dir, "config": config,
                                    "device": device})
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"fine-tuned weights")
        return {"train_items": 3, "calibration_items": 1, "skipped": 0,
                "temperature": 0.9, "temperature_by_options": {"next_action": 0.9},
                "n_by_bucket": {"ANSWER": 1}, "abstention_thresholds": {"ANSWER": 0.5},
                "epoch_loss": [0.5, 0.4]}


def _cfg(**layer) -> dict:
    laya = {
        "model": "convaiinnovations/laya-multilingual", "revision": None, "frozen": "rlcd",
        "max_len": 768, "head_max_len": 192, "micro_batch": 1, "grad_accum": 8, "epochs": 3,
        "encoder_lr": 2.5e-5, "head_lr": 1e-4, "seed": 42, "freeze_encoder": False,
        "target_error": 0.1, "min_abstain_n": 10, "calib_max": 400, "calib_frac": 0.1,
        "calib_seed": 20260922, "device": "auto",
    }
    laya.update(layer)
    return {"laya": laya}


def _teacher_jsonl(path: Path, qids=PILOT_QIDS) -> Path:
    lines = []
    for qid in qids:
        row = laya_record(state=f"Câu hỏi {qid}", expected="ANSWER",
                          gold_distribution={"ANSWER": 0.7, "SEARCH_BM25": 0.1,
                                             "SEARCH_DENSE": 0.1, "ABSTAIN": 0.1})
        row["meta"] = {"qid": qid, "state_text_version": "v2"}
        lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pilot_manifest(path: Path, *, pilot=PILOT_QIDS, validation=VALIDATION_QIDS) -> Path:
    manifest = {"run_id": "vixstory-pilot-300-test",
                "splits": {"pilot_300": {"qids": list(pilot)},
                           "validation_246": {"qids": list(validation)},
                           "train_dev_200": {"qids": []}}}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _base_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"published weights")
    return path


def _prepare(tmp_path, monkeypatch):
    """Local base, TRAIN300 teacher, pilot manifest and the offline stub seams."""
    stub = _StubTrainModule()
    monkeypatch.setattr(train_laya, "load_train_module", lambda: stub)
    monkeypatch.setattr(train_laya, "normalize_checkpoint", lambda _out: None)
    monkeypatch.setattr(train_laya, "hub_sha",
                        lambda _m: pytest.fail("the Hub must not be queried"))
    return {
        "stub": stub,
        "base": _base_dir(tmp_path / "base_ckpt"),
        "data": _teacher_jsonl(tmp_path / "train300.jsonl"),
        "pilot_manifest": _pilot_manifest(tmp_path / "manifest.json"),
        "out_root": tmp_path / "laya",
    }


def _run(env, **overrides):
    kwargs = {"data": env["data"], "base": str(env["base"]),
              "pilot_manifest": env["pilot_manifest"], "out_root": env["out_root"],
              "loss": "both", "device": "cpu", "cfg": _cfg(), "train_module": env["stub"]}
    kwargs.update(overrides)
    return run(**kwargs)


# --- happy path --------------------------------------------------------------------------

def test_both_arms_train_into_pilot_dirs_and_write_hashed_manifests(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    results = _run(env, epochs=7, seed=123)

    assert set(results) == set(LOSS_ARMS.values())
    for arm, loss in (("E04_ce", "soft-ce"), ("E05_rlcd", "rlcd")):
        out = env["out_root"] / arm
        assert (out / "rl_agent_config.json").is_file()  # a P6-readable checkpoint
        assert results[arm]["out_dir"] == str(out)
        assert results[arm]["model_hash"] == checkpoint_model_hash(out)

        manifest = json.loads((env["out_root"] / f"{arm}.manifest.json").read_text("utf-8"))
        assert manifest["stage"] == "P5_laya_pilot" and manifest["status"] == "SUCCEEDED"
        assert manifest["arm"] == arm and manifest["loss"] == loss
        assert manifest["regime"] == train_laya.REGIMES[loss]
        assert manifest["epochs"] == 7 and manifest["seed"] == 123
        assert manifest["device"] == "cpu"
        assert manifest["data"]["sha256"] == sha256_file(env["data"])
        assert manifest["state_text_version"] == "v2"
        assert manifest["max_len"] == 768 and manifest["head_max_len"] == 192
        assert manifest["data"]["qids"] == len(PILOT_QIDS)
        assert manifest["scope_guard"] == {"validation_rows_in_train": 0,
                                           "qids_outside_pilot": 0}
        assert set(manifest["base"]["files"]) == {"rl_agent_config.json", "model.safetensors"}
        assert manifest["output"]["model_hash"] == results[arm]["model_hash"]
        assert manifest["calibration"]["calibration_items"] == 1

    # The stub saw the pilot paths, not any ViQuAD default.
    assert {(c["output_dir"], c["device"]) for c in env["stub"].finetune_calls} == {
        (str(env["out_root"] / "E04_ce"), "cpu"), (str(env["out_root"] / "E05_rlcd"), "cpu")}
    assert {c["model_dir"] for c in env["stub"].finetune_calls} == {str(env["base"])}
    assert {c["data"] for c in env["stub"].finetune_calls} == {str(env["data"])}


def test_single_arm_selection(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    results = _run(env, loss="rlcd")
    assert set(results) == {"E05_rlcd"}
    assert not (env["out_root"] / "E04_ce").exists()


def test_unknown_loss_is_refused(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="loss must be one of"):
        _run(env, loss="bogus")


# --- TRAIN300-only scope guard -----------------------------------------------------------

def test_refuses_a_validation_qid_in_the_teacher_file(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    env["data"] = _teacher_jsonl(tmp_path / "leaky.jsonl", [*PILOT_QIDS, VALIDATION_QIDS[0]])
    with pytest.raises(ValueError, match="validation QIDs"):
        _run(env)
    assert env["stub"].finetune_calls == []


def test_refuses_a_qid_outside_the_pilot(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    env["data"] = _teacher_jsonl(tmp_path / "extra.jsonl", [*PILOT_QIDS, "train:9999"])
    with pytest.raises(ValueError, match="outside pilot_300"):
        _run(env)


def test_guard_counts_clean_data():
    assert guard_teacher_scope({"a", "b"}, {"a", "b", "c"}, {"z"}) == {
        "validation_rows_in_train": 0, "qids_outside_pilot": 0}


def test_row_without_a_qid_is_refused(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"state": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no meta.qid"):
        read_teacher_rows(path)


def test_run_requires_teacher_provenance_before_training(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="teacher provenance"):
        _run(env, train_module=None, teacher_provenance=None)
    assert env["stub"].finetune_calls == []


def test_run_rejects_teacher_contract_mismatch_before_training(tmp_path, monkeypatch):
    from src.decision import vixstory_state_text
    from src.utils.hashing import sha256_json

    env = _prepare(tmp_path, monkeypatch)
    expected = {"sha256": sha256_file(env["data"]), "state_text_version": "v2",
                "state_text_sha256": sha256_file(vixstory_state_text.__file__),
                "max_len": 768, "head_max_len": 192,
                "tokenizer_contract_sha256": sha256_json({"tokenizer": "different"})}
    provenance = env["data"].with_suffix(".provenance.json")
    provenance.write_text(json.dumps(expected), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizer contract"):
        _run(env, teacher_provenance=provenance)
    assert env["stub"].finetune_calls == []


def test_synthetic_teacher_laya_inference_contract_with_local_tokenizer(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from src.decision.laya_adapter import LayaAdapter
    from src.decision import vixstory_state_text
    from src.decision.vixstory_state_text import serialize_vixstory_state
    from src.pipeline.vixstory_arms import LayaPrior
    from src.utils.hashing import sha256_json

    env = _prepare(tmp_path, monkeypatch)
    tok_dir = (Path(__file__).resolve().parents[1] /
               "outputs/experiments/vixstory-pilot-300-2026-10-11/laya/E04_ce/tokenizer")
    if not tok_dir.is_dir():
        pytest.skip("local Laya tokenizer not available")
    (env["base"] / "tokenizer").symlink_to(tok_dir, target_is_directory=True)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    observation = {"question": "Thành phố nào?", "seen_doc_ids": ("d1",),
                   "seen_doc_texts": ("Hà Nội là thủ đô.",),
                   "candidate": {"text": "Hà Nội", "cited_ids": ("d1",),
                                 "citation_valid": True},
                   "history": (), "optional_used": 0, "remaining_optional_searches": 2,
                   "action_mask": [True, True, True, True]}
    state = serialize_vixstory_state(observation, tokenizer=tokenizer)
    rows = [json.loads(line) for line in env["data"].read_text("utf-8").splitlines()]
    for row in rows:
        row["state"] = state
    env["data"].write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) +
                           "\n", encoding="utf-8")
    contract = json.loads(tokenizer.backend_tokenizer.to_str())
    contract.pop("truncation", None)
    provenance = {"sha256": sha256_file(env["data"]), "state_text_version": "v2",
                  "state_text_sha256": sha256_file(vixstory_state_text.__file__),
                  "max_len": 768, "head_max_len": 192,
                  "tokenizer_contract_sha256": sha256_json(contract)}
    provenance_path = env["data"].with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    result = _run(env, loss="soft-ce", teacher_provenance=provenance_path)
    checkpoint = Path(result["E04_ce"]["out_dir"])

    class FakeAdapter:
        _agent = type("Agent", (), {"tok": tokenizer, "cfg": {"max_len": 768,
                                                   "head_max_len": 192}})()

        def model_hash(self):
            return checkpoint_model_hash(checkpoint)

        def predict_probs(self, texts):
            assert texts == [state]
            return [[0.7, 0.1, 0.1, 0.1]]

        def close(self):
            pass

    monkeypatch.setattr(LayaAdapter, "load", lambda *args, **kwargs: FakeAdapter())
    prior = LayaPrior(checkpoint, device="cpu")
    try:
        assert list(prior(observation)) == [0.7, 0.1, 0.1, 0.1]
    finally:
        prior.close()


def test_real_laya_parser_uses_teacher_soft_targets_and_keeps_every_row(tmp_path):
    from transformers import AutoTokenizer
    from laya.train import items_from_rows

    tok_dir = (Path(__file__).resolve().parents[1] /
               "outputs/experiments/vixstory-pilot-300-2026-10-11/laya/E04_ce/tokenizer")
    if not tok_dir.is_dir():
        pytest.skip("local Laya tokenizer not available")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    path = _teacher_jsonl(tmp_path / "teacher.jsonl")
    rows = read_teacher_rows(path)
    items, skipped = items_from_rows(tokenizer, rows, 768, 192)
    assert len(items) == len(rows) and not skipped
    for item, row in zip(items, rows):
        assert item["target"] == pytest.approx([
            row["gold"]["next_action"]["probabilities"][action]
            for action in row["questions"]["next_action"]["criteria"]])


def test_v1_teacher_rows_cannot_be_trained_as_v2(tmp_path):
    path = _teacher_jsonl(tmp_path / "v2.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["meta"]["state_text_version"] = "v1"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not state text v2"):
        read_teacher_rows(path)


def test_empty_teacher_file_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_teacher_rows(path)


# --- preexisting output and missing inputs ----------------------------------------------

def test_refuses_a_preexisting_checkpoint(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    existing = env["out_root"] / "E04_ce"
    existing.mkdir(parents=True)
    (existing / "rl_agent_config.json").write_text('{"old": true}', encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _run(env)
    assert env["stub"].finetune_calls == []


def test_refuse_preexisting_allows_a_fresh_or_empty_directory(tmp_path):
    refuse_preexisting(tmp_path / "does-not-exist")  # no error
    empty = tmp_path / "empty"
    empty.mkdir()
    refuse_preexisting(empty)  # no error
    (empty / "model.safetensors").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        refuse_preexisting(empty)


def test_missing_teacher_file_is_refused(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="teacher JSONL not found"):
        _run(env, data=tmp_path / "nope.jsonl")


def test_missing_pilot_manifest_is_refused(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="pilot manifest not found"):
        _run(env, pilot_manifest=tmp_path / "nope.json")


# --- offline base ------------------------------------------------------------------------

def test_local_base_needs_no_hub(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(train_laya, "resolve_revision",
                        lambda *a, **k: pytest.fail("a local base needs no revision lookup"))
    results = _run(env, loss="soft-ce")
    manifest = json.loads(Path(results["E04_ce"]["manifest"]).read_text("utf-8"))
    assert manifest["base"]["downloaded"] is False
    assert manifest["base"]["revision_source"] == "local"
    assert manifest["base"]["revision"] is None


def test_uncached_hub_base_is_refused_without_network(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(train_laya, "hub_sha",
                        lambda _m: pytest.fail("no network when the cache is absent"))
    monkeypatch.setattr(train_laya, "materialize_checkpoint", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="not in the local Hugging Face cache"):
        _run(env, base="convaiinnovations/laya-multilingual", allow_download=False)


def test_allow_download_opts_into_the_hub(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    captured = {}

    def fake_materialize(model, revision, *, download=True, token=None):
        captured["download"] = download
        return {"dir": str(env["base"]), "downloaded": download, "revision": revision,
                "files": {}}

    monkeypatch.setattr(train_laya, "materialize_checkpoint", fake_materialize)
    monkeypatch.setattr(train_laya, "hub_sha", lambda _m: "a" * 40)
    _run(env, base="convaiinnovations/laya-multilingual", loss="rlcd", allow_download=True)
    assert captured["download"] is True


# --- dry run -----------------------------------------------------------------------------

def test_dry_run_trains_nothing_and_writes_no_checkpoint(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    results = _run(env, dry_run=True)
    assert env["stub"].finetune_calls == []
    for arm in LOSS_ARMS.values():
        assert results[arm]["dry_run"] is True and results[arm]["model_hash"] is None
        assert not (env["out_root"] / arm).exists()
        assert not (env["out_root"] / f"{arm}.manifest.json").exists()


# --- manifest reading / CLI --------------------------------------------------------------

def test_pilot_split_qids_refuses_a_manifest_without_splits(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no 'splits'"):
        pilot_split_qids(path)


def test_train_only_manifest_can_train_without_validation_qids(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    manifest = json.loads(env["pilot_manifest"].read_text("utf-8"))
    del manifest["splits"]["validation_246"]
    env["pilot_manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert pilot_split_qids(env["pilot_manifest"]) == (set(PILOT_QIDS), set())
    results = _run(env, loss="soft-ce")
    recorded = json.loads(Path(results["E04_ce"]["manifest"]).read_text("utf-8"))
    assert recorded["pilot_manifest"]["validation_246"] == 0


def test_cli_requires_explicit_inputs():
    with pytest.raises(SystemExit):  # argparse rejects a missing --data / --base / --out-root
        main(["--loss", "both"])


def test_cli_requires_teacher_provenance_before_stub_training(tmp_path, monkeypatch):
    env = _prepare(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        main(["--data", str(env["data"]), "--base", str(env["base"]),
              "--pilot-manifest", str(env["pilot_manifest"]),
              "--out-root", str(env["out_root"]), "--loss", "soft-ce", "--device", "cpu"])
    assert env["stub"].finetune_calls == []


# --- import hygiene ----------------------------------------------------------------------

def test_import_has_no_gpu_or_laya_side_effects():
    code = ("import sys, torch; import scripts.vixstory_pilot_laya; "
            "assert 'laya' not in sys.modules, 'laya.train must load lazily'; "
            "assert not torch.cuda.is_initialized(), 'no CUDA context at import'")
    import os

    from src.utils.config import REPO_ROOT

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True,
                          text=True, env=env)
    assert proc.returncode == 0, proc.stderr

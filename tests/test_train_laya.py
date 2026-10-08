"""The train_laya stage: base pinning, command equivalence, manifest, resume caveat.

The library is replaced by a stub module mirroring ``laya.train`` 0.4.0 (same ``TrainConfig``
fields and defaults, same keyword signatures), so these tests run with no torch, no
tokenizer, no checkpoint and no network. What they pin down is this stage's own contract:
the base is resolved and pinned, the recorded command is equivalent, the output lands where
P6 reads it, and the resume limitation is recorded rather than implied.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from src.decision.laya_adapter import checkpoint_model_hash
from src.decision.schema import ACTION_ORDER, laya_record
from src.pipeline import train_laya, worlds
from src.utils.io import read_json, read_jsonl, write_yaml
from tests import p5_fixtures as fx

PIN_SHA = "e" * 40
HUB_SHA = "f" * 40
MODEL = "convaiinnovations/laya-multilingual"


@dataclasses.dataclass
class _TrainConfig:
    """Mirror of ``laya.train.TrainConfig`` 0.4.0: same fields, same defaults, same checks."""

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
        for name in ("epochs", "micro_batch", "grad_accum"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be a positive integer")


class _StubTrainModule:
    """The ``laya.train`` seam; records calls and can write a fake fine-tuned checkpoint."""

    TrainConfig = _TrainConfig

    def __init__(self, *, write_checkpoint: bool = True) -> None:
        self.write_checkpoint = write_checkpoint
        self.dry_run_calls: list[dict] = []
        self.finetune_calls: list[dict] = []

    def dry_run(self, *, data, model_dir, config):
        self.dry_run_calls.append({"data": data, "model_dir": model_dir, "config": config})
        return {"rows_read": 2, "valid_items": 1, "skipped": 1, "max_len": config.max_len,
                "head_max_len": config.head_max_len}

    def calibration_records(self, model, tok, items, device, *args, **kwargs):
        """Mirrors ``laya.train`` so ``train_laya``'s device wrapper has a target to wrap."""
        return []

    def finetune(self, *, data, model_dir, output_dir, config, device):
        self.finetune_calls.append({"data": data, "model_dir": model_dir,
                                    "output_dir": output_dir, "config": config,
                                    "device": device})
        if self.write_checkpoint:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
            (out / "model.safetensors").write_bytes(b"fine-tuned weights")
        return {"train_items": 1, "calibration_items": 1, "skipped": 1,
                "temperature": 0.87, "temperature_by_options": {"next_action": 0.87},
                "n_by_bucket": {"ANSWER": 1}, "abstention_thresholds": {"ANSWER": 0.5},
                "epoch_loss": [0.42], "max_len": config.max_len}


def _config(tmp_path, *, layer=None):
    cfg = fx.p5_config(tmp_path)
    if layer:
        cfg["laya"] = {**cfg["laya"], **layer}
    return cfg


def _teacher_data(cfg, rows=2):
    path = Path(cfg["paths"]["data_processed"]) / "laya_train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(laya_record(
        state=f"Câu hỏi: fixture {index}",
        expected="ANSWER",
        gold_distribution={"ANSWER": 0.7, "SEARCH_BM25": 0.1, "SEARCH_DENSE": 0.1,
                           "ABSTAIN": 0.1})) for index in range(rows)) + "\n", encoding="utf-8")
    return path


def _checkpoint_dir(tmp_path, name="base_ckpt"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"published weights")
    return root


def _statuses(cfg):
    rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl"))
    return [(row["status"], row["stage"]) for row in rows]


# ------------------------------------------------------------------------ base resolution

def test_local_directory_needs_no_revision(tmp_path):
    local = _checkpoint_dir(tmp_path)
    assert train_laya.resolve_revision(str(local), None, pins={MODEL: PIN_SHA},
                                       hub_sha=_fail) == (None, "local")
    # An explicit revision is ignored for a local dir: there is nothing to check out.
    assert train_laya.resolve_revision(str(local), "v1", pins={}, hub_sha=_fail) == (None, "local")


def test_explicit_revision_wins_over_the_pin():
    assert train_laya.resolve_revision("some/repo", "abc123", pins={"some/repo": PIN_SHA},
                                       hub_sha=_fail) == ("abc123", "explicit")


def test_the_pin_is_read_case_insensitively():
    for key in (MODEL, MODEL.upper(), MODEL.lower()):
        assert train_laya.resolve_revision(MODEL, None, pins={key: PIN_SHA},
                                           hub_sha=_fail) == (PIN_SHA, "pinned")


def test_an_unpinned_repo_is_queried_and_the_sha_recorded():
    seen = []

    def fake_hub(model):
        seen.append(model)
        return HUB_SHA

    assert train_laya.resolve_revision("new/repo", None, pins={MODEL: PIN_SHA},
                                       hub_sha=fake_hub) == (HUB_SHA, "queried")
    assert seen == ["new/repo"]


def test_a_hub_without_a_sha_fails_loudly():
    def no_sha(model):
        raise RuntimeError("the Hub returned no commit SHA for 'x'")

    with pytest.raises(RuntimeError, match="no commit SHA"):
        train_laya.resolve_revision("x", None, pins={}, hub_sha=no_sha)


def test_load_pins_reads_the_library_table_at_run_time():
    from laya.revisions import PINNED_REVISIONS

    assert train_laya.load_pins() == dict(PINNED_REVISIONS)
    assert MODEL in PINNED_REVISIONS, "the reviewed table must still pin the base we use"


def test_materialize_checkpoint_uses_a_local_dir_as_is(tmp_path):
    local = _checkpoint_dir(tmp_path)
    info = train_laya.materialize_checkpoint(str(local), None)
    assert info["dir"] == str(local) and info["downloaded"] is False
    assert set(info["files"]) == {"rl_agent_config.json", "model.safetensors"}


def test_materialize_checkpoint_rejects_a_directory_that_is_not_a_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a Laya checkpoint"):
        train_laya.materialize_checkpoint(str(tmp_path), None)


def test_materialize_checkpoint_offline_returns_none_for_an_uncached_repo(monkeypatch):
    import huggingface_hub

    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    assert train_laya.materialize_checkpoint("org/model", PIN_SHA, download=False) is None


def test_materialize_checkpoint_with_download_propagates_a_fetch_failure(monkeypatch):
    import huggingface_hub

    def missing(*args, **kwargs):
        raise LocalEntryNotFoundError("offline")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", missing)
    with pytest.raises(LocalEntryNotFoundError):
        train_laya.materialize_checkpoint("org/model", PIN_SHA, download=True)


def test_materialize_checkpoint_hashes_a_downloaded_snapshot(tmp_path, monkeypatch):
    fetched = _checkpoint_dir(tmp_path, "fetched")
    calls = []

    def fake_download(model, **kwargs):
        calls.append((model, kwargs))
        return str(fetched)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    info = train_laya.materialize_checkpoint("org/model", PIN_SHA)
    assert calls == [("org/model", {"revision": PIN_SHA,
                                    "allow_patterns": list(train_laya.ALLOW_PATTERNS),
                                    "token": None, "local_files_only": False})]
    assert info["dir"] == str(fetched) and info["downloaded"] is True
    assert set(info["files"]) == {"rl_agent_config.json", "model.safetensors"}


# -------------------------------------------------------------------- config and command

def test_train_config_maps_every_configured_field():
    cfg = _config(Path("/nonexistent"))
    config = train_laya.train_config_from_cfg(cfg, loss="soft-ce",
                                              train_module=_StubTrainModule())
    laya = cfg["laya"]
    assert (config.epochs, config.micro_batch, config.grad_accum) == (
        laya["epochs"], laya["micro_batch"], laya["grad_accum"])
    assert (config.encoder_lr, config.head_lr) == (laya["encoder_lr"], laya["head_lr"])
    assert config.loss == "soft-ce" and config.seed == laya["seed"]
    assert (config.max_len, config.head_max_len) == (768, 192)
    assert config.freeze_encoder is False
    # The calibration split is not CLI-configurable; it must come from the config, or a
    # recorded command would not reproduce the run.
    assert (config.calib_max, config.calib_frac, config.calib_seed) == (
        laya["calib_max"], laya["calib_frac"], laya["calib_seed"])
    assert (config.target_error, config.min_abstain_n) == (laya["target_error"],
                                                           laya["min_abstain_n"])
    config.validate()


def test_overrides_replace_only_what_they_name():
    cfg = _config(Path("/nonexistent"), layer={"max_len": None})
    config = train_laya.train_config_from_cfg(cfg, loss="rlcd", train_module=_StubTrainModule(),
                                              epochs=1, micro_batch=1, grad_accum=2,
                                              max_len=64)
    assert (config.epochs, config.micro_batch, config.grad_accum, config.max_len) == (1, 1, 2, 64)
    assert config.head_max_len == cfg["laya"]["head_max_len"]


def test_config_diff_names_exactly_the_overrides():
    cfg = _config(Path("/nonexistent"))
    config = train_laya.train_config_from_cfg(cfg, loss="soft-ce",
                                              train_module=_StubTrainModule())
    assert train_laya.config_diff(config) == {"epochs": 3, "micro_batch": 1, "loss": "soft-ce",
                                              "seed": 42, "max_len": 768, "head_max_len": 192}


def test_the_cli_command_carries_every_recorded_value():
    cfg = _config(Path("/nonexistent"))
    config = train_laya.train_config_from_cfg(cfg, loss="rlcd", train_module=_StubTrainModule())
    command = train_laya.cli_command(config, "data/processed/laya_train.jsonl",
                                     "/base/dir", "outputs/models/laya_rlcd", "auto")
    assert command[:2] == ["laya-train", "--data"]
    joined = " ".join(command)
    for fragment in ("--loss rlcd", "--epochs 3", "--micro-batch 1", "--grad-accum 8",
                     "--seed 42", "--device auto", "--max-len 768", "--head-max-len 192",
                     "--target-error 0.1", "--min-abstain-n 10", "--base /base/dir"):
        assert fragment in joined, fragment
    assert "--freeze-encoder" not in command


# ------------------------------------------------------------------------------ the stage

def test_dry_run_resolves_base_and_command_without_training(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    data = _teacher_data(cfg)
    base = _checkpoint_dir(tmp_path, "base_ckpt")
    stub = _StubTrainModule()
    monkeypatch.setattr(train_laya, "load_train_module", lambda: stub)

    summary = train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", base=str(base),
                             dry_run=True)

    assert summary["dry_run"] is True and summary["regime"] == "ce"
    assert summary["base"]["revision_source"] == "local"
    assert summary["base"]["dir"] == str(base) and summary["base"]["downloaded"] is False
    assert summary["base"]["files"]["model.safetensors"] == \
        train_laya.sha256_file(base / "model.safetensors")
    assert summary["data"] == {"path": str(data), "sha256": train_laya.sha256_file(data)}
    assert "--loss soft-ce" in summary["command"]
    assert f"--data {data}" in summary["command"] and f"--base {base}" in summary["command"]
    assert summary["config_diff"]["loss"] == "soft-ce"
    assert summary["resume"]["supported"] is False and "epoch 0" in summary["resume"]["reason"]
    assert len(stub.dry_run_calls) == 1 and stub.dry_run_calls[0]["model_dir"] == str(base)
    assert stub.finetune_calls == []
    assert not Path(summary["output_dir"]).exists(), "a dry run must not create the output"
    assert _statuses(cfg) == [("RUNNING", "P5_laya_train"), ("SKIPPED", "P5_laya_train")]


def test_a_local_base_skips_the_hub_entirely(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _teacher_data(cfg)
    base = _checkpoint_dir(tmp_path, "base_ckpt")
    monkeypatch.setattr(train_laya, "load_train_module", lambda: _StubTrainModule())

    def fail(model):  # pragma: no cover - would run only on a regression
        raise AssertionError("the Hub must not be queried for a local base")

    monkeypatch.setattr(train_laya, "hub_sha", fail)
    summary = train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", base=str(base),
                             dry_run=True)
    assert summary["base"]["revision"] is None


def test_training_writes_the_checkpoint_p6_reads_and_a_full_manifest(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    data = _teacher_data(cfg)
    base = _checkpoint_dir(tmp_path, "base_ckpt")
    stub = _StubTrainModule()
    monkeypatch.setattr(train_laya, "load_train_module", lambda: stub)

    summary = train_laya.run(cfg=cfg, profile="smoke", loss="rlcd", base=str(base),
                             epochs=1, micro_batch=1)

    out = worlds.laya_checkpoint_dir(cfg, "rlcd")
    assert out == Path(summary["out_dir"]) and (out / "rl_agent_config.json").is_file()
    assert summary["model_hash"] == checkpoint_model_hash(out)
    assert summary["epochs"] == 1 and summary["temperature"] == 0.87
    assert summary["train_items"] == 1 and summary["calibration_items"] == 1
    assert summary["walltime_s"] >= 0.0 and "peak_vram_mib" in summary
    assert stub.finetune_calls[0]["device"] == "auto"
    assert stub.finetune_calls[0]["model_dir"] == str(base)

    manifest = read_json(summary["manifest"])
    assert manifest["status"] == "SUCCEEDED" and manifest["regime"] == "rlcd"
    assert manifest["epochs"] == 1 and manifest["seed"] == 42
    assert manifest["base"]["files"] and manifest["base"]["revision_source"] == "local"
    assert manifest["data"]["sha256"] == train_laya.sha256_file(data)
    assert manifest["output"]["model_hash"] == summary["model_hash"]
    assert manifest["equivalent_cli_command"] == summary["command"]
    assert shlex.join(shlex.split(summary["command"])) == summary["command"]
    assert manifest["calibration"]["temperature"] == 0.87
    assert manifest["calibration"]["calib_seed"] == cfg["laya"]["calib_seed"]
    assert manifest["epoch_loss"] == [0.42]
    assert manifest["resume"] == train_laya.RESUME
    assert isinstance(manifest["gpu"]["cuda"], bool)  # recorded whatever the host is
    assert "code_git_sha" in manifest and "library_versions" in manifest
    assert manifest["creation_script_sha"] == train_laya.sha256_file(train_laya.__file__)

    rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl"))
    assert [row["status"] for row in rows[-2:]] == ["RUNNING", "SUCCEEDED"]
    assert set(rows[-1]["artifacts"]) == {str(out), str(summary["manifest"])}
    assert rows[-1]["resume_supported"] is False


def test_an_existing_checkpoint_is_never_silently_replaced(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    _teacher_data(cfg)
    base = _checkpoint_dir(tmp_path, "base_ckpt")
    monkeypatch.setattr(train_laya, "load_train_module", lambda: _StubTrainModule())

    first = train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", base=str(base),
                           epochs=1, micro_batch=1)
    original = (Path(first["out_dir"]) / "model.safetensors").read_bytes()
    with pytest.raises(FileExistsError, match="cannot resume"):
        train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", base=str(base),
                       epochs=1, micro_batch=1)
    assert (Path(first["out_dir"]) / "model.safetensors").read_bytes() == original
    assert [status for status, _ in _statuses(cfg)][-2:] == ["RUNNING", "FAILED"]

    forced = train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", base=str(base),
                            epochs=1, micro_batch=1, force=True)
    assert (Path(forced["out_dir"]) / "model.safetensors").is_file()
    assert [status for status, _ in _statuses(cfg)][-2:] == ["RUNNING", "SUCCEEDED"]


def test_missing_teacher_data_is_an_error_not_an_empty_run(tmp_path, monkeypatch):
    cfg = _config(tmp_path)
    monkeypatch.setattr(train_laya, "load_train_module", lambda: _StubTrainModule())
    with pytest.raises(FileNotFoundError, match="make_teacher"):
        train_laya.run(cfg=cfg, profile="smoke", loss="soft-ce", dry_run=True)
    assert _statuses(cfg) == [("RUNNING", "P5_laya_train"), ("FAILED", "P5_laya_train")]
    assert not worlds.laya_checkpoint_dir(cfg, "ce").exists()


def test_an_unknown_loss_is_refused(tmp_path):
    with pytest.raises(ValueError, match="loss must be one of"):
        train_laya.run(cfg=_config(tmp_path), profile="smoke", loss="dpo")


def test_the_cli_dry_run_prints_the_resolved_plan(tmp_path, capsys, monkeypatch):
    cfg = _config(tmp_path)
    _teacher_data(cfg)
    base = _checkpoint_dir(tmp_path, "base_ckpt")
    monkeypatch.setattr(train_laya, "load_train_module", lambda: _StubTrainModule())
    overlay = tmp_path / "overlay.yaml"
    write_yaml(overlay, cfg)

    code = train_laya.main(["--config", str(overlay), "--profile", "smoke", "--loss", "soft-ce",
                            "--base", str(base), "--dry-run"])
    assert code == 0
    lines = capsys.readouterr().out
    assert "dry-run: loss=soft-ce" in lines and "local" in lines
    assert "--loss soft-ce" in lines and "config diff vs laya defaults" in lines
    assert "resume: not supported" not in lines  # that trailer is the training path's
    snapshot = Path(cfg["paths"]["manifests"]) / "P5_train_laya_ce_smoke_config.yaml"
    assert snapshot.is_file()
    assert not list(Path(cfg["paths"]["manifests"]).glob("P5_train_laya_*.json"))


# ------------------------------------------------- laya 0.4.0 CUDA calibration ordering

def test_the_calibration_patch_moves_the_model_first_and_wraps_once(monkeypatch):
    """The stage's workaround for ``laya/train.py:1086`` (calibration before ``train_model``
    moves the model at ``laya/train.py:638``): the wrapper must move the model before the
    original runs -- provable on cpu because the spy sees ``.to(device)`` already applied --
    and applying it twice must not double-wrap."""
    torch = pytest.importorskip("torch")
    import laya.train as laya_train

    seen = []

    class RecordingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.moves = []
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def to(self, *args, **kwargs):
            self.moves.append(args)
            return super().to(*args, **kwargs)

    def spy(model, tok, items, device, *args, **kwargs):
        seen.append({"moves": list(model.moves), "items": items, "extra": args})
        return ["record"]

    monkeypatch.setattr(laya_train, "calibration_records", spy)
    train_laya.patch_calibration_records(laya_train)

    device = torch.device("cpu")
    model = RecordingModule()
    assert laya_train.calibration_records(model, None, ["item"], device, 512, 192) == ["record"]
    assert seen[0]["moves"] == [(device,)], "the original must see a model already moved"
    assert seen[0]["items"] == ["item"] and seen[0]["extra"] == (512, 192)

    train_laya.patch_calibration_records(laya_train)  # second application must be a no-op
    model.moves.clear()
    seen.clear()
    assert laya_train.calibration_records(model, None, ["item"], device, 512, 192) == ["record"]
    assert seen[0]["moves"] == [(device,)], "a double wrap would move the model twice"


def _fail(*args, **kwargs):  # pragma: no cover - guards the "must not be called" paths
    raise AssertionError("this code path must not run in this test")

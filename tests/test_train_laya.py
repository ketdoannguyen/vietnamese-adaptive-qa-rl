"""The shared ``src.pipeline.train_laya`` helpers: base pinning, config/command, calibration.

``scripts/vixstory_pilot_laya.py`` wires these helpers into the VixSTORY pilot stage; these
tests pin down the helpers' own contract: the base checkpoint is resolved and pinned, the
``TrainConfig``/command mapping is exact, the retained named profiles resolve the same
effective Laya train config, and the calibration patch moves the model before the library's
first calibration pass. The library is replaced by a stub module mirroring ``laya.train``
0.4.0's ``TrainConfig`` (same fields, same defaults), so these run with no torch, no
tokenizer, no checkpoint and no network.
"""

from __future__ import annotations

import dataclasses

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from src.pipeline import train_laya
from src.utils.config import load_config

PIN_SHA = "e" * 40
HUB_SHA = "f" * 40
MODEL = "convaiinnovations/laya-multilingual"

#: The retained Laya knobs exactly as ``configs/base.yaml`` declares them; the profile parity
#: test below holds the real configs to this.
_LAYA = {
    "model": MODEL,
    "revision": None,
    "frozen": "rlcd",
    "max_len": 768,
    "head_max_len": 192,
    "micro_batch": 1,
    "grad_accum": 8,
    "epochs": 3,
    "encoder_lr": 2.5e-05,
    "head_lr": 0.0001,
    "seed": 42,
    "freeze_encoder": False,
    "target_error": 0.1,
    "min_abstain_n": 10,
    "calib_max": 400,
    "calib_frac": 0.1,
    "calib_seed": 20260922,
    "device": "auto",
}

#: Exactly the ``laya`` keys ``train_config_from_cfg`` reads into the effective train config.
_PROJECTION = ("epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr", "freeze_encoder",
               "seed", "max_len", "head_max_len", "target_error", "min_abstain_n", "calib_max",
               "calib_frac", "calib_seed")

#: The retained named profiles -> the ``data.profile`` each must resolve to.
_PROFILES = ((None, "full"), ("smoke", "smoke"), ("pilot", "pilot"), ("small", "small"),
             ("medium", "medium"), ("full", "full"))


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
    """The ``laya.train`` seam: the config builder reads only ``TrainConfig``."""

    TrainConfig = _TrainConfig


def _config(*, layer: dict | None = None) -> dict:
    """The minimal config ``train_config_from_cfg`` needs: the retained Laya knobs."""
    return {"data": {"profile": "smoke"}, "laya": {**_LAYA, **(layer or {})}}


def _checkpoint_dir(tmp_path, name="base_ckpt"):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "rl_agent_config.json").write_text('{"policy": "laya"}', encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"published weights")
    return root


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
    cfg = _config()
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
    cfg = _config(layer={"max_len": None})
    config = train_laya.train_config_from_cfg(cfg, loss="rlcd", train_module=_StubTrainModule(),
                                              epochs=1, micro_batch=1, grad_accum=2,
                                              max_len=64)
    assert (config.epochs, config.micro_batch, config.grad_accum, config.max_len) == (1, 1, 2, 64)
    assert config.head_max_len == cfg["laya"]["head_max_len"]


def test_config_diff_names_exactly_the_overrides():
    cfg = _config()
    config = train_laya.train_config_from_cfg(cfg, loss="soft-ce",
                                              train_module=_StubTrainModule())
    assert train_laya.config_diff(config) == {"epochs": 3, "micro_batch": 1, "loss": "soft-ce",
                                              "seed": 42, "max_len": 768, "head_max_len": 192}


def test_the_cli_command_carries_every_recorded_value():
    cfg = _config()
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


# ------------------------------------------------------------------ config parity profiles

def test_named_profiles_resolve_the_same_effective_laya_train_config():
    """Config parity gate: every retained profile resolves to the exact same effective Laya
    train config, so pruning ViQuAD-only config keys cannot silently move a knob."""
    for profile, data_profile in _PROFILES:
        cfg = load_config(profile)
        assert cfg["data"]["profile"] == data_profile
        assert cfg["laya"] == _LAYA
        config = train_laya.train_config_from_cfg(cfg, loss="soft-ce",
                                                  train_module=_StubTrainModule())
        assert {name: getattr(config, name) for name in _PROJECTION} == {
            name: _LAYA[name] for name in _PROJECTION}


# ------------------------------------------------- laya 0.4.0 CUDA calibration ordering

def test_the_calibration_patch_moves_the_model_first_and_wraps_once(monkeypatch):
    """The helper's workaround for ``laya/train.py:1086`` (calibration before ``train_model``
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

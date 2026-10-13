"""Laya fine-tune seams shared with the VixSTORY pilot stage.

These are the helper pieces ``scripts/vixstory_pilot_laya.py`` wires together to fine-tune
Laya on VixSTORY teacher data: base checkpoint materialization, revision pinning, the
``TrainConfig`` builder and its diff, the equivalent ``laya-train`` command, and the
calibration-ordering / tokenizer-normalization patches the library needs. The calling stage
owns its data, scope guards and outputs; this module holds no CLI and no stage defaults.

**Base checkpoint pinning.** ``laya-train`` resolves a Hub id with no revision argument, so a
moving ``main`` upstream would silently change the base under a finished run.
:func:`resolve_revision` resolves the revision itself (explicit revision, else the reviewed
SHA for the repo id read at run time from ``laya.revisions.PINNED_REVISIONS`` — never
hardcoded here — else the Hub commit SHA queried and recorded), and
:func:`materialize_checkpoint` fetches the snapshot into the local HF cache at exactly that
revision and returns its directory plus the SHA256 of its identity files.

**Calibration ordering.** ``laya.train.finetune`` calls ``calibration_records`` to score the
base checkpoint (``laya/train.py:1086``) before ``train_model``, the first place that runs
``model.to(device)`` (``laya/train.py:638``). :func:`patch_calibration_records` moves the
model onto the device before delegating, in-process only, and self-neutralizes once the
library moves the model itself. :func:`normalize_checkpoint` repairs
``tokenizer/tokenizer_config.json`` the way every later load will, so recorded hashes match
what later loads — and the probability cache — see.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Mapping

from src.utils.hashing import sha256_file

#: Loss -> regime name recorded next to a fine-tuned checkpoint.
REGIMES = {"soft-ce": "ce", "rlcd": "rlcd"}

#: Files ``snapshot_download`` needs for one Laya checkpoint; anything else is dead weight.
ALLOW_PATTERNS = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")

#: Base-checkpoint files whose SHA256 identifies it. ``tokenizer/tokenizer_config.json`` is
#: deliberately absent: the library repairs it in place on first load, so its digest is a
#: property of the local cache, not of the published checkpoint.
BASE_IDENTITY_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                       "tokenizer/tokenizer.json")


# ------------------------------------------------------------------------------ laya glue

def load_pins() -> dict[str, str]:
    """The reviewed-SHA table, read at run time so an upstream re-pin is picked up."""
    from laya.revisions import PINNED_REVISIONS

    return dict(PINNED_REVISIONS)


def hub_sha(model: str) -> str:
    """The commit SHA the Hub currently serves for ``model``."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(model)
    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(f"the Hub returned no commit SHA for {model!r}")
    return str(sha)


def load_train_module() -> Any:
    """The ``laya.train`` module; the single seam tests replace with a stub."""
    from laya import train

    return train


def patch_calibration_records(train: Any) -> None:
    """Move the model onto ``device`` before ``finetune``'s first calibration pass.

    ``laya.train.finetune`` calls ``calibration_records`` to score the base checkpoint
    (``laya/train.py:1086``) before ``train_model``, the first place that runs
    ``model.to(device)`` (``laya/train.py:638``). ``load_checkpoint`` returns the model on
    cpu, so on CUDA the calibration batch is moved to the device while the encoder is not:
    ``RuntimeError ... _CUDA__index_select``. Moving the model before delegating fixes this
    in-process only (site-packages is never edited) and self-neutralizes: once the library
    moves the model itself, ``model.to(device)`` is a no-op. Re-applying is a no-op too.
    """
    records = train.calibration_records
    if getattr(records, "_moves_model_first", False):
        return

    def _calibration_records_on_device(model, tok, items, device, *args, **kwargs):
        model.to(device)
        return records(model, tok, items, device, *args, **kwargs)

    _calibration_records_on_device._moves_model_first = True
    train.calibration_records = _calibration_records_on_device


def normalize_checkpoint(checkpoint_dir: str | Path) -> None:
    """Repair ``tokenizer/tokenizer_config.json`` the way every load will.

    ``save_checkpoint`` writes the tokenizer as-is; the library's first later load repairs
    that config in place (``laya.agent._fix_tokenizer_config``, called by ``laya.load`` and
    by ``laya.train``'s own ``load_checkpoint``/``dry_run``). Repairing here, before hashing,
    makes the recorded output hash and the recorded artifact checksums equal to what every
    later load sees, and stops the first load from mutating a checkpoint that has already
    been checksummed.
    """
    from laya.agent import _fix_tokenizer_config

    _fix_tokenizer_config(str(checkpoint_dir))


# ------------------------------------------------------------------------ base resolution

def resolve_revision(model: str, revision: str | None, *, pins: Mapping[str, str],
                     hub_sha: Any) -> tuple[str | None, str]:
    """``(revision, source)`` for the base checkpoint; ``None`` means "the Hub default".

    Precedence: a local directory needs no revision; an explicit revision wins; the reviewed
    pin for a Hub repo id (case-insensitive, over the table read at run time) comes next;
    otherwise the Hub commit SHA is queried and recorded, so a manifest identifies the exact
    base even before upstream review adds a pin.
    """
    if Path(model).expanduser().is_dir():
        return None, "local"
    if revision:
        return str(revision), "explicit"
    wanted = str(model).lower()
    for repo, sha in pins.items():
        if str(repo).lower() == wanted:
            return str(sha), "pinned"
    return str(hub_sha(model)), "queried"


def identity_shas(root: Path) -> dict[str, str]:
    """SHA256 of each :data:`BASE_IDENTITY_FILES` member that exists."""
    return {name: sha256_file(root / name) for name in BASE_IDENTITY_FILES
            if (root / name).is_file()}


def materialize_checkpoint(model: str, revision: str | None, *, download: bool = True,
                           token: str | None = None) -> dict[str, Any] | None:
    """A local directory holding the base checkpoint, plus its file digests.

    A local ``model`` directory is used as-is. Otherwise the snapshot is fetched at
    ``revision`` with the checkpoint allow-list; with ``download=False`` an un-cached model
    returns ``None`` rather than hitting the network.
    """
    root = Path(model).expanduser()
    if root.is_dir():
        if not (root / "rl_agent_config.json").is_file():
            raise FileNotFoundError(
                f"{root} is a directory but not a Laya checkpoint (no rl_agent_config.json)")
        return {"dir": str(root), "downloaded": False, "revision": revision,
                "files": identity_shas(root)}

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        resolved = snapshot_download(str(model), revision=revision,
                                     allow_patterns=list(ALLOW_PATTERNS), token=token,
                                     local_files_only=not download)
    except LocalEntryNotFoundError:
        if download:
            raise
        return None
    root = Path(resolved)
    if not (root / "rl_agent_config.json").is_file():
        raise FileNotFoundError(f"{root}: not a Laya checkpoint (no rl_agent_config.json)")
    return {"dir": str(root), "downloaded": True, "revision": revision,
            "files": identity_shas(root)}


# ------------------------------------------------------------------------- config, command

def _int_or_none(value: Any) -> int | None:
    return None if value in (None, "", "null") else int(value)


def train_config_from_cfg(cfg: Mapping[str, Any], *, loss: str, train_module: Any,
                          epochs: int | None = None, max_len: int | None = None,
                          micro_batch: int | None = None,
                          grad_accum: int | None = None) -> Any:
    """The library's ``TrainConfig`` from ``cfg["laya"]`` plus the CLI overrides."""
    laya = cfg["laya"]
    return train_module.TrainConfig(
        epochs=int(epochs if epochs is not None else laya["epochs"]),
        micro_batch=int(micro_batch if micro_batch is not None else laya["micro_batch"]),
        grad_accum=int(grad_accum if grad_accum is not None else laya["grad_accum"]),
        encoder_lr=float(laya["encoder_lr"]),
        head_lr=float(laya["head_lr"]),
        loss=loss,
        freeze_encoder=bool(laya["freeze_encoder"]),
        seed=int(laya["seed"]),
        max_len=int(max_len) if max_len is not None else _int_or_none(laya.get("max_len")),
        head_max_len=_int_or_none(laya.get("head_max_len")),
        target_error=float(laya["target_error"]),
        min_abstain_n=int(laya["min_abstain_n"]),
        calib_max=int(laya["calib_max"]),
        calib_frac=float(laya["calib_frac"]),
        calib_seed=int(laya["calib_seed"]),
    )


def config_dict(config: Any) -> dict[str, Any]:
    return dataclasses.asdict(config)


def config_diff(config: Any) -> dict[str, Any]:
    """Fields this run sets away from the library's own defaults, for the dry run."""
    defaults = type(config)()
    return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)
            if getattr(config, field.name) != getattr(defaults, field.name)}


def cli_command(config: Any, data: str | Path, base: str, out: str | Path,
                device: str) -> list[str]:
    """The equivalent ``laya-train`` argv.

    The calibration split (``calib_max`` / ``calib_frac`` / ``calib_seed``) and the RLCD
    sampler knobs are ``TrainConfig``-only, so this command is only exactly equivalent while
    those stay at the configured values — which the run manifest records.
    """
    args = ["laya-train", "--data", str(data), "--base", str(base), "--out", str(out),
            "--loss", str(config.loss), "--epochs", str(config.epochs),
            "--micro-batch", str(config.micro_batch), "--grad-accum", str(config.grad_accum),
            "--encoder-lr", repr(config.encoder_lr), "--head-lr", repr(config.head_lr),
            "--seed", str(config.seed), "--device", str(device),
            "--target-error", repr(config.target_error),
            "--min-abstain-n", str(config.min_abstain_n)]
    if config.max_len is not None:
        args += ["--max-len", str(config.max_len)]
    if config.head_max_len is not None:
        args += ["--head-max-len", str(config.head_max_len)]
    if config.freeze_encoder:
        args.append("--freeze-encoder")
    return args

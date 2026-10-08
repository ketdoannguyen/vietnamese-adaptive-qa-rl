"""P5 CLI: fine-tune Laya on the teacher data with soft-CE or RLCD (`02_IMPL/04`, spec 06 §1).

    python -m src.pipeline.train_laya --config configs/pilot.yaml --loss soft-ce
    python -m src.pipeline.train_laya --config configs/pilot.yaml --loss rlcd
    python -m src.pipeline.train_laya --config configs/pilot.yaml --loss soft-ce --dry-run

Both losses use the same data and the same config; the only difference is the objective, so
the CE/RLCD pair is a controlled comparison. Outputs land in the directories the P6 rollouts
read: ``outputs/models/laya_ce`` and ``outputs/models/laya_rlcd``
(:data:`src.pipeline.worlds.LAYA_CHECKPOINT_DIRS` — one definition, both sides).

**Base checkpoint pinning.** ``laya-train`` resolves a Hub id with no revision argument, so a
moving ``main`` upstream would silently change the base under a finished run. This stage
resolves the revision itself (:func:`resolve_revision`: explicit config revision, else the
reviewed SHA for the repo id read at run time from ``laya.revisions.PINNED_REVISIONS`` —
never hardcoded here — else the Hub commit SHA queried and recorded), materializes the
snapshot into the local HF cache at exactly that revision
(:func:`materialize_checkpoint`), and passes the local directory to the library. The run
manifest records the revision, how it was chosen, and the SHA256 of the base's identity files.

**Training is in-process** (``laya.train.finetune``), so the manifest also carries the
library's own returned summary — items trained, calibration counts, fitted temperature,
abstention thresholds, per-epoch loss — and the calibration split is explicit in the recorded
config. ``--dry-run`` builds the items and prints the exact equivalent ``laya-train`` command
and the config diff without loading model weights or touching the GPU.

**Resume.** ``laya.train.finetune`` loads the base checkpoint and starts at epoch 0: there is
no optimizer or scheduler state to resume. The run summary says so explicitly (spec 06 §5)
and registers the limitation, so a restart is never mistaken for a continuation.
"""

from __future__ import annotations

import argparse
import dataclasses
import shlex
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.decision.laya_adapter import checkpoint_model_hash
from src.pipeline import worlds
from src.pipeline.make_teacher import DATA_FILE, PRIMARY_SPLIT
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, resolve, \
    snapshot_config
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, git_sha, now_iso

#: Loss -> output regime name. The regime names are the ones `worlds` looks up.
REGIMES = {"soft-ce": "ce", "rlcd": "rlcd"}

#: Files `snapshot_download` needs for one Laya checkpoint; anything else is dead weight.
ALLOW_PATTERNS = ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")

#: Base-checkpoint files whose SHA256 identifies it. ``tokenizer/tokenizer_config.json`` is
#: deliberately absent: the library repairs it in place on first load, so its digest is a
#: property of the local cache, not of the published checkpoint.
BASE_IDENTITY_FILES = ("model.safetensors", "rl_agent_config.json", "encoder/config.json",
                       "tokenizer/tokenizer.json")

#: Verbatim resume statement recorded in every run manifest (spec 06 §5).
RESUME = {
    "supported": False,
    "reason": ("laya 0.4.0 train.finetune() loads the base checkpoint and starts at epoch 0; "
               "it writes no optimizer/scheduler state, so there is nothing to resume from"),
    "restart_policy": "rerun the stage; training restarts from the base checkpoint",
}


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
    later load -- and every P6 probability cache -- sees, and stops the first load from
    mutating a checkpoint the registry has already checksummed.
    """
    from laya.agent import _fix_tokenizer_config

    _fix_tokenizer_config(str(checkpoint_dir))


# ------------------------------------------------------------------------ base resolution

def resolve_revision(model: str, revision: str | None, *, pins: Mapping[str, str],
                     hub_sha: Any) -> tuple[str | None, str]:
    """``(revision, source)`` for the base checkpoint; ``None`` means "the Hub default".

    Precedence: a local directory needs no revision; an explicit revision wins; the reviewed
    pin for a Hub repo id (case-insensitive, over the table read at run time) comes next;
    otherwise the Hub commit SHA is queried and recorded, so the manifest identifies the exact
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
    returns ``None`` rather than hitting the network (``--dry-run``).
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
    those stay at this file's configured values — which the run manifest records.
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


# ------------------------------------------------------------------------------- hardware

def gpu_info() -> dict[str, Any]:
    """GPU facts for the manifest; an absent CUDA device is reported, not hidden."""
    info: dict[str, Any] = {"cuda": False, "torch": None}
    try:
        import torch
    except ImportError:  # pragma: no cover - torch ships with the training extra
        return info
    info["torch"] = torch.__version__
    if not torch.cuda.is_available():
        return info
    props = torch.cuda.get_device_properties(0)
    info.update({"cuda": True, "name": props.name,
                 "capability": f"{props.major}.{props.minor}",
                 "total_mib": round(props.total_memory / 2 ** 20, 1),
                 "device_count": torch.cuda.device_count(),
                 "cuda_version": torch.version.cuda})
    return info


def reset_vram_peak() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_vram_mib() -> float | None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)


# ----------------------------------------------------------------------------------- run

def run(*, cfg: dict, profile: str, loss: str, data: str | Path | None = None,
        base: str | None = None, out_dir: str | Path | None = None,
        epochs: int | None = None, max_len: int | None = None,
        micro_batch: int | None = None, grad_accum: int | None = None,
        device: str | None = None, dry_run: bool = False, force: bool = False,
        registry: RunRegistry | None = None, train_module: Any | None = None) -> dict[str, Any]:
    if loss not in REGIMES:
        raise ValueError(f"loss must be one of {sorted(REGIMES)}, got {loss!r}")
    regime = REGIMES[loss]
    laya = cfg["laya"]
    model = base or str(laya["model"])
    out_path = (Path(out_dir) if out_dir is not None
                else resolve(cfg, "models") / worlds.LAYA_CHECKPOINT_DIRS[regime])
    data_path = Path(data) if data is not None else (
        resolve(cfg, "data_processed") / DATA_FILE[PRIMARY_SPLIT])
    device_name = str(device or laya["device"])
    run_id = f"p5-laya-{regime}-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P5_laya_train", status="RUNNING", profile=profile,
                    regime=regime, loss=loss, dry_run=dry_run, base=model,
                    output_dir=str(out_path), resume_supported=False)
    try:
        if not data_path.is_file():
            raise FileNotFoundError(
                f"teacher data not found: {data_path}; run make_teacher first")
        revision, revision_source = resolve_revision(model, laya.get("revision"), pins=load_pins(),
                                                     hub_sha=hub_sha)
        train_module = train_module or load_train_module()
        config = train_config_from_cfg(cfg, loss=loss, train_module=train_module, epochs=epochs,
                                       max_len=max_len, micro_batch=micro_batch,
                                       grad_accum=grad_accum)
        config.validate()
        command = cli_command(config, data_path, model, out_path, device_name)
        diff = config_diff(config)
        base_info = materialize_checkpoint(model, revision, download=not dry_run)
        data_sha = sha256_file(data_path)
        base_record = {
            "model": model, "revision": revision, "revision_source": revision_source,
            "dir": None if base_info is None else base_info["dir"],
            "downloaded": None if base_info is None else base_info["downloaded"],
            "files": {} if base_info is None else base_info["files"],
            "train_config": config_dict(config),
        }

        if dry_run:
            dataset = None
            if base_info is not None:
                dataset = train_module.dry_run(data=str(data_path), model_dir=base_info["dir"],
                                               config=config)
            summary = {
                "run_id": run_id, "loss": loss, "regime": regime, "dry_run": True,
                "base": base_record, "data": {"path": str(data_path), "sha256": data_sha},
                "output_dir": str(out_path),
                "command": shlex.join(command), "config_diff": diff, "dataset": dataset,
                "resume": RESUME,
            }
            registry.append(run_id=run_id, stage="P5_laya_train", status="SKIPPED",
                            profile=profile, regime=regime, loss=loss,
                            reason="dry-run: no training performed", output_dir=str(out_path),
                            resume_supported=False)
            return summary

        if (out_path / "rl_agent_config.json").is_file() and not force:
            raise FileExistsError(
                f"{out_path} already holds a fine-tuned checkpoint. Training cannot resume "
                "(no optimizer state is written), so a rerun would replace it from scratch — "
                "pass --force if that is intended")
        manifest_path = (resolve(cfg, "manifests")
                         / f"P5_train_laya_{regime}_{profile}.json")
        gpu = gpu_info()
        reset_vram_peak()
        started = time.monotonic()
        # laya 0.4.0 CUDA ordering, see the docstring; a no-op once upstream moves the model
        # before its first calibration pass.
        patch_calibration_records(train_module)
        library_summary = train_module.finetune(
            data=str(data_path), model_dir=base_info["dir"], output_dir=str(out_path),
            config=config, device=device_name)
        walltime = round(time.monotonic() - started, 1)
        normalize_checkpoint(out_path)
        output_hash = checkpoint_model_hash(out_path)
        manifest = {
            "run_id": run_id, "stage": "P5_laya_train", "status": "SUCCEEDED",
            "loss": loss, "regime": regime, "profile": profile,
            "epochs": int(config.epochs), "seed": int(config.seed),
            "base": base_record,
            "data": {"path": str(data_path), "sha256": data_sha},
            "output": {"dir": str(out_path), "model_hash": output_hash},
            "config": config_dict(config), "config_diff": diff,
            "equivalent_cli_command": shlex.join(command),
            "calibration": {
                "train_items": library_summary.get("train_items"),
                "calibration_items": library_summary.get("calibration_items"),
                "skipped": library_summary.get("skipped"),
                "calib_max": int(config.calib_max), "calib_frac": float(config.calib_frac),
                "calib_seed": int(config.calib_seed),
                "temperature": library_summary.get("temperature"),
                "temperature_by_options": library_summary.get("temperature_by_options"),
                "n_by_bucket": library_summary.get("n_by_bucket"),
                "abstention_thresholds": library_summary.get("abstention_thresholds"),
            },
            "epoch_loss": library_summary.get("epoch_loss"),
            "resume": RESUME,
            "gpu": gpu, "peak_vram_mib": peak_vram_mib(), "walltime_s": walltime,
            "code_git_sha": git_sha(REPO_ROOT),
            "library_versions": capture_library_versions(),
            "creation_script_sha": sha256_file(Path(__file__)),
            "created_utc": now_iso(),
            "library_summary": library_summary,
        }
        atomic_write_json(manifest_path, manifest)
    except Exception as exc:
        registry.append(run_id=run_id, stage="P5_laya_train", status="FAILED", profile=profile,
                        regime=regime, loss=loss, error=f"{type(exc).__name__}: {exc}")
        raise

    registry.record_artifact(run_id, "P5_laya_train", [out_path, manifest_path],
                             profile=profile, regime=regime, loss=loss,
                             epochs=int(config.epochs), seed=int(config.seed),
                             base_revision=revision, base_files=base_record["files"],
                             data_sha=data_sha, output_model_hash=output_hash,
                             walltime_s=walltime, peak_vram_mib=manifest["peak_vram_mib"],
                             temperature=library_summary.get("temperature"),
                             resume_supported=False)
    return {"run_id": run_id, "loss": loss, "regime": regime, "dry_run": False,
            "out_dir": str(out_path), "manifest": str(manifest_path), "base": base_record,
            "model_hash": output_hash, "walltime_s": walltime,
            "peak_vram_mib": manifest["peak_vram_mib"], "epochs": int(config.epochs),
            "seed": int(config.seed), "temperature": library_summary.get("temperature"),
            "train_items": library_summary.get("train_items"),
            "calibration_items": library_summary.get("calibration_items"),
            "epoch_loss": library_summary.get("epoch_loss"), "resume": RESUME,
            "command": shlex.join(command)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.train_laya",
        description="P5: fine-tune Laya on the teacher data (soft-CE or RLCD)")
    add_profile_args(parser)
    parser.add_argument("--loss", required=True, choices=sorted(REGIMES),
                        help="training objective; selects the output checkpoint directory")
    parser.add_argument("--data", default=None,
                        help="teacher JSONL (default: data/processed/laya_train.jsonl)")
    parser.add_argument("--base", default=None,
                        help="base checkpoint: Hub id or local directory (default: laya.model)")
    parser.add_argument("--out", default=None,
                        help="output directory (default: outputs/models/laya_{ce,rlcd})")
    parser.add_argument("--epochs", type=int, default=None, help="override laya.epochs")
    parser.add_argument("--max-len", type=int, default=None, help="override laya.max_len")
    parser.add_argument("--micro-batch", type=int, default=None, help="override laya.micro_batch")
    parser.add_argument("--grad-accum", type=int, default=None, help="override laya.grad_accum")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda (default: laya.device)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved base and command and the config diff; no "
                             "model weights, no GPU")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing output checkpoint (training never resumes)")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    if args.seed is not None:
        cfg["laya"]["seed"] = args.seed
    regime = REGIMES[args.loss]
    snapshot_config(cfg, resolve(cfg, "manifests")
                    / f"P5_train_laya_{regime}_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, loss=args.loss, data=args.data, base=args.base,
                  out_dir=args.out, epochs=args.epochs, max_len=args.max_len,
                  micro_batch=args.micro_batch, grad_accum=args.grad_accum,
                  device=args.device, dry_run=args.dry_run, force=args.force)
    if args.json:
        import json

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    base = summary["base"]
    origin = (f"{base['revision'][:12]}… ({base['revision_source']})"
              if base["revision"] else base["revision_source"])
    if summary["dry_run"]:
        print(f"dry-run: loss={summary['loss']} base={base['model']}@{origin}")
        print(f"  base dir: {base['dir']}")
        print(f"  data: {summary['data']['path']} sha256={summary['data']['sha256'][:12]}…")
        if summary["dataset"] is not None:
            dataset = summary["dataset"]
            print(f"  items: rows={dataset['rows_read']} valid={dataset['valid_items']} "
                  f"skipped={dataset['skipped']} max_len={dataset['max_len']} "
                  f"head_max_len={dataset['head_max_len']}")
        else:
            print("  base not materialized locally; dataset stats skipped "
                  "(pass --base <local dir> for a full dry run)")
        print(f"  command: {summary['command']}")
        import json

        print(f"  config diff vs laya defaults: "
              f"{json.dumps(summary['config_diff'], ensure_ascii=False)}")
        return 0
    print(f"run_id={summary['run_id']} loss={summary['loss']} out={summary['out_dir']}")
    print(f"  base={base['model']}@{origin} files={len(base['files'])}")
    print(f"  items: train={summary['train_items']} calib={summary['calibration_items']} "
          f"temperature={summary['temperature']} epoch_loss={summary['epoch_loss']}")
    print(f"  walltime={summary['walltime_s']}s "
          f"peak_vram={summary['peak_vram_mib']}MiB model_hash={summary['model_hash'][:16]}…")
    print(f"  resume: not supported — {RESUME['reason']} "
          f"({RESUME['restart_policy']}; spec 06 §5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

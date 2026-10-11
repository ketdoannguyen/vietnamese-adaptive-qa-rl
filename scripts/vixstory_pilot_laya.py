"""Pilot Laya fine-tune stage for VixSTORY: trains E04 (soft-CE) and E05 (RLCD).

Wires the existing ``src.pipeline.train_laya`` seams -- :func:`materialize_checkpoint`,
:func:`patch_calibration_records`, :func:`normalize_checkpoint` and
``laya.train.finetune`` -- so the pilot gets its own, explicitly supplied inputs and its own
pilot-scoped outputs, without touching the ViQuAD defaults (``data/processed/laya_train.jsonl``,
``outputs/models/laya_{ce,rlcd}``):

    python -m scripts.vixstory_pilot_laya \\
        --data outputs/experiments/vixstory-pilot-300-2026-10-11/teacher/train300.jsonl \\
        --base convaiinnovations/laya-multilingual \\
        --pilot-manifest outputs/experiments/vixstory-pilot-300-2026-10-11/manifest.json \\
        --out-root outputs/experiments/vixstory-pilot-300-2026-10-11/laya \\
        --loss both --epochs 3 --device cuda:0

Guarantees, each refused with a named error rather than silently defaulted:

* ``--data``, ``--base``, ``--pilot-manifest`` and ``--out-root`` are all **required**: the
  stage never falls back to a ViQuAD path.
* The teacher JSONL must be **TRAIN300-only**. Every row's ``meta.qid`` must be a
  ``pilot_300`` QID and none may be a ``validation_246`` QID, so neither the training rows nor
  the library's internal calibration split (drawn from the same file) can contain validation.
* A preexisting, non-empty output directory (``E04_ce`` / ``E05_rlcd``) is refused: this
  stage cannot resume and would replace it from scratch.
* The base checkpoint is materialized with ``download=False`` -- local ``--base`` directory and
  an already-cached pinned snapshot are used **without any network**. A missing cache is a
  blocker unless ``--allow-download`` is passed explicitly.

Each trained arm writes ``<out-root>/<arm>.manifest.json`` recording the data SHA256, the base
identity-file digests, the output model hash, the exact train config and the calibration counts.
No model weights are loaded at import time, and nothing touches the GPU until ``finetune``.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.decision.laya_adapter import checkpoint_model_hash
from src.pipeline import train_laya
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, load_config
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_json
from src.utils.runs import git_sha, now_iso

#: Pilot arm names, one per loss. The ``E04``/``E05`` prefixes are the VixSTORY evaluation arms.
LOSS_ARMS = {"soft-ce": "E04_ce", "rlcd": "E05_rlcd"}

#: Length of the frozen pilot splits the teacher scope is checked against.
PILOT_SPLIT = "pilot_300"
VALIDATION_SPLIT = "validation_246"


# ------------------------------------------------------------------- inputs and scope guard

def pilot_split_qids(manifest_path: str | Path) -> tuple[set[str], set[str]]:
    """Pilot QIDs and optional validation QIDs from the frozen manifest."""
    manifest = read_json(manifest_path)
    splits = manifest.get("splits") if isinstance(manifest, Mapping) else None
    if not isinstance(splits, Mapping):
        raise ValueError(f"{manifest_path}: manifest carries no 'splits' mapping")
    pilot = splits.get(PILOT_SPLIT)
    pilot_qids = pilot.get("qids") if isinstance(pilot, Mapping) else None
    if not pilot_qids:
        raise ValueError(f"{manifest_path}: no {PILOT_SPLIT!r} QID list")
    validation = splits.get(VALIDATION_SPLIT)
    if validation is not None and not isinstance(validation, Mapping):
        raise ValueError(f"{manifest_path}: malformed {VALIDATION_SPLIT!r} split")
    validation_qids = validation.get("qids") if validation is not None else ()
    if validation_qids is None or isinstance(validation_qids, (str, bytes)):
        raise ValueError(f"{manifest_path}: malformed {VALIDATION_SPLIT!r} QID list")
    return {str(qid) for qid in pilot_qids}, {str(qid) for qid in validation_qids}


def read_teacher_rows(data_path: str | Path) -> list[dict[str, Any]]:
    """Teacher JSONL rows; every row must carry a ``meta.qid`` and v2 text."""
    from src.decision.vixstory_state_text import STATE_TEXT_VERSION
    from src.utils.io import read_jsonl

    rows = list(read_jsonl(data_path))
    if not rows:
        raise ValueError(f"{data_path}: teacher JSONL is empty")
    for index, row in enumerate(rows):
        meta = row.get("meta") if isinstance(row, Mapping) else None
        if not isinstance(meta, Mapping) or not meta.get("qid"):
            raise ValueError(f"{data_path}: row {index} has no meta.qid")
        if meta.get("state_text_version") != STATE_TEXT_VERSION:
            raise ValueError(f"{data_path}: row {index} is not state text {STATE_TEXT_VERSION}")
    return rows


def guard_teacher_scope(row_qids: set[str], pilot_qids: set[str],
                        validation_qids: set[str]) -> dict[str, int]:
    """Refuse teacher data that is not exactly a subset of the ``pilot_300`` QIDs."""
    leaked = sorted(row_qids & validation_qids)
    if leaked:
        raise ValueError(
            f"teacher data contains {len(leaked)} validation QIDs (e.g. {leaked[:3]}); "
            "training and calibration must never see validation rows")
    extra = sorted(row_qids - pilot_qids)
    if extra:
        raise ValueError(
            f"teacher data contains {len(extra)} QIDs outside {PILOT_SPLIT} "
            f"(e.g. {extra[:3]}); the pilot teacher must be TRAIN300-only")
    return {"validation_rows_in_train": 0, "qids_outside_pilot": 0}


# --------------------------------------------------------------- base checkpoint (no network)

def base_checkpoint(base: str, revision: str | None, *, allow_download: bool) -> dict[str, Any]:
    """A local base checkpoint directory, resolved without downloading unless permitted.

    A local ``--base`` directory or a pinned snapshot already in the HF cache is used as-is
    (``download=False`` never reaches the network). Otherwise the stage refuses; the Hub is
    queried only when ``allow_download`` is set.
    """
    if Path(base).expanduser().is_dir():
        info = train_laya.materialize_checkpoint(base, revision, download=False)
        return {**info, "revision_source": "local"}

    pins = train_laya.load_pins()
    if allow_download:
        resolved, source = train_laya.resolve_revision(
            base, revision, pins=pins, hub_sha=train_laya.hub_sha)
        info = train_laya.materialize_checkpoint(base, resolved, download=True)
    else:
        source = "pinned"
        resolved = revision or pins.get(base) or pins.get(base.lower())
        info = train_laya.materialize_checkpoint(base, resolved, download=False)
    if info is None:
        raise FileNotFoundError(
            f"{base!r} is not in the local Hugging Face cache: no network is used when the "
            f"pinned base is absent. Warm the cache, pass a local --base directory, or pass "
            f"--allow-download to fetch it (revision={revision!r}).")
    return {**info, "revision_source": source}


# ---------------------------------------------------------------------------- output refusal

def refuse_preexisting(out_dir: str | Path) -> None:
    """Refuse a non-empty output directory; this stage cannot resume and would overwrite it."""
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty. Fine-tuning writes no optimizer state, so "
            "a rerun would replace the checkpoint from scratch -- move it aside first.")


def arm_output_dir(out_root: str | Path, arm: str) -> Path:
    return Path(out_root) / arm


# ------------------------------------------------------------------------------------- arm run

def _arm_manifest(*, arm: str, loss: str, regime: str, status: str, data_path: Path,
                  data_sha: str, rows: list[dict[str, Any]], guard: Mapping[str, int],
                  pilot_manifest: Path, pilot_n: int, validation_n: int,
                  base_info: Mapping[str, Any], base_arg: str, out_dir: Path,
                  config: Any, config_diff: Mapping[str, Any], command: Sequence[str],
                  device: str, summary: Mapping[str, Any] | None,
                  model_hash: str | None, teacher_contract: Mapping[str, Any] | None) -> dict[str, Any]:
    row_qids = {str(row["meta"]["qid"]) for row in rows}
    calibration = None
    if summary is not None:
        calibration = {
            "train_items": summary.get("train_items"),
            "calibration_items": summary.get("calibration_items"),
            "skipped": summary.get("skipped"),
            "temperature": summary.get("temperature"),
            "n_by_bucket": summary.get("n_by_bucket"),
            "abstention_thresholds": summary.get("abstention_thresholds"),
        }
    from src.decision.vixstory_state_text import STATE_TEXT_VERSION

    base_tokenizer = Path(base_info["dir"]) / "tokenizer"
    tokenizer_contract_sha = None
    if base_tokenizer.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(base_tokenizer, local_files_only=True)
        contract = json.loads(tokenizer.backend_tokenizer.to_str())
        contract.pop("truncation", None)
        tokenizer_contract_sha = sha256_json(contract)
    return {
        "stage": "P5_laya_pilot",
        "state_text_version": STATE_TEXT_VERSION,
        "tokenizer_contract_sha256": tokenizer_contract_sha,
        "teacher_contract": None if teacher_contract is None else dict(teacher_contract),
        "max_len": int(config.max_len), "head_max_len": int(config.head_max_len),
        "status": status,
        "arm": arm,
        "loss": loss,
        "regime": regime,
        "pilot_manifest": {"path": str(pilot_manifest), "sha256": sha256_file(pilot_manifest),
                           PILOT_SPLIT: pilot_n, VALIDATION_SPLIT: validation_n},
        "data": {"path": str(data_path), "sha256": data_sha, "rows": len(rows),
                 "qids": len(row_qids), "qids_sha256": sha256_json(sorted(row_qids))},
        "scope_guard": dict(guard),
        "base": {"model": base_arg, "dir": base_info["dir"],
                 "downloaded": base_info["downloaded"], "revision": base_info["revision"],
                 "revision_source": base_info.get("revision_source"), "files": base_info["files"]},
        "output": {"dir": str(out_dir), "model_hash": model_hash},
        "config": train_laya.config_dict(config),
        "config_diff": dict(config_diff),
        "equivalent_cli_command": shlex.join(command),
        "epochs": int(config.epochs),
        "seed": int(config.seed),
        "device": device,
        "calibration": calibration,
        "library_summary": None if summary is None else dict(summary),
        "creation_script_sha": sha256_file(Path(__file__)),
        "code_git_sha": git_sha(REPO_ROOT),
        "created_utc": now_iso(),
    }


def run(*, data: str | Path, base: str, pilot_manifest: str | Path, out_root: str | Path,
        loss: str = "both", revision: str | None = None, epochs: int | None = None,
        seed: int | None = None, device: str | None = None, allow_download: bool = False,
        dry_run: bool = False, profile: str | None = None, cfg: Mapping[str, Any] | None = None,
        train_module: Any | None = None,
        teacher_provenance: str | Path | None = None) -> dict[str, Any]:
    """Train the requested pilot arm(s) on the TRAIN300-only teacher file.

    Returns ``{arm: {out_dir, manifest, model_hash, dry_run, epochs, seed, device}}``.
    """
    if loss != "both" and loss not in LOSS_ARMS:
        raise ValueError(f"loss must be one of {sorted(LOSS_ARMS)} or 'both', got {loss!r}")
    arms = list(LOSS_ARMS) if loss == "both" else [loss]

    data_path = Path(data)
    if not data_path.is_file():
        raise FileNotFoundError(f"teacher JSONL not found: {data_path}; pass --data explicitly")
    pilot_manifest_path = Path(pilot_manifest)
    if not pilot_manifest_path.is_file():
        raise FileNotFoundError(
            f"pilot manifest not found: {pilot_manifest_path}; pass --pilot-manifest")
    out_root = Path(out_root)

    pilot_qids, validation_qids = pilot_split_qids(pilot_manifest_path)
    rows = read_teacher_rows(data_path)
    guard = guard_teacher_scope({str(row["meta"]["qid"]) for row in rows}, pilot_qids,
                                validation_qids)
    data_sha = sha256_file(data_path)
    teacher_contract = None
    if teacher_provenance is None and train_module is None:
        raise ValueError("teacher provenance is required before a real Laya run")
    if teacher_provenance is not None:
        if Path(teacher_provenance).resolve() != data_path.with_suffix(".provenance.json").resolve():
            raise ValueError("teacher provenance must be adjacent to the supplied JSONL")
        teacher_contract = read_json(teacher_provenance)
        if teacher_contract.get("sha256") != data_sha:
            raise ValueError("teacher provenance data SHA differs from supplied JSONL")

    for arm_loss in arms:  # fail fast before any training
        refuse_preexisting(arm_output_dir(out_root, LOSS_ARMS[arm_loss]))

    base_info = base_checkpoint(base, revision, allow_download=allow_download)
    config_cfg = cfg if cfg is not None else load_config(profile)
    tm = train_module if train_module is not None else train_laya.load_train_module()

    results: dict[str, Any] = {}
    for arm_loss in arms:
        arm = LOSS_ARMS[arm_loss]
        out_dir = arm_output_dir(out_root, arm)
        config = train_laya.train_config_from_cfg(config_cfg, loss=arm_loss, train_module=tm,
                                                  epochs=epochs)
        if seed is not None:
            config.seed = int(seed)
        config.validate()
        if teacher_contract is not None:
            from src.decision import vixstory_state_text
            from src.decision.vixstory_state_text import STATE_TEXT_VERSION
            from transformers import AutoTokenizer

            if (teacher_contract.get("state_text_sha256") !=
                    sha256_file(Path(vixstory_state_text.__file__))):
                raise ValueError("teacher state renderer source differs from this training run")
            tokenizer_dir = Path(base_info["dir"]) / "tokenizer"
            if not tokenizer_dir.is_dir():
                raise ValueError("teacher tokenizer contract requires a local base tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
            contract = json.loads(tokenizer.backend_tokenizer.to_str())
            contract.pop("truncation", None)
            if (teacher_contract.get("state_text_version") != STATE_TEXT_VERSION
                    or teacher_contract.get("max_len") != config.max_len
                    or teacher_contract.get("head_max_len") != config.head_max_len
                    or teacher_contract.get("tokenizer_contract_sha256") != sha256_json(contract)):
                raise ValueError("teacher tokenizer contract, state version or sequence lengths differ")
        device_name = str(device or config_cfg["laya"]["device"])
        command = train_laya.cli_command(config, data_path, base_info["dir"], out_dir, device_name)
        diff = train_laya.config_diff(config)

        summary = None
        model_hash = None
        if not dry_run:
            train_laya.patch_calibration_records(tm)
            summary = tm.finetune(data=str(data_path), model_dir=base_info["dir"],
                                  output_dir=str(out_dir), config=config, device=device_name)
            train_laya.normalize_checkpoint(out_dir)
            model_hash = checkpoint_model_hash(out_dir)

        manifest = _arm_manifest(
            arm=arm, loss=arm_loss, regime=train_laya.REGIMES[arm_loss],
            status="DRY_RUN" if dry_run else "SUCCEEDED", data_path=data_path, data_sha=data_sha,
            rows=rows, guard=guard, pilot_manifest=pilot_manifest_path, pilot_n=len(pilot_qids),
            validation_n=len(validation_qids), base_info=base_info, base_arg=base,
            out_dir=out_dir, config=config, config_diff=diff, command=command,
            device=device_name, summary=summary, model_hash=model_hash,
            teacher_contract=teacher_contract)
        manifest_path = out_root / f"{arm}.manifest.json"
        if not dry_run:
            out_root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(manifest_path, manifest)

        results[arm] = {
            "out_dir": str(out_dir), "manifest": str(manifest_path), "loss": arm_loss,
            "model_hash": model_hash, "dry_run": dry_run, "epochs": int(config.epochs),
            "seed": int(config.seed), "device": device_name,
            "calibration_items": None if summary is None else summary.get("calibration_items"),
        }
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.vixstory_pilot_laya",
        description="VixSTORY pilot: fine-tune Laya E04 (soft-CE) and E05 (RLCD) on TRAIN300.")
    add_profile_args(parser)
    parser.add_argument("--data", required=True,
                        help="TRAIN300-only teacher JSONL (no default; ViQuAD paths are never used)")
    parser.add_argument("--teacher-provenance", type=Path, required=True,
                        help="teacher-stage JSON metadata binding state, tokenizer and sequence budget")
    parser.add_argument("--base", required=True,
                        help="base checkpoint: local directory or Hub id (cached, no network)")
    parser.add_argument("--revision", default=None, help="base revision (for a Hub id)")
    parser.add_argument("--pilot-manifest", required=True,
                        help="frozen pilot manifest.json for the TRAIN300/validation scope guard")
    parser.add_argument("--out-root", required=True,
                        help="output root; checkpoints land in <out-root>/E04_ce and E05_rlcd")
    parser.add_argument("--loss", default="both", choices=[*sorted(LOSS_ARMS), "both"],
                        help="which arm(s) to train (default: both E04 and E05)")
    parser.add_argument("--epochs", type=int, default=None, help="override laya.epochs")
    parser.add_argument("--device", default=None, help="auto | cpu | cuda[:N] (default: laya.device)")
    parser.add_argument("--allow-download", action="store_true",
                        help="permit Hub network access when the pinned base is not cached")
    parser.add_argument("--dry-run", action="store_true",
                        help="build configs and check the base without training")
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    results = run(data=args.data, base=args.base, pilot_manifest=args.pilot_manifest,
                  out_root=args.out_root, loss=args.loss, revision=args.revision,
                  epochs=args.epochs, seed=args.seed, device=args.device,
                  allow_download=args.allow_download, dry_run=args.dry_run, cfg=cfg,
                  teacher_provenance=args.teacher_provenance)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

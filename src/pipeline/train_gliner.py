"""Small, guarded GLiNER LoRA pilot on the same hard-label teacher rows as Laya."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from src.decision.schema import ACTION_ORDER, build_questions_field
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, resolve
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, read_jsonl

BASE = REPO_ROOT / "outputs/experiments/2026-10-09-gliner-p0/hf/model"
PINS = REPO_ROOT / "outputs/experiments/2026-10-09-gliner-p0/env/wheel_sha256.txt"
REVISION = "e173bca1f0c4217d7a55b3b0c705d5ece8a0218d"


def verify_base() -> dict[str, str]:
    pins = {}
    for line in PINS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 64:
            pins[parts[1]] = parts[0]
    observed = {}
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        key = f"hf/model/{name}"
        observed[name] = sha256_file(BASE / name)
        if observed[name] != pins.get(key):
            raise ValueError(f"pinned GLiNER base hash mismatch: {key}")
    return observed


def examples_from_rows(rows):
    from gliner2 import InputExample
    from gliner2.training.data import Classification

    expected_schema = build_questions_field()["next_action"]
    examples = []
    for i, row in enumerate(rows):
        meta = row.get("meta") or {}
        action = (row.get("expected") or {}).get("next_action")
        question = (row.get("questions") or {}).get("next_action")
        if (meta.get("split") != "train_strict" or meta.get("privileged")
                or action not in ACTION_ORDER or question != expected_schema
                or meta.get("state_text_version") != "v2"
                or not isinstance(row.get("state"), str) or not row["state"]):
            raise ValueError(f"invalid pilot teacher row {i}")
        examples.append(InputExample(text=row["state"], classifications=[Classification(
            task="next_action", labels=list(ACTION_ORDER), true_label=action,
            label_descriptions=dict(expected_schema["criteria"]),
            prompt=expected_schema["instructions"]) ]))
    if not examples:
        raise ValueError("empty teacher pilot")
    return examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_profile_args(parser)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args(argv)
    if not 1 <= args.epochs <= 2:
        parser.error("pilot epochs must be 1 or 2")
    cfg = config_from_args(args)
    data = resolve(cfg, "data_processed") / "laya_train.jsonl"
    out = resolve(cfg, "models") / "gliner_sft"
    report = resolve(cfg, "manifests") / "gliner_pilot.json"
    if out.exists() or report.exists():
        raise FileExistsError(f"GLiNER pilot output exists: {out} or {report}")
    if not data.is_file():
        raise FileNotFoundError(data)
    base_hashes = verify_base()
    rows = list(read_jsonl(data))
    from gliner2 import AutoExtractor
    from gliner2.training import ExtractorTrainer, TrainingConfig
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("GLiNER pilot requires a CUDA device with BF16")
    # Laya's fixed calibration shuffle excludes these same 30 rows from gradient updates.
    # Keep GLiNER's weight-training examples exactly aligned, not merely the source file.
    shuffled = list(range(len(rows)))
    random.Random(int(cfg["laya"]["calib_seed"])).shuffle(shuffled)
    excluded = set(shuffled[:min(int(cfg["laya"]["calib_max"]),
                              int(len(rows) * float(cfg["laya"]["calib_frac"])))])
    examples = examples_from_rows(row for i, row in enumerate(rows) if i not in excluded)
    torch.cuda.reset_peak_memory_stats()
    config = TrainingConfig(
        output_dir=str(out), experiment_name="gliner-laya-pilot",
        num_epochs=args.epochs, batch_size=2, gradient_accumulation_steps=1,
        encoder_lr=1e-5, task_lr=5e-4, weight_decay=0.01,
        fp16=False, bf16=True, use_lora=True, lora_r=8, lora_alpha=16.0,
        lora_dropout=0.05, lora_target_modules=["encoder"], save_adapter_only=True,
        eval_strategy="no", logging_steps=1, seed=42, num_workers=0,
        max_len=4096, strict_training=True, report_to_wandb=False,
    )
    started = time.monotonic()
    trainer = ExtractorTrainer(AutoExtractor.from_pretrained(str(BASE)), config)
    summary = trainer.train(train_data=examples)
    adapter = out / "final" / "adapter_model.safetensors"
    if not adapter.is_file() or summary["total_steps"] <= 0:
        raise RuntimeError("GLiNER training did not produce a final adapter and steps")
    losses = [entry["loss"] for entry in summary["train_metrics_history"]]
    import math
    if not losses or not all(math.isfinite(float(loss)) for loss in losses):
        raise RuntimeError("GLiNER training loss is missing or non-finite")
    result = {
        "status": "SUCCEEDED", "base": str(BASE), "base_revision": REVISION,
        "base_sha256": base_hashes, "teacher_path": str(data),
        "teacher_sha256": sha256_file(data), "n_teacher_rows": len(rows),
        "n_weight_train_rows": len(examples), "laya_calibration_excluded_indices": sorted(excluded),
        "epochs_requested": args.epochs, "steps": summary["total_steps"],
        "epochs_actual": summary["total_epochs"], "losses": losses,
        "train_seconds": time.monotonic() - started,
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "adapter_path": str(adapter), "adapter_sha256": sha256_file(adapter),
        "precision": "bf16", "lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
    }
    atomic_write_json(report, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

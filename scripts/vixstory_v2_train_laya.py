"""Train V2 Laya's decision head with the whole encoder frozen, in bf16 on length-bucketed batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import types
from pathlib import Path
from unittest.mock import patch

import torch

from src.decision.schema_v2 import laya_record
from src.decision.vixstory_state_text_v2 import HEAD_MAX_LEN, MAX_LEN, render_with_retention
from src.pipeline.vixstory_teacher_v2 import VixStoryTeacherV2
from src.pipeline.vixstory_worlds_v2 import load_worlds
from src.utils.hashing import sha256_file

EXPECTED_TRAIN_QIDS = 2553
EXPECTED_TRAIN_STATES = 17871
LAYA_REVISION = "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
BUCKET = 64


def teacher_rows(worlds, tokenizer):
    for world in worlds:
        teacher = VixStoryTeacherV2(world)
        for obs in teacher.enumerate_states():
            state, audit = render_with_retention(obs, tokenizer=tokenizer)
            if audit["truncated"] or not audit["essential_fields_present"]:
                raise RuntimeError(f"invalid public state for {world.qid}")
            yield laya_record(state, teacher.probabilities(obs))


def freeze_all_encoder_blocks(model) -> dict:
    """Freeze the complete ModernBERT encoder; only the two-layer decision head trains."""
    blocks = getattr(model.encoder, "layers", None)
    if blocks is None or len(blocks) != 22 or len(model.head.layers) != 2:
        raise RuntimeError("expected pinned 22-block ModernBERT and 2-block Laya head")
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.parameters():
        if parameter.device.type != "cpu":
            raise RuntimeError("freeze before moving the loaded checkpoint to GPU")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    expected = {name for name, _ in model.named_parameters() if not name.startswith("encoder.")}
    if trainable != expected:
        raise RuntimeError("unexpected head-only trainable parameter set")
    return {"encoder_blocks_total": len(blocks), "encoder_blocks_trainable": [],
            "head_blocks_trainable": len(model.head.layers),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad)}


def frozen_parameter_digest(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class BucketedRandom(random.Random):
    """random.Random whose shuffle sorts by state length and shuffles inside BUCKET-item groups.

    Padding every micro-batch to its longest member wastes a third of the compute on the
    full-length mixture; keeping near-equal lengths adjacent costs 1-6% instead. Plain
    (non-item) sequences keep the uniform shuffle, e.g. option orders in `draw_option_order`.
    """

    def shuffle(self, x):
        if not x or not isinstance(x[0], dict):
            return super().shuffle(x)
        x.sort(key=lambda item: len(item["state_ids"]))
        for start in range(0, len(x), BUCKET):
            group = x[start:start + BUCKET]
            super().shuffle(group)
            x[start:start + BUCKET] = group
        buckets = [x[start:start + BUCKET] for start in range(0, len(x), BUCKET)]
        super().shuffle(buckets)
        x[:] = [item for bucket in buckets for item in bucket]


def bf16_forward(model, batch, device, amp, detach_encoder):
    """laya.train._forward with bfloat16 autocast instead of the hardcoded float16."""
    args = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
            batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
    kwargs = {"detach_encoder": detach_encoder}
    if "option_ids" in batch:
        kwargs.update(position_ids=batch["position_ids"].to(device),
                      option_ids=batch["option_ids"].to(device))
    with torch.autocast(device.type, dtype=torch.bfloat16):
        logits, _act = model(*args, **kwargs)
    return logits.float()


def run(run_dir: Path, output: Path, arm: str, *, micro_batch: int = 4,
        grad_accum: int = 2, epochs: int = 3) -> None:
    from laya.train import TrainConfig, items_from_rows, load_checkpoint, save_checkpoint, train_model

    if arm not in ("E04", "E05"):
        raise ValueError("only E04 soft CE and E05 RLCD are in the V2 Laya protocol")
    if micro_batch < 1 or grad_accum < 1 or epochs != 3:
        raise ValueError("micro_batch and grad_accum must be positive; partial run has 3 epochs")
    if not torch.cuda.is_available():
        raise RuntimeError("full V2 Laya training requires the local GPU")
    audit = json.loads((run_dir / "audit/retention_summary.json").read_text())
    if (audit.get("status") != "PASS_NO_EVIDENCE_CUT"
            or audit.get("max_len") != MAX_LEN or audit.get("head_max_len") != HEAD_MAX_LEN
            or audit["splits"]["train"]["n_states"] != EXPECTED_TRAIN_STATES):
        raise RuntimeError("full-data retention gate is not satisfied")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite training artifacts: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    journal = output / "progress.jsonl"

    def record(entry):
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    try:
        started = time.monotonic()
        checkpoint = (Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya-multilingual"
                      / "snapshots" / LAYA_REVISION)
        model, tokenizer, cfg = load_checkpoint(str(checkpoint))
        freeze = freeze_all_encoder_blocks(model)
        frozen_before = frozen_parameter_digest(model)
        record({"stage": "head_only_freeze_verified", "arm": arm, **freeze,
                "frozen_parameter_sha256_before": frozen_before})
        worlds = load_worlds(run_dir, "train")
        if len(worlds) != EXPECTED_TRAIN_QIDS or len({w.qid for w in worlds}) != EXPECTED_TRAIN_QIDS:
            raise RuntimeError("training must cover exactly 2553 distinct official train QIDs")
        record({"stage": "loaded_train", "arm": arm, "qids": len(worlds),
                "elapsed_seconds": time.monotonic() - started})
        items, skipped = items_from_rows(tokenizer, teacher_rows(worlds, tokenizer), MAX_LEN,
                                         HEAD_MAX_LEN)
        if len(items) != EXPECTED_TRAIN_STATES or skipped:
            raise RuntimeError(f"Teacher item gate failed: {len(items)} items; skipped {skipped}")
        del worlds
        record({"stage": "all_items_built", "arm": arm, "items": len(items),
                "skipped": skipped, "elapsed_seconds": time.monotonic() - started})

        config = TrainConfig(epochs=epochs, loss="soft-ce" if arm == "E04" else "rlcd",
                             seed=42, micro_batch=micro_batch, grad_accum=grad_accum,
                             max_len=MAX_LEN, head_max_len=HEAD_MAX_LEN, amp=True,
                             gradient_checkpointing=False, freeze_encoder=True,
                             head_lr=2e-4, log_every=100)
        config.validate()
        expected_steps_per_epoch = math.ceil(math.ceil(len(items) / micro_batch) / grad_accum)
        expected_steps = expected_steps_per_epoch * config.epochs
        record({"stage": "training_start", "arm": arm, "precision": "bf16-autocast",
                "batching": "length-bucketed-64",
                "micro_batch": micro_batch, "grad_accum": grad_accum,
                "epochs": epochs, "expected_optimizer_steps": expected_steps,
                "checkpointing": False, "head_lr": config.head_lr, "trainable": freeze})
        saved_cfg = {**cfg, "max_len": MAX_LEN, "head_max_len": HEAD_MAX_LEN,
                     "vixstory_head_only_training": {"seed": 42, "epochs": epochs,
                                                     "head_lr": config.head_lr,
                                                     "freeze_encoder": True,
                                                     "precision": "bf16-autocast",
                                                     "batching": "length-bucketed-64", **freeze}}
        actual_steps = 0
        gradient_checks = 0
        loss_calls = 0
        import laya.train as laya_train

        original_step = torch.optim.AdamW.step
        original_clip = torch.nn.utils.clip_grad_norm_
        original_rlcd = laya_train.rlcd_loss
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        frozen_parameters = [p for p in model.parameters() if not p.requires_grad]
        head_parameters = [p for n, p in model.named_parameters()
                           if not n.startswith("encoder.") and p.requires_grad]

        def checked_step(optimizer, *args, **kwargs):
            nonlocal actual_steps
            if actual_steps == 0:
                if (len(optimizer.param_groups) != 1
                        or [group["lr"] for group in optimizer.param_groups] != [config.head_lr]
                        or [set(map(id, group["params"])) for group in optimizer.param_groups]
                        != [set(map(id, head_parameters))]):
                    raise RuntimeError("optimizer must update only the decision head")
            result = original_step(optimizer, *args, **kwargs)
            actual_steps += 1
            return result

        def checked_clip(parameters, *args, **kwargs):
            nonlocal gradient_checks
            params = list(parameters)
            if (set(map(id, params)) != set(map(id, trainable_parameters))
                    or any(param.grad is not None for param in frozen_parameters)):
                raise RuntimeError("frozen encoder received a gradient or entered clipping")
            for param in params:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError("non-finite pre-clip gradient")
            norm = original_clip(params, *args, **kwargs)
            if not torch.isfinite(norm) or norm <= 0:
                raise FloatingPointError(f"invalid pre-clip gradient norm: {norm}")
            gradient_checks += 1
            return norm

        def counted_rlcd(*args, **kwargs):
            nonlocal loss_calls
            loss_calls += 1
            return original_rlcd(*args, **kwargs)

        def epoch_end(epoch, mean_loss):
            if not math.isfinite(mean_loss):
                raise FloatingPointError(f"non-finite epoch {epoch + 1} mean loss")
            if actual_steps != (epoch + 1) * expected_steps_per_epoch:
                raise RuntimeError(f"epoch {epoch + 1} optimizer steps: {actual_steps} != "
                                   f"{(epoch + 1) * expected_steps_per_epoch}")
            if gradient_checks != actual_steps:
                raise RuntimeError("gradient checks do not match actual optimizer steps")
            if frozen_parameter_digest(model) != frozen_before:
                raise RuntimeError("frozen encoder parameters changed during training")
            epoch_dir = output / f"epoch-{epoch + 1}"
            if epoch_dir.exists():
                raise FileExistsError(epoch_dir)
            save_checkpoint(model, tokenizer, saved_cfg, str(epoch_dir))
            record({"stage": "epoch_complete", "arm": arm, "epoch": epoch + 1,
                    "mean_loss": float(mean_loss), "optimizer_steps": actual_steps,
                    "gradient_checks": gradient_checks,
                    "frozen_parameter_sha256": frozen_before,
                    "checkpoint_sha256": sha256_file(epoch_dir / "model.safetensors"),
                    "elapsed_seconds": time.monotonic() - started})

        torch.cuda.reset_peak_memory_stats()
        with (patch.object(torch.optim.AdamW, "step", checked_step),
              patch.object(torch.nn.utils, "clip_grad_norm_", checked_clip),
              patch.object(laya_train, "rlcd_loss", counted_rlcd),
              patch.object(laya_train, "_forward", bf16_forward),
              patch.object(laya_train, "random", types.SimpleNamespace(Random=BucketedRandom))):
            history = train_model(model, tokenizer, items, config, torch.device("cuda"),
                                  MAX_LEN, HEAD_MAX_LEN, on_epoch_end=epoch_end)
        if actual_steps != expected_steps or gradient_checks != expected_steps:
            raise RuntimeError(f"incomplete training: steps={actual_steps}, "
                               f"gradient checks={gradient_checks}, expected={expected_steps}")
        if (arm == "E05" and loss_calls != math.ceil(len(items) / micro_batch) * epochs) or (
                arm == "E04" and loss_calls != 0):
            raise RuntimeError(f"unexpected RLCD loss call count: {loss_calls}")
        record({"stage": "training_complete", "arm": arm, "loss_history": history,
                "optimizer_steps": actual_steps, "rlcd_loss_calls": loss_calls,
                "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "elapsed_seconds": time.monotonic() - started})
    except BaseException as exc:
        record({"stage": "failed", "arm": arm, "error": repr(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=("E04", "E05"), required=True)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    run(args.run_dir, args.output, args.arm, micro_batch=args.micro_batch,
        grad_accum=args.grad_accum, epochs=args.epochs)


if __name__ == "__main__":
    main()

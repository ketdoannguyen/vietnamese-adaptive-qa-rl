"""SFT fine-tuning of the Vietnamese extractive reader (`03_...md` §2, §3, §9).

Starts from ``FacebookAI/xlm-roberta-base`` with a **fresh** QA head: the point of using the
plain pretrained checkpoint is that no ViQuAD2-tuned checkpoint (and therefore no test
overlap) enters the pipeline. Training data is ``train_strict`` only — never val, never test.

The loss is written out explicitly in :class:`QATrainer` rather than left to the model's
built-in ``labels`` handling, so the objective (mean of the start and end cross-entropies)
is visible and identical across transformers versions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from src.utils.hashing import file_manifest, sha256_json, sha256_text
from src.utils.io import atomic_write_json
from src.utils.reproducibility import seed_everything

DEFAULT_BASE_MODEL = "FacebookAI/xlm-roberta-base"
WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def load_tokenizer(model_name: str = DEFAULT_BASE_MODEL, revision: str | None = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)


def load_model(model_name: str = DEFAULT_BASE_MODEL, revision: str | None = None):
    from transformers import AutoModelForQuestionAnswering
    return AutoModelForQuestionAnswering.from_pretrained(model_name, revision=revision)


def tokenizer_hash(tokenizer: Any) -> str:
    """Hash the tokenizer's serialized files (vocab + merges), not its in-memory object."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tokenizer.save_pretrained(tmp)
        manifest = file_manifest(Path(tmp).glob("*"), root=tmp)
    return sha256_json(manifest)


def reader_model_hash(model_dir: str | Path) -> str:
    """Hash of the saved reader weights; any change invalidates the QA cache."""
    model_dir = Path(model_dir)
    weights = [model_dir / name for name in WEIGHT_FILES if (model_dir / name).exists()]
    if not weights:
        raise FileNotFoundError(f"no weights ({', '.join(WEIGHT_FILES)}) under {model_dir}")
    return sha256_json(file_manifest(weights, root=model_dir))


class QATrainer:
    """Factory for the Trainer subclass (kept lazy so importing this module needs no model)."""

    @staticmethod
    def build(*, model, args, train_dataset, data_collator=None):
        from transformers import Trainer, default_data_collator

        data_collator = data_collator or default_data_collator

        class _QATrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                start = inputs.pop("start_positions")
                end = inputs.pop("end_positions")
                outputs = model(**inputs)
                loss = (F.cross_entropy(outputs.start_logits, start)
                        + F.cross_entropy(outputs.end_logits, end)) / 2
                return (loss, outputs) if return_outputs else loss

        return _QATrainer(model=model, args=args, train_dataset=train_dataset,
                          data_collator=data_collator)


def total_steps(n_examples: int, *, batch_size: int, grad_accum: int, epochs: int) -> int:
    per_step = max(1, batch_size * grad_accum)
    steps_per_epoch = max(1, n_examples // per_step)
    return steps_per_epoch * epochs


def train(*, model, tokenizer, rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
          epochs: int = 2, lr: float = 2e-5, batch_size: int = 4, grad_accum: int = 8,
          warmup_ratio: float = 0.1, weight_decay: float = 0.01, fp16: bool = False,
          bf16: bool = False, max_steps: int | None = None, seed: int = 42,
          logging_steps: int = 10, gradient_checkpointing: bool = False,
          save_steps: int | None = None) -> dict[str, Any]:
    """Fine-tune ``model`` on pre-built feature rows and save it to ``out_dir``."""
    from datasets import Dataset
    from transformers import TrainingArguments

    if not rows:
        raise ValueError("no reader training features")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    steps = max_steps or total_steps(len(rows), batch_size=batch_size, grad_accum=grad_accum,
                                     epochs=epochs)
    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        max_steps=steps,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        warmup_steps=int(steps * warmup_ratio),
        weight_decay=weight_decay,
        lr_scheduler_type="linear",
        optim="adamw_torch",
        fp16=fp16,
        bf16=bf16,
        seed=seed,
        data_seed=seed,
        logging_steps=logging_steps,
        save_strategy="no" if save_steps is None else "steps",
        save_steps=save_steps or 10 ** 9,
        report_to=[],
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    trainer = QATrainer.build(model=model, args=args,
                              train_dataset=Dataset.from_list([dict(r) for r in rows]))
    result = trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    history = [row for row in trainer.state.log_history if "loss" in row]
    summary = {
        "examples": len(rows),
        "steps": steps,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "warmup_steps": args.warmup_steps,
        "fp16": fp16,
        "bf16": bf16,
        "seed": seed,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "final_loss": history[-1]["loss"] if history else None,
        "loss_history": [{"step": row.get("step"), "loss": row["loss"],
                          "lr": row.get("learning_rate")} for row in history],
        "feature_sha": sha256_json([list(r["input_ids"]) for r in rows[:200]]),
        "library_versions": _versions(),
    }
    atomic_write_json(out_dir / "training_args.json", summary)
    return summary


def overfit_smoke(*, model, tokenizer, rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                  steps: int = 40, lr: float = 5e-5, batch_size: int = 4, seed: int = 42,
                  fp16: bool = False, bf16: bool = False) -> dict[str, Any]:
    """The P3 acceptance gate: 32 examples must show a real loss decrease.

    If the loss does not fall here, the bug is in the data/feature path and RL must not
    start (`03_...md` §8.2). ``fp16``/``bf16`` are mutually exclusive; the caller decides
    (see ``use_bf16`` in the P3 pipeline).
    """
    if len(rows) < 32:
        raise ValueError(f"need 32 examples for the overfit gate, got {len(rows)}")
    subset = [dict(row) for row in rows[:32]]
    summary = train(model=model, tokenizer=tokenizer, rows=subset, out_dir=out_dir,
                    epochs=1, lr=lr, batch_size=batch_size, grad_accum=1, warmup_ratio=0.0,
                    max_steps=steps, seed=seed, logging_steps=1, fp16=fp16, bf16=bf16)
    history = summary["loss_history"]
    first = history[0]["loss"] if history else None
    last = history[-1]["loss"] if history else None
    summary.update({
        "overfit_examples": len(subset),
        "loss_first": first,
        "loss_last": last,
        "loss_decreased": bool(first is not None and last is not None and last < first),
    })
    atomic_write_json(Path(out_dir) / "overfit_smoke.json", summary)
    return summary


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"torch": torch.__version__}
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except ImportError:
        out["transformers"] = None
    return out


def save_model_card(out_dir: str | Path, *, base_model: str, revision: str | None,
                    tokenizer_sha: str, summary: Mapping[str, Any],
                    metrics: Mapping[str, Any] | None = None) -> Path:
    """Model card for the frozen reader checkpoint (`03_...md` §9)."""
    out_dir = Path(out_dir)
    weights_sha = reader_model_hash(out_dir)
    lines = [
        "# QA reader — XLM-R base (Vietnamese extractive QA)",
        "",
        f"- base checkpoint: `{base_model}` (revision `{revision or 'unresolved'}`), MIT",
        "- QA head: freshly initialised (`AutoModelForQuestionAnswering`); no ViQuAD2-tuned",
        "  checkpoint was used, so no test-split overlap enters the reader.",
        f"- training data: `train_strict` only ({summary.get('examples')} windows, "
        f"{summary.get('steps')} steps, lr {summary.get('learning_rate')}, "
        f"effective batch {summary.get('effective_batch')})",
        f"- tokenizer sha256: `{tokenizer_sha}`",
        f"- weights sha256: `{weights_sha}`",
        f"- final train loss: {summary.get('final_loss')}",
        "",
        "## Scores are not probabilities",
        "",
        "`non_null_score` / `null_score` are raw logit sums. They are used only for ranking",
        "candidates; nothing in this project reports them as calibrated correctness.",
        "",
    ]
    if metrics:
        lines += ["## Validation metrics (strict val)", "", "```json",
                  json.dumps(metrics, ensure_ascii=False, indent=2), "```", ""]
    path = out_dir / "model_card.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def config_fingerprint(config: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(config, ensure_ascii=False, sort_keys=True))

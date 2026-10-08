"""Frozen-reader inference (`03_...md` §5, §6): visible evidence -> scored span candidates.

One (question, document) pair at a time, exactly as the environment sees it: the reader never
receives gold, never receives another document's text, and its output is a span sliced out of
the document it was given. The caller decides what to do with the candidates.

:func:`inference_config_sha` covers everything that can change a decoded candidate — model
weights, windowing, the null threshold and the numeric precision — so a QA cache keyed on it
cannot silently go stale when the threshold is retuned.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.qa.features import build_pair_features
from src.qa.postprocess import BestSpan, decode_windows
from src.utils.hashing import sha256_json

DEFAULT_BATCH_SIZE = 16


@dataclass(frozen=True)
class PairInput:
    """One unit of reader work: a question and one visible document."""

    qid: str
    doc_id: str
    question: str
    context: str


def use_fp16(config: Mapping[str, Any]) -> bool:
    """fp16 inference only when the config asks for it and CUDA is actually available."""
    import torch
    return bool(config.get("fp16")) and torch.cuda.is_available()


def inference_config_sha(config: Mapping[str, Any], *, threshold: float, model_hash: str) -> str:
    return sha256_json({
        "reader_model_hash": model_hash,
        "max_length": int(config["max_length"]),
        "doc_stride": int(config["doc_stride"]),
        "max_answer_length": int(config["max_answer_length"]),
        "null_score_diff_threshold": float(threshold),
        "fp16": bool(config.get("fp16", False)),
        "bf16": bool(config.get("bf16", False)),
    })


class ReaderInference:
    """Batched inference over a saved reader checkpoint."""

    def __init__(self, *, model: Any, tokenizer: Any, model_hash: str, tokenizer_sha: str,
                 max_length: int = 384, doc_stride: int = 128, max_answer_length: int = 30,
                 batch_size: int = DEFAULT_BATCH_SIZE, device: str = "cpu",
                 use_fp16: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_hash = model_hash
        self.tokenizer_sha = tokenizer_sha
        self.max_length = int(max_length)
        self.doc_stride = int(doc_stride)
        self.max_answer_length = int(max_answer_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.use_fp16 = use_fp16

    @classmethod
    def load(cls, model_dir: str | Path, *, max_length: int = 384, doc_stride: int = 128,
             max_answer_length: int = 30, batch_size: int = DEFAULT_BATCH_SIZE,
             device: str | None = None, fp16: bool = False) -> "ReaderInference":
        """Load a *saved* reader checkpoint (the frozen artifact, not the base model)."""
        import torch
        from src.qa.train_reader import load_model, load_tokenizer, reader_model_hash, tokenizer_hash

        model_dir = Path(model_dir)
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"{model_dir} is not a saved reader checkpoint")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = load_tokenizer(str(model_dir))
        model = load_model(str(model_dir))
        model.to(device)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, model_hash=reader_model_hash(model_dir),
                   tokenizer_sha=tokenizer_hash(tokenizer), max_length=max_length,
                   doc_stride=doc_stride, max_answer_length=max_answer_length,
                   batch_size=batch_size, device=device, use_fp16=fp16)

    def predict(self, pairs: Sequence[PairInput]) -> dict[tuple[str, str], BestSpan]:
        """Best span per (qid, doc_id) across all sliding windows of that pair."""
        import numpy as np
        import torch

        if not pairs:
            return {}
        features, metas, context_by_doc = [], [], {}
        for pair in pairs:
            window_features, window_metas = build_pair_features(
                pair.question, pair.context, self.tokenizer, qid=pair.qid, doc_id=pair.doc_id,
                max_length=self.max_length, doc_stride=self.doc_stride)
            features.extend(window_features)
            metas.extend(window_metas)
            context_by_doc[pair.doc_id] = pair.context

        start_logits = np.empty((len(features), self.max_length), dtype=np.float32)
        end_logits = np.empty((len(features), self.max_length), dtype=np.float32)
        autocast = self.use_fp16 and self.device.startswith("cuda")
        for first in range(0, len(features), self.batch_size):
            batch = features[first:first + self.batch_size]
            inputs: dict[str, Any] = {
                "input_ids": torch.tensor([f.input_ids for f in batch], device=self.device),
                "attention_mask": torch.tensor([f.attention_mask for f in batch],
                                               device=self.device),
            }
            if batch[0].token_type_ids is not None:
                inputs["token_type_ids"] = torch.tensor([f.token_type_ids for f in batch],
                                                        device=self.device)
            with torch.no_grad():
                if autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)
            rows = len(batch)
            start_logits[first:first + rows] = outputs.start_logits.float().cpu().numpy()
            end_logits[first:first + rows] = outputs.end_logits.float().cpu().numpy()

        return decode_windows(metas, start_logits, end_logits, context_by_doc,
                              max_answer_length=self.max_answer_length)

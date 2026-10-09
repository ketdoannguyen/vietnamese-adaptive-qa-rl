"""P9 demo CLI: one Vietnamese question, one adaptive episode, one cited answer.

    python -m src.inference.cli --question 'Phạm Văn Đồng bắt đầu giữ chức Thủ tướng \
        CHXHCN Việt Nam từ năm nào?' --config configs/final_inference.yaml --show-trace

Rules this module follows (`02_IMPLEMENTATION/07_INFERENCE_DEMO_AND_MODEL_EXPORT.md`):

* **Extractive only.** The answer string is a reader span copied out of a retrieved passage;
  nothing is generated. The deterministic template ``Theo tài liệu [doc_id], đáp án là: …``
  is applied at display time, never stored as the answer.
* **No labels in the demo.** The episode's :class:`~src.rl.reward.EpisodePrivate` is empty (no
  gold answers, no answerability, no reachability), so reward gold and strict test labels
  cannot influence or leak into a demo answer; ``test_strict`` is refused as a demo corpus.
  User corpora come from ``--corpus-jsonl`` and never change the evaluation corpus.
* **The same episode as evaluation.** The loop is :class:`~src.rl.env.AdaptiveQAEnv` with
  :func:`src.pipeline.worlds.env_config`, i.e. the masks, horizon, forced bootstrap and search
  budget of the evaluated arms. Only the *reader* differs: a live checkpoint instead of the
  P3 state cache, because a demo question has no pre-cached states. The policy is applied
  exactly as ``src.rl.reinforce.rollout(greedy=True)`` does (same feature extractor, same
  scaler, same masked argmax).
* **Honest degradation.** A missing frozen artifact stops the run with the path and the stage
  that produces it; a dense-encoder failure degrades to BM25 and says so in the trace; a state
  with no valid span abstains with the spec's exact message. Nothing is ever fabricated.

Action probabilities are always labelled ``xác suất chọn hành động`` ("probability of choosing
the action"), never as a probability that an answer is correct.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from src.data import artifacts
from src.decision.laya_adapter import LayaAdapter
from src.decision.laya_cache import LayaProbCache
from src.decision.schema import ACTION_ORDER
from src.decision.state_text import serialize_state, state_hash
from src.pipeline import train_reader
from src.pipeline import worlds as world_builder
from src.pipeline.cache_reader import read_contract
from src.qa.infer import PairInput, ReaderInference, use_fp16
from src.qa.postprocess import apply_threshold
from src.qa.state_cache import best_state_prediction
from src.retrieval.bm25 import BM25Index
from src.retrieval.dense import DenseBank, DenseEncoder, rank_by_inner_product
from src.rl.env import ANSWER, AdaptiveQAEnv, EpisodeWorld
from src.rl.features import FeatureExtractor, FeatureScaler
from src.rl.policy import MASK_FILL, PolicyNet, TorchScaler, greedy_action
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig
from src.utils.config import add_profile_args, config_from_args, resolve
from src.utils.hashing import sha256_json, sha256_text
from src.utils.io import read_jsonl

#: Exact abstention message from spec 07 §2 — the demo must not paraphrase it.
ABSTAIN_MESSAGE = "Chưa tìm thấy bằng chứng đủ để trả lời trong bộ tài liệu hiện tại."
#: The only label under which action probabilities may be shown (spec 07 §1).
ACTION_PROB_LABEL = "xác suất chọn hành động"
ACTION_PROB_NOTE = ("Đây là xác suất chọn hành động của mô hình quyết định, "
                    "không phải xác suất đáp án đúng.")
CONTRADICTORY_NOTE = ("Chưa có cơ chế giải quyết mâu thuẫn: các trích đoạn mâu thuẫn được "
                      "đánh dấu để người dùng tự kiểm chứng, không tự chọn theo nhãn.")
DISPLAY_TEMPLATE = "Theo tài liệu [{doc_id}], đáp án là: {answer}"

DEFAULT_CORPUS_SPLIT = "val_strict"
#: Splits the demo must never read: strict test labels stay locked behind the final eval.
REFUSED_CORPUS_SPLITS = ("test_strict",)
DEMO_SPLIT = "demo"

#: P6 artifact directory names: spec 07 §4 calls it ``policy_reinforce_best``, while
#: ``train_rl`` currently writes ``rl_policy`` — the first existing candidate wins, and the
#: chosen path is recorded in the trace. ``scaler.json`` is what ``ReinforceTrainer.save``
#: writes today; ``features_scaler.json`` is the spec's name.
POLICY_DIR_NAMES = ("policy_reinforce_best", "rl_policy")
POLICY_FILE = "policy.safetensors"
SCALER_FILES = ("features_scaler.json", "scaler.json")
#: The demo keeps its own Laya probability cache: sharing the P6 cache would let a demo
#: checkpoint mismatch invalidate (delete) the probabilities the RL stage trained against.
DEMO_LAYA_CACHE_DIR = "laya_probs_cache_demo"

DEFAULT_TOP_K = 32
DEFAULT_SNIPPET_CHARS = 300
CORPUS_LIMIT_HINT = "sorted doc_id order"

TIMING_KEYS = ("setup_ms", "retrieval_ms", "reader_ms", "laya_ms", "policy_ms")


class NotReadyError(RuntimeError):
    """A required frozen artifact is missing. The message names path and producing stage."""


# ------------------------------------------------------------------------------- corpus

@dataclass(frozen=True)
class Corpus:
    """Demo corpus: documents plus display titles. Documents carry no labels."""

    docs: Mapping[str, str]
    titles: Mapping[str, str]
    source: str

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self.docs)

    @property
    def sha256(self) -> str:
        return sha256_json({"docs": sorted(self.docs.items())})


def load_corpus(cfg: Mapping[str, Any], *, corpus_jsonl: str | None = None,
                split: str | None = None, limit: int | None = None) -> Corpus:
    """User corpus (``--corpus-jsonl``) or the configured split's documents.

    The configured default reads the processed *documents* (passages) only — never the
    label store — so the default demo cannot see an answer key.
    """
    section = inference_cfg(cfg)
    if corpus_jsonl:
        return _corpus_from_jsonl(Path(corpus_jsonl), limit=limit)
    split = str(split or section["corpus_split"])
    if split in REFUSED_CORPUS_SPLITS:
        raise ValueError(
            f"refusing to use {split!r} as the demo corpus: strict test labels stay locked "
            "behind the final evaluation (spec 07 §3); pass --corpus-jsonl or a validation split")
    return _corpus_from_artifacts(cfg, split, limit=limit)


def _corpus_from_jsonl(path: Path, *, limit: int | None) -> Corpus:
    if not path.is_file():
        raise FileNotFoundError(f"corpus file not found: {path}")
    docs: dict[str, str] = {}
    titles: dict[str, str] = {}
    for line_no, row in enumerate(read_jsonl(path), start=1):
        doc_id = row.get("doc_id", row.get("id"))
        text = row.get("text", row.get("context", row.get("passage")))
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{path}:{line_no}: row needs a non-empty 'doc_id'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{line_no}: row {doc_id!r} needs non-empty 'text'")
        if doc_id in docs:
            raise ValueError(f"{path}:{line_no}: duplicate doc_id {doc_id!r}")
        docs[doc_id] = text
        titles[doc_id] = str(row.get("title") or row.get("title_group") or doc_id)
    if not docs:
        raise ValueError(f"{path}: no documents found")
    return _limited(Corpus(docs=docs, titles=titles, source=f"user corpus {path}"), limit)


def _corpus_from_artifacts(cfg: Mapping[str, Any], split: str, *, limit: int | None) -> Corpus:
    processed = resolve(cfg, "data_processed")
    path = artifacts.docs_path(processed, split)
    rows = artifacts.read_doc_rows(processed, split)
    docs, titles = {}, {}
    for row in rows:
        doc_id = row["doc_id"]
        if doc_id in docs:
            raise ValueError(f"{path}: duplicate doc_id {doc_id!r}")
        docs[doc_id] = row["text"]
        titles[doc_id] = str(row.get("title_group") or doc_id)
    if not docs:
        raise ValueError(f"{path}: no documents found for split {split!r}")
    return _limited(Corpus(docs=docs, titles=titles, source=f"{path} ({split}, {len(docs)} docs)"),
                    limit)


def _limited(corpus: Corpus, limit: int | None) -> Corpus:
    """Cap the corpus by sorted doc_id so a demo run is bounded and reproducible."""
    if limit is None or limit <= 0 or limit >= len(corpus.docs):
        return corpus
    keep = corpus.doc_ids[:int(limit)]
    return Corpus(docs={d: corpus.docs[d] for d in keep},
                  titles={d: corpus.titles[d] for d in keep},
                  source=f"{corpus.source} [first {len(keep)} docs by {CORPUS_LIMIT_HINT}]")


# ------------------------------------------------------------------------------- config

def inference_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``inference:`` section with defaults; no key is required in the config file."""
    section = dict(cfg.get("inference") or {})
    defaults: dict[str, Any] = {
        "corpus_split": DEFAULT_CORPUS_SPLIT, "corpus_jsonl": None, "top_k": DEFAULT_TOP_K,
        "dense": True, "snippet_chars": DEFAULT_SNIPPET_CHARS, "decision_rule": "auto",
        "policy_dir": None, "laya_cache_dir": None,
    }
    for key, value in defaults.items():
        section.setdefault(key, value)
    return section


def _policy_dir(cfg: Mapping[str, Any]) -> Path:
    """The existing P6 checkpoint dir (spec or ``train_rl`` name); spec name if none exists."""
    override = inference_cfg(cfg).get("policy_dir")
    if override:
        return Path(override)
    candidates = [resolve(cfg, "models") / name for name in POLICY_DIR_NAMES]
    for candidate in candidates:
        if (candidate / POLICY_FILE).is_file():
            return candidate
    return candidates[0]  # nothing exists: the error must name the spec's expected path


# ------------------------------------------------------------------------------- session

@dataclass
class Session:
    """Loaded frozen artifacts + demo corpus; question-independent, built once per process."""

    cfg: Mapping[str, Any]
    corpus: Corpus
    reader: Any  # duck-typed ReaderInference: .predict(pairs) -> {(qid, doc_id): BestSpan}
    reader_model_hash: str
    reader_threshold: float
    decision_name: str
    tools: tuple[str, ...] = ("bm25",)
    dense_bank: DenseBank | None = None
    dense_encoder_factory: Callable[[], Any] | None = None
    laya: Any | None = None
    laya_cache: LayaProbCache | None = None
    laya_model_hash: str | None = None
    laya_regime: str | None = None
    policy: PolicyNet | None = None
    scaler: FeatureScaler | None = None
    extractor: FeatureExtractor | None = None
    notes: list[str] = field(default_factory=list)
    paths: dict[str, str] = field(default_factory=dict)
    #: Wall time spent loading the frozen artifacts; 0.0 for a hand-built session.
    setup_ms: float = 0.0

    def decision(self, timings: dict[str, float] | None = None) -> "DecisionMaker":
        laya_fn = self.build_laya_fn(timings)
        if self.policy is not None:
            extractor = self.extractor or FeatureExtractor()
            if self.scaler is not None:
                # Mirrors ReinforceTrainer: the extractor holds the scaler *and* the torch
                # scaler is applied, exactly as evaluate()/rollout(greedy=True) does.
                extractor.scaler = self.scaler
            scaler_t = None if self.scaler is None else TorchScaler(
                np.asarray(self.scaler.mean, dtype=np.float32),
                np.asarray(self.scaler.std, dtype=np.float32), self.scaler.clip)
            return DecisionMaker(name=self.decision_name, net=self.policy, extractor=extractor,
                                 scaler_t=scaler_t, laya_fn=laya_fn)
        if laya_fn is not None:
            return DecisionMaker(name=self.decision_name, laya_fn=laya_fn)
        raise NotReadyError(
            "no decision model loaded: neither the P6 policy nor the P5 Laya checkpoint is "
            "available, and the demo never invents a decision rule")

    def build_laya_fn(self, timings: dict[str, float] | None = None
                      ) -> Callable[[dict], np.ndarray] | None:
        """The live prior closure for one episode, or ``None`` without a loaded adapter."""
        if self.laya is None or self.laya_cache is None:
            return None
        return _laya_fn(self.laya, self.laya_cache, timings)


@dataclass(frozen=True)
class _Artifacts:
    """What :func:`open_session` must load, decided before any weight is read."""

    reader_dir: Path
    contract: Mapping[str, Any]
    policy_dir: Path
    use_policy: bool
    laya_dir: Path | None
    laya_cache_dir: Path | None


def _require_artifacts(cfg: Mapping[str, Any], regime: str | None,
                       paths: dict[str, str]) -> _Artifacts:
    """Check every frozen artifact exists (fail fast) before loading anything heavy."""
    reader_dir, contract = _reader_contract(cfg)
    paths["reader_dir"] = str(reader_dir)
    rule = str(inference_cfg(cfg)["decision_rule"]).lower()
    if rule not in ("auto", "policy", "laya"):
        raise ValueError(f"inference.decision_rule must be auto|policy|laya, got {rule!r}")
    policy_dir = _policy_dir(cfg)
    paths["policy_dir"] = str(policy_dir)
    policy_path = policy_dir / POLICY_FILE
    use_policy = False
    if rule != "laya":
        if policy_path.is_file():
            scaler_path = next((policy_dir / name for name in SCALER_FILES
                                if (policy_dir / name).is_file()), None)
            if scaler_path is None:
                raise NotReadyError(
                    f"{policy_dir}: no feature scaler ({' or '.join(SCALER_FILES)}) — the "
                    "policy must be applied with the scaler it was trained with (P6, "
                    "train_rl); refusing to run unscaled features")
            paths["scaler"] = str(scaler_path)
            use_policy = True
        elif rule == "policy":
            raise NotReadyError(
                f"{policy_path} not found — the frozen RL policy is produced by the P6 stage "
                "(python -m src.pipeline.train_rl)")
        else:
            paths["policy_missing"] = str(policy_path)
    laya_dir = None
    if regime is not None:
        laya_dir = world_builder.laya_checkpoint_dir(cfg, regime)
        paths["laya_dir"] = str(laya_dir)
        if not (laya_dir / "model.safetensors").is_file() or \
                not (laya_dir / "rl_agent_config.json").is_file():
            loss = "soft-ce" if regime == "ce" else "rlcd"
            raise NotReadyError(
                f"{laya_dir}: not a Laya checkpoint (needs model.safetensors + "
                f"rl_agent_config.json) — produced by the P5 stage "
                f"(python -m src.pipeline.train_laya --loss {loss})")
    if not use_policy and laya_dir is None:
        raise NotReadyError(
            "no decision model available: the P6 policy is absent (train it with train_rl) "
            "and the Laya prior is disabled or missing (P5, train_laya); the demo has no "
            "fallback decision rule")
    cache_dir = inference_cfg(cfg).get("laya_cache_dir") or (
        resolve(cfg, "models") / DEMO_LAYA_CACHE_DIR)
    return _Artifacts(reader_dir=reader_dir, contract=contract, policy_dir=policy_dir,
                      use_policy=use_policy, laya_dir=laya_dir,
                      laya_cache_dir=Path(cache_dir) if laya_dir is not None else None)


def open_session(cfg: Mapping[str, Any], *, corpus_jsonl: str | None = None,
                 corpus_split: str | None = None, corpus_limit: int | None = None,
                 laya_regime: str = world_builder.DEFAULT_LAYA_SENTINEL) -> Session:
    """Load every frozen artifact the demo needs, or fail with the path and the stage."""
    started = time.perf_counter()
    section = inference_cfg(cfg)
    notes: list[str] = []
    paths: dict[str, str] = {}
    corpus = load_corpus(cfg, corpus_jsonl=corpus_jsonl, split=corpus_split, limit=corpus_limit)
    regime = world_builder.resolve_laya_regime(cfg, laya_regime)
    required = _require_artifacts(cfg, regime, paths)

    reader_cfg = cfg["reader"]
    reader = ReaderInference.load(
        required.reader_dir, max_length=int(reader_cfg["max_length"]),
        doc_stride=int(reader_cfg["doc_stride"]),
        max_answer_length=int(reader_cfg["max_answer_length"]),
        batch_size=int(reader_cfg["inference_batch_size"]), fp16=use_fp16(reader_cfg))
    if reader.model_hash != required.contract["reader_model_hash"]:
        raise NotReadyError(
            f"{required.reader_dir} does not match its own contract: weights hash "
            f"{reader.model_hash[:16]}… != reader_inference.json "
            f"{str(required.contract['reader_model_hash'])[:16]}…; re-run the P3 stage "
            "(train_reader)")

    policy, scaler = (_load_policy(required.policy_dir) if required.use_policy else (None, None))
    laya, laya_cache, laya_hash = _load_laya(required.laya_dir, required.laya_cache_dir,
                                             paths)
    if laya_cache is not None:
        paths["laya_cache_dir"] = str(laya_cache.cache_dir)

    tools: tuple[str, ...] = ("bm25",)
    dense_bank = None
    dense_factory = None
    if section["dense"]:
        try:
            dense_bank = _build_demo_bank(cfg, corpus)
            dense_factory = lambda: DenseEncoder(  # noqa: E731 - small, local factory
                model_name=cfg["retrieval"]["dense"]["model"],
                batch_size=int(cfg["retrieval"]["dense"]["batch_size"]),
                max_length=int(cfg["retrieval"]["dense"]["max_length"]), device=_device())
            tools = ("bm25", "dense")
            paths["dense_model"] = str(cfg["retrieval"]["dense"]["model"])
        except Exception as exc:  # encoder unavailable offline/OOM: BM25-only, disclosed
            notes.append(f"dense encoder unavailable ({type(exc).__name__}: {exc}); "
                         "the demo ran with BM25 only")
    else:
        notes.append("dense retrieval disabled by the inference config; BM25 only")

    if policy is not None:
        # The artifact directory name is the honest policy name: it says which checkpoint this
        # run actually used, and paths["policy_dir"] gives the full trace.
        base = required.policy_dir.name
        decision_name = base if laya is not None else f"{base}[no_prior]"
    else:
        decision_name = f"laya_{regime}_argmax"
    session = Session(
        cfg=cfg, corpus=corpus, reader=reader, reader_model_hash=reader.model_hash,
        reader_threshold=float(required.contract["null_score_diff_threshold"]),
        decision_name=decision_name, tools=tools, dense_bank=dense_bank,
        dense_encoder_factory=dense_factory, laya=laya, laya_cache=laya_cache,
        laya_model_hash=laya_hash, laya_regime=regime, policy=policy, scaler=scaler,
        extractor=FeatureExtractor() if policy is not None else None, notes=notes, paths=paths,
        setup_ms=(time.perf_counter() - started) * 1000.0)
    return session


def _reader_contract(cfg: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    """The reader checkpoint dir and its P3 contract (``reader_inference.json``).

    ``read_contract`` returns the *contract file* path; loading needs the checkpoint
    directory that holds it (weights + config.json), so return that dir, not the JSON.
    """
    try:
        _, contract = read_contract(cfg)
    except FileNotFoundError as exc:
        raise NotReadyError(
            f"{exc} — the reader checkpoint is produced by the P3 stage "
            "(python -m src.pipeline.train_reader); the demo never fakes an answer "
            "without a loaded reader") from exc
    return train_reader.reader_dir(cfg), contract


def _load_policy(policy_dir: Path) -> tuple[PolicyNet, FeatureScaler]:
    """Load the frozen P6 head; the scaler must match the feature layout this build uses."""
    policy_path = policy_dir / POLICY_FILE
    try:
        net = PolicyNet.load(policy_path)
    except FileNotFoundError as exc:
        raise NotReadyError(
            f"{policy_path}: checkpoint config sidecar missing ({exc}) — produced by the P6 "
            "stage (train_rl)") from exc
    scaler_path = next((policy_dir / name for name in SCALER_FILES
                        if (policy_dir / name).is_file()), None)
    if scaler_path is None:  # pragma: no cover - _require_artifacts already checked
        raise NotReadyError(f"{policy_dir}: no feature scaler ({' or '.join(SCALER_FILES)})")
    scaler = FeatureScaler.load(scaler_path)
    if len(scaler.mean) != FeatureExtractor().dim:
        raise NotReadyError(
            f"{scaler_path}: scaler has {len(scaler.mean)} features, the extractor has "
            f"{FeatureExtractor().dim}; the policy belongs to another feature layout")
    return net, scaler


def _load_laya(checkpoint: Path | None, cache_dir: Path | None, paths: dict[str, str]
               ) -> tuple[LayaAdapter | None, LayaProbCache | None, str | None]:
    """Load the frozen P5 checkpoint and bind it to its own demo probability cache."""
    if checkpoint is None or cache_dir is None:
        return None, None, None
    adapter = LayaAdapter.load(checkpoint)
    paths["laya_cache_dir"] = str(cache_dir)
    return adapter, LayaProbCache(cache_dir), adapter.model_hash()


def _build_demo_bank(cfg: Mapping[str, Any], corpus: Corpus) -> DenseBank:
    """Encode the demo corpus with the configured E5 checkpoint (same as P2)."""
    ret = cfg["retrieval"]["dense"]
    encoder = DenseEncoder(model_name=ret["model"], batch_size=int(ret["batch_size"]),
                           max_length=int(ret["max_length"]), device=_device())
    try:
        return DenseBank.build({d: corpus.docs[d] for d in corpus.doc_ids}, encoder)
    finally:
        encoder.unload()  # never leave the encoder resident next to the reader/Laya


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        return "cpu"


# ----------------------------------------------------------------------- decision layer

class DecisionMaker:
    """Masked action choice: the P6 policy when loaded, else the Laya prior's masked argmax."""

    def __init__(self, *, name: str, laya_fn: Callable[[dict], np.ndarray] | None = None,
                 net: PolicyNet | None = None, extractor: FeatureExtractor | None = None,
                 scaler_t: Any | None = None) -> None:
        if net is None and laya_fn is None:
            raise ValueError("DecisionMaker needs a policy or a Laya prior")
        self.name = name
        self.laya_fn = laya_fn
        self.net = net
        self.extractor = extractor
        self.scaler_t = scaler_t

    def choose(self, obs: Mapping[str, Any], timings: dict[str, float]
               ) -> tuple[int, np.ndarray, np.ndarray | None]:
        """``(action, action_probabilities, laya_probabilities)`` under the env's mask."""
        mask = np.asarray(obs["action_mask"], dtype=bool)
        laya = self.laya_fn(obs) if self.laya_fn is not None else None
        if self.net is None:
            # Masking is the environment's job: an illegal action gets no probability mass.
            probs = np.where(mask, np.asarray(laya, dtype=np.float64), 0.0)
            total = float(probs.sum())
            probs = probs / total if total > 0 else mask.astype(np.float64) / float(mask.sum())
            return int(np.argmax(probs)), probs, laya
        assert self.extractor is not None
        raw = self.extractor.vector(dict(obs), laya)
        x = torch.as_tensor(np.asarray(raw, dtype=np.float32)).unsqueeze(0)
        if self.scaler_t is not None:
            x = self.scaler_t(x)
        mask_t = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        started = time.perf_counter()
        with torch.no_grad():
            logits, _ = self.net(x)
            probs_t = torch.softmax(logits.masked_fill(~mask_t, MASK_FILL), dim=-1)
        timings["policy_ms"] += (time.perf_counter() - started) * 1000.0
        probs = probs_t.squeeze(0).numpy().astype(np.float64)
        return greedy_action(logits.squeeze(0), mask_t.squeeze(0)), probs, laya


def _laya_fn(adapter: Any, cache: LayaProbCache, timings: dict[str, float] | None
             ) -> Callable[[dict], np.ndarray]:
    """Live Laya prior for one episode: ``serialize_state`` → ``state_hash`` → cache/adapter."""
    stats = timings if timings is not None else {}

    def lookup(obs: dict) -> np.ndarray:
        text = serialize_state(obs)
        qid, key = str(obs["qid"]), state_hash(text)
        started = time.perf_counter()
        hit = cache.lookup(qid, key)
        if hit is None:
            hit = cache.get_or_compute(adapter, [(qid, key, text)])[(qid, key)]
        stats["laya_ms"] = stats.get("laya_ms", 0.0) + (time.perf_counter() - started) * 1000.0
        return hit

    return lookup


# ------------------------------------------------------------------------------- reader

class LiveReader:
    """Memoized live reader over the demo corpus: state → ReaderPrediction.

    Uses the same primitives the P3 cache was built with (``ReaderInference.predict`` →
    ``apply_threshold`` → ``best_state_prediction``), so one visible-document state yields the
    same candidate the RL stage would read from the cache. A miss cannot happen here: every
    state is computed on demand from the loaded checkpoint.
    """

    def __init__(self, session: Session, question: str, qid: str,
                 timings: dict[str, float]) -> None:
        self.session = session
        self.question = question
        self.qid = qid
        self.timings = timings
        self.per_doc: dict[tuple[str, ...], dict[str, Any]] = {}
        self._states: dict[tuple[str, ...], Any] = {}

    def __call__(self, visible_doc_ids: tuple[str, ...]) -> Any:
        key = tuple(visible_doc_ids)
        if key not in self._states:
            pairs = [PairInput(qid=self.qid, doc_id=doc_id, question=self.question,
                               context=self.session.corpus.docs[doc_id]) for doc_id in key]
            started = time.perf_counter()
            spans = self.session.reader.predict(pairs)
            self.timings["reader_ms"] += (time.perf_counter() - started) * 1000.0
            predictions = apply_threshold(
                [spans[(self.qid, doc_id)] for doc_id in key],
                threshold=self.session.reader_threshold,
                reader_model_hash=self.session.reader_model_hash,
                context_by_doc=self.session.corpus.docs)
            self._states[key] = best_state_prediction(list(predictions.values()))
            self.per_doc[key] = dict(predictions)
        return self._states[key]


# ----------------------------------------------------------------------------- retrieval

def rank_question(session: Session, question: str, timings: dict[str, float]
                  ) -> tuple[dict[str, list[str]], dict[str, dict[str, float]]]:
    """Live BM25 (+ optional dense) ranking of the demo corpus for one question.

    The same primitives as ``src.retrieval.engine.rank_scenario``, with the corpus restricted
    to the demo documents; so the demo walks exactly the retrieval tree the environment walks.
    """
    started = time.perf_counter()
    ret = session.cfg["retrieval"]
    top_n = int(inference_cfg(session.cfg)["top_k"])
    index = BM25Index.build({doc_id: session.corpus.docs[doc_id] for doc_id in session.corpus.doc_ids},
                            k1=float(ret["bm25"]["k1"]), b=float(ret["bm25"]["b"]))
    pairs = index.rank(question, top_n=top_n)
    rankings: dict[str, list[str]] = {"bm25": [doc_id for doc_id, _ in pairs]}
    scores: dict[str, dict[str, float]] = {"bm25": {doc_id: float(s) for doc_id, s in pairs}}
    if "dense" in session.tools:
        assert session.dense_bank is not None and session.dense_encoder_factory is not None
        encoder = session.dense_encoder_factory()
        try:
            query = encoder.encode_queries([question])
        finally:
            encoder.unload()
        dense = rank_by_inner_product(query, session.dense_bank.vectors,
                                      session.dense_bank.doc_ids)[:top_n]
        rankings["dense"] = [doc_id for doc_id, _ in dense]
        scores["dense"] = {doc_id: float(s) for doc_id, s in dense}
    timings["retrieval_ms"] += (time.perf_counter() - started) * 1000.0
    return rankings, scores


# ------------------------------------------------------------------------------ episode

def answer_question(session: Session, question: str) -> dict[str, Any]:
    """Run one adaptive episode and return the result JSON of spec 07 §1."""
    question = " ".join(str(question).split())
    if not question:
        raise ValueError("empty question")
    started = time.perf_counter()
    timings = {key: 0.0 for key in TIMING_KEYS}
    qid = "demo:" + sha256_text(question)[:12]
    decision = session.decision(timings)
    env_config = world_builder.env_config(session.cfg)
    reward = RewardCalculator(RewardConfig.from_dict(session.cfg))

    rankings, scores = rank_question(session, question, timings)
    reader = LiveReader(session, question, qid, timings)
    world = EpisodeWorld(
        qid=qid, question=question, split=DEMO_SPLIT, scenario_id=f"demo:{qid}",
        corpus=dict(session.corpus.docs), rankings=rankings,
        # No labels anywhere in the demo: empty private side, no gold, no answerability.
        private=EpisodePrivate(qid=qid, answer_texts=(), is_impossible=False,
                               gold_doc_id=None, gold_in_corpus=False, gold_reachable=None),
        reader=reader)
    # The environment reads real per-tool scores off the world when present; otherwise it
    # falls back to a rank proxy the trained features never saw.
    world._scores_bm25 = scores.get("bm25", {})
    world._scores_dense = scores.get("dense", {})
    env = AdaptiveQAEnv([world], reward, env_config)

    trace: list[dict[str, Any]] = [_setup_trace(session, env_config, reward, rankings)]
    obs, _ = env.reset(seed=0, options={"qid": qid})
    last_action, done = None, False
    while not done:
        action, probs, _ = decision.choose(obs, timings)
        entry = _step_entry(obs, action, probs, reward)
        # The step reward the environment returns is discarded: with an empty private side
        # there is no gold to score against, and a fabricated utility must never be shown.
        obs, _, terminated, truncated, _ = env.step(action)
        entry["cost_so_far"] = round(_cost_so_far(env), 6)
        entry["terminal"] = bool(terminated or truncated)
        trace.append(entry)
        last_action, done = action, bool(terminated or truncated)

    seen = tuple(env.seen_doc_ids)
    contradictions = _contradictions(reader.per_doc.get(seen, {}))
    trace[-1]["contradictory_evidence"] = contradictions
    if contradictions:
        trace[-1]["contradictory_evidence_note"] = CONTRADICTORY_NOTE
    candidate = env.candidate
    answered = (last_action == ANSWER and candidate.answer_text is not None
                and bool(candidate.provenance_ok))
    citations = _citations(session, seen, candidate if answered else None)
    timings["setup_ms"] = session.setup_ms
    timings_out = {"total_ms": round((time.perf_counter() - started) * 1000.0, 3)}
    timings_out.update({key: round(value, 3) for key, value in timings.items()})
    return {
        "question": question,
        "answer": candidate.answer_text if answered else None,
        "abstained": not answered,
        "reason_short": (f"Đáp án trích từ tài liệu {candidate.doc_id}."
                         if answered else ABSTAIN_MESSAGE),
        "citations": citations,
        "retrieval_steps": int(env.n_bm25 + env.n_dense),
        "tools_used": [tool for tool in ("bm25", "dense") if _uses(env, tool)],
        "total_cost": round(_cost_so_far(env), 6),
        "policy_name": session.decision_name,
        "model_hash": session.reader_model_hash,
        "stage_trace": trace,
        "timings": timings_out,
    }


def run_inference(cfg: Mapping[str, Any], question: str, *, corpus_jsonl: str | None = None,
                  corpus_split: str | None = None, corpus_limit: int | None = None,
                  laya_regime: str = world_builder.DEFAULT_LAYA_SENTINEL,
                  session: Session | None = None) -> dict[str, Any]:
    """``open_session`` + ``answer_question``: the whole demo path in one call."""
    session = session or open_session(cfg, corpus_jsonl=corpus_jsonl, corpus_split=corpus_split,
                                      corpus_limit=corpus_limit, laya_regime=laya_regime)
    return answer_question(session, question)


def _setup_trace(session: Session, env_config: Any, reward: RewardCalculator,
                 rankings: Mapping[str, list[str]]) -> dict[str, Any]:
    return {
        "timestep": -1, "event": "setup", "policy_name": session.decision_name,
        "action_probabilities_label": ACTION_PROB_LABEL,
        "corpus": {"source": session.corpus.source, "sha256": session.corpus.sha256,
                   "n_docs": len(session.corpus.docs)},
        "tools_available": list(session.tools),
        "reader_model_hash": session.reader_model_hash,
        "reader_threshold": session.reader_threshold,
        "laya_regime": session.laya_regime, "laya_model_hash": session.laya_model_hash,
        "env_config": asdict(env_config),
        "reward_config": reward.config.to_dict(),
        "reward_note": ("demo không có nhãn: không tính utility/reward, chỉ tính chi phí "
                        "truy hồi công khai theo reward_config"),
        "rankings": {tool: {"n": len(ids), "top": list(ids[:5])}
                     for tool, ids in sorted(rankings.items())},
        "artifacts": dict(session.paths),
        "notes": list(session.notes),
    }


def _step_entry(obs: Mapping[str, Any], action: int, probs: np.ndarray,
                reward: RewardCalculator) -> dict[str, Any]:
    mask = np.asarray(obs["action_mask"], dtype=bool)
    candidate = obs["candidate"]
    return {
        "timestep": int(obs["timestep"]), "event": "step",
        "action": ACTION_ORDER[action],
        "action_probs": {name: round(float(probs[i]), 6)
                         for i, name in enumerate(ACTION_ORDER)},
        "action_probs_label": ACTION_PROB_LABEL,
        "action_mask": {name: bool(mask[i]) for i, name in enumerate(ACTION_ORDER)},
        "seen_doc_ids": list(obs["seen_doc_ids"]),
        "candidate": {"doc_id": candidate["doc_id"], "text": candidate["text"],
                      "non_null_score": round(float(candidate["non_null_score"]), 6),
                      "null_score": round(float(candidate["null_score"]), 6),
                      "null_margin": round(float(candidate["null_margin"]), 6),
                      "provenance_ok": bool(candidate["provenance_ok"])},
        "retrieval": {key: round(float(value), 6)
                      for key, value in obs["retrieval"].items()},
        "optional_used": int(obs["optional_used"]),
        "step_cost": round(float(reward.search_cost(ACTION_ORDER[action])), 6),
    }


def _cost_so_far(env: AdaptiveQAEnv) -> float:
    """Retrieval cost charged so far, exactly as the environment charges it."""
    reward = env.reward
    optional_bm25 = max(0, env.n_bm25 - 1)  # the first BM25 call is the forced bootstrap
    total = (optional_bm25 * reward.search_cost("SEARCH_BM25")
             + env.n_dense * reward.search_cost("SEARCH_DENSE"))
    if env.config.bootstrap_tool == "bm25":
        total += reward.config.bootstrap_bm25_cost
    else:
        total += reward.config.dense_cost
    return float(total)


def _uses(env: AdaptiveQAEnv, tool: str) -> bool:
    return bool(env.n_bm25 if tool == "bm25" else env.n_dense)


def _contradictions(predictions: Mapping[Any, Any]) -> list[dict[str, str]]:
    """Distinct non-null spans among the visible docs, flagged, never resolved silently."""
    seen: dict[str, str] = {}
    for prediction in predictions.values():
        if prediction.answer_text is None or not prediction.provenance_ok:
            continue
        seen.setdefault(" ".join(str(prediction.answer_text).split()), prediction.doc_id)
    if len(seen) < 2:
        return []
    return [{"doc_id": doc_id, "text": text} for text, doc_id in sorted(seen.items())]


def _citations(session: Session, seen: Sequence[str], candidate: Any | None
               ) -> list[dict[str, str]]:
    """The answered span's source, or every seen passage when abstaining."""
    width = int(inference_cfg(session.cfg)["snippet_chars"])
    if candidate is not None and candidate.answer_text is not None:
        docs = [(candidate.doc_id, candidate.start_char, candidate.end_char)]
    else:
        docs = [(doc_id, None, None) for doc_id in seen]
    out = []
    for doc_id, start, end in docs:
        text = session.corpus.docs.get(doc_id)
        if text is None:
            continue
        out.append({"doc_id": doc_id, "snippet": _snippet(text, start, end, width),
                    "title": session.corpus.titles.get(doc_id, doc_id)})
    return out


def _snippet(text: str, start: int | None, end: int | None, width: int) -> str:
    if start is None or end is None:
        collapsed = " ".join(text.split())
        return collapsed[:width] + ("…" if len(collapsed) > width else "")
    half = max(0, (width - max(0, end - start)) // 2)
    low, high = max(0, start - half), min(len(text), end + half)
    out = " ".join(text[low:high].split())
    return f"{'…' if low > 0 else ''}{out}{'…' if high < len(text) else ''}"


# ------------------------------------------------------------------------------ display

def format_result(result: Mapping[str, Any], *, show_trace: bool = False) -> str:
    """Human-readable rendering; the JSON contract stays untouched."""
    lines = [f"Câu hỏi: {result['question']}"]
    if result["abstained"]:
        lines.append(f"Trả lời: {ABSTAIN_MESSAGE}")
    else:
        doc_id = result["citations"][0]["doc_id"] if result["citations"] else "(không rõ)"
        lines.append(DISPLAY_TEMPLATE.format(doc_id=doc_id, answer=result["answer"]))
    lines.append(f"Trích dẫn ({len(result['citations'])}):")
    for citation in result["citations"]:
        lines.append(f"  - {citation['doc_id']} | {citation['title']}: \"{citation['snippet']}\"")
    lines.append(f"Mô hình quyết định: {result['policy_name']} | "
                 f"model_hash (reader): {str(result['model_hash'])[:16]}…")
    tools = ", ".join(result["tools_used"]) or "(không truy hồi)"
    lines.append(f"Truy hồi: {result['retrieval_steps']} bước | công cụ: {tools} | "
                 f"chi phí: {result['total_cost']}")
    if show_trace:
        lines.append("Truy vết từng bước:")
        for entry in result["stage_trace"]:
            if entry["event"] == "setup":
                lines.append(f"  [setup] {entry['corpus']['source']} | "
                             f"{entry['corpus']['n_docs']} docs | {entry['policy_name']}")
                lines.extend(f"      ghi chú: {note}" for note in entry["notes"])
                continue
            probs = " ".join(f"{name}={value:.3f}"
                             for name, value in entry["action_probs"].items())
            lines.append(f"  t={entry['timestep']} {entry['action']} | "
                         f"{ACTION_PROB_LABEL}: {probs}")
            candidate = entry["candidate"]
            shown = candidate["text"] if candidate["text"] else "(không có ứng viên)"
            lines.append(f"      đã xem={entry['seen_doc_ids']} | ứng viên: {shown} | "
                         f"hợp lệ: {candidate['provenance_ok']} | "
                         f"chi phí bước: {entry['step_cost']}")
            for item in entry.get("contradictory_evidence", []):
                lines.append(f"      mâu thuẫn: {item['doc_id']} → {item['text']!r}")
        lines.append(f"  Ghi chú: {ACTION_PROB_NOTE}")
    timings = " ".join(f"{key}={value:.1f}ms" for key, value in result["timings"].items())
    lines.append(f"Thời gian: {timings}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------- main

def _questions_from_args(args: argparse.Namespace) -> list[str]:
    questions = []
    if args.question:
        questions += [line for line in str(args.question).splitlines() if line.strip()]
    if args.question_file:
        path = Path(args.question_file)
        if not path.is_file():
            raise FileNotFoundError(f"question file not found: {path}")
        questions += [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
    if not questions:
        raise ValueError("nothing to answer: pass --question or --question-file")
    return questions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.inference.cli",
        description="Demo: one Vietnamese question through the adaptive QA agent "
                    "(extractive answer with citations, or an honest abstention)")
    add_profile_args(parser)
    parser.add_argument("--question", help="one question (multi-line input is split)")
    parser.add_argument("--question-file", help="file with one question per line")
    parser.add_argument("--corpus-jsonl",
                        help="JSONL demo corpus ({doc_id, text, title?}); defaults to the "
                             "configured split's processed documents")
    parser.add_argument("--corpus-split", help=f"corpus split (default "
                                               f"{DEFAULT_CORPUS_SPLIT}; {REFUSED_CORPUS_SPLITS[0]} refused)")
    parser.add_argument("--corpus-limit", type=int,
                        help="cap the demo corpus (sorted doc_id) for a quick run")
    parser.add_argument("--laya-regime", default=world_builder.DEFAULT_LAYA_SENTINEL,
                        choices=[world_builder.DEFAULT_LAYA_SENTINEL, "ce", "rlcd", "none"],
                        help="Laya prior: __default__ uses cfg['laya']['frozen'], none = no prior")
    parser.add_argument("--show-trace", action="store_true", help="print the per-step trace")
    parser.add_argument("--json", action="store_true", help="print the result JSON only")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    try:
        questions = _questions_from_args(args)
        session = open_session(cfg, corpus_jsonl=args.corpus_jsonl,
                               corpus_split=args.corpus_split, corpus_limit=args.corpus_limit,
                               laya_regime=args.laya_regime)
    except (NotReadyError, FileNotFoundError, ValueError) as exc:
        print(f"LỖI: {exc}", file=sys.stderr)
        return 3
    for question in questions:
        try:
            result = answer_question(session, question)
        except NotReadyError as exc:
            print(f"LỖI: {exc}", file=sys.stderr)
            return 3
        # ``--json`` prints one result per line (JSONL), so a question file stays pipeable.
        print(json.dumps(result, ensure_ascii=False) if args.json
              else format_result(result, show_trace=args.show_trace))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

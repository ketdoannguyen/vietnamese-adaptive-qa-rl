#!/usr/bin/env python
"""P0 read-only audit: how much of each teacher state survives each backend's real tokenizer.

Source plan §P0 (`scripts/audit_decision_inputs.py`): measure input/head/token retention
with each backend's real tokenizer, stratified by depth/action, and write one JSON audit
under an experiment namespace. Run on v2 as the baseline; rerun on v3 at P4.

What is measured, from real package APIs (no model weights, no forward pass, CPU only):

* **Laya** (``laya==0.4.0``, baseline ``.venv``): the exact sequence the checkpoint builds —
  ``laya.train.to_internal`` -> ``laya.common.build_head`` -> the state clamp
  ``room = max_len - head_tokens - 1`` -> ``laya.common.encode_text``. The checkpoint's own
  ``rl_agent_config.json`` supplies ``max_len`` / ``head_max_len``. The room formula is
  re-verified at runtime against ``build_sequence(return_truncation_stats=True)``, so a
  layout change in the package fails the audit instead of silently skewing numbers.
* **GLiNER** (``gliner2==2.0.0``, ``.venv-gliner``): the exact classification input path the
  adapter uses — ``load_extractor_tokenizer`` -> ``SchemaTransformer`` ->
  ``compile_schema`` -> ``collate_fn_inference``. ``max_len`` here is a **word** cap (words,
  not subwords); the checkpoint config's 4096 is the trainer cap, while the adapter's
  inference call passes ``max_len=None``. The classification path applies **no subword
  truncation**; that is declared in the report, not hidden.

Both backends share one pure core: segment spans from the v2 line layout, unit<->char
mapping, retention flags, stratification, and the JSON report. Only the thin adapters touch
a real tokenizer, so the unit tests run against fake offsets and fake processors.

Read-only guarantees: rows whose ``meta.split`` is not ``train_strict``/``val_strict`` are
refused (test labels never enter the audit); output goes to
``<experiments-root>/<namespace>/audit/decision_inputs_<backend>_<version>.json``; an existing
file is never overwritten; no network, no training, no writes outside that path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision.schema import ACTION_CRITERIA, ACTION_ORDER, INSTRUCTIONS, build_questions_field
from src.utils.hashing import canonical_json, sha256_file, sha256_text
from src.utils.io import atomic_write_json, read_jsonl
from src.utils.runs import now_iso

#: Mirrors src.decision.laya_adapter.QUESTION_ID and src.decision.gliner_adapter.QUESTION_ID.
QUESTION_ID = "next_action"

DEFAULT_DATA = ROOT / "data/processed/laya_train.jsonl"
DEFAULT_EXPERIMENTS_ROOT = ROOT / "outputs/experiments"
DEFAULT_LAYA_CHECKPOINT = ROOT / "outputs/models/laya_ce"

#: Only these teacher splits may be audited; ``test_strict`` is refused at the row level.
ALLOWED_SPLITS = ("train_strict", "val_strict")

#: v2 line headers (`src.decision.state_text.serialize_state`). ``retrieval`` and ``history``
#: only bound the blocks around the audited segments; they are not scored themselves.
SEGMENT_HEADERS = {
    "evidence": "Bằng chứng đã xem:",
    "candidate": "Ứng viên trả lời:",
    "retrieval": "Truy hồi:",
    "history": "Hành động đã thực hiện:",
    "budget": "Số lượt tìm đã dùng:",
    "action_mask": "Hành động hợp lệ:",
}

#: Segments whose retention the report declares (plan §4.3 and §P4 pass criteria).
AUDITED_SEGMENTS = ("evidence", "candidate", "budget", "action_mask")

_NAMESPACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# --------------------------------------------------------------------------------- rows


@dataclass(frozen=True)
class Row:
    """One teacher row reduced to what the audit reads (no gold, no teacher metadata)."""

    state: str
    qid: str
    split: str
    depth: int
    action: str
    state_text_version: str = "v2"
    questions: Mapping[str, Any] | None = None


def load_rows(paths: Sequence[Path]) -> list[Row]:
    """Load train/val teacher rows; any other split is an error (no test labels).

    Every structural assumption the audit relies on is checked here so a malformed file
    fails at load time rather than producing numbers for the wrong text.
    """
    rows: list[Row] = []
    for path in paths:
        for raw in read_jsonl(path):
            meta = raw.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"{path}: row has no meta object")
            split = meta.get("split")
            if split not in ALLOWED_SPLITS:
                raise ValueError(f"{path}: meta.split {split!r} is not one of {ALLOWED_SPLITS}; "
                                 f"test labels never enter this audit")
            state = raw.get("state")
            if not isinstance(state, str) or not state.strip():
                raise ValueError(f"{path}: row {meta.get('qid')!r} has no usable state text")
            depth = meta.get("depth")
            if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
                raise ValueError(f"{path}: row {meta.get('qid')!r} meta.depth must be a "
                                 f"non-negative int, got {depth!r}")
            action = meta.get("expected")
            if action is None:
                expected = raw.get("expected")
                action = expected.get("next_action") if isinstance(expected, dict) else None
            if action not in ACTION_ORDER:
                raise ValueError(f"{path}: row {meta.get('qid')!r} action {action!r} is not "
                                 f"one of {ACTION_ORDER}")
            version = meta.get("state_text_version")
            if not isinstance(version, str) or not version:
                raise ValueError(f"{path}: row {meta.get('qid')!r} has no meta.state_text_version")
            rows.append(Row(state=state, qid=str(meta.get("qid", "")), split=split, depth=depth,
                            action=action, state_text_version=version,
                            questions=raw.get("questions")))
    return rows


def uniform_state_text_version(rows: Sequence[Row]) -> str:
    """The single state-text version in the audit; mixed files are an error."""
    versions = sorted({row.state_text_version for row in rows})
    if len(versions) != 1:
        raise ValueError(f"mixed state_text_version values across rows: {versions}")
    return versions[0]


def uniform_questions(rows: Sequence[Row]) -> Mapping[str, Any]:
    """The single questions field in the audit; a mixed field would change the head tokens."""
    seen: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row.questions is not None:
            seen.setdefault(canonical_json(row.questions), row.questions)
    if len(seen) > 1:
        raise ValueError("rows carry different questions fields; head tokens would vary")
    return next(iter(seen.values())) if seen else build_questions_field()


def validate_laya_questions(questions: Mapping[str, Any]) -> None:
    if canonical_json(questions) != canonical_json(build_questions_field()):
        raise ValueError("teacher questions differ from runtime Laya schema; audit head invalid")


def sample_rows(rows: Sequence[Row], limit: int | None) -> tuple[list[Row], int]:
    """Deterministic stride sample of at most ``limit`` rows; returns ``(rows, stride)``."""
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be positive, got {limit}")
    if limit is None or limit >= len(rows):
        return list(rows), 1
    stride = math.ceil(len(rows) / limit)
    return list(rows[::stride]), stride


# ------------------------------------------------------------------- pure measurement core


def _lines_with_offsets(text: str) -> Iterator[tuple[int, str]]:
    offset = 0
    for line in text.split("\n"):
        yield offset, line
        offset += len(line) + 1


def segment_spans(text: str) -> dict[str, tuple[int, int] | None]:
    """Char spans of the audited v2 segments, keyed by name; ``None`` when absent.

    Headers are matched at line starts, so evidence prose that happens to contain a header
    phrase mid-line cannot fake a boundary. Each block ends at the next known header line
    (any of them), which keeps the parse correct for a reordered v3 layout too.
    """
    starts: dict[str, int] = {}
    for offset, line in _lines_with_offsets(text):
        for name, header in SEGMENT_HEADERS.items():
            if line.startswith(header):
                if name in starts:
                    raise ValueError(f"duplicate line-start header for {name}: {header}")
                starts[name] = offset
    ordered = sorted(starts.values())
    spans: dict[str, tuple[int, int] | None] = {}
    for name in AUDITED_SEGMENTS:
        start = starts.get(name)
        if start is None:
            spans[name] = None
            continue
        following = [position for position in ordered if position > start]
        spans[name] = (start, min(following) if following else len(text))
    return spans


def token_span(offsets: Sequence[tuple[int, int]], start: int, end: int) -> tuple[int, int] | None:
    """``(first, last)`` unit indices overlapping ``[start, end)``; ``None`` when empty."""
    if end <= start:
        return None
    first: int | None = None
    last: int | None = None
    for index, (unit_start, unit_end) in enumerate(offsets):
        if unit_end > start and unit_start < end:
            if first is None:
                first = index
            last = index
    return None if first is None else (first, last)


def audit_row(row: Row, offsets: Sequence[tuple[int, int]], *, retained_units: int,
              head_units: int, full_subwords: int, retained_subwords: int,
              unit_kind: str) -> dict:
    """One row's retention record: unit counts, subword counts, per-segment flags.

    ``retained_units`` is how many leading units (subwords for Laya, words for GLiNER) the
    encoder actually receives; a segment is *retained* when its first unit is inside that
    window and *fully retained* when its last unit is too. Evidence char coverage is the
    share of the segment's characters inside the window, measured from real offsets.
    """
    spans = segment_spans(row.state)
    last_retained_char = offsets[retained_units - 1][1] if 0 < retained_units <= len(offsets) else 0
    segments: dict[str, dict] = {}
    for name in AUDITED_SEGMENTS:
        span = spans[name]
        if span is None:
            segments[name] = {"present": False, "retained": False, "fully_retained": False,
                              "char_coverage": 0.0}
            continue
        start, end = span
        hit = token_span(offsets, start, end)
        first, last = hit if hit is not None else (None, None)
        width = end - start
        coverage = 0.0 if width <= 0 else min(max(last_retained_char - start, 0), width) / width
        segments[name] = {"present": True,
                          "retained": first is not None and first < retained_units,
                          "fully_retained": last is not None and last < retained_units,
                          "char_coverage": coverage}
    return {"qid": row.qid, "split": row.split, "depth": row.depth, "action": row.action,
            "unit_kind": unit_kind, "full_units": len(offsets), "retained_units": retained_units,
            "head_units": head_units, "full_subwords": full_subwords,
            "retained_subwords": retained_subwords, "segments": segments}


def measure_laya_rows(rows: Sequence[Row], *, encode, head_tokens: int,
                      max_len: int) -> list[dict]:
    """Laya retention from a real encode function; ``encode(state)`` -> ids + offset_mapping.

    ``build_sequence`` clamps the state to ``room = max_len - len(head_ids) - 1`` and then to
    ``ids[:max_len]``; since ``head + room + 1 == max_len`` the clamp is exactly
    ``min(len(state_ids), room)``, which is what this computes (and ``run_laya`` verifies
    against ``build_sequence`` itself).
    """
    room = max(0, max_len - head_tokens - 1)
    metrics: list[dict] = []
    for row in rows:
        encoded = encode(row.state)
        ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        retained = min(len(ids), room)
        metrics.append(audit_row(row, offsets, retained_units=retained, head_units=head_tokens,
                                 full_subwords=head_tokens + len(ids) + 1,
                                 retained_subwords=head_tokens + retained + 1,
                                 unit_kind="subword"))
    return metrics


def normalize_gliner_text(text: str) -> str:
    """The punctuation mutation ``SchemaTransformer._collate_batch`` applies before splitting."""
    if not text:
        return "."
    if not text.endswith((".", "!", "?")):
        return text + "."
    return text


def _gliner_head_subwords(mappings: Sequence[tuple]) -> int:
    """Subwords before the first text word: schema tokens + ``[SEP_TEXT]``."""
    for index, (segment_type, _orig_index, _schema_index) in enumerate(mappings):
        if segment_type == "text":
            return index
    return len(mappings)


def measure_gliner_rows(rows: Sequence[Row], *, processor, model_schema: Any,
                        word_cap: int | None, batch_size: int = 128) -> list[dict]:
    """GLiNER retention from the real processor; units are words, subwords are declared.

    Mirrors ``ClassificationScorer.batch_score``: ``collate_fn_inference(rows, max_len=...)``
    where the cap counts *words* after splitting. The full batch runs uncapped to get the
    true word/subword totals; a row whose cap binds is re-collated alone to measure the
    capped subword count. ``error_policy="raise"`` is stricter than the adapter's default:
    a record that would silently become a dummy in production fails the audit instead.
    """
    metrics: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start:start + batch_size])
        texts = [normalize_gliner_text(row.state) for row in chunk]
        batch = processor.collate_fn_inference([(text, model_schema) for text in texts],
                                               max_len=None, error_policy="raise")
        for index, row in enumerate(chunk):
            entries = list(processor.word_splitter(texts[index], lower=True))
            words = [token for token, _start, _end in entries]
            full_words = batch.text_word_counts[index]
            if full_words != len(words):
                raise ValueError(f"{row.qid}: processor counted {full_words} text words but "
                                 f"word_splitter yielded {len(words)}")
            retained_words = full_words if word_cap is None else min(full_words, word_cap)
            if retained_words == full_words:
                retained_subwords = batch.original_lengths[index]
            else:
                capped = processor.collate_fn_inference([(texts[index], model_schema)],
                                                        max_len=word_cap, error_policy="raise")
                retained_subwords = capped.original_lengths[0]
            metrics.append(audit_row(
                row, [(start_char, end_char) for _token, start_char, end_char in entries],
                retained_units=retained_words,
                head_units=_gliner_head_subwords(batch.mapped_indices[index]),
                full_subwords=batch.original_lengths[index],
                retained_subwords=retained_subwords, unit_kind="word"))
    return metrics


# ----------------------------------------------------------------------------- aggregation


def quantile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank quantile (deterministic, no interpolation); ``None`` for empty input."""
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(metrics: Sequence[Mapping[str, Any]]) -> dict:
    """One stratum's retention summary: unit/subword counts and per-segment shares."""
    n = len(metrics)
    shares = [m["retained_units"] / m["full_units"] if m["full_units"] else 1.0 for m in metrics]
    encoder_shares = [m["retained_subwords"] / m["full_subwords"] if m["full_subwords"] else 1.0
                      for m in metrics]
    segments: dict[str, dict] = {}
    for name in AUDITED_SEGMENTS:
        present = [m["segments"][name] for m in metrics if m["segments"][name]["present"]]
        segments[name] = {
            "present_share": len(present) / n if n else 0.0,
            "retained_share": _mean([1.0 if s["retained"] else 0.0 for s in present]),
            "fully_retained_share": _mean([1.0 if s["fully_retained"] else 0.0 for s in present]),
            "char_coverage_mean": _mean([s["char_coverage"] for s in present]),
            "char_coverage_p10": quantile([s["char_coverage"] for s in present], 0.1),
        }
    full_units = [m["full_units"] for m in metrics]
    retained_units = [m["retained_units"] for m in metrics]
    full_subwords = [m["full_subwords"] for m in metrics]
    retained_subwords = [m["retained_subwords"] for m in metrics]
    return {
        "n": n,
        "truncated_share": _mean([1.0 if m["retained_units"] < m["full_units"] else 0.0
                                  for m in metrics]),
        "unit_retention_share": {"mean": _mean(shares), "p10": quantile(shares, 0.1),
                                 "p50": quantile(shares, 0.5), "p90": quantile(shares, 0.9)},
        "full_units": {"mean": _mean(full_units), "p50": quantile(full_units, 0.5),
                       "p90": quantile(full_units, 0.9), "p95": quantile(full_units, 0.95),
                       "max": max(full_units) if full_units else None},
        "retained_units": {"mean": _mean(retained_units), "p50": quantile(retained_units, 0.5)},
        "full_subwords": {"mean": _mean(full_subwords), "p95": quantile(full_subwords, 0.95),
                          "max": max(full_subwords) if full_subwords else None},
        "retained_subwords": {"mean": _mean(retained_subwords),
                              "p50": quantile(retained_subwords, 0.5)},
        "encoder_input_retention_share": {"mean": _mean(encoder_shares)},
        "segments": segments,
    }


def _histogram(rows: Sequence[Row], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, attribute))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_report(*, backend: str, rows: Sequence[Row], metrics: Sequence[Mapping[str, Any]],
                 data_paths: Sequence[Path], tokenizer_info: Mapping[str, Any],
                 limits: Mapping[str, Any], version: str, namespace: str, output_path: Path,
                 argv: Sequence[str], caveats: Sequence[str],
                 source_verification: Mapping[str, Any], environment: Mapping[str, Any],
                 sampling: Mapping[str, Any]) -> dict:
    """Assemble the audit JSON: identity, limits, overall + depth/action strata, caveats."""
    by_depth = {str(depth): summarize_group([m for m, r in zip(metrics, rows) if r.depth == depth])
                for depth in sorted({row.depth for row in rows})}
    by_action = {action: summarize_group([m for m, r in zip(metrics, rows) if r.action == action])
                 for action in ACTION_ORDER}
    return {
        "audit": "decision_inputs",
        "backend": backend,
        "created_utc": now_iso(),
        "namespace": namespace,
        "output": str(output_path),
        "argv": list(argv),
        "state_text_version": version,
        "environment": dict(environment),
        "data": {
            "paths": [{"path": str(path), "sha256": sha256_file(path)} for path in data_paths],
            "rows_audited": len(rows),
            "sampling": dict(sampling),
            "splits": _histogram(rows, "split"),
            "depth": _histogram(rows, "depth"),
            "action": _histogram(rows, "action"),
        },
        "tokenizer": dict(tokenizer_info),
        "limits": dict(limits),
        "source_verification": dict(source_verification),
        "overall": summarize_group(metrics),
        "by_depth": by_depth,
        "by_action": by_action,
        "caveats": list(caveats),
    }


# --------------------------------------------------------------------------------- output


def validate_namespace(name: str) -> str:
    """A namespace must be one safe path segment: no separators, no leading dot."""
    if not _NAMESPACE_RE.match(name or ""):
        raise ValueError(f"namespace {name!r} must be a single safe path segment "
                         f"(letters, digits, dot, dash, underscore; no leading dot)")
    return name


def output_path(root: Path, namespace: str, backend: str, version: str) -> Path:
    validate_namespace(namespace)
    validate_namespace(version)
    return Path(root) / namespace / "audit" / f"decision_inputs_{backend}_{version}.json"


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomic write that refuses to overwrite an existing audit."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {path}")
    atomic_write_json(path, report)


# --------------------------------------------------------------------------- real backends


@dataclass
class BackendContext:
    metrics: list[dict]
    tokenizer_info: dict
    limits: dict
    source_verification: dict
    caveats: list[str]


def _package_version(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _environment(backend: str) -> dict:
    package = {"laya": "laya", "gliner": "gliner2"}[backend]
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "backend_package": f"{package}=={_package_version(package)}",
            "transformers": _package_version("transformers")}


def _tokenizer_info(tokenizer, source: Path, identity_files: Sequence[str]) -> dict:
    model_max = getattr(tokenizer, "model_max_length", None)
    if not isinstance(model_max, int) or isinstance(model_max, bool) or model_max > 10**12:
        model_max = None  # HF uses a sentinel for "unset"
    identity = {name: sha256_file(source / name) for name in identity_files
                if (source / name).is_file()}
    return {"source": str(source), "class": type(tokenizer).__name__,
            "is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "vocab_size": int(tokenizer.vocab_size), "model_max_length": model_max,
            "identity_files": identity}


def load_laya_tokenizer(checkpoint: Path):
    """Load the checkpoint tokenizer without mutating its files."""
    from transformers import AutoTokenizer

    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"{checkpoint}: no tokenizer/ directory (not a Laya checkpoint)")
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_dir))
    except AttributeError as exc:
        if "'list' object has no attribute 'keys'" not in str(exc):
            raise
        raise RuntimeError(
            f"{tokenizer_dir}: tokenizer cannot be loaded without changing checkpoint files; "
            "audit refuses to repair a historical artifact") from exc


def _laya_limits(checkpoint: Path) -> tuple[int, int]:
    config_path = checkpoint / "rl_agent_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{checkpoint}: no rl_agent_config.json (not a Laya checkpoint)")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return int(config.get("max_len", 512)), int(config.get("head_max_len", 192))


def _verify_laya_room(tokenizer, internal_question: Mapping[str, Any], max_len: int,
                      head_max_len: int, room: int) -> None:
    """Assert the measured room against ``build_sequence``'s own truncation stats."""
    from laya.common import build_sequence, encode_text

    probe = "\n".join(["Câu hỏi: probe"] + ["ứng viên " * 2000])
    state_ids = encode_text(tokenizer, probe, add_special_tokens=False)["input_ids"]
    _ids, _markers, _stats, truncation = build_sequence(
        tokenizer, probe, internal_question, max_len, head_max_len, state_ids=state_ids,
        return_stats=True, return_truncation_stats=True)
    used = truncation["state_tokens_used"]
    expected = min(len(state_ids), room)
    if used != expected:
        raise RuntimeError(f"laya layout changed: build_sequence used {used} state tokens, "
                           f"expected {expected} (room={room}); audit numbers would be wrong")


def run_laya(rows: Sequence[Row], *, checkpoint: Path) -> BackendContext:
    """Measure Laya retention with the checkpoint's real tokenizer, head and limits."""
    from laya.common import build_head, encode_text
    from laya.train import to_internal

    tokenizer = load_laya_tokenizer(checkpoint)
    max_len, head_max_len = _laya_limits(checkpoint)
    questions = uniform_questions(rows)
    validate_laya_questions(questions)
    if QUESTION_ID not in questions:
        raise ValueError(f"questions field has no {QUESTION_ID!r} entry: {sorted(questions)}")
    # The Agent converts the per-qid definition (`questions[qid]`), not the outer mapping.
    internal_question = to_internal(QUESTION_ID, dict(questions[QUESTION_ID]))
    head_ids, _markers, _stats = build_head(tokenizer, internal_question, head_max_len)
    head_tokens = len(head_ids)
    room = max(0, max_len - head_tokens - 1)
    _verify_laya_room(tokenizer, internal_question, max_len, head_max_len, room)

    mask_token = tokenizer.mask_token

    def encode(state: str) -> Mapping[str, Any]:
        # Mirrors laya.train.encode_state. Replacement would shift character offsets,
        # so refuse rows that need it rather than report misleading retention spans.
        if mask_token and mask_token in state:
            raise ValueError("state contains Laya mask token; offset audit cannot map replacement")
        return encode_text(tokenizer, state, add_special_tokens=False,
                           return_offsets_mapping=True)

    metrics = measure_laya_rows(rows, encode=encode, head_tokens=head_tokens, max_len=max_len)
    return BackendContext(
        metrics=metrics,
        tokenizer_info=_tokenizer_info(tokenizer, checkpoint / "tokenizer",
                                       ("tokenizer.json", "tokenizer_config.json")),
        limits={"kind": "subword", "max_len": max_len, "head_max_len": head_max_len,
                "head_tokens": head_tokens, "state_room": room,
                "truncation": "right: the state start is kept, the tail is dropped",
                "inference_max_len": max_len},
        source_verification={
            "package": f"laya=={_package_version('laya')}",
            "apis": ["transformers.AutoTokenizer.from_pretrained", "laya.train.to_internal",
                     "laya.common.build_head", "laya.common.build_sequence",
                     "laya.common.encode_text"],
            "room_check": (f"build_sequence(return_truncation_stats=True) reported "
                           f"state_tokens_used={room} on a probe state, matching "
                           f"max_len - head_tokens - 1"),
            "questions_sha256": sha256_text(canonical_json(questions)),
        },
        caveats=[
            "The state is right-truncated to state_room subwords; the v2 layout puts "
            "candidate/budget/action-mask last, so long evidence pushes them out first.",
            "Action-mask loss does not by itself admit illegal actions: the environment "
            "applies the mask outside the text.",
            "Only the checkpoint tokenizer and layout ran here; no model weights, no forward "
            "pass, no training.",
        ],
    )


def run_gliner(rows: Sequence[Row], *, base_dir: Path, batch_size: int) -> BackendContext:
    """Measure GLiNER retention with the real processor, compiled schema and word cap."""
    from gliner2.classification.compiler import compile_schema
    from gliner2.classification.schema import ClassificationSchema
    from gliner2.models.base import load_extractor_tokenizer
    from gliner2.processor import SchemaTransformer

    if not (base_dir / "config.json").is_file():
        raise FileNotFoundError(f"{base_dir}: no config.json (not a GLiNER checkpoint dir)")
    tokenizer = load_extractor_tokenizer(str(base_dir))
    processor = SchemaTransformer(tokenizer=tokenizer)
    schema = ClassificationSchema()
    schema.single(QUESTION_ID, dict(ACTION_CRITERIA), instruction=INSTRUCTIONS)
    compiled = compile_schema(schema)

    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    word_cap = config.get("max_len")
    word_cap = int(word_cap) if word_cap is not None else None
    encoder_config_path = base_dir / "encoder_config" / "config.json"
    encoder_config = (json.loads(encoder_config_path.read_text(encoding="utf-8"))
                      if encoder_config_path.is_file() else {})
    encoder_max = encoder_config.get("max_position_embeddings")

    metrics = measure_gliner_rows(rows, processor=processor, model_schema=compiled.build(),
                                  word_cap=None, batch_size=batch_size)
    head_subwords = metrics[0]["head_units"] if metrics else None
    over_encoder_limit = None
    if encoder_max:
        over_encoder_limit = _mean([1.0 if m["full_subwords"] > encoder_max else 0.0
                                    for m in metrics])
    state_over_encoder_limit = None
    if encoder_max:
        state_over_encoder_limit = _mean([
            1.0 if m["full_subwords"] - m["head_units"] > encoder_max else 0.0
            for m in metrics])
    return BackendContext(
        metrics=metrics,
        tokenizer_info=_tokenizer_info(tokenizer, base_dir,
                                       ("tokenizer.json", "tokenizer_config.json", "config.json",
                                        "encoder_config/config.json")),
        limits={"kind": "word", "trainer_word_cap": word_cap,
                "inference_max_len": None, "head_subwords": head_subwords,
                "subword_truncation": "none in the audited classification path",
                "encoder_max_position_embeddings": encoder_max,
                "encoder_position_biased_input": encoder_config.get("position_biased_input"),
                "encoder_relative_attention": encoder_config.get("relative_attention"),
                "encoder_max_relative_positions": encoder_config.get("max_relative_positions"),
                "head_share_of_declared_position_count": (
                    head_subwords / encoder_max if head_subwords is not None and encoder_max
                    else None),
                "rows_over_encoder_max_position_embeddings": over_encoder_limit,
                "rows_with_state_subwords_over_declared_position_count": (
                    state_over_encoder_limit)},
        source_verification={
            "package": f"gliner2=={_package_version('gliner2')}",
            "apis": ["gliner2.models.base.load_extractor_tokenizer",
                     "gliner2.processor.SchemaTransformer.collate_fn_inference",
                     "gliner2.classification.schema.ClassificationSchema.single",
                     "gliner2.classification.compiler.compile_schema"],
            "schema_fingerprint": compiled.fingerprint,
            "word_cap_check": ("the cap counts words after splitting and drops words before "
                               "subword tokenization; no subword truncation exists in this path"),
        },
        caveats=[
            "Classification inference passes max_len=None and applies no word or subword "
            "truncation: all-1.0 retention figures are vacuous, not evidence of model quality.",
            "Checkpoint config max_len=4096 words applies to training, not this adapter's "
            "inference call; max observed input is below that training cap.",
            "The encoder config declares max_position_embeddings=512, "
            "position_biased_input=false and relative_attention=true; this is not an absolute "
            "input-length clamp in this DeBERTa v2 encoder. The classification path sends full "
            "inputs, but effects on quality at these lengths require model forwards/evaluation.",
            "Only the tokenizer/processor/schema ran here; no model weights, no forward pass.",
            "The processor appends '.' when the state does not end in .!? (mirrored here); "
            "char offsets are measured on that mutated text.",
            "GLiNER's cap counts words, not subwords; it must not be compared to Laya's "
            "subword max_len.",
        ],
    )


# ------------------------------------------------------------------------------------ CLI


def _print_summary(report: Mapping[str, Any], output: Path) -> None:
    overall = report["overall"]
    print(f"[{report['backend']}] {report['data']['rows_audited']} states, "
          f"truncated {overall['truncated_share']:.1%}, "
          f"unit retention mean {overall['unit_retention_share']['mean']:.1%}")
    for name in AUDITED_SEGMENTS:
        segment = overall["segments"][name]
        print(f"  {name:11s} present {segment['present_share']:.1%} "
              f"retained {segment['retained_share']:.1%} "
              f"fully {segment['fully_retained_share']:.1%} "
              f"coverage {segment['char_coverage_mean']:.1%}")
    print(f"  wrote {output}")


def _resolve(path: Path) -> Path:
    """CLI paths are repo-relative unless absolute (the script is repo-scoped)."""
    return path if path.is_absolute() else ROOT / path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("laya", "gliner"), required=True,
                        help="laya runs in the baseline .venv; gliner runs in .venv-gliner")
    parser.add_argument("--data", type=Path, action="append", default=None,
                        help="teacher JSONL, train/val only (repeatable; "
                             f"default: {DEFAULT_DATA})")
    parser.add_argument("--limit", type=int, default=None,
                        help="audit a deterministic stride sample of N rows")
    parser.add_argument("--namespace", required=True,
                        help="experiment namespace under the experiments root")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_LAYA_CHECKPOINT,
                        help="Laya checkpoint dir (backend=laya)")
    parser.add_argument("--base", type=Path, default=None,
                        help="GLiNER base model dir (backend=gliner)")
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="GLiNER collate batch size")
    args = parser.parse_args(argv)

    namespace = validate_namespace(args.namespace)
    data_paths = [_resolve(path) for path in (args.data or [DEFAULT_DATA])]
    for path in data_paths:
        if not path.is_file():
            parser.error(f"data file not found: {path}")
    rows = load_rows(data_paths)
    if not rows:
        parser.error("no rows loaded")
    version = uniform_state_text_version(rows)
    sampled, stride = sample_rows(rows, args.limit)
    output = output_path(_resolve(args.experiments_root), namespace, args.backend, version)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing audit: {output}")

    if args.backend == "laya":
        context = run_laya(sampled, checkpoint=_resolve(args.checkpoint))
    else:
        if args.base is None:
            parser.error("--base is required for --backend gliner")
        context = run_gliner(sampled, base_dir=_resolve(args.base), batch_size=args.batch_size)

    report = build_report(
        backend=args.backend, rows=sampled, metrics=context.metrics, data_paths=data_paths,
        tokenizer_info=context.tokenizer_info, limits=context.limits, version=version,
        namespace=namespace, output_path=output, argv=sys.argv[1:], caveats=context.caveats,
        source_verification=context.source_verification, environment=_environment(args.backend),
        sampling={"limit": args.limit, "stride": stride})
    write_report(output, report)
    _print_summary(report, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for ``scripts/audit_decision_inputs.py`` — the P0 retention audit.

The audit is split so its math is pure: segment spans, token/char mapping, retention
classification, stratification and the JSON report are all functions over plain data,
tested here with fake offsets and fake tokenizer/processor objects. The thin backend
adapters only translate a real tokenizer/processor into those plain inputs, so no
checkpoint, GPU or ``gliner2`` install is needed for this suite.

What is pinned by these tests:
* the v2 state layout (and the reordered v3 layout) is parsed by line-start headers,
  not by substring occurrence, so evidence prose cannot fake a header;
* retention means "first unit inside the retained window", full retention means "last
  unit inside it", and evidence char coverage is measured from real offsets;
* test-split rows are refused outright (no test labels reach the audit);
* reports land under ``outputs/experiments/<namespace>/audit/`` and never overwrite.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_decision_inputs.py"

spec = importlib.util.spec_from_file_location("audit_decision_inputs", SCRIPT)
assert spec is not None and spec.loader is not None
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit  # dataclasses resolve annotations via sys.modules
spec.loader.exec_module(audit)


def _state_v2() -> str:
    """The canonical v2 layout; the snippet deliberately embeds a header-like phrase."""
    return "\n".join([
        "Câu hỏi: Ai là tác giả của Truyện Kiều?",
        "Bằng chứng đã xem: 1 đoạn",
        "[1] doc=abc123 Đoạn văn nói Ứng viên trả lời: không phải tiêu đề thật.",
        "Ứng viên trả lời: Nguyễn Du",
        "Nguồn ứng viên: doc=abc123 | hợp lệ: có",
        "Điểm đọc: non_null=1.000 | null=0.000 | chênh lệch null=-1.000",
        "Truy hồi: BM25 top=1.000 (khoảng cách 0.000) | Dense top=1.000 (khoảng cách 0.000)",
        "Hành động đã thực hiện: (chưa có)",
        "Số lượt tìm đã dùng: 0 | còn lại: 2",
        "Hành động hợp lệ: ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN",
    ])


def _state_v3_order() -> str:
    """P4's planned order: candidate first, budget/mask next, evidence last."""
    return "\n".join([
        "Câu hỏi: Ai là tác giả của Truyện Kiều?",
        "Ứng viên trả lời: Nguyễn Du",
        "Nguồn ứng viên: doc=abc123 | hợp lệ: có",
        "Điểm đọc: non_null=1.000 | null=0.000 | chênh lệch null=-1.000",
        "Số lượt tìm đã dùng: 0 | còn lại: 2",
        "Hành động hợp lệ: ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN",
        "Truy hồi: BM25 top=1.000 (khoảng cách 0.000) | Dense top=1.000 (khoảng cách 0.000)",
        "Hành động đã thực hiện: (chưa có)",
        "Bằng chứng đã xem: 1 đoạn",
        "[1] doc=abc123 Đoạn văn.",
    ])


def _row(state: str = None, *, depth: int = 0, action: str = "ANSWER",
         qid: str = "train:0001", split: str = "train_strict", version: str = "v2") -> dict:
    """A JSONL-shaped teacher row (the subset the loader reads)."""
    return {
        "state": state if state is not None else _state_v2(),
        "questions": {"next_action": {"type": "choice"}},
        "expected": {"next_action": action},
        "meta": {"qid": qid, "split": split, "depth": depth, "expected": action,
                 "state_text_version": version},
    }


# ------------------------------------------------------------------ segment spans


def test_segment_spans_finds_v2_blocks_at_line_starts():
    text = _state_v2()
    spans = audit.segment_spans(text)
    candidate_start = text.index("\nỨng viên trả lời: ") + 1
    retrieval_start = text.index("\nTruy hồi: ") + 1
    budget_start = text.index("\nSố lượt tìm đã dùng: ") + 1
    mask_start = text.index("\nHành động hợp lệ: ") + 1
    assert spans["evidence"] == (text.index("Bằng chứng đã xem: "), candidate_start)
    assert spans["candidate"] == (candidate_start, retrieval_start)
    assert spans["budget"] == (budget_start, mask_start)
    assert spans["action_mask"] == (mask_start, len(text))


def test_segment_spans_ignores_header_like_text_inside_evidence():
    text = _state_v2()
    # The snippet contains "Ứng viên trả lời:" mid-line; the real header is the line start.
    spans = audit.segment_spans(text)
    assert spans["candidate"][0] == text.index("\nỨng viên trả lời: ") + 1
    assert spans["evidence"][1] == spans["candidate"][0]


def test_segment_spans_survives_the_v3_reordering():
    text = _state_v3_order()
    spans = audit.segment_spans(text)
    assert spans["candidate"][0] < spans["budget"][0] < spans["action_mask"][0]
    assert spans["action_mask"][0] < spans["evidence"][0]
    assert spans["evidence"][1] == len(text)
    assert spans["candidate"][1] == spans["budget"][0]


def test_segment_spans_rejects_duplicate_line_start_headers():
    text = _state_v2() + "\nHành động hợp lệ: forged duplicate"
    with pytest.raises(ValueError, match="duplicate.*action_mask"):
        audit.segment_spans(text)


def test_segment_spans_returns_none_for_missing_headers():
    spans = audit.segment_spans("Câu hỏi: x\nchỉ có dòng này")
    assert set(spans) == {"evidence", "candidate", "budget", "action_mask"}
    assert all(value is None for value in spans.values())


# ------------------------------------------------------------------ token mapping


@pytest.mark.parametrize("start, end, expected", [
    (0, 3, (0, 0)),
    (4, 8, (1, 2)),
    (2, 4, (0, 1)),
    (0, 0, None),
    (10, 12, None),
    (9, 10, (2, 2)),
])
def test_token_span_overlaps_char_range(start, end, expected):
    offsets = [(0, 3), (3, 7), (7, 10)]
    assert audit.token_span(offsets, start, end) == expected


def test_token_span_on_empty_offsets_is_none():
    assert audit.token_span([], 0, 5) is None


# ------------------------------------------------------------------ per-row audit


def _char_offsets(text: str) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(len(text))]


def _row_for_audit() -> audit.Row:
    return audit.Row(state=_state_v2(), qid="train:0001", split="train_strict",
                     depth=0, action="ANSWER")


def test_audit_row_retention_and_segment_flags():
    row = _row_for_audit()
    text = row.state
    offsets = _char_offsets(text)
    candidate_start = text.index("\nỨng viên trả lời: ") + 1
    evidence_start = text.index("Bằng chứng đã xem: ")
    # Window ends exactly at the candidate header: evidence fully inside, candidate outside.
    metric = audit.audit_row(row, offsets, retained_units=candidate_start, head_units=92,
                             full_subwords=92 + len(text) + 1,
                             retained_subwords=92 + candidate_start + 1,
                             unit_kind="subword")
    assert metric["full_units"] == len(text)
    assert metric["retained_units"] == candidate_start
    evidence = metric["segments"]["evidence"]
    assert evidence["present"] and evidence["retained"] and evidence["fully_retained"]
    assert evidence["char_coverage"] == pytest.approx(1.0)
    candidate = metric["segments"]["candidate"]
    assert candidate["present"] and not candidate["retained"]
    assert metric["segments"]["budget"]["retained"] is False
    assert metric["segments"]["action_mask"]["retained"] is False


def test_audit_row_partial_evidence_coverage():
    row = _row_for_audit()
    text = row.state
    offsets = _char_offsets(text)
    evidence_start = text.index("Bằng chứng đã xem: ")
    candidate_start = text.index("\nỨng viên trả lời: ") + 1
    window = evidence_start + 5
    metric = audit.audit_row(row, offsets, retained_units=window, head_units=1,
                             full_subwords=len(text) + 2, retained_subwords=window + 2,
                             unit_kind="subword")
    evidence = metric["segments"]["evidence"]
    assert evidence["retained"] and not evidence["fully_retained"]
    assert evidence["char_coverage"] == pytest.approx(
        5.0 / (candidate_start - evidence_start))


def test_audit_row_absent_segments_are_reported_as_absent():
    row = audit.Row(state="Câu hỏi: x\nchỉ có dòng này", qid="q", split="train_strict",
                    depth=1, action="ABSTAIN")
    metric = audit.audit_row(row, [(0, 1), (1, 2)], retained_units=2, head_units=0,
                             full_subwords=2, retained_subwords=2, unit_kind="subword")
    for name in ("evidence", "candidate", "budget", "action_mask"):
        assert metric["segments"][name]["present"] is False
        assert metric["segments"][name]["retained"] is False


def test_audit_row_with_zero_retained_units():
    row = _row_for_audit()
    offsets = _char_offsets(row.state)
    metric = audit.audit_row(row, offsets, retained_units=0, head_units=10,
                             full_subwords=len(row.state) + 11, retained_subwords=11,
                             unit_kind="subword")
    assert metric["segments"]["evidence"]["retained"] is False
    assert metric["segments"]["evidence"]["char_coverage"] == 0.0


# ------------------------------------------------------------------ aggregation


def _metric(*, depth: int, action: str, full: int, retained: int,
            segments: dict | None = None) -> dict:
    return {"qid": f"q{depth}{action}", "split": "train_strict", "depth": depth,
            "action": action, "unit_kind": "subword",
            "full_units": full, "retained_units": retained,
            "head_units": 2, "full_subwords": full + 3, "retained_subwords": retained + 3,
            "segments": segments or {
                "evidence": {"present": True, "retained": True, "fully_retained": True,
                             "char_coverage": 1.0},
                "candidate": {"present": True, "retained": retained >= full,
                              "fully_retained": retained >= full,
                              "char_coverage": 1.0 if retained >= full else 0.5},
                "budget": {"present": True, "retained": retained >= full,
                           "fully_retained": retained >= full,
                           "char_coverage": 1.0 if retained >= full else 0.0},
                "action_mask": {"present": True, "retained": retained >= full,
                                "fully_retained": retained >= full,
                                "char_coverage": 1.0 if retained >= full else 0.0},
            }}


def test_summarize_group_stratifies_depth_action_and_segments():
    rows = [
        _metric(depth=0, action="ANSWER", full=100, retained=100),
        _metric(depth=0, action="ABSTAIN", full=100, retained=50),
        _metric(depth=2, action="ABSTAIN", full=200, retained=0),
    ]
    summary = audit.summarize_group(rows)
    assert summary["n"] == 3
    assert summary["truncated_share"] == pytest.approx(2 / 3)
    assert summary["unit_retention_share"]["mean"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert summary["full_units"]["max"] == 200
    assert summary["segments"]["candidate"]["retained_share"] == pytest.approx(1 / 3)
    assert summary["segments"]["candidate"]["fully_retained_share"] == pytest.approx(1 / 3)
    assert summary["segments"]["action_mask"]["present_share"] == pytest.approx(1.0)


def test_summarize_group_counts_absent_segments_separately():
    absent = _metric(depth=0, action="ANSWER", full=10, retained=10)
    absent["segments"]["budget"] = {"present": False, "retained": False,
                                    "fully_retained": False, "char_coverage": 0.0}
    summary = audit.summarize_group([absent, _metric(depth=0, action="ANSWER", full=10, retained=10)])
    budget = summary["segments"]["budget"]
    assert budget["present_share"] == pytest.approx(0.5)
    assert budget["retained_share"] == pytest.approx(1.0)  # among the present ones


def test_summarize_group_handles_the_empty_group():
    summary = audit.summarize_group([])
    assert summary["n"] == 0
    assert summary["truncated_share"] == 0.0
    assert summary["unit_retention_share"]["mean"] == 0.0
    assert summary["segments"]["candidate"]["present_share"] == 0.0


@pytest.mark.parametrize("values, q, expected", [
    ([1, 2, 3, 4], 0.5, 2),
    ([1, 2, 3, 4], 0.9, 4),
    ([1, 2, 3, 4], 0.1, 1),
    ([5], 0.5, 5),
    ([], 0.5, None),
])
def test_quantile_is_nearest_rank(values, q, expected):
    assert audit.quantile(values, q) == expected


def test_sample_rows_rejects_nonpositive_limit():
    rows = [audit.Row(state="state", qid="q", split="train_strict", depth=0,
                      action="ANSWER")]
    for limit in (0, -1):
        with pytest.raises(ValueError, match="limit"):
            audit.sample_rows(rows, limit)


def test_sample_rows_uses_a_deterministic_stride():
    rows = [audit.Row(state=f"s{i}", qid=f"q{i}", split="train_strict", depth=0,
                      action="ANSWER") for i in range(10)]
    sampled, stride = audit.sample_rows(rows, 4)
    assert stride == 3
    assert [row.qid for row in sampled] == ["q0", "q3", "q6", "q9"]
    everything, stride_all = audit.sample_rows(rows, None)
    assert everything == rows and stride_all == 1


# ------------------------------------------------------------------ data loading


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")
    return path


def test_validate_laya_questions_matches_runtime_schema():
    audit.validate_laya_questions(audit.build_questions_field())
    with pytest.raises(ValueError, match="questions.*runtime"):
        audit.validate_laya_questions({"next_action": {"type": "choice"}})


def test_load_rows_parses_teacher_rows(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", [_row(depth=1, action="SEARCH_BM25")])
    rows = audit.load_rows([path])
    assert len(rows) == 1
    assert rows[0].depth == 1 and rows[0].action == "SEARCH_BM25"
    assert audit.uniform_state_text_version(rows) == "v2"


def test_load_rows_refuses_test_split_rows(tmp_path):
    path = _write_jsonl(tmp_path / "test.jsonl", [_row(split="test_strict")])
    with pytest.raises(ValueError, match="test_strict"):
        audit.load_rows([path])


def test_load_rows_refuses_a_missing_split(tmp_path):
    row = _row()
    row["meta"].pop("split")
    path = _write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(ValueError, match="meta.split"):
        audit.load_rows([path])


def test_load_rows_refuses_an_empty_state(tmp_path):
    row = _row()
    row["state"] = "   "
    path = _write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(ValueError, match="state"):
        audit.load_rows([path])


def test_load_rows_refuses_an_unknown_action(tmp_path):
    row = _row()
    row["meta"]["expected"] = "NOT_AN_ACTION"
    row["expected"]["next_action"] = "NOT_AN_ACTION"
    path = _write_jsonl(tmp_path / "bad.jsonl", [row])
    with pytest.raises(ValueError, match="action"):
        audit.load_rows([path])


def test_uniform_state_text_version_refuses_mixed_files(tmp_path):
    path = _write_jsonl(tmp_path / "mixed.jsonl",
                        [_row(version="v2"), _row(qid="train:0002", version="v3")])
    rows = audit.load_rows([path])
    with pytest.raises(ValueError, match="mixed"):
        audit.uniform_state_text_version(rows)


# ------------------------------------------------------------------ backend adapters


def _fake_encode(text: str) -> dict:
    return {"input_ids": list(range(len(text))),
            "offset_mapping": _char_offsets(text)}


def test_measure_laya_rows_computes_room_retention_and_subwords():
    row = _row_for_audit()
    metrics = audit.measure_laya_rows([row], encode=_fake_encode, head_tokens=3, max_len=20)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric["unit_kind"] == "subword"
    assert metric["full_units"] == len(row.state)
    assert metric["retained_units"] == 16  # max_len - head - 1
    assert metric["full_subwords"] == 3 + len(row.state) + 1
    assert metric["retained_subwords"] == 3 + 16 + 1
    assert metric["depth"] == 0 and metric["action"] == "ANSWER"


def test_measure_laya_rows_never_retains_more_than_the_state():
    row = audit.Row(state="ngắn", qid="q", split="train_strict", depth=0, action="ANSWER")
    metrics = audit.measure_laya_rows([row], encode=_fake_encode, head_tokens=2, max_len=100)
    assert metrics[0]["retained_units"] == metrics[0]["full_units"] == 4
    assert metrics[0]["segments"]["candidate"]["present"] is False


class _FakeBatch:
    def __init__(self, lengths, word_counts, mappings):
        self.original_lengths = lengths
        self.text_word_counts = word_counts
        self.mapped_indices = mappings


class _FakeProcessor:
    """Mirrors the parts of gliner2's SchemaTransformer the adapter consumes."""

    HEAD = 2  # schema + [SEP_TEXT] subwords, before the first text word

    #: The real WhitespaceTokenSplitter pattern: words, then any non-space char.
    _PATTERN = re.compile(r"\w+(?:[-_]\w+)*|\S")

    def word_splitter(self, text, lower=True):
        for match in self._PATTERN.finditer(text):
            token = match.group()
            yield (token.lower() if lower else token, match.start(), match.end())

    def collate_fn_inference(self, rows, max_len=None, error_policy=None):
        lengths, counts, mappings = [], [], []
        for text, _schema in rows:
            if text and not text.endswith((".", "!", "?")):
                text = text + "."
            words = [word for word, _, _ in self.word_splitter(text)]
            if max_len is not None:
                words = words[:max_len]
            subwords = self.HEAD + sum(2 if len(word) > 3 else 1 for word in words)
            lengths.append(subwords)
            counts.append(len(words))
            mappings.append(
                [("schema", 0, 0), ("sep", 1, 0)]
                + [("text", self.HEAD + index, 1) for index in range(len(words))])
        return _FakeBatch(lengths, counts, mappings)


def test_measure_gliner_rows_uses_word_units_and_reports_subwords():
    row = audit.Row(state="một hai ba", qid="q", split="train_strict", depth=0,
                    action="ANSWER")
    metrics = audit.measure_gliner_rows([row], processor=_FakeProcessor(),
                                        model_schema={}, word_cap=4096)
    metric = metrics[0]
    assert metric["unit_kind"] == "word"
    assert metric["full_units"] == 4  # "một", "hai", "ba", appended "."
    assert metric["retained_units"] == 4
    assert metric["head_units"] == _FakeProcessor.HEAD
    # words "một"/"hai"/"ba" are 3 chars -> 1 subword; the appended "." is 1 char -> 1.
    assert metric["full_subwords"] == _FakeProcessor.HEAD + 4
    assert metric["retained_subwords"] == metric["full_subwords"]


def test_measure_gliner_rows_applies_the_word_cap():
    row = audit.Row(state="một hai ba bốn năm", qid="q", split="train_strict", depth=0,
                    action="ANSWER")
    metrics = audit.measure_gliner_rows([row], processor=_FakeProcessor(),
                                        model_schema={}, word_cap=2)
    metric = metrics[0]
    assert metric["full_units"] == 6  # five words + appended "."
    assert metric["retained_units"] == 2
    assert metric["full_subwords"] == _FakeProcessor.HEAD + 6
    assert metric["retained_subwords"] == _FakeProcessor.HEAD + 2
    assert metric["segments"]["candidate"]["present"] is False


# ------------------------------------------------------------------ output contract


@pytest.mark.parametrize("namespace", ["2026-10-09-gliner-p0", "p4.v3_audit"])
def test_validate_namespace_accepts_a_single_safe_segment(namespace):
    assert audit.validate_namespace(namespace) == namespace


@pytest.mark.parametrize("namespace", ["", ".hidden", "a/b", "..", "a\\b", "a b"])
def test_validate_namespace_rejects_traversal_and_separators(namespace):
    with pytest.raises(ValueError):
        audit.validate_namespace(namespace)


def test_output_path_is_namespaced_and_versioned(tmp_path):
    path = audit.output_path(tmp_path, "2026-10-09-gliner-p0", "laya", "v2")
    assert path == tmp_path / "2026-10-09-gliner-p0" / "audit" / "decision_inputs_laya_v2.json"


def test_write_report_refuses_to_overwrite(tmp_path):
    report = {"audit": "decision_inputs", "backend": "laya"}
    path = audit.output_path(tmp_path, "ns", "laya", "v2")
    audit.write_report(path, report)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        audit.write_report(path, report)


def test_build_report_records_identity_and_strata(tmp_path):
    path = _write_jsonl(tmp_path / "train.jsonl", [_row(depth=0), _row(qid="train:0002", depth=1)])
    rows = audit.load_rows([path])
    metrics = [audit.audit_row(row, _char_offsets(row.state),
                               retained_units=len(row.state), head_units=1,
                               full_subwords=len(row.state) + 2,
                               retained_subwords=len(row.state) + 2, unit_kind="subword")
               for row in rows]
    report = audit.build_report(
        backend="laya", rows=rows, metrics=metrics, data_paths=[path],
        tokenizer_info={"class": "Fake"}, limits={"kind": "subword", "max_len": 768},
        version="v2", namespace="ns", output_path=tmp_path / "ns" / "audit" / "x.json",
        argv=["audit_decision_inputs.py"], caveats=["caveat"], source_verification={"a": 1},
        environment={"python": "3.10"}, sampling={"limit": None, "stride": 1})
    assert report["audit"] == "decision_inputs"
    assert report["backend"] == "laya"
    assert report["state_text_version"] == "v2"
    assert report["overall"]["n"] == 2
    assert report["by_depth"]["0"]["n"] == 1 and report["by_depth"]["1"]["n"] == 1
    assert report["by_action"]["ANSWER"]["n"] == 2
    assert report["data"]["paths"][0]["sha256"]
    assert report["caveats"] == ["caveat"]

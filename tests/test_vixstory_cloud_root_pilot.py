"""Cloud root dev40 pilot contracts (CPU/synthetic, no network, no local reader load)."""

import json
import random
import re
from pathlib import Path

import pytest

from scripts import vixstory_cloud_root_pilot as mod
from scripts.vixstory_reader_eval import (
    MAX_PROMPT_CHARS,
    MODEL,
    CloudReader,
    build_identity,
)


def _rows(n: int = 200) -> list[dict]:
    return [{"id": f"qa_{i:06d}", "question": f"câu hỏi {i}", "answer": f"đáp án {i}",
             "context_ids": [f"gold_{i}"]} for i in range(n)]


def _json_body(answer: str, cited: list[str]) -> str:
    """A JSON reader response, built without shadowing the ``json`` request keyword."""
    return '{"answer":%s,"citations":%s}' % (json.dumps(answer, ensure_ascii=False),
                                             json.dumps(cited, ensure_ascii=False))


@pytest.fixture(scope="module")
def real_identity():
    """A real cache identity, exercising ``build_identity(corpus_path, rankings_path)``."""
    return build_identity(corpus_path=mod.CORPUS_PATH, rankings_path=mod.RANKINGS_PATH)


class _Response:
    def __init__(self, payload: dict) -> None:
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """A synthetic OpenAI-compatible client; it cites the one passage each prompt shows.

    ``answers`` maps a prompt to the exact answer text to return, so a test can drive the F1
    and gated-success metrics deterministically without a network call.
    """

    def __init__(self, answers: dict[str, str] | None = None, model: str = MODEL) -> None:
        self.answers = answers or {}
        self.model = model
        self.prompts: list[str] = []

    def post(self, url: str, headers=None, json=None):  # noqa: A002 - httpx keyword name
        content = json["messages"][0]["content"]
        self.prompts.append(content)
        match = re.search(r"Đoạn văn:\n\[([^\]]+)\]", content)
        cited = [match.group(1)] if match else []
        body = _json_body(self.answers.get(content, "trả lời ngắn"), cited)
        return _Response({"model": self.model, "system_fingerprint": "fp",
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                    "total_tokens": 15},
                          "choices": [{"message": {"content": body},
                                       "finish_reason": "stop"}]})


# --- selection: fixed, deterministic, dev200-only ------------------------------------------

def test_select_dev_sample_is_deterministic_and_matches_seed42_rule():
    rows = _rows(200)
    selected = mod.select_dev_sample(rows, 40, seed=42)
    assert len(selected) == 40
    assert [r["id"] for r in selected] == [r["id"] for r in mod.select_dev_sample(rows, 40, seed=42)]
    # Independent reproduction of the copied rule (ids are zero-padded -> order is stable).
    expected = sorted(random.Random(42).sample(range(len(rows)), 40))
    ordered = sorted(rows, key=lambda r: r["id"])
    assert [r["id"] for r in selected] == [ordered[i]["id"] for i in expected]


def test_select_dev_sample_rejects_out_of_range_and_passes_none_through():
    rows = _rows(5)
    with pytest.raises(ValueError):
        mod.select_dev_sample(rows, 6, seed=42)
    with pytest.raises(ValueError):
        mod.select_dev_sample(rows, 0, seed=42)
    assert [r["id"] for r in mod.select_dev_sample(rows, None, seed=42)] == [r["id"] for r in rows]


def test_split_qids_resolves_alias_and_validates():
    manifest = {"splits": {"train_dev_200": {"qids": ["qa_a", "qa_b"]}}}
    assert mod.split_qids(manifest, mod.DEV_SPLIT_ALIASES) == ["qa_a", "qa_b"]
    assert mod.split_qids({"splits": {"dev": ["qa_c"]}}, mod.DEV_SPLIT_ALIASES) == ["qa_c"]
    with pytest.raises(ValueError):
        mod.split_qids({}, mod.DEV_SPLIT_ALIASES)
    with pytest.raises(ValueError):
        mod.split_qids({"splits": {"dev": [1, 2]}}, mod.DEV_SPLIT_ALIASES)


# --- root-only prompt: one passage, no gold ------------------------------------------------

def test_root_passage_is_the_bm25_top1_and_prompt_hides_gold():
    docs = {"root": {"text": "Đoạn gốc được truy hồi."}, "gold": {"text": "Đoạn vàng."}}
    row = {"id": "qa_1", "question": "Hỏi gì?", "answer": "đáp án vàng", "context_ids": ["gold"]}
    rankings = {"qa_1": {"bm25": ["root"], "dense": ["gold"]}}
    root_id, passages = mod.root_passage(row, docs, rankings)
    assert root_id == "root"
    assert [p["context_id"] for p in passages] == ["root"]
    prompt = mod.reader_prompt(row["question"], passages)
    assert "Đoạn gốc được truy hồi." in prompt
    assert "Đoạn vàng." not in prompt   # gold passage never shown
    assert "[gold]" not in prompt       # gold id never shown
    assert prompt.count("[") == 2       # the one passage + the JSON skeleton only


def test_root_passage_requires_a_bm25_ranking():
    docs = {"root": {"text": "x"}}
    with pytest.raises(KeyError):
        mod.root_passage({"id": "qa_1"}, docs, {})
    with pytest.raises(KeyError):
        mod.root_passage({"id": "qa_1"}, docs, {"qa_1": {"bm25": ["absent"], "dense": []}})


def test_build_prompts_shows_one_bounded_passage_per_row():
    rows = _rows(3)
    docs = {f"r{i}": {"text": "nội dung " + "x" * 20} for i in range(3)}
    rankings = {f"qa_{i:06d}": {"bm25": [f"r{i}"], "dense": []} for i in range(3)}
    visible, prompts = mod.build_prompts(rows, docs, rankings)
    assert [p[0]["context_id"] for p in visible] == ["r0", "r1", "r2"]
    assert len(prompts) == 3 and all(len(p) <= MAX_PROMPT_CHARS for p in prompts)


def test_smoke_prompt_is_synthetic_and_one_passage():
    prompt = mod.smoke_prompt()
    assert "SMOKE" in prompt and len(prompt) <= MAX_PROMPT_CHARS
    assert prompt.split("Câu hỏi:")[0].count("[") == 1


# --- local scoring: F1, citation validity, gated success -----------------------------------

def test_score_computes_f1_citation_and_gated_success():
    rows = [{"id": "qa_1", "answer": "mười đồng"}, {"id": "qa_2", "answer": "hai"}]
    visible = [[{"context_id": "r1", "text": "x"}], [{"context_id": "r2", "text": "y"}]]
    responses = ['{"answer": "mười đồng", "citations": ["r1"]}',   # exact text + valid citation
                 '{"answer": "hai", "citations": ["nope"]}']       # exact text, bad citation
    records = mod.score(rows, visible, responses)
    assert records[0]["f1"] == 1.0 and records[0]["citation_valid"] is True
    assert records[0]["success"] is True
    assert records[1]["f1"] == 1.0 and records[1]["citation_valid"] is False
    assert records[1]["success"] is False  # gated success needs a valid citation
    assert mod.metrics(records) == {"n": 2, "mean_f1": 1.0, "answer_rate": 1.0,
                                    "citation_valid_rate": 0.5, "gated_success_rate": 0.5}


def test_metrics_of_empty_records_is_zero_n():
    assert mod.metrics([]) == {"n": 0}


# --- credential guard ----------------------------------------------------------------------

def test_require_api_key_raises_when_absent(monkeypatch):
    monkeypatch.delenv("AIBOX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="AIBOX_API_KEY"):
        mod.require_api_key()
    monkeypatch.setenv("AIBOX_API_KEY", "test-key")
    mod.require_api_key()  # present -> no raise


def test_run_fails_before_loading_qa_when_key_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("AIBOX_API_KEY", raising=False)
    loaded: list[int] = []
    monkeypatch.setattr(mod, "load_dev40", lambda *a, **k: loaded.append(1))
    with pytest.raises(RuntimeError, match="AIBOX_API_KEY"):
        mod.run(out=tmp_path)
    assert loaded == []  # no QA row was read


# --- overwrite guard -----------------------------------------------------------------------

def test_run_refuses_overwrite_before_loading(tmp_path, monkeypatch):
    monkeypatch.setenv("AIBOX_API_KEY", "test-key")
    (tmp_path / mod.REPORT_NAME).write_text("{}", encoding="utf-8")
    loaded: list[int] = []
    monkeypatch.setattr(mod, "load_dev40", lambda *a, **k: loaded.append(1))
    with pytest.raises(FileExistsError):
        mod.run(out=tmp_path)
    assert loaded == []


# --- end to end with a synthetic client (no network) ---------------------------------------

def test_run_end_to_end_with_synthetic_smoke(tmp_path, monkeypatch, real_identity):
    monkeypatch.setenv("AIBOX_API_KEY", "test-key")
    docs, rows = mod.load_dev40(mod.MANIFEST_PATH, mod.CORPUS_PATH, mod.TRAIN_PATH,
                               n=mod.DEV_N, seed=mod.DEV_SEED)
    rankings = json.loads(mod.RANKINGS_PATH.read_text(encoding="utf-8"))
    _visible, prompts = mod.build_prompts(rows, docs, rankings)
    answers = {prompt: row["answer"] for row, prompt in zip(rows, prompts)}

    client = _FakeClient(answers=answers)
    reader = CloudReader(tmp_path / "cache.json", identity=real_identity,
                         max_requests=mod.MAX_REQUESTS, client=client)
    report = mod.run(out=tmp_path, reader=reader, max_requests=mod.MAX_REQUESTS, smoke=True)

    # One smoke request + 40 unique question prompts, all against the same reader/budget.
    assert len(client.prompts) == 41
    assert client.prompts[0] == mod.smoke_prompt()
    assert report["requests_total"] == 41
    assert report["smoke_requests"] == 1
    assert report["generated_this_run"] == 41
    assert report["cache_entries"] == 41
    assert len(report["usage"]) == 41

    # Every non-smoke prompt is exactly its question plus its BM25 top-1 passage, nothing else.
    for row, prompt in zip(rows, client.prompts[1:]):
        root_id = rankings[row["id"]]["bm25"][0]
        expected = mod.reader_prompt(
            row["question"], [{"context_id": root_id, "text": docs[root_id]["text"]}])
        assert prompt == expected

    # The synthetic client echoes the gold answer and cites the shown root -> full credit.
    assert report["metrics"] == {"n": 40, "mean_f1": 1.0, "answer_rate": 1.0,
                                 "citation_valid_rate": 1.0, "gated_success_rate": 1.0}
    assert report["data"]["official_test_loaded"] is False
    assert report["data"]["official_validation_used"] is False
    assert report["data"]["gold_evidence_in_prompt"] is False

    # The report is persisted, and neither the credential nor a local reader name leaks in.
    text = (tmp_path / mod.REPORT_NAME).read_text(encoding="utf-8")
    assert "test-key" not in text and "AIBOX_API_KEY" not in text
    assert json.loads(text)["metrics"]["n"] == 40


def test_run_shares_the_request_cap_with_the_smoke(tmp_path, monkeypatch, real_identity):
    monkeypatch.setenv("AIBOX_API_KEY", "test-key")
    reader = CloudReader(tmp_path / "cache.json", identity=real_identity, max_requests=2,
                         client=_FakeClient())
    with pytest.raises(RuntimeError):
        mod.run(out=tmp_path, reader=reader, max_requests=2, smoke=True)
    assert reader.requests == 0  # insufficient budget fails before any network request


# --- the pilot never imports a local reader or the training module -------------------------

def test_module_does_not_import_a_local_reader_or_training_module():
    source = Path(mod.__file__).read_text(encoding="utf-8")
    for forbidden in ("transformers", "vixstory_reader_train", "AutoModelForCausalLM"):
        assert forbidden not in source

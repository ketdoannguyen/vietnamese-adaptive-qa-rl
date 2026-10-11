"""Pilot ranking builder: label-free, deterministic, complete, non-overwriting.

Every test is synthetic and CPU-only — a deterministic stand-in encoder replaces E5 so no
model is loaded and no GPU is touched. The two contracts under test are that retrieval never
reads an answer or a gold passage id (validation included), and that the artifacts cover the
whole frozen manifest and refuse to overwrite an existing file.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import vixstory_pilot_rankings as mod
from src.retrieval.bm25 import BM25Index
from src.utils.hashing import sha256_file, sha256_text, sha256_json


class FakeEncoder:
    """Deterministic, dependency-free stand-in for ``DenseEncoder`` (no model, no GPU)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.passage_calls = 0
        self.query_calls = 0

    def _encode(self, texts):
        rows = []
        for text in texts:
            rng = np.random.default_rng(int(sha256_text(text)[:16], 16))
            vector = rng.standard_normal(self.dim).astype(np.float32)
            rows.append(vector / np.linalg.norm(vector))
        return np.vstack(rows) if rows else np.zeros((0, self.dim), dtype=np.float32)

    def encode_passages(self, texts):
        self.passage_calls += 1
        return self._encode(texts)

    def encode_queries(self, texts):
        self.query_calls += 1
        return self._encode(texts)


class NoLabels(dict):
    """A row that raises if any label column is read."""

    def __getitem__(self, key):
        if key in mod.LABEL_FIELDS:
            raise AssertionError(f"label field {key!r} was read during retrieval")
        return super().__getitem__(key)

    def __contains__(self, key):
        if key in mod.LABEL_FIELDS:
            raise AssertionError(f"label field {key!r} was probed during retrieval")
        return super().__contains__(key)


def corpus(n: int = 20) -> dict:
    return {f"CTX-{i:02d}": f"tài liệu số {i} về trường đại học Việt Nam" for i in range(n)}


def manifest_dict(pilot, dev, val, *, revision="deadbeef"):
    def split(qids):
        qids = sorted(qids)
        return {"n": len(qids), "qids": qids, "qids_sha256": sha256_json(qids)}

    return {"schema_version": 1, "run_id": "synthetic-pilot", "issue": 30,
            "source": {"resolved_revision": revision},
            "splits": {"pilot_300": split(pilot), "train_dev_200": split(dev),
                       "validation_246": split(val)}}


# --- question selection is label-free ------------------------------------------------

def test_select_questions_never_reads_answers_or_gold_ids():
    rows = [NoLabels(id="qa_1", question="Thủ đô là gì?", answer="Hà Nội",
                     context_ids=["CTX-01"], contexts=[{"context_id": "CTX-01", "text": "x"}])]
    assert mod.select_questions(rows, ["qa_1"]) == {"qa_1": "Thủ đô là gì?"}


def test_select_questions_errors_on_a_missing_manifest_qid():
    with pytest.raises(ValueError, match="absent"):
        mod.select_questions([{"id": "qa_1", "question": "x"}], ["qa_1", "qa_2"])


# --- rankings are complete, deterministic and scored over the full corpus ------------

def test_build_rankings_is_complete_deterministic_and_uses_the_whole_corpus():
    docs = corpus()
    questions = {"qa_2": "câu hỏi hai", "qa_1": "câu hỏi một", "qa_3": "câu hỏi ba"}
    first = mod.build_rankings(docs, questions, top_n=mod.DEFAULT_TOP_N, encoder=FakeEncoder())
    second = mod.build_rankings(docs, questions, top_n=mod.DEFAULT_TOP_N, encoder=FakeEncoder())
    assert first == second
    assert set(first) == set(questions)
    for lists in first.values():
        assert set(lists) == {"bm25", "dense"}
        for ranking in lists.values():
            assert len(ranking) == mod.DEFAULT_TOP_N
            assert len(set(ranking)) == len(ranking)          # no duplicate passages
            assert set(ranking) <= set(docs)                  # ids come from the corpus


def test_bm25_ranking_is_the_statistical_retriever_over_the_full_corpus():
    docs = corpus()
    question = "trường đại học Việt Nam số 7"
    expected = [doc_id for doc_id, _ in BM25Index.build(docs).rank(question, top_n=mod.MIN_TOP_N)]
    produced = mod.build_rankings(docs, {"qa_1": question}, top_n=mod.MIN_TOP_N,
                                  encoder=FakeEncoder())["qa_1"]["bm25"]
    assert produced == expected


def test_rankings_do_not_depend_on_answers_or_gold_passages():
    docs = corpus()
    clean = [{"id": "qa_1", "question": "câu hỏi một"}]
    leaky = [dict(clean[0], answer="ĐÁP ÁN BÍ MẬT", context_ids=["CTX-19"],
                  contexts=[{"context_id": "CTX-19", "text": "gold"}])]
    q_clean = mod.select_questions(clean, ["qa_1"])
    q_leaky = mod.select_questions(leaky, ["qa_1"])
    assert mod.build_rankings(docs, q_clean, encoder=FakeEncoder()) == \
        mod.build_rankings(docs, q_leaky, encoder=FakeEncoder())


# --- top_n floor and corpus sanity ---------------------------------------------------

def test_build_rankings_rejects_a_ranking_shorter_than_the_reachability_floor():
    with pytest.raises(ValueError, match="too short"):
        mod.build_rankings(corpus(), {"qa_1": "x"}, top_n=mod.MIN_TOP_N - 1, encoder=FakeEncoder())


def test_build_rankings_rejects_a_corpus_smaller_than_top_n():
    with pytest.raises(ValueError, match="fewer than"):
        mod.build_rankings({f"CTX-{i}": "x" for i in range(5)}, {"qa_1": "x"},
                           top_n=mod.MIN_TOP_N, encoder=FakeEncoder())


# --- manifest validation -------------------------------------------------------------

def test_load_manifest_rejects_a_hash_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = manifest_dict(["qa_1"], ["qa_2"], ["qa_3"])
    manifest["splits"]["pilot_300"]["qids_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        mod.load_manifest(path)


# --- output files are never overwritten ----------------------------------------------

def test_write_outputs_refuses_to_overwrite(tmp_path):
    (tmp_path / mod.rankings_name(mod.DEFAULT_TOP_N)).write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        mod.write_outputs(tmp_path, {"qa_1": {"bm25": [], "dense": []}}, {}, top_n=mod.DEFAULT_TOP_N)


# --- end-to-end run over a synthetic manifest (no GPU) -------------------------------

def write_synthetic_inputs(tmp_path, *, pilot, dev, val):
    docs = corpus()
    pq.write_table(pa.Table.from_pylist(
        [{"context_id": cid, "school_id": "S", "text": text} for cid, text in docs.items()]),
        tmp_path / "corpus.parquet")

    def rows(qids, split_name):
        return [{"id": qid, "question": f"câu hỏi cho {qid}", "answer": f"SECRET-{split_name}",
                 "context_ids": ["CTX-00"], "hop_level": "1", "question_type": "factual"}
                for qid in qids]

    pq.write_table(pa.Table.from_pylist(rows(pilot + dev, "train")), tmp_path / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows(val, "val")), tmp_path / "validation.parquet")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_dict(pilot, dev, val)), encoding="utf-8")
    return docs, manifest_path


def test_train_only_rankings_never_read_or_hash_validation(tmp_path, monkeypatch):
    pilot, dev, val = ["qa_1"], ["qa_2"], ["va_1"]
    _, manifest_path = write_synthetic_inputs(tmp_path, pilot=pilot, dev=dev, val=val)
    validation = tmp_path / "validation.parquet"
    original_read = mod.pq.read_table
    monkeypatch.setattr(mod.pq, "read_table", lambda path, **kw:
                        pytest.fail("validation read") if str(path) == str(validation)
                        else original_read(path, **kw))
    original_hash = mod.sha256_file
    monkeypatch.setattr(mod, "sha256_file", lambda path:
                        pytest.fail("validation hash") if str(path) == str(validation)
                        else original_hash(path))
    result = mod.run(manifest_path=manifest_path, corpus_path=tmp_path / "corpus.parquet",
                     train_path=tmp_path / "train.parquet", validation_path=validation,
                     out_dir=tmp_path, encoder=FakeEncoder(),
                     expected_corpus_sha256=sha256_file(tmp_path / "corpus.parquet"))
    assert result["n_qids"] == len(pilot) + len(dev)
    provenance = json.loads((tmp_path / mod.PROVENANCE_NAME).read_text("utf-8"))
    assert set(provenance["splits"]) == {"pilot_300", "train_dev_200"}
    assert "validation_sha256" not in provenance["source"]


def test_validation_opt_in_requires_validation_qids_before_qa_read(tmp_path, monkeypatch):
    _, manifest_path = write_synthetic_inputs(tmp_path, pilot=["qa_1"], dev=["qa_2"],
                                              val=["va_1"])
    manifest = json.loads(manifest_path.read_text("utf-8"))
    del manifest["splits"]["validation_246"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(mod.pq, "read_table", lambda *_a, **_kw: pytest.fail("QA read"))
    with pytest.raises(ValueError, match="validation_246"):
        mod.run(manifest_path=manifest_path, corpus_path=tmp_path / "corpus.parquet",
                train_path=tmp_path / "train.parquet", out_dir=tmp_path,
                include_validation=True, encoder=FakeEncoder())


def test_run_writes_label_free_artifacts_and_refuses_a_second_run(tmp_path):
    pilot, dev, val = ["qa_1", "qa_2", "qa_3"], ["qa_4", "qa_5"], ["va_1", "va_2"]
    docs, manifest_path = write_synthetic_inputs(tmp_path, pilot=pilot, dev=dev, val=val)
    result = mod.run(manifest_path=manifest_path, corpus_path=tmp_path / "corpus.parquet",
                     train_path=tmp_path / "train.parquet",
                     validation_path=tmp_path / "validation.parquet", out_dir=tmp_path,
                     encoder=FakeEncoder(), include_validation=True,
                     expected_corpus_sha256=sha256_file(tmp_path / "corpus.parquet"))

    assert result["n_qids"] == len(pilot) + len(dev) + len(val)
    assert result["corpus_docs"] == len(docs)
    assert result["top_n"] == mod.DEFAULT_TOP_N

    rankings = json.loads((tmp_path / mod.rankings_name(mod.DEFAULT_TOP_N)).read_text("utf-8"))
    assert set(rankings) == set(pilot) | set(dev) | set(val)
    for lists in rankings.values():
        assert len(lists["bm25"]) == len(lists["dense"]) == mod.DEFAULT_TOP_N
        assert set(lists["bm25"] + lists["dense"]) <= set(docs)

    provenance = json.loads((tmp_path / mod.PROVENANCE_NAME).read_text("utf-8"))
    assert provenance["top_n"] == mod.DEFAULT_TOP_N
    assert provenance["n_qids"] == result["n_qids"]
    assert provenance["corpus"] == {"n_docs": len(docs), "doc_ids_sha256": sha256_json(sorted(docs))}
    assert provenance["legacy_train500_reuse"]["reused"] is False
    assert set(provenance["splits"]) == set(mod.SPLITS)
    assert provenance["no_gold_leakage"]

    # No reference answer may appear anywhere in the ranking artifact.
    assert "SECRET-train" not in (tmp_path / mod.rankings_name(mod.DEFAULT_TOP_N)).read_text("utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        mod.run(manifest_path=manifest_path, corpus_path=tmp_path / "corpus.parquet",
                train_path=tmp_path / "train.parquet",
                validation_path=tmp_path / "validation.parquet", out_dir=tmp_path,
                encoder=FakeEncoder(), include_validation=True,
                expected_corpus_sha256=sha256_file(tmp_path / "corpus.parquet"))


def test_run_rejects_a_corpus_whose_hash_is_not_pinned(tmp_path):
    _, manifest_path = write_synthetic_inputs(tmp_path, pilot=["qa_1"], dev=["qa_2"], val=["va_1"])
    with pytest.raises(ValueError, match="pinned"):
        mod.run(manifest_path=manifest_path, corpus_path=tmp_path / "corpus.parquet",
                train_path=tmp_path / "train.parquet",
                validation_path=tmp_path / "validation.parquet", out_dir=tmp_path,
                encoder=FakeEncoder(), expected_corpus_sha256="0" * 64)

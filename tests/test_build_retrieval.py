"""P2 pipeline wiring: inputs -> cached rankings -> reachability -> report.

Runs against a synthetic dataset written into ``tmp_path`` (never the real data dirs) with
the deterministic stub encoder standing in for E5, so the whole stage is exercised without
a GPU or a model download.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data import artifacts
from src.data.scenarios import scenario_id_for
from src.pipeline import build_retrieval as br
from src.utils.config import load_config
from src.utils.io import read_jsonl, write_jsonl, write_parquet
from src.utils.runs import RunRegistry
from tests import retrieval_stub

SPLIT = "train_strict"


def scenario_id(qid: str) -> str:
    return scenario_id_for(SPLIT, qid)


DOCS = {k: v for k, v in retrieval_stub.DOCS.items() if k != "doc_outside"}
QUESTIONS = {"q:capital": retrieval_stub.QUESTION, "q:pho": "Phở là gì?",
             "q:noanswer": "Ai là tổng thống Pháp?"}
SCENARIOS = [
    {"scenario_id": scenario_id("q:capital"), "qid": "q:capital", "split": SPLIT,
     "corpus_doc_ids": ["doc_hanoi", "doc_hcm", "doc_pho"], "scenario_seed": 42,
     "gold_doc_id": "doc_hanoi", "reachable_bm25": None, "reachable_dense": None,
     "difficulty": {"answerable": True}},
    # Three documents minimum: with N=2 a term present in exactly one document gets an IDF
    # of zero and BM25 ranks by document length noise instead of by the query terms.
    {"scenario_id": scenario_id("q:pho"), "qid": "q:pho", "split": SPLIT,
     "corpus_doc_ids": ["doc_pho", "doc_hanoi", "doc_river"], "scenario_seed": 42,
     "gold_doc_id": "doc_pho", "reachable_bm25": None, "reachable_dense": None,
     "difficulty": {"answerable": True}},
    {"scenario_id": scenario_id("q:noanswer"), "qid": "q:noanswer", "split": SPLIT,
     "corpus_doc_ids": ["doc_river", "doc_hcm"], "scenario_seed": 42,
     "gold_doc_id": None, "reachable_bm25": None, "reachable_dense": None,
     "difficulty": {"answerable": False}},
]


class _StubEncoder(retrieval_stub.StubEncoder):
    """Accepts DenseEncoder's keyword arguments and never touches a model."""

    def __init__(self, **_kwargs) -> None:
        super().__init__()

    def unload(self) -> None:
        return None


@pytest.fixture()
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(br, "DenseEncoder", _StubEncoder)
    conf = load_config(None)
    processed, scenarios = tmp_path / "processed", tmp_path / "scenarios"
    processed.mkdir(), scenarios.mkdir()
    write_parquet(processed / f"docs_{SPLIT}.parquet",
                  [{"doc_id": k, "text": v} for k, v in DOCS.items()])
    write_parquet(processed / f"questions_{SPLIT}.parquet",
                  [{"qid": q, "question": t} for q, t in QUESTIONS.items()])
    write_jsonl(artifacts.scenarios_path(scenarios, SPLIT), SCENARIOS)
    conf["paths"] = {**conf["paths"],
                     "data_processed": str(processed), "data_scenarios": str(scenarios),
                     "data_indexes": str(tmp_path / "indexes"),
                     "data_cache": str(tmp_path / "cache"),
                     "reports": str(tmp_path / "reports"), "tables": str(tmp_path / "reports"),
                     "manifests": str(tmp_path / "manifests"), "runs": str(tmp_path / "runs")}
    conf["data"]["profile"] = "smoke"
    return conf


def _run(cfg, **kwargs):
    return br.run(cfg=cfg, profile="smoke", splits=(SPLIT,),
                  registry=RunRegistry(cfg["paths"]["runs"] + "/index.jsonl"), **kwargs)


def test_pipeline_writes_rankings_reachability_and_report(cfg):
    summary = _run(cfg)
    assert summary["splits"][SPLIT] == {"cache_hits": 0, "ranked": 3, "scenarios": 3}

    rankings = list(read_jsonl(br.rankings_path(cfg, SPLIT)))
    assert len(rankings) == 3
    for row in rankings:
        assert set(row["rankings"]) == {"bm25", "dense"}
        assert row["scores"]["dense"]
        assert row["config_sha"] == summary["config_sha"]

    reach = list(read_jsonl(br.reachability_path(cfg, SPLIT)))
    by_id = {r["scenario_id"]: r for r in reach}
    negative = by_id[scenario_id("q:noanswer")]
    assert negative["answerable"] is False
    assert negative["reachable_bm25"] is None and negative["reachable_dense"] is None
    answerable = [r for r in reach if r["answerable"]]
    assert len(answerable) == 2
    assert all(r["gold_rank_bm25"] == 1 for r in answerable)  # lexical gold match
    assert all(r["gold_rank_dense"] is not None for r in answerable)

    report = summary["report"]
    assert report["rankings_total"] == 3
    assert report["n_answerable"] == 2
    assert report["recall_at_1"]["bm25"] == 1.0
    assert report["recall_at_1"]["dense"] in (0.0, 1.0)
    assert report["reach_budget"] == {"bm25": 3, "dense": 2}
    # the gold-rank histogram covers answerable scenarios only
    assert sum(report["gold_rank_bm25_histogram"].values()) == report["n_answerable"] == 2
    assert report["unreachable_rate"] == 0.0


def test_missing_queries_are_batched_without_changing_rankings(cfg, monkeypatch):
    calls = []
    original = _StubEncoder.encode_queries

    def count_queries(self, texts):
        calls.append(list(texts))
        return original(self, texts)

    monkeypatch.setattr(_StubEncoder, "encode_queries", count_queries)
    summary = _run(cfg)
    assert summary["splits"][SPLIT]["ranked"] == len(SCENARIOS)
    assert calls == [[QUESTIONS[s["qid"]] for s in SCENARIOS]]

    actual = {r["scenario_id"]: r for r in read_jsonl(br.rankings_path(cfg, SPLIT))}
    docs = br.load_docs(cfg, (SPLIT,))
    bank, _ = br.DenseBank.load(str(br.bank_path(cfg)))
    baseline = {
        s["scenario_id"]: br.rank_scenario(
            scenario_id=s["scenario_id"], qid=s["qid"], question=QUESTIONS[s["qid"]],
            docs=docs, corpus_doc_ids=s["corpus_doc_ids"], dense_bank=bank,
            dense_encoder=_StubEncoder(), top_n=cfg["retrieval"]["bm25"]["top_n_cached"],
            bm25_k1=cfg["retrieval"]["bm25"]["k1"],
            bm25_b=cfg["retrieval"]["bm25"]["b"],
        ).to_row() for s in SCENARIOS
    }
    for key, row in baseline.items():
        assert actual[key]["rankings"] == row["rankings"]
        for tool in ("bm25", "dense"):
            for doc_id, score in row["scores"][tool].items():
                assert actual[key]["scores"][tool][doc_id] == pytest.approx(score, abs=1e-6)


def test_bad_batched_query_shape_refuses_before_writing_rankings(cfg, monkeypatch):
    def bad_queries(self, texts):
        return self.encode_passages(["one"])  # the returned row count cannot match todo

    monkeypatch.setattr(_StubEncoder, "encode_queries", bad_queries)
    with pytest.raises(ValueError, match="dense query embeddings"):
        _run(cfg)
    assert not br.rankings_path(cfg, SPLIT).exists()


def test_cached_rows_encode_only_missing_questions(cfg, monkeypatch):
    _run(cfg)
    path = br.rankings_path(cfg, SPLIT)
    rows = list(read_jsonl(path))
    missing = rows.pop()
    write_jsonl(path, rows)
    calls = []
    original = _StubEncoder.encode_queries

    def count_queries(self, texts):
        calls.append(list(texts))
        return original(self, texts)

    monkeypatch.setattr(_StubEncoder, "encode_queries", count_queries)
    result = _run(cfg)
    assert result["splits"][SPLIT] == {"cache_hits": 2, "ranked": 1, "scenarios": 3}
    assert calls == [[QUESTIONS[missing["qid"]]]]


def test_second_run_is_a_cache_hit_and_cache_only_reads_it(cfg):
    _run(cfg)
    second = _run(cfg)
    assert second["splits"][SPLIT]["ranked"] == 0
    assert second["splits"][SPLIT]["cache_hits"] == 3

    only_cached = _run(cfg, cache_only=True)
    assert only_cached["splits"][SPLIT]["ranked"] == 0
    assert only_cached["dense_bank"] is None  # nothing loaded the encoder
    assert only_cached["report"]["rankings_total"] == 3


def test_cache_only_without_cache_fails_loudly(cfg):
    with pytest.raises(FileNotFoundError, match="cache-only"):
        _run(cfg, cache_only=True)


def test_dense_revision_change_invalidates_rankings_and_bank(cfg):
    first = _run(cfg)
    cfg["retrieval"]["dense"]["revision"] = "new-revision"
    assert br.retrieval_config_sha(cfg) != first["config_sha"]
    second = _run(cfg)
    assert second["splits"][SPLIT]["ranked"] == 3
    assert second["dense_bank"]["rebuilt"] is True
    assert second["dense_bank"]["revision"] == "new-revision"


def test_changing_the_retrieval_config_invalidates_cached_rows(cfg):
    _run(cfg)
    cfg["retrieval"]["bm25"]["k1"] = 1.2
    assert _run(cfg)["splits"][SPLIT]["ranked"] == 3


def test_query_encoder_uses_revision_of_bank(cfg, monkeypatch):
    seen_revisions = []
    initial = br.DenseEncoder

    class TrackingEncoder(initial):
        def __init__(self, **kwargs):
            seen_revisions.append(kwargs.get("revision"))
            super().__init__(**kwargs)

    cfg["retrieval"]["dense"]["revision"] = "pinned-revision"
    monkeypatch.setattr(br, "DenseEncoder", TrackingEncoder)
    _run(cfg)
    assert seen_revisions == ["pinned-revision", "pinned-revision"]


def test_changed_document_text_invalidates_bank_and_cached_rankings(cfg):
    _run(cfg)
    processed = Path(cfg["paths"]["data_processed"])
    changed = dict(DOCS)
    changed["doc_hanoi"] += " Câu bổ sung."
    write_parquet(processed / f"docs_{SPLIT}.parquet",
                  [{"doc_id": k, "text": v} for k, v in changed.items()])
    result = _run(cfg)
    assert result["dense_bank"]["rebuilt"] is True
    assert result["splits"][SPLIT] == {"cache_hits": 1, "ranked": 2, "scenarios": 3}


def test_changed_question_text_invalidates_only_its_ranking(cfg):
    _run(cfg)
    updated = dict(QUESTIONS)
    updated["q:capital"] += " Mới?"
    write_parquet(Path(cfg["paths"]["data_processed"]) / f"questions_{SPLIT}.parquet",
                  [{"qid": qid, "question": text} for qid, text in updated.items()])
    result = _run(cfg)
    assert result["dense_bank"]["rebuilt"] is False
    assert result["splits"][SPLIT] == {"cache_hits": 2, "ranked": 1, "scenarios": 3}
    cache = {row["qid"]: row for row in read_jsonl(br.rankings_path(cfg, SPLIT))}
    assert cache["q:capital"]["question_sha"]


def test_cache_only_refuses_rankings_after_question_text_change(cfg):
    _run(cfg)
    updated = dict(QUESTIONS)
    updated["q:capital"] += " Mới?"
    write_parquet(Path(cfg["paths"]["data_processed"]) / f"questions_{SPLIT}.parquet",
                  [{"qid": qid, "question": text} for qid, text in updated.items()])
    with pytest.raises(FileNotFoundError, match="cache-only"):
        _run(cfg, cache_only=True)


def test_cache_only_refuses_rankings_after_document_text_change(cfg):
    _run(cfg)
    changed = dict(DOCS)
    changed["doc_hanoi"] += " New passage text."
    write_parquet(Path(cfg["paths"]["data_processed"]) / f"docs_{SPLIT}.parquet",
                  [{"doc_id": k, "text": v} for k, v in changed.items()])
    with pytest.raises(FileNotFoundError, match="cache-only"):
        _run(cfg, cache_only=True)


def test_a_cached_row_for_a_changed_corpus_is_rebuilt(cfg):
    """D011: the same scenario_id can get a different corpus at a different profile cap.

    A cached ranking is reused only while its corpus fingerprint matches; for the changed
    scenario the old row (no fingerprint) must be re-ranked over the new corpus.
    """
    _run(cfg)
    rows = list(read_jsonl(br.rankings_path(cfg, SPLIT)))
    assert all(row["corpus_sha"] for row in rows)

    scenarios_path = artifacts.scenarios_path(Path(cfg["paths"]["data_scenarios"]), SPLIT)
    scenarios = list(read_jsonl(scenarios_path))
    changed = scenarios[0]
    changed["corpus_doc_ids"] = ["doc_hanoi", "doc_hcm", "doc_river"]
    write_jsonl(scenarios_path, scenarios)

    second = _run(cfg)
    assert second["splits"][SPLIT] == {"cache_hits": 2, "ranked": 1, "scenarios": 3}
    rebuilt = {row["scenario_id"]: row for row in read_jsonl(br.rankings_path(cfg, SPLIT))}
    assert rebuilt[changed["scenario_id"]]["corpus_sha"] != \
        next(r for r in rows if r["scenario_id"] == changed["scenario_id"])["corpus_sha"]
    assert "doc_river" in rebuilt[changed["scenario_id"]]["rankings"]["bm25"]


def test_superseded_scenario_versions_are_pruned_from_the_cache(cfg):
    """A row whose scenario id carries an older version can never be served again (ids are
    versioned), and keeping it makes two rows share one qid — which ``cache_reader`` refuses.
    The next run must drop it and rewrite the file even when nothing needs re-ranking.
    """
    _run(cfg)
    path = br.rankings_path(cfg, SPLIT)
    rows = list(read_jsonl(path))
    old = dict(rows[0], scenario_id=f"sc:{SPLIT}:{rows[0]['qid']}:v0")
    write_jsonl(path, [old, *rows])
    before = path.stat().st_mtime_ns

    second = _run(cfg)
    assert second["splits"][SPLIT] == {"cache_hits": 3, "ranked": 0, "scenarios": 3}
    assert path.stat().st_mtime_ns != before  # pruned rows reached disk
    kept = list(read_jsonl(path))
    assert "v0" not in "".join(row["scenario_id"] for row in kept)
    qids = [row["qid"] for row in kept]
    assert len(qids) == len(set(qids)) == 3  # cache_reader's qid-unique contract holds


def test_force_rebuilds_the_dense_bank(cfg):
    first = _run(cfg)
    assert first["dense_bank"]["rebuilt"] is True
    assert _run(cfg)["dense_bank"]["rebuilt"] is False
    assert _run(cfg, force=True)["dense_bank"]["rebuilt"] is True


def test_report_tables_and_registry_entries(cfg):
    summary = _run(cfg)
    recall_md = br.resolve(cfg, "tables") / "retrieval_recall_at_k.md"
    examples_md = br.resolve(cfg, "tables") / "bm25_vs_dense_examples.md"
    assert recall_md.exists() and "recall@1" in recall_md.read_text(encoding="utf-8")
    assert examples_md.exists() and "BM25 vs E5" in examples_md.read_text(encoding="utf-8")

    registry = RunRegistry(cfg["paths"]["runs"] + "/index.jsonl")
    rows = list(br.read_jsonl(registry.path))
    assert [r["status"] for r in rows] == ["RUNNING", "SUCCEEDED"]
    artifacts = rows[-1]["artifacts"]
    for path in summary["artifacts"]:
        assert artifacts[path]["sha256"]  # every artifact is checksummed


def test_report_json_is_valid_and_records_the_bank(cfg):
    summary = _run(cfg)
    report = json.loads((br.resolve(cfg, "reports") / "retrieval_report.json").read_text("utf-8"))
    assert report["dense_bank"]["docs"] == len(DOCS)
    assert report["dense_bank"]["dim"] == retrieval_stub.DIM
    assert "library_versions" in report
    assert report["unreachable_examples"] == []

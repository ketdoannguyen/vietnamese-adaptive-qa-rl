"""VixSTORY V2 prepare stages (preflight + rankings): strict IDs, privacy, no overwrite.

All inputs are synthetic parquet files written into tmp dirs — the real 2,553/246 rows, the
real pinned digests and the GPU model are never touched here. The contracts under test:

* the preflight reads ``id``/``question`` (public) and ``id``/``context_ids`` (private) in
  separate projected reads and never requests ``answer``/``contexts``;
* non-pinned sources (csv, ``test.parquet``) are refused by filename before any read;
* counts/IDs/non-empty/unique/in-corpus checks fail loudly and leave no partial artifacts;
* the private gold file is written with 0600 permissions and public artifacts carry no
  labels; exact/near content duplicates are reported without changing the splits;
* rankings write the unified ``rankings_top12.json`` with hashes, refuse overwrite, never
  read the private gold and are deterministic for a deterministic encoder.
"""

from __future__ import annotations

import stat

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import vixstory_v2_prepare as mod
from src.pipeline.vixstory_worlds_v2 import TOP_N, private_gold_name, public_questions_name
from src.retrieval.bm25 import BM25Index
from src.utils.hashing import sha256_file, sha256_json, sha256_text
from src.utils.io import read_json

PIN = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"

CORPUS = {f"CTX-{i:02d}": f"đoạn văn số {i} về một trường đại học Việt Nam" for i in range(20)}
TRAIN = {
    "qa_1": ("Câu hỏi về trường A?", ["CTX-00"]),
    "qa_2": ("Trường B có bao nhiêu khoa?", ["CTX-01", "CTX-02"]),
    "qa_3": ("Trường C ở đâu?", ["CTX-03"]),
    "qa_4": ("Trường D thành lập năm nào?", ["CTX-05"]),
    "qa_5": ("Trường E có bao nhiêu sinh viên?", ["CTX-07"]),
}
VALIDATION = {
    "va_1": ("Câu hỏi kiểm tra một?", ["CTX-10"]),
    "va_2": ("Câu hỏi kiểm tra hai?", ["CTX-11"]),
    "va_3": ("Câu hỏi kiểm tra ba?", ["CTX-12"]),
}


def _rows(mapping):
    return [{"id": qid, "question": question, "answer": f"SECRET-{qid}",
             "context_ids": list(gold)} for qid, (question, gold) in mapping.items()]


class FakeEncoder:
    """Deterministic dim-8 stand-in; the rankings stage only needs the encode methods."""

    effective_batch_size = None

    def _encode(self, texts):
        rows = []
        for text in texts:
            seed = int(sha256_text(text)[:16], 16)
            vec = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
            rows.append(vec / np.linalg.norm(vec))
        return np.vstack(rows)

    def encode_passages(self, texts):
        return self._encode(texts)

    def encode_queries(self, texts):
        return self._encode(texts)

    def dimension(self):
        return 8


def make_inputs(tmp_path, *, train_rows=None, validation_rows=None):
    """Write the three pinned-style parquet files; return paths, digests and row counts."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_rows = _rows(TRAIN) if train_rows is None else train_rows
    validation_rows = _rows(VALIDATION) if validation_rows is None else validation_rows
    corpus_path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(
        [{"context_id": cid, "school_id": "S", "text": text} for cid, text in CORPUS.items()]),
        corpus_path)
    train_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows), train_path)
    validation_path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(validation_rows), validation_path)
    paths = {"corpus": corpus_path, "train": train_path, "validation": validation_path}
    digests = {role: sha256_file(path) for role, path in paths.items()}
    rows = {"corpus": len(CORPUS), "train": len(train_rows), "validation": len(validation_rows)}
    return paths, digests, rows


def run_stage(tmp_path, *, train_rows=None, validation_rows=None, run_name="run", **overrides):
    """Run the preflight over synthetic inputs; ``run_name`` isolates negative cases."""
    paths, digests, rows = make_inputs(tmp_path, train_rows=train_rows,
                                       validation_rows=validation_rows)
    run_dir = tmp_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = overrides.pop("expected_rows", rows)
    manifest = mod.run_preflight(run_dir, corpus_path=paths["corpus"], train_path=paths["train"],
                                 validation_path=paths["validation"], expected_sha256=digests,
                                 expected_rows=expected_rows, **overrides)
    return manifest, run_dir


# --- preflight happy path ----------------------------------------------------------------

def test_preflight_writes_public_private_and_manifest(tmp_path):
    manifest, run_dir = run_stage(tmp_path)

    public = read_json(run_dir / public_questions_name("train"))
    assert set(public) == {"split", "n", "qids", "qids_sha256", "questions", "questions_sha256"}
    assert public["split"] == "train" and public["n"] == len(TRAIN)
    assert public["qids"] == sorted(TRAIN)
    assert public["qids_sha256"] == sha256_json(sorted(TRAIN))
    assert public["questions"] == {qid: question for qid, (question, _) in TRAIN.items()}

    private = read_json(run_dir / private_gold_name("train"))
    assert private["gold"] == {qid: sorted(gold) for qid, (_, gold) in TRAIN.items()}
    assert private["gold_sha256"] == sha256_json(private["gold"])

    assert manifest["stage"] == "preflight" and manifest["run_id"] == run_dir.name
    assert manifest["model_pin"]["revision"] == mod.MODEL_REVISION == PIN
    assert manifest["parameters"]["top_n"] == TOP_N == 12
    assert manifest["inputs"]["dataset"]["official_test_loaded"] is False
    assert manifest["inputs"]["train"]["sha256"] == sha256_file(tmp_path / "train.parquet")
    assert manifest["inputs"]["validation"]["rows"] == len(VALIDATION)
    assert manifest["splits"]["train"]["n"] == len(TRAIN)
    assert manifest["splits"]["train"]["qids_sha256"] == sha256_json(sorted(TRAIN))
    assert manifest["disjointness"] == {"train_validation": 0, "all_disjoint": True}
    assert manifest["time"]["seconds"] >= 0

    for name, entry in manifest["outputs"].items():
        assert entry["sha256"] == sha256_file(run_dir / name)
        assert entry["bytes"] == (run_dir / name).stat().st_size


def test_public_artifacts_carry_no_labels(tmp_path):
    _, run_dir = run_stage(tmp_path)
    for split in ("train", "validation"):
        text = (run_dir / public_questions_name(split)).read_text(encoding="utf-8")
        assert "SECRET" not in text and "CTX-" not in text and "answer" not in text


def test_private_gold_files_are_owner_only(tmp_path):
    _, run_dir = run_stage(tmp_path)
    for split in ("train", "validation"):
        mode = stat.S_IMODE((run_dir / private_gold_name(split)).stat().st_mode)
        assert mode == 0o600


def test_preflight_refuses_to_overwrite_artifacts(tmp_path):
    manifest, run_dir = run_stage(tmp_path)
    public_path = run_dir / public_questions_name("train")
    before = public_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        run_stage(tmp_path)
    assert public_path.read_text(encoding="utf-8") == before


# --- column projection and source pinning ------------------------------------------------

def test_preflight_never_projects_answer_or_contexts_columns(tmp_path, monkeypatch):
    seen = []
    original = mod.pq.read_table

    def spy(path, **kwargs):
        columns = tuple(kwargs.get("columns") or ())
        seen.append(columns)
        for forbidden in ("answer", "contexts"):
            assert forbidden not in columns
        return original(path, **kwargs)

    monkeypatch.setattr(mod.pq, "read_table", spy)
    run_stage(tmp_path)
    assert ("id", "question") in seen
    assert ("id", "context_ids") in seen
    assert ("context_id", "text") in seen


def test_preflight_rejects_non_pinned_sources(tmp_path):
    csv_path = tmp_path / "qa.csv"
    csv_path.write_text("id,question\n", encoding="utf-8")
    test_path = tmp_path / "test.parquet"
    pq.write_table(pa.Table.from_pylist(_rows(TRAIN)), test_path)
    with pytest.raises(ValueError, match="refus"):
        mod.read_public_rows(csv_path)
    with pytest.raises(ValueError, match="refus"):
        mod.read_public_rows(test_path)
    # The preflight itself refuses a csv/test source before any read or write.
    paths, digests, rows = make_inputs(tmp_path)
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="refus"):
        mod.run_preflight(run_dir, corpus_path=paths["corpus"], train_path=test_path,
                          validation_path=paths["validation"], expected_sha256=digests,
                          expected_rows=rows)
    assert not run_dir.exists() or not list(run_dir.iterdir())
    with pytest.raises(ValueError, match="unknown pinned source"):
        mod.pinned_source("test")


# --- strict ID / gold validation ---------------------------------------------------------

def test_preflight_rejects_a_corpus_digest_mismatch(tmp_path):
    paths, digests, rows = make_inputs(tmp_path)
    digests["corpus"] = "0" * 64
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="pinned"):
        mod.run_preflight(run_dir, corpus_path=paths["corpus"], train_path=paths["train"],
                          validation_path=paths["validation"], expected_sha256=digests,
                          expected_rows=rows)


def test_preflight_rejects_a_row_count_mismatch(tmp_path):
    _, _, rows = make_inputs(tmp_path)
    rows["train"] += 1
    with pytest.raises(ValueError, match="rows"):
        run_stage(tmp_path, expected_rows=rows)


def test_preflight_rejects_bad_ids_and_questions(tmp_path):
    duplicate = _rows(TRAIN) + [{"id": "qa_1", "question": "khác", "answer": "x",
                                 "context_ids": ["CTX-00"]}]
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        run_stage(tmp_path, train_rows=duplicate, run_name="dup")

    empty_question = _rows(TRAIN)
    empty_question[0]["question"] = "   "
    with pytest.raises(ValueError, match="empty"):
        run_stage(tmp_path, train_rows=empty_question, run_name="empty")


def test_preflight_rejects_bad_private_gold(tmp_path):
    cases = {"empty": [], "duplicate": ["CTX-00", "CTX-00"], "outside": ["CTX-99"]}
    for name, bad_gold in cases.items():
        rows = _rows(TRAIN)
        rows[0]["context_ids"] = bad_gold
        with pytest.raises(ValueError):
            run_stage(tmp_path, train_rows=rows, run_name=name)
        assert not list((tmp_path / name).iterdir())     # nothing written on failure


def test_preflight_rejects_overlapping_train_validation_qids(tmp_path):
    validation_rows = _rows({"qa_1": ("Câu hỏi trùng QID?", ["CTX-13"])})
    with pytest.raises(ValueError, match="disjoint"):
        run_stage(tmp_path, validation_rows=validation_rows)


# --- content duplicate report (report only, never a resplit) ----------------------------

def test_duplicate_report_flags_exact_and_near_duplicates_without_resplitting(tmp_path):
    train = {
        "dup_a": ("Câu hỏi trùng nhau?", ["CTX-00"]),
        "dup_b": ("câu HỎI trùng nhau?", ["CTX-01"]),     # exact after NFC+casefold
        "near_a": ("trường đại học bách khoa hà nội thành lập năm nào", ["CTX-02"]),
        "solo": ("Trường Z ở tỉnh nào?", ["CTX-03"]),
    }
    validation = {
        "near_b": ("trường đại học bách khoa hà nội thành lập năm nào vậy", ["CTX-10"]),
        "other": ("Câu hỏi khác hẳn?", ["CTX-11"]),
    }
    _, run_dir = run_stage(tmp_path, train_rows=_rows(train),
                           validation_rows=_rows(validation))
    report = read_json(run_dir / "dup_report.json")

    assert report["splits_unchanged"] is True
    assert report["counts"]["exact_duplicate_groups"] == 1
    group = report["exact_duplicate_groups"][0]
    assert {member["qid"] for member in group["members"]} == {"dup_a", "dup_b"}
    assert report["counts"]["near_duplicate_pairs"] == 1
    pair = report["near_duplicate_pairs"][0]
    assert {pair["a"]["qid"], pair["b"]["qid"]} == {"near_a", "near_b"}
    assert pair["jaccard"] >= 0.9
    assert pair["scope"] == "train_vs_validation"

    # The splits themselves are untouched: the public artifacts keep the original QIDs.
    public = read_json(run_dir / public_questions_name("train"))
    assert public["qids"] == sorted(train)


# --- rankings stage ----------------------------------------------------------------------

def test_rankings_write_unified_top12_and_never_read_gold(tmp_path, monkeypatch):
    _, run_dir = run_stage(tmp_path)
    private_before = sha256_file(run_dir / private_gold_name("train"))
    monkeypatch.setattr(mod, "read_private_rows",
                        lambda *a, **k: pytest.fail("rankings read private gold"))
    monkeypatch.setattr(mod, "download_model",
                        lambda *a, **k: pytest.fail("injected encoder must skip the download"))

    result = mod.run_rankings(run_dir, encoder=FakeEncoder())
    assert result["n_qids"] == len(TRAIN) + len(VALIDATION)
    assert result["top_n"] == TOP_N and result["corpus_docs"] == len(CORPUS)

    rankings = read_json(run_dir / "rankings_top12.json")
    assert set(rankings) == set(TRAIN) | set(VALIDATION)
    corpus_ids = set(CORPUS)
    for entry in rankings.values():
        assert set(entry) == {"bm25", "dense"}
        for ranking in entry.values():
            assert len(ranking) == TOP_N
            assert len(set(ranking)) == TOP_N
            assert set(ranking) <= corpus_ids
    # BM25 is the shared BM25Index over the pinned corpus, not a reimplementation.
    expected = [doc_id for doc_id, _ in
                BM25Index.build(dict(sorted(CORPUS.items()))).rank(TRAIN["qa_1"][0],
                                                                   top_n=TOP_N)]
    assert rankings["qa_1"]["bm25"] == expected

    provenance = read_json(run_dir / "rankings_manifest.json")
    assert provenance["model"]["source"] == "injected"
    assert provenance["preflight_manifest"]["sha256"] == sha256_file(run_dir / "manifest.json")
    assert provenance["outputs"]["rankings_top12.json"]["sha256"] == \
        sha256_file(run_dir / "rankings_top12.json")
    assert provenance["time"]["seconds"] >= 0
    assert {"started_utc", "finished_utc"} <= set(provenance["time"])
    assert "SECRET" not in (run_dir / "rankings_top12.json").read_text(encoding="utf-8")
    assert sha256_file(run_dir / private_gold_name("train")) == private_before

    with pytest.raises(FileExistsError, match="overwrite"):
        mod.run_rankings(run_dir, encoder=FakeEncoder())


def test_rankings_require_the_preflight_manifest(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="preflight"):
        mod.run_rankings(run_dir, encoder=FakeEncoder())


def test_rankings_are_deterministic_across_run_dirs(tmp_path):
    _, first_dir = run_stage(tmp_path / "a")
    _, second_dir = run_stage(tmp_path / "b")
    mod.run_rankings(first_dir, encoder=FakeEncoder())
    mod.run_rankings(second_dir, encoder=FakeEncoder())
    assert (first_dir / "rankings_top12.json").read_text(encoding="utf-8") == \
        (second_dir / "rankings_top12.json").read_text(encoding="utf-8")

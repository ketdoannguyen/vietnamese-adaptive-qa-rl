"""VixSTORY V2 worlds loader: hash-chain verification, strict IDs, lazy env import.

Synthetic run dirs are produced by the real prepare stages (preflight + rankings driven by
an injected encoder), then tampered file by file — the loader must refuse anything that
does not hash-chain back to the preflight manifest, must never touch the answer/contexts
columns or the raw QA parquet files, and must import ``src.rl.vixstory_env_v2`` lazily so
the loader stays importable while the V2 environment is still being built.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import vixstory_v2_prepare as prep
from src.pipeline import vixstory_worlds_v2 as mod
from src.utils.hashing import sha256_file, sha256_json, sha256_text
from src.utils.io import read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]

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


class FakeEncoder:
    """Deterministic dim-8 stand-in for the pinned dense encoder."""

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


def _rows(mapping):
    return [{"id": qid, "question": question, "answer": f"SECRET-{qid}",
             "context_ids": list(gold)} for qid, (question, gold) in mapping.items()]


def build_run(root, *, with_rankings_manifest=True):
    """A complete, consistent run dir: preflight artifacts + unified rankings."""
    root.mkdir(parents=True, exist_ok=True)
    corpus_path = root / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist(
        [{"context_id": cid, "school_id": "S", "text": text} for cid, text in CORPUS.items()]),
        corpus_path)
    train_path = root / "train.parquet"
    pq.write_table(pa.Table.from_pylist(_rows(TRAIN)), train_path)
    validation_path = root / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(_rows(VALIDATION)), validation_path)
    digests = {role: sha256_file(path) for role, path in
               (("corpus", corpus_path), ("train", train_path),
                ("validation", validation_path))}
    rows = {"corpus": len(CORPUS), "train": len(TRAIN), "validation": len(VALIDATION)}
    run_dir = root / "run"
    run_dir.mkdir()
    prep.run_preflight(run_dir, corpus_path=corpus_path, train_path=train_path,
                       validation_path=validation_path, expected_sha256=digests,
                       expected_rows=rows)
    prep.run_rankings(run_dir, encoder=FakeEncoder())
    if not with_rankings_manifest:
        (run_dir / mod.RANKINGS_MANIFEST_NAME).unlink()
    return run_dir


def refresh_output_hash(manifest_path, artifact_path):
    payload = read_json(manifest_path)
    payload["outputs"][artifact_path.name] = {"sha256": sha256_file(artifact_path),
                                              "bytes": artifact_path.stat().st_size}
    write_json(manifest_path, payload)


def rewrite_artifact(manifest_path, artifact_path, mutate):
    """Consistently tamper an artifact: the manifest is refreshed to match the lie."""
    payload = read_json(artifact_path)
    mutate(payload)
    write_json(artifact_path, payload)
    refresh_output_hash(manifest_path, artifact_path)
    return payload


def inject_fake_world(monkeypatch):
    """A fake ``src.rl.vixstory_env_v2`` module so no real V2 env code is required."""
    created = []

    class FakeWorld:
        def __init__(self, qid, question, corpus, rankings, gold_context_ids):
            self.qid = qid
            self.question = question
            self.corpus = corpus
            self.rankings = rankings
            self.gold_context_ids = gold_context_ids
            created.append(self)

    module = types.ModuleType("src.rl.vixstory_env_v2")
    module.VixStoryWorldV2 = FakeWorld
    monkeypatch.setitem(sys.modules, "src.rl.vixstory_env_v2", module)
    return created


# --- happy path ---------------------------------------------------------------------------

def test_load_worlds_builds_one_world_per_qid(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    created = inject_fake_world(monkeypatch)
    worlds = mod.load_worlds(run_dir, "train")
    assert [world.qid for world in worlds] == sorted(TRAIN)
    assert len(created) == len(TRAIN)
    first = worlds[0]
    assert first.question == TRAIN["qa_1"][0]
    assert first.corpus == CORPUS
    assert set(first.rankings) == {"bm25", "dense"}
    assert len(first.rankings["bm25"]) == len(first.rankings["dense"]) == mod.TOP_N == 12
    assert first.gold_context_ids == ("CTX-00",)
    assert worlds[1].gold_context_ids == ("CTX-01", "CTX-02")
    assert len(mod.load_worlds(run_dir, "validation")) == len(VALIDATION)


def test_worlds_loader_reads_only_the_pinned_corpus_parquet(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    seen = []
    original = mod.pq.read_table

    def spy(path, **kwargs):
        seen.append((Path(path).name, tuple(kwargs.get("columns") or ())))
        return original(path, **kwargs)

    monkeypatch.setattr(mod.pq, "read_table", spy)
    mod.load_worlds(run_dir, "train")
    assert seen == [("corpus.parquet", ("context_id", "text"))]


def test_unknown_split_is_rejected(tmp_path):
    run_dir = build_run(tmp_path)
    for bad in ("dev", "test", "TRAIN"):
        with pytest.raises(ValueError, match="unknown split"):
            mod.load_worlds(run_dir, bad)


# --- lazy env import ---------------------------------------------------------------------

def test_env_module_is_imported_lazily():
    code = ("import sys\n"
            "import src.pipeline.vixstory_worlds_v2\n"
            "assert 'src.rl.vixstory_env_v2' not in sys.modules\n"
            "print('lazy-ok')\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "lazy-ok" in proc.stdout


def test_missing_env_module_raises_a_clear_import_error(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    monkeypatch.setitem(sys.modules, "src.rl.vixstory_env_v2", None)
    with pytest.raises(ImportError, match="vixstory_env_v2"):
        mod.load_worlds(run_dir, "train")


# --- artifact tampering ------------------------------------------------------------------

def test_tampered_public_artifact_fails_the_hash_chain(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    path = run_dir / mod.public_questions_name("train")
    payload = read_json(path)
    payload["questions"]["qa_1"] = "Câu hỏi bị sửa?"
    write_json(path, payload)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_worlds(run_dir, "train")


def test_empty_question_is_rejected_even_with_a_consistent_manifest(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)

    def blank_out(payload):
        payload["questions"]["qa_1"] = "   "

    rewrite_artifact(run_dir / mod.MANIFEST_NAME,
                     run_dir / mod.public_questions_name("train"), blank_out)
    with pytest.raises(ValueError, match="empty"):
        mod.load_worlds(run_dir, "train")


def test_bad_gold_is_rejected_even_with_a_consistent_manifest(tmp_path, monkeypatch):
    inject_fake_world(monkeypatch)
    cases = {
        "outside": (["CTX-99"], "corpus"),
        "duplicate": (["CTX-01", "CTX-01"], "duplicate"),
        "empty": ([], "empty"),
    }
    for name, (bad_gold, match) in cases.items():
        run_dir = build_run(tmp_path / name)

        def mutate(payload, bad_gold=bad_gold):
            payload["gold"]["qa_1"] = list(bad_gold)
            payload["gold_sha256"] = sha256_json(payload["gold"])

        rewrite_artifact(run_dir / mod.MANIFEST_NAME,
                         run_dir / mod.private_gold_name("train"), mutate)
        with pytest.raises(ValueError, match=match):
            mod.load_worlds(run_dir, "train")


def test_corpus_drift_is_refused(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    corpus_path = Path(read_json(run_dir / mod.MANIFEST_NAME)["inputs"]["corpus"]["path"])
    rows = pq.read_table(corpus_path).to_pylist()
    rows[0]["text"] += " bị sửa"
    pq.write_table(pa.Table.from_pylist(rows), corpus_path)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_worlds(run_dir, "train")


def test_non_pinned_source_names_are_refused(tmp_path, monkeypatch):
    inject_fake_world(monkeypatch)

    csv_run = build_run(tmp_path / "csv", with_rankings_manifest=False)
    manifest_path = csv_run / mod.MANIFEST_NAME
    manifest = read_json(manifest_path)
    manifest["inputs"]["train"]["path"] = "data/train.csv"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="refus"):
        mod.load_worlds(csv_run, "train")

    test_run = build_run(tmp_path / "official-test", with_rankings_manifest=False)
    manifest_path = test_run / mod.MANIFEST_NAME
    manifest = read_json(manifest_path)
    manifest["inputs"]["validation"]["path"] = "data/test.parquet"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="refus"):
        mod.load_worlds(test_run, "train")


# --- rankings validation ------------------------------------------------------------------

def test_bad_rankings_are_rejected(tmp_path, monkeypatch):
    inject_fake_world(monkeypatch)

    def drop_qid(payload):
        del payload["qa_3"]

    def shorten(payload):
        payload["qa_2"]["dense"] = payload["qa_2"]["dense"][:11]

    def duplicate(payload):
        payload["qa_1"]["bm25"][1] = payload["qa_1"]["bm25"][0]

    def unknown_id(payload):
        payload["qa_4"]["dense"][0] = "NOT-IN-CORPUS"

    cases = {"missing": (drop_qid, "missing"), "short": (shorten, "12"),
             "duplicate": (duplicate, "duplicate"), "unknown": (unknown_id, "corpus")}
    for name, (mutate, match) in cases.items():
        run_dir = build_run(tmp_path / name)
        path = run_dir / mod.RANKINGS_NAME
        payload = read_json(path)
        mutate(payload)
        write_json(path, payload)
        with pytest.raises(ValueError, match=match):
            mod.load_worlds(run_dir, "train")


def test_missing_rankings_manifest_fails_closed(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path, with_rankings_manifest=False)
    inject_fake_world(monkeypatch)
    with pytest.raises(ValueError, match="rankings manifest"):
        mod.load_worlds(run_dir, "train")


def test_manifest_split_qids_must_match_public_artifact(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    manifest = read_json(run_dir / mod.MANIFEST_NAME)
    manifest["splits"]["train"]["qids"] = ["wrong-qid"]
    write_json(run_dir / mod.MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="split.*manifest|manifest.*split"):
        mod.load_worlds(run_dir, "train")


def test_train_source_hash_is_rechecked_without_reading_columns(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    manifest = read_json(run_dir / mod.MANIFEST_NAME)
    train = Path(manifest["inputs"]["train"]["path"])
    with train.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="train.*sha256"):
        mod.load_worlds(run_dir, "train")


def test_extra_rankings_entries_are_allowed(tmp_path, monkeypatch):
    run_dir = build_run(tmp_path)
    inject_fake_world(monkeypatch)
    path = run_dir / mod.RANKINGS_NAME
    payload = read_json(path)
    payload["extra_qid"] = {"bm25": payload["qa_1"]["bm25"], "dense": payload["qa_1"]["dense"]}
    write_json(path, payload)
    refresh_output_hash(run_dir / mod.RANKINGS_MANIFEST_NAME, path)
    assert len(mod.load_worlds(run_dir, "train")) == len(TRAIN)


def test_rankings_manifest_cross_checks(tmp_path, monkeypatch):
    inject_fake_world(monkeypatch)

    # (a) the ranking artifact changed after the rankings manifest was written
    run_dir = build_run(tmp_path / "stale")
    path = run_dir / mod.RANKINGS_NAME
    payload = read_json(path)
    payload["qa_1"]["bm25"] = payload["qa_1"]["bm25"][::-1]
    write_json(path, payload)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_worlds(run_dir, "train")

    # (b) the rankings were built with a different model revision
    run_dir = build_run(tmp_path / "revision")
    rman_path = run_dir / mod.RANKINGS_MANIFEST_NAME
    rman = read_json(rman_path)
    rman["model"]["resolved_revision"] = "0" * 40
    write_json(rman_path, rman)
    with pytest.raises(ValueError, match="revision"):
        mod.load_worlds(run_dir, "train")

    # (c) the rankings point at a different preflight manifest
    run_dir = build_run(tmp_path / "manifest")
    rman_path = run_dir / mod.RANKINGS_MANIFEST_NAME
    rman = read_json(rman_path)
    rman["preflight_manifest"]["sha256"] = "0" * 64
    write_json(rman_path, rman)
    with pytest.raises(ValueError, match="sha256"):
        mod.load_worlds(run_dir, "train")

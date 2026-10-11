"""The pilot split manifest must be deterministic, disjoint and label-free.

Selection is frozen from official train only; the official test split is never read. The
manifest records QIDs and provenance, never a question, answer or passage.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import vixstory_pilot_manifest as mod
from scripts.vixstory_suitability import sample_train
from src.utils.hashing import sha256_file, sha256_json

# Exact field names that would leak labels into the manifest (substring match would wrongly
# reject the legitimate ``question_type`` key).
FORBIDDEN_KEYS = {
    "answer", "answers", "answer_text", "question", "question_text",
    "context", "contexts", "context_ids", "context_text", "passage", "passages", "text",
}


def train_rows(n, *, prefix="qa"):
    hops = [1, 2, 3, 4, 5]
    kinds = ["factual", "comparative", "procedural"]
    return [{"id": f"{prefix}_{i:06d}", "hop_level": str(hops[i % len(hops)]),
             "question_type": kinds[i % len(kinds)]} for i in range(1, n + 1)]


def all_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from all_keys(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from all_keys(value)


def test_select_pilot_is_sized_deterministic_and_disjoint():
    train = train_rows(2553)
    pilot, dev = mod.select_pilot(train)
    pids = [row["id"] for row in pilot]
    dids = [row["id"] for row in dev]
    assert len(pids) == 300 and len(set(pids)) == 300
    assert len(dids) == 200 and len(set(dids)) == 200
    assert set(pids).isdisjoint(dids)
    assert set(pids) | set(dids) <= {row["id"] for row in train}
    # Stable under input reordering: selection depends only on the pinned rows and seeds.
    again = mod.select_pilot(list(reversed(train)))
    assert [row["id"] for row in again[0]] == pids
    assert [row["id"] for row in again[1]] == dids
    assert pids == sorted(pids) and dids == sorted(dids)


def test_sampling_is_seed_sensitive_and_pilot_seed_reproducible():
    train = train_rows(2553)
    pilot, dev = mod.select_pilot(train)
    assert [row["id"] for row in pilot] != [row["id"] for row in
                                            sample_train(train, n=300, seed=43)]
    assert [row["id"] for row in dev] == [row["id"] for row in
                                          mod.select_pilot(train)[1]]
    with pytest.raises(ValueError, match="not enough"):
        sample_train(train, n=len(train) + 1, seed=42)


def test_manifest_records_provenance_disjointness_and_no_labels():
    train = train_rows(600)
    pilot, dev = mod.select_pilot(train, pilot_n=120, pilot_seed=42, dev_n=80, dev_seed=43)
    val = [{"id": f"va_{i:06d}", "hop_level": "1", "question_type": "factual"}
           for i in range(246)]
    manifest = mod.build_manifest(
        revision="deadbeef", source_files={"data/train.parquet": {"sha256": "x", "bytes": 1}},
        raw_rows={"train": 600, "validation": 246}, pilot_rows=pilot, dev_rows=dev,
        val_rows=val)
    assert manifest["source"]["resolved_revision"] == "deadbeef"
    assert manifest["source"]["raw_rows"] == {"train": 600, "validation": 246}
    assert manifest["source"]["official_test_loaded"] is False
    assert manifest["sampling"]["version"] == mod.SAMPLING_VERSION
    assert manifest["sampling"]["sampler"] == "scripts.vixstory_suitability.sample_train"
    assert manifest["selection_exploratory"] is True
    assert "500" in manifest["exploratory_reason"]
    assert manifest["disjointness"] == {
        "pilot_vs_train_dev": 0, "pilot_vs_validation": 0,
        "train_dev_vs_validation": 0, "all_disjoint": True}
    for name, n in (("pilot_300", 120), ("train_dev_200", 80), ("validation_246", 246)):
        split = manifest["splits"][name]
        assert split["n"] == len(split["qids"]) == n
        assert split["qids"] == sorted(split["qids"])
        assert split["qids_sha256"] == sha256_json(split["qids"])
    assert FORBIDDEN_KEYS.isdisjoint(set(all_keys(manifest)))


def test_manifest_rejects_overlapping_splits():
    row = {"id": "qa_1", "hop_level": "1", "question_type": "factual"}
    other = {"id": "va_1", "hop_level": "1", "question_type": "factual"}
    with pytest.raises(ValueError, match="not disjoint"):
        mod.build_manifest(revision="x", source_files={}, raw_rows={"train": 1, "validation": 1},
                           pilot_rows=[row], dev_rows=[row], val_rows=[other])


def test_run_refuses_existing_output(tmp_path):
    out = tmp_path / "manifest.json"
    out.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mod.run(out=out)


def test_train_only_run_never_downloads_reads_or_hashes_validation(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows(mod.TRAIN_COUNT)), train_path)
    out = tmp_path / "train-only.json"
    monkeypatch.setattr(mod, "hf_hub_download",
                        lambda _repo, name, **_kw: str(train_path) if name == "data/train.parquet"
                        else pytest.fail("validation download"))
    original_hash = mod.sha256_file
    monkeypatch.setattr(mod, "sha256_file",
                        lambda path: original_hash(path) if str(path) == str(train_path)
                        else pytest.fail("validation hash"))
    original_read = mod.pq.read_table
    monkeypatch.setattr(mod.pq, "read_table",
                        lambda path, **kw: original_read(path, **kw)
                        if str(path) == str(train_path) else pytest.fail("validation read"))
    monkeypatch.setattr(mod, "EXPECTED_SHA256", {"data/train.parquet": original_hash(train_path)})
    manifest = mod.run(out=out)
    assert set(manifest["splits"]) == {"pilot_300", "train_dev_200"}
    assert set(manifest["source"]["files"]) == {"data/train.parquet"}
    assert "validation" not in manifest["source"]["raw_rows"]
    assert FORBIDDEN_KEYS.isdisjoint(set(all_keys(manifest)))


def test_run_writes_manifest_without_labels_and_is_idempotent(tmp_path, monkeypatch):
    train_path = tmp_path / "train.parquet"
    val_path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist(train_rows(mod.TRAIN_COUNT)), train_path)
    pq.write_table(pa.Table.from_pylist([{"id": f"va_{i:06d}", "hop_level": "1",
                                          "question_type": "factual"}
                                         for i in range(mod.VALIDATION_COUNT)]), val_path)
    mapping = {"data/train.parquet": str(train_path), "data/validation.parquet": str(val_path)}
    monkeypatch.setattr(mod, "hf_hub_download",
                        lambda repo, name, repo_type, revision: mapping[name])
    monkeypatch.setattr(mod, "EXPECTED_SHA256",
                        {name: sha256_file(path) for name, path in mapping.items()})
    out = tmp_path / "manifest.json"
    manifest = mod.run(out=out, include_validation=True)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == manifest
    assert written["splits"]["pilot_300"]["n"] == 300
    assert written["splits"]["train_dev_200"]["n"] == 200
    assert written["disjointness"]["all_disjoint"] is True
    assert FORBIDDEN_KEYS.isdisjoint(set(all_keys(written)))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        mod.run(out=out, include_validation=True)

"""Read-only validation/test lineage checks for train-cap experiments."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_split_identity import verify_split_identity
from src.utils.io import read_json
from tests.fixtures import tiny_raw as fx


def _manifest(root: Path) -> Path:
    return root / "processed" / "split_manifest.json"


def test_train_cap_preserves_frozen_artifacts_without_reading_labels(tmp_path, monkeypatch):
    base, _ = fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    limited, _ = fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                                         caps={"medium": 20}, train_cap=6)
    assert base["strict_split"]["splits"]["train_strict"]["rows_emitted"] > 6
    import pyarrow.parquet as pq

    monkeypatch.setattr(pq, "read_table", lambda *_args, **_kwargs: pytest.fail("read labels"))
    result = verify_split_identity(_manifest(tmp_path / "base"),
                                   _manifest(tmp_path / "limited"))
    assert result["matched_artifacts"] == 8
    assert result["train_rows"] == 6
    assert result["val_rows"] == base["strict_split"]["splits"]["val_strict"]["rows_emitted"]


def test_identity_check_rejects_changed_val_file_or_manifest(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    reference = _manifest(tmp_path / "base")
    candidate = _manifest(tmp_path / "limited")
    data = read_json(candidate)
    Path(data["artifacts"]["qids_val_strict"]["path"]).write_text("tampered\n")
    with pytest.raises(ValueError, match="qids_val_strict"):
        verify_split_identity(reference, candidate)


def test_identity_check_rejects_same_manifest_as_candidate(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    reference = _manifest(tmp_path / "base")
    with pytest.raises(ValueError, match="candidate manifest aliases reference"):
        verify_split_identity(reference, reference)


def test_identity_check_rejects_candidate_pointing_to_historical_artifact(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    from src.utils.io import atomic_write_json

    reference = _manifest(tmp_path / "base")
    candidate = _manifest(tmp_path / "limited")
    data = read_json(candidate)
    data["artifacts"]["qids_val_strict"]["path"] = read_json(reference)["artifacts"][
        "qids_val_strict"]["path"]
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="qids_val_strict.*historical"):
        verify_split_identity(reference, candidate)


def test_identity_check_rejects_changed_scenario_corpus(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    from src.utils.io import atomic_write_json

    reference = _manifest(tmp_path / "base")
    candidate = _manifest(tmp_path / "limited")
    data = read_json(candidate)
    data["scenarios"]["splits"]["val_strict"]["scenarios"] += 1
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="scenario summary"):
        verify_split_identity(reference, candidate)


def test_identity_check_rejects_changed_val_metadata_even_if_files_match(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    reference = _manifest(tmp_path / "base")
    candidate = _manifest(tmp_path / "limited")
    from src.utils.io import atomic_write_json

    data = read_json(candidate)
    data["strict_split"]["splits"]["val_strict"]["rows_emitted"] -= 1
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="val_strict.rows_emitted"):
        verify_split_identity(reference, candidate)


def test_identity_check_rejects_changed_raw_input(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    candidate = _manifest(tmp_path / "limited")
    source = read_json(tmp_path / "limited" / "raw" / "source_manifest.json")
    raw = tmp_path / "limited" / "raw" / source["splits"]["train"]["parquet"]
    with raw.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="raw train artifact"):
        verify_split_identity(_manifest(tmp_path / "base"), candidate)


def test_identity_check_rejects_valid_but_different_raw_with_same_revision(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    from src.utils.hashing import sha256_file
    from src.utils.io import atomic_write_json

    candidate = _manifest(tmp_path / "limited")
    source_path = tmp_path / "limited" / "raw" / "source_manifest.json"
    source = read_json(source_path)
    raw = source_path.parent / source["splits"]["validation"]["parquet"]
    # Different intact bytes, with both levels of candidate manifest updated consistently.
    with raw.open("ab") as handle:
        handle.write(b"different source")
    source["splits"]["validation"]["sha256"] = sha256_file(raw)
    atomic_write_json(source_path, source)
    data = read_json(candidate)
    data["source"]["source_manifest_sha256"] = sha256_file(source_path)
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="raw validation differs"):
        verify_split_identity(_manifest(tmp_path / "base"), candidate)


def test_identity_check_rejects_different_raw_content_with_same_revision(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    from src.utils.hashing import sha256_file
    from src.utils.io import atomic_write_json

    candidate = _manifest(tmp_path / "limited")
    source_path = tmp_path / "limited" / "raw" / "source_manifest.json"
    source = read_json(source_path)
    source["splits"]["train"]["sha256"] = "0" * 64
    atomic_write_json(source_path, source)
    data = read_json(candidate)
    data["source"]["source_manifest_sha256"] = sha256_file(source_path)
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="raw train artifact"):
        verify_split_identity(_manifest(tmp_path / "base"), candidate)


def test_identity_check_rejects_source_revision_change(tmp_path):
    fx.run_fixture_pipeline(tmp_path / "base", profile="full", caps={"full": None})
    fx.run_fixture_pipeline(tmp_path / "limited", profile="medium",
                            caps={"medium": 20}, train_cap=6)
    reference = _manifest(tmp_path / "base")
    candidate = _manifest(tmp_path / "limited")
    from src.utils.io import atomic_write_json

    data = read_json(candidate)
    data["source"]["resolved_revision"] = "different"
    atomic_write_json(candidate, data)
    with pytest.raises(ValueError, match="resolved_revision"):
        verify_split_identity(reference, candidate)

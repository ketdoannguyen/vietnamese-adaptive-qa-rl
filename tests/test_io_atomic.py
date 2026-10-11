"""A failed parquet write must not replace a previously registered artifact."""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from src.utils.hashing import sha256_file
from src.utils.io import write_parquet


def test_failed_parquet_rewrite_preserves_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "artifact.parquet"
    old = write_parquet(path, [{"qid": "first"}])
    original = pq.write_table

    def fail_after_write(table, target, **kwargs):
        original(table, target, **kwargs)
        raise RuntimeError("interrupted write")

    monkeypatch.setattr(pq, "write_table", fail_after_write)
    with pytest.raises(RuntimeError, match="interrupted write"):
        write_parquet(path, [{"qid": "replacement"}])

    assert sha256_file(path) == old["sha256"]
    assert pq.read_table(path).to_pylist() == [{"qid": "first"}]
    assert list(tmp_path.iterdir()) == [path]

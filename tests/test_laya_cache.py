"""The Laya probability cache and the ``cache_laya`` stage (parquet, invalidation, warm-up).

Staleness is what this file is about: a row computed from checkpoint A must never be served
for checkpoint B, a ``state_hash`` from an old text layout must never be served at all, and a
run that would silently mix two priors must stop instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from src.decision.laya_cache import CACHE_VERSION, COLUMNS, LayaProbCache, ModelHashMismatch
from src.decision.state_text import STATE_TEXT_VERSION, state_hash
from src.pipeline import cache_laya, worlds
from src.utils.io import read_json, read_jsonl, read_yaml, write_yaml
from tests import p5_fixtures as fx

CACHE_DIR_NAME = "cache"
HASH_A = "a" * 64
HASH_B = "b" * 64

REQUESTS = [("train_strict:q000", state_hash(f"text {index}"), f"text {index}")
            for index in range(5)]


def _rows_on_disk(cache: LayaProbCache) -> list[dict]:
    return pq.read_table(str(cache.parquet_path)).to_pylist()


def test_misses_are_computed_once_and_then_hit(tmp_path):
    cache = LayaProbCache(tmp_path / CACHE_DIR_NAME)
    adapter = fx.StubAdapter(model_hash=HASH_A)

    first = cache.get_or_compute(adapter, REQUESTS)
    assert set(first) == {(qid, digest) for qid, digest, _ in REQUESTS}
    assert adapter.calls == 1 and len(adapter.texts) == len(REQUESTS)
    assert (cache.misses, cache.hits) == (len(REQUESTS), 0)

    second = cache.get_or_compute(adapter, REQUESTS)
    assert adapter.calls == 1, "a second call must be answered from the cache"
    assert (cache.misses, cache.hits) == (len(REQUESTS), len(REQUESTS))
    for key, probs in first.items():
        assert probs.shape == (4,) and probs.sum() == pytest.approx(1.0)
        assert np.allclose(second[key], probs)


def test_repeated_keys_are_computed_once_and_empty_requests_are_free(tmp_path):
    cache = LayaProbCache(tmp_path / CACHE_DIR_NAME)
    adapter = fx.StubAdapter(model_hash=HASH_A)
    assert cache.get_or_compute(adapter, []) == {} and adapter.calls == 0

    once = REQUESTS[:1]
    out = cache.get_or_compute(adapter, once + once)
    assert len(out) == 1 and adapter.calls == 1 and len(adapter.texts) == 1
    assert cache.misses == 1


def test_rows_survive_a_reopen_with_the_same_versions(tmp_path):
    directory = tmp_path / CACHE_DIR_NAME
    written = LayaProbCache(directory).get_or_compute(fx.StubAdapter(model_hash=HASH_A),
                                                      REQUESTS)
    fresh = LayaProbCache(directory)
    assert len(fresh) == len(REQUESTS) and fresh.model_hash == HASH_A
    for (qid, digest), probs in written.items():
        stored = fresh.lookup(qid, digest)
        assert stored is not None and np.allclose(stored, probs)
    meta = read_json(fresh.meta_path)
    assert meta["cache_version"] == CACHE_VERSION
    assert meta["state_text_version"] == STATE_TEXT_VERSION
    assert meta["model_hash"] == HASH_A and meta["n_rows"] == len(REQUESTS)
    assert [column for column in _rows_on_disk(fresh)[0]] == list(COLUMNS)


def test_an_unknown_state_is_a_miss(tmp_path):
    cache = LayaProbCache(tmp_path / CACHE_DIR_NAME)
    cache.get_or_compute(fx.StubAdapter(model_hash=HASH_A), REQUESTS)
    assert cache.lookup("train_strict:q000", "0" * 64) is None
    assert cache.lookup("train_strict:q999", REQUESTS[0][1]) is None


def test_require_model_hash_refuses_a_foreign_checkpoint_without_touching_rows(tmp_path):
    directory = tmp_path / CACHE_DIR_NAME
    cache = LayaProbCache(directory)
    cache.get_or_compute(fx.StubAdapter(model_hash=HASH_A), REQUESTS)
    with pytest.raises(ModelHashMismatch, match="rebuild the cache"):
        cache.require_model_hash(HASH_B)
    assert len(cache) == len(REQUESTS)
    assert cache.lookup(*REQUESTS[0][:2]) is not None
    assert read_json(cache.meta_path)["model_hash"] == HASH_A


def test_get_or_compute_under_a_new_checkpoint_invalidates_everything(tmp_path):
    directory = tmp_path / CACHE_DIR_NAME
    cache = LayaProbCache(directory)
    original = cache.get_or_compute(fx.StubAdapter(model_hash=HASH_A), REQUESTS)

    second = fx.StubAdapter(model_hash=HASH_B)
    replaced = cache.get_or_compute(second, REQUESTS)
    assert cache.invalidated_rows == len(REQUESTS)
    assert second.calls == 1 and len(second.texts) == len(REQUESTS)
    assert cache.model_hash == HASH_B and len(cache) == len(REQUESTS)
    assert all(row["model_hash"] == HASH_B for row in _rows_on_disk(cache))
    # The stub's rows depend on the state text only, so the probabilities agree; what changed
    # is the stamp. The point is that the old rows were *replaced*, never served.
    assert read_json(cache.meta_path)["model_hash"] == HASH_B
    for key, probs in original.items():
        assert np.allclose(replaced[key], probs)

    reopened = LayaProbCache(directory)
    assert reopened.model_hash == HASH_B and len(reopened) == len(REQUESTS)


def test_a_state_text_version_change_discards_the_rows(tmp_path):
    directory = tmp_path / CACHE_DIR_NAME
    LayaProbCache(directory).get_or_compute(fx.StubAdapter(model_hash=HASH_A), REQUESTS)
    other = LayaProbCache(directory, state_text_version=STATE_TEXT_VERSION + "-next")
    assert len(other) == 0 and other.invalidated_rows == len(REQUESTS)
    assert other.lookup(*REQUESTS[0][:2]) is None


def test_an_adapter_returning_the_wrong_shape_is_an_error(tmp_path):
    class _BadShape(fx.StubAdapter):
        def predict_probs(self, states):
            self.calls += 1
            return np.zeros((len(states), 3), dtype=np.float64)

    cache = LayaProbCache(tmp_path / CACHE_DIR_NAME)
    with pytest.raises(ValueError, match="adapter returned"):
        cache.get_or_compute(_BadShape(model_hash=HASH_A), REQUESTS)
    assert len(cache) == 0 and not cache.parquet_path.exists()


# --------------------------------------------------------------------- the cache_laya stage

def _pipeline_fixture(tmp_path, *, model_hash=HASH_A, include_checkpoint=True):
    cfg = fx.p5_config(tmp_path)
    fx.write_all(cfg, n_train=2, n_val=1)
    if include_checkpoint:
        worlds.laya_checkpoint_dir(cfg, "ce").mkdir(parents=True, exist_ok=True)
    return cfg


def _run(cfg, *, adapter=None, force=False, splits=("train_strict",)):
    return cache_laya.run(cfg=cfg, profile="smoke", regime="ce", splits=splits, limit=None,
                          batch=2, force=force, adapter=adapter or fx.StubAdapter(model_hash=HASH_A))


class _StubLoader:
    """Stands in for :class:`LayaAdapter` so the CLI path needs no real checkpoint."""

    @staticmethod
    def load(checkpoint_dir, *, device=None, batch_size=8):
        return fx.StubAdapter(model_hash=HASH_A)


def test_run_caches_every_reachable_state_and_registers_the_artifacts(tmp_path):
    cfg = _pipeline_fixture(tmp_path)
    adapter = fx.StubAdapter(model_hash=HASH_A)
    summary = _run(cfg, adapter=adapter)

    assert summary["regime"] == "ce" and summary["model_hash"] == HASH_A
    per_split = summary["splits"]["train_strict"]
    assert per_split["questions"] == 2 and per_split["states"] >= per_split["tree_states"] >= 2
    assert summary["states"] == per_split["states"]
    assert summary["misses"] == summary["states"] and summary["hits"] == 0
    assert adapter.calls >= 1 and not adapter.closed, "the run must not close an injected adapter"

    written = LayaProbCache(worlds.laya_cache_dir(cfg))
    assert len(written) == summary["states"] and written.model_hash == HASH_A
    summary_path = Path(cfg["paths"]["manifests"]) / "P5_cache_laya_ce_smoke.json"
    assert summary_path.is_file() and read_json(summary_path)["states"] == summary["states"]

    rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl"))
    assert [row["status"] for row in rows] == ["RUNNING", "SUCCEEDED"]
    assert rows[1]["stage"] == "P5_cache_laya" and set(rows[1]["artifacts"]) == {
        str(written.parquet_path), str(written.meta_path), str(summary_path)}


def test_a_second_run_is_all_hits(tmp_path):
    cfg = _pipeline_fixture(tmp_path)
    first = _run(cfg)
    second = _run(cfg)
    assert second["states"] == first["states"]
    assert second["misses"] == 0 and second["hits"] == first["states"]


def test_a_foreign_checkpoint_stops_the_run_and_keeps_the_rows(tmp_path):
    cfg = _pipeline_fixture(tmp_path)
    first = _run(cfg)
    with pytest.raises(ModelHashMismatch):
        _run(cfg, adapter=fx.StubAdapter(model_hash=HASH_B))
    cached = LayaProbCache(worlds.laya_cache_dir(cfg))
    assert len(cached) == first["states"] and cached.model_hash == HASH_A
    rows = list(read_jsonl(Path(cfg["paths"]["runs"]) / "index.jsonl"))
    assert [row["status"] for row in rows] == ["RUNNING", "SUCCEEDED", "RUNNING", "FAILED"]
    assert "ModelHashMismatch" in rows[-1]["error"]


def test_force_replaces_a_foreign_checkpoints_rows(tmp_path):
    cfg = _pipeline_fixture(tmp_path)
    first = _run(cfg)
    summary = _run(cfg, adapter=fx.StubAdapter(model_hash=HASH_B), force=True)
    assert summary["invalidated_rows"] == first["states"]
    assert summary["misses"] == summary["states"]
    cached = LayaProbCache(worlds.laya_cache_dir(cfg))
    assert cached.model_hash == HASH_B and len(cached) == summary["states"]


def test_a_missing_checkpoint_is_an_error_before_anything_runs(tmp_path):
    cfg = _pipeline_fixture(tmp_path, include_checkpoint=False)
    with pytest.raises(FileNotFoundError, match="train it with the P5 stage"):
        _run(cfg)
    assert not (Path(cfg["paths"]["runs"]) / "index.jsonl").exists()


def test_main_names_the_snapshot_after_the_resolved_regime(tmp_path, capsys, monkeypatch):
    cfg = _pipeline_fixture(tmp_path)
    monkeypatch.setattr(cache_laya, "LayaAdapter", _StubLoader)
    # The overlay carries the fixture's absolute paths, so the CLI never writes into the repo.
    overlay = tmp_path / "overlay.yaml"
    write_yaml(overlay, cfg)
    assert read_yaml(overlay)["paths"]["manifests"] == cfg["paths"]["manifests"]

    code = cache_laya.main(["--config", str(overlay), "--profile", "smoke", "--regime", "ce",
                            "--splits", "train_strict", "--limit", "1", "--json"])
    assert code == 0
    names = [path.name for path in Path(cfg["paths"]["manifests"]).iterdir()]
    assert "P5_cache_laya_ce_smoke_config.yaml" in names
    assert not [name for name in names if worlds.DEFAULT_LAYA_SENTINEL in name]
    out = capsys.readouterr().out
    assert '"regime": "ce"' in out and '"states"' in out

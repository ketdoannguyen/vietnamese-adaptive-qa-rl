"""Pinned Vietnamese_Embedding dense encoder: config verification, no remote code, no
prefixes, OOM-only batch reduction.

Everything here is synthetic: a fake SentenceTransformer stands in for the 2.3 GB model,
so no weights are downloaded or loaded and no GPU is touched. The contracts under test are
the ones the VixSTORY V2 plan pins: revision resolution, CLS/1024 config verification,
``trust_remote_code=False``, prefix-free normalized encoding, halving on CUDA OOM and a
hard stop (never an E5 fallback) at batch size 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types

import numpy as np
import pytest

from src.retrieval import dense_v2 as mod

PIN = "dea33aa1ab339f38d66ae0a40e6c40e0a9249568"


def write_snapshot(root, *, modules=None, pooling=None, dim=1024, max_seq_length=8192):
    """A minimal on-disk snapshot with the pinned model's file layout."""
    root.mkdir(parents=True, exist_ok=True)
    modules = modules if modules is not None else [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "", "type": "sentence_transformers.models.Pooling"},
        {"idx": 2, "name": "2", "path": "", "type": "sentence_transformers.models.Normalize"},
    ]
    (root / "modules.json").write_text(json.dumps(modules), encoding="utf-8")
    (root / "1_Pooling").mkdir(exist_ok=True)
    cfg = {"word_embedding_dimension": dim, "pooling_mode_cls_token": True,
           "pooling_mode_mean_tokens": False, "pooling_mode_max_tokens": False,
           "pooling_mode_mean_sqrt_len_tokens": False,
           "pooling_mode_weightedmean_tokens": False, "pooling_mode_lasttoken": False}
    if pooling:
        cfg.update(pooling)
    (root / "1_Pooling" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (root / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": max_seq_length, "do_lower_case": False}), encoding="utf-8")
    for name in mod.MODEL_FILES:                      # placeholder payloads for required files
        path = root / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"pinned-model-file")
    return root


class FakeST:
    """Deterministic stand-in for SentenceTransformer; records every encode call."""

    def __init__(self, *, dim=1024, oom_above=None, scale=1.0, nan=False):
        self.dim = dim
        self.oom_above = oom_above
        self.scale = scale
        self.nan = nan
        self.max_seq_length = None
        self.calls = []

    def get_embedding_dimension(self):
        return self.dim

    def encode(self, texts, *, batch_size, normalize_embeddings, convert_to_numpy, show_progress_bar):
        self.calls.append({"texts": list(texts), "batch_size": batch_size,
                           "normalize_embeddings": normalize_embeddings})
        if self.oom_above is not None and batch_size > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
            vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec) * self.scale
            if self.nan:
                vec[0] = np.nan
            rows.append(vec)
        return np.vstack(rows)


def fake_encoder(tmp_path, fake, monkeypatch, **kwargs):
    snapshot = write_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(mod, "_build_sentence_transformer",
                        lambda model_path, device=None: fake)
    return mod.VietnameseEmbeddingEncoder(snapshot, **kwargs), snapshot


# --- download set and revision resolution ------------------------------------------------

def test_model_files_exclude_onnx_and_remote_code():
    assert "model.safetensors" in mod.MODEL_FILES
    assert "1_Pooling/config.json" in mod.MODEL_FILES
    assert not any("onnx" in name for name in mod.MODEL_FILES)
    assert "evaluation_model.py" not in mod.MODEL_FILES


def test_download_model_verifies_the_resolved_revision(tmp_path, monkeypatch):
    def fake_download(repo_id, revision=None, allow_patterns=None, cache_dir=None):
        snapshot = tmp_path / "models" / PIN
        return str(write_snapshot(snapshot))

    monkeypatch.setattr(mod, "snapshot_download", fake_download)
    path, files = mod.download_model()
    assert path == tmp_path / "models" / PIN
    assert set(files) == set(mod.MODEL_FILES)
    assert files["config.json"]["bytes"] > 0
    assert len(files["config.json"]["sha256"]) == 64


def test_download_model_rejects_an_unresolved_revision(tmp_path, monkeypatch):
    def fake_download(repo_id, revision=None, allow_patterns=None, cache_dir=None):
        return str(write_snapshot(tmp_path / "models" / "some-other-commit"))

    monkeypatch.setattr(mod, "snapshot_download", fake_download)
    with pytest.raises(ValueError, match="resolved"):
        mod.download_model()


# --- pinned config verification ----------------------------------------------------------

def test_verify_model_files_reports_the_pinned_configuration(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot")
    report = mod.verify_model_files(snapshot)
    assert report["pooling"] == "cls"
    assert report["embedding_dimension"] == 1024
    assert report["config_max_seq_length"] == 8192
    assert report["modules"] == ["Transformer", "Pooling", "Normalize"]


def test_verify_rejects_a_remote_code_module(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot", modules=[
        {"idx": 0, "name": "0", "path": "evaluation_model.py",
         "type": "evaluation_model.Model"}])
    with pytest.raises(ValueError, match="remote|code|refus"):
        mod.verify_model_files(snapshot)


def test_verify_rejects_a_non_cls_pooling_config(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot",
                              pooling={"pooling_mode_cls_token": False,
                                       "pooling_mode_mean_tokens": True})
    with pytest.raises(ValueError, match="[Cc][Ll][Ss]"):
        mod.verify_model_files(snapshot)


def test_verify_rejects_a_dimension_mismatch(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot", dim=768)
    with pytest.raises(ValueError, match="1024"):
        mod.verify_model_files(snapshot)


def test_verify_rejects_a_missing_normalize_module(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot", modules=[
        {"idx": 0, "name": "0", "path": "",
         "type": "sentence_transformers.models.Transformer"}])
    with pytest.raises(ValueError, match="Normalize"):
        mod.verify_model_files(snapshot)


def test_verify_rejects_a_missing_required_file(tmp_path):
    snapshot = write_snapshot(tmp_path / "snapshot")
    (snapshot / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="tokenizer.json"):
        mod.verify_model_files(snapshot)


# --- no arbitrary remote code ------------------------------------------------------------

def test_build_sentence_transformer_disables_remote_code(monkeypatch):
    captured = {}

    class Recorder:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = model_path
            captured.update(kwargs)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = Recorder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    mod._build_sentence_transformer("/tmp/pinned-model", device="cuda")
    assert captured["model_path"] == "/tmp/pinned-model"
    assert captured["trust_remote_code"] is False
    assert captured["device"] == "cuda"


# --- encoder contract --------------------------------------------------------------------

def test_encoder_load_pins_max_length_and_verifies_dimension(tmp_path, monkeypatch):
    fake = FakeST()
    encoder, _ = fake_encoder(tmp_path, fake, monkeypatch, batch_size=4)
    assert encoder.load() is fake
    assert fake.max_seq_length == mod.MAX_SEQ_LENGTH == 2048
    assert encoder.dimension() == mod.EMBED_DIM == 1024
    assert encoder.config_report["pooling"] == "cls"


def test_encoder_rejects_a_non_1024_model(tmp_path, monkeypatch):
    encoder, _ = fake_encoder(tmp_path, FakeST(dim=768), monkeypatch)
    with pytest.raises(ValueError, match="1024"):
        encoder.load()


def test_encoder_requires_a_sentence_transformers_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    encoder = mod.VietnameseEmbeddingEncoder(plain)
    with pytest.raises(ValueError, match="modules.json"):
        encoder.load()


def test_encode_is_prefix_free_normalized_and_deterministic(tmp_path, monkeypatch):
    fake = FakeST()
    encoder, _ = fake_encoder(tmp_path, fake, monkeypatch, batch_size=4)
    passages = ["trường đại học bách khoa hà nội", "đoạn văn số hai"]
    first = encoder.encode_passages(passages)
    second = encoder.encode_passages(passages)
    assert fake.calls[0]["texts"] == passages            # no query:/passage: prefixes
    assert fake.calls[0]["normalize_embeddings"] is True
    assert first.shape == (2, 1024)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-3)
    assert np.array_equal(first, second)

    queries = ["câu hỏi một?"]
    encoder.encode_queries(queries)
    assert fake.calls[-1]["texts"] == queries


def test_encode_rejects_empty_batches(tmp_path, monkeypatch):
    encoder, _ = fake_encoder(tmp_path, FakeST(), monkeypatch)
    with pytest.raises(ValueError, match="empty"):
        encoder.encode_passages([])


def test_encode_rejects_non_normalized_output(tmp_path, monkeypatch):
    encoder, _ = fake_encoder(tmp_path, FakeST(scale=2.0), monkeypatch)
    with pytest.raises(ValueError, match="normaliz"):
        encoder.encode_passages(["một đoạn văn"])


def test_encode_rejects_non_finite_output(tmp_path, monkeypatch):
    encoder, _ = fake_encoder(tmp_path, FakeST(nan=True), monkeypatch)
    with pytest.raises(ValueError, match="finite"):
        encoder.encode_passages(["một đoạn văn"])


# --- OOM policy: halve, then stop --------------------------------------------------------

def test_oom_halves_the_batch_and_records_the_effective_size(tmp_path, monkeypatch):
    fake = FakeST(oom_above=8)
    encoder, _ = fake_encoder(tmp_path, fake, monkeypatch, batch_size=32)
    vectors = encoder.encode_passages(["văn bản một", "văn bản hai"])
    assert [call["batch_size"] for call in fake.calls] == [32, 16, 8]
    assert encoder.effective_batch_size == 8
    assert vectors.shape == (2, 1024)


def test_oom_at_batch_size_one_stops_without_any_fallback(tmp_path, monkeypatch):
    fake = FakeST(oom_above=0)
    encoder, _ = fake_encoder(tmp_path, fake, monkeypatch, batch_size=8)
    with pytest.raises(RuntimeError, match="batch_size=1"):
        encoder.encode_passages(["văn bản"])
    assert [call["batch_size"] for call in fake.calls] == [8, 4, 2, 1]


def test_unload_releases_the_model(tmp_path, monkeypatch):
    encoder, _ = fake_encoder(tmp_path, FakeST(), monkeypatch)
    encoder.load()
    encoder.unload()
    assert encoder._model is None

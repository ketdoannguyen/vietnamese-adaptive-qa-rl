"""Shared retrieval primitives: tokenizer anchors, BM25 ranking, dense bank.

Retained from the P2 acceptance suite after the ViQuAD pipeline removal. The scenario
ranking/cursor/cache tests went with ``src/retrieval/{engine,rankings}.py`` and their
in-repo ranking stub; what remains runs offline on hand-built inputs — the dense encoder
is exercised through a stand-in model, never the real E5 weights.
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Index, rank_documents
from src.retrieval.dense import PASSAGE_PREFIX, QUERY_PREFIX, DenseBank, DenseEncoder, rank_by_inner_product
from src.retrieval.tokenize import self_test as tokenizer_self_test
from src.retrieval.tokenize import tokenize_vi, tokenizer_name

# --------------------------------------------------------------------------- tokenizer


def test_tokenizer_self_test_anchors_pass():
    results = tokenizer_self_test()
    assert len(results) >= 2
    assert all(results.values()), results


def test_tokenizer_preserves_diacritics():
    assert tokenize_vi("hoà bình") != tokenize_vi("hoa bình")
    assert tokenize_vi("Hà Nội") == ["hà", "nội"]


def test_tokenizer_normalises_decomposed_input_to_nfc():
    decomposed = unicodedata.normalize("NFD", "Thành phố Hồ Chí Minh")
    assert decomposed != "Thành phố Hồ Chí Minh"
    assert tokenize_vi(decomposed) == tokenize_vi("Thành phố Hồ Chí Minh")


def test_tokenizer_stopword_mode_is_explicit_and_named():
    assert tokenizer_name() == "regex_fallback"
    assert "và" not in tokenize_vi("và Hà Nội", remove_stopwords=True)
    assert "và" in tokenize_vi("và Hà Nội", remove_stopwords=False)


# ------------------------------------------------------------------------------- BM25


def test_bm25_ranks_the_lexical_match_first():
    docs = {"d1": "Hà Nội là thủ đô của Việt Nam.",
            "d2": "Phở là món ăn truyền thống.",
            "d3": "Sông Mekong chảy qua nhiều quốc gia."}
    pairs = rank_documents("thủ đô Việt Nam", docs)
    assert pairs[0][0] == "d1"


def test_bm25_ties_break_on_doc_id_for_stability():
    docs = {"doc_z": "Hà Nội là thủ đô.", "doc_a": "Hà Nội là thủ đô."}
    first, second = BM25Index.build(docs).rank("Hà Nội")[:2]
    assert (first[0], second[0]) == ("doc_a", "doc_z")
    assert first[1] == second[1]


def test_bm25_handles_empty_documents_and_empty_queries_without_nan():
    docs = {"empty": "", "filled": "Hà Nội là thủ đô của Việt Nam."}
    index = BM25Index.build(docs)
    for pairs in (index.rank("Hà Nội"), index.rank(""), index.rank("và của là")):
        assert len(pairs) == 2
        assert all(math.isfinite(score) for _, score in pairs)


def test_bm25_requires_at_least_one_document():
    with pytest.raises(ValueError):
        BM25Index.build({})


def test_bm25_top_n_truncates():
    docs = {f"d{i}": f"Hà Nội số {i}" for i in range(5)}
    assert len(BM25Index.build(docs).rank("Hà Nội", top_n=2)) == 2


# ----------------------------------------------------------- dense encoder (no weights)


class _FakeModel:
    """Stands in for SentenceTransformer so the wrapper can be tested without weights."""

    def __init__(self, output) -> None:
        self.output = output
        self.seen: list[dict] = []

    def encode(self, texts, **kwargs):
        self.seen.append({"texts": list(texts), "kwargs": kwargs})
        return self.output


def _encoder_with(output, **kwargs) -> tuple[DenseEncoder, _FakeModel]:
    encoder = DenseEncoder(**kwargs)
    model = _FakeModel(output)
    encoder._model = model  # bypass load(): no download, no torch
    return encoder, model


def test_e5_prefixes_are_applied_on_the_correct_side():
    encoder, model = _encoder_with(np.ones((2, 4), dtype=np.float32))
    encoder.encode_passages(["Hà Nội", "Sông Mekong"])
    encoder.encode_queries(["Thủ đô là gì?"])
    assert model.seen[0]["texts"] == [f"{PASSAGE_PREFIX}Hà Nội", f"{PASSAGE_PREFIX}Sông Mekong"]
    assert model.seen[1]["texts"] == [f"{QUERY_PREFIX}Thủ đô là gì?"]
    assert all(call["kwargs"].get("normalize_embeddings") is True for call in model.seen)
    assert PASSAGE_PREFIX != QUERY_PREFIX


def test_encoder_rejects_non_finite_vectors():
    encoder, _ = _encoder_with(np.full((1, 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        encoder.encode_queries(["Thủ đô là gì?"])


def test_encoder_rejects_non_2d_output():
    encoder, _ = _encoder_with(np.ones(4, dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        encoder.encode_queries(["Thủ đô là gì?"])


# -------------------------------------------------------------------------- DenseBank


def _synthetic_bank() -> DenseBank:
    """Two L2-normalised 2-D vectors; no encoder, no fixture, no I/O."""
    return DenseBank(["doc_hanoi", "doc_pho"],
                     np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_inner_product_ranking_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        rank_by_inner_product(np.ones((1, 8), dtype=np.float32),
                              np.ones((2, 4), dtype=np.float32), ["a", "b"])


def test_dense_bank_rejects_id_vector_count_mismatch():
    with pytest.raises(ValueError, match="2 ids but 1 vectors"):
        DenseBank(["a", "b"], np.ones((1, 4), dtype=np.float32))


def test_dense_bank_subset_rejects_unknown_document():
    bank = _synthetic_bank()
    with pytest.raises(KeyError):
        bank.subset(["doc_hanoi", "doc_not_in_bank"])


def test_dense_bank_save_load_round_trip_without_pickle(tmp_path):
    bank = _synthetic_bank()
    path = tmp_path / "bank.npz"
    bank.save(str(path), meta={"model": "synthetic", "revision": "v1"})
    loaded, meta = DenseBank.load(str(path))
    assert loaded.doc_ids == bank.doc_ids
    assert all(isinstance(d, str) for d in loaded.doc_ids)
    assert np.allclose(loaded.vectors, bank.vectors)
    assert meta == {"model": "synthetic", "revision": "v1"}

"""P2 acceptance tests (`02_IMPLEMENTATION/02_BM25_DENSE_RETRIEVAL.md` §8).

Gates covered, in the spec's order:
1. rankings are invariant for the same scenario (in-process and in a fresh process,
   under different PYTHONHASHSEED values);
2. a second search never returns an already-seen document;
3. every returned doc_id belongs to the scenario corpus (no cross-split leakage);
4. embeddings have the right dimension, are finite, and are L2-normalised;
5. E5 `query: `/`passage: ` prefixes are applied on the correct side;
6. recall@k / overlap / difference machinery is exact (the *real-data* recall table is
   produced by the pipeline stage, and by the env-guarded test here);
7. the cache-only path runs with no torch/sentence-transformers import at all.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from src.retrieval.bm25 import BM25Index, rank_documents
from src.retrieval.dense import (PASSAGE_PREFIX, QUERY_PREFIX, DenseBank, DenseEncoder,
                                 rank_by_inner_product)
from src.retrieval.engine import RankingCache, ScenarioRanking, next_unseen, rank_scenario
from src.retrieval.rankings import (differing_examples, novelty_rate, recall_at_k,
                                    reciprocal_rank, summarise_retrieval, top_k_overlap)
from src.retrieval.tokenize import self_test as tokenizer_self_test
from src.retrieval.tokenize import tokenize_vi, tokenizer_name
from tests import retrieval_stub
from tests.retrieval_stub import CORPUS_IDS, DOCS, QUESTION, StubEncoder, build_ranking

ROOT = Path(__file__).resolve().parents[1]
RUN_REAL_MODEL = os.environ.get("RETRIEVAL_REAL_MODEL") == "1"


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


# ----------------------------------------------------------------- rank_scenario (P2 §8.3)


def test_all_returned_ids_belong_to_the_scenario_corpus():
    ranking = build_ranking()
    for tool in ("bm25", "dense"):
        returned = ranking.ranked(tool)
        assert returned, f"{tool} returned nothing"
        assert set(returned) <= set(CORPUS_IDS)
        assert "doc_outside" not in returned
    # The excluded document is a perfect content match, so its absence proves the
    # restriction, not luck.
    assert "thủ đô" in DOCS["doc_outside"]


def test_scores_match_rankings_and_dense_scores_descend():
    ranking = build_ranking()
    for tool in ("bm25", "dense"):
        assert set(ranking.scores[tool]) == set(ranking.ranked(tool))
    dense_scores = [ranking.score("dense", d) for d in ranking.ranked("dense")]
    assert dense_scores == sorted(dense_scores, reverse=True)


def test_missing_corpus_document_raises():
    with pytest.raises(KeyError, match="missing from docs"):
        build_ranking(corpus_ids=CORPUS_IDS + ["doc_absent"])


def test_duplicate_corpus_ids_raise():
    with pytest.raises(ValueError, match="duplicate"):
        build_ranking(corpus_ids=CORPUS_IDS + ["doc_pho"])


def test_unknown_tool_lookup_raises():
    ranking = build_ranking()
    with pytest.raises(KeyError, match="no ranking for tool"):
        ranking.ranked("sparse")


def test_rank_scenario_uses_query_side_encoding_only():
    encoder = StubEncoder()
    build_ranking(encoder=encoder)
    assert encoder.calls == ["passage", "query"]  # bank first, then the question


# ------------------------------------------------------------------------- cursor (P2 §8.2)


def test_second_search_returns_a_document_never_seen_before():
    ranking = build_ranking()
    ranked = ranking.ranked("bm25")
    assert len(ranked) >= 2
    seen: list[str] = []
    for _ in range(len(ranked)):
        fresh = next_unseen(ranked, seen, 1)
        assert len(fresh) == 1
        assert fresh[0] not in seen
        seen.extend(fresh)
    assert set(seen) == set(ranked)
    assert next_unseen(ranked, seen, 1) == []


def test_cursor_can_fetch_several_passages_at_once_without_repeats():
    ranked = build_ranking().ranked("dense")
    first_two = next_unseen(ranked, [], 2)
    assert len(first_two) == len(set(first_two)) == 2
    next_two = next_unseen(ranked, first_two, 2)
    assert set(next_two).isdisjoint(first_two)


def test_novelty_rate_counts_unseen_share():
    assert novelty_rate(["a", "b", "c", "d"], ["a"], 3) == pytest.approx(2 / 3)
    assert novelty_rate(["a", "b"], ["a", "b"], 2) == 0.0
    assert novelty_rate([], [], 3) == 0.0


# ----------------------------------------------------- determinism (P2 §8.1, cross-process)


def test_ranking_is_byte_identical_across_independent_builds():
    assert retrieval_stub.rows_hash() == retrieval_stub.rows_hash()


def test_ranking_is_identical_in_a_fresh_process_under_any_hash_seed():
    reference = retrieval_stub.rows_hash()
    for hash_seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, "-m", "tests.retrieval_stub"],
                              cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == reference


# ------------------------------------------------------------ dense encoder (P2 §8.4, §8.5)


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


def test_stub_bank_vectors_are_unit_norm():
    bank = retrieval_stub.build_bank()
    norms = np.linalg.norm(bank.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_inner_product_ranking_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension mismatch"):
        rank_by_inner_product(np.ones((1, 8), dtype=np.float32),
                              np.ones((2, 4), dtype=np.float32), ["a", "b"])


def test_dense_bank_rejects_id_vector_count_mismatch():
    with pytest.raises(ValueError, match="2 ids but 1 vectors"):
        DenseBank(["a", "b"], np.ones((1, 4), dtype=np.float32))


def test_dense_bank_subset_rejects_unknown_document():
    bank = retrieval_stub.build_bank()
    with pytest.raises(KeyError):
        bank.subset(["doc_hanoi", "doc_not_in_bank"])


def test_dense_bank_save_load_round_trip_without_pickle(tmp_path):
    bank = retrieval_stub.build_bank()
    path = tmp_path / "bank.npz"
    bank.save(str(path), meta={"model": "stub", "revision": "v1"})
    loaded, meta = DenseBank.load(str(path))
    assert loaded.doc_ids == bank.doc_ids
    assert all(isinstance(d, str) for d in loaded.doc_ids)
    assert np.allclose(loaded.vectors, bank.vectors)
    assert meta == {"model": "stub", "revision": "v1"}


@pytest.mark.skipif(not RUN_REAL_MODEL,
                    reason="set RETRIEVAL_REAL_MODEL=1 to exercise the real E5 encoder")
def test_real_e5_embeddings_are_dimensioned_finite_and_normalised():
    encoder = DenseEncoder(device="cpu")
    try:
        passages = encoder.encode_passages(list(DOCS.values()))
        assert passages.shape == (len(DOCS), 384)
        assert np.all(np.isfinite(passages))
        assert np.allclose(np.linalg.norm(passages, axis=1), 1.0, atol=1e-3)

        query = encoder.encode_queries([QUESTION])
        same_text_other_prefix = encoder.encode_passages([QUESTION])
        assert query.shape == (1, 384)
        assert not np.allclose(query, same_text_other_prefix, atol=1e-3)

        ranking = build_ranking(encoder=encoder)
        assert set(ranking.ranked("dense")) <= set(CORPUS_IDS)
        assert "doc_hanoi" in ranking.ranked("dense")[:2]
    finally:
        encoder.unload()


# -------------------------------------------------------------- cache-only mode (P2 §8.7)


def test_ranking_cache_round_trip_and_loud_miss():
    ranking = build_ranking()
    cache = RankingCache.from_rows([ranking.to_row()])
    assert len(cache) == 1
    assert ranking.scenario_id in cache
    restored = cache.get(ranking.scenario_id)
    assert restored.to_row() == ranking.to_row()
    with pytest.raises(KeyError, match="fail loudly"):
        cache.get("scenario-that-does-not-exist")


def test_cache_only_path_imports_no_torch_stack():
    script = (
        "import sys\n"
        "from src.retrieval.engine import RankingCache, next_unseen\n"
        "row = {'scenario_id': 's1', 'qid': 'q1',\n"
        "       'rankings': {'bm25': ['d1', 'd2'], 'dense': ['d2', 'd1']},\n"
        "       'scores': {'bm25': {'d1': 1.0, 'd2': 0.5}, 'dense': {'d1': 0.4, 'd2': 0.9}}}\n"
        "cache = RankingCache.from_rows([row])\n"
        "print(next_unseen(cache.get('s1').ranked('bm25'), [], 1)[0])\n"
        "heavy = sorted(m for m in ('torch', 'sentence_transformers', 'transformers')\n"
        "               if m in sys.modules)\n"
        "print('HEAVY:' + ','.join(heavy))\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.strip().splitlines()
    assert lines[0] == "d1"
    assert lines[1] == "HEAVY:"


# --------------------------------------------------- recall/overlap metrics (P2 §8.6)


def _synthetic_rankings() -> list[ScenarioRanking]:
    return [
        ScenarioRanking("s1", "q1", {"bm25": ["a", "b", "c"], "dense": ["b", "a", "c"]}, {}),
        ScenarioRanking("s2", "q2", {"bm25": ["x", "y", "z"], "dense": ["y", "z", "w"]}, {}),
        ScenarioRanking("s3", "q3", {"bm25": ["a", "b"], "dense": ["a", "b"]}, {}),
    ]


def test_recall_and_mrr_use_the_gold_document_position():
    rankings = _synthetic_rankings()
    assert recall_at_k(rankings[0].ranked("bm25"), "a", 1) == 1.0
    assert recall_at_k(rankings[0].ranked("dense"), "a", 1) == 0.0
    assert recall_at_k(rankings[0].ranked("dense"), "a", 2) == 1.0
    assert reciprocal_rank(rankings[0].ranked("dense"), "a") == pytest.approx(0.5)
    assert reciprocal_rank(rankings[0].ranked("bm25"), "zz") == 0.0
    # Negatives have no gold document and must not silently count as failure.
    assert math.isnan(recall_at_k(rankings[1].ranked("bm25"), None, 1))
    assert math.isnan(reciprocal_rank(rankings[1].ranked("bm25"), None))


def test_top_k_overlap_is_jaccard_on_the_top_k():
    assert top_k_overlap(["a", "b", "c"], ["b", "a", "c"], 3) == 1.0
    assert top_k_overlap(["a", "b", "c"], ["b", "z", "w"], 3) == pytest.approx(0.2)  # 1 / 5
    assert top_k_overlap([], [], 3) == 1.0


def test_summarise_retrieval_aggregates_over_negatives_and_positives():
    rankings = _synthetic_rankings()
    gold = {"s1": "a", "s2": None, "s3": "a"}
    report = summarise_retrieval(rankings, gold,
                                 reach_budget={"bm25": 3, "dense": 3}, overlap_k=3).to_dict()

    assert report["n_scenarios"] == 3
    assert report["n_answerable"] == 2  # s2 is a negative
    assert report["recall_at_1"] == {"bm25": 1.0, "dense": 0.5}  # s1 dense misses at k=1
    assert report["recall_at_3"] == {"bm25": 1.0, "dense": 1.0}
    assert report["mrr"]["bm25"] == 1.0
    assert report["mrr"]["dense"] == pytest.approx(0.75)  # (0.5 + 1.0) / 2
    assert report["union_recall_at_5"] == 1.0
    assert report["reachable_rate"] == {"bm25": 1.0, "dense": 1.0}
    # overlap: s1 = 1.0, s2 = 2/4, s3 = 1.0
    assert report["topk_overlap_bm25_dense"] == pytest.approx((1.0 + 0.5 + 1.0) / 3)
    assert report["diff_rate"] == pytest.approx(1 / 3)


def test_differing_examples_list_the_disagreements_only():
    examples = differing_examples(_synthetic_rankings(), k=3, limit=20)
    assert [e["scenario_id"] for e in examples] == ["s2"]
    assert examples[0]["top_bm25"] == ["x", "y", "z"]
    assert examples[0]["top_dense"] == ["y", "z", "w"]
    assert len(differing_examples(_synthetic_rankings(), k=3, limit=1)) == 1

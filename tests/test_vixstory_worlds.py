"""VixSTORY pilot world-builder contracts (CPU/synthetic, no model load, no GPU).

Coverage matches the task's three required cases — a synthetic split assembles real worlds,
a cache miss fails instead of generating, and a multi-gold private side is preserved — plus the
surrounding fail-don't-drop contract: overlong prompts are an explicit blocker, a cache written
under a different reader identity is refused, and the official test split is never read.
"""

from __future__ import annotations

import json

import pytest

from scripts.vixstory_reader_eval import PILOT_MAX_PROMPT_CHARS, CacheIdentity, GenerationCache
from src.data.schemas import find_forbidden_keys
from src.pipeline.vixstory_worlds import (
    MANIFEST_SPLITS,
    assemble_split,
    build_reader_cache,
    manifest_split_qids,
    prompt_budget_blockers,
    read_qa_rows,
    read_rankings,
    reader_cache_fingerprints,
    reader_state_prompts,
)
from src.retrieval.states import reachable_states
from src.rl.vixstory_env import (
    ANSWER,
    SEARCH_BM25,
    SEARCH_DENSE,
    VixStoryEnv,
    VixStoryEnvConfig,
)
from src.utils.hashing import sha256_json
from src.utils.io import write_json

# --- synthetic pinned inputs ----------------------------------------------------------
DOC_0 = "VixSTORY: trường A toạ lạc tại Hà Nội."
DOC_1 = "Trường A thành lập năm 1995 tại Hà Nội."
DOC_2 = "Trường A hiện có 12 khoa và 3 viện nghiên cứu."
DOC_3 = "Đoạn nhiễu không liên quan tới câu hỏi."
CORPUS = {"d0": DOC_0, "d1": DOC_1, "d2": DOC_2, "d3": DOC_3}
#: Bootstrap (BM25, 1) -> d0; the two tools then diverge to d1 / d2.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
#: Free-form multi-gold answer, a fact from d1 joined with a fact from d2.
GOLD_MULTI = "Trường A thành lập năm 1995 và có 12 khoa"
ROW = {"id": "q1", "question": "Trường A có đặc điểm gì?", "answer": GOLD_MULTI,
       "context_ids": ["d1", "d2"], "hop_level": 2, "question_type": "multi_hop"}
ROWS = {"q1": ROW}
RANKINGS_BY_QID = {"q1": RANKINGS}
#: Every reachable ordered state of q1 and the (cached) reader response it maps to.
RESPONSES = {
    ("d0",): '{"answer":"","citations":[]}',
    ("d0", "d1"): '{"answer":"1995","citations":["d1"]}',
    ("d0", "d2"): '{"answer":"12 khoa","citations":["d2"]}',
    ("d0", "d1", "d2"): json.dumps({"answer": GOLD_MULTI, "citations": ["d1", "d2"]}),
    ("d0", "d2", "d1"): json.dumps({"answer": GOLD_MULTI, "citations": ["d1", "d2"]}),
}


def _cache_identity(**overrides) -> CacheIdentity:
    base = dict(model="ds/deepseek-flash", model_revision="provider-managed-alias",
                base_url="https://api.ai-box.vn/v1", prompt_version="v1",
                decoding="temperature=0", max_new_tokens=160,
                corpus_sha256="corpus-sha", rankings_sha256="rankings-sha",
                message_version="chat-user-v1")
    base.update(overrides)
    return CacheIdentity(**base)


def _generation_cache(tmp_path, *, drop=(), rows=ROWS, corpus=CORPUS, rankings=RANKINGS_BY_QID,
                      identity=None):
    """A frozen generation cache holding one response per reachable state (except ``drop``)."""
    cache = GenerationCache(tmp_path / "reader-cache-v2.json", identity or _cache_identity())
    prompts = reader_state_prompts(qids=["q1"], rows=rows, corpus=corpus, rankings=rankings)
    for state_prompt in prompts:
        if state_prompt.state in drop:
            continue
        cache.put(state_prompt.prompt, RESPONSES[state_prompt.state])
    return cache


# --- synthetic split ------------------------------------------------------------------
def test_synthetic_split_builds_worlds_and_reader_cache(tmp_path):
    identity = _cache_identity()
    bundle = assemble_split(role="dev", qids=["q1"], corpus=CORPUS, rows=ROWS,
                            rankings=RANKINGS_BY_QID,
                            cache=_generation_cache(tmp_path, identity=identity))
    assert bundle.role == "dev"
    assert bundle.qids == ("q1",)
    assert len(bundle.worlds) == 1
    # The bundle carries the one identity the cache was validated under, verbatim.
    assert bundle.identity == identity
    # The cache covers exactly the states reachable_states enumerates under the same budget.
    expected = reachable_states(RANKINGS)
    assert bundle.n_states == len(expected) == 5

    world = bundle.worlds[0]
    assert world.qid == "q1" and world.split == "dev"
    assert world.rankings["bm25"] == RANKINGS["bm25"]
    assert world.reader(("d0", "d1")).text == "1995"
    assert world.reader(("d0", "d1")).citation_valid is True
    assert world.reader(("d0",)).text is None  # the reader declined on the noise passage


def test_build_reader_cache_only_covers_reachable_states(tmp_path):
    cache = _generation_cache(tmp_path)
    bundle = assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS,
                            rankings=RANKINGS_BY_QID, cache=cache)
    world = bundle.worlds[0]
    for state in reachable_states(RANKINGS):
        assert world.reader(state) is not None
    # A state the budget never reaches is not in the cache: the environment must not invent it.
    with pytest.raises(KeyError, match="cache miss"):
        world.reader(("d1", "d0"))


# --- cache miss -----------------------------------------------------------------------
def test_a_missing_cached_prompt_fails_instead_of_generating(tmp_path):
    cache = _generation_cache(tmp_path, drop=(("d0", "d1", "d2"),))
    with pytest.raises(KeyError, match="generation cache"):
        assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS,
                       rankings=RANKINGS_BY_QID, cache=cache)


def test_missing_ranking_fails_rather_than_dropping_the_qid(tmp_path):
    with pytest.raises(KeyError, match="ranking"):
        assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS, rankings={},
                       cache=_generation_cache(tmp_path))


# --- multi-gold private ---------------------------------------------------------------
def test_multi_gold_private_is_preserved_and_scored(tmp_path):
    bundle = assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS,
                            rankings=RANKINGS_BY_QID, cache=_generation_cache(tmp_path))
    world = bundle.worlds[0]
    assert world.private.gold_context_ids == ("d1", "d2")
    assert world.private.answer_texts == (GOLD_MULTI,)
    assert world.private.hop_level == 2 and world.private.question_type == "multi_hop"

    # Single-path evidence: only the ordered state holding both gold passages scores.
    env = VixStoryEnv(list(bundle.worlds), VixStoryEnvConfig())
    obs, _ = env.reset(options={"qid": "q1"})
    obs, _, _, _, _ = env.step(SEARCH_BM25)
    obs, _, _, _, _ = env.step(SEARCH_DENSE)  # seen = (d0, d1, d2)
    _, reward, terminated, _, _ = env.step(ANSWER)
    assert terminated and reward == 1.0
    outcome = env.private_outcome()
    assert outcome["correct"] is True and outcome["gold_evidence_recall"] == 1.0
    assert find_forbidden_keys(obs) == []
    assert find_forbidden_keys(env._safe_info()) == []


def test_partial_multi_gold_path_does_not_score(tmp_path):
    bundle = assemble_split(role="train", qids=["q1"], corpus=CORPUS, rows=ROWS,
                            rankings=RANKINGS_BY_QID, cache=_generation_cache(tmp_path))
    env = VixStoryEnv(list(bundle.worlds), VixStoryEnvConfig())
    env.reset(options={"qid": "q1"})
    env.step(SEARCH_BM25)  # seen = (d0, d1): only one gold passage
    _, reward, terminated, _, _ = env.step(ANSWER)
    assert terminated and reward == 0.0
    assert env.private_outcome()["gold_evidence_recall"] == pytest.approx(0.5)


# --- long documents: blocker, not truncation ------------------------------------------
def test_world_uses_same_pilot_budget_as_cache_warmup(tmp_path):
    corpus = {**CORPUS, "d0": "x" * 13000}
    prompts = reader_state_prompts(qids=["q1"], rows=ROWS, corpus=corpus,
                                   rankings=RANKINGS_BY_QID)
    assert prompt_budget_blockers(prompts) == []
    cache = GenerationCache(tmp_path / "reader-cache-v2.json", _cache_identity())
    for state_prompt in prompts:
        cache.put(state_prompt.prompt, RESPONSES[state_prompt.state])
    bundle = assemble_split(role="train", qids=["q1"], corpus=corpus, rows=ROWS,
                            rankings=RANKINGS_BY_QID, cache=cache)
    assert bundle.n_states == len(prompts)


def test_overlong_prompt_is_an_explicit_blocker_not_a_truncation(tmp_path):
    long_doc = "x" * (PILOT_MAX_PROMPT_CHARS + 1)
    corpus = {**CORPUS, "d0": long_doc}
    rows = {"q1": {**ROW, "context_ids": ["d0"]}}
    prompts = reader_state_prompts(qids=["q1"], rows=rows, corpus=corpus,
                                   rankings=RANKINGS_BY_QID)
    blockers = prompt_budget_blockers(prompts)
    assert blockers and all(b["qid"] == "q1" for b in blockers)
    assert blockers[0]["prompt_chars"] > PILOT_MAX_PROMPT_CHARS

    cache = GenerationCache(tmp_path / "reader-cache-v2.json", _cache_identity())
    with pytest.raises(ValueError, match="truncation is not implemented"):
        assemble_split(role="train", qids=["q1"], corpus=corpus, rows=rows,
                       rankings=RANKINGS_BY_QID, cache=cache)


# --- cache identity binding -----------------------------------------------------------
def test_reader_cache_fingerprints_come_from_the_only_identity(tmp_path):
    identity = _cache_identity()
    reader_cache = build_reader_cache(
        qids=["q1"], rows=ROWS, corpus=CORPUS, rankings=RANKINGS_BY_QID,
        cache=_generation_cache(tmp_path, identity=identity))
    assert reader_cache.reader_sha == reader_cache_fingerprints(identity)["reader_sha"]
    assert reader_cache.tokenizer_sha == identity.message_version
    assert reader_cache.inference_config_sha == \
        reader_cache_fingerprints(identity)["inference_config_sha"]

    # A run pinned to a different reader produces a different key: no cross-reader reuse.
    other = _cache_identity(model="different-reader")
    other_cache = build_reader_cache(
        qids=["q1"], rows=ROWS, corpus=CORPUS, rankings=RANKINGS_BY_QID,
        cache=_generation_cache(tmp_path, identity=other))
    assert other_cache.key("q1", ("d0", "d1")) != reader_cache.key("q1", ("d0", "d1"))


def test_cache_written_under_a_different_reader_identity_is_refused(tmp_path):
    path = tmp_path / "reader-cache-v2.json"
    written = GenerationCache(path, _cache_identity())
    written.put("p", "r")
    written.flush()
    # A different cloud model, message contract or provider revision cannot reuse responses.
    for drift in ({"model": "other-reader"},
                  {"message_version": "other-message"},
                  {"model_revision": "rev2"}):
        with pytest.raises(ValueError, match="different reader identity"):
            GenerationCache(path, _cache_identity(**drift))


# --- pinned-input loaders -------------------------------------------------------------
def test_official_test_split_is_never_read(tmp_path):
    path = tmp_path / "test.parquet"
    path.write_bytes(b"not really parquet")
    with pytest.raises(ValueError, match="never read"):
        read_qa_rows(path)


def test_manifest_split_qids_maps_roles_and_checks_hashes():
    qids = ["q1", "q2"]
    record = {"n": len(qids), "qids": list(qids), "qids_sha256": sha256_json(qids)}
    manifest = {"splits": {name: dict(record) for name in MANIFEST_SPLITS.values()}}
    mapped = manifest_split_qids(manifest)
    assert mapped == {"train": ("q1", "q2"), "dev": ("q1", "q2"), "validation": ("q1", "q2")}

    manifest["splits"]["pilot_300"]["qids_sha256"] = "deadbeef"
    with pytest.raises(ValueError, match="hash"):
        manifest_split_qids(manifest)


def test_train_only_manifest_maps_train_and_dev_but_not_validation():
    record = lambda qid: {"n": 1, "qids": [qid], "qids_sha256": sha256_json([qid])}
    manifest = {"splits": {"pilot_300": record("t"), "train_dev_200": record("d")}}
    assert manifest_split_qids(manifest) == {"train": ("t",), "dev": ("d",)}


def test_read_rankings_rejects_a_malformed_tool_list(tmp_path):
    path = tmp_path / "rankings.json"
    write_json(path, {"q1": {"bm25": ["d0"], "dense": "not-a-list"}})
    with pytest.raises(ValueError, match="dense"):
        read_rankings(path)


def test_unknown_role_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown pilot role"):
        assemble_split(role="test", qids=["q1"], corpus=CORPUS, rows=ROWS,
                       rankings=RANKINGS_BY_QID, cache=_generation_cache(tmp_path))

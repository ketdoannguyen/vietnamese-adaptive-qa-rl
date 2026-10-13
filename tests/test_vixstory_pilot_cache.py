"""VixSTORY pilot reader-cache warm-up contracts (CPU/synthetic, no model load, no GPU).

The warm-up pass plans every reachable ordered retrieval state, fails fast on invalid inputs,
reports cloud failures as incomplete coverage, persists incrementally so a long run resumes,
and binds the cache to the reader identity. Every
test injects a deterministic fake reader and reads from synthetic in-memory fixtures; the real
model is never loaded and the official test split is never touched.
"""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.vixstory_pilot_cache import (
    COVERAGE_NAME,
    MANIFEST_SPLITS,
    PILOT_DIR,
    EpisodeBudget,
    StatePrompt,
    assert_pilot_namespace,
    char_blockers,
    load_manifest_splits,
    next_free_path,
    read_corpus,
    read_questions,
    read_rankings,
    run,
    state_prompts,
    warm_stage,
)
from scripts.vixstory_reader_eval import PILOT_MAX_PROMPT_CHARS, CacheIdentity, GenerationCache
from src.retrieval.states import reachable_states
from src.utils.hashing import sha256_json
from src.utils.io import write_json

# --- synthetic pinned inputs (the same tiny tree the worlds tests use) ----------------
DOC_0 = "VixSTORY: trường A toạ lạc tại Hà Nội."
DOC_1 = "Trường A thành lập năm 1995 tại Hà Nội."
DOC_2 = "Trường A hiện có 12 khoa và 3 viện nghiên cứu."
DOC_3 = "Đoạn nhiễu không liên quan tới câu hỏi."
CORPUS = {"d0": DOC_0, "d1": DOC_1, "d2": DOC_2, "d3": DOC_3}
#: Bootstrap (BM25, 1) -> d0; the two tools then diverge to d1 / d2. d3 is never reachable.
RANKINGS = {"bm25": ["d0", "d1", "d2", "d3"], "dense": ["d0", "d2", "d1", "d3"]}
QUESTION = "Trường A có đặc điểm gì?"
GOLD_ANSWER = "Trường A thành lập năm 1995 và có 12 khoa"  # must never reach a prompt
BUDGET = EpisodeBudget()  # bm25 / 1 / 1 / 2 -> 5 reachable states
N_STATES = 5


def _identity(**overrides) -> CacheIdentity:
    base = dict(model="ds/deepseek-flash", model_revision="provider-managed-alias",
                base_url="https://api.ai-box.vn/v1", prompt_version="v1",
                decoding="temperature=0", max_new_tokens=160,
                corpus_sha256="corpus-sha", rankings_sha256="rankings-sha",
                message_version="chat-user-v1")
    base.update(overrides)
    return CacheIdentity(**base)


class FakeReader:
    """Deterministic stand-in for ``CloudReader``: generates missing prompts into the cache.

    ``boom`` makes ``generate`` raise on any prompt containing one of its markers, so a test can
    interrupt a warm-up mid-way and observe the incrementally flushed cache.
    """

    def __init__(self, cache: GenerationCache, *, model_context=None, max_new_tokens=None,
                 tokenizer=None, boom=()) -> None:
        self.cache = cache
        self.tokenizer = tokenizer
        self.model_context = model_context
        self.max_new_tokens = max_new_tokens
        self.boom = tuple(boom)
        self.generated = 0
        self.flushes = 0

    def generate(self, prompts, *, workers=1):
        for prompt in dict.fromkeys(prompts):
            if any(marker in prompt for marker in self.boom):
                raise RuntimeError("boom")
            if prompt not in self.cache:
                self.cache.put(prompt, json.dumps({"answer": "x", "citations": []}))
                self.generated += 1
        return self.cache.get(prompts)

    def flush(self) -> None:
        self.cache.flush()
        self.flushes += 1


def _reader(tmp_path, **kwargs):
    cache_path = tmp_path / "reader-cache-pilot.json"
    return cache_path, FakeReader(GenerationCache(cache_path, _identity()), **kwargs)


# --- planning: every reachable state, no labels ---------------------------------------
def test_state_prompts_cover_every_reachable_ordered_state():
    prompts = state_prompts(qids=["q1"], questions={"q1": QUESTION}, rankings={"q1": RANKINGS},
                            corpus=CORPUS, budget=BUDGET)
    assert [sp.state for sp in prompts] == reachable_states(RANKINGS)
    assert len(prompts) == N_STATES


def test_state_prompts_show_visible_passages_only_and_hide_gold():
    prompts = state_prompts(qids=["q1"], questions={"q1": QUESTION}, rankings={"q1": RANKINGS},
                            corpus=CORPUS, budget=BUDGET)
    for sp in prompts:
        assert QUESTION in sp.prompt
        for doc_id in sp.state:
            assert CORPUS[doc_id] in sp.prompt
        assert GOLD_ANSWER not in sp.prompt           # a reference answer never enters a prompt
        assert CORPUS["d3"] not in sp.prompt          # an unreachable passage is never shown
    # The deepest state reveals three documents; the bootstrap state reveals exactly one.
    assert len(prompts[0].state) == 1 and len(max(prompts, key=lambda sp: len(sp.state)).state) == 3


def test_episode_budget_required_top_n_matches_the_deepest_state():
    assert BUDGET.required_top_n() == 1 + 1 * 2


# --- fail fast: no dropped state ------------------------------------------------------
def test_over_char_prompt_is_reported_per_state_never_silently_dropped():
    long_corpus = dict(CORPUS)
    long_corpus["d0"] = "x" * (PILOT_MAX_PROMPT_CHARS + 1)
    prompts = state_prompts(qids=["q1"], questions={"q1": QUESTION}, rankings={"q1": RANKINGS},
                            corpus=long_corpus, budget=BUDGET)
    blockers = char_blockers(prompts)
    # Every state carries the over-long bootstrap passage, so every state is reported as a
    # blocker rather than being filtered out of the plan.
    assert len(blockers) == N_STATES
    assert all(b["qid"] == "q1" and b["prompt_chars"] > PILOT_MAX_PROMPT_CHARS for b in blockers)


def test_pilot_accepts_state_over_historical_char_cap_when_within_token_budget(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    corpus = dict(CORPUS)
    corpus["d0"] = "x" * 13000
    record = warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION},
                        corpus=corpus, rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET)
    assert record["n_states"] == N_STATES and record["states_verified"]


def test_over_char_prompt_fails_fast_without_generating(tmp_path):
    cache_path, reader = _reader(tmp_path)
    long_corpus = dict(CORPUS)
    long_corpus["d0"] = "x" * (PILOT_MAX_PROMPT_CHARS + 1)
    with pytest.raises(ValueError, match="MAX_PROMPT_CHARS"):
        warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION},
                   corpus=long_corpus, rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET)
    assert reader.generated == 0 and len(reader.cache) == 0


class _TokenizerStub:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return messages[0]["content"]

    def __call__(self, rendered, add_special_tokens=False, return_length=False):
        return {"length": [10**6] * len(rendered)}


def test_over_context_prompt_fails_fast_without_generating(tmp_path):
    cache_path, reader = _reader(tmp_path, model_context=128, max_new_tokens=16,
                                 tokenizer=_TokenizerStub())
    with pytest.raises(ValueError, match="exceeds"):
        warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=CORPUS,
                   rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET)
    assert reader.generated == 0 and len(reader.cache) == 0


def test_missing_question_row_is_never_skipped(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    with pytest.raises(KeyError, match="question row"):
        warm_stage(role="train", qids=["q1", "q2"], questions={"q1": QUESTION},
                   corpus=CORPUS, rankings={"q1": RANKINGS, "q2": RANKINGS},
                   reader=reader, budget=BUDGET)


def test_missing_ranking_is_never_skipped(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    with pytest.raises(KeyError, match="ranking"):
        warm_stage(role="train", qids=["q1", "q2"], questions={"q1": QUESTION, "q2": QUESTION},
                   corpus=CORPUS, rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET)


def test_ranking_shorter_than_the_deepest_state_fails_fast(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    short = {"bm25": ["d0"], "dense": ["d0"]}
    with pytest.raises(ValueError, match="shorter"):
        warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=CORPUS,
                   rankings={"q1": short}, reader=reader, budget=BUDGET)


# --- generation + incremental persistence ---------------------------------------------
def test_warm_generates_every_state_and_flushes_per_batch(tmp_path):
    cache_path, reader = _reader(tmp_path)
    record = warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=CORPUS,
                        rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET, batch_size=2)
    assert record["n_states"] == N_STATES and record["n_generated"] == N_STATES
    assert record["states_verified"] is True
    assert reader.flushes == 3  # 5 missing prompts at 2 per batch -> three flushes
    # The flushed file alone holds every state, so a fresh reader over it owes nothing.
    reloaded = GenerationCache(cache_path, _identity())
    assert len(reloaded) == N_STATES


def test_rerun_resumes_and_regenerates_nothing(tmp_path):
    cache_path, reader = _reader(tmp_path)
    warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=CORPUS,
               rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET)
    resumed = FakeReader(GenerationCache(cache_path, _identity()))
    record = warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=CORPUS,
                        rankings={"q1": RANKINGS}, reader=resumed, budget=BUDGET)
    assert record["n_generated"] == 0 and resumed.generated == 0
    assert record["states_verified"] is True


def test_a_crash_mid_run_keeps_the_flushed_batches(tmp_path):
    cache_path = tmp_path / "reader-cache-pilot.json"
    reader = FakeReader(GenerationCache(cache_path, _identity()), boom=("BOOM",))
    corpus = dict(CORPUS)
    corpus["d2"] = DOC_2 + " BOOM"  # only the states that reveal d2 trip the failure
    with pytest.raises(RuntimeError, match="boom"):
        warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION}, corpus=corpus,
                   rankings={"q1": RANKINGS}, reader=reader, budget=BUDGET, batch_size=2)
    # The first batch (states 1 and 2) was flushed before the crashing batch was generated.
    assert len(GenerationCache(cache_path, _identity())) == 2


def test_exhausted_prompt_is_skipped_while_other_roles_continue(tmp_path):
    cache_path, reader = _reader(tmp_path)
    failed = state_prompts(qids=["q1"], questions={"q1": _toy_questions()["q1"]},
                           rankings={"q1": RANKINGS}, corpus=CORPUS, budget=BUDGET)[0].prompt
    reader.cache.retries[failed] = 5
    reader.flush()
    coverage = run(stages=("train", "dev"), manifest=_toy_manifest(), corpus=CORPUS,
                   rankings={"q1": RANKINGS, "q2": RANKINGS},
                   train_questions=_toy_questions(), validation_questions={}, reader=reader,
                   budget=BUDGET, out_dir=tmp_path, inputs=_toy_inputs(), batch_size=2)
    assert coverage["all_states_verified"] is False
    assert coverage["splits"]["train"]["n_missing_prompts"] == 1
    assert coverage["splits"]["train"]["n_failed_prompts"] == 1
    assert coverage["splits"]["train"]["n_missing_states"] == 1
    assert coverage["splits"]["dev"]["states_verified"] is True
    assert coverage["totals"]["n_missing_states"] == 1
    assert reader.generated == 2 * N_STATES - 1
    assert reader.cache.retries[failed] == 5 and failed not in reader.cache
    assert len(GenerationCache(cache_path, _identity())) == 2 * N_STATES - 1


class OneFailedPromptReader(FakeReader):
    def __init__(self, cache, failed):
        super().__init__(cache)
        self.failed = failed

    def generate(self, prompts, *, workers=1):
        if self.failed in prompts:
            self.cache.retries[self.failed] = 5
            self.flush()
            raise RuntimeError("cloud reader request failed (transport error); response and credentials withheld")
        return super().generate(prompts, workers=workers)


def test_failed_request_does_not_stop_later_prompts_in_batch(tmp_path):
    cache_path, fake = _reader(tmp_path)
    first = state_prompts(qids=["q1"], questions={"q1": QUESTION},
                          rankings={"q1": RANKINGS}, corpus=CORPUS, budget=BUDGET)[0].prompt
    reader = OneFailedPromptReader(fake.cache, first)
    result = warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION},
                        corpus=CORPUS, rankings={"q1": RANKINGS}, reader=reader,
                        budget=BUDGET, batch_size=N_STATES)
    assert result["n_missing_states"] == 1 and reader.generated == N_STATES - 1
    assert len(GenerationCache(cache_path, _identity())) == N_STATES - 1


def test_terminal_identity_failure_stops_warm_up(tmp_path):
    _path, fake = _reader(tmp_path)
    first = state_prompts(qids=["q1"], questions={"q1": QUESTION},
                          rankings={"q1": RANKINGS}, corpus=CORPUS, budget=BUDGET)[0].prompt
    reader = OneFailedPromptReader(fake.cache, first)
    def fail_identity(prompts, *, workers=1):
        reader.cache.failures[first] = "cloud model identity changed during run"
        reader.flush()
        raise RuntimeError("cloud reader request failed (cloud model identity changed during run); response and credentials withheld")
    reader.generate = fail_identity
    with pytest.raises(RuntimeError, match="identity changed"):
        warm_stage(role="train", qids=["q1"], questions={"q1": QUESTION},
                   corpus=CORPUS, rankings={"q1": RANKINGS}, reader=reader,
                   budget=BUDGET)


def test_cache_is_bound_to_the_reader_identity(tmp_path):
    cache_path = tmp_path / "reader-cache-pilot.json"
    cache = GenerationCache(cache_path, _identity())
    cache.put("p", "r")
    cache.flush()
    with pytest.raises(ValueError):
        GenerationCache(cache_path, _identity(model_revision="rev2"))


# --- coverage manifest -----------------------------------------------------------------
def _toy_manifest():
    return {"train": ("q1",), "dev": ("q2",), "validation": ("q3",)}


def _toy_questions():
    # Distinct question text per QID, so each split's state prompts are their own cache entries
    # (the real pilot splits are disjoint, so no qid is shared across roles).
    return {qid: f"{QUESTION} [{qid}]" for qid in ("q1", "q2", "q3")}


def _toy_inputs():
    return {"corpus": {"path": "corpus.parquet", "sha256": "corpus-sha"},
            "rankings": {"path": "pilot_rankings_top12.json", "sha256": "rankings-sha"}}


def test_run_writes_coverage_with_counts_hashes_and_verification(tmp_path):
    cache_path, reader = _reader(tmp_path)
    coverage = run(stages=("train", "dev", "validation"), manifest=_toy_manifest(),
                   corpus=CORPUS, rankings={"q1": RANKINGS, "q2": RANKINGS, "q3": RANKINGS},
                   train_questions=_toy_questions(), validation_questions=_toy_questions(),
                   reader=reader, budget=BUDGET, out_dir=tmp_path, inputs=_toy_inputs())
    assert coverage["all_states_verified"] is True
    assert coverage["totals"] == {"n_qids": 3, "n_states": 3 * N_STATES,
                                  "n_missing_states": 0,
                                  "generated_this_run": 3 * N_STATES}
    assert coverage["inputs"]["official_test_loaded"] is False
    assert set(coverage["splits"]) == {"train", "dev", "validation"}
    assert coverage["splits"]["train"]["qids_sha256"] == sha256_json(["q1"])
    assert coverage["cache"]["identity"]["model_revision"] == "provider-managed-alias"
    assert coverage["cache"]["cache_version"] == 2
    written = tmp_path / COVERAGE_NAME
    assert written.is_file() and coverage["written"] == str(written)
    on_disk = json.loads(written.read_text(encoding="utf-8"))
    assert on_disk["all_states_verified"] is True and "written" not in on_disk


def test_run_never_overwrites_a_historic_coverage_file(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    kwargs = dict(manifest=_toy_manifest(), corpus=CORPUS,
                  rankings={"q1": RANKINGS, "q2": RANKINGS, "q3": RANKINGS},
                  train_questions=_toy_questions(), validation_questions=_toy_questions(),
                  budget=BUDGET, out_dir=tmp_path, inputs=_toy_inputs())
    first = run(stages=("train",), reader=reader, **kwargs)
    second = run(stages=("train",), reader=reader, **kwargs)
    assert (tmp_path / COVERAGE_NAME).is_file()
    assert first["written"] != second["written"]
    assert second["written"].endswith("cache-coverage.v2.json")


def test_unknown_stage_is_rejected(tmp_path):
    _cache_path, reader = _reader(tmp_path)
    with pytest.raises(ValueError, match="unknown stage"):
        run(stages=("test",), manifest=_toy_manifest(), corpus=CORPUS, rankings={},
            train_questions={}, validation_questions={}, reader=reader, budget=BUDGET,
            out_dir=tmp_path, inputs=_toy_inputs())


# --- loaders and namespace -------------------------------------------------------------
def test_next_free_path_versions_instead_of_overwriting(tmp_path):
    path = tmp_path / COVERAGE_NAME
    assert next_free_path(path) == path
    path.write_text("{}", encoding="utf-8")
    assert next_free_path(path).name == "cache-coverage.v2.json"


def test_assert_pilot_namespace_accepts_inside_and_rejects_outside():
    assert_pilot_namespace(PILOT_DIR / "reader-cache-pilot.json", PILOT_DIR)
    with pytest.raises(ValueError, match="pilot namespace"):
        assert_pilot_namespace("/tmp/elsewhere/reader-cache-pilot.json", PILOT_DIR)


def test_load_manifest_splits_checks_qid_hashes(tmp_path):
    path = tmp_path / "manifest.json"
    record = {"n": 1, "qids": ["q1"], "qids_sha256": sha256_json(["q1"])}
    write_json(path, {"splits": {name: dict(record) for name in MANIFEST_SPLITS.values()}})
    assert load_manifest_splits(path) == {"train": ("q1",), "dev": ("q1",), "validation": ("q1",)}
    write_json(path, {"splits": {name: {"n": 1, "qids": ["q1"], "qids_sha256": "deadbeef"}
                                 for name in MANIFEST_SPLITS.values()}})
    with pytest.raises(ValueError, match="hash"):
        load_manifest_splits(path)


def test_read_questions_projects_only_id_and_question(tmp_path):
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "q1", "question": "Q?", "answer": "SECRET"}]),
                   path)
    assert read_questions(path) == {"q1": "Q?"}
    with pytest.raises(ValueError, match="never read"):
        read_questions(tmp_path / "test.parquet")


def test_read_corpus_roundtrips_and_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "corpus.parquet"
    pq.write_table(pa.Table.from_pylist([{"context_id": "d0", "text": "t0"},
                                         {"context_id": "d1", "text": "t1"}]), path)
    assert read_corpus(path) == {"d0": "t0", "d1": "t1"}
    pq.write_table(pa.Table.from_pylist([{"context_id": "d0", "text": "t0"},
                                         {"context_id": "d0", "text": "dup"}]), path)
    with pytest.raises(ValueError, match="duplicate"):
        read_corpus(path)


def test_read_rankings_rejects_a_malformed_tool_list(tmp_path):
    path = tmp_path / "rankings.json"
    write_json(path, {"q1": {"bm25": ["d0"], "dense": "not-a-list"}})
    with pytest.raises(ValueError, match="dense"):
        read_rankings(path)


def test_cli_requires_explicit_stage_before_opening_inputs(monkeypatch):
    import sys

    from scripts import vixstory_pilot_cache as module

    monkeypatch.setattr(sys, "argv", ["vixstory_pilot_cache"])
    monkeypatch.setattr(module, "load_manifest_splits", lambda _: pytest.fail("manifest opened"))
    with pytest.raises(SystemExit, match="2"):
        module.main()


@pytest.mark.parametrize("stage", ["train", "dev", "validation", "all"])
def test_cli_opens_and_hashes_only_requested_qa_sources(tmp_path, monkeypatch, stage):
    import sys
    from pathlib import Path

    from scripts import vixstory_pilot_cache as module

    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    expected = ({train_path, validation_path} if stage == "all" else
                {validation_path} if stage == "validation" else {train_path})
    reads, hashes = [], []

    def read_questions_spy(path):
        assert path in expected, f"unrequested QA read: {path}"
        reads.append(path)
        return _toy_questions()

    def hash_spy(path):
        path = Path(path)
        if path in (train_path, validation_path):
            assert path in expected, f"unrequested QA hash: {path}"
            hashes.append(path)
        return "synthetic-sha"

    _, reader = _reader(tmp_path)
    monkeypatch.setattr(module, "load_manifest_splits", lambda _: _toy_manifest())
    monkeypatch.setattr(module, "read_corpus", lambda _: CORPUS)
    monkeypatch.setattr(module, "read_rankings", lambda _: {
        q: RANKINGS for q in _toy_questions()})
    monkeypatch.setattr(module, "read_questions", read_questions_spy)
    monkeypatch.setattr(module, "sha256_file", hash_spy)
    monkeypatch.setattr(module, "build_identity", lambda **_: _identity())
    monkeypatch.setattr(module, "CloudReader", lambda *a, **kw: reader)
    monkeypatch.setattr(sys, "argv", ["vixstory_pilot_cache", "--stage", stage,
        "--pilot-dir", str(tmp_path), "--train", str(train_path),
        "--validation", str(validation_path)])

    assert module.main() == 0
    assert set(reads) == expected and set(hashes) == expected
    coverage = json.loads((tmp_path / COVERAGE_NAME).read_text())
    assert ("validation" in coverage["inputs"]) == (validation_path in expected)
    assert ("train" in coverage["inputs"]) == (train_path in expected)


def test_cache_lock_refuses_another_writer(tmp_path):
    from src.utils.runs import LockHeld, OutputNamespaceLock

    lock_path = tmp_path / "cache.json.lock"
    with OutputNamespaceLock(lock_path):
        with pytest.raises(LockHeld):
            with OutputNamespaceLock(lock_path):
                pytest.fail("second writer entered")
    with OutputNamespaceLock(lock_path):
        pass


def test_state_prompt_is_a_frozen_value():
    prompt = StatePrompt(qid="q1", state=("d0",), prompt="p")
    assert prompt.state == ("d0",) and prompt.prompt == "p"

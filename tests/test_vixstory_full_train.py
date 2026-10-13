"""Full official-train warm-up contracts with synthetic data and no network."""

import json

from scripts import vixstory_full_train as full
from scripts.vixstory_pilot_cache import EpisodeBudget, state_prompts
from scripts.vixstory_reader_eval import CacheIdentity, GenerationCache


CORPUS = {"a": "first", "b": "second", "c": "third", "d": "fourth"}
RANKING = {"bm25": ["a", "b", "c", "d"], "dense": ["a", "c", "b", "d"]}
QUESTIONS = {"q1": "First question?", "q2": "Second question?"}


def _identity():
    return CacheIdentity(model="ds/deepseek-flash", model_revision="provider-managed-alias",
                         base_url="https://api.ai-box.vn/v1", prompt_version="v3-complete-concise",
                         decoding="temperature=0", max_new_tokens=16384,
                         corpus_sha256="corpus", rankings_sha256="ranking",
                         message_version="chat-user-v1")


class FakeReader:
    def __init__(self, path, *, identity, max_requests):
        self.cache = GenerationCache(path, identity)
        self.max_new_tokens = identity.max_new_tokens

    def generate(self, prompts, *, workers=1):
        for prompt in dict.fromkeys(prompts):
            self.cache.put(prompt, '{"answer":"yes","citations":[]}')
        return self.cache.get(prompts)

    def flush(self):
        self.cache.flush()


def test_full_warm_refuses_more_than_authorized_attempts(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="30000"):
        full.warm(tmp_path, max_requests=30001)


def test_full_training_refuses_incomplete_reader_cache(tmp_path, monkeypatch):
    import pytest

    rankings = {qid: RANKING for qid in QUESTIONS}
    monkeypatch.setattr(full, "_inputs", lambda *_: (sorted(QUESTIONS), QUESTIONS, CORPUS, rankings))
    monkeypatch.setattr(full, "build_identity", lambda **_: _identity())
    monkeypatch.setattr(full, "TRAIN_COUNT", len(QUESTIONS))
    with pytest.raises(RuntimeError, match="coverage incomplete"):
        full.train_inputs(tmp_path)
    assert not (tmp_path / full.TEACHER).exists()


def test_full_warm_skips_failed_question_but_completes_other_qids(tmp_path, monkeypatch):
    rankings = {qid: RANKING for qid in QUESTIONS}
    monkeypatch.setattr(full, "_inputs", lambda *_: (sorted(QUESTIONS), QUESTIONS, CORPUS, rankings))
    monkeypatch.setattr(full, "build_identity", lambda **_: _identity())
    monkeypatch.setattr(full, "CloudReader", FakeReader)
    failed = state_prompts(qids=["q1"], questions=QUESTIONS, corpus=CORPUS,
                           rankings=rankings, budget=EpisodeBudget())[0].prompt
    cache = GenerationCache(tmp_path / full.CACHE, _identity())
    cache.retries[failed] = 5
    cache.flush()

    result = full.warm(tmp_path, max_requests=20, qid_batch_size=1)
    assert result["n_official_train"] == 2
    assert result["n_complete_qids"] == 1 and result["complete_qids"] == ["q2"]
    assert result["incomplete"] == {"q1": {"missing_states": 1,
                                            "reasons": ["attempts_exhausted"]}}
    assert result["dev200_is_held_out"] is False
    assert json.loads((tmp_path / full.FAILED).read_text()) == result
    assert GenerationCache(tmp_path / full.CACHE, _identity()).retries[failed] == 5

    resumed = full.warm(tmp_path, max_requests=20, qid_batch_size=1)
    assert resumed["entries"] == result["entries"]
    assert resumed["n_complete_qids"] == result["n_complete_qids"]
    assert (tmp_path / "failed-qids.v2.json").is_file()
    assert json.loads((tmp_path / full.FAILED).read_text()) == result


def test_full_coverage_keeps_unattempted_qids_in_pending_queue(tmp_path):
    rankings = {qid: RANKING for qid in QUESTIONS}
    cache = GenerationCache(tmp_path / full.CACHE, _identity())
    result = full._coverage(sorted(QUESTIONS), QUESTIONS, CORPUS, rankings, cache,
                            EpisodeBudget())
    assert result["n_complete_qids"] == 0 and result["n_incomplete_qids"] == 2
    assert set(result["incomplete"]) == set(QUESTIONS)
    assert all(item["reasons"] == ["pending"] for item in result["incomplete"].values())

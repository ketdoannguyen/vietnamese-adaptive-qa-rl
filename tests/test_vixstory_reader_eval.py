"""VixSTORY reader and search-advantage contracts (CPU/synthetic, no model load)."""

import json
import sys

import pytest

from scripts.vixstory_reader_eval import (
    CACHE_VERSION,
    CacheIdentity,
    GenerationCache,
    answer_reward,
    assert_prompts_fit,
    CloudReader,
    BASE_URL,
    MODEL,
    parse_reader_output,
    prompt_token_budget,
    reader_prompt,
    refuse_overwrite,
    result_paths,
    search_advantage,
    select_scored_rows,
    stage_gold,
    stage_states,
)


def _identity(**overrides) -> CacheIdentity:
    base = dict(model=MODEL, model_revision="r1", base_url=BASE_URL, prompt_version="v1",
                decoding="temperature=0", max_new_tokens=160, corpus_sha256="c",
                rankings_sha256="k", message_version="chat-user-v1")
    base.update(overrides)
    return CacheIdentity(**base)


# --- prompt / parsing ----------------------------------------------------------------

def test_reader_prompt_shows_only_supplied_passages():
    prompt = reader_prompt("Học phí bao nhiêu?", [
        {"context_id": "ctx-a", "text": "Học phí là 10 đồng."},
    ])
    assert "ctx-a" in prompt
    assert "10 đồng" in prompt
    assert "ctx-b" not in prompt
    assert "gold" not in prompt.lower()


def test_parse_accepts_bare_json_and_fenced_json():
    text, cited, valid = parse_reader_output('{"answer":"10 đồng","citations":["a"]}', ["a"])
    assert text == "10 đồng" and cited == ("a",) and valid
    fenced = '```json\n{"answer": "10 đồng", "citations": ["a"]}\n```'
    assert parse_reader_output(fenced, ["a"])[:1] == ("10 đồng",)
    # Trailing prose after the object must not defeat the parse.
    assert parse_reader_output('{"answer":"x","citations":["a"]} (theo đoạn a)', ["a"])[0] == "x"


def test_parse_rejects_empty_answer_and_unknown_citation():
    assert parse_reader_output('{"answer":"   ","citations":["a"]}', ["a"])[0] is None
    assert parse_reader_output('{"answer":"x","citations":["b"]}', ["a"])[2] is False
    assert parse_reader_output('{"answer":"x","citations":[]}', ["a"])[2] is False
    assert parse_reader_output("không phải json", ["a"])[0] is None


def test_parse_keeps_answer_when_citations_are_absent():
    """A missing citation invalidates provenance, but the answer text is still scorable."""
    text, cited, valid = parse_reader_output('{"answer":"10 đồng"}', ["a"])
    assert text == "10 đồng" and cited == () and valid is False


def test_parse_keeps_answer_when_citations_are_null_or_not_strings():
    """``null`` / non-string citations invalidate provenance without discarding the answer."""
    assert parse_reader_output('{"answer":"x","citations":null}', ["a"]) == ("x", (), False)
    assert parse_reader_output('{"answer":"x","citations":[1,2]}', ["a"]) == ("x", (), False)
    assert parse_reader_output('{"answer":"x","citations":{"a":1}}', ["a"]) == ("x", (), False)
    assert parse_reader_output('{"answer":"x","citation":42}', ["a"]) == ("x", (), False)


def test_parse_decodes_a_single_object_ignoring_trailing_braces():
    """Only the first complete object is decoded; trailing braces or prose never defeat it."""
    assert parse_reader_output('{"answer":"x","citations":["a"]}}}', ["a"]) == ("x", ("a",), True)
    assert parse_reader_output('{"answer":"x","citations":["a"]} {"extra":1}', ["a"]) == (
        "x", ("a",), True)


def test_select_scored_rows_excludes_over_budget_and_reports_them():
    rows = [{"id": "a", "context_ids": ["x"]}, {"id": "b", "context_ids": ["y", "z"]}]
    docs = {"x": {"text": "t" * 10}, "y": {"text": "t" * 8}, "z": {"text": "t" * 8}}
    kept, excluded = select_scored_rows(rows, docs, max_prompt_passage_chars=12)
    assert [r["id"] for r in kept] == ["a"]
    assert excluded == ["b"]


# --- reward: answer F1 vs citation-ID validity --------------------------------------

def test_answer_reward_keeps_f1_and_citation_validity_separate():
    # A correct answer with a valid citation pays; a correct answer with a bad citation does
    # not, under the locked gate — a valid ID is provenance, not entailment.
    assert answer_reward(0.9, True) == 1.0
    assert answer_reward(0.9, False) == 0.0
    # Text below the threshold never pays, citation or not.
    assert answer_reward(0.4, True) == 0.0
    assert answer_reward(0.5, True) == 1.0


@pytest.mark.parametrize("f1,expected", [(0.5 - 2e-12, 0.0),
                                              (0.5 - 5e-13, 1.0),
                                              (0.5, 1.0), (0.5 + 2e-12, 1.0)])
def test_reader_eval_reward_matches_environment_threshold(f1, expected):
    assert answer_reward(f1, True) == expected
    assert answer_reward(f1, False) == 0.0


def test_answer_reward_sensitivity_arm_drops_the_citation_gate():
    assert answer_reward(0.9, False, gate_citation=False) == 1.0
    assert answer_reward(0.4, False, gate_citation=False) == 0.0
    # The gate only ever removes reward; it never adds it.
    assert answer_reward(0.9, True, gate_citation=False) == answer_reward(0.9, True)


# --- prompt token budget: fail fast, never truncate ----------------------------------

def test_prompt_token_budget_reserves_output_tokens():
    assert prompt_token_budget(32768, 160) == 32608
    with pytest.raises(ValueError):
        prompt_token_budget(100, 160)


def test_assert_prompts_fit_rejects_over_budget_instead_of_truncating():
    assert_prompts_fit([10, 32608], model_context=32768, max_new_tokens=160)
    with pytest.raises(ValueError):
        assert_prompts_fit([32609], model_context=32768, max_new_tokens=160)


# --- generation cache: identity binding -----------------------------------------------

def test_cache_roundtrip_and_identity_drift(tmp_path):
    path = tmp_path / "cache.json"
    cache = GenerationCache(path, _identity())
    assert len(cache) == 0
    cache.put("p", "resp")
    cache.flush()

    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["cache_version"] == CACHE_VERSION
    assert set(envelope["identity"]) == set(_identity().as_dict())

    reloaded = GenerationCache(path, _identity())
    assert reloaded.get(["p"]) == ["resp"]
    # A different model revision is a different reader; reuse must be refused.
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(model_revision="r2"))
    # So is a different prompt version, corpus, or ranking fingerprint.
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(prompt_version="v2"))
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(corpus_sha256="other"))
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(rankings_sha256="other"))


def test_cache_refuses_different_cloud_message_contract(tmp_path):
    path = tmp_path / "cache.json"
    cache = GenerationCache(path, _identity())
    cache.put("p", "resp")
    cache.flush()
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(message_version="other"))
    with pytest.raises(ValueError):
        GenerationCache(path, _identity(base_url="https://other.example/v1"))


# --- result files: never clobber a historical measurement -----------------------------

def test_result_paths_match_the_stage_outputs(tmp_path):
    assert result_paths("gold", tmp_path) == (tmp_path / "reader-gold-train500.json",
                                              tmp_path / "reader-gold-train500-rows.json")
    assert result_paths("states", tmp_path, "-boot2") == (
        tmp_path / "search-advantage-train500-boot2.json",
        tmp_path / "search-advantage-train500-boot2-rows.json")


def test_refuse_overwrite_passes_when_absent_and_names_existing(tmp_path):
    refuse_overwrite((tmp_path / "nope.json",))  # nothing to clobber
    present = tmp_path / "there.json"
    present.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="there.json"):
        refuse_overwrite((present,))


def test_stage_gold_refuses_overwrite_before_generation(tmp_path):
    (tmp_path / "reader-gold-train500-rows.json").write_text("[]", encoding="utf-8")
    # A dummy reader is never touched: the refusal happens before any generation.
    with pytest.raises(FileExistsError):
        stage_gold(object(), {}, [], out=tmp_path)


def test_stage_states_refuses_overwrite_before_generation(tmp_path):
    (tmp_path / "search-advantage-train500.json").write_text("{}", encoding="utf-8")
    # The rankings file does not even exist; the guard must fire first.
    with pytest.raises(FileExistsError):
        stage_states(object(), {}, [], out=tmp_path, rankings_path=tmp_path / "missing.json")


def test_cli_refuses_overwrite_before_loading(tmp_path, monkeypatch):
    from scripts import vixstory_reader_eval as mod

    (tmp_path / "reader-gold-train500.json").write_text("{}", encoding="utf-8")

    def _boom():
        raise AssertionError("data must not load when a result file already exists")

    monkeypatch.setattr(mod, "_load", _boom)
    monkeypatch.setattr(sys, "argv", ["vixstory_reader_eval", "--stage", "gold",
                                      "--out", str(tmp_path)])
    with pytest.raises(FileExistsError):
        mod.main()


def test_cloud_reader_sends_only_prompt_and_observes_request_cap(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setenv("AIBOX_API_KEY", "synthetic-secret")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={"model": MODEL, "choices": [{"message": {
            "content": '{"answer":"10 đồng","citations":["a"]}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10}})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    reader = CloudReader(tmp_path / "cloud.json", identity=_identity(), max_requests=1,
                         client=client)
    assert reader.generate(["visible question and passage a"]) == [
        '{"answer":"10 đồng","citations":["a"]}']
    assert reader.requests == reader.generated == 1
    assert reader.generate(["visible question and passage a"]) == reader.cache.get([
        "visible question and passage a"])
    with pytest.raises(RuntimeError, match="cap"):
        reader.generate(["another prompt"])
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["messages"] == [{"role": "user", "content": "visible question and passage a"}]
    assert requests[0].url == f"{BASE_URL}/chat/completions"
    assert reader.usage == [{"prompt_tokens": 20, "completion_tokens": 10}]
    assert "synthetic-secret" not in (tmp_path / "cloud.json").read_text()


def test_cloud_reader_requires_env_and_redacts_provider_errors(tmp_path, monkeypatch):
    import httpx

    monkeypatch.delenv("AIBOX_API_KEY", raising=False)
    reader = CloudReader(tmp_path / "cache.json", identity=_identity(),
                         client=httpx.Client(transport=httpx.MockTransport(lambda r: None)))
    with pytest.raises(RuntimeError, match="AIBOX_API_KEY"):
        reader.generate(["p"])
    monkeypatch.setenv("AIBOX_API_KEY", "synthetic-secret")
    client = httpx.Client(transport=httpx.MockTransport(
        lambda _r: httpx.Response(401, text="synthetic-secret")))
    reader = CloudReader(tmp_path / "cache.json", identity=_identity(), client=client)
    with pytest.raises(RuntimeError) as exc:
        reader.generate(["p"])
    assert "synthetic-secret" not in str(exc.value)
    assert exc.value.__context__ is None
    assert reader.requests == 1 and reader.cache.attempts == 1 and len(reader.cache) == 0


def test_cache_refuses_legacy_prompt_only_json(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"some prompt": "some response"}), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy"):
        GenerationCache(path, _identity())


def test_cache_refuses_unknown_version_and_malformed_entries(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"cache_version": CACHE_VERSION + 1, "identity": {},
                                "entries": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="cache_version"):
        GenerationCache(path, _identity())
    path.write_text(json.dumps({"cache_version": CACHE_VERSION,
                                "identity": _identity().as_dict(), "entries": {"p": 1}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        GenerationCache(path, _identity())


def test_cache_miss_raises_on_the_locked_read_path(tmp_path):
    cache = GenerationCache(tmp_path / "cache.json", _identity())
    cache.put("p", "resp")
    assert cache.get(["p", "p"]) == ["resp", "resp"]
    # A miss is an error: the locked path must never fall back to a fresh sample.
    with pytest.raises(KeyError):
        cache.get(["q"])


def test_missing_cache_file_starts_empty(tmp_path):
    cache = GenerationCache(tmp_path / "does-not-exist.json", _identity())
    assert len(cache) == 0 and "p" not in cache


# --- search advantage (regression preserved: cost applied exactly once) --------------

def test_search_advantage_requires_strictly_better_search():
    terminal = {"ANSWER": 0.0, "ABSTAIN": 0.0}
    # A search branch that only ties the terminal action is not an advantage.
    assert search_advantage({"SEARCH_BM25": 0.0}, terminal, search_cost=0.0)["positive"] is False
    assert search_advantage({"SEARCH_BM25": 0.5}, terminal, search_cost=0.05)["positive"] is True
    # Branch Q is already net of cost; do not subtract it a second time.
    assert search_advantage({"SEARCH_BM25": 0.03}, terminal, search_cost=0.05)["positive"] is True
    assert search_advantage({"SEARCH_BM25": -0.03}, terminal, search_cost=0.05)["positive"] is False


def test_search_advantage_prefers_terminal_on_ties_and_names_the_action():
    result = search_advantage({"SEARCH_DENSE": 0.4, "SEARCH_BM25": 0.4}, {"ANSWER": 0.4, "ABSTAIN": 0.0},
                              search_cost=0.0)
    assert result["optimal"] == "ANSWER"
    assert result["positive"] is False

"""VixSTORY pilot world builder: frozen manifest QIDs -> concrete episode worlds.

This is the VixSTORY counterpart of :mod:`src.pipeline.worlds`. One call turns the frozen
pilot inputs into the exact episode set a rollout may see:

==================  ==========================================================
split QIDs          the pilot manifest (``pilot_300`` / ``train_dev_200`` /
                    ``validation_246``), each re-checked against its QID hash
corpus              the pinned 3,462-passage ``corpus.parquet`` (``context_id`` -> text)
question + labels   the pinned ``train`` / ``validation`` rows; the official test split
                    (``data/test.parquet``) is refused, never read
rankings            ``pilot_rankings_top{N}.json`` (``{qid: {bm25, dense}}``)
reader answers      the identity-bound :class:`~scripts.vixstory_reader_eval.GenerationCache`
==================  ==========================================================

Four rules this module exists to enforce:

* **The reader is never invented.** Every reachable state's prompt is built with
  :func:`scripts.vixstory_reader_eval.reader_prompt` over *visible* passages only and read
  from the frozen :class:`GenerationCache`; a miss raises. Generation is confined to the
  warm-up pass, never here and never inside the environment.
* **Fail, don't drop.** A missing ranking, a missing cached prompt, or a prompt longer than
  ``PILOT_MAX_PROMPT_CHARS`` raises. Evaluation QIDs are never silently excluded.
* **Long documents are a blocker, not a truncation.** Deterministic truncation consistent
  across states is deliberately *not* implemented: shortening a visible passage would change
  the evidence the reader scored. :func:`prompt_budget_blockers` reports the offending states
  and :func:`build_reader_cache` refuses to proceed.
* **One validated cache identity.** The generation cache is bound by
  :func:`scripts.vixstory_reader_eval.build_identity` to the cloud reader configuration,
  prompt contract, corpus and rankings. :class:`GenerationCache` refuses a different identity;
  the environment's reader keys are derived from that validated identity. A provider-managed
  model alias does not guarantee immutable remote weights.

The worlds use the isolated :class:`src.rl.vixstory_env.VixStoryWorld`; the gold answer, gold
context ids and hop level live only in ``private`` and never reach an observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.vixstory_reader_eval import (
    PILOT_MAX_PROMPT_CHARS,
    CacheIdentity,
    GenerationCache,
    build_identity,
    parse_reader_output,
    reader_prompt,
)
from src.retrieval.states import reachable_states
from src.rl.vixstory_env import (
    VixStoryEnvConfig,
    VixStoryPrivate,
    VixStoryReaderCache,
    VixStoryReaderPrediction,
    VixStoryWorld,
)
from src.utils.hashing import sha256_json
from src.utils.io import read_json

#: Retrieval tools a world's rankings must provide; both are action choices in the MDP.
#: Mirrors :data:`src.retrieval.states.TOOLS` and the reader-eval stage.
TOOLS = ("bm25", "dense")

#: Manifest split record -> pilot role. ``train`` is the 300-question tuning split,
#: ``dev`` the 200-question disjoint tuning split, ``validation`` the 246 scored once.
MANIFEST_SPLITS = {"train": "pilot_300", "dev": "train_dev_200", "validation": "validation_246"}
SPLIT_ROLES = tuple(MANIFEST_SPLITS)

#: The official test split is never opened; a QA-file path carrying this name is refused.
FORBIDDEN_QA_FILENAME = "test.parquet"


@dataclass(frozen=True)
class StatePrompt:
    """One reachable state's reader prompt, built from its visible passages only."""

    qid: str
    state: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class SplitWorlds:
    """Everything one pilot role contributes to a rollout."""

    role: str
    qids: tuple[str, ...]
    worlds: tuple[VixStoryWorld, ...]
    n_states: int
    identity: CacheIdentity


def reader_cache_fingerprints(identity: CacheIdentity) -> dict[str, str]:
    """Derive the environment's three reader keys from the validated cloud cache identity."""
    return {
        "reader_sha": f"{identity.base_url}/{identity.model}@{identity.model_revision}",
        "tokenizer_sha": identity.message_version,
        "inference_config_sha": (f"{identity.prompt_version}:{identity.decoding}:"
                                 f"{identity.max_new_tokens}"),
    }


# ------------------------------------------------------------------------------- loaders

def manifest_split_qids(manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    """Map the frozen pilot manifest's split records to ``{role: sorted QIDs}``.

    Each record is re-checked against its own count, sort order and ``qids_sha256`` so a
    tampered or truncated manifest cannot silently change the pilot's question set.
    """
    splits = manifest.get("splits") if isinstance(manifest, Mapping) else None
    if not isinstance(splits, Mapping):
        raise ValueError("manifest carries no 'splits' mapping")
    out: dict[str, tuple[str, ...]] = {}
    for role, name in MANIFEST_SPLITS.items():
        record = splits.get(name)
        if role == "validation" and record is None:
            continue
        if not isinstance(record, Mapping):
            raise ValueError(f"manifest has no {name!r} split (role {role!r})")
        qids = record.get("qids")
        if not isinstance(qids, list) or not all(isinstance(q, str) for q in qids):
            raise ValueError(f"manifest split {name!r} qids must be a list of strings")
        if record.get("n") != len(qids) or list(qids) != sorted(qids):
            raise ValueError(
                f"manifest split {name!r} is not a sorted list of {record.get('n')} QIDs")
        if record.get("qids_sha256") != sha256_json(qids):
            raise ValueError(f"manifest split {name!r} QID hash does not match its QIDs")
        out[role] = tuple(qids)
    return out


def _reject_official_test(path: str | Path) -> None:
    if Path(path).name == FORBIDDEN_QA_FILENAME:
        raise ValueError(
            f"the official VixSTORY test split ({FORBIDDEN_QA_FILENAME}) is never read")


def read_corpus(path: str | Path) -> dict[str, str]:
    """``context_id -> text`` for the pinned shared corpus."""
    import pyarrow.parquet as pq

    corpus: dict[str, str] = {}
    for row in pq.read_table(path, columns=["context_id", "text"]).to_pylist():
        context_id = row["context_id"]
        if context_id in corpus:
            raise ValueError(f"{path}: duplicate corpus context_id {context_id!r}")
        corpus[context_id] = row["text"]
    return corpus


def read_qa_rows(path: str | Path) -> dict[str, Mapping[str, Any]]:
    """``qid -> row`` for a pinned train/validation split; the official test file is refused."""
    _reject_official_test(path)
    import pyarrow.parquet as pq

    columns = ["id", "question", "answer", "context_ids", "hop_level", "question_type"]
    rows: dict[str, Mapping[str, Any]] = {}
    for row in pq.read_table(path, columns=columns).to_pylist():
        qid = row["id"]
        if qid in rows:
            raise ValueError(f"{path}: duplicate QA id {qid!r}")
        rows[qid] = row
    return rows


def read_rankings(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """``{qid: {"bm25": [...], "dense": [...]}}`` from the frozen pilot rankings JSON."""
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: rankings must be a mapping of qid -> tool lists")
    out: dict[str, dict[str, list[str]]] = {}
    for qid, entry in data.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: entry {qid!r} is not a mapping")
        for tool in TOOLS:
            ranking = entry.get(tool)
            if not isinstance(ranking, list) or not all(isinstance(d, str) for d in ranking):
                raise ValueError(f"{path}: {qid!r} has no {tool!r} list of doc ids")
        out[str(qid)] = {"bm25": list(entry["bm25"]), "dense": list(entry["dense"])}
    return out


# --------------------------------------------------------------------------- reader cache

def _require(mapping: Mapping[str, Any], key: str, what: str, *, qid: str | None = None) -> Any:
    if key not in mapping:
        where = f"{qid!r}: " if qid is not None else ""
        raise KeyError(f"{where}no {what} for {key!r}")
    return mapping[key]


def _states_for(ranking: Mapping[str, Sequence[str]], config: VixStoryEnvConfig
                ) -> list[tuple[str, ...]]:
    return reachable_states(
        {"bm25": list(ranking["bm25"]), "dense": list(ranking["dense"])},
        bootstrap_tool=config.bootstrap_tool, bootstrap_passages=config.bootstrap_passages,
        passage_per_search_action=config.passage_per_search_action,
        max_optional_search=config.max_optional_search, tools=TOOLS)


def reader_state_prompts(*, qids: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
                         corpus: Mapping[str, str],
                         rankings: Mapping[str, Mapping[str, Sequence[str]]],
                         config: VixStoryEnvConfig | None = None) -> list[StatePrompt]:
    """The reader prompt of every reachable state, over visible passages only.

    Enumerates the tree with :func:`src.retrieval.states.reachable_states` under the *same*
    budget the environment walks, so the cache this feeds cannot miss a state a rollout visits.
    """
    config = config or VixStoryEnvConfig()
    prompts: list[StatePrompt] = []
    for qid in qids:
        row = _require(rows, qid, "question row")
        ranking = _require(rankings, qid, "ranking")
        for state in _states_for(ranking, config):
            passages = [{"context_id": doc_id,
                         "text": _require(corpus, doc_id, "corpus passage", qid=qid)}
                        for doc_id in state]
            prompts.append(StatePrompt(qid=qid, state=state,
                                       prompt=reader_prompt(row["question"], passages)))
    return prompts


def prompt_budget_blockers(state_prompts: Sequence[StatePrompt]) -> list[dict[str, Any]]:
    """Explicit blockers for the long-document path that is intentionally not truncated.

    Deterministic truncation consistent across states is *not* implemented: shortening a
    visible passage would change the evidence the reader was scored on. A state prompt longer
    than ``PILOT_MAX_PROMPT_CHARS`` is therefore a blocker the caller must resolve, never a row
    dropped. Returns one record per offending state.
    """
    return [{"qid": sp.qid, "n_visible_docs": len(sp.state), "prompt_chars": len(sp.prompt)}
            for sp in state_prompts if len(sp.prompt) > PILOT_MAX_PROMPT_CHARS]


def build_reader_cache(*, qids: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
                       corpus: Mapping[str, str],
                       rankings: Mapping[str, Mapping[str, Sequence[str]]],
                       cache: GenerationCache,
                       config: VixStoryEnvConfig | None = None) -> VixStoryReaderCache:
    """Read every reachable state's cached reader answer into a :class:`VixStoryReaderCache`.

    The generation cache is read-only here (``.get`` raises on a miss): the environment performs
    no inference, and a prompt that would need truncation, or that was never generated, is a
    hard error rather than a silently dropped or shortened state.

    The returned cache is keyed from ``cache.identity`` itself, so its fingerprints can only ever
    be the ones :class:`GenerationCache` validated on load.
    """
    state_prompts = reader_state_prompts(qids=qids, rows=rows, corpus=corpus, rankings=rankings,
                                         config=config)

    blockers = prompt_budget_blockers(state_prompts)
    if blockers:
        raise ValueError(
            f"{len(blockers)} state prompt(s) exceed PILOT_MAX_PROMPT_CHARS={PILOT_MAX_PROMPT_CHARS} and "
            "deterministic truncation is not implemented; refusing to shorten or drop evidence. "
            f"blockers={blockers}")

    missing = [sp for sp in state_prompts if sp.prompt not in cache]
    if missing:
        raise KeyError(
            f"{len(missing)} state prompt(s) absent from the frozen generation cache at "
            f"{cache.path}; refusing to generate during world assembly. "
            f"first missing: qid={missing[0].qid!r} state={list(missing[0].state)}")

    responses = cache.get([sp.prompt for sp in state_prompts])
    reader_cache = VixStoryReaderCache(**reader_cache_fingerprints(cache.identity))
    for sp, response in zip(state_prompts, responses):
        text, cited, valid = parse_reader_output(response, sp.state)
        reader_cache.put(sp.qid, sp.state, VixStoryReaderPrediction(
            qid=sp.qid, text=text, cited_ids=cited, citation_valid=valid))
    return reader_cache


# -------------------------------------------------------------------------------- worlds

def build_worlds(*, role: str, qids: Sequence[str], rows: Mapping[str, Mapping[str, Any]],
                 corpus: Mapping[str, str],
                 rankings: Mapping[str, Mapping[str, Sequence[str]]],
                 reader_cache: VixStoryReaderCache) -> list[VixStoryWorld]:
    """Assemble the role's :class:`VixStoryWorld` list from the shared corpus and the cache.

    Gold labels (free-form answer, multi gold context ids, hop level, question type) enter only
    the world's ``private`` side; the observation carries visible state alone.
    """
    worlds: list[VixStoryWorld] = []
    for qid in qids:
        row = _require(rows, qid, "question row")
        ranking = _require(rankings, qid, "ranking")
        gold_context_ids = tuple(str(doc_id) for doc_id in (row.get("context_ids") or ()))
        worlds.append(VixStoryWorld(
            qid=qid, question=str(row["question"]), split=role,
            scenario_id=f"vixstory:{role}:{qid}", corpus=corpus,
            rankings={"bm25": list(ranking["bm25"]), "dense": list(ranking["dense"])},
            private=VixStoryPrivate(
                qid=qid, answer_texts=(str(row["answer"]),), gold_context_ids=gold_context_ids,
                hop_level=int(row["hop_level"]), question_type=str(row["question_type"])),
            reader=reader_cache.reader(qid), reader_sha=reader_cache.reader_sha))
    return worlds


def assemble_split(*, role: str, qids: Sequence[str], corpus: Mapping[str, str],
                   rows: Mapping[str, Mapping[str, Any]],
                   rankings: Mapping[str, Mapping[str, Sequence[str]]],
                   cache: GenerationCache, config: VixStoryEnvConfig | None = None) -> SplitWorlds:
    """Assemble one pilot role's worlds from in-memory manifest QIDs, rows, rankings and cache."""
    if role not in MANIFEST_SPLITS:
        raise ValueError(f"unknown pilot role {role!r}; known: {list(SPLIT_ROLES)}")
    reader_cache = build_reader_cache(qids=qids, rows=rows, corpus=corpus, rankings=rankings,
                                      cache=cache, config=config)
    worlds = build_worlds(role=role, qids=qids, rows=rows, corpus=corpus, rankings=rankings,
                          reader_cache=reader_cache)
    return SplitWorlds(role=role, qids=tuple(qids), worlds=tuple(worlds),
                       n_states=len(reader_cache), identity=cache.identity)


def assemble_pilot_split(*, role: str, manifest_path: str | Path, corpus_path: str | Path,
                         qa_path: str | Path, rankings_path: str | Path, cache_path: str | Path,
                         config: VixStoryEnvConfig | None = None) -> SplitWorlds:
    """Read the frozen artifacts and assemble one role's worlds (``train``/``dev``/``validation``).

    Bind the cache to the cloud reader configuration and visible corpus/rankings. Opening
    :class:`GenerationCache` refuses generations from another reader configuration.
    """
    if role not in MANIFEST_SPLITS:
        raise ValueError(f"unknown pilot role {role!r}; known: {list(SPLIT_ROLES)}")
    splits = manifest_split_qids(read_json(manifest_path))
    if role not in splits:
        raise ValueError(f"manifest has no split for role {role!r}")
    qids = splits[role]
    corpus = read_corpus(corpus_path)
    rows = read_qa_rows(qa_path)
    rankings = read_rankings(rankings_path)
    identity = build_identity(corpus_path=corpus_path, rankings_path=rankings_path)
    cache = GenerationCache(cache_path, identity)
    return assemble_split(role=role, qids=qids, corpus=corpus, rows=rows, rankings=rankings,
                          cache=cache, config=config)

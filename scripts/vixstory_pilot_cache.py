"""Warm the cloud-reader generation cache for the VixSTORY 300/200/246 pilot.

This is the optional cloud warm-up pass that makes ``src.pipeline.vixstory_worlds`` and the rollout
environment possible: it generates one reader answer per **reachable ordered retrieval state**
of every pilot question, and nothing else. The locked evaluation path (world assembly, the
environment) only ever *reads* the cache — a miss there is a hard error — so every state a
rollout can visit must be present here, or the episode would be unrunnable.

What one run does (``--stage train|dev|validation|all``):

==================  ==========================================================
split QIDs          the frozen pilot manifest (``pilot_300`` / ``train_dev_200`` /
                    ``validation_246``), each re-checked against its QID hash
corpus + question   the pinned ``corpus.parquet`` and the pinned ``train`` /
                    ``validation`` rows; only the ``id`` and ``question`` columns are
                    loaded, so a reference answer or gold passage id is *unreadable*
rankings            ``pilot_rankings_top{N}.json`` over the full pinned corpus
states              ``src.retrieval.states.reachable_states`` under the *same* budget the
                    environment walks, so the cache cannot miss a state a rollout visits
reader prompts      built with ``scripts.vixstory_reader_eval.reader_prompt`` over visible
                    passages only; generation uses temperature=0 and is identity-bound
==================  ==========================================================

Four rules this script exists to enforce:

* **No silently dropped states.** Missing rankings, question rows, or over-budget prompts
  raise before generation. Individual exhausted cloud prompts remain absent and are reported
  as incomplete coverage; locked world assembly rejects their QIDs. Evidence is never shortened.
* **Use the validated cloud cache identity.** ``build_identity`` binds the reader,
  prompt contract, corpus and rankings; ``GenerationCache`` rejects a different identity.
  No additional fingerprint is invented here.
* **Generation never sees a label.** The corpus text of *visible* documents and the question
  text are the only inputs; the reference answer and gold passage ids are never loaded.
* **Progress survives a crash.** The cache is flushed to disk after every batch and every
  stage, and a re-run loads the existing entries and regenerates only the prompts still
  missing — so a long warm-up resumes instead of restarting. Coverage files are written to the
  next free versioned name; a historical measurement is never overwritten.

The official test split (``data/test.parquet``) is never read. No GPU work happens at import
or in the tests: the reader is injected and the tests use synthetic fixtures.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from scripts.vixstory_reader_eval import (
    CACHE_VERSION,
    CORPUS_PATH,
    MAX_ATTEMPTS_PER_PROMPT,
    PILOT_MAX_PROMPT_CHARS,
    SNAP,
    CloudReader,
    assert_prompts_fit,
    build_identity,
    reader_prompt,
)
from src.retrieval.states import reachable_states
from src.utils.hashing import sha256_file, sha256_json
from src.utils.io import atomic_write_json, read_json
from src.utils.runs import OutputNamespaceLock

# --- pilot namespace (new outputs only) -----------------------------------
RUN_ID = "vixstory-cloud-pilot-300-2026-10-11"
PILOT_DIR = Path("outputs") / "experiments" / RUN_ID
SOURCE_DIR = Path("outputs/experiments/vixstory-pilot-300-2026-10-11")
MANIFEST_PATH = SOURCE_DIR / "manifest.json"
RANKINGS_PATH = SOURCE_DIR / "pilot_rankings_top12.json"
#: Versioned reader cache for the pilot. It is disjoint from reader-eval's ``reader-cache-v2.json``
#: so a warm-up pass never rewrites another run's generations.
CACHE_NAME = "reader-cache-cloud-v3-max16384.json"
#: Coverable split roles -> their frozen-manifest record names.
MANIFEST_SPLITS = {"train": "pilot_300", "dev": "train_dev_200", "validation": "validation_246"}
STAGES = tuple(MANIFEST_SPLITS)
FORBIDDEN_QA_FILENAME = "test.parquet"
COVERAGE_NAME = "cache-coverage.json"
COVERAGE_ARTIFACT = "vixstory_pilot_reader_cache_coverage"
SCHEMA_VERSION = 1
DEFAULT_BATCH_SIZE = 64


@dataclass(frozen=True)
class EpisodeBudget:
    """The retrieval budget the environment walks; the same one enumerates the states here."""

    bootstrap_tool: str = "bm25"
    bootstrap_passages: int = 1
    passage_per_search_action: int = 1
    max_optional_search: int = 2

    def required_top_n(self) -> int:
        """Passages a tool's ranking must expose to reach the deepest state."""
        return self.bootstrap_passages + self.passage_per_search_action * self.max_optional_search


@dataclass(frozen=True)
class StatePrompt:
    """One reachable state's reader prompt, built from its visible passages only."""

    qid: str
    state: tuple[str, ...]
    prompt: str


# --- paths, manifest, inputs -----------------------------------------------

def next_free_path(path: str | Path) -> Path:
    """``path``, or the first ``<stem>.v<N><suffix>`` that does not exist. Never overwrites."""
    path = Path(path)
    if not path.exists():
        return path
    for version in range(2, 1000):
        candidate = path.with_name(f"{path.stem}.v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no free version of {path}")


def assert_pilot_namespace(path: str | Path, pilot_dir: str | Path = PILOT_DIR) -> None:
    """Refuse to write the reader cache outside the pilot run directory."""
    resolved = Path(path).resolve()
    root = Path(pilot_dir).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"{path} is outside the pilot namespace {pilot_dir}; the warm-up cache is written "
            "under the pilot run directory only")


def load_manifest_splits(path: str | Path = MANIFEST_PATH) -> dict[str, tuple[str, ...]]:
    """``{role: sorted QIDs}`` from the frozen manifest, each record re-checked against its hash.

    A tampered or truncated manifest raises instead of silently changing the pilot's question
    set, so no QID is ever dropped from a warm-up pass by a bad manifest.
    """
    manifest = read_json(path)
    splits = manifest.get("splits") if isinstance(manifest, Mapping) else None
    if not isinstance(splits, Mapping):
        raise ValueError(f"{path}: manifest carries no 'splits' mapping")
    out: dict[str, tuple[str, ...]] = {}
    for role, name in MANIFEST_SPLITS.items():
        record = splits.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: manifest has no {name!r} split (role {role!r})")
        qids = record.get("qids")
        if not isinstance(qids, list) or not all(isinstance(q, str) for q in qids):
            raise ValueError(f"{path}: split {name!r} qids must be a list of strings")
        if record.get("n") != len(qids) or list(qids) != sorted(qids):
            raise ValueError(f"{path}: split {name!r} is not a sorted list of {record.get('n')} QIDs")
        if record.get("qids_sha256") != sha256_json(qids):
            raise ValueError(f"{path}: split {name!r} QID hash does not match its QIDs")
        out[role] = tuple(qids)
    return out


def _reject_official_test(path: str | Path) -> None:
    if Path(path).name == FORBIDDEN_QA_FILENAME:
        raise ValueError(f"the official VixSTORY test split ({FORBIDDEN_QA_FILENAME}) is never read")


def read_corpus(path: str | Path = CORPUS_PATH) -> dict[str, str]:
    """``context_id -> text`` for the pinned shared corpus."""
    import pyarrow.parquet as pq

    corpus: dict[str, str] = {}
    for row in pq.read_table(path, columns=["context_id", "text"]).to_pylist():
        context_id = row["context_id"]
        if context_id in corpus:
            raise ValueError(f"{path}: duplicate corpus context_id {context_id!r}")
        corpus[context_id] = row["text"]
    return corpus


def read_questions(path: str | Path) -> dict[str, str]:
    """``qid -> question`` for a pinned train/validation split.

    Only the ``id`` and ``question`` columns are projected at read time, so a reference answer
    or gold passage id is not merely unused — it is never loaded, and generation cannot read it.
    The official test file is refused.
    """
    _reject_official_test(path)
    import pyarrow.parquet as pq

    questions: dict[str, str] = {}
    for row in pq.read_table(path, columns=["id", "question"]).to_pylist():
        qid = row["id"]
        if qid in questions:
            raise ValueError(f"{path}: duplicate QA id {qid!r}")
        questions[qid] = row["question"]
    return questions


def read_rankings(path: str | Path) -> dict[str, dict[str, list[str]]]:
    """``{qid: {"bm25": [...], "dense": [...]}}`` from the frozen pilot rankings JSON."""
    data = read_json(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: rankings must be a mapping of qid -> tool lists")
    out: dict[str, dict[str, list[str]]] = {}
    for qid, entry in data.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: entry {qid!r} is not a mapping")
        for tool in ("bm25", "dense"):
            ranking = entry.get(tool)
            if not isinstance(ranking, list) or not all(isinstance(d, str) for d in ranking):
                raise ValueError(f"{path}: {qid!r} has no {tool!r} list of doc ids")
        out[str(qid)] = {"bm25": list(entry["bm25"]), "dense": list(entry["dense"])}
    return out


# --- state planning (pure, CPU) ---------------------------------------------

def states_for(ranking: Mapping[str, Sequence[str]], budget: EpisodeBudget) -> list[tuple[str, ...]]:
    """Every ordered reachable state of one question under ``budget``."""
    return reachable_states(
        {"bm25": list(ranking["bm25"]), "dense": list(ranking["dense"])},
        bootstrap_tool=budget.bootstrap_tool, bootstrap_passages=budget.bootstrap_passages,
        passage_per_search_action=budget.passage_per_search_action,
        max_optional_search=budget.max_optional_search, tools=("bm25", "dense"))


def _passages(corpus: Mapping[str, str], state: Sequence[str], *, qid: str) -> list[dict]:
    missing = [doc_id for doc_id in state if doc_id not in corpus]
    if missing:
        raise KeyError(f"{qid}: ranking references {len(missing)} doc(s) absent from the corpus, "
                       f"e.g. {missing[0]!r}")
    return [{"context_id": doc_id, "text": corpus[doc_id]} for doc_id in state]


def state_prompts(*, qids: Sequence[str], questions: Mapping[str, str],
                  rankings: Mapping[str, Mapping[str, Sequence[str]]],
                  corpus: Mapping[str, str], budget: EpisodeBudget) -> list[StatePrompt]:
    """The reader prompt of every reachable state, over visible passages only.

    Enumerates the tree with :func:`src.retrieval.states.reachable_states` under the *same*
    budget the environment walks. The question text and the visible passages are the only
    inputs: no reference answer, no gold passage id.
    """
    out: list[StatePrompt] = []
    for qid in qids:
        question = questions[qid]
        for state in states_for(rankings[qid], budget):
            out.append(StatePrompt(qid=qid, state=state,
                                   prompt=reader_prompt(question, _passages(corpus, state, qid=qid))))
    return out


def char_blockers(state_prompts: Sequence[StatePrompt],
                  max_prompt_chars: int = PILOT_MAX_PROMPT_CHARS) -> list[dict]:
    """Explicit blockers for the long-document path that is intentionally not truncated.

    Deterministic truncation consistent across states is *not* implemented: shortening a visible
    passage would change the evidence the reader is scored on. A state prompt longer than
    ``max_prompt_chars`` is therefore a blocker, never a state dropped from the cache.
    """
    return [{"qid": sp.qid, "n_visible_docs": len(sp.state), "prompt_chars": len(sp.prompt)}
            for sp in state_prompts if len(sp.prompt) > max_prompt_chars]


def prompt_token_lengths(reader: object, prompts: Sequence[str]) -> list[int] | None:
    """Token lengths of the rendered chat prompts, or ``None`` when the reader exposes no tokenizer.

    Renders the same chat template the reader will apply, so the measured length is the length
    the model context window actually has to hold.
    """
    tokenizer = getattr(reader, "tokenizer", None)
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None or not callable(tokenizer):
        return None
    rendered = [apply([{"role": "user", "content": prompt}], add_generation_prompt=True,
                      tokenize=False) for prompt in prompts]
    return [int(n) for n in tokenizer(rendered, add_special_tokens=False, return_length=True)["length"]]


def assert_within_budget(state_prompts: Sequence[StatePrompt], *,
                         max_prompt_chars: int = PILOT_MAX_PROMPT_CHARS, reader: object | None = None) -> None:
    """Fail fast — before any generation — when a state prompt is over the char or token budget.

    The character check is deterministic and needs no tokenizer; the token check runs only when
    the reader exposes a tokenizer and its context window. A failure here aborts the warm-up
    pass rather than silently dropping or shortening the offending state.
    """
    blockers = char_blockers(state_prompts, max_prompt_chars)
    if blockers:
        raise ValueError(
            f"{len(blockers)} state prompt(s) exceed MAX_PROMPT_CHARS={max_prompt_chars}; refusing "
            "to warm a cache that would drop or truncate a state. "
            f"first blocker: qid={blockers[0]['qid']!r} chars={blockers[0]['prompt_chars']}")
    if reader is None:
        return
    lengths = prompt_token_lengths(reader, [sp.prompt for sp in state_prompts])
    model_context = getattr(reader, "model_context", None)
    max_new_tokens = getattr(reader, "max_new_tokens", None)
    if lengths is not None and model_context and max_new_tokens:
        assert_prompts_fit(lengths, model_context=int(model_context),
                           max_new_tokens=int(max_new_tokens))


# --- generation (reader-backed, persisted incrementally) --------------------

def warm_prompts(reader: object, prompts: Sequence[str], *, batch_size: int = DEFAULT_BATCH_SIZE,
                 workers: int = 1, progress: Callable[[int, int], None] | None = None) -> dict:
    """Try missing prompts, skipping only recorded/exhausted failures.

    Successful answers and actual attempts remain in the identity-bound cache. A failed prompt
    is never replaced with a dummy answer; the caller reports incomplete coverage.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    unique = list(dict.fromkeys(prompts))
    before = len(reader.cache)
    missing = [p for p in unique if p not in reader.cache]
    skippable = {"cloud response exhausted output budget"}
    for prompt, failure in reader.cache.failures.items():
        if failure not in skippable:
            raise RuntimeError(f"cloud cache contains terminal failure ({failure}); stopping")
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        while True:
            eligible = [p for p in batch if p not in reader.cache and
                        p not in reader.cache.failures and
                        reader.cache.retries.get(p, 0) < MAX_ATTEMPTS_PER_PROMPT]
            if not eligible:
                break
            try:
                reader.generate(eligible, workers=workers)
            except RuntimeError as exc:
                blocked = [p for p in eligible if p not in reader.cache and
                           (p in reader.cache.failures or
                            reader.cache.retries.get(p, 0) >= MAX_ATTEMPTS_PER_PROMPT)]
                if not blocked or not str(exc).startswith("cloud reader request failed ("):
                    raise
                if any(reader.cache.failures.get(p) not in (None, *skippable)
                       for p in blocked):
                    raise
            reader.flush()
        if progress is not None:
            progress(min(start + len(batch), len(missing)), len(missing))
    absent = [p for p in unique if p not in reader.cache]
    blocked = [p for p in absent if p in reader.cache.failures or
               reader.cache.retries.get(p, 0) >= MAX_ATTEMPTS_PER_PROMPT]
    if len(blocked) != len(absent):
        raise RuntimeError("cache warm-up left prompts missing without a recorded failure or "
                           "exhausted attempt budget")
    return {"n_unique": len(unique), "n_generated": len(reader.cache) - before,
            "n_missing": len(absent), "n_failed": len(blocked)}


def warm_stage(*, role: str, qids: Sequence[str], questions: Mapping[str, str],
               corpus: Mapping[str, str], rankings: Mapping[str, Mapping[str, Sequence[str]]],
               reader: object, budget: EpisodeBudget,
               max_prompt_chars: int = PILOT_MAX_PROMPT_CHARS,
               batch_size: int = DEFAULT_BATCH_SIZE, workers: int = 1,
               progress: Callable[[int, int], None] | None = None) -> dict:
    """Warm every reachable state of one pilot role; return its coverage record.

    Every manifest QID must have a question row and a deep-enough ranking, and every planned
    missing generation remains explicitly recorded as incomplete coverage.
    """
    missing_rankings = [qid for qid in qids if qid not in rankings]
    if missing_rankings:
        raise KeyError(f"{role}: {len(missing_rankings)} manifest QID(s) have no ranking, e.g. "
                       f"{missing_rankings[0]!r}; refusing to skip them")
    missing_questions = [qid for qid in qids if qid not in questions]
    if missing_questions:
        raise KeyError(f"{role}: {len(missing_questions)} manifest QID(s) have no question row, "
                       f"e.g. {missing_questions[0]!r}; refusing to skip them")
    required = budget.required_top_n()
    short = [qid for qid in qids
             if min(len(rankings[qid]["bm25"]), len(rankings[qid]["dense"])) < required]
    if short:
        raise ValueError(f"{role}: {len(short)} ranking(s) shorter than the {required} passages "
                         f"the deepest state needs, e.g. {short[0]!r}")

    prompts = state_prompts(qids=qids, questions=questions, rankings=rankings, corpus=corpus,
                            budget=budget)
    if not prompts:
        raise ValueError(f"{role}: no reachable states planned")
    assert_within_budget(prompts, max_prompt_chars=max_prompt_chars, reader=reader)

    warm = warm_prompts(reader, [sp.prompt for sp in prompts], batch_size=batch_size,
                        workers=workers, progress=progress)
    absent = [sp for sp in prompts if sp.prompt not in reader.cache]
    per_qid = Counter(sp.qid for sp in prompts)
    states_per_qid = [per_qid[qid] for qid in qids]
    return {"role": role, "n_qids": len(qids), "n_states": len(prompts),
            "n_unique_prompts": warm["n_unique"], "n_generated": warm["n_generated"],
            "n_missing_prompts": warm["n_missing"], "n_failed_prompts": warm["n_failed"],
            "n_missing_states": len(absent),
            "qids_sha256": sha256_json(sorted(qids)),
            "states_per_qid_mean": round(sum(states_per_qid) / len(states_per_qid), 2),
            "states_per_qid_max": max(states_per_qid),
            "states_verified": not absent}


# --- coverage manifest ------------------------------------------------------

def build_coverage(*, stages: Sequence[str], per_split: Mapping[str, Mapping],
                   budget: EpisodeBudget, reader: object, inputs: Mapping[str, object],
                   max_prompt_chars: int, batch_size: int) -> dict:
    """Assemble the final coverage manifest: counts, hashes and the all-states-verified proof."""
    split_block = {role: {key: per_split[role][key] for key in (
        "n_qids", "n_states", "n_unique_prompts", "n_generated", "n_missing_prompts",
        "n_failed_prompts", "n_missing_states", "qids_sha256", "states_per_qid_mean",
        "states_per_qid_max", "states_verified")} for role in stages}
    return {
        "artifact": COVERAGE_ARTIFACT, "schema_version": SCHEMA_VERSION, "run_id": RUN_ID,
        "stages": list(stages),
        "purpose": "Warm the frozen-reader generation cache over every reachable ordered "
                   "retrieval state of the pilot splits, so world assembly and rollouts never "
                   "miss a state. Generation reads no answer and no gold passage id.",
        "inputs": {**inputs, "official_test_loaded": False},
        "budget": asdict(budget),
        "config": {"max_prompt_chars": max_prompt_chars, "batch_size": batch_size,
                   "max_new_tokens": getattr(reader, "max_new_tokens", None)},
        "cache": {"path": str(reader.cache.path), "cache_version": CACHE_VERSION,
                  "entries": len(reader.cache), "identity": reader.cache.identity.as_dict()},
        "splits": split_block,
        "totals": {"n_qids": sum(r["n_qids"] for r in per_split.values()),
                   "n_states": sum(r["n_states"] for r in per_split.values()),
                   "n_missing_states": sum(r["n_missing_states"] for r in per_split.values()),
                   "generated_this_run": sum(r["n_generated"] for r in per_split.values())},
        "all_states_verified": all(r["states_verified"] for r in per_split.values()),
    }


def run(*, stages: Sequence[str], manifest: Mapping[str, Sequence[str]],
        corpus: Mapping[str, str], rankings: Mapping[str, Mapping[str, Sequence[str]]],
        train_questions: Mapping[str, str], validation_questions: Mapping[str, str],
        reader: object, budget: EpisodeBudget, out_dir: str | Path,
        inputs: Mapping[str, object], max_prompt_chars: int = PILOT_MAX_PROMPT_CHARS,
        batch_size: int = DEFAULT_BATCH_SIZE, workers: int = 1,
        progress: Callable[[int, int], None] | None = None) -> dict:
    """Warm every requested split, write the coverage manifest, and return it.

    Each requested split is a stage: its cache is flushed when it finishes, so a crash mid-run
    keeps everything generated so far. The coverage manifest is written to the next free
    versioned name, so a re-run never clobbers a historical one.
    """
    unknown = [stage for stage in stages if stage not in MANIFEST_SPLITS]
    if unknown:
        raise ValueError(f"unknown stage(s) {unknown}; known: {list(STAGES)}")
    questions_by_role = {"train": train_questions, "dev": train_questions,
                         "validation": validation_questions}
    per_split: dict[str, dict] = {}
    for role in stages:
        per_split[role] = warm_stage(role=role, qids=manifest[role], questions=questions_by_role[role],
                                     corpus=corpus, rankings=rankings, reader=reader, budget=budget,
                                     max_prompt_chars=max_prompt_chars, batch_size=batch_size,
                                     workers=workers, progress=progress)
        reader.flush()  # stage-end flush: the stage's states survive a later-stage crash

    coverage = build_coverage(stages=stages, per_split=per_split, budget=budget, reader=reader,
                              inputs=inputs, max_prompt_chars=max_prompt_chars,
                              batch_size=batch_size)
    target = next_free_path(Path(out_dir) / COVERAGE_NAME)
    atomic_write_json(target, coverage)
    coverage["written"] = str(target)
    return coverage


# --- CLI --------------------------------------------------------------------

def _pilot_inputs(*, manifest_path: Path, rankings_path: Path, corpus_path: Path,
                  train_path: Path | None, validation_path: Path | None,
                  identity) -> dict:
    paths = {"manifest": manifest_path, "train": train_path, "validation": validation_path}
    result = {name: {"path": str(path), "sha256": sha256_file(path)}
              for name, path in paths.items() if path is not None}
    result["rankings"] = {"path": str(rankings_path), "sha256": identity.rankings_sha256}
    result["corpus"] = {"path": str(corpus_path), "sha256": identity.corpus_sha256}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=(*STAGES, "all"), required=True,
                        help="pilot split to warm; validation requires explicit selection")
    parser.add_argument("--pilot-dir", type=Path, default=PILOT_DIR,
                        help="pilot run directory; the cache and coverage live inside it")
    parser.add_argument("--manifest", type=Path, default=None,
                        help=f"frozen source manifest (default: {MANIFEST_PATH})")
    parser.add_argument("--rankings", type=Path, default=None,
                        help=f"frozen source rankings (default: {RANKINGS_PATH})")
    parser.add_argument("--cache", type=Path, default=None,
                        help=f"identity-bound reader cache inside the pilot dir "
                             f"(default: <pilot-dir>/{CACHE_NAME})")
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--train", type=Path, default=SNAP / "data/train.parquet")
    parser.add_argument("--validation", type=Path, default=SNAP / "data/validation.parquet")
    parser.add_argument("--max-requests", type=int, default=50,
                        help="maximum cumulative cloud attempts in this cache across resumed runs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="prompts per progress batch")
    parser.add_argument("--workers", type=int, default=10,
                        help="concurrent cloud requests (1-10); one process owns the cache")
    parser.add_argument("--max-prompt-chars", type=int, default=PILOT_MAX_PROMPT_CHARS)
    parser.add_argument("--preflight-only", action="store_true",
                        help="count planned train/dev prompts without API calls or writing a cache")
    parser.add_argument("--bootstrap-tool", default=EpisodeBudget.bootstrap_tool)
    parser.add_argument("--bootstrap-passages", type=int, default=EpisodeBudget.bootstrap_passages)
    parser.add_argument("--passage-per-search-action", type=int,
                        default=EpisodeBudget.passage_per_search_action)
    parser.add_argument("--max-optional-search", type=int, default=EpisodeBudget.max_optional_search)
    args = parser.parse_args()

    pilot_dir = args.pilot_dir
    cache_path = args.cache or (pilot_dir / CACHE_NAME)
    assert_pilot_namespace(cache_path, pilot_dir)  # the warm-up cache is pilot-scoped only
    if not 1 <= args.workers <= 10:
        parser.error("--workers must be between 1 and 10")
    if args.preflight_only:
        return _warm_cli(args, pilot_dir, cache_path)
    with OutputNamespaceLock(cache_path.with_name(cache_path.name + ".lock")):
        with OutputNamespaceLock(pilot_dir / ".reader-cache-warm.lock"):
            return _warm_cli(args, pilot_dir, cache_path)


def _warm_cli(args, pilot_dir: Path, cache_path: Path) -> int:
    manifest_path = args.manifest or MANIFEST_PATH
    rankings_path = args.rankings or RANKINGS_PATH

    stages = STAGES if args.stage == "all" else (args.stage,)
    manifest = load_manifest_splits(manifest_path)
    corpus = read_corpus(args.corpus)
    rankings = read_rankings(rankings_path)
    train_path = args.train if any(stage in stages for stage in ("train", "dev")) else None
    validation_path = args.validation if "validation" in stages else None
    train_questions = read_questions(train_path) if train_path is not None else {}
    validation_questions = read_questions(validation_path) if validation_path is not None else {}
    budget = EpisodeBudget(bootstrap_tool=args.bootstrap_tool,
                           bootstrap_passages=args.bootstrap_passages,
                           passage_per_search_action=args.passage_per_search_action,
                           max_optional_search=args.max_optional_search)

    identity = build_identity(corpus_path=args.corpus, rankings_path=rankings_path)
    reader = CloudReader(cache_path, identity=identity, max_requests=args.max_requests)
    planned = {sp.prompt for role in stages for sp in state_prompts(
        qids=manifest[role],
        questions=train_questions if role != "validation" else validation_questions,
        rankings=rankings, corpus=corpus, budget=budget)}
    remaining = sum(prompt not in reader.cache for prompt in planned)
    eligible = sum(prompt not in reader.cache and prompt not in reader.cache.failures and
                   reader.cache.retries.get(prompt, 0) < MAX_ATTEMPTS_PER_PROMPT
                   for prompt in planned)
    attempts = getattr(reader.cache, "attempts", 0)
    print(f"cloud cache preflight: {len(planned)} unique prompts, {remaining} missing "
          f"({eligible} eligible), {attempts}/{args.max_requests} attempts used", flush=True)
    if attempts >= args.max_requests and eligible:
        raise RuntimeError("cloud cache request cap exhausted; no new attempts allowed")
    if args.preflight_only:
        return 0
    inputs = _pilot_inputs(manifest_path=manifest_path, rankings_path=rankings_path,
                           corpus_path=args.corpus, train_path=train_path,
                           validation_path=validation_path, identity=identity)

    def _progress(done: int, total: int) -> None:
        print(f"  generated {done}/{total} missing prompt(s); cache holds {len(reader.cache)}",
              flush=True)

    try:
        coverage = run(stages=stages, manifest=manifest, corpus=corpus, rankings=rankings,
                       train_questions=train_questions, validation_questions=validation_questions,
                       reader=reader, budget=budget, out_dir=pilot_dir, inputs=inputs,
                       max_prompt_chars=args.max_prompt_chars, batch_size=args.batch_size,
                       workers=args.workers, progress=_progress)
    except KeyboardInterrupt:
        reader.flush()
        print("interrupted; the generation cache was flushed and will resume on re-run", flush=True)
        raise
    print(json.dumps({k: v for k, v in coverage.items() if k != "inputs"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical episode-world builder shared by the P5 teacher states and the P6 RL stage.

One call assembles, for a strict split, exactly what a rollout may see:

==================  ==========================================================
scenarios           ``data/scenarios/scenarios_{short}.jsonl`` (P1)
rankings + scores   ``data/cache/retrieval/{split}_rankings_v1.jsonl`` (P2)
question text       ``questions_{split}.parquet``, public column only
documents           ``docs_*.parquet`` merged over the strict splits
reader candidate    the exhaustive P3 state cache, keyed like the teacher
Laya prior          the frozen P5 checkpoint, pre-warmed into a probability cache
==================  ==========================================================

Three rules this module exists to enforce:

* **One definition of the episode.** Teacher, baselines and RL must build their environments
  from these worlds with :func:`env_config`, so masks, horizon, bootstrap and retrieval
  budget cannot diverge between the arms of a comparison.
* **The reader is never invented.** ``EpisodeWorld.reader`` answers from the P3 state cache
  under :func:`src.qa.state_cache.state_cache_key` and raises on a miss; a substituted
  prediction would let the agent answer on evidence it never retrieved.
* **Decisions stay online.** The Laya prior is pre-warmed for *every* reachable state before
  any rollout and is then a pure in-memory lookup (:func:`prior_lookup`) — the model runs
  while pre-warming, never inside a training step (`05_...md` §8, `09_...md` §3).

State serialisation and keying for Laya follow :mod:`src.decision.teacher` exactly:
:func:`src.decision.state_text.serialize_state` -> :func:`src.decision.state_text.state_hash`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.data import artifacts
from src.decision.laya_adapter import LayaAdapter
from src.decision.laya_cache import LayaProbCache
from src.decision.state_text import serialize_state, state_hash
from src.pipeline import build_retrieval
from src.pipeline import cache_reader
from src.pipeline.cache_reader import state_cache_paths
from src.qa.reader import ReaderCache, ReaderPrediction
from src.qa.state_cache import load_state_cache, load_state_cache_meta, state_cache_key
from src.retrieval.states import reachable_states, retrieved_doc_ids
from src.rl.env import SEARCH_BM25, SEARCH_DENSE, AdaptiveQAEnv, EnvConfig, EpisodeWorld
from src.rl.reward import EpisodePrivate, RewardCalculator, RewardConfig
from src.utils.config import resolve
from src.utils.io import read_json, read_jsonl

#: Retrieval tools a world's rankings must provide; both are action choices in the MDP.
TOOLS = ("bm25", "dense")

#: Laya checkpoints the P5 stage writes, by regime name (spec 04 §5).
LAYA_REGIMES = ("ce", "rlcd")
LAYA_CHECKPOINT_DIRS = {"ce": "laya_ce", "rlcd": "laya_rlcd"}
#: Default frozen prior for the mainline agent: `laya_rlcd` (`05_...md` §1).
DEFAULT_LAYA_REGIME = "rlcd"
#: ``cfg["laya"]["frozen"]`` is the config's way of saying "use the config default".
DEFAULT_LAYA_SENTINEL = "__default__"

#: ``get_or_compute`` batches misses internally; this only bounds peak memory per call.
PREWARM_BATCH = 256

#: P1 export of the automated negative audit; the label-store column is the fallback.
AMBIGUOUS_FILE = "ambiguous_negatives.json"
AMBIGUOUS_COLUMN = "ambiguous_negative"

#: The Laya probability cache lives next to the other model artifacts, one subdirectory per
#: regime (``laya_probs_cache/ce`` / ``laya_probs_cache/rlcd``): stored rows are stamped with
#: the bound checkpoint's hash, so a shared directory would let a P7 run of one regime
#: invalidate the other regime's frozen cache (and the artifact P5 registered).
LAYA_CACHE_DIR = "laya_probs_cache"


@dataclass(frozen=True)
class WorldBundle:
    """Everything one strict split contributes to a rollout."""

    split: str
    worlds: list[EpisodeWorld]
    questions: dict[str, str]
    docs: dict[str, str]
    laya_fn: Callable[[dict], np.ndarray | None] | None
    ambiguous_qids: frozenset[str]


# ------------------------------------------------------------------------------- config

def resolve_laya_regime(cfg: Mapping[str, Any], regime: str | None = DEFAULT_LAYA_SENTINEL
                        ) -> str | None:
    """Map a regime name to ``"ce" | "rlcd" | None`` (``None`` = the A3 ablation, no prior)."""
    if regime is None or regime == "none":
        return None
    if regime == DEFAULT_LAYA_SENTINEL:
        frozen = (cfg.get("laya") or {}).get("frozen", DEFAULT_LAYA_REGIME)
        return None if frozen in (None, "none") else str(frozen)
    if regime not in LAYA_REGIMES:
        raise ValueError(f"unknown Laya regime {regime!r}; known: {list(LAYA_REGIMES)}")
    return regime


def laya_checkpoint_dir(cfg: Mapping[str, Any], regime: str) -> Path:
    """Directory the P5 stage wrote for ``regime`` (never created here)."""
    if regime not in LAYA_CHECKPOINT_DIRS:
        raise ValueError(f"unknown Laya regime {regime!r}; known: {list(LAYA_REGIMES)}")
    return resolve(cfg, "models") / LAYA_CHECKPOINT_DIRS[regime]


def laya_cache_dir(cfg: Mapping[str, Any], regime: str) -> Path:
    """The regime's probability-cache directory (rows are never shared across regimes)."""
    if regime not in LAYA_REGIMES:
        raise ValueError(f"unknown Laya regime {regime!r}; known: {list(LAYA_REGIMES)}")
    return resolve(cfg, "models") / LAYA_CACHE_DIR / regime


def env_config(cfg: Mapping[str, Any]) -> EnvConfig:
    """The one environment definition every stage must use (spec 05 §2 fairness).

    The optional-search budget comes from ``cfg["rl"]["max_optional_search"]`` (default 2,
    `05_...md` §6); the bootstrap and reveal sizes come from ``cfg["retrieval"]`` so the
    environment walks exactly the retrieval tree the P3 state cache was built over.
    """
    retrieval = cfg.get("retrieval") or {}
    rl = cfg.get("rl") or {}
    return EnvConfig(
        max_optional_search=int(rl.get("max_optional_search", 2)),
        bootstrap_tool=str(retrieval.get("initial_bootstrap", "bm25")),
        bootstrap_passages=int(retrieval.get("bootstrap_passages", 1)),
        passage_per_search_action=int(retrieval.get("passage_per_search_action", 1)),
        preview_unseen_scores=True,
        max_steps=16,
        strict_invalid_actions=True,
    )


def ambiguous_negatives_path(cfg: Mapping[str, Any]) -> Path:
    return resolve(cfg, "data_processed") / AMBIGUOUS_FILE


def load_ambiguous_qids(cfg: Mapping[str, Any], split: str) -> frozenset[str]:
    """Qids of ambiguous negatives for ``split`` — excluded from train *reward* only.

    Primary source is P1's ``ambiguous_negatives.json`` split export (also tolerant of
    a list of qids, audit rows, or a ``{"qids": [...]}`` wrapper). Until that file exists the
    label store's ``ambiguous_negative`` column is used, which is what the auditor writes
    today; with neither, nothing is flagged and the set is empty.
    """
    path = ambiguous_negatives_path(cfg)
    if path.is_file():
        return frozenset(_parse_ambiguous(read_json(path), split))
    rows = artifacts.read_questions(resolve(cfg, "data_processed"), split)
    return frozenset(str(row["qid"]) for row in rows if row.get(AMBIGUOUS_COLUMN))


def _parse_ambiguous(data: Any, split: str) -> list[str]:
    """Read the audit export defensively; a row's own verdict is never overridden."""
    if isinstance(data, dict):
        if "splits" in data:
            splits = data["splits"]
            if not isinstance(splits, dict) or split not in splits:
                raise ValueError(f"{AMBIGUOUS_FILE}: missing split {split!r}")
            row = splits[split]
            if not isinstance(row, dict) or not isinstance(row.get("flagged_qids"), list):
                raise ValueError(f"{AMBIGUOUS_FILE}: {split!r}.flagged_qids must be a list")
            data = row["flagged_qids"]
        if isinstance(data, dict):
            for key in ("qids", "ambiguous_qids", "ambiguous"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                data = [{"qid": qid, **(row if isinstance(row, dict) else {})}
                        for qid, row in data.items()]
    if not isinstance(data, list):
        raise ValueError(f"{AMBIGUOUS_FILE}: expected a list or a mapping, got {type(data).__name__}")
    qids: list[str] = []
    for item in data:
        if isinstance(item, str):
            qids.append(item)
            continue
        if not isinstance(item, dict) or not item.get("qid"):
            raise ValueError(f"{AMBIGUOUS_FILE}: unusable row {item!r}")
        if item.get("split") not in (None, split):
            continue
        flag = item.get("ambiguous_flag")
        if item.get("verdict") == "clean" or flag in (False, "false"):
            continue
        qids.append(str(item["qid"]))
    return qids


# -------------------------------------------------------------------------- Laya prior

def prior_lookup(probs: Mapping[tuple[str, str], np.ndarray]
                 ) -> Callable[[dict], np.ndarray]:
    """A pure in-memory lookup over pre-warmed ``(qid, state_hash)`` probabilities.

    A missing key raises: falling back to a live model call inside a training rollout would
    make the episode depend on when the cache was warmed, and silently returning a uniform
    prior would fabricate a decision input.
    """

    def lookup(obs: dict) -> np.ndarray:
        text = serialize_state(obs)
        key = (str(obs["qid"]), state_hash(text))
        hit = probs.get(key)
        if hit is None:
            raise KeyError(
                f"Laya prior for qid={key[0]!r} state_hash={key[1][:16]}… was never "
                "pre-warmed; build_worlds must warm every reachable state before a rollout")
        return hit

    return lookup


def merge_laya_fns(pairs: Sequence[tuple[Sequence[str], Callable[[dict], np.ndarray]]]
                   ) -> Callable[[dict], np.ndarray]:
    """Route an observation to the bundle that pre-warmed its qid.

    Train and validation worlds are warmed per split (strict splits have disjoint qids), so
    the trainer — which shares one prior across both environments — needs this dispatcher.
    """
    index: dict[str, Callable[[dict], np.ndarray]] = {}
    for qids, fn in pairs:
        for qid in qids:
            other = index.setdefault(qid, fn)
            if other is not fn:
                raise ValueError(f"qid {qid!r} is pre-warmed by two different bundles")

    def lookup(obs: dict) -> np.ndarray:
        fn = index.get(str(obs["qid"]))
        if fn is None:
            raise KeyError(f"no Laya prior was pre-warmed for qid {obs['qid']!r}")
        return fn(obs)

    return lookup


def reachable_state_requests(worlds: Sequence[EpisodeWorld], reward: RewardCalculator,
                             config: EnvConfig) -> list[tuple[str, str, str]]:
    """``(qid, state_hash, state_text)`` for every state a rollout can decide from.

    Action sequences are replayed through the real environment, so the enumeration cannot
    drift from what ``rollout`` visits (same argument as the teacher's state walk): the
    environment is deterministic, so one replay per reachable history covers the tree.
    """
    env = AdaptiveQAEnv(list(worlds), reward, config)
    requests: list[tuple[str, str, str]] = []
    for world in worlds:
        frontier: list[tuple[tuple[int, ...], dict]] = [((), _replay(env, world.qid, ())[0])]
        while frontier:
            history, obs = frontier.pop()
            text = serialize_state(obs)
            requests.append((world.qid, state_hash(text), text))
            mask = np.asarray(obs["action_mask"], dtype=bool)
            for action in (SEARCH_BM25, SEARCH_DENSE):
                if not mask[action]:
                    continue
                next_history = history + (action,)
                next_obs, done = _replay(env, world.qid, next_history)
                if not done:
                    frontier.append((next_history, next_obs))
    return requests


def _replay(env: AdaptiveQAEnv, qid: str, history: tuple[int, ...]
            ) -> tuple[dict, bool]:
    obs, _ = env.reset(seed=0, options={"qid": qid})
    terminated = truncated = False
    for action in history:
        obs, _, terminated, truncated, _ = env.step(action)
    return obs, bool(terminated or truncated)


def prewarm_prior(cfg: Mapping[str, Any], regime: str, checkpoint: str | Path,
                  requests: Sequence[tuple[str, str, str]], *,
                  cache_dir: str | Path | None = None) -> Callable[[dict], np.ndarray]:
    """Warm ``regime``'s Laya probability cache for ``requests`` and return the rollout lookup.

    The adapter is always loaded, even when every row is already cached: binding it is what
    invalidates stored rows when the checkpoint changed, and serving a stale probability would
    be a silent scientific error (see :mod:`src.decision.laya_cache`).
    """
    cache = LayaProbCache(cache_dir if cache_dir is not None else laya_cache_dir(cfg, regime))
    adapter = LayaAdapter.load(checkpoint)
    try:
        probs: dict[tuple[str, str], np.ndarray] = {}
        for start in range(0, len(requests), PREWARM_BATCH):
            probs.update(cache.get_or_compute(adapter, list(requests[start:start + PREWARM_BATCH])))
    finally:
        adapter.close()
    return prior_lookup(probs)


# ------------------------------------------------------------------------------ worlds

def build_worlds(cfg: Mapping[str, Any], split: str, *, limit: int | None = None,
                 laya_regime: str | None = DEFAULT_LAYA_SENTINEL) -> WorldBundle:
    """Assemble one strict split's worlds from the frozen P1/P2/P3 caches.

    ``limit`` caps the scenario list after sorting by ``scenario_id``, so a cached split and
    a limited rebuild select the same questions (the P3 cache was built the same way).
    """
    regime = resolve_laya_regime(cfg, laya_regime)
    checkpoint = None
    if regime is not None:
        checkpoint = laya_checkpoint_dir(cfg, regime)
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"Laya checkpoint for regime {regime!r} is missing: {checkpoint} — train it "
                "with the P5 stage, or use the no-prior ablation (laya_regime=None / "
                "--laya-regime none)")

    processed = resolve(cfg, "data_processed")
    docs = artifacts.read_docs_all(processed)
    questions = artifacts.question_texts(processed, split)
    question_rows = artifacts.questions_by_qid(artifacts.read_questions(processed, split))
    scenarios = sorted(artifacts.read_scenarios(resolve(cfg, "data_scenarios"), split),
                       key=lambda scenario: scenario.scenario_id)
    if limit is not None:
        scenarios = scenarios[:int(limit)]
    if not scenarios:
        raise ValueError(f"no {split} scenarios to build worlds from (limit={limit})")

    ranking_rows = _load_rankings(cfg, split)
    meta = load_state_cache_meta(state_cache_paths(cfg, split)[1])
    reader_cache = load_state_cache(state_cache_paths(cfg, split)[0])
    config = env_config(cfg)
    _check_state_cache_budget(meta, config)
    _check_state_cache_sources(cfg, split, meta)
    _check_state_cache_reader(cfg, meta)

    worlds: list[EpisodeWorld] = []
    for scenario in scenarios:
        row = ranking_rows.get(scenario.scenario_id)
        if row is None:
            raise KeyError(f"{scenario.scenario_id}: no cached ranking for {split}; "
                           "run build_retrieval first")
        if row["qid"] != scenario.qid:
            raise ValueError(f"{scenario.scenario_id}: ranking qid {row['qid']!r} != "
                             f"scenario qid {scenario.qid!r}")
        missing = [doc_id for doc_id in scenario.corpus_doc_ids if doc_id not in docs]
        if missing:
            raise KeyError(f"{scenario.scenario_id}: {len(missing)} corpus doc(s) absent from "
                           f"the processed documents, e.g. {missing[0]!r}")
        ranking = {tool: list(row["rankings"][tool]) for tool in TOOLS}
        question = questions.get(scenario.qid)
        if question is None:
            raise KeyError(f"{scenario.qid}: scenario has no question row in {split}")
        gold = question_rows.get(scenario.qid)
        if gold is None:
            raise KeyError(f"{scenario.qid}: scenario has no label row in {split}")
        corpus = {doc_id: docs[doc_id] for doc_id in scenario.corpus_doc_ids}
        world = EpisodeWorld(
            qid=scenario.qid, question=question, split=split,
            scenario_id=scenario.scenario_id, corpus=corpus, rankings=ranking,
            private=_private(scenario, gold, corpus, ranking, config),
            reader=_reader(reader_cache, meta, scenario.qid))
        # The environment reads per-tool score maps off the world when they exist and falls
        # back to a rank proxy otherwise; real BM25/E5 scores are what the feature names mean.
        scores = row.get("scores") or {}
        world._scores_bm25 = dict(scores.get("bm25", {}))
        world._scores_dense = dict(scores.get("dense", {}))
        worlds.append(world)

    laya_fn = None
    if regime is not None and checkpoint is not None:
        reward = RewardCalculator(RewardConfig.from_dict(cfg))
        laya_fn = prewarm_prior(cfg, regime, checkpoint,
                                reachable_state_requests(worlds, reward, config))
    return WorldBundle(split=split, worlds=worlds, questions=questions, docs=docs,
                       laya_fn=laya_fn, ambiguous_qids=load_ambiguous_qids(cfg, split))


def _load_rankings(cfg: Mapping[str, Any], split: str) -> dict[str, dict[str, Any]]:
    path = artifacts.require(build_retrieval.rankings_path(cfg, split),
                             "run the P2 retrieval step first (build_retrieval)")
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        scenario_id = row["scenario_id"]
        if scenario_id in rows:
            raise ValueError(f"{path}: duplicate scenario_id {scenario_id!r}")
        rows[scenario_id] = row
    return rows


def _check_state_cache_budget(meta: Mapping[str, Any], config: EnvConfig) -> None:
    """The episode tree must be a subset of the tree the P3 cache was built over."""
    budget = meta.get("retrieval_budget") or {}
    expected = {"bootstrap_tool": config.bootstrap_tool,
                "bootstrap_passages": config.bootstrap_passages,
                "passage_per_search_action": config.passage_per_search_action}
    drifted = {key: (budget.get(key), value) for key, value in expected.items()
               if budget.get(key) != value}
    if drifted or int(budget.get("optional_searches", 0)) < config.max_optional_search:
        raise ValueError(
            "the P3 state cache was built with a different retrieval budget "
            f"({budget}), the environment would run {expected} with "
            f"max_optional_search={config.max_optional_search}; rebuild the cache "
            "(cache_reader) instead of rolling out states it does not cover")


def _check_state_cache_sources(cfg: Mapping[str, Any], split: str,
                               meta: Mapping[str, Any]) -> None:
    """The cache is only valid for the exact inputs it recorded (D011).

    The cache records the sha256 of the rankings/docs/questions/scenarios it was built from;
    a file that changed since (e.g. rankings rebuilt after the D011 fix) makes every cached
    state stale, and rolling out a stale state would show the agent evidence from the wrong
    corpus. Fail fast instead.
    """
    recorded = meta.get("sources") or {}
    current = cache_reader.source_checksums(cfg, split)
    drifted = [name for name in sorted(current)
               if (recorded.get(name) or {}).get("sha256") != current[name]["sha256"]]
    if drifted:
        raise ValueError(
            f"the P3 state cache for {split} is stale: {', '.join(drifted)} changed since "
            "it was built; rerun cache_reader (P3) before building worlds")


def _check_state_cache_reader(cfg: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    """The cache bakes reader outputs, threshold included; the reader contract must match.

    ``train_reader --reselect`` (D010) can rewrite ``null_score_diff_threshold`` without
    touching the weights, and the state cache would keep serving candidate decisions made
    with the old threshold while the contract reports the new one. Fail fast instead: the
    threshold-dependent states are rebuilt by cache_reader (P3).
    """
    _, contract = cache_reader.read_contract(cfg)
    fields = ("reader_model_hash", "tokenizer_sha", "inference_config_sha",
              "null_score_diff_threshold")
    drifted = [field for field in fields if meta.get(field) != contract.get(field)]
    if drifted:
        raise ValueError(
            f"the P3 state cache was built with a different reader contract "
            f"(drifted: {', '.join(drifted)}); rerun cache_reader (P3) so the cached "
            "candidates match the frozen reader")


def _private(scenario: Any, gold_row: Mapping[str, Any],
             corpus: Mapping[str, str], ranking: Mapping[str, Sequence[str]],
             config: EnvConfig) -> EpisodePrivate:
    gold_doc_id = gold_row.get("gold_doc_id")
    reachable = None
    if gold_doc_id is not None:
        states = reachable_states(
            ranking, bootstrap_tool=config.bootstrap_tool,
            bootstrap_passages=config.bootstrap_passages,
            passage_per_search_action=config.passage_per_search_action,
            max_optional_search=config.max_optional_search)
        reachable = gold_doc_id in retrieved_doc_ids(states)
    return EpisodePrivate(
        qid=scenario.qid,
        # The display pair indexes the *normalized* text the reader and its offsets use.
        answer_texts=tuple(str(text) for text in (gold_row.get("answer_texts_display") or [])),
        is_impossible=bool(gold_row.get("is_impossible")),
        gold_doc_id=gold_doc_id,
        gold_in_corpus=bool(gold_doc_id is not None and gold_doc_id in corpus),
        gold_reachable=reachable)


def _reader(cache: ReaderCache, meta: Mapping[str, Any], qid: str
            ) -> Callable[[tuple[str, ...]], ReaderPrediction]:
    def read(visible_doc_ids: tuple[str, ...]) -> ReaderPrediction:
        key = state_cache_key(meta, qid, visible_doc_ids)
        try:
            return cache.lookup(key)
        except KeyError as exc:
            raise KeyError(
                f"reader state cache miss for {qid!r} with visible docs "
                f"{list(visible_doc_ids)} — the P3 cache does not cover this state; rebuild it "
                "(cache_reader) rather than substitute a prediction") from exc

    return read

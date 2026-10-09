"""P2 CLI: build + cache retrieval rankings, the dense index, and recall diagnostics.

    python -m src.pipeline.build_retrieval --profile smoke
    python -m src.pipeline.build_retrieval --profile smoke --cache-only

Interface with P1: consumed exclusively through :mod:`src.data.artifacts`, the supported
reader for the P1 files (``docs_{split}.parquet`` / ``questions_{split}.parquet`` under
``data/processed``, ``scenarios_{short}.jsonl`` under ``data/scenarios``). Only questions
are read from the label store; the private columns are ignored.

Outputs, all registered with SHA256 in ``outputs/runs/index.jsonl``:

* ``data/indexes/dense_<model>.npz``             — passage vectors (pickle-free npz)
* ``data/cache/retrieval/{split}_rankings_v1.jsonl``    — full BM25 + dense rankings
* ``data/cache/retrieval/{split}_reachability_v1.jsonl`` — gold rank + reachability
* ``reports/retrieval_report.json`` + two markdown tables (recall@k, BM25-vs-dense cases)

Reachability is a *private diagnostic*: it is written to its own file and only aggregates
ever reach a report. Nothing here reads an answer string, and rankings are keyed on the
question text alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.data import artifacts, scenarios as scenario_mod
from src.data.schemas import ScenarioRecord
from src.retrieval.dense import DenseBank, DenseEncoder
from src.retrieval.engine import ScenarioRanking, rank_scenario, scenario_corpus_sha
from src.retrieval.rankings import differing_examples, summarise_retrieval
from src.utils.config import add_profile_args, config_from_args, resolve, snapshot_config
from src.utils.hashing import sha256_json
from src.utils.io import atomic_write_json, read_jsonl, write_jsonl
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, now_iso

CACHE_VERSION = "v1"
SPLITS = artifacts.STRICT_SPLITS
DIFF_EXAMPLES = 20


# ---------------------------------------------------------------------------- inputs

def load_docs(cfg: dict, splits: Sequence[str] = SPLITS) -> dict[str, str]:
    """doc_id -> passage text across the requested strict splits. Documents carry no
    labels, so this map is safe to hold."""
    return artifacts.read_docs_all(resolve(cfg, "data_processed"), splits)


def load_questions(cfg: dict, split: str) -> dict[str, str]:
    """qid -> question text (``QuestionRecord.public_view`` only)."""
    return artifacts.question_texts(resolve(cfg, "data_processed"), split)


def load_scenarios(cfg: dict, split: str) -> list[ScenarioRecord]:
    return artifacts.read_scenarios(resolve(cfg, "data_scenarios"), split)


# ------------------------------------------------------------------------ config hash

def retrieval_config_sha(cfg: dict) -> str:
    """Hash of everything that can change a cached ranking."""
    ret = cfg["retrieval"]
    return sha256_json({
        "cache_version": CACHE_VERSION,
        "bm25": {k: ret["bm25"][k] for k in ("tokenize", "k1", "b", "top_n_cached")},
        "dense": {k: ret["dense"][k] for k in
                  ("model", "max_length", "normalize", "query_prefix", "passage_prefix")},
        "passage_per_search_action": ret["passage_per_search_action"],
    })


def reach_budgets(cfg: dict) -> dict[str, int]:
    """How many passages each tool can surface in one episode.

    BM25 supplies the forced bootstrap passage plus every optional search; dense only the
    optional searches. Reachability is reported per tool against these budgets so a
    bootstrap advantage cannot be mistaken for retrieval quality.
    """
    ret = cfg["retrieval"]
    searches = int(ret["optional_searches"]) * int(ret["passage_per_search_action"])
    bootstrap = int(ret["bootstrap_passages"])
    if ret["initial_bootstrap"] != "bm25":
        raise ValueError(f"initial_bootstrap={ret['initial_bootstrap']!r} not wired for reachability")
    return {"bm25": bootstrap + searches, "dense": searches}


# ----------------------------------------------------------------------------- index

def model_slug(model: str) -> str:
    return model.replace("/", "__")


def resolve_dense_revision(cfg: dict) -> tuple[str | None, str]:
    """Pin the encoder revision from the Hub; fall back to the local cache offline."""
    pinned = cfg["retrieval"]["dense"].get("revision")
    if pinned:
        return str(pinned), "config"
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(cfg["retrieval"]["dense"]["model"]).sha, "hub"
    except Exception as exc:  # offline is legitimate once the weights are cached
        print(f"WARNING: could not resolve the model revision from the Hub ({type(exc).__name__}); "
              f"proceeding from the local cache and recording revision_source=offline_cache")
        return None, "offline_cache"


def build_bank(cfg: dict, docs: Mapping[str, str], needed: Iterable[str]) -> tuple[DenseBank, dict]:
    ret = cfg["retrieval"]["dense"]
    revision, source = resolve_dense_revision(cfg)
    encoder = DenseEncoder(model_name=ret["model"], revision=revision,
                           batch_size=int(ret["batch_size"]), max_length=int(ret["max_length"]),
                           device=_device())
    try:
        ordered = sorted(set(needed))
        missing = [d for d in ordered if d not in docs]
        if missing:
            raise KeyError(f"{len(missing)} scenario doc(s) missing from the processed "
                           f"docs parquet files, e.g. {missing[:3]}")
        bank = DenseBank.build({d: docs[d] for d in ordered}, encoder)
    finally:
        encoder.unload()  # never leave the encoder resident next to the reader/Laya
    meta = {"model": ret["model"], "revision": revision, "revision_source": source,
            "dim": int(bank.vectors.shape[1]), "docs": len(bank.doc_ids), "rebuilt": True,
            "config_sha": retrieval_config_sha(cfg)}
    bank.save(str(bank_path(cfg)), meta=meta)
    return bank, meta


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def bank_path(cfg: dict) -> Path:
    path = resolve(cfg, "data_indexes") / f"dense_{model_slug(cfg['retrieval']['dense']['model'])}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_bank(cfg: dict, docs: Mapping[str, str], needed: Iterable[str],
                *, force: bool) -> tuple[DenseBank, dict]:
    """Reuse a bank that already covers ``needed`` under the same config, else rebuild."""
    needed = set(needed)
    path = bank_path(cfg)
    if path.exists() and not force:
        bank, meta = DenseBank.load(str(path))
        if meta.get("config_sha") == retrieval_config_sha(cfg) and needed <= set(bank.doc_ids):
            return bank, {**meta, "rebuilt": False}
    return build_bank(cfg, docs, needed)


# ------------------------------------------------------------------------ build/rank

def rankings_path(cfg: dict, split: str) -> Path:
    path = resolve(cfg, "data_cache") / "retrieval" / f"{split}_rankings_{CACHE_VERSION}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def reachability_path(cfg: dict, split: str) -> Path:
    path = resolve(cfg, "data_cache") / "retrieval" / f"{split}_reachability_{CACHE_VERSION}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _reusable_rows(path: Path, *, cfg: dict, force: bool) -> tuple[dict[str, dict], int]:
    """Cached rows keyed by scenario_id, plus the count of discarded superseded-version rows.

    A scenario id embeds ``SCENARIO_VERSION`` (``src.data.scenarios``), so a row from a
    superseded version can never be served again — the current scenario set only ever asks
    for current-version ids. Keeping such rows is dead weight and breaks qid-unique consumers
    (``cache_reader.load_rankings`` refuses a duplicate qid), so they are discarded here; the
    caller rewrites the file so the pruning reaches disk even when nothing needs re-ranking.
    """
    if force or not path.exists():
        return {}, 0
    stamp = retrieval_config_sha(cfg)
    suffix = f":v{scenario_mod.SCENARIO_VERSION}"
    kept, stale, superseded = {}, 0, 0
    for row in read_jsonl(path):
        if row.get("config_sha") != stamp:
            stale += 1
        elif not str(row.get("scenario_id", "")).endswith(suffix):
            superseded += 1
        else:
            kept[row["scenario_id"]] = row
    if stale:
        print(f"{path.name}: discarding {stale} row(s) from a different retrieval config")
    if superseded:
        print(f"{path.name}: discarding {superseded} row(s) from superseded scenario "
              f"versions (current v{scenario_mod.SCENARIO_VERSION})")
    return kept, superseded


def _reachability_row(scenario: ScenarioRecord, ranking: ScenarioRanking,
                      budgets: Mapping[str, int]) -> dict:
    gold = scenario.gold_doc_id

    def rank_of(tool: str) -> int | None:
        ranked = ranking.ranked(tool)
        return ranked.index(gold) + 1 if gold is not None and gold in ranked else None

    def reachable(tool: str, rank: int | None) -> bool | None:
        return None if gold is None else bool(rank is not None and rank <= budgets[tool])

    bm25_rank, dense_rank = rank_of("bm25"), rank_of("dense")
    return {
        "scenario_id": scenario.scenario_id, "qid": scenario.qid, "split": scenario.split,
        "answerable": gold is not None,
        "gold_rank_bm25": bm25_rank, "gold_rank_dense": dense_rank,
        "reachable_bm25": reachable("bm25", bm25_rank),
        "reachable_dense": reachable("dense", dense_rank),
        "budget_bm25": budgets["bm25"], "budget_dense": budgets["dense"],
    }


def build_split(cfg: dict, split: str, *, docs: Mapping[str, str], questions: Mapping[str, str],
                scenarios: Sequence[ScenarioRecord], bank: DenseBank | None, config_sha: str,
                force: bool, cache_only: bool) -> tuple[list[ScenarioRanking], list[dict], dict]:
    ret = cfg["retrieval"]
    top_n = int(ret["bm25"]["top_n_cached"])
    path = rankings_path(cfg, split)
    rows, superseded = _reusable_rows(path, cfg=cfg, force=force)
    # A row is reusable only for the corpus it was built over (D011): the same scenario_id
    # gets different distractors at different profile caps, so an id match alone can serve
    # a ranking of a different corpus. Older rows without the fingerprint are rebuilt.
    todo = [s for s in scenarios
            if rows.get(s.scenario_id, {}).get("corpus_sha")
            != scenario_corpus_sha(list(s.corpus_doc_ids))]
    if todo and cache_only:
        raise FileNotFoundError(
            f"--cache-only but {len(todo)} of {len(scenarios)} {split} scenarios have no cached "
            f"ranking in {path} (e.g. {todo[0].scenario_id}); run without --cache-only first")
    if todo:
        if bank is None:
            raise ValueError("dense bank required to build rankings; run without --cache-only")
        encoder = DenseEncoder(model_name=ret["dense"]["model"],
                               batch_size=int(ret["dense"]["batch_size"]),
                               max_length=int(ret["dense"]["max_length"]), device=_device())
        try:
            for scenario in todo:
                if scenario.qid not in questions:
                    raise KeyError(f"{scenario.qid} from {split} scenarios has no question row in "
                                   f"{artifacts.questions_path(resolve(cfg, 'data_processed'), split)}")
                ranking = rank_scenario(
                    scenario_id=scenario.scenario_id, qid=scenario.qid,
                    question=questions[scenario.qid],
                    docs=docs, corpus_doc_ids=list(scenario.corpus_doc_ids),
                    dense_bank=bank, dense_encoder=encoder, top_n=top_n,
                    bm25_k1=float(ret["bm25"]["k1"]), bm25_b=float(ret["bm25"]["b"]))
                row = ranking.to_row()
                row["config_sha"] = config_sha
                rows[scenario.scenario_id] = row
        finally:
            encoder.unload()
        # Keep rows for scenarios outside this profile: they stay valid under the same
        # config and let a larger profile reuse them.
        write_jsonl(path, [rows[key] for key in sorted(rows)])
    elif superseded:
        # Only dead rows were dropped; rewrite so the pruning reaches disk (no re-ranking).
        write_jsonl(path, [rows[key] for key in sorted(rows)])

    cached = {s.scenario_id: rows[s.scenario_id] for s in scenarios if s.scenario_id in rows}
    budgets = reach_budgets(cfg)
    reach_rows = [_reachability_row(s, ScenarioRanking.from_row(cached[s.scenario_id]), budgets)
                  for s in scenarios if s.scenario_id in cached]
    write_jsonl(reachability_path(cfg, split), reach_rows)
    return ([ScenarioRanking.from_row(cached[s.scenario_id]) for s in scenarios
             if s.scenario_id in cached],
            reach_rows,
            {"cache_hits": len(scenarios) - len(todo), "ranked": len(todo),
             "scenarios": len(scenarios)})


# ---------------------------------------------------------------------------- reports

def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out) + "\n"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return "nan" if value != value else f"{value:.{digits}f}"
    return str(value)


def write_report(cfg: dict, *, rankings: list[ScenarioRanking], reach_rows: list[dict],
                 gold: dict[str, str | None], meta: dict) -> dict:
    budgets = reach_budgets(cfg)
    report = summarise_retrieval(rankings, gold, reach_budget=budgets).to_dict()
    answerable = [r for r in reach_rows if r["answerable"]]
    unreachable = [r for r in answerable if not (r["reachable_bm25"] or r["reachable_dense"])]
    report.update({
        "config_sha": retrieval_config_sha(cfg),
        "reach_budget": budgets,
        "cache_version": CACHE_VERSION,
        "rankings_total": len(rankings),
        "gold_rank_bm25_histogram": _histogram([r["gold_rank_bm25"] for r in answerable]),
        "gold_rank_dense_histogram": _histogram([r["gold_rank_dense"] for r in answerable]),
        "unreachable_scenarios": len(unreachable),
        "unreachable_rate": (len(unreachable) / len(answerable)) if answerable else None,
        "unreachable_examples": [r["scenario_id"] for r in unreachable[:20]],
        "library_versions": capture_library_versions(),
        **meta,
    })

    tables = resolve(cfg, "tables")
    tables.mkdir(parents=True, exist_ok=True)
    recall_md = tables / "retrieval_recall_at_k.md"
    examples_md = tables / "bm25_vs_dense_examples.md"
    report_path = resolve(cfg, "reports") / "retrieval_report.json"

    recall_md.write_text(md_table(
        ["tool", "recall@1", "recall@3", "recall@5", "MRR", "reachable@budget"],
        [[tool, _fmt(report["recall_at_1"].get(tool)), _fmt(report["recall_at_3"].get(tool)),
          _fmt(report["recall_at_5"].get(tool)), _fmt(report["mrr"].get(tool)),
          _fmt(report["reachable_rate"].get(tool))] for tool in ("bm25", "dense")]
    ) + "\n" + md_table(
        ["metric", "value"],
        [["answerable scenarios", report["n_answerable"]],
         ["union recall@5", _fmt(report["union_recall_at_5"])],
         ["BM25/E5 top-3 overlap", _fmt(report["topk_overlap_bm25_dense"])],
         ["BM25/E5 difference rate", _fmt(report["diff_rate"])],
         ["unreachable rate", _fmt(report["unreachable_rate"])]]
    ), encoding="utf-8")

    examples = differing_examples(rankings, k=3, limit=DIFF_EXAMPLES)
    example_rows = [[e["qid"], (gold.get(e["scenario_id"]) or "—"),
                     ", ".join(e["top_bm25"]), ", ".join(e["top_dense"])] for e in examples]
    examples_md.write_text(
        f"# BM25 vs E5 — {len(examples)} cases where the top-3 differs\n\n"
        + md_table(["qid", "gold doc id (private)", "top-3 BM25", "top-3 E5"], example_rows),
        encoding="utf-8")

    atomic_write_json(report_path, report)
    return {"report": report_path, "recall_table": recall_md, "examples_table": examples_md}


def _histogram(values: Sequence[int | None]) -> dict[str, int]:
    """Bucket gold ranks into '1', '2-3', '4-5', '6-10', '>10', 'missing'."""
    buckets = {"1": 0, "2-3": 0, "4-5": 0, "6-10": 0, ">10": 0, "missing": 0}
    for value in values:
        if not value:
            buckets["missing"] += 1
        elif value == 1:
            buckets["1"] += 1
        elif value <= 3:
            buckets["2-3"] += 1
        elif value <= 5:
            buckets["4-5"] += 1
        elif value <= 10:
            buckets["6-10"] += 1
        else:
            buckets[">10"] += 1
    return buckets


# ------------------------------------------------------------------------------- run

def run(*, cfg: dict, profile: str, splits: Sequence[str] = SPLITS, force: bool = False,
        cache_only: bool = False, registry: RunRegistry | None = None) -> dict:
    run_id = f"p2-retrieval-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P2_retrieval", status="RUNNING", profile=profile,
                    splits=list(splits), force=force, cache_only=cache_only)
    try:
        docs = load_docs(cfg, splits)
        config_sha = retrieval_config_sha(cfg)
        scenarios_by_split = {s: load_scenarios(cfg, s) for s in splits}
        needed = {d for scenarios in scenarios_by_split.values() for s in scenarios
                  for d in s.corpus_doc_ids}
        bank, bank_meta = (None, {}) if cache_only else ensure_bank(cfg, docs, needed, force=force)

        rankings, reach_rows, per_split = [], [], {}
        for split in splits:
            split_rankings, split_reach, stats = build_split(
                cfg, split, docs=docs, questions=load_questions(cfg, split),
                scenarios=scenarios_by_split[split], bank=bank, config_sha=config_sha,
                force=force, cache_only=cache_only)
            rankings += split_rankings
            reach_rows += split_reach
            per_split[split] = stats

        gold = {s.scenario_id: s.gold_doc_id for scenarios in scenarios_by_split.values()
                for s in scenarios}
        meta = {"profile": profile, "splits": dict(per_split), "docs": len(docs),
                "dense_bank": {"path": str(bank_path(cfg)), "docs": len(bank.doc_ids),
                               "dim": int(bank.vectors.shape[1]), **bank_meta} if bank else None}
        paths = write_report(cfg, rankings=rankings, reach_rows=reach_rows, gold=gold, meta=meta)
    except Exception as exc:
        registry.append(run_id=run_id, stage="P2_retrieval", status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise

    artifacts = [rankings_path(cfg, s) for s in splits] + [reachability_path(cfg, s) for s in splits]
    if not cache_only:
        artifacts.append(bank_path(cfg))
    artifacts += [paths["report"], paths["recall_table"], paths["examples_table"]]
    registry.record_artifact(run_id, "P2_retrieval", artifacts, profile=profile,
                             config_sha=config_sha, splits=dict(per_split))
    return {"run_id": run_id, "config_sha": config_sha, "splits": per_split,
            "dense_bank": meta["dense_bank"],
            "report": json.loads(Path(paths["report"]).read_text(encoding="utf-8")),
            "artifacts": [str(a) for a in artifacts]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.build_retrieval",
                                     description="P2: build BM25 + E5 rankings and diagnostics")
    add_profile_args(parser)
    parser.add_argument("--splits", default=",".join(SPLITS), help="comma-separated strict splits")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    parser.add_argument("--cache-only", action="store_true",
                        help="consume cached rankings only; never load the encoder")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    snapshot_config(cfg, resolve(cfg, "manifests") / f"P2_retrieval_{profile}_config.yaml")
    summary = run(cfg=cfg, profile=profile, splits=splits, force=args.force,
                  cache_only=args.cache_only)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        report = summary["report"]
        print(f"run_id={summary['run_id']} config_sha={summary['config_sha']}")
        for split, stats in summary["splits"].items():
            print(f"  {split}: {stats['ranked']} ranked, {stats['cache_hits']} cached "
                  f"of {stats['scenarios']}")
        print(f"  union recall@5={_fmt(report['union_recall_at_5'])} "
              f"diff_rate={_fmt(report['diff_rate'])} "
              f"unreachable={_fmt(report['unreachable_rate'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

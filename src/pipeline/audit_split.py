"""P1 normalize + audit + strict split + scenario CLI.

    python -m src.pipeline.audit_split --profile smoke

Reads the pinned raw parquet (``data/raw/source_manifest.json``), normalizes every row,
builds group-disjoint strict splits **from the original ``train`` split only**, subsamples
the profile cap, builds one episodic scenario per question and writes:

* ``data/processed/questions_{train,val,test}_strict.parquet``  (label store — private)
* ``data/processed/docs_{train,val,test}_strict.parquet``       (content only)
* ``data/processed/qids_{split}_strict.txt``
* ``data/processed/split_manifest.json``
* ``data/processed/ambiguous_negatives.json``
* ``data/scenarios/scenarios_{train,val,test}.jsonl``           (``cfg['paths']['data_scenarios']``,
  the layout :mod:`src.data.artifacts` — the supported reader — expects)
* ``data/diagnostics/{schema_report.md,leakage_report.json,negative_audit.csv,
  exclusions.csv,doc_sha256.csv}``

Every number in those files is measured in this run — none is typed by hand.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.data import auditor, loader, scenarios as scenario_mod, splitter
from src.data.normalizer import NormalizedRow, RecordNormalizer
from src.data.schemas import DocumentRecord
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, resolve
from src.utils.hashing import sha256_file
from src.utils.io import atomic_write_json, write_jsonl
from src.utils.reproducibility import capture_library_versions
from src.utils.runs import RunRegistry, now_iso

OFFICIAL_SPLITS = ("train", "validation", "test")
#: title groups whose documents are this similar (TF-IDF cosine) are put in one component
#: and therefore land in the same strict split.
MERGE_NEAR_DUP_THRESHOLD = 0.90
#: reported (warning) near-duplicate threshold; pairs in [REPORT, MERGE) across strict
#: splits are reported with counts and examples instead of being silently ignored.
REPORT_NEAR_DUP_THRESHOLD = 0.85

QUESTIONS_SCHEMA = pa.schema([
    ("qid", pa.string()), ("source_split", pa.string()), ("strict_split", pa.string()),
    ("title_group", pa.string()), ("question", pa.string()),
    ("answer_texts", pa.list_(pa.string())), ("answer_starts_raw", pa.list_(pa.int64())),
    ("answer_starts_display", pa.list_(pa.int64())), ("answer_texts_display", pa.list_(pa.string())),
    ("is_impossible", pa.bool_()), ("plausible_texts", pa.list_(pa.string())),
    ("gold_doc_id", pa.string()), ("offset_issue", pa.string()), ("notes", pa.string()),
    ("ambiguous_negative", pa.bool_()),
])

DOCS_SCHEMA = pa.schema([
    ("doc_id", pa.string()), ("source_split", pa.string()), ("strict_split", pa.string()),
    ("title_group", pa.string()), ("text", pa.string()), ("text_raw", pa.string()),
])


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------

def _rel(path: str | Path) -> str:
    """Repo-relative path when possible (portable manifests), absolute otherwise."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_raw_split(path: str | Path, batch_size: int = 1024):
    """Stream one raw parquet split row by row (bounded memory)."""
    parquet = pq.ParquetFile(str(path))
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def normalize_raw_split(path: str | Path, split: str, *, keep_rows: bool):
    """Normalize a whole raw split -> (rows, stats, qids, failures)."""
    stats = auditor.SchemaStats(f"raw:{split}")
    normalizer = RecordNormalizer()
    rows: list[NormalizedRow] = []
    qids: set[str] = set()
    failures: list[dict[str, str]] = []
    for example in read_raw_split(path):
        try:
            row = normalizer.normalize(example, split)
        except (ValueError, KeyError, TypeError) as exc:
            reason = f"{type(exc).__name__}:{exc}"
            stats.add_malformed(reason)
            failures.append({"qid": f"{split}:{example.get('id', '<missing id>')}", "split": split,
                             "stage": "normalize", "reason": reason})
            continue
        stats.add(row)
        qids.add(row.question.qid)
        if keep_rows:
            rows.append(row)
    return rows, stats, qids, failures


def question_row(row: NormalizedRow, strict_split: str,
                 ambiguous_negative: bool = False) -> dict[str, Any]:
    """Label-bearing row for the label store parquet (never an observation).

    ``ambiguous_negative`` is the automated distractor-ambiguity flag (spec §6); it is
    always ``False`` for answerable rows and never true for a row that was not evaluated.
    """
    private = row.question.private
    return {
        "qid": row.question.qid,
        "source_split": row.question.source_split,
        "strict_split": strict_split,
        "title_group": row.question.title_group,
        "question": row.question.question,
        "answer_texts": list(private.answer_texts),
        "answer_starts_raw": list(private.answer_starts),
        "answer_starts_display": list(row.answer_starts_display),
        "answer_texts_display": list(row.answer_texts_display),
        "is_impossible": private.is_impossible,
        "plausible_texts": list(private.plausible_texts),
        "gold_doc_id": private.gold_doc_id,
        "offset_issue": private.offset_issue,
        "notes": ";".join(row.notes),
        "ambiguous_negative": False if not private.is_impossible else bool(ambiguous_negative),
    }


def doc_row(doc: DocumentRecord, text_raw: str, strict_split: str) -> dict[str, Any]:
    return {"doc_id": doc.doc_id, "source_split": doc.source_split,
            "strict_split": strict_split, "title_group": doc.title_group,
            "text": doc.text, "text_raw": text_raw}


def write_parquet(rows: Sequence[Mapping[str, Any]], path: Path, schema: pa.Schema) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(r) for r in rows], schema=schema)
    pq.write_table(table, str(path), compression="zstd")
    return {"rows": len(rows), "sha256": sha256_file(path), "bytes": path.stat().st_size}


# --------------------------------------------------------------------------------------
# main pipeline
# --------------------------------------------------------------------------------------

def run(
    *,
    cfg: Mapping[str, Any],
    profile: str,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    registry: RunRegistry | None = None,
) -> dict[str, Any]:
    data = cfg["data"]
    caps = data["caps"]
    if profile not in caps:
        raise ValueError(f"unknown profile {profile!r}; known profiles: {sorted(caps)}")
    cap = caps[profile]
    train_cap = data.get("train_cap", cap)
    if train_cap is not None and (isinstance(train_cap, bool) or not isinstance(train_cap, int)
                                  or train_cap < 1):
        raise ValueError("data.train_cap must be a positive integer or null")
    split_seed = int(data["split_seed"])
    scenario_seed = int(data["scenario_seed"])
    n_distractors = int(data["n_distractors"])
    hard_negative_ratio = float(data.get("hard_negative_ratio", 0.0))
    retrieval_bm25 = (cfg.get("retrieval") or {}).get("bm25") or {}
    bm25_k1 = float(retrieval_bm25.get("k1", 1.5))
    bm25_b = float(retrieval_bm25.get("b", 0.75))
    audit_size = int(data["negative_audit_size"])
    train_ratio = float(data["strict_train_ratio"])
    val_ratio = float(data["strict_val_ratio"])
    ratios = {"train": train_ratio, "val": val_ratio, "test": round(1.0 - train_ratio - val_ratio, 10)}

    raw_dir = Path(raw_dir if raw_dir is not None else resolve(cfg, "data_raw"))
    processed_dir = Path(processed_dir if processed_dir is not None else resolve(cfg, "data_processed"))
    diagnostics_dir = Path(diagnostics_dir if diagnostics_dir is not None
                           else resolve(cfg, "data_diagnostics"))
    scenarios_dir = resolve(cfg, "data_scenarios")
    for directory in (processed_dir, diagnostics_dir, scenarios_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_id = f"p1-audit-split-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P1_audit_split", status="RUNNING", profile=profile,
                    cap_per_split=cap, train_cap=train_cap, split_seed=split_seed,
                    scenario_seed=scenario_seed,
                    n_distractors=n_distractors, hard_negative_ratio=hard_negative_ratio)
    written: list[Path] = []
    try:
        source = loader.load_source_manifest(raw_dir)
        raw_paths = loader.raw_split_paths(raw_dir, source)
        missing = [name for name in OFFICIAL_SPLITS if name not in raw_paths]
        if missing:
            raise FileNotFoundError(
                f"raw split(s) {missing} missing — run `python -m src.pipeline.download` first")

        # ---- 1. normalize every official split; keep the raw train rows in memory ----
        raw_stats: dict[str, auditor.SchemaStats] = {}
        official_qids: dict[str, set[str]] = {}
        exclusions: list[dict[str, str]] = []
        train_rows: list[NormalizedRow] = []
        for name in OFFICIAL_SPLITS:
            rows, stats, qids, failures = normalize_raw_split(
                raw_paths[name], name, keep_rows=(name == "train"))
            raw_stats[name] = stats
            official_qids[name] = qids
            exclusions.extend(failures)
            if name == "train":
                train_rows = rows
        if not train_rows:
            raise RuntimeError("no usable rows in the raw train split")

        docs_map: dict[str, DocumentRecord] = {}
        docs_raw: dict[str, str] = {}
        for row in train_rows:
            docs_map.setdefault(row.document.doc_id, row.document)
            docs_raw.setdefault(row.document.doc_id, row.context_raw)

        # ---- 2. strict split by title_group, after merging near-duplicate groups ----
        order = sorted(docs_map)
        merge_sim = splitter.self_similarity(
            order, [docs_map[doc_id].text for doc_id in order], threshold=MERGE_NEAR_DUP_THRESHOLD)
        group_of = {doc_id: docs_map[doc_id].title_group for doc_id in order}
        group_pairs = {(group_of[a], group_of[b]) for a, b, _ in merge_sim["all"]}
        groups_by_doc: dict[str, set[str]] = defaultdict(set)
        for row in train_rows:
            groups_by_doc[row.document.doc_id].add(row.question.title_group)
        for names in groups_by_doc.values():
            ordered = sorted(names)
            group_pairs.update(zip(ordered, ordered[1:]))
        group_merge = splitter.merge_groups(group_pairs, groups=set(group_of.values()))
        component_of = dict(group_merge)

        component_stats: dict[str, splitter.GroupStat] = defaultdict(splitter.GroupStat)
        for row in train_rows:
            component = component_of[row.question.title_group]
            component_stats[component] = component_stats[component] + splitter.GroupStat(
                1, 1 if row.question.private.is_impossible else 0)
        component_assignment = splitter.allocate_groups(
            component_stats, ratios=ratios, seed_salt=split_seed)
        group_assignment = {group: component_assignment[comp]
                            for group, comp in component_of.items()}

        strict_rows: dict[str, list[NormalizedRow]] = {name: [] for name in splitter.STRICT_SPLITS}
        for row in train_rows:
            strict_rows[group_assignment[row.question.title_group]].append(row)
        for rows in strict_rows.values():
            rows.sort(key=lambda r: r.question.qid)

        # ---- 3. profile cap per strict split + the document pool of the emitted rows ----
        selected: dict[str, list[NormalizedRow]] = {}
        caps_info: dict[str, splitter.CapInfo] = {}
        docs_selected: dict[str, dict[str, DocumentRecord]] = {}
        docs_raw_selected: dict[str, dict[str, str]] = {}
        strict_stats: dict[str, auditor.SchemaStats] = {}
        for name in splitter.STRICT_SPLITS:
            split_cap = train_cap if name == "train" else cap
            rows, cap_info = splitter.cap_rows(strict_rows[name], split_cap, salt=split_seed)
            selected[name] = rows
            caps_info[name] = cap_info
            docs_selected[name] = {}
            for row in rows:
                docs_selected[name].setdefault(row.document.doc_id, row.document)
            docs_raw_selected[name] = {doc_id: docs_raw[doc_id] for doc_id in docs_selected[name]}
            stats = auditor.SchemaStats(f"strict:{name}_strict")
            for row in rows:
                stats.add(row)
            strict_stats[name] = stats

        # ---- 4. scenarios + exclusions ----
        scenarios_by_split: dict[str, list[scenario_mod.ScenarioRecord]] = {}
        scenario_exclusions: list[dict[str, str]] = []
        for name in splitter.STRICT_SPLITS:
            scenarios, excl = scenario_mod.build_scenarios(
                selected[name], docs_selected[name], split=f"{name}_strict",
                n_distractors=n_distractors, scenario_seed=scenario_seed,
                hard_negative_ratio=hard_negative_ratio, bm25_k1=bm25_k1, bm25_b=bm25_b)
            scenarios_by_split[name] = scenarios
            scenario_exclusions.extend(excl)
        exclusions.extend(scenario_exclusions)
        all_scenarios = [s for name in splitter.STRICT_SPLITS for s in scenarios_by_split[name]]

        # ---- 5. leakage: identity sets, the two overlap views, near duplicates ----
        strict_sets = {name: auditor.row_sets(strict_rows[name]) for name in splitter.STRICT_SPLITS}
        overlap_full = {key: auditor.overlap_sets({n: strict_sets[n][key] for n in strict_sets})
                        for key in auditor.OVERLAP_KEYS}
        overlap_emitted = auditor.overlap_block(
            {f"{name}_strict": selected[name] for name in splitter.STRICT_SPLITS})

        full_docs: dict[str, dict[str, DocumentRecord]] = {}
        for name in splitter.STRICT_SPLITS:
            bucket: dict[str, DocumentRecord] = {}
            for row in strict_rows[name]:
                bucket.setdefault(row.document.doc_id, row.document)
            full_docs[f"{name}_strict"] = bucket

        near_duplicates: dict[str, Any] = {}
        for i, a in enumerate(splitter.STRICT_SPLITS):
            for b in splitter.STRICT_SPLITS[i + 1:]:
                docs_a, docs_b = full_docs[f"{a}_strict"], full_docs[f"{b}_strict"]
                keys_a, keys_b = sorted(docs_a), sorted(docs_b)
                result = splitter.cross_similarity(
                    keys_a, [docs_a[d].text for d in keys_a],
                    keys_b, [docs_b[d].text for d in keys_b],
                    threshold=REPORT_NEAR_DUP_THRESHOLD)
                report = splitter.near_duplicate_report(result, limit=10)
                report["pairs_at_or_above_merge_threshold"] = sum(
                    1 for _, _, score in result["all"] if score >= MERGE_NEAR_DUP_THRESHOLD)
                near_duplicates[f"{a}_strict~{b}_strict"] = report

        official_sets: dict[str, dict[str, set[str]]] = {}
        for name in OFFICIAL_SPLITS:
            stats = raw_stats[name]
            official_sets[name] = {
                "qid": set(official_qids[name]),
                "title_group": set(stats.unique_title_group),
                "doc_id": set(stats.unique_doc_id),
                "raw_context_sha256": set(stats.unique_context_raw),
                "row_id": {qid.split(":", 1)[1] for qid in official_qids[name]},
            }
        official_overlap = {key: auditor.overlap_sets({n: official_sets[n][key] for n in OFFICIAL_SPLITS})
                            for key in auditor.OVERLAP_KEYS}
        combined: dict[str, dict[str, set[str]]] = {}
        for key in auditor.OVERLAP_KEYS:
            bucket = {f"strict:{n}": strict_sets[n][key] for n in splitter.STRICT_SPLITS}
            bucket.update({f"official:{n}": official_sets[n][key] for n in ("validation", "test")})
            combined[key] = bucket
        strict_vs_official = {key: auditor.overlap_sets(combined[key]) for key in auditor.OVERLAP_KEYS}

        def _disjoint(key: str) -> bool:
            return all(entry["count"] == 0 for entry in overlap_full[key].values())

        invariants = {
            "qid_disjoint_across_strict_splits": _disjoint("qid"),
            "title_group_disjoint_across_strict_splits": _disjoint("title_group"),
            "doc_sha256_disjoint_across_strict_splits": _disjoint("doc_id"),
            "raw_context_disjoint_across_strict_splits": _disjoint("raw_context_sha256"),
            "near_duplicate_pairs_at_or_above_merge_threshold": sum(
                entry["pairs_at_or_above_merge_threshold"] for entry in near_duplicates.values()),
            "answerable_scenarios_all_have_gold_context": all(
                s.difficulty.get("gold_context_in_corpus") is not False for s in all_scenarios),
        }
        invariants["pass"] = (
            invariants["qid_disjoint_across_strict_splits"]
            and invariants["title_group_disjoint_across_strict_splits"]
            and invariants["doc_sha256_disjoint_across_strict_splits"]
            and invariants["raw_context_disjoint_across_strict_splits"]
            and invariants["near_duplicate_pairs_at_or_above_merge_threshold"] == 0
            and invariants["answerable_scenarios_all_have_gold_context"])

        # ---- 6. negative audit (automated heuristics — no human review claimed) ----
        rows_by_qid = {row.question.qid: row for name in splitter.STRICT_SPLITS
                       for row in selected[name]}
        docs_by_id = {doc_id: doc for name in splitter.STRICT_SPLITS
                      for doc_id, doc in docs_selected[name].items()}
        audit_rows, audit_summary, ambiguous_flags = auditor.audit_negatives(
            all_scenarios, rows_by_qid, docs_by_id, size=audit_size, seed=split_seed)
        # Data hygiene applied identically to all splits (never tuned on test_strict): the
        # flag travels with every scenario so downstream stages cannot miss it. Scenario
        # dataclasses are frozen but `difficulty` is a plain dict; this is the one place it
        # is completed, and it happens before anything is written.
        for scenario in all_scenarios:
            scenario.difficulty["ambiguous_negative"] = ambiguous_flags.get(scenario.qid, False)

        # ---- 7. artifacts ----
        artifacts: dict[str, dict[str, Any]] = {}

        def record(key: str, info: dict[str, Any], path: Path) -> None:
            info["path"] = _rel(path)
            artifacts[key] = info
            written.append(path)

        for name in splitter.STRICT_SPLITS:
            strict_name = f"{name}_strict"
            q_path = processed_dir / f"questions_{name}_strict.parquet"
            d_path = processed_dir / f"docs_{name}_strict.parquet"
            s_path = scenarios_dir / f"scenarios_{name}.jsonl"
            qid_path = processed_dir / f"qids_{name}_strict.txt"
            record(f"questions_{strict_name}",
                   write_parquet(
                       [question_row(r, strict_name, ambiguous_flags.get(r.question.qid, False))
                        for r in selected[name]], q_path, QUESTIONS_SCHEMA), q_path)
            record(f"docs_{strict_name}",
                   write_parquet([doc_row(docs_selected[name][d], docs_raw_selected[name][d],
                                          strict_name) for d in sorted(docs_selected[name])],
                                 d_path, DOCS_SCHEMA), d_path)
            n_lines = write_jsonl(s_path, [s.to_row() for s in scenarios_by_split[name]])
            record(f"scenarios_{name}",
                   {"rows": n_lines, "sha256": sha256_file(s_path), "bytes": s_path.stat().st_size},
                   s_path)
            qid_path.write_text("\n".join(sorted(r.question.qid for r in selected[name])) + "\n",
                                encoding="utf-8")
            record(f"qids_{strict_name}",
                   {"rows": len(selected[name]), "sha256": sha256_file(qid_path),
                    "bytes": qid_path.stat().st_size}, qid_path)

        exclusions_path = diagnostics_dir / "exclusions.csv"
        auditor.write_csv(exclusions_path, exclusions, ["qid", "split", "stage", "reason"])
        written.append(exclusions_path)
        negative_path = diagnostics_dir / "negative_audit.csv"
        auditor.write_csv(negative_path, audit_rows,
                          ["qid", "split", "docs", "verdict", "ambiguous_flag", "reviewer_note"])
        written.append(negative_path)
        # Machine-readable flag list: the RL trainer reads this to skip flagged TRAIN
        # negatives from reward-bearing updates in the primary analysis (spec §6), while
        # val_strict/test_strict keep their flags for clean-vs-ambiguous reporting.
        ambiguous_path = processed_dir / "ambiguous_negatives.json"
        atomic_write_json(ambiguous_path, {
            "method": audit_summary["method"],
            "generated_by": f"python -m src.pipeline.audit_split --profile {profile} ({run_id})",
            "splits": {
                f"{name}_strict": {
                    "flagged_qids": sorted(
                        s.qid for s in scenarios_by_split[name]
                        if s.difficulty.get("ambiguous_negative")),
                    "counts": dict(audit_summary["per_split"].get(
                        f"{name}_strict",
                        {"negatives": 0, "flagged": 0, "clean": 0, "flagged_rate": None,
                         "gold_doc_signal_negatives": 0})),
                } for name in splitter.STRICT_SPLITS},
            "note": "automated heuristic only; no human review",
        })
        written.append(ambiguous_path)
        doc_sha_path = diagnostics_dir / "doc_sha256.csv"
        auditor.write_csv(doc_sha_path, auditor.doc_sha_rows(full_docs),
                          ["doc_id", "source_split", "strict_split", "title_group", "n_chars"])
        written.append(doc_sha_path)

        leakage = {
            "generated_utc": now_iso(),
            "profile": profile,
            "cap_per_split": cap,
            "split_seed": split_seed,
            "scope": {f"raw:{n}": raw_stats[n].to_dict() for n in OFFICIAL_SPLITS},
            "thresholds": {"merge_near_duplicate_cosine": MERGE_NEAR_DUP_THRESHOLD,
                           "report_near_duplicate_cosine": REPORT_NEAR_DUP_THRESHOLD},
            "strict_splits": {
                "overlap_full_pool": overlap_full,
                "overlap_emitted": overlap_emitted,
                "full_pool_sizes": {f"{n}_strict": {
                    "rows": len(strict_rows[n]),
                    "title_groups": len(strict_sets[n]["title_group"]),
                    "docs": len(full_docs[f"{n}_strict"])} for n in splitter.STRICT_SPLITS},
            },
            "near_duplicates_across_strict_splits": near_duplicates,
            "official_splits": {
                "overlap": official_overlap,
                "note": ("The upstream `train`/`validation`/`test` splits overlap each other. "
                         "They are kept ONLY as a secondary, contamination-disclosed benchmark; "
                         "the strict splits are built from the upstream `train` split alone."),
            },
            "strict_vs_official": {"overlap": strict_vs_official},
            "invariants": invariants,
        }
        leakage_path = diagnostics_dir / "leakage_report.json"
        atomic_write_json(leakage_path, leakage)
        written.append(leakage_path)

        # ---- 8. schema report ----
        stats_for_report: dict[str, dict[str, Any]] = {}
        for name in OFFICIAL_SPLITS:
            stats_for_report[f"raw:{name}"] = raw_stats[name].to_dict()
        for name in splitter.STRICT_SPLITS:
            stats_for_report[f"strict:{name}_strict"] = strict_stats[name].to_dict()

        test_stats = raw_stats["test"].to_dict()
        flagged_by_split = {split: b["flagged"]
                            for split, b in sorted(audit_summary["per_split"].items())}
        notes = [
            f"Strict splits are built from the raw `train` split only "
            f"({raw_stats['train'].rows} rows). Raw `validation` and `test` are never used "
            f"to build them.",
            f"Rows in strict scopes are the *emitted* (profile-capped) rows; pool sizes are in "
            f"`data/processed/split_manifest.json` (profile `{profile}`, cap per split "
            f"`{cap if cap is not None else 'none (full)'}`, train cap "
            f"`{train_cap if train_cap is not None else 'none (full)'}`).",
            f"Near-duplicate merge: {merge_sim['pairs']} document pairs at cosine >= "
            f"{MERGE_NEAR_DUP_THRESHOLD} plus "
            f"{sum(1 for names in groups_by_doc.values() if len(names) > 1)} documents referenced "
            f"by more than one title merged "
            f"{sum(1 for g, comp in component_of.items() if g != comp)} title groups into "
            f"{len(component_stats)} components before allocation, so no two strict splits "
            f"contain a near-duplicate pair at or above that threshold and no document is "
            f"placed in two splits.",
            f"Near-duplicate pairs in [{REPORT_NEAR_DUP_THRESHOLD}, {MERGE_NEAR_DUP_THRESHOLD}) "
            f"across strict splits are reported (not silently ignored) in "
            f"`data/diagnostics/leakage_report.json`.",
            f"Offset policy: exact match kept; an unambiguous repair is applied but always "
            f"recorded in `offset_issue` of the row; anything else is left untouched and "
            f"reported as `offset_unrecoverable`. Measured on the raw `train` split: "
            f"{raw_stats['train'].offset_exact} exact, {raw_stats['train'].offset_recovered} "
            f"recovered, {raw_stats['train'].offset_unrecoverable} unrecoverable.",
            f"Positive rows with an unrecoverable gold offset are excluded from scenarios with "
            f"a logged reason in `data/diagnostics/exclusions.csv` (never silently repaired or "
            f"dropped): {len(scenario_exclusions)} scenario-stage exclusions, "
            f"{len(exclusions)} exclusions in total.",
            f"Negative audit (automated, distractor documents only): all "
            f"{audit_summary['negatives_available']} negative scenarios were evaluated and "
            f"{audit_summary['flagged_negatives_all']} are flagged `ambiguous_negative` "
            f"(overall rate {audit_summary['flagged_rate_all']}); the CSV covers the "
            f"deterministic sample of {audit_summary['audited']}. Signals on a question's own "
            f"gold doc are expected for `is_impossible` rows (measured: "
            f"{audit_summary['gold_doc_signal_negatives']} negatives carry such gold-doc "
            f"signals) and are reported informationally, never counted as ambiguity. NO human "
            f"review was performed — the CSV says so in every note.",
            f"`ambiguous_negative` policy (spec §6): flagged train_strict negatives are skipped "
            f"by the primary RL trainer for reward-bearing updates; flagged val_strict/"
            f"test_strict negatives are kept for clean-vs-ambiguous comparison. Flagged per "
            f"split: {flagged_by_split}; full list and counts in "
            f"`data/processed/ambiguous_negatives.json`. The heuristic is applied identically "
            f"to all splits and was judged on pilot-scale data — never tuned on test_strict.",
            f"`questions_*.parquet` is a LABEL STORE (gold answers, `is_impossible`, "
            f"`plausible_texts`, `gold_doc_id`). It must never be read into an observation or a "
            f"prompt; policies only ever see `QuestionRecord.public_view()` and "
            f"`ScenarioRecord.public_meta()`.",
            f"`data/diagnostics/doc_sha256.csv` lists every document of the strict splits "
            f"(full pool, before the profile cap) so split isolation can be re-checked.",
        ]
        if test_stats["answers_field_missing"] == test_stats["rows"] and test_stats["rows"]:
            notes.append(
                f"Raw `test` ships WITHOUT labels: all {test_stats['answers_field_missing']}/"
                f"{test_stats['rows']} rows have `answers=None` and `is_impossible=False`, so it "
                f"cannot be scored as a secondary benchmark; it is retained only as a format "
                f"reference.")
        train_val = official_overlap["qid"].get("train~validation", {"count": 0})
        train_test = official_overlap["qid"].get("train~test", {"count": 0})
        val_test = official_overlap["qid"].get("validation~test", {"count": 0})
        notes.append(
            f"Upstream contamination (measured): qid overlap train~validation="
            f"{train_val['count']}, train~test={train_test['count']}, "
            f"validation~test={val_test['count']}; context-sha overlap train~validation="
            f"{official_overlap['raw_context_sha256'].get('train~validation', {'count': 0})['count']}, "
            f"train~test="
            f"{official_overlap['raw_context_sha256'].get('train~test', {'count': 0})['count']}, "
            f"validation~test="
            f"{official_overlap['raw_context_sha256'].get('validation~test', {'count': 0})['count']}.")
        schema_path = diagnostics_dir / "schema_report.md"
        auditor.write_schema_report(
            stats_for_report, schema_path,
            context={
                "profile": profile,
                "cap per split": cap if cap is not None else "none (full)",
                "train cap": train_cap if train_cap is not None else "none (full)",
                "repo_id": source["repo_id"],
                "resolved revision": source["resolved_revision"],
                "split seed": split_seed,
                "scenario seed": scenario_seed,
                "scenario version": scenario_mod.SCENARIO_VERSION,
                "n distractors per scenario": n_distractors,
                "hard-negative ratio (reference BM25, other-title)": hard_negative_ratio,
                "merge near-dup cosine": MERGE_NEAR_DUP_THRESHOLD,
                "report near-dup cosine": REPORT_NEAR_DUP_THRESHOLD,
                "generated (UTC)": now_iso(),
            },
            exclusions=exclusions, negative_summary=audit_summary, notes=notes)
        written.append(schema_path)

        # ---- 9. split manifest ----
        manifest_path = processed_dir / "split_manifest.json"
        manifest: dict[str, Any] = {
            "stage": "P1_data",
            "profile": profile,
            "cap_per_split": cap,
            "train_cap": train_cap,
            "caps_definition": dict(caps),
            "split_manifest": _rel(manifest_path),
            "created_utc": now_iso(),
            "source": {
                "repo_id": source["repo_id"],
                "resolved_revision": source["resolved_revision"],
                "source_manifest": _rel(loader.manifest_path(raw_dir)),
                "source_manifest_sha256": sha256_file(loader.manifest_path(raw_dir)),
                "raw_rows": {name: source["splits"][name]["rows"] for name in source["splits"]},
                "raw_split_rows_used_for_strict": raw_stats["train"].rows,
            },
            "normalization": {
                "qid_scheme": "<source_split>:<raw id> (duplicates get a checksum suffix)",
                "display_text": "NFC + whitespace collapse with a verified raw->display index map",
                "context_raw_kept": True,
                "offset_policy": ("exact kept; unambiguous repair logged in offset_issue; "
                                  "otherwise unrecoverable and reported, never silently fixed"),
            },
            "strict_split": {
                "source_split": "train",
                "seed": split_seed,
                "ratios_target": ratios,
                "n_title_groups": len(set(group_of.values())),
                "n_components_after_near_duplicate_merge": len(component_stats),
                "n_groups_merged": sum(1 for g, comp in component_of.items() if g != comp),
                "n_near_duplicate_document_pairs": len(merge_sim["all"]),
                "n_documents_referenced_by_multiple_titles": sum(
                    1 for names in groups_by_doc.values() if len(names) > 1),
                "near_duplicate_merge_threshold": MERGE_NEAR_DUP_THRESHOLD,
                "splits": {f"{name}_strict": {
                    "rows_pool": len(strict_rows[name]),
                    "rows_emitted": len(selected[name]),
                    "answerable_emitted": strict_stats[name].answerable,
                    "impossible_emitted": strict_stats[name].impossible,
                    "title_groups_pool": len(strict_sets[name]["title_group"]),
                    "docs_emitted": len(docs_selected[name]),
                } for name in splitter.STRICT_SPLITS},
                "caps": {f"{name}_strict": caps_info[name].to_dict()
                         for name in splitter.STRICT_SPLITS},
            },
            "leakage": invariants,
            "leakage_report": _rel(leakage_path),
            "scenarios": {
                "version": scenario_mod.SCENARIO_VERSION,
                "scenario_seed": scenario_seed,
                "n_distractors": n_distractors,
                "hard_negative_ratio": hard_negative_ratio,
                "hard_negative_reference": {"tool": "bm25", "k1": bm25_k1, "b": bm25_b},
                "splits": {f"{name}_strict": scenario_mod.scenario_summary(scenarios_by_split[name])
                           for name in splitter.STRICT_SPLITS},
            },
            "negative_audit": audit_summary,
            "ambiguous_negatives": {
                "path": _rel(ambiguous_path),
                "policy": ("primary analysis: flagged train_strict negatives are skipped by the "
                           "RL trainer for reward-bearing updates; val_strict/test_strict keep "
                           "them so clean vs ambiguous negatives can be compared (spec §6)"),
                "human_review": False,
            },
            "exclusions": {
                "total": len(exclusions),
                "scenario_stage": len(scenario_exclusions),
                "by_reason": dict(Counter(e["reason"].split(":")[0] for e in exclusions)),
            },
            "artifacts": artifacts,
            "library_versions": capture_library_versions(),
        }
        atomic_write_json(manifest_path, manifest)
        written.append(manifest_path)
        result = manifest
    except Exception as exc:
        registry.append(run_id=run_id, stage="P1_audit_split", status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise

    registry.record_artifact(run_id, "P1_audit_split", written, profile=profile, cap_per_split=cap,
                             train_cap=train_cap, invariants_pass=result["leakage"]["pass"],
                             rows={name: info["rows_emitted"]
                                   for name, info in result["strict_split"]["splits"].items()})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.audit_split",
                                     description="P1: normalize, strict split, scenarios, audits")
    add_profile_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    manifest = run(cfg=cfg, profile=profile)
    summary = {
        "profile": manifest["profile"],
        "cap_per_split": manifest["cap_per_split"],
        "strict_split": manifest["strict_split"]["splits"],
        "leakage": manifest["leakage"],
        "scenarios": manifest["scenarios"]["splits"],
        "negative_audit": manifest["negative_audit"],
        "exclusions": manifest["exclusions"],
        "manifest": manifest["split_manifest"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

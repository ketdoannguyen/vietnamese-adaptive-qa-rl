"""Freeze a train-only balanced scenario QID curriculum from measured frozen worlds.

No validation labels are used for selection; no synthetic ranking manipulation. All
scenarios keep their original BM25/Dense rankings, documents, reader and reward. A
question appears once. This is a diagnostic stress train set, not natural validation.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.hashing import sha256_file, sha256_json, stable_seed

DIAGNOSIS = Path("outputs/experiments/2026-10-11-rl-bottleneck/diagnosis.json")
OUTPUT = DIAGNOSIS.with_name("train_search_curriculum.json")


def select(rows: list[dict], *, flagged: set[str], per_bucket: int = 100, seed: int = 42) -> dict:
    clean = [r for r in rows if r["qid"] not in flagged]
    cohorts = {
        "search_recovers": [r for r in clean if r["search_recovers_reader"]
                            and r["gold_position"] in ("first_search", "second_search")],
        "already_correct": [r for r in clean if not r["is_impossible"]
                            and r["bootstrap_reader"] == "correct"],
        "search_fails": [r for r in clean if not r["is_impossible"]
                         and r["gold_position"] in ("first_search", "second_search")
                         and not r["search_recovers_reader"]],
        "negative": [r for r in clean if r["is_impossible"]],
    }
    output = {}
    for name, members in cohorts.items():
        chosen = sorted(members, key=lambda row: (stable_seed(seed, name, row["qid"]),
                                                  row["qid"]))[:per_bucket]
        if len(chosen) < per_bucket:
            raise ValueError(f"{name}: only {len(chosen)}/{per_bucket} clean training QIDs")
        output[name] = sorted(r["qid"] for r in chosen)
    all_qids = [qid for bucket in output.values() for qid in bucket]
    if len(set(all_qids)) != 4 * per_bucket:
        raise ValueError("curriculum strata overlap")
    return {"n_per_stratum": per_bucket, "seed": seed, "n": len(all_qids),
            "qids_sha256": sha256_json(sorted(all_qids)),
            "source_strata_sizes": {key: len(value) for key, value in cohorts.items()},
            "selected": output}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite: {OUTPUT}")
    report = json.loads(DIAGNOSIS.read_text(encoding="utf-8"))
    rows = report["per_qid"]["train"]
    # The audit's clean train sample already reports how many ambiguous negatives were
    # excluded; retrieve exactly that set from the P1 audit file rather than inferring it.
    from src.pipeline.worlds import _parse_ambiguous
    from src.utils.io import read_json
    flagged = set(_parse_ambiguous(read_json("data/processed/ambiguous_negatives.json"),
                                    "train_strict"))
    result = select(rows, flagged=flagged)
    result["diagnosis_sha256"] = sha256_file(DIAGNOSIS)
    with OUTPUT.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, sort_keys=True, indent=2)
        out.write("\n")
    print("selected", result["n"], "train-only original scenarios; stratum sizes",
          result["source_strata_sizes"], "wrote", OUTPUT)


if __name__ == "__main__":
    main()

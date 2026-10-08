"""P1 download CLI: resolve the dataset revision, pin it, persist raw parquet + manifest.

    python -m src.pipeline.download --profile smoke
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from src.data import loader
from src.utils.config import add_profile_args, config_from_args, resolve
from src.utils.runs import RunRegistry, now_iso


def run(*, cfg: dict[str, Any], profile: str, repo: str | None = None,
        revision: str | None = None, force: bool = False,
        registry: RunRegistry | None = None) -> dict[str, Any]:
    raw_dir = resolve(cfg, "data_raw")
    repo = repo or cfg["data"]["hf_repo"]
    run_id = f"p1-download-{profile}-{now_iso().replace(':', '')}"
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    registry.append(run_id=run_id, stage="P1_download", status="RUNNING",
                    profile=profile, repo_id=repo, revision=revision, force=force)
    try:
        manifest = loader.persist_raw(raw_dir, repo_id=repo, revision=revision, force=force)
    except Exception as exc:
        registry.append(run_id=run_id, stage="P1_download", status="FAILED",
                        profile=profile, error=f"{type(exc).__name__}: {exc}")
        raise
    artifacts = [loader.manifest_path(raw_dir)] + [raw_dir / e["parquet"]
                                                   for e in manifest["splits"].values()]
    registry.record_artifact(run_id, "P1_download", artifacts, profile=profile,
                             revision=manifest["resolved_revision"], reused=manifest["reused"],
                             rows={k: v["rows"] for k, v in manifest["splits"].items()})
    summary = {
        "run_id": run_id,
        "repo_id": manifest["repo_id"],
        "resolved_revision": manifest["resolved_revision"],
        "reused_existing_download": manifest["reused"],
        "splits": {k: {"rows": v["rows"], "answerable_rows": v["answerable_rows"],
                       "impossible_rows": v["impossible_rows"],
                       "answers_field_none_rows": v["answers_field_none_rows"],
                       "sha256": v["sha256"]} for k, v in manifest["splits"].items()},
        "manifest": str(loader.manifest_path(raw_dir)),
        "license_verified": manifest["license_verified"],
    }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline.download",
                                     description="P1: pin + download UIT-ViQuAD2.0 raw data")
    add_profile_args(parser)
    parser.add_argument("--repo", default=None, help="override data.hf_repo")
    parser.add_argument("--revision", default=None, help="explicit commit SHA to pin")
    parser.add_argument("--force", action="store_true", help="re-download even if intact")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    summary = run(cfg=cfg, profile=profile, repo=args.repo, revision=args.revision,
                  force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

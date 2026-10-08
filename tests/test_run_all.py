"""Tests for ``scripts/run_all.sh`` and ``scripts/reproduce.sh`` — hermetic temp-repo runs.

Each test builds a minimal copy of the repository (scripts, configs, src/utils) under
``tmp_path`` and invokes the scripts through ``bash`` with the current interpreter, so
the real registry/outputs are never touched. No stage body ever executes real work: every
plan is a ``--dry-run``, stops at a stage whose gate is already satisfied, or runs a stub
stage module that only writes a registry row. The P7 stub in the reproduce tests writes a
parquet the test prepared — no model is loaded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from src.utils.config import load_config
from src.utils.hashing import sha256_file, sha256_text

REAL = Path(__file__).resolve().parents[1]
DAG = ["download", "audit_split", "build_retrieval", "train_reader", "cache_reader",
       "make_teacher", "train_laya_ce", "train_laya_rlcd", "cache_laya", "train_rl",
       "train_correctness", "eval_baselines_val", "eval_baselines_test", "report"]

STUB_STAGE = """\
import argparse
from pathlib import Path

from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args, _ = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    artifact = repo / "data/raw/raw.parquet"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"stub-download")
    RunRegistry(repo / "outputs/runs/index.jsonl").append(
        run_id="p1-download-stub-1", stage="P1_download", status="SUCCEEDED",
        profile=args.profile, artifacts={str(artifact): {
            "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}})


if __name__ == "__main__":
    main()
"""


def make_repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("run_all.sh", "reproduce.sh"):
        shutil.copy2(REAL / "scripts" / name, root / "scripts" / name)
    shutil.copytree(REAL / "configs", root / "configs")
    (root / "src" / "pipeline").mkdir(parents=True)
    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pipeline" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "utils").mkdir()
    for name in ("__init__.py", "config.py", "hashing.py", "io.py", "logging.py",
                 "reproducibility.py", "runs.py"):
        shutil.copy2(REAL / "src" / "utils" / name, root / "src" / "utils" / name)
    for rel in ("outputs/runs", "outputs/manifests/run_all", "outputs/diagnostics",
                "outputs/predictions", "data/raw"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def run_script(root: Path, name: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / name), *args], cwd=str(root),
        capture_output=True, text=True,
        env={**os.environ, "PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"})


def pilot_config_sha() -> str:
    """The sha the scripts compute for --profile pilot (configs are byte-identical copies)."""
    cfg = load_config("pilot")
    return sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2))


def registry(root: Path) -> list[dict]:
    path = root / "outputs" / "runs" / "index.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_registry(root: Path, rows: list[dict]) -> None:
    path = root / "outputs" / "runs" / "index.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def satisfied_artifact(root: Path, rel: str, payload: bytes = b"payload") -> tuple[str, dict]:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path), {"sha256": sha256_file(path), "bytes": path.stat().st_size}


def succeeded_row(root: Path, stage: str, run_id: str, rel: str,
                  profile: str = "pilot") -> tuple[dict, dict]:
    """A registry row shaped like the real stages write them (absolute artifact keys)."""
    artifact, meta = satisfied_artifact(root, rel)
    return ({"run_id": run_id, "stage": stage, "status": "SUCCEEDED", "profile": profile,
             "ts": "2026-10-07T00:00:00+00:00", "artifacts": {artifact: meta}}, meta)


def write_manifest(root: Path, key: str, run_id: str, outputs: dict, config_sha: str,
                   profile: str = "pilot", overlay: dict | None = None) -> None:
    """A stage manifest shaped like run_all writes them (repo-relative output keys)."""
    path = root / "outputs" / "manifests" / "run_all" / f"{key}.json"
    path.write_text(json.dumps({
        "stage_key": key, "registry_run_id": run_id, "profile": profile,
        "config_sha": config_sha, "outputs_sha": outputs, "overlay": overlay or {},
    }, ensure_ascii=False), encoding="utf-8")


PLAN_RE = re.compile(r"^\s*(\d+)\s+(\S+)\s+(CHẠY|BỎ QUA)\s+(.*)$")


def plan_lines(stdout: str) -> dict[str, tuple[str, str]]:
    """Parse the printed plan table into {stage_key: (verdict, reason)} (print order)."""
    plan = {}
    for line in stdout.splitlines():
        match = PLAN_RE.match(line)
        if match:
            plan[match.group(2)] = (match.group(3), match.group(4).strip())
    return plan


# ---------------------------------------------------------------------------- run_all

def test_scripts_parse_with_bash_n():
    for name in ("run_all.sh", "reproduce.sh"):
        proc = subprocess.run(["bash", "-n", str(REAL / "scripts" / name)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{name}: {proc.stderr}"


def test_dry_run_prints_the_dag_in_order_and_writes_nothing(tmp_path):
    root = make_repo(tmp_path)
    before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    plan = plan_lines(proc.stdout)
    assert list(plan) == DAG  # exact DAG order, no extra or missing stages
    assert all(verdict == "CHẠY" for verdict, _ in plan.values())
    assert "DRY-RUN" in proc.stdout
    assert "đĩa" in proc.stdout and "RAM" in proc.stdout and "GPU" in proc.stdout
    assert "best_selection.json" in proc.stdout  # the freeze precondition is disclosed
    after = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert after == before  # --dry-run activates nothing and writes nothing
    assert not (root / "outputs" / "runs" / "index.jsonl").exists()


def test_gate_skips_satisfied_stages_and_forces_downstream_reruns(tmp_path):
    root = make_repo(tmp_path)
    sha = pilot_config_sha()
    row_d, meta_d = succeeded_row(root, "P1_download", "p1-download-pilot-1",
                                  "data/raw/raw.parquet")
    row_b, meta_b = succeeded_row(root, "P2_retrieval", "p2-retrieval-pilot-1",
                                  "data/cache/bank.npz")
    row_a, _ = succeeded_row(root, "P1_audit_split", "p1-audit-split-pilot-1",
                             "data/processed/qids.txt")
    write_registry(root, [row_d, row_b, row_a])
    # download's manifest records a different config sha -> the stage must re-run...
    write_manifest(root, "download", "p1-download-pilot-1",
                   {"data/raw/raw.parquet": meta_d}, "0" * 64)
    # ...and a re-running download invalidates build_retrieval even though its own gate passes.
    write_manifest(root, "build_retrieval", "p2-retrieval-pilot-1",
                   {"data/cache/bank.npz": meta_b}, sha)
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    plan = plan_lines(proc.stdout)
    assert plan["download"][0] == "CHẠY" and "config đã đổi" in plan["download"][1]
    assert plan["build_retrieval"][0] == "CHẠY"
    assert "phụ thuộc download" in plan["build_retrieval"][1]
    # audit_split succeeded with intact artifacts but never got a run_all manifest
    assert plan["audit_split"][0] == "CHẠY"
    assert "chưa có stage_manifest" in plan["audit_split"][1]


def test_satisfied_stages_skip_when_nothing_upstream_changed(tmp_path):
    root = make_repo(tmp_path)
    sha = pilot_config_sha()
    row_d, meta_d = succeeded_row(root, "P1_download", "p1-download-pilot-1",
                                  "data/raw/raw.parquet")
    row_a, meta_a = succeeded_row(root, "P1_audit_split", "p1-audit-split-pilot-1",
                                  "data/processed/qids.txt")
    write_registry(root, [row_d, row_a])
    write_manifest(root, "download", "p1-download-pilot-1",
                   {"data/raw/raw.parquet": meta_d}, sha)
    write_manifest(root, "audit_split", "p1-audit-split-pilot-1",
                   {"data/processed/qids.txt": meta_a}, sha)
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--dry-run")
    assert proc.returncode == 0, proc.stderr
    plan = plan_lines(proc.stdout)
    assert plan["download"][0] == "BỎ QUA"
    assert "đã xong p1-download-pilot-1" in plan["download"][1]
    assert plan["audit_split"][0] == "BỎ QUA"
    assert plan["build_retrieval"][0] == "CHẠY"  # nothing recorded for it yet


def test_a_dead_running_stage_aborts_without_resume(tmp_path):
    root = make_repo(tmp_path)
    write_registry(root, [{"run_id": "p2-retrieval-pilot-1", "stage": "P2_retrieval",
                           "status": "RUNNING", "profile": "pilot",
                           "ts": "2026-10-07T00:00:00+00:00"}])
    before = (root / "outputs" / "runs" / "index.jsonl").read_text(encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot")
    assert proc.returncode == 3
    assert "RUNNING" in proc.stderr and "--resume" in proc.stderr
    assert (root / "outputs" / "runs" / "index.jsonl").read_text(encoding="utf-8") == before
    assert not (root / "outputs" / "diagnostics" / "run_all").exists()


def test_a_successful_stage_is_gated_by_the_manifest_the_run_just_wrote(tmp_path):
    """One stub stage runs for real (no training): the manifest round-trip gates the rerun."""
    root = make_repo(tmp_path)
    (root / "src" / "pipeline" / "download.py").write_text(STUB_STAGE, encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "download")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[1/14] download (P1_download) — CHẠY" in proc.stdout and "XONG" in proc.stdout
    manifest = json.loads(
        (root / "outputs" / "manifests" / "run_all" / "download.json").read_text("utf-8"))
    assert manifest["stage_key"] == "download" and manifest["registry_stage"] == "P1_download"
    assert manifest["registry_run_id"] == "p1-download-stub-1"
    assert manifest["config_sha"] == pilot_config_sha()
    assert manifest["exit_code"] == 0
    assert manifest["outputs_sha"] == {
        "data/raw/raw.parquet": {"sha256": sha256_file(root / "data/raw/raw.parquet"),
                                 "bytes": 13}}
    assert (root / manifest["log"]).is_file()
    rows = registry(root)
    assert any(row.get("event") == "stage" and row.get("stage_key") == "download"
               and row["status"] == "SUCCEEDED" for row in rows)
    end = [row for row in rows if row.get("event") == "end"][0]
    assert end["status"] == "SUCCEEDED" and end["executed"] == ["download"]
    # ...and the gate accepts what the run wrote: the same stage now skips.
    again = run_script(root, "run_all.sh", "--profile", "pilot", "--dry-run")
    plan = plan_lines(again.stdout)
    assert plan["download"][0] == "BỎ QUA"
    assert "đã xong p1-download-stub-1" in plan["download"][1]


def test_resume_marks_interrupted_and_continues_from_the_gated_stage(tmp_path):
    root = make_repo(tmp_path)
    sha = pilot_config_sha()
    row_d, meta_d = succeeded_row(root, "P1_download", "p1-download-pilot-1",
                                  "data/raw/raw.parquet")
    write_manifest(root, "download", "p1-download-pilot-1",
                   {"data/raw/raw.parquet": meta_d}, sha)
    write_registry(root, [row_d, {"run_id": "p2-retrieval-pilot-1", "stage": "P2_retrieval",
                                  "status": "RUNNING", "profile": "pilot",
                                  "ts": "2026-10-07T00:00:00+00:00"}])
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                      "--stop-after", "download")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "RECOVERY" in proc.stdout and "dừng theo yêu cầu" in proc.stdout
    rows = registry(root)
    interrupted = [row for row in rows if row.get("status") == "INTERRUPTED"]
    assert [row["stage"] for row in interrupted] == ["P2_retrieval"]
    assert "INTERRUPTED" in interrupted[0]["reason"]
    assert any(row.get("event") == "recovery" for row in rows)
    assert any(row.get("event") == "end" and row["status"] == "SUCCEEDED" for row in rows)
    # the stale row is still there (append-only) and no extra P2_retrieval row was added
    assert [row["stage"] for row in rows if row.get("stage") == "P2_retrieval"] == [
        "P2_retrieval", "P2_retrieval"]
    assert list((root / "outputs" / "diagnostics" / "run_all").glob("*/config_pilot.json"))


# --------------------------------------------------------------------------- reproduce

STUB_EVAL = """\
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile")
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    rows = json.loads((root / "outputs/predictions/stub_rows.json").read_text("utf-8"))
    out = root / "outputs/predictions/reproduced_stub.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(out))
    print(json.dumps({"run_id": "stub", "split": args.split,
                      "predictions": {"path": str(out)}}))


if __name__ == "__main__":
    main()
"""

PREDICTION_ROWS = [
    {"exp_id": "E06", "seed": 7, "qid": "val:0001", "terminal_action": "ANSWER",
     "action_name_trace": ["SEARCH_BM25", "ANSWER"], "final_candidate": "1955",
     "eval_correct": 1, "eval_is_impossible": False, "eval_correctness_f1": 1.0,
     "eval_correctness_em": 1.0, "retrieval_cost": 0.2, "eval_episode_return": 1.8,
     "optional_used": 2, "bm25_calls": 2, "dense_calls": 0, "total_retrievals": 2,
     "forced_terminal": False, "final_candidate_doc_id": "doc_a"},
    {"exp_id": "E06", "seed": 7, "qid": "val:0002", "terminal_action": "ABSTAIN",
     "action_name_trace": ["ABSTAIN"], "final_candidate": None,
     "eval_correct": 1, "eval_is_impossible": True, "eval_correctness_f1": 0.4,
     "eval_correctness_em": 0.0, "retrieval_cost": 0.1, "eval_episode_return": 0.7,
     "optional_used": 1, "bm25_calls": 1, "dense_calls": 0, "total_retrievals": 1,
     "forced_terminal": False, "final_candidate_doc_id": None},
]

P7_RUN_ID = "p7-baselines-pilot-val-7-1"


def make_p7_world(root: Path, with_manifest: bool = False) -> Path:
    """A recorded P7 run whose stored predictions and the P7 stub's output are the same rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    stored = root / "outputs" / "predictions" / "p7-stored_val_strict.parquet"
    pq.write_table(pa.Table.from_pylist(PREDICTION_ROWS), str(stored))
    (root / "outputs" / "predictions" / "stub_rows.json").write_text(
        json.dumps(PREDICTION_ROWS), encoding="utf-8")
    (root / "src" / "pipeline" / "eval_baselines.py").write_text(STUB_EVAL, encoding="utf-8")
    meta = {"sha256": sha256_file(stored), "bytes": stored.stat().st_size}
    write_registry(root, [{
        "run_id": P7_RUN_ID, "stage": "P7_baselines", "status": "SUCCEEDED",
        "profile": "pilot", "split": "val_strict", "seed": 7, "dev": False,
        "ts": "2026-10-07T00:00:00+00:00", "artifacts": {str(stored): meta}}])
    if with_manifest:
        write_manifest(root, "eval_baselines_val", P7_RUN_ID, {}, pilot_config_sha())
    return stored


def test_reproduce_verifies_hashes_config_and_a_prediction_subset(tmp_path):
    root = make_repo(tmp_path)
    stored = make_p7_world(root, with_manifest=True)
    stored_sha = sha256_file(stored)
    proc = run_script(root, "reproduce.sh", "--run-id", P7_RUN_ID, "--mode", "eval")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "[PASS] hash nguồn & artifact" in proc.stdout
    assert "[PASS] cấu hình eval_baselines_val" in proc.stdout
    assert "[PASS] tái tạo subset predictions" in proc.stdout
    assert "KẾT QUẢ: PASS" in proc.stdout
    assert (root / "outputs" / "predictions" / "reproduced_stub.parquet").is_file()
    assert list((root / "outputs/diagnostics/reproduce" / P7_RUN_ID).glob("eval_*.log"))
    assert sha256_file(stored) == stored_sha  # the recorded artifact is never rewritten


def test_reproduce_reports_a_changed_prediction_as_fail(tmp_path):
    root = make_repo(tmp_path)
    make_p7_world(root)
    changed = [dict(PREDICTION_ROWS[0]), dict(PREDICTION_ROWS[1])]
    changed[1]["eval_correctness_f1"] = 0.9  # stored says 0.4
    (root / "outputs" / "predictions" / "stub_rows.json").write_text(
        json.dumps(changed), encoding="utf-8")
    proc = run_script(root, "reproduce.sh", "--run-id", P7_RUN_ID, "--mode", "eval")
    assert proc.returncode == 1
    assert "[FAIL] tái tạo subset predictions" in proc.stdout
    assert "eval_correctness_f1" in proc.stdout
    assert "[NOT_RUN] cấu hình (config_sha)" in proc.stdout  # no run_all manifest recorded it
    assert "KẾT QUẢ: FAIL" in proc.stdout


def test_reproduce_refuses_an_unknown_run_id(tmp_path):
    root = make_repo(tmp_path)
    write_registry(root, [{"run_id": "p2-retrieval-pilot-1", "stage": "P2_retrieval",
                           "status": "SUCCEEDED", "profile": "pilot"}])
    proc = run_script(root, "reproduce.sh", "--run-id", "nope", "--mode", "eval")
    assert proc.returncode == 3
    assert "nope" in proc.stderr and "p2-retrieval-pilot-1" in proc.stderr

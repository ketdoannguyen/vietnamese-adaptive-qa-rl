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
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.utils.config import load_config
from src.utils.hashing import sha256_file, sha256_text
from src.utils.io import read_yaml, write_yaml

REAL = Path(__file__).resolve().parents[1]
DAG = ["download", "audit_split", "build_retrieval", "train_reader", "cache_reader",
       "make_teacher", "train_laya_ce", "train_laya_rlcd", "cache_laya", "train_rl",
       "select_rl", "train_correctness", "eval_baselines_val", "cache_reader_test",
       "eval_baselines_test", "report"]

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
    for name in ("__init__.py", "config.py", "hashing.py", "io.py",
                 "reproducibility.py", "runs.py"):
        shutil.copy2(REAL / "src" / "utils" / name, root / "src" / "utils" / name)
    for rel in ("outputs/runs", "outputs/manifests/run_all", "outputs/diagnostics",
                "outputs/predictions", "data/raw"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


def script_env() -> dict:
    return {**os.environ, "PYTHON": sys.executable, "PYTHONDONTWRITEBYTECODE": "1"}


def run_script(root: Path, name: str, *args: str, timeout: int = 120
               ) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / name), *args], cwd=str(root),
        capture_output=True, text=True, timeout=timeout, env=script_env())


def pilot_config_sha() -> str:
    """The sha the scripts compute for --profile pilot (configs are byte-identical copies)."""
    cfg = load_config("pilot")
    return sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2))


def test_small_profile_has_distinct_identity_and_1000_row_cap() -> None:
    small = load_config("small")
    assert small["data"]["profile"] == "small"
    assert small["data"]["caps"]["small"] == 1000
    assert config_sha_for("small", {}) != config_sha_for("full", {})


def test_pilot_overlays_keep_full_validation_and_separate_outputs() -> None:
    configs = []
    for size in (1000, 5000):
        overlay = read_yaml(REAL / "configs" / f"gliner_laya_{size}.yaml")
        cfg = load_config("medium", extra=overlay)
        assert cfg["data"]["profile"] == "medium"
        assert cfg["data"]["train_cap"] == size
        assert cfg["data"]["caps"]["medium"] == 5000
        assert cfg["retrieval"]["dense"]["revision"] == \
            "614241f622f53c4eeff9890bdc4f31cfecc418b3"
        assert cfg["reader"]["revision"] == "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
        assert cfg["limits"]["max_gpu_hours"] == 48
        roots = {Path(value).parts[:3] for value in cfg["paths"].values()}
        assert roots == {("outputs", "experiments", f"gliner-laya-{size}")}
        configs.append(cfg)
    assert configs[0]["paths"]["data_processed"] != configs[1]["paths"]["data_processed"]
    assert config_sha_for("medium", read_yaml(REAL / "configs" / "gliner_laya_1000.yaml")) != \
        config_sha_for("medium", read_yaml(REAL / "configs" / "gliner_laya_5000.yaml"))


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
                   profile: str = "pilot", overlay: dict | None = None,
                   base_config_sha: str | None = None) -> None:
    """A stage manifest shaped like run_all writes them (repo-relative output keys)."""
    path = root / "outputs" / "manifests" / "run_all" / f"{key}.json"
    path.write_text(json.dumps({
        "stage_key": key, "registry_run_id": run_id, "profile": profile,
        "config_sha": config_sha,
        "base_config_sha": base_config_sha or config_sha_for(profile, {}),
        "outputs_sha": outputs, "overlay": overlay or {},
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


def test_ledger_close_failure_leaves_interval_open_and_run_all_nonzero(tmp_path):
    root = make_repo(tmp_path)
    # The child appends a malformed row after its interval opens but exits successfully.
    corrupt = STUB_STAGE.replace(
        "    RunRegistry(repo /",
        "    with (repo / 'outputs/experiments/gpu_ledger.jsonl').open('a') as ledger:\n"
        "        ledger.write('not json\\n')\n"
        "    RunRegistry(repo /",
    )
    (root / "src/pipeline/download.py").write_text(corrupt, encoding="utf-8")
    result = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "download")
    assert result.returncode != 0
    assert "ledger" in (result.stdout + result.stderr).lower()
    lines = (root / "outputs/experiments/gpu_ledger.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["event"] == "open"
    assert lines[1:] == ["not json"]  # no successful close row or ledger rewrite
    assert not (root / "outputs/manifests/run_all/download.json").exists()
    assert any(row.get("event") == "stage" and row.get("stage_key") == "download"
               and row.get("status") == "FAILED" for row in registry(root))
    assert any(row.get("event") == "end" and row.get("status") == "FAILED"
               for row in registry(root))


def test_malformed_shared_ledger_refuses_cpu_stage_before_side_effects(tmp_path):
    root = make_repo(tmp_path)
    (root / "src/pipeline/download.py").write_text(STUB_STAGE, encoding="utf-8")
    ledger = root / "outputs/experiments/gpu_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("not json\n", encoding="utf-8")
    result = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "download")
    assert result.returncode != 0
    assert not (root / "data/raw/raw.parquet").exists()
    assert ledger.read_text(encoding="utf-8") == "not json\n"


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


def test_unknown_running_stage_refuses_resume_without_writing(tmp_path):
    root = make_repo(tmp_path)
    write_registry(root, [{"run_id": "p1-download-live", "stage": "P1_download",
                           "status": "RUNNING", "profile": "pilot"}])
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    pgrep = fake_bin / "pgrep"
    pgrep.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")
    pgrep.chmod(0o755)
    env = {**script_env(), "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    before = (root / "outputs/runs/index.jsonl").read_bytes()
    proc = subprocess.run(["bash", str(root / "scripts/run_all.sh"), "--profile", "pilot",
                           "--resume", "--stop-after", "download"], cwd=root,
                          capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "pgrep" in proc.stderr
    assert (root / "outputs/runs/index.jsonl").read_bytes() == before
    assert not list((root / "outputs/diagnostics").rglob("config_pilot.json"))


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
    assert "[1/16] download (P1_download) — CHẠY" in proc.stdout and "XONG" in proc.stdout
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


def test_resume_marker_preserves_filtered_stage_identity(tmp_path):
    root = make_repo(tmp_path)
    sha = pilot_config_sha()
    row, meta = succeeded_row(root, "P1_download", "download-1", "data/raw/raw.parquet")
    write_manifest(root, "download", "download-1", {"data/raw/raw.parquet": meta}, sha)
    write_registry(root, [row, {"run_id": "ce-killed-1", "stage": "P5_laya_train",
                               "status": "RUNNING", "profile": "pilot", "loss": "soft-ce"}])
    resumed = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                         "--stop-after", "download")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    plan = plan_lines(resumed.stdout)
    assert plan["train_laya_ce"][1].startswith("lần chạy trước: INTERRUPTED")
    marker = [r for r in registry(root) if r.get("status") == "INTERRUPTED"]
    assert len(marker) == 1 and marker[0]["loss"] == "soft-ce"
    again = run_script(root, "run_all.sh", "--profile", "pilot",
                       "--stop-after", "download")
    assert again.returncode == 0, again.stdout + again.stderr


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


# ----------------------------------------------------- P0: output lock & GPU ledger
#
# Contract exercised here (implementation expected in src/utils/runs.py + run_all.sh):
# * a real invocation takes an exclusive lock over the output namespace before it writes
#   anything; a second live invocation refuses (exit 3) and writes nothing, while
#   --dry-run stays read-only and needs no lock;
# * every launched stage records an allocation interval in
#   outputs/experiments/gpu_ledger.jsonl; CPU-only stages record gpu=false and spend nothing;
# * train_reader (P3) is a GPU stage, download (P1) is CPU-only;
# * before launching a GPU stage run_all checks the cumulative ledger: a ledger that is
#   over limits.max_gpu_hours, malformed, or carrying an open interval from a killed
#   invocation fails closed and the stage is never launched.

LEDGER_REL = "outputs/experiments/gpu_ledger.jsonl"


def tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def iso_utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds")


def ledger_intervals(root: Path) -> list[dict]:
    """Fold ``gpu_ledger.jsonl`` open/close events into ``[{open, close|None}, ...]``."""
    path = root / LEDGER_REL
    if not path.is_file():
        return []
    opens: dict[str, dict] = {}
    closes: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "open":
            opens[row["interval_id"]] = row
        elif row.get("event") == "close":
            closes[row["interval_id"]] = row
    return [{"open": row, "close": closes.get(interval_id)}
            for interval_id, row in opens.items()]


def gpu_seconds_from_ledger(root: Path) -> float:
    now = time.time()
    total = 0.0
    for interval in ledger_intervals(root):
        opened, closed = interval["open"], interval["close"]
        if not opened.get("gpu", True):
            continue
        if closed is not None:
            total += float(closed.get("seconds") or 0.0)
        else:
            started = datetime.fromisoformat(opened["started"]).timestamp()
            total += max(0.0, now - started)
    return total


def append_ledger_interval(root: Path, *, run_id: str, stage_key: str, seconds: float,
                           status: str = "SUCCEEDED", gpu: bool = True) -> None:
    """A completed interval exactly as a previous invocation would have left it."""
    interval_id = f"seed-{stage_key}-{run_id}"
    end = time.time()
    rows = [
        {"event": "open", "interval_id": interval_id, "run_id": run_id,
         "stage_key": stage_key, "stage": stage_key, "gpu": gpu,
         "started": iso_utc(end - seconds)},
        {"event": "close", "interval_id": interval_id, "ended": iso_utc(end),
         "status": status, "seconds": seconds},
    ]
    path = root / LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_open_interval(root: Path, *, run_id: str, stage_key: str, age_seconds: float,
                         gpu: bool = True) -> None:
    """An interval whose invocation was killed before it could close the row."""
    path = root / LEDGER_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "event": "open", "interval_id": f"open-{stage_key}-{run_id}",
            "run_id": run_id, "stage_key": stage_key, "stage": stage_key, "gpu": gpu,
            "started": iso_utc(time.time() - age_seconds)}, ensure_ascii=False) + "\n")


def satisfy_through_build_retrieval(root: Path, overlay: dict | None = None) -> None:
    """Rows + manifests so the first stage the plan wants to run is train_reader (P3)."""
    sha = config_sha_for("pilot", overlay) if overlay else pilot_config_sha()
    rows = []
    for key, stage, run_id, rel in (
            ("download", "P1_download", "p1-download-pilot-1", "data/raw/raw.parquet"),
            ("audit_split", "P1_audit_split", "p1-audit-split-pilot-1",
             "data/processed/qids.txt"),
            ("build_retrieval", "P2_retrieval", "p2-retrieval-pilot-1",
             "data/cache/bank.npz")):
        row, meta = succeeded_row(root, stage, run_id, rel)
        rows.append(row)
        write_manifest(root, key, run_id, {rel: meta}, sha,
                       overlay=overlay, base_config_sha=sha)
    write_registry(root, rows)


GPU_STUB = """\
import argparse
import time
from pathlib import Path

from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ran.flag").write_text("ran", encoding="utf-8")
    time.sleep(0.3)  # stands in for a real GPU allocation
    artifact = out / "model.bin"
    artifact.write_bytes(b"gpu-stub")
    RunRegistry(repo / "outputs/runs/index.jsonl").append(
        run_id="p3-reader-stub-1", stage="P3_reader", status="SUCCEEDED",
        profile=args.profile,
        artifacts={str(artifact): {"sha256": sha256_file(artifact),
                                   "bytes": artifact.stat().st_size}})


if __name__ == "__main__":
    main()
"""

GPU_FAIL_STUB = """\
import argparse
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ran.flag").write_text("ran", encoding="utf-8")
    time.sleep(0.3)  # a real GPU allocation that then dies
    print("stub train_reader: simulated failure", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
"""

BLOCKING_STUB = """\
import argparse
import time
from pathlib import Path

from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    ready = repo / "data/raw/.stub_ready"
    release = repo / "data/raw/.stub_release"
    ready.parent.mkdir(parents=True, exist_ok=True)
    ready.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 30
    while not release.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    artifact = repo / "data/raw/raw.parquet"
    artifact.write_bytes(b"stub-download")
    RunRegistry(repo / "outputs/runs/index.jsonl").append(
        run_id="p1-download-stub-1", stage="P1_download", status="SUCCEEDED",
        profile=args.profile,
        artifacts={str(artifact): {"sha256": sha256_file(artifact),
                                   "bytes": artifact.stat().st_size}})


if __name__ == "__main__":
    main()
"""


def test_runs_only_overlay_cannot_split_lock_or_gpu_ledger(tmp_path):
    root = make_repo(tmp_path)
    (root / "src/pipeline/download.py").write_text(BLOCKING_STUB, encoding="utf-8")
    alternate = tmp_path / "runs_only.yaml"
    write_yaml(alternate, {"paths": {"runs": "outputs/alternate/runs"}})
    first = subprocess.Popen(
        ["bash", str(root / "scripts/run_all.sh"), "--profile", "pilot",
         "--stop-after", "download"], cwd=root, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, env=script_env())
    try:
        ready = root / "data/raw/.stub_ready"
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline and first.poll() is None:
            time.sleep(0.05)
        assert ready.exists(), "first invocation never reached its blocking stage"
        second = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                            str(alternate), "--stop-after", "download")
        assert second.returncode == 3, second.stdout + second.stderr
        assert not (root / "outputs/alternate/runs/index.jsonl").exists()
        assert not (root / "outputs/manifests/run_all/download.json").exists()
    finally:
        (root / "data/raw/.stub_release").write_text("go", encoding="utf-8")
        try:
            out, err = first.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            first.kill()
            out, err = first.communicate()
            pytest.fail(f"first invocation did not finish: {out}{err}")
    assert first.returncode == 0, out + err

    # Different namespaces must still charge the same device-wide allocation ledger.
    assert (root / LEDGER_REL).is_file()
    assert not (root / "outputs/alternate/runs/gpu_ledger.jsonl").exists()


def test_two_invocations_are_exclusive_and_dry_run_stays_read_only(tmp_path):
    root = make_repo(tmp_path)
    (root / "src" / "pipeline" / "download.py").write_text(BLOCKING_STUB, encoding="utf-8")
    run1 = subprocess.Popen(
        ["bash", str(root / "scripts" / "run_all.sh"), "--profile", "pilot",
         "--stop-after", "download"],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=script_env())
    ready = root / "data/raw/.stub_ready"
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and time.monotonic() < deadline and run1.poll() is None:
            time.sleep(0.05)
        if not ready.exists():
            out, err = run1.communicate(timeout=10)
            pytest.fail(f"first invocation never reached its blocking stage:\n{out}{err}")

        # while the first invocation owns the namespace, --dry-run must still work and
        # must not create, remove or modify anything.
        before = tree(root)
        dry = run_script(root, "run_all.sh", "--profile", "pilot", "--dry-run")
        assert dry.returncode == 0, dry.stdout + dry.stderr
        assert list(plan_lines(dry.stdout)) == DAG
        assert tree(root) == before

        # a second real invocation must refuse before writing anything.
        registry_before = (root / "outputs" / "runs" / "index.jsonl").read_bytes()
        second = run_script(root, "run_all.sh", "--profile", "pilot",
                            "--stop-after", "download")
        assert second.returncode == 3, second.stdout + second.stderr
        combined = (second.stdout + second.stderr).lower()
        assert any(token in combined for token in ("lock", "khoá", "khóa", "độc quyền",
                                                   "namespace"))
        assert (root / "outputs" / "runs" / "index.jsonl").read_bytes() == registry_before
    finally:
        (root / "data/raw/.stub_release").write_text("go", encoding="utf-8")
        try:
            out, err = run1.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            run1.kill()
            out, err = run1.communicate()
            pytest.fail("first invocation did not finish after release")
    assert run1.returncode == 0, out + err

    # the lock is free again after a completed invocation
    third = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "download")
    assert third.returncode == 0, third.stdout + third.stderr


def test_new_experiment_gpu_run_all_requires_bounded_runner(tmp_path):
    root = make_repo(tmp_path)
    overlay = root / "experiment.yaml"
    write_yaml(overlay, {"paths": {"runs": "outputs/experiments/probe/runs",
                                   "manifests": "outputs/experiments/probe/manifests",
                                   "diagnostics": "outputs/experiments/probe/diagnostics"},
                         "limits": {"max_gpu_hours": 48}})
    cfg = read_yaml(overlay)
    satisfy_through_build_retrieval(root, cfg)
    (root / "src/pipeline/train_reader.py").write_text(GPU_STUB, encoding="utf-8")
    result = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                        str(overlay), "--stop-after", "train_reader")
    assert result.returncode != 0
    assert "watchdog" in (result.stdout + result.stderr).lower()
    assert not (root / "outputs/models/reader_stub/ran.flag").exists()
    assert not (root / LEDGER_REL).exists()
    assert not (root / "outputs/experiments/probe/runs/index.jsonl").exists()


def test_experiment_report_path_also_blocks_run_all_before_side_effects(tmp_path):
    root = make_repo(tmp_path)
    overlay = root / "report_only.yaml"
    write_yaml(overlay, {"paths": {"reports": "outputs/experiments/probe/reports"}})
    (root / "src/pipeline/download.py").write_text(STUB_STAGE, encoding="utf-8")
    result = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                        str(overlay), "--stop-after", "download")
    assert result.returncode == 3
    assert not (root / "data/raw/raw.parquet").exists()
    assert not (root / LEDGER_REL).exists()


def test_cumulative_gpu_ledger_blocks_the_next_gpu_stage(tmp_path):
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    (root / "src" / "pipeline" / "train_reader.py").write_text(GPU_STUB, encoding="utf-8")
    first = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after",
                       "train_reader")
    assert first.returncode == 0, first.stdout + first.stderr
    marker = root / "outputs/models/reader_stub/ran.flag"
    assert marker.is_file()

    # invocation 1 billed the GPU stage it launched
    intervals = [i for i in ledger_intervals(root)
                 if i["open"].get("stage_key") == "train_reader"]
    assert intervals and intervals[0]["open"].get("gpu") is True
    assert intervals[0]["close"] is not None
    assert float(intervals[0]["close"]["seconds"]) >= 0.2

    # simulate an earlier invocation that already spent 13 h of the 12 h budget
    append_ledger_interval(root, run_id="old-run", stage_key="train_laya_ce",
                           seconds=13 * 3600)
    assert gpu_seconds_from_ledger(root) > 12 * 3600

    # force train_reader to want to run again, then prove it is blocked before launch
    (root / "outputs/models/reader_stub/model.bin").unlink()
    marker.unlink()
    second = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after",
                        "train_reader")
    assert second.returncode != 0, second.stdout + second.stderr
    assert not marker.exists(), "GPU stage launched despite an exhausted budget"
    combined = (second.stdout + second.stderr).lower()
    assert any(token in combined for token in ("ngân sách", "budget", "vượt", "hạn mức"))


def test_failed_gpu_stage_still_accounts_its_allocated_time(tmp_path):
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    (root / "src" / "pipeline" / "train_reader.py").write_text(GPU_FAIL_STUB,
                                                               encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after",
                      "train_reader")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert (root / "outputs/models/reader_stub/ran.flag").is_file()
    intervals = [i for i in ledger_intervals(root)
                 if i["open"].get("stage_key") == "train_reader"]
    assert intervals, "the failed GPU allocation was not recorded"
    close = intervals[0]["close"]
    assert close is not None and close["status"] == "FAILED"
    assert float(close["seconds"]) >= 0.2


def test_malformed_ledger_fails_closed_before_launching_gpu(tmp_path):
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    (root / "src" / "pipeline" / "train_reader.py").write_text(GPU_STUB, encoding="utf-8")
    ledger = root / LEDGER_REL
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{not json}\n", encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after",
                      "train_reader")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (root / "outputs/models/reader_stub/ran.flag").exists()


def test_open_interval_from_a_previous_invocation_fails_closed(tmp_path):
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    (root / "src" / "pipeline" / "train_reader.py").write_text(GPU_STUB, encoding="utf-8")
    append_open_interval(root, run_id="crashed-run", stage_key="train_laya_ce",
                         age_seconds=13 * 3600)
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after",
                      "train_reader")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (root / "outputs/models/reader_stub/ran.flag").exists()


def test_cpu_only_stage_spends_no_gpu_time(tmp_path):
    root = make_repo(tmp_path)
    (root / "src" / "pipeline" / "download.py").write_text(STUB_STAGE, encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "download")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert gpu_seconds_from_ledger(root) == 0.0


# --------------------------------------- --config overlay forwarding to stage children
#
# Regression: run_all resolved its own registry/manifests/diagnostics from the --config
# base overlay but launched every stage child without it, so an experiment-path overlay
# (paths.runs/models) made the children write historical artifacts while the orchestrator
# believed it was isolated. Every child must receive the base overlay merged under the
# stage's own overlay (a previous retry's reduced-resource knobs win on conflicts), and
# the recorded config sha / stage-manifest gating must stay deterministic.

NAMESPACE_STUB = """\
import argparse
from pathlib import Path

from src.utils.config import config_from_args, resolve
from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry

STAGE, RUN_ID = "{stage}", "{run_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args, _ = parser.parse_known_args()
    cfg = config_from_args(args)
    runs, models = resolve(cfg, "runs"), resolve(cfg, "models")
    print(f"RESOLVED runs={{runs}} models={{models}}", flush=True)
    artifact = models / f"{{RUN_ID}}/artifact.bin"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(f"{{RUN_ID}}".encode())
    RunRegistry(runs / "index.jsonl").append(
        run_id=RUN_ID, stage=STAGE, status="SUCCEEDED", profile=args.profile,
        artifacts={{str(artifact): {{"sha256": sha256_file(artifact),
                                    "bytes": artifact.stat().st_size}}}})


if __name__ == "__main__":
    main()
"""

EXP1_OVERLAY = {
    "paths": {
        "runs": "outputs/exp1/runs",
        "models": "outputs/exp1/models",
        "manifests": "outputs/exp1/manifests",
        "diagnostics": "outputs/exp1/diagnostics",
    },
}


def config_sha_for(profile: str, overlay: dict) -> str:
    """The config sha the scripts compute for a profile + overlay (byte-identical copies)."""
    cfg = load_config(profile, extra=overlay)
    return sha256_text(json.dumps(cfg, ensure_ascii=False, sort_keys=True, indent=2))


def test_config_overlay_reaches_every_stage_child_in_its_namespace(tmp_path):
    """The child resolves its own config from --config: it must see the experiment
    namespace and never touch the historical outputs; the manifest records the merged
    overlay and its config sha, and the gate round-trips deterministically on it."""
    root = make_repo(tmp_path)
    for key, stage, run_id in (("download", "P1_download", "p1-download-stub-1"),
                               ("audit_split", "P1_audit_split", "p1-audit-split-stub-1")):
        (root / "src" / "pipeline" / f"{key}.py").write_text(
            NAMESPACE_STUB.format(stage=stage, run_id=run_id), encoding="utf-8")
    overlay_path = tmp_path / "exp1.yaml"
    write_yaml(overlay_path, EXP1_OVERLAY)

    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(overlay_path),
                      "--stop-after", "audit_split")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    exp_runs = root / "outputs" / "exp1" / "runs"
    exp_models = root / "outputs" / "exp1" / "models"
    assert proc.stdout.count(f"RESOLVED runs={exp_runs} models={exp_models}") == 2, (
        "a stage child did not resolve the overlay's paths (--config was not forwarded)")
    assert (exp_models / "p1-download-stub-1/artifact.bin").is_file()
    assert (exp_models / "p1-audit-split-stub-1/artifact.bin").is_file()

    # every launch hands the child the resolved overlay itself
    launch = sorted((root / "outputs" / "exp1" / "diagnostics").rglob("*.overlay.yaml"))
    assert [p.name for p in launch] == [
        "01_download_r1.overlay.yaml", "02_audit_split_r1.overlay.yaml"]
    assert all(read_yaml(p) == EXP1_OVERLAY for p in launch)

    # the orchestrator's own writes (registry rows, stage manifests) are in the namespace
    rows = [json.loads(line) for line in
            (exp_runs / "index.jsonl").read_text("utf-8").splitlines() if line.strip()]
    assert {row.get("stage") for row in rows if row.get("status") == "SUCCEEDED"} == {
        "P1_download", "P1_audit_split", "run_all"}
    for key, run_id in (("download", "p1-download-stub-1"),
                        ("audit_split", "p1-audit-split-stub-1")):
        manifest = json.loads(
            (root / "outputs/exp1/manifests/run_all" / f"{key}.json").read_text("utf-8"))
        assert manifest["overlay"] == EXP1_OVERLAY
        assert manifest["base_config_sha"] == config_sha_for("pilot", EXP1_OVERLAY)
        assert manifest["config_sha"] == config_sha_for("pilot", EXP1_OVERLAY)
        assert list(manifest["outputs_sha"]) == [f"outputs/exp1/models/{run_id}/artifact.bin"]

    # nothing was written into the historical namespace the overlay moved away from
    assert not (root / "outputs" / "runs" / "index.jsonl").exists()
    assert not (root / "outputs" / "models").exists()
    assert not list((root / "outputs" / "manifests" / "run_all").iterdir())

    # the same overlay gates identically on a second invocation (deterministic round trip)
    again = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(overlay_path),
                       "--dry-run")
    assert again.returncode == 0, again.stdout + again.stderr
    plan = plan_lines(again.stdout)
    assert plan["download"][0] == "BỎ QUA" and plan["audit_split"][0] == "BỎ QUA"
    assert "đã xong p1-audit-split-stub-1" in plan["audit_split"][1]


def test_changing_config_in_same_registry_reruns_into_requested_namespace(tmp_path):
    root = make_repo(tmp_path)
    for key, stage in (("download", "P1_download"), ("audit_split", "P1_audit_split")):
        (root / "src" / "pipeline" / f"{key}.py").write_text(
            NAMESPACE_STUB.format(stage=stage, run_id=f"{key}-stub"), encoding="utf-8")
    first_path, second_path = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first = {"paths": {"models": "outputs/first/models"}}
    second = {"paths": {"models": "outputs/second/models"}}
    write_yaml(first_path, first)
    write_yaml(second_path, second)
    initial = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                         str(first_path), "--stop-after", "audit_split")
    assert initial.returncode == 0, initial.stdout + initial.stderr

    preview = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                         str(second_path), "--dry-run")
    assert preview.returncode == 0, preview.stdout + preview.stderr
    assert plan_lines(preview.stdout)["download"][0] == "CHẠY"
    assert plan_lines(preview.stdout)["audit_split"][0] == "CHẠY"

    changed = run_script(root, "run_all.sh", "--profile", "pilot", "--config",
                         str(second_path), "--stop-after", "audit_split")
    assert changed.returncode == 0, changed.stdout + changed.stderr
    second_models = root / "outputs/second/models"
    assert changed.stdout.count(f"models={second_models}") == 2
    assert (second_models / "download-stub/artifact.bin").is_file()
    assert (second_models / "audit_split-stub/artifact.bin").is_file()
    assert (root / "outputs/first/models/download-stub/artifact.bin").is_file()
    assert (root / "outputs/first/models/audit_split-stub/artifact.bin").is_file()


def test_empty_config_file_with_seed_is_a_valid_overlay(tmp_path):
    root = make_repo(tmp_path)
    empty = tmp_path / "empty.yaml"
    empty.write_text("# deliberately empty\n", encoding="utf-8")
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(empty),
                      "--seed", "7", "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert list(plan_lines(proc.stdout)) == DAG


def test_changing_seed_invalidates_stage_manifest(tmp_path):
    root = make_repo(tmp_path)
    (root / "src/pipeline/download.py").write_text(STUB_STAGE, encoding="utf-8")
    initial = run_script(root, "run_all.sh", "--profile", "pilot", "--seed", "1",
                         "--stop-after", "download")
    assert initial.returncode == 0, initial.stdout + initial.stderr
    same = run_script(root, "run_all.sh", "--profile", "pilot", "--seed", "1",
                      "--dry-run")
    assert plan_lines(same.stdout)["download"][0] == "BỎ QUA"
    changed = run_script(root, "run_all.sh", "--profile", "pilot", "--seed", "2",
                         "--dry-run")
    assert plan_lines(changed.stdout)["download"][0] == "CHẠY"


# ----------------------------------------------------- regressions: retry, recovery, P6 GPU
#
# Review findings covered here (tests must fail until the orchestrator/ledger fix lands):
# * an OOM retry is a new GPU launch: when the failed attempt already reached
#   limits.max_gpu_hours, run_all must not relaunch it;
# * a GPU interval left open by a killed invocation must be closed by --resume itself,
#   conservatively and fail-closed: only when pgrep definitively proves the stage process
#   is gone (exit 1), and at the recovery instant — an upper bound that charges the idle
#   downtime and never underbills (a silent job's stage-log mtime freezes at launch and
#   must never be read as the stop time); an uncertain liveness oracle (pgrep missing or
#   exiting with a status other than 0/1) leaves the interval open and blocks the next
#   GPU stage — never by hand-editing the ledger;
# * train_rl pre-warms Laya, which calls the model on the GPU when its probability cache
#   misses, so the launch must be billed and budget-guarded as a GPU stage;
# * the ledger must reject tampered close rows and unusable limits (tests/test_runs.py).

OOM_RETRY_STUB = """\
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "attempts.txt").open("a", encoding="utf-8") as fh:
        fh.write("attempt\\n")
    time.sleep(0.5)  # a real GPU allocation that OOMs every time
    print("CUDA out of memory (simulated)", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
"""

BLOCKING_GPU_STUB = """\
import argparse
import time
from pathlib import Path

from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    (out / "killed.flag").write_text("ran", encoding="utf-8")
    RunRegistry(repo / "outputs/runs/index.jsonl").append(
        run_id="p3-reader-killed-1", stage="P3_reader", status="RUNNING",
        profile=args.profile)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        (out / "heartbeat").write_text("alive", encoding="utf-8")
        print("heartbeat", flush=True)
        time.sleep(0.1)


if __name__ == "__main__":
    main()
"""

SILENT_GPU_STUB = """\
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    (out / "silent.flag").write_text("started", encoding="utf-8")
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        time.sleep(0.1)  # holds the "GPU" silently: not one log line after launch


if __name__ == "__main__":
    main()
"""

TRAIN_RL_STUB = """\
import argparse
from pathlib import Path

from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default=None)
    args, _ = parser.parse_known_args()
    repo = Path(__file__).resolve().parents[2]
    out = repo / "outputs/models/rl_stub"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / "policy.bin"
    artifact.write_bytes(b"rl-stub")
    RunRegistry(repo / "outputs/runs/index.jsonl").append(
        run_id="p6-rl-stub-1", stage="P6_rl", status="SUCCEEDED",
        profile=args.profile,
        artifacts={str(artifact): {"sha256": sha256_file(artifact),
                                   "bytes": artifact.stat().st_size}})


if __name__ == "__main__":
    main()
"""


@pytest.mark.parametrize("cap", [None, 0, 49, float("nan")])
def test_run_all_refuses_invalid_or_unauthorized_gpu_cap_before_any_write(tmp_path, cap):
    root = make_repo(tmp_path)
    (root / "src/pipeline/download.py").write_text(STUB_STAGE, encoding="utf-8")
    overlay = tmp_path / "invalid_budget.yaml"
    write_yaml(overlay, {"limits": {"max_gpu_hours": cap}})

    result = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(overlay),
                        "--stop-after", "download")

    assert result.returncode != 0
    assert "max_gpu_hours" in (result.stdout + result.stderr)
    assert not (root / "data/raw/raw.parquet").exists()
    assert registry(root) == []


def test_resume_recovers_cpu_interval_with_invalid_budget_but_launches_nothing(tmp_path):
    root = make_repo(tmp_path)
    write_registry(root, [{"run_id": "crashed-run", "stage": "run_all", "status": "RUNNING",
                           "profile": "pilot", "event": "start",
                           "ts": "2026-10-07T00:00:00+00:00"}])
    append_open_interval(root, run_id="crashed-run", stage_key="download",
                         age_seconds=1, gpu=False)
    overlay = tmp_path / "invalid_resume_budget.yaml"
    write_yaml(overlay, {"limits": {"max_gpu_hours": None}})

    result = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(overlay),
                        "--resume", "--stop-after", "download")

    assert result.returncode == 3
    recovered = [i for i in ledger_intervals(root)
                 if i["open"]["interval_id"] == "open-download-crashed-run"][0]
    assert recovered["close"]["status"] == "INTERRUPTED"
    assert not (root / "data/raw/raw.parquet").exists()
    assert "không chạy stage mới" in result.stderr


def test_oom_retry_is_refused_once_the_failed_attempt_exhausted_the_budget(tmp_path):
    root = make_repo(tmp_path)
    budget_overlay = {"limits": {"max_gpu_hours": 0.00005}}
    satisfy_through_build_retrieval(root, budget_overlay)
    (root / "src" / "pipeline" / "train_reader.py").write_text(OOM_RETRY_STUB,
                                                               encoding="utf-8")
    overlay = tmp_path / "tiny_gpu_budget.yaml"
    write_yaml(overlay, budget_overlay)  # 0.18 s
    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(overlay),
                      "--stop-after", "train_reader")

    attempts = (root / "outputs/models/reader_stub/attempts.txt").read_text(encoding="utf-8")
    assert attempts.splitlines() == ["attempt"], (
        "the OOM retry relaunched a GPU stage whose failed attempt had already spent the "
        "whole budget")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not list((root / "outputs" / "diagnostics").rglob("*_r2.log")), (
        "a second attempt log exists even though the retry was over budget")
    combined = (proc.stdout + proc.stderr).lower()
    assert any(token in combined for token in ("ngân sách", "budget", "vượt", "hạn mức"))


OOM_ONCE_STUB = """\
import argparse
from pathlib import Path

from src.utils.config import config_from_args, resolve
from src.utils.hashing import sha256_file
from src.utils.runs import RunRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args, _ = parser.parse_known_args()
    cfg = config_from_args(args)
    runs = resolve(cfg, "runs")
    out = resolve(cfg, "models") / "reader_stub"
    out.mkdir(parents=True, exist_ok=True)
    print(f"RESOLVED runs={runs} batch_size={cfg['reader']['batch_size']} "
          f"epochs={cfg['reader']['epochs']}", flush=True)
    attempts_file = out / "attempts.txt"
    attempts = int(attempts_file.read_text(encoding="utf-8")) if attempts_file.is_file() else 0
    attempts_file.write_text(str(attempts + 1), encoding="utf-8")
    if attempts == 0:
        print("CUDA out of memory (simulated)", flush=True)
        raise SystemExit(1)
    artifact = out / "model.bin"
    artifact.write_bytes(b"reader-stub")
    RunRegistry(runs / "index.jsonl").append(
        run_id="p3-reader-stub-1", stage="P3_reader", status="SUCCEEDED",
        profile=args.profile, artifacts={str(artifact): {
            "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}})


if __name__ == "__main__":
    main()
"""


def test_oom_retry_launch_merges_the_base_overlay_under_the_generated_overlay(tmp_path):
    """A retry is a stage launch too: the generated OOM overlay must be merged over the
    CLI base overlay (the reduced knob wins, the rest of the base survives), and the
    manifest must record that merged overlay with its deterministic config sha."""
    root = make_repo(tmp_path)
    base = {"reader": {"batch_size": 8, "epochs": 1}}
    satisfy_through_build_retrieval(root, base)
    (root / "src" / "pipeline" / "train_reader.py").write_text(OOM_ONCE_STUB,
                                                               encoding="utf-8")
    base_path = tmp_path / "reader_base.yaml"
    write_yaml(base_path, base)

    proc = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(base_path),
                      "--stop-after", "train_reader")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (root / "outputs/models/reader_stub/attempts.txt").read_text("utf-8") == "2"
    assert "batch_size=8 epochs=1" in proc.stdout  # attempt 1 got the base overlay
    assert "batch_size=4 epochs=1" in proc.stdout  # attempt 2: OOM knob over the base

    merged = {"reader": {"batch_size": 4, "epochs": 1}}
    first = list((root / "outputs" / "diagnostics").rglob("*_train_reader_r1.overlay.yaml"))
    assert len(first) == 1 and read_yaml(first[0]) == base
    retry = list((root / "outputs" / "diagnostics").rglob("*_train_reader_r2.overlay.yaml"))
    assert len(retry) == 1 and read_yaml(retry[0]) == merged, (
        "the retry launch did not receive the merged base + OOM overlay")
    manifest = json.loads(
        (root / "outputs/manifests/run_all/train_reader.json").read_text("utf-8"))
    assert manifest["overlay"] == merged
    assert manifest["base_config_sha"] == config_sha_for("pilot", base)
    assert manifest["config_sha"] == config_sha_for("pilot", merged)

    again = run_script(root, "run_all.sh", "--profile", "pilot", "--config", str(base_path),
                       "--dry-run")
    assert again.returncode == 0, again.stdout + again.stderr
    plan = plan_lines(again.stdout)
    assert plan["train_reader"][0] == "BỎ QUA"
    assert "overlay giảm tài nguyên" in plan["train_reader"][1]


def test_resume_recovers_killed_cpu_interval_without_gpu_charge(tmp_path):
    root = make_repo(tmp_path)
    rows = registry(root)
    rows.append({"run_id": "crashed-run", "stage": "run_all", "status": "RUNNING",
                 "profile": "pilot", "event": "start", "ts": "2026-10-07T00:00:00+00:00"})
    row, artifact = succeeded_row(root, "P1_download", "download-pilot-1", "data/raw/raw.parquet")
    rows.append(row)
    write_registry(root, rows)
    write_manifest(root, "download", "download-pilot-1", {"data/raw/raw.parquet": artifact},
                   pilot_config_sha())
    append_open_interval(root, run_id="crashed-run", stage_key="download",
                         age_seconds=1, gpu=False)
    resumed = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                         "--stop-after", "download")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    recovered = [i for i in ledger_intervals(root)
                 if i["open"].get("interval_id") == "open-download-crashed-run"][0]
    assert recovered["close"]["status"] == "INTERRUPTED"
    assert gpu_seconds_from_ledger(root) == 0


def test_resume_refuses_cpu_stage_when_prior_child_is_still_running(tmp_path):
    root = make_repo(tmp_path)
    (root / "src/pipeline/download.py").write_text("import time\ntime.sleep(60)", encoding="utf-8")
    rows = [{"run_id": "crashed-run", "stage": "run_all", "status": "RUNNING",
             "profile": "pilot", "event": "start", "ts": "2026-10-07T00:00:00+00:00"}]
    write_registry(root, rows)
    append_open_interval(root, run_id="crashed-run", stage_key="download",
                         age_seconds=1, gpu=False)
    live = subprocess.Popen([sys.executable, "-m", "src.pipeline.download", "--profile", "pilot"],
                            cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True)
    try:
        # Exercise the startup window before the child has written its RUNNING registry row.
        result = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                            "--stop-after", "download")
        assert result.returncode != 0, result.stdout + result.stderr
        assert [i for i in ledger_intervals(root) if i["open"]["stage_key"] == "download"
                and i["open"]["run_id"] != "crashed-run"] == []
    finally:
        if live.poll() is None:
            os.killpg(live.pid, signal.SIGKILL)
        live.wait(timeout=5)


def test_resume_recovers_a_killed_gpu_interval_conservatively(tmp_path):
    """The rerun closes the orphaned interval itself: only when the stage process is
    provably gone, at the recovery instant — conservative, charging kill->rerun idle."""
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    stub = root / "src" / "pipeline" / "train_reader.py"
    stub.write_text(BLOCKING_GPU_STUB, encoding="utf-8")
    killed_run = subprocess.Popen(
        ["bash", str(root / "scripts" / "run_all.sh"), "--profile", "pilot",
         "--stop-after", "train_reader"],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=script_env(), start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        opened = None
        while time.monotonic() < deadline:
            intervals = [i for i in ledger_intervals(root)
                         if i["open"].get("stage_key") == "train_reader"]
            if intervals and (root / "outputs/models/reader_stub/heartbeat").exists():
                opened = intervals[0]["open"]
                break
            if killed_run.poll() is not None:
                out, err = killed_run.communicate()
                pytest.fail(f"the to-be-killed invocation exited early "
                            f"(rc={killed_run.returncode}):\n{out}{err}")
            time.sleep(0.05)
        assert opened is not None, "no GPU interval was opened before the kill"
        time.sleep(1.2)  # the stage verifiably ran for over a second
        os.killpg(os.getpgid(killed_run.pid), signal.SIGTERM)
        try:
            killed_run.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(killed_run.pid), signal.SIGKILL)
            killed_run.wait(timeout=15)
    finally:
        if killed_run.poll() is None:
            os.killpg(os.getpgid(killed_run.pid), signal.SIGKILL)
            killed_run.wait(timeout=15)
    t_kill = time.time()
    assert [i for i in ledger_intervals(root) if i["close"] is None], (
        "fixture: the kill was expected to leave an open GPU interval")
    time.sleep(4.0)  # idle downtime that conservative recovery must also charge
    t_resume = time.time()

    stub.write_text(GPU_STUB, encoding="utf-8")  # the retried stage succeeds
    resumed = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                         "--stop-after", "train_reader")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (root / "outputs/models/reader_stub/ran.flag").is_file()
    recovered = [i for i in ledger_intervals(root)
                 if i["open"].get("interval_id") == opened["interval_id"]][0]
    close = recovered["close"]
    assert close is not None, ("--resume left the killed GPU interval open; the operator "
                               "must not have to edit the ledger by hand")
    assert close["status"] == "INTERRUPTED"
    started = datetime.fromisoformat(opened["started"]).timestamp()
    ended = datetime.fromisoformat(close["ended"]).timestamp()
    assert close["seconds"] >= 0.5, "the killed interval was closed as if it never ran"
    assert started <= ended
    # Conservative accounting: the close instant is the recovery NOW, so it may overbill
    # (it charges the kill->rerun idle downtime) but can never bill below the real run.
    assert ended >= t_resume, "recovery closed before the rerun even started"
    assert close["seconds"] >= (t_kill - started), "recovery billed less than the real run"
    assert close["seconds"] >= (t_resume - started), (
        "the idle downtime between the kill and the rerun was not charged")


def test_resume_never_underbills_a_silent_killed_gpu_interval(tmp_path):
    """A silent GPU job leaves the stage log's mtime frozen at launch, so closing at that
    mtime would bill ~0 s. Recovery must instead charge the whole launch->recovery window:
    at least the real run, and conservatively at least up to the recovery instant."""
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    stub = root / "src" / "pipeline" / "train_reader.py"
    stub.write_text(SILENT_GPU_STUB, encoding="utf-8")
    killed_run = subprocess.Popen(
        ["bash", str(root / "scripts" / "run_all.sh"), "--profile", "pilot",
         "--stop-after", "train_reader"],
        cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=script_env(), start_new_session=True)
    try:
        deadline = time.monotonic() + 30
        opened = None
        while time.monotonic() < deadline:
            intervals = [i for i in ledger_intervals(root)
                         if i["open"].get("stage_key") == "train_reader"]
            if intervals and (root / "outputs/models/reader_stub/silent.flag").exists():
                opened = intervals[0]["open"]
                break
            if killed_run.poll() is not None:
                out, err = killed_run.communicate()
                pytest.fail(f"the to-be-killed invocation exited early "
                            f"(rc={killed_run.returncode}):\n{out}{err}")
            time.sleep(0.05)
        assert opened is not None, "no GPU interval was opened before the kill"
        time.sleep(1.5)  # the silent job verifiably ran for over a second
        os.killpg(os.getpgid(killed_run.pid), signal.SIGTERM)
        try:
            killed_run.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(killed_run.pid), signal.SIGKILL)
            killed_run.wait(timeout=15)
    finally:
        if killed_run.poll() is None:
            os.killpg(os.getpgid(killed_run.pid), signal.SIGKILL)
            killed_run.wait(timeout=15)
    t_kill = time.time()
    started = datetime.fromisoformat(opened["started"]).timestamp()
    assert t_kill - started >= 1.0, "fixture: the interval did not really run a second"
    stage_log = next((root / "outputs" / "diagnostics").rglob("*_train_reader_r1.log"))
    assert t_kill - stage_log.stat().st_mtime >= 1.0, (
        "fixture: the silent job's log was expected to stay frozen near launch")
    assert [i for i in ledger_intervals(root) if i["close"] is None], (
        "fixture: the kill was expected to leave an open GPU interval")
    time.sleep(4.0)  # idle downtime between the kill and the rerun
    t_resume = time.time()

    stub.write_text(GPU_STUB, encoding="utf-8")  # the retried stage succeeds
    resumed = run_script(root, "run_all.sh", "--profile", "pilot", "--resume",
                         "--stop-after", "train_reader")
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (root / "outputs/models/reader_stub/ran.flag").is_file()
    recovered = [i for i in ledger_intervals(root)
                 if i["open"].get("interval_id") == opened["interval_id"]][0]
    close = recovered["close"]
    assert close is not None, ("--resume left the silent killed interval open; the "
                               "operator must not have to edit the ledger by hand")
    assert close["status"] == "INTERRUPTED"
    ended = datetime.fromisoformat(close["ended"]).timestamp()
    assert close["seconds"] >= (t_kill - started), (
        "recovery billed less than the silent job actually ran (the log mtime says ~0 s; "
        "it must never be used as the stop time)")
    assert ended >= t_resume, "recovery closed before the rerun even started"
    assert close["seconds"] >= (t_resume - started), (
        "recovery underbilled: the kill->rerun downtime was dropped")


def test_uncertain_pgrep_leaves_the_interval_open_and_blocks_the_gpu_stage(tmp_path):
    """Fail closed on uncertainty: pgrep exiting with a status other than 0/1 proves
    neither liveness nor death, so the interval stays open and the pre-launch GPU gate
    refuses to start the stage."""
    root = make_repo(tmp_path)
    satisfy_through_build_retrieval(root)
    (root / "src" / "pipeline" / "train_reader.py").write_text(GPU_STUB, encoding="utf-8")
    rows = registry(root)
    rows.append({"run_id": "crashed-run", "stage": "run_all", "status": "RUNNING",
                 "profile": "pilot", "event": "start",
                 "log_dir": "outputs/diagnostics/run_all/crashed",
                 "ts": "2026-10-07T00:00:00+00:00"})
    write_registry(root, rows)
    append_open_interval(root, run_id="crashed-run", stage_key="train_reader",
                         age_seconds=60)
    # a log exists at the claimed log_dir, so an mtime-based close *could* have produced a
    # plausible stop time — liveness, not a log, is what must decide
    crashed_logs = root / "outputs" / "diagnostics" / "run_all" / "crashed"
    crashed_logs.mkdir(parents=True)
    (crashed_logs / "01_train_reader_r1.log").write_text("stale log\n", encoding="utf-8")
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    pgrep = fake_bin / "pgrep"
    pgrep.write_text("#!/bin/sh\nexit 2\n", encoding="utf-8")  # usage error: no answer
    pgrep.chmod(0o755)
    env = {**script_env(), "PATH": f"{fake_bin}:{os.environ['PATH']}"}
    proc = subprocess.run(
        ["bash", str(root / "scripts" / "run_all.sh"), "--profile", "pilot", "--resume",
         "--stop-after", "train_reader"],
        cwd=str(root), capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (root / "outputs/models/reader_stub/ran.flag").exists(), (
        "the GPU stage was launched although liveness could not be verified")
    assert [i for i in ledger_intervals(root) if i["close"] is None], (
        "an interval whose liveness could not be verified was closed anyway")
    assert "pgrep" in proc.stdout + proc.stderr  # the uncertainty is reported, not hidden


SATISFIED_BEFORE_TRAIN_RL = (
    ("download", "P1_download", {}, "data/raw/raw.parquet"),
    ("audit_split", "P1_audit_split", {}, "data/processed/qids.txt"),
    ("build_retrieval", "P2_retrieval", {}, "data/cache/bank.npz"),
    ("train_reader", "P3_reader", {}, "outputs/models/reader_prior/model.bin"),
    ("cache_reader", "P3_reader_cache", {"splits": ["train_strict", "val_strict"]},
     "outputs/state_cache/train_states.parquet"),
    ("make_teacher", "P5_teacher", {}, "data/teacher/states.jsonl"),
    ("train_laya_ce", "P5_laya_train", {"loss": "soft-ce"},
     "outputs/models/laya_ce/model.safetensors"),
    ("train_laya_rlcd", "P5_laya_train", {"loss": "rlcd"},
     "outputs/models/laya_rlcd/model.safetensors"),
    ("cache_laya", "P5_cache_laya", {"regime": "rlcd"},
     "outputs/models/laya_probs_cache/rlcd/laya_probs.parquet"),
)


def satisfy_stages_before_train_rl(root: Path) -> None:
    """Rows + manifests so the only stage the plan wants to run is train_rl (P6)."""
    sha = pilot_config_sha()
    rows = []
    for key, stage, extra, rel in SATISFIED_BEFORE_TRAIN_RL:
        run_id = f"{key}-pilot-1"
        row, meta = succeeded_row(root, stage, run_id, rel)
        row.update(extra)
        rows.append(row)
        write_manifest(root, key, run_id, {rel: meta}, sha)
    write_registry(root, rows)


def test_train_rl_is_billed_and_guarded_as_a_gpu_stage(tmp_path):
    """train_rl pre-warms Laya: a cache miss calls the model on the GPU (unbilled today)."""
    root = make_repo(tmp_path)
    satisfy_stages_before_train_rl(root)
    (root / "src" / "pipeline" / "train_rl.py").write_text(TRAIN_RL_STUB, encoding="utf-8")
    artifact = root / "outputs/models/rl_stub/policy.bin"

    first = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "train_rl")
    assert first.returncode == 0, first.stdout + first.stderr
    assert artifact.is_file()
    billed = [i for i in ledger_intervals(root) if i["open"].get("stage_key") == "train_rl"]
    assert billed, "the train_rl launch was not recorded in the GPU ledger"
    assert billed[0]["open"].get("gpu") is True, (
        "train_rl can call the Laya model on the GPU (cache miss) but was billed as CPU-only")

    artifact.unlink()  # force the gate to want train_rl again
    append_ledger_interval(root, run_id="old-run", stage_key="train_laya_ce",
                           seconds=13 * 3600)  # over limits.max_gpu_hours=12
    second = run_script(root, "run_all.sh", "--profile", "pilot", "--stop-after", "train_rl")
    assert second.returncode != 0, second.stdout + second.stderr
    assert not artifact.exists(), "train_rl launched although the GPU budget was exhausted"
    combined = (second.stdout + second.stderr).lower()
    assert any(token in combined for token in ("ngân sách", "budget", "vượt", "hạn mức"))


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

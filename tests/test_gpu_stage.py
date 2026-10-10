"""A direct GPU launch must share run_all's lock, ledger, cap and experiment paths."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.utils.runs import GpuLedger, OutputNamespaceLock
from tests.test_run_all import make_repo

REAL = Path(__file__).resolve().parents[1]
STAGE = "src.pipeline.train_gliner"


def setup(tmp_path: Path, code: str = "print('done')") -> Path:
    root = make_repo(tmp_path)
    shutil.copy2(REAL / "scripts/gpu_stage.py", root / "scripts/gpu_stage.py")
    (root / "src/pipeline/train_gliner.py").write_text(code, encoding="utf-8")
    for env in (".venv", ".venv-gliner"):
        (root / env / "bin").mkdir(parents=True)
        (root / env / "bin/python").symlink_to(sys.executable)
    overlay = root / "experiment.yaml"
    from src.utils.io import write_yaml
    keys = ("data_raw", "data_processed", "data_scenarios", "data_indexes", "data_cache",
            "data_diagnostics", "outputs", "models", "runs", "predictions", "manifests",
            "diagnostics", "reports", "plots", "tables")
    write_yaml(overlay, {"limits": {"max_gpu_hours": 0.05},
                         "paths": {key: f"outputs/experiments/probe/{key}" for key in keys}})
    return root


def cmd(root: Path, *options: str) -> list[str]:
    return [sys.executable, str(root / "scripts/gpu_stage.py"),
            "--profile", "pilot", "--config", str(root / "experiment.yaml"),
            "--module", STAGE, "--forecast-seconds", "1", "--timeout-seconds", "5",
            *options]


def run(root: Path, *options: str) -> subprocess.CompletedProcess:
    invocation = cmd(root)
    if options == ("--recover",):
        invocation.insert(2, "--recover")
    else:
        invocation += list(options)
    return subprocess.run(invocation, cwd=root, capture_output=True, text=True,
                          timeout=30, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


def ledger(root: Path) -> GpuLedger:
    return GpuLedger(root / "outputs/experiments/gpu_ledger.jsonl")


def test_success_and_failed_launch_are_both_billed(tmp_path):
    root = setup(tmp_path, "print('done')")
    assert run(root).returncode == 0
    first = ledger(root)._rows()
    assert [event["event"] for event in first] == ["open", "close"]
    assert first[1]["status"] == "SUCCEEDED" and first[1]["seconds"] > 0
    logs = list((root / "outputs/experiments/probe/diagnostics/gpu_stage").glob("*.log"))
    assert len(logs) == 1
    launched = json.loads(logs[0].read_text().splitlines()[0])["command"]
    assert launched[launched.index("--config") + 1].startswith(str(root / "outputs/experiments"))
    (root / "src/pipeline/train_gliner.py").write_text("raise RuntimeError('failure')")
    bad = run(root)
    assert bad.returncode != 0
    rows = ledger(root)._rows()
    assert [event["event"] for event in rows] == ["open", "close", "open", "close"]
    assert rows[-1]["status"] == "FAILED"
    assert ledger(root).spent_seconds() >= first[1]["seconds"]


def test_lock_and_budget_refuse_before_launch(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    with OutputNamespaceLock(root / "outputs/experiments/.run_all.lock"):
        blocked = run(root)
    assert blocked.returncode != 0 and not (root / "marker").exists()
    assert not ledger(root).path.exists()
    over = run(root, "--forecast-seconds", "180")
    assert over.returncode != 0 and not (root / "marker").exists()
    assert not ledger(root).path.exists()


def test_unwrapped_live_module_blocks_recovery_and_launch(tmp_path):
    root = setup(tmp_path, "import time\ntime.sleep(60)")
    plain = subprocess.Popen([sys.executable, "-m", STAGE], cwd=root,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
    try:
        interval = ledger(root).open_interval(run_id="direct-other", stage_key=STAGE, stage=STAGE)
        refusal = run(root, "--recover")
        assert refusal.returncode != 0 and ledger(root).open_intervals()[0]["interval_id"] == interval
    finally:
        os.killpg(plain.pid, signal.SIGKILL)
        plain.wait(timeout=5)
    # A terminated process group may still be visible briefly during kernel teardown.
    for _ in range(30):
        recovered = run(root, "--recover")
        if recovered.returncode == 0:
            break
        time.sleep(0.1)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr


def test_malformed_ledger_and_open_interval_fail_closed(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    ledger(root).path.parent.mkdir(parents=True, exist_ok=True)
    ledger(root).path.write_text("not json\n")
    assert run(root).returncode != 0
    assert not (root / "marker").exists()
    ledger(root).path.unlink()
    ledger(root).open_interval(run_id="direct-stale", stage_key="probe", stage=STAGE)
    assert run(root).returncode != 0
    assert not (root / "marker").exists()


def test_config_cannot_raise_device_ceiling_above_48_hours(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    from src.utils.io import read_yaml, write_yaml
    overlay = root / "experiment.yaml"
    cfg = read_yaml(overlay)
    cfg["limits"]["max_gpu_hours"] = 49
    write_yaml(overlay, cfg)
    result = run(root)
    assert result.returncode != 0
    assert "48" in (result.stdout + result.stderr)
    assert not (root / "marker").exists() and not ledger(root).path.exists()


def test_config_paths_are_isolated(tmp_path):
    root = setup(tmp_path)
    (root / "experiment.yaml").write_text("limits:\n  max_gpu_hours: 48\n")
    refused = run(root)
    assert refused.returncode != 0 and not ledger(root).path.exists()


def test_stage_args_cannot_override_isolated_config_or_profile(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    for override in ("--config=/tmp/historical.yaml", "--profile=full"):
        result = run(root, "--", override)
        assert result.returncode != 0 and not (root / "marker").exists()
    assert not ledger(root).path.exists()


def test_stage_args_cannot_override_training_paths_or_force_artifacts(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    for override in ("--out", "--data=/tmp/evil", "--base", "--force", "--dev", "--reselect"):
        result = run(root, "--", override, "/tmp/evil")
        assert result.returncode != 0, override
    assert not (root / "marker").exists() and not ledger(root).path.exists()
    valid = run(root, "--", "--loss", "soft-ce")
    assert valid.returncode == 0, valid.stdout + valid.stderr


def test_watchdog_cannot_exceed_remaining_budget(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    assert run(root, "--timeout-seconds", "160").returncode != 0
    assert not (root / "marker").exists() and not ledger(root).path.exists()


def test_mixed_namespace_and_preflight_are_refused(tmp_path):
    root = setup(tmp_path)
    from src.utils.io import read_yaml, write_yaml
    config = root / "experiment.yaml"
    overlay = read_yaml(config)
    overlay["paths"]["models"] = "outputs/experiments/other/models"
    write_yaml(config, overlay)
    assert run(root).returncode != 0
    write_yaml(config, {**overlay, "paths": {**overlay["paths"],
                                             "models": "outputs/experiments/probe/models"}})
    invocation = cmd(root)
    invocation[invocation.index(STAGE)] = "src.pipeline.preflight"
    result = subprocess.run(invocation, cwd=root, capture_output=True, text=True)
    assert result.returncode != 0 and not ledger(root).path.exists()


def test_cpu_teacher_launch_is_namespaced_and_not_billed(tmp_path):
    root = setup(tmp_path)
    (root / "src/pipeline/make_teacher.py").write_text(
        "import argparse\nfrom pathlib import Path\n"
        "p = argparse.ArgumentParser()\np.add_argument('--profile')\np.add_argument('--config')\n"
        "args = p.parse_args()\n"
        "Path(args.config).with_suffix('.seen').write_text(args.profile)\n",
        encoding="utf-8",
    )
    command = cmd(root, "--cpu-stage")
    command[command.index(STAGE)] = "src.pipeline.make_teacher"
    command[command.index("--forecast-seconds"):command.index("--timeout-seconds")] = []
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    events = ledger(root)._rows()
    assert events[0]["gpu"] is False and events[-1]["status"] == "SUCCEEDED"
    assert ledger(root).spent_seconds() == 0
    logs = list((root / "outputs/experiments/probe/diagnostics/cpu_stage").glob("*.log"))
    assert len(logs) == 1
    info = json.loads(logs[0].read_text().splitlines()[0])
    assert info["command"][info["command"].index("--config") + 1].startswith(
        str(root / "outputs/experiments/probe")
    )
    assert len(list((root / "outputs/experiments/probe/diagnostics/cpu_stage").glob("*.seen"))) == 1


def test_cpu_stage_rejects_gpu_modules_and_unapproved_options(tmp_path):
    root = setup(tmp_path, "from pathlib import Path\nPath('marker').write_text('launched')")
    cases = [cmd(root, "--cpu-stage"), cmd(root, "--cpu-stage", "--", "--split", "val_strict")]
    cases[1][cases[1].index(STAGE)] = "src.pipeline.make_teacher"
    cases[1][cases[1].index("--forecast-seconds"):cases[1].index("--timeout-seconds")] = []
    (root / "src/pipeline/make_teacher.py").write_text("print('should not run')")
    for invocation in cases:
        result = subprocess.run(invocation, cwd=root, capture_output=True, text=True)
        assert result.returncode != 0
    assert not (root / "marker").exists() and not ledger(root).path.exists()


def test_cpu_stage_failure_and_timeout_record_status_without_gpu_charge(tmp_path):
    root = setup(tmp_path)
    module = root / "src/pipeline/make_teacher.py"
    command = cmd(root, "--cpu-stage")
    command[command.index(STAGE)] = "src.pipeline.make_teacher"
    command[command.index("--forecast-seconds"):command.index("--timeout-seconds")] = []
    module.write_text("raise RuntimeError('failure')", encoding="utf-8")
    failed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert failed.returncode != 0 and ledger(root)._rows()[-1]["status"] == "FAILED"
    module.write_text("import time\ntime.sleep(60)", encoding="utf-8")
    command[command.index("--timeout-seconds") + 1] = "0.2"
    timed_out = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=20)
    assert timed_out.returncode != 0 and ledger(root)._rows()[-1]["status"] == "FAILED"
    assert ledger(root).spent_seconds() == 0


def test_dead_cpu_interval_requires_recovery_before_next_stage(tmp_path):
    root = setup(tmp_path)
    (root / "src/pipeline/make_teacher.py").write_text("print('done')", encoding="utf-8")
    pending = ledger(root).open_interval(run_id="direct-stale", stage_key="src.pipeline.make_teacher",
                                         stage="src.pipeline.make_teacher", gpu=False)
    command = cmd(root, "--cpu-stage")
    command[command.index(STAGE)] = "src.pipeline.make_teacher"
    command[command.index("--forecast-seconds"):command.index("--timeout-seconds")] = []
    blocked = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
    assert blocked.returncode != 0 and ledger(root).open_intervals()[0]["interval_id"] == pending
    recovered = subprocess.run([*command[:2], "--recover", *command[2:]], cwd=root,
                               capture_output=True, text=True, timeout=30)
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert ledger(root)._rows()[-1]["status"] == "INTERRUPTED"
    assert ledger(root).spent_seconds() == 0
    assert subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30).returncode == 0


def test_run_all_cpu_interval_does_not_block_direct_stages(tmp_path):
    root = setup(tmp_path)
    (root / "src/pipeline/make_teacher.py").write_text("print('done')", encoding="utf-8")
    pending = ledger(root).open_interval(run_id="run_all-stale", stage_key="download",
                                         stage="src.pipeline.download", gpu=False)
    command = cmd(root, "--cpu-stage")
    command[command.index(STAGE)] = "src.pipeline.make_teacher"
    command[command.index("--forecast-seconds"):command.index("--timeout-seconds")] = []
    assert subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30).returncode == 0
    assert ledger(root).open_intervals()[0]["interval_id"] == pending
    assert ledger(root).spent_seconds() == 0
    recovered = subprocess.run([*command[:2], "--recover", *command[2:]], cwd=root,
                               capture_output=True, text=True, timeout=30)
    assert recovered.returncode == 0 and "unbilled" in recovered.stdout
    assert ledger(root).open_intervals()[0]["interval_id"] == pending


def test_recovery_ignores_invalid_launch_budget(tmp_path):
    root = setup(tmp_path)
    pending = ledger(root).open_interval(run_id="direct-stale", stage_key=STAGE, stage=STAGE)
    from src.utils.io import read_yaml, write_yaml
    cfg = read_yaml(root / "experiment.yaml")
    cfg["limits"]["max_gpu_hours"] = 49
    write_yaml(root / "experiment.yaml", cfg)
    result = run(root, "--recover")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger(root)._rows()[-1]["status"] == "INTERRUPTED"
    assert ledger(root).open_intervals() == []


def test_recovery_accepts_null_launch_limits(tmp_path):
    root = setup(tmp_path)
    ledger(root).open_interval(run_id="direct-stale", stage_key=STAGE, stage=STAGE)
    from src.utils.io import read_yaml, write_yaml
    cfg = read_yaml(root / "experiment.yaml")
    cfg["limits"] = None
    write_yaml(root / "experiment.yaml", cfg)
    result = run(root, "--recover")
    assert result.returncode == 0, result.stdout + result.stderr
    assert ledger(root).open_intervals() == []


def test_timeout_closes_failed_interval(tmp_path):
    root = setup(tmp_path, "import time\ntime.sleep(60)")
    result = subprocess.run(cmd(root, "--timeout-seconds", "0.2"), cwd=root,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0, result.stdout + result.stderr
    assert ledger(root)._rows()[-1]["status"] == "FAILED"


def test_recovery_fails_closed_for_live_process_and_recovers_dead_one(tmp_path):
    root = setup(tmp_path, "import time\ntime.sleep(60)")
    process = subprocess.Popen(cmd(root, "--timeout-seconds", "60"), cwd=root,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               start_new_session=True)
    for _ in range(150):
        if ledger(root).path.exists() and ledger(root).open_intervals():
            break
        time.sleep(.02)
    else:
        pytest.fail("GPU stage never opened an interval")
    try:
        assert run(root, "--recover").returncode != 0
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
        result = run(root, "--recover")
        assert result.returncode == 0, result.stdout + result.stderr
        assert not ledger(root).open_intervals()
        assert ledger(root)._rows()[-1]["status"] == "INTERRUPTED"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)

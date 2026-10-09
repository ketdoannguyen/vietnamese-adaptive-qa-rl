"""P9 report stage: every ``reports/`` product generated from recorded artifacts.

    python -m src.pipeline.report --config configs/pilot.yaml

Source-of-truth policy (`05_TEMPLATES/06` §1): a number enters this report only from a file —
a run-registry row, an evaluation CSV/JSON, a manifest or a model artifact. Missing inputs are
rendered ``NOT_RUN``/``N/A`` with the path that was looked for; nothing is estimated and no
``[FROM_ARTIFACT: …]`` placeholder survives (the run fails if one does).

Written by this module (and nothing else in the repo):

* ``reports/results.csv`` / ``reports/results.json`` — one row per (E00–E07 × val/test), status
  from the recorded rows, metrics ``N/A``/``null`` when missing, never 0.
* ``reports/REPORT.md`` + ``reports/REPORT.txt`` — the Vietnamese report of
  ``05_TEMPLATES/02``, every table carrying N, split, seed and its source artifact path.
* ``reports/MODEL_CARD.md`` (``05_TEMPLATES/04``), ``reports/REPRODUCE.md``,
  ``reports/FINAL_HANDOFF.md``, ``reports/artifact_manifest.sha256`` and
  ``docs/TEST_ACCESS_LOG.md`` (honest verdict — "KHÔNG THỂ CHỨNG MINH" when the artifacts
  cannot prove test access order).
* one ``P9_report`` registry row; ``--zip`` adds a size-guarded ``reports/handoff_metadata.zip``
  (code/config/docs/report only — never model weights).

Raw run outputs are read-only here. ``reports/results.csv`` keeps rows it already held for
experiments outside E00–E07 (other stages append their rows there) instead of dropping them.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.eval import reporting
from src.utils.config import REPO_ROOT, add_profile_args, config_from_args, resolve
from src.utils.hashing import sha256_file, sha256_text
from src.utils.io import atomic_write_json
from src.utils.runs import RunRegistry, git_sha, now_iso

STAGE = "P9_report"

#: Generated files, in write order. ``artifact_manifest.sha256`` excludes itself.
REPORT_MD = "REPORT.md"
REPORT_TXT = "REPORT.txt"
RESULTS_CSV = "results.csv"
RESULTS_JSON = "results.json"
MODEL_CARD_MD = "MODEL_CARD.md"
REPRODUCE_MD = "REPRODUCE.md"
MANIFEST_FILE = "artifact_manifest.sha256"
HANDOFF_MD = "FINAL_HANDOFF.md"
ZIP_FILE = "handoff_metadata.zip"
TEST_ACCESS_LOG = Path("docs") / "TEST_ACCESS_LOG.md"

SPLITS = ("val_strict", "test_strict")
SPLIT_SHORT = {"val_strict": "val", "test_strict": "test"}

#: ``reporting.RESULTS_COLUMNS`` + the source file of every measured cell.
RESULTS_OUT_COLUMNS = tuple(reporting.RESULTS_COLUMNS) + ("source",)

#: Everything ``--zip`` may contain: documentation, code, config, scripts — never weights.
ZIP_INCLUDE_DIRS = ("reports", "docs", "scripts", "configs", "src", "tests")
ZIP_INCLUDE_FILES = ("README.md", "pyproject.toml")
ZIP_EXCLUDE_NAMES = {ZIP_FILE, "__pycache__", ".pytest_cache"}


# --------------------------------------------------------------------------- experiments

@dataclass(frozen=True)
class Experiment:
    """One row of the pre-registered comparison matrix (`03_EXPERIMENTS/00` §3)."""

    exp_id: str
    baseline: str
    decision: str
    prior: str | None = None
    note: str = ""


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("E00", "B1", "Rule bootstrap: trả lời/từ chối ngay sau bootstrap", None,
               "zero adaptation"),
    Experiment("E01", "B2", "Fixed BM25 tới hết ngân sách tìm tùy chọn", None,
               "fixed keyword retrieval"),
    Experiment("E02", "B3", "Fixed Dense tới hết ngân sách tìm tùy chọn", None,
               "fixed semantic retrieval"),
    Experiment("E03", "B5", "Ngưỡng p_correct hiệu chuẩn trên val", None,
               "classical adaptive baseline"),
    Experiment("E04", "B6", "Laya CE: masked argmax", "ce", "supervised decision model"),
    Experiment("E05", "B7", "Laya RLCD: masked argmax", "rlcd", "calibrated decision model"),
    Experiment("E06", "B8", "REINFORCE MLP trên prior RLCD (agent chính)", "rlcd",
               "main RL agent"),
    Experiment("E07", "B8", "REINFORCE MLP trên prior CE (tùy chọn)", "ce",
               "isolates the RLCD increment"),
)
EXP_IDS = tuple(experiment.exp_id for experiment in EXPERIMENTS)

#: Pre-registered ablations (`03_EXPERIMENTS/00` §4) — reported, never invented.
ABLATIONS: tuple[tuple[str, str], ...] = (
    ("A1", "reward không có chi phí tìm kiếm (0) so với mặc định"),
    ("A2", "chỉ hành động BM25 so với hai công cụ"),
    ("A3", "bỏ prior Laya (MLP chỉ dùng feature số) để đo đóng góp của Laya"),
    ("A4", "prior CE so với prior RLCD với cùng policy REINFORCE (xem E06/E07)"),
    ("A5", "n_distractors = 4/12/24"),
    ("A6", "ngân sách tìm tùy chọn 1/2/3"),
    ("A7", "không cho phép abstain so với có abstain"),
    ("A8", "bật/tắt hiệu chuẩn null của Reader (không rò rỉ ngưỡng từ test)"),
)


@dataclass(frozen=True)
class Stage:
    """One stage of the DAG (`02_IMPLEMENTATION/06` §2) and where its evidence lives."""

    phase: str
    key: str
    registry_stage: str | None
    command: str
    outputs: tuple[str, ...]
    regime: str | None = None


STAGES: tuple[Stage, ...] = (
    Stage("P0", "preflight", None, "bash scripts/preflight.sh",
          ("outputs/diagnostics/preflight.md", "outputs/diagnostics/preflight_verdict.json")),
    Stage("P1", "download", "P1_download",
          "python -m src.pipeline.download --config configs/{profile}.yaml",
          ("data/raw/source_manifest.json",)),
    Stage("P1", "audit_split", "P1_audit_split",
          "python -m src.pipeline.audit_split --config configs/{profile}.yaml",
          ("data/processed/split_manifest.json",)),
    Stage("P2", "build_retrieval", "P2_retrieval",
          "python -m src.pipeline.build_retrieval --config configs/{profile}.yaml",
          ("reports/retrieval_report.json", "data/cache/retrieval/")),
    Stage("P3", "train_reader", "P3_reader",
          "python -m src.pipeline.train_reader --config configs/{profile}.yaml",
          ("outputs/models/qa_reader/reader_inference.json",
           "outputs/models/qa_reader/reader_eval.json")),
    Stage("P3", "cache_reader", "P3_reader_cache",
          "python -m src.pipeline.cache_reader --config configs/{profile}.yaml",
          ("outputs/models/qa_reader/reader_qa_state_cache_train_strict.parquet",
           "outputs/models/qa_reader/reader_qa_state_cache_val_strict.parquet")),
    Stage("P5", "make_teacher", "P5_teacher",
          "python -m src.pipeline.make_teacher --config configs/{profile}.yaml",
          ("data/processed/laya_train.jsonl",)),
    Stage("P5", "train_laya_ce", "P5_laya_train",
          "python -m src.pipeline.train_laya --config configs/{profile}.yaml --loss soft-ce",
          ("outputs/models/laya_ce/",), regime="ce"),
    Stage("P5", "train_laya_rlcd", "P5_laya_train",
          "python -m src.pipeline.train_laya --config configs/{profile}.yaml --loss rlcd",
          ("outputs/models/laya_rlcd/",), regime="rlcd"),
    Stage("P5", "cache_laya", "P5_cache_laya",
          "python -m src.pipeline.cache_laya --config configs/{profile}.yaml",
          ("outputs/models/laya_probs_cache/",)),
    Stage("P6", "train_rl", "P6_rl",
          "python -m src.pipeline.train_rl --config configs/{profile}.yaml",
          ("outputs/models/rl_policy/", "reports/rl_training_report.json")),
    Stage("P6", "select_rl", "P6_select",
          "python -m src.pipeline.select_rl --config configs/{profile}.yaml",
          ("outputs/manifests/best_selection.json",)),
    Stage("P8", "train_correctness", "B5_correctness",
          "python -m src.pipeline.train_correctness --config configs/{profile}.yaml",
          ("outputs/models/correctness_estimator/", "reports/correctness_calibration.md")),
    Stage("P7", "eval_baselines", "P7_baselines",
          "python -m src.pipeline.eval_baselines --config configs/{profile}.yaml "
          "--split {split}",
          ("reports/tables/baseline_results_val.csv", "reports/tables/baseline_results_test.csv")),
    Stage("P9", "report", "P9_report",
          "python -m src.pipeline.report --config configs/{profile}.yaml",
          ("reports/REPORT.md", "reports/results.csv")),
)


# ------------------------------------------------------------------------------- sources

@dataclass
class Sources:
    """Every artifact this report reads; ``None`` means "not found at this path"."""

    cfg: Mapping[str, Any]
    profile: str
    now: str
    reports: Path
    tables: Path
    plots: Path
    registry_path: Path
    registry: list[dict[str, Any]] = field(default_factory=list)
    #: sha256 of the registry lines *excluding* this stage's own rows (see :func:`registry_sha`).
    registry_sha: str | None = None
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    baselines: dict[str, dict[str, Any]] = field(default_factory=dict)
    b5_row: dict[str, Any] | None = None
    b5_row_path: Path | None = None
    retrieval: dict[str, Any] | None = None
    retrieval_path: Path | None = None
    reader_eval: dict[str, Any] | None = None
    reader_eval_path: Path | None = None
    reader_contract: dict[str, Any] | None = None
    reader_contract_path: Path | None = None
    rl_report: dict[str, Any] | None = None
    rl_report_path: Path | None = None
    rl_config: dict[str, Any] | None = None
    rl_config_path: Path | None = None
    selection: dict[str, Any] | None = None
    selection_path: Path | None = None
    split_manifest: dict[str, Any] | None = None
    split_manifest_path: Path | None = None
    source_manifest: dict[str, Any] | None = None
    source_manifest_path: Path | None = None
    gpu: dict[str, Any] | None = None
    gpu_path: Path | None = None
    system: dict[str, Any] | None = None
    system_path: Path | None = None
    preflight: dict[str, Any] | None = None
    preflight_path: Path | None = None
    stage_manifests: dict[str, dict[str, Any]] = field(default_factory=dict)
    baseline_reports: dict[str, dict[str, Any]] = field(default_factory=dict)
    predictions: list[Path] = field(default_factory=list)


def _maybe_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def registry_sha(path: Path) -> str | None:
    """sha256 of the registry content with the ``P9_report`` rows removed.

    This stage appends rows for every report it writes, so hashing the raw file would make
    re-runs of an unchanged repo differ. The digest (and its label) always says exactly what
    it covers.
    """
    if not path.is_file():
        return None
    kept = [line for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line).get("stage") != STAGE]
    return sha256_text("\n".join(kept) + "\n") if kept else None


def _stage_status(rows: Sequence[Mapping[str, Any]], stage: Stage, profile: str
                  ) -> dict[str, Any]:
    """Latest terminal status of one stage, ``INTERRUPTED`` when only RUNNING rows remain."""
    relevant = [row for row in rows
                if row.get("stage") == stage.registry_stage
                and (stage.regime is None or row.get("regime") == stage.regime)
                and row.get("profile") in (None, profile)]
    if stage.registry_stage is None:
        return {"status": "NOT_RUN", "run_id": None, "ts": None,
                "reason": "stage không ghi vào runs/index.jsonl (xem outputs/diagnostics)"}
    if not relevant:
        return {"status": "NOT_RUN", "run_id": None, "ts": None,
                "reason": f"không có bản ghi stage={stage.registry_stage!r} "
                          f"profile={profile!r} trong runs/index.jsonl"}
    terminal = [row for row in relevant if row.get("status") != "RUNNING"]
    if terminal:
        last = terminal[-1]
        return {"status": str(last["status"]), "run_id": str(last["run_id"]),
                "ts": last.get("ts"), "reason": last.get("error") or last.get("reason")}
    last = relevant[-1]
    return {"status": "INTERRUPTED", "run_id": str(last["run_id"]), "ts": last.get("ts"),
            "reason": ("bản ghi RUNNING cuối cùng không có dòng kết thúc — tiến trình đã dừng "
                       "(02_IMPLEMENTATION/06 §5)")}


def load_sources(cfg: Mapping[str, Any], profile: str, now: str) -> Sources:
    reports, tables, plots = resolve(cfg, "reports"), resolve(cfg, "tables"), resolve(cfg, "plots")
    sources = Sources(cfg=cfg, profile=profile, now=now, reports=reports, tables=tables,
                      plots=plots, registry_path=resolve(cfg, "runs") / "index.jsonl")
    # This stage's own rows are excluded: it appends one on every report it writes, and a
    # report that quoted its own earlier runs could never be byte-identical across re-runs
    # of unchanged sources (see ``registry_sha``).
    sources.registry = [row for row in _read_jsonl(sources.registry_path)
                        if row.get("stage") != STAGE]
    sources.registry_sha = registry_sha(sources.registry_path)
    for stage in STAGES:
        sources.stages[stage.key] = _stage_status(sources.registry, stage, profile)
    # The current run's own P9_report row is appended after this read (it can only say
    # SUCCEEDED when every product exists), so the entry states exactly that.
    sources.stages["report"] = {
        "status": "SUCCEEDED", "run_id": None, "ts": now,
        "reason": "lần chạy sinh báo cáo hiện tại (dòng P9_report được ghi sau khi đọc nguồn)"}

    for split in SPLITS:
        short = SPLIT_SHORT[split]
        path = tables / f"baseline_results_{short}.csv"
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [{column: _parse_cell(value) for column, value in row.items()}
                    for row in reader]
        sources.baselines[short] = {"path": path, "sha256": sha256_file(path), "rows": rows}
        report_path = reports / f"baseline_report_{short}.json"
        loaded = _maybe_json(report_path)
        if loaded is not None:
            sources.baseline_reports[short] = loaded

    def _load(attr: str, path: Path) -> None:
        loaded = _maybe_json(path)
        if loaded is not None:
            setattr(sources, attr, loaded)
            setattr(sources, f"{attr}_path", path)

    _load("b5_row", reports / "results_b5_row.json")
    _load("retrieval", reports / "retrieval_report.json")
    _load("reader_eval", resolve(cfg, "models") / "qa_reader" / "reader_eval.json")
    _load("reader_contract", resolve(cfg, "models") / "qa_reader" / "reader_inference.json")
    _load("rl_report", reports / "rl_training_report.json")
    _load("rl_config", resolve(cfg, "models") / "rl_policy" / "config.json")
    _load("selection", resolve(cfg, "manifests") / "best_selection.json")
    _load("split_manifest", resolve(cfg, "data_processed") / "split_manifest.json")
    _load("source_manifest", resolve(cfg, "data_raw") / "source_manifest.json")
    _load("gpu", resolve(cfg, "diagnostics") / "gpu_info.json")
    _load("system", resolve(cfg, "diagnostics") / "system_info.json")
    _load("preflight", resolve(cfg, "diagnostics") / "preflight_verdict.json")

    manifests = resolve(cfg, "manifests")
    if manifests.is_dir():
        for path in sorted(manifests.glob("P5_train_laya_*.json")):
            loaded = _maybe_json(path)
            if loaded is not None:
                sources.stage_manifests[path.stem] = loaded
    predictions_dir = resolve(cfg, "predictions")
    if predictions_dir.is_dir():
        found = sorted(predictions_dir.glob("*.parquet"))
        preferred = [p for p in found if f"-{profile}-" in p.name] or found
        sources.predictions = preferred
    return sources


def _parse_cell(value: str | None) -> Any:
    """Typed cell of a source CSV, round-tripping through :func:`reporting.format_cell`."""
    if value is None or value == "" or value == "N/A":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


# ------------------------------------------------------------------------------- results

def build_results_rows(sources: Sources) -> list[dict[str, Any]]:
    """One row per (E00–E07 × val/test): measured where recorded, ``NOT_RUN`` fillers else."""
    rows: list[dict[str, Any]] = []
    for exp_id in EXP_IDS:
        spec = next(experiment for experiment in EXPERIMENTS if experiment.exp_id == exp_id)
        for split in SPLITS:
            short = SPLIT_SHORT[split]
            source = sources.baselines.get(short)
            found = None
            if source is not None:
                found = next((row for row in source["rows"]
                              if row.get("exp_id") == exp_id
                              and row.get("split") in (split, short)), None)
            if found is not None:
                row = {column: found.get(column) for column in reporting.RESULTS_COLUMNS}
                row["source"] = str(source["path"])
            else:
                row = _not_run_row(spec, split, sources)
            rows.append(row)
    # Foreign rows (other experiments/stages appended to our file) are kept, not dropped.
    rows.extend(_foreign_rows(sources.reports / RESULTS_CSV))
    return rows


def _not_run_row(spec: Experiment, split: str, sources: Sources) -> dict[str, Any]:
    short = SPLIT_SHORT[split]
    row: dict[str, Any] = dict(reporting.empty_metrics())
    row.update({"run_id": None, "exp_id": spec.exp_id, "policy": spec.baseline,
                "policy_variant": None, "prior_regime": spec.prior, "split": split,
                "dev": None, "seed": None, "tau": None, "status": "NOT_RUN",
                "reason": (f"chưa có dòng {spec.exp_id} cho {split} trong "
                           f"reports/tables/baseline_results_{short}.csv "
                           "(chạy python -m src.pipeline.eval_baselines)"),
                "source": None})
    return row


def _foreign_rows(path: Path) -> list[dict[str, Any]]:
    """Rows already in ``results.csv`` that are not part of the E00–E07 matrix."""
    if not path.is_file():
        return []
    keep: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("exp_id") or "") not in EXP_IDS:
                keep.append({column: _parse_cell(row.get(column))
                             for column in RESULTS_OUT_COLUMNS})
    return keep


def write_results_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(RESULTS_OUT_COLUMNS)
        for row in rows:
            writer.writerow([reporting.format_cell(row.get(column))
                             for column in RESULTS_OUT_COLUMNS])
    return path


def write_results_json(path: Path, rows: Sequence[Mapping[str, Any]], sources: Sources
                       ) -> Path:
    source_files: dict[str, dict[str, str]] = {}
    for name in ("baseline_results_val", "baseline_results_test"):
        short = name.rsplit("_", 1)[1]
        if short in sources.baselines:
            source_files[name] = {"path": str(sources.baselines[short]["path"]),
                                  "sha256": sources.baselines[short]["sha256"]}
    if sources.registry_sha is not None:
        source_files["registry"] = {
            "path": str(sources.registry_path), "sha256": sources.registry_sha,
            "note": "sha256 không gồm các dòng P9_report của chính báo cáo"}
    for name, source_path in (("retrieval_report", sources.retrieval_path),
                              ("reader_eval", sources.reader_eval_path),
                              ("rl_training_report", sources.rl_report_path),
                              ("b5_training_row", sources.b5_row_path),
                              ("selection_freeze", sources.selection_path)):
        if source_path is not None and source_path.is_file():
            source_files[name] = {"path": str(source_path),
                                  "sha256": sha256_file(source_path)}
    payload = {
        "generated_at": sources.now, "profile": sources.profile, "stage": STAGE,
        "columns": list(RESULTS_OUT_COLUMNS),
        "rows": [{column: row.get(column) for column in RESULTS_OUT_COLUMNS} for row in rows],
        "sources": source_files,
        "note": ("null = không có artifact đo được (KHÔNG phải 0); "
                 "mỗi dòng ghi đường dẫn file nguồn ở cột 'source'"),
    }
    atomic_write_json(path, payload)
    return path


# ------------------------------------------------------------------------------- rendering

def _fmt(value: Any) -> str:
    return reporting.format_cell(value)


def _pick(mapping: Mapping[str, Any] | None, *names: str) -> Any:
    if not mapping:
        return None
    for name in names:
        if mapping.get(name) is not None:
            return mapping[name]
    return None


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(cell) for cell in row) + " |")
    return lines


def _durations(sources: Sources) -> list[list[Any]]:
    """Measured stage walltime: first RUNNING timestamp to the last terminal row."""
    durations = []
    for stage in STAGES:
        if stage.registry_stage is None:
            continue
        relevant = [row for row in sources.registry
                    if row.get("stage") == stage.registry_stage
                    and (stage.regime is None or row.get("regime") == stage.regime)
                    and row.get("profile") in (None, sources.profile)]
        starts = [row["ts"] for row in relevant if row.get("status") == "RUNNING"]
        ends = [row["ts"] for row in relevant if row.get("status") != "RUNNING"]
        seconds = None
        if starts and ends:
            try:
                delta = (_dt.datetime.fromisoformat(ends[-1])
                         - _dt.datetime.fromisoformat(starts[0]))
                seconds = round(delta.total_seconds(), 1)
            except ValueError:
                seconds = None
        durations.append([stage.phase, stage.key, sources.stages[stage.key]["status"], seconds])
    return durations


def _laya_rows(sources: Sources) -> list[list[Any]]:
    rows = []
    for regime, checkpoint in (("ce", "laya_ce"), ("rlcd", "laya_rlcd")):
        manifest = sources.stage_manifests.get(f"P5_train_laya_{regime}_{sources.profile}")
        calibration = (manifest or {}).get("calibration") or {}
        rows.append([
            checkpoint, regime, (manifest or {}).get("loss"),
            (manifest or {}).get("data", {}).get("sha256"),
            _pick(manifest, "train_items") or calibration.get("train_items"),
            calibration.get("calibration_items"), calibration.get("temperature"),
            _pick(manifest, "epoch_loss"), _pick(manifest, "walltime_s"),
            _pick(manifest, "peak_vram_mib"),
            "SUCCEEDED" if manifest else "NOT_RUN",
        ])
    return rows


def _failure_cases(sources: Sources, limit: int = 20) -> tuple[list[list[Any]], str]:
    """Real failed episodes from the prediction artifacts; never a fabricated example."""
    import pyarrow.parquet as pq

    failures: list[list[Any]] = []
    seen_rows = 0
    for path in sources.predictions:
        table = pq.read_table(str(path))
        for row in table.to_pylist():
            seen_rows += 1
            answered = row.get("terminal_action") == "ANSWER"
            correct = row.get("eval_correct")
            impossible = bool(row.get("eval_is_impossible"))
            if not (impossible and answered) and not (answered and correct == 0):
                continue
            cause = ("trả lời câu không có đáp án" if impossible and answered
                     else "trả lời sai (EM=0)")
            failures.append([
                row.get("qid"), row.get("split"), row.get("exp_id"), row.get("seed"),
                row.get("action_name_trace"), row.get("total_retrievals"),
                _pick(row, "eval_correctness_f1"),
                (row.get("final_candidate") or "(không có)"), cause, str(path),
            ])
            if len(failures) >= limit:
                break
        if len(failures) >= limit:
            break
    if not sources.predictions:
        return [], "NOT_RUN — chưa có outputs/predictions/*.parquet"
    if seen_rows == 0:
        return [], "NOT_RUN — file predictions rỗng"
    if not failures:
        return [], (f"Không có episode thất bại trong {seen_rows} dòng predictions đã đọc "
                    "(mọi câu trả lời đều đúng theo EM) — đây là kết quả đo được, "
                    "không phải bỏ sót mẫu.")
    return failures, f"{len(failures)} mẫu thất bại thực tế trong {seen_rows} dòng predictions"


def _plot_inventory(sources: Sources) -> list[list[Any]]:
    names = ("reward_learning_curve.png", "action_histogram.png",
             "answer_f1_vs_retrieval_cost.png", "risk_coverage.png", "retrieval_recall.png",
             "correctness_reliability.png")
    rows = []
    for name in names:
        path = sources.plots / name
        rows.append([name, "có" if path.is_file() else "CHƯA TẠO",
                     str(path) if path.is_file() else None])
    return rows


def _experiment_table(rows: Sequence[Mapping[str, Any]], split: str) -> list[list[Any]]:
    out = []
    for row in rows:
        if row.get("exp_id") not in EXP_IDS or row.get("split") != split:
            continue
        out.append([row.get("exp_id"), row.get("policy"), row.get("seed"),
                    row.get("n_episodes"), row.get("mean_episode_reward"),
                    row.get("token_f1"), row.get("answerable_answer_f1"),
                    row.get("false_answer_rate_on_unanswerables"), row.get("coverage"),
                    row.get("avg_retrieval_cost"), row.get("status"), row.get("source")])
    return out


def _row(rows: Sequence[Mapping[str, Any]], exp_id: str, split: str) -> Mapping[str, Any] | None:
    return next((row for row in rows if row.get("exp_id") == exp_id
                 and row.get("split") == split and row.get("status") == "SUCCEEDED"), None)


def _delta(rows: Sequence[Mapping[str, Any]], split: str, left: str, right: str) -> str:
    a, b = _row(rows, left, split), _row(rows, right, split)
    if a is None or b is None:
        return f"{left} vs {right}: chưa đủ dữ liệu (cần cả hai dòng SUCCEEDED ở {split})"
    parts = []
    for column, label in (("mean_episode_reward", "reward"), ("token_f1", "token F1"),
                          ("avg_retrieval_cost", "chi phí tìm")):
        av, bv = a.get(column), b.get(column)
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            parts.append(f"Δ{label}={av - bv:+.6f}")
    seeds = {a.get("seed"), b.get("seed")}
    caveat = " (một seed — exploratory)" if len(seeds) == 1 else ""
    return f"{left} vs {right}: " + ", ".join(parts) + caveat


def _hypotheses(rows: Sequence[Mapping[str, Any]], sources: Sources) -> list[list[Any]]:
    ce = sources.stage_manifests.get(f"P5_train_laya_ce_{sources.profile}")
    rlcd = sources.stage_manifests.get(f"P5_train_laya_rlcd_{sources.profile}")
    main_vs_threshold = _row(rows, "E06", "val_strict") and _row(rows, "E03", "val_strict")

    def verdict(available: bool) -> str:
        return "inconclusive" if available else "NOT_RUN (thiếu artifact)"

    return [
        ["H1 — RLCD giảm ECE/Brier so với Soft-CE trên cùng target", "RQ1",
         verdict(bool(ce and rlcd)),
         "cần artifact đo calibration của hai checkpoint (P5 manifest + đánh giá teacher)"],
        ["H2 — REINFORCE tăng mean reward so với threshold", "RQ2",
         verdict(bool(main_vs_threshold)), _delta(rows, "val_strict", "E06", "E03")],
        ["H3 — REINFORCE giảm chi phí tìm ở mức F1 tương đương", "RQ3",
         verdict(bool(main_vs_threshold)), _delta(rows, "val_strict", "E06", "E01")],
        ["H4 — trade-off accuracy/coverage theo hướng penalty", "RQ4",
         verdict(bool(_row(rows, "E06", "val_strict"))),
         "chỉ kết luận được khi có bảng risk–coverage của agent"],
    ]


def _sources_block(sources: Sources) -> list[list[Any]]:
    """Appendix table: every artifact this report read, with its digest (or CHƯA CÓ).

    The registry row cites only the filtered digest (:func:`registry_sha`), so a re-run sees
    exactly the same cells whether or not the file already holds earlier P9_report rows.
    """
    registry = str(sources.registry_path)
    rows: list[list[Any]] = [[
        "runs/index.jsonl",
        "có" if sources.registry_sha else "CHƯA CÓ dòng ngoài P9_report",
        f"{registry} (sha256 {sources.registry_sha[:16]}…, không gồm dòng P9_report)"
        if sources.registry_sha else registry]]
    entries = [
        ("baseline_results_val.csv", (sources.baselines.get("val") or {}).get("path")),
        ("baseline_results_test.csv", (sources.baselines.get("test") or {}).get("path")),
        ("retrieval_report.json", sources.retrieval_path),
        ("reader_inference.json", sources.reader_contract_path),
        ("reader_eval.json", sources.reader_eval_path),
        ("rl_training_report.json", sources.rl_report_path),
        ("results_b5_row.json", sources.b5_row_path),
        ("best_selection.json", sources.selection_path),
        ("split_manifest.json", sources.split_manifest_path),
        ("source_manifest.json", sources.source_manifest_path),
        ("gpu_info.json", sources.gpu_path),
    ]
    for name, path in entries:
        if path is None or not Path(path).is_file():
            rows.append([name, "CHƯA CÓ", None])
        else:
            digest = sha256_file(Path(path))
            rows.append([name, "có", f"{path} (sha256 {digest[:16]}…)"])
    return rows


def render_report(sources: Sources, rows: Sequence[Mapping[str, Any]],
                  failures: Sequence[Sequence[Any]], failure_note: str,
                  registry_digest: str | None) -> str:
    """The Vietnamese report of ``05_TEMPLATES/02`` — every number carries its source."""
    git = git_sha(REPO_ROOT) or "không xác định (không có git trong môi trường này)"
    split_manifest = sources.split_manifest or {}
    strict_rows = (split_manifest.get("strict_split") or {}).get("rows") or {}
    reader_hash = (sources.reader_contract or {}).get("reader_model_hash")
    threshold = (sources.reader_contract or {}).get("null_score_diff_threshold")
    best = max((row for row in rows if row.get("status") == "SUCCEEDED"
                and isinstance(row.get("token_f1"), (int, float))),
               key=lambda row: row["token_f1"], default=None)
    lines: list[str] = []

    lines += [
        "# BÁO CÁO CUỐI KỲ — HỆ TÁC TỬ HỎI ĐÁP TIẾNG VIỆT THÍCH NGHI",
        "",
        f"- **Tên đề tài:** Hệ tác tử hỏi đáp tiếng Việt thích nghi sử dụng Reinforcement "
        f"Learning và mô hình quyết định Jev-like.",
        f"- **Tên Anh:** Vietnamese Adaptive QA Agent with RLCD and Sequential Policy Learning.",
        f"- **Học phần:** Reinforcement Learning / Học tăng cường.",
        f"- **Nhóm/Sinh viên/Giảng viên/Đơn vị:** chưa có trong artifact — điền tay tại đây.",
        f"- **Ngày hoàn thành (sinh báo cáo):** {sources.now}",
        f"- **Git SHA:** {git}",
        f"- **Reader model hash:** {_fmt(reader_hash)} | **ngưỡng null-score-diff:** "
        f"{_fmt(threshold)}",
        "",
        "## Tóm tắt / Abstract",
        "",
        "Đồ án xây dựng một tác tử hỏi đáp trích xuất tiếng Việt trên kho văn bản đóng, trong đó "
        "tác tử quyết định tuần tự khi nào tìm thêm bằng chứng bằng BM25 hay Dense, khi nào trả "
        "lời và khi nào từ chối, với chi phí tìm kiếm nằm trong hàm thưởng. Mô hình quyết định "
        "gồm một prior Laya được chuyên biệt hóa bằng RLCD (so sánh với Soft-CE) và một policy "
        "REINFORCE tối ưu episodic reward trên cùng môi trường, cùng reader và cùng ngân sách "
        "tìm kiếm với mọi baseline. Báo cáo này được sinh hoàn toàn từ artifact đã ghi: các "
        "bảng E00–E07 lấy từ reports/tables/baseline_results_*.csv, chỉ số retrieval từ "
        "reports/retrieval_report.json, cấu hình và thời lượng từ runs/index.jsonl và các "
        "manifest. Kết quả chính: "
        + (f"dòng tốt nhất theo token F1 hiện có là {best['exp_id']} trên {best['split']} "
           f"(F1={_fmt(best['token_f1'])}, N={_fmt(best['n_episodes'])}); "
           if best is not None else
           "chưa có dòng kết quả nào chạy xong trong artifact, nên mọi so sánh định lượng được "
           "ghi NOT_RUN thay vì suy đoán; ")
        + "các giả thuyết H1–H4 và trạng thái supported/not supported/inconclusive được ghi "
          "tại mục 10 kèm bằng chứng file. Từ khóa: RL, Reinforcement Learning, Vietnamese QA, "
          "Adaptive Retrieval, RLCD, Calibration, Abstention.",
        "",
        "## 1. Giới thiệu và phát biểu vấn đề",
        "",
        "- Bài toán: trả lời đúng câu hỏi tiếng Việt trên kho tài liệu đóng, hạn chế hallucination "
        "và chi phí truy hồi; câu trả lời phải trích từ bằng chứng đã thấy, hoặc từ chối.",
        "- RAG cố định (retrieve-k rồi trả lời) so với tác tử chủ động quyết định thu thập "
        "bằng chứng từng bước; đây là bài toán điều khiển, không phải sinh văn bản.",
        "- Không gian hành động 4 lựa chọn: ANSWER, SEARCH_BM25, SEARCH_DENSE, ABSTAIN; episode "
        "kết thúc khi trả lời/từ chối hoặc hết ngân sách tìm kiếm, sau đó hệ thống buộc quyết định.",
        "- Bốn câu hỏi nghiên cứu (pre-registered, `01_SPECIFICATIONS/01` §3):",
        "  - **RQ1 (calibrated decisions):** trên cùng dữ liệu teacher, RLCD so với Soft-CE có "
        "cải thiện calibration, action accuracy và tỷ lệ chọn hành động đúng không?",
        "  - **RQ2 (sequential RL):** policy REINFORCE trên feature Laya có tăng episodic reward "
        "so với threshold, SFT policy và chiến lược cố định dưới cùng chi phí không?",
        "  - **RQ3 (retrieval choice):** policy biết chọn BM25 vs Dense và thời điểm dừng có "
        "đem lại Pareto trade-off Answer F1 / retrieval cost tốt hơn không?",
        "  - **RQ4 (robustness, tùy chọn):** chính sách có chịu được thay đổi noise, chất lượng "
        "retrieval, ngân sách tìm kiếm và câu hỏi không có đáp án không?",
        "- Giả thuyết H1–H4 (`01_SPECIFICATIONS/01` §4) và kết quả kiểm định ở mục 10; không "
        "tuyên bố kết quả trước khi đo.",
        "",
        "## 2. Cơ sở lý thuyết",
        "",
        "- MDP hữu hạn với quan sát đã thấy: state gồm câu hỏi, các đoạn đã xem, candidate của "
        "reader, thống kê truy hồi và lịch sử hành động; policy ngẫu nhiên theo REINFORCE "
        "(policy gradient, baseline, entropy regularization, gamma=1).",
        "- RLCD/Laya/Jev: mô hình ra **quyết định** (choice + probabilities) tách biệt với "
        "reader trả **câu trả lời**; không fine-tune Jev gốc và không claim như vậy.",
        f"- Calibration (ECE/Brier) đo chất lượng xác suất p_correct của baseline threshold; "
        "xác suất chọn hành động của policy **không phải** xác suất đáp án đúng.",
        "- Retrieval BM25 (Okapi) so với dense E5 multilingual (prefix query:/passage:); reader "
        "XLM-R base được SFT trích xuất với cơ chế no-answer kiểu SQuAD2.",
        "- Related work: Adaptive-RAG (chọn chiến lược theo độ khó), Search-R1 (RL cho multi-turn "
        "search) — khác biệt ở quy mô nhỏ, tiếng Việt, cost-aware và có abstention.",
        "",
        "## 3. Dữ liệu, giấy phép và chia tập",
        "",
        f"- Nguồn: {_fmt((sources.source_manifest or {}).get('repo_id') or 'taidng/UIT-ViQuAD2.0')}"
        f", revision {_fmt((split_manifest.get('source') or {}).get('resolved_revision'))} "
        "(pin tại thời điểm tải, ghi trong data/raw/source_manifest.json).",
        f"- Số dòng raw đã dùng: {_fmt((split_manifest.get('source') or {}).get('raw_rows'))}; "
        f"số dòng strict mỗi tập: {_fmt(strict_rows)}; profile={sources.profile}.",
        f"- Chia tập strict theo title-group (không rò rỉ tiêu đề giữa các tập); chi tiết merge "
        f"near-duplicate và tỷ lệ trong data/processed/split_manifest.json.",
        f"- Ngữ nghĩa `is_impossible`: câu hỏi không có đáp án trong ngữ cảnh; strict test labels "
        f"chỉ được mở cho lần đánh giá cuối (xem docs/TEST_ACCESS_LOG.md).",
        f"- Giấy phép dữ liệu: card_license="
        f"{_fmt((sources.source_manifest or {}).get('card_license'))} — "
        "**UNVERIFIED**: license phân phối chưa được xác thực độc lập, không xuất bản lại raw "
        "dataset/checkpoint nếu chưa rà soát (05_TEMPLATES/04).",
        "",
        "## 4. Phương pháp đề xuất",
        "",
        "- Kiến trúc: BM25 + Dense index trên kho đóng → QA reader trích xuất (XLM-R SFT, ngưỡng "
        "null-score-diff chọn trên val) → mô hình quyết định (Laya RLCD/CE hoặc policy "
        "REINFORCE MLP) → hành động, lặp lại tới khi kết thúc episode.",
        "- Câu trả lời luôn là span trích từ đoạn đã truy hồi (kèm doc_id và offset); không sinh "
        "văn bản mới; câu từ chối cố định: “Chưa tìm thấy bằng chứng đủ để trả lời trong bộ tài "
        "liệu hiện tại.”",
        "- State/action mask/terminal theo `src/rl/env.py`; episode mặc định: 1 lượt bootstrap "
        "bắt buộc + tối đa 2 lượt tìm tùy chọn; mỗi lượt tìm mở thêm 1 đoạn.",
        "- Reward (configs/base.yaml `reward:`): "
        + ", ".join(f"{key}={value}" for key, value in sorted(sources.cfg["reward"].items()))
        + f" — chi phí bootstrap: {_fmt(sources.cfg['reward'].get('bootstrap_bm25_cost'))}.",
        "- Teacher cho Laya: quy tắc soft (temperature, tie-order) từ reward, không dùng oracle "
        "gold để chọn hành động; anti-leakage kiểm tra bằng `find_forbidden_keys`.",
        "- RL: REINFORCE on-policy, greedy eval, feature 33 chiều (số liệu + prior Laya), scaler "
        "chuẩn hóa đi kèm checkpoint; cache Laya probabilities theo (qid, state_hash).",
        "- Tài nguyên 24GB: microbatch nhỏ + gradient accumulation, cache state/predictions trên "
        "đĩa, giải phóng encoder sau khi dùng.",
        "",
        "## 5. Cài đặt và môi trường",
        "",
    ]
    gpu = ((sources.gpu or {}).get("gpus") or [{}])[0]
    torch_info = (sources.gpu or {}).get("torch") or {}
    lines += [
        f"- GPU: {_fmt(gpu.get('name'))} | VRAM {_fmt(gpu.get('memory_total_mib'))} MiB | "
        f"driver {_fmt(gpu.get('driver_version'))}; Torch {_fmt(torch_info.get('version'))}, "
        f"CUDA runtime {_fmt(torch_info.get('cuda_runtime'))}.",
        f"- Host: {_fmt((sources.system or {}).get('platform'))}; CPU "
        f"{_fmt((sources.system or {}).get('cpu_count'))} lõi; RAM "
        f"{_fmt(((sources.system or {}).get('memory_mib') or {}).get('MemTotal'))} MiB; "
        f"disk free {_fmt(((((sources.system or {}).get('disks') or {}).get('root')) or {}).get('free_gb'))} GB.",
        f"- Preflight: {_fmt((sources.preflight or {}).get('status'))}; blockers: "
        f"{_fmt((sources.preflight or {}).get('blockers'))}.",
        f"- Git SHA: {git}; seed mặc định {_fmt(sources.cfg.get('seed'))}; profile "
        f"{sources.profile}; reader max_len={_fmt((sources.reader_contract or {}).get('max_length'))}, "
        f"doc_stride={_fmt((sources.reader_contract or {}).get('doc_stride'))}, "
        f"max_answer_length={_fmt((sources.reader_contract or {}).get('max_answer_length'))}.",
        "- Lệnh chạy từng stage: xem REPRODUCE.md và bảng ở FINAL_HANDOFF.md; mọi stage ghi "
        "manifest + dòng registry với SHA256.",
        "",
        "## 6. Thiết kế thí nghiệm",
        "",
    ]
    lines += _table(["Mã", "Baseline", "Quyết định", "Prior", "Dữ liệu train", "Ghi chú"],
                    [[experiment.exp_id, experiment.baseline, experiment.decision,
                      experiment.prior, "train strict" if experiment.prior else "—",
                      experiment.note] for experiment in EXPERIMENTS])
    lines += [
        "",
        "- Cùng qid, cùng docs/scenario seed, cùng reader, cùng ranking, cùng mask/horizon và "
        "cùng trọng số reward cho mọi dòng E00–E07 (fairness trong "
        "reports/baseline_report_val.json).",
        f"- Profile: {sources.profile}; selection: mean episodic reward trên validation, "
        f"freeze trước test — trạng thái best_selection.json: "
        f"{'có' if sources.selection else 'CHƯA CÓ (chưa thể mở test comparison chính thức)'}.",
        "- Test strict chỉ chạy một lần sau khi khoá lựa chọn; access log ở "
        "docs/TEST_ACCESS_LOG.md; CI theo cụm title-group (cluster bootstrap) khi đủ dữ liệu.",
        "",
        "## 7. Kết quả quantitative — KHÔNG ĐƯỢC BỎ",
        "",
        "**Bảng 1 — Môi trường và thời lượng đo được** (nguồn: outputs/diagnostics/*, "
        "runs/index.jsonl; giây = RUNNING đầu → dòng kết thúc cuối):",
        "",
    ]
    lines += _table(["Giai đoạn", "Stage", "Trạng thái", "Giây (đo được)"], _durations(sources))
    retrieval = sources.retrieval or {}
    oracle = (sources.reader_eval or {}).get("oracle_context") or {}
    retrieved = (sources.reader_eval or {}).get("retrieved_context") or {}
    acceptance = (sources.reader_eval or {}).get("acceptance") or {}
    failed_checks = [name for name, check in acceptance.items()
                     if isinstance(check, dict) and check.get("passed") is False]
    lines += [
        "",
        f"**Bảng 2a — Retrieval** (nguồn: reports/retrieval_report.json; scenarios="
        f"{_fmt(retrieval.get('n_scenarios'))}, câu trả lời được="
        f"{_fmt(retrieval.get('n_answerable'))}):",
        "",
    ]
    lines += _table(
        ["Chỉ số", "BM25", "Dense", "Ghi chú"],
        [["Recall@1", _pick(retrieval.get("recall_at_1"), "bm25"),
          _pick(retrieval.get("recall_at_1"), "dense"), None],
         ["Recall@3", _pick(retrieval.get("recall_at_3"), "bm25"),
          _pick(retrieval.get("recall_at_3"), "dense"), None],
         ["Recall@5", _pick(retrieval.get("recall_at_5"), "bm25"),
          _pick(retrieval.get("recall_at_5"), "dense"),
          f"union={_fmt(retrieval.get('union_recall_at_5'))}"],
         ["MRR", _pick(retrieval.get("mrr"), "bm25"), _pick(retrieval.get("mrr"), "dense"),
          None],
         ["Tỷ lệ reachable trong ngân sách", _pick(retrieval.get("reachable_rate"), "bm25"),
          _pick(retrieval.get("reachable_rate"), "dense"),
          f"ngân sách: bm25={_fmt(_pick(retrieval.get('reach_budget'), 'bm25'))}, "
          f"dense={_fmt(_pick(retrieval.get('reach_budget'), 'dense'))}"],
         ["Top-k overlap / diff rate", None, None,
          f"{_fmt(retrieval.get('topk_overlap_bm25_dense'))} / "
          f"{_fmt(retrieval.get('diff_rate'))}"]])
    lines += [
        "",
        f"**Bảng 2b — Reader** (nguồn: outputs/models/qa_reader/reader_eval.json; "
        f"n={_fmt(oracle.get('n'))}, ngưỡng null-score-diff={_fmt(threshold)}, "
        f"model hash={_fmt(reader_hash)}):",
        "",
    ]
    lines += _table(
        ["Chỉ số", "Oracle-context", "Retrieved-context", "Ghi chú"],
        [["EM (câu trả lời được)", oracle.get("em_answerable"), retrieved.get("em_answerable"),
          "trần của agent khi reader thấy ngữ cảnh vàng"],
         ["F1 (câu trả lời được)", oracle.get("f1_answerable"), retrieved.get("f1_answerable"),
          "bằng chứng thực tế agent thấy"],
         ["EM (toàn bộ)", oracle.get("em_overall"), retrieved.get("em_overall"),
          "gồm cả câu không có đáp án"],
         ["No-answer accuracy", oracle.get("no_answer_accuracy"),
          retrieved.get("no_answer_accuracy"), None],
         ["False answer rate (impossible)", oracle.get("false_answer_rate"),
          retrieved.get("false_answer_rate"), "càng thấp càng tốt"],
         ["Provenance failures", oracle.get("provenance_failures"),
          retrieved.get("provenance_failures"), "span phải nằm trong context đã thấy"],
         ["Đã trả lời (answerable)", oracle.get("answered_answerable"),
          retrieved.get("answered_answerable"),
          f"trên {_fmt(oracle.get('n_answerable'))} câu trả lời được"]])
    lines += [
        "",
        "Acceptance của P3 (`train_reader`): "
        + ("**có kiểm tra KHÔNG đạt — " + ", ".join(failed_checks) + "**; reader chỉ được coi "
           "là đạt khi mọi kiểm tra pass (xem reader_eval.json)"
           if failed_checks else
           f"mọi kiểm tra đạt ({len(acceptance)} mục)" if acceptance else
           "chưa có reader_eval.json để đối chiếu"),
    ]
    lines += [
        "",
        f"**Bảng 3 — Laya CE vs RLCD** (nguồn: outputs/manifests/P5_train_laya_*.json; các chỉ "
        f"số teacher-action accuracy/ECE cần đánh giá riêng — chưa có artifact, ghi NOT_RUN):",
        "",
    ]
    lines += _table(["Checkpoint", "Regime", "Loss", "Data SHA256", "Train N", "Calib N",
                     "Temperature", "Epoch loss", "Walltime (s)", "Peak VRAM (MiB)", "Status"],
                    _laya_rows(sources))
    lines += [
        "",
        "**Bảng 4 — E00–E07 trên val_strict** (nguồn: reports/tables/baseline_results_val.csv; "
        "mỗi ô N/A là chưa đo, không phải 0):",
        "",
    ]
    lines += _table(["Mã", "Policy", "Seed", "N", "Mean reward ↑", "Token F1 ↑",
                     "Answerable F1 ↑", "WAR(câu không đáp án) ↓", "Coverage ↑",
                     "Chi phí TB ↓", "Status", "Nguồn"],
                    _experiment_table(rows, "val_strict"))
    lines += [
        "",
        "**Bảng 4b — E00–E07 trên test_strict** (chỉ có ý nghĩa sau khi khoá lựa chọn):",
        "",
    ]
    lines += _table(["Mã", "Policy", "Seed", "N", "Mean reward ↑", "Token F1 ↑",
                     "Answerable F1 ↑", "WAR(câu không đáp án) ↓", "Coverage ↑",
                     "Chi phí TB ↓", "Status", "Nguồn"],
                    _experiment_table(rows, "test_strict"))
    b5_metrics = (sources.b5_row or {}).get("row") or {}
    b5_summary = (", ".join(f"{key}={_fmt(b5_metrics.get(key))}"
                            for key in ("variant", "tau_answer", "ece", "brier", "coverage",
                                        "selective_risk", "n_episodes"))
                  if b5_metrics else "CHƯA CÓ")
    lines += [
        "",
        f"**Dòng huấn luyện B5** (nguồn: reports/results_b5_row.json = {b5_summary}; nếu "
        "train_correctness đã nối dòng vào reports/results.csv thì dòng đó được giữ nguyên "
        "trong file — split val_model_selection, khác split của E03):",
        "",
        "**Bảng 5 — Ablations pre-registered** (`03_EXPERIMENTS/00` §4):",
        "",
    ]
    lines += _table(["Mã", "Nội dung", "Trạng thái"],
                    [[code, text, "NOT_RUN (không có artifact ablation trong repo) "]
                     for code, text in ABLATIONS])
    lines += [
        "",
        "**Hình** (nguồn: reports/plots/):",
        "",
    ]
    lines += _table(["Tên", "Trạng thái", "Đường dẫn"], _plot_inventory(sources))
    lines += [
        "",
        "Mọi bảng trên nêu N, split, seed (khi có) và đường dẫn artifact; không có số nào được "
        "nhập tay.",
        "",
        "## 8. Phân tích",
        "",
        f"- {_delta(rows, 'val_strict', 'E06', 'E03')}",
        f"- {_delta(rows, 'val_strict', 'E06', 'E01')}",
        f"- {_delta(rows, 'val_strict', 'E06', 'E02')}",
        f"- {_delta(rows, 'val_strict', 'E05', 'E04')} (RLCD vs CE ở tầng prior)",
        "- Khi RL chỉ giảm chi phí nhưng không tăng chất lượng trả lời, kết luận là trade-off "
        "theo chi phí, không gọi “tốt hơn toàn diện”.",
        "- Câu hỏi không có đáp án: xem WAR và unanswerable_abstain_accuracy trong Bảng 4; Dense "
        "hữu ích nhất khi BM25 bỏ lỡ diễn đạt khác từ (reports/tables/bm25_vs_dense_examples.md).",
        "- Trần của reader/retrieval: so sánh oracle-context so với retrieved-context ở Bảng 2; "
        "nếu retrieved-context thấp hơn, lỗi nằm ở truy hồi chứ không phải reader.",
        "",
        "## 9. Failure cases thực tế",
        "",
        f"{failure_note}",
        "",
    ]
    if failures:
        lines += _table(["qid", "Split", "Exp", "Seed", "Chuỗi hành động", "Số lượt tìm",
                         "F1", "Câu trả lời cuối", "Nguyên nhân", "Nguồn"], failures)
        lines.append("")
    lines += [
        "- Không trích dẫn toàn văn tài liệu dài trong báo cáo; chỉ nêu span trả lời và qid để "
        "tra cứu trong artifact dự đoán.",
        "",
        "## 10. Đánh giá giả thuyết",
        "",
    ]
    lines += _table(["Giả thuyết", "RQ", "Kết luận", "Bằng chứng"], _hypotheses(rows, sources))
    lines += [
        "",
        "- Nếu chỉ có một seed hoặc N nhỏ, kết quả được ghi rõ là exploratory và thiếu "
        "statistical power; CI vắng mặt là do không thể tính đúng, không phải bỏ qua.",
        "",
        "## 11. Hạn chế",
        "",
        "- Nhãn ViQuAD2.0 gắn với ngữ cảnh (context-bounded), không phải chân lý mở; abstain "
        "đúng theo corpus chứ không theo thế giới.",
        "- Giấy phép dữ liệu chưa xác thực độc lập; nguy cơ contamination với các checkpoint "
        "QA tiếng Việt có sẵn (đã tránh dùng trong primary path).",
        "- Chất lượng reader có thể chi phối trần của agent; reward là heuristic có thiên lệch.",
        "- Một GPU 24GB, context cắt ngắn, số seed hạn chế; policy học trên phân phối hẹp, "
        "khả năng chuyển sang web chưa kiểm chứng.",
        "",
        "## 12. Đạo đức / an toàn",
        "",
        "- Câu trả lời luôn neo vào tài liệu đã thấy; không thực thi chỉ dẫn nằm trong văn bản "
        "người dùng/tài liệu; không chia sẻ bí mật; không xuất bản dữ liệu/checkpoint khi chưa "
        "rà soát license.",
        "- Xác suất chọn hành động không được trình bày như bảo đảm đáp án đúng.",
        "",
        "## 13. Tái lập",
        "",
        "- Lệnh đầy đủ: xem REPRODUCE.md (tạo .venv → chạy stage → đánh giá → sinh báo cáo) và "
        "`bash scripts/reproduce.sh --run-id <id> --mode eval` để kiểm chứng lại một run đã ghi "
        "bằng hash + so sánh dự đoán trên subset (không train lại).",
        f"- Hash nguồn chính: runs/index.jsonl sha256={_fmt(registry_digest)} "
        f"(không gồm dòng P9_report của chính báo cáo); "
        f"reader model hash={_fmt(reader_hash)}; policy dir="
        f"{_fmt((sources.rl_config or {}).get('run_id'))}.",
        "- Phụ thuộc: .venv + pyproject.toml; dữ liệu thô tải từ HF revision đã pin; toàn bộ "
        "artifact kèm SHA256 trong reports/artifact_manifest.sha256.",
        "- Kiểm định: `.venv/bin/python -m pytest tests/ -q` (unit/integration, fixture tổng hợp); "
        "stage này không chạy test và không chạy GPU.",
        "- Nondeterminism đã biết: kernel CUDA và thứ tự giảm gradient; eval greedy + no_grad + "
        "seed cố định nên sai khác chỉ ở mức số học và được so với tolerance khi verify.",
        "",
        "## 14. Kết luận",
        "",
        "- Đóng góp đo được: môi trường episodic tiếng Việt tái lập được với trace hành động, "
        "mô hình quyết định Laya (CE/RLCD) so sánh trực tiếp, tầng REINFORCE thực và bộ đánh giá "
        "E00–E07 + báo cáo truy vết từng số về file nguồn.",
        "- Thành công/thất bại cụ thể: xem Bảng 4 và mục 10; kết quả âm hoặc chưa chạy hết vẫn "
        "được báo nguyên trạng (NOT_RUN/FAILED) thay vì suy diễn.",
        "- Hướng tiếp theo: PPO/group-relative, viết lại truy vấn động, mở rộng open-world với "
        "benchmark và rủi ro khác.",
        "",
        "## 15. Tài liệu tham khảo",
        "",
        "Nguồn chính (đã đối chiếu ngày 08/10/2026; xem "
        "`06_REFERENCES/00_SOURCES_VERIFIED_AS_OF_2026_10_08.md`):",
        "",
        "1. Laya multilingual checkpoint — https://huggingface.co/convaiinnovations/laya-multilingual "
        "(truy cập 08/10/2026).",
        "2. Laya fine-tuning CLI docs — https://nandhakishorm.github.io/laya/finetune/ (08/10/2026).",
        "3. XLM-RoBERTa-base — https://huggingface.co/FacebookAI/xlm-roberta-base (08/10/2026).",
        "4. multilingual-E5-small — https://huggingface.co/intfloat/multilingual-e5-small "
        "(08/10/2026).",
        "5. UIT-ViQuAD2.0 (HF mirror) — https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0 "
        "(08/10/2026).",
        "6. UIT-ViQuAD2.0 shared task — https://arxiv.org/abs/2203.11400 (08/10/2026).",
        "7. Williams, R. J. (1992). Simple statistical gradient-following algorithms for "
        "connectionist reinforcement learning. Machine Learning 8:229–256. "
        "https://doi.org/10.1007/BF00992696",
        "8. DeepSeekMath (GRPO) — https://arxiv.org/abs/2402.03300 (08/10/2026).",
        "9. Adaptive-RAG (NAACL 2024) — https://aclanthology.org/2024.naacl-long.389/ (08/10/2026).",
        "10. Search-R1 — https://arxiv.org/abs/2503.09516 (08/10/2026).",
        "",
        "## Phụ lục",
        "",
        "- Lệnh từng stage: xem FINAL_HANDOFF.md (bảng “What ran”) và REPRODUCE.md.",
        "- Reward/env/config đầy đủ: configs/base.yaml + snapshot trong outputs/manifests/.",
        "- Action schema: ANSWER=0, SEARCH_BM25=1, SEARCH_DENSE=2, ABSTAIN=3 "
        "(`src/decision/schema.py`).",
        "- Cache manifests: outputs/models/qa_reader/reader_qa_state_cache_*.json, "
        "outputs/models/laya_probs_cache/laya_probs.meta.json.",
        "- Nguồn của mọi bảng trong báo cáo này:",
        "",
    ]
    lines += _table(["Artifact", "Trạng thái", "SHA256/Đường dẫn"], _sources_block(sources))
    lines += ["",
              "- artifact_manifest.sha256 được ghi TRƯỚC khi dòng P9_report SUCCEEDED của chính "
              "lần chạy được nối vào runs/index.jsonl, nên dòng registry trong manifest phản "
              "ánh file ngay trước khi báo cáo hoàn tất (registry là log chỉ-nối-thêm).",
              "- Test access log: docs/TEST_ACCESS_LOG.md; registry: outputs/runs/index.jsonl.",
              "",
              "**Tự kiểm cuối:** báo cáo này do `python -m src.pipeline.report` sinh; nếu thiếu "
              "nguồn cho một số, chương trình ghi NOT_RUN/N/A thay vì số ước lượng, và từ chối "
              "kết xuất nếu còn placeholder dạng FROM_ARTIFACT trong nguồn.", ""]
    text = "\n".join(lines)
    if "[FROM_ARTIFACT:" in text:
        raise RuntimeError("report still contains an unfilled [FROM_ARTIFACT: …] placeholder")
    return text


def markdown_to_plaintext(markdown: str) -> str:
    """Deterministic markdown→text: keep Unicode, drop syntax, underline headings."""
    out: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            out.append(title)
            out.append("=" * len(title))
            continue
        stripped = line.replace("**", "").replace("`", "")
        if stripped.strip() == "---":
            out.append("-" * 40)
            continue
        out.append(stripped)
    return "\n".join(out).rstrip() + "\n"


def render_model_card(sources: Sources, rows: Sequence[Mapping[str, Any]]) -> str:
    split_manifest = sources.split_manifest or {}
    strict_rows = (split_manifest.get("strict_split") or {}).get("rows") or {}
    reader_hash = (sources.reader_contract or {}).get("reader_model_hash")
    rl = sources.rl_config or {}
    lines = [
        "# MODEL CARD — Vietnamese Adaptive QA RL",
        "",
        "## Model identifier",
        "",
        "`Vietnamese Adaptive QA RL` — pipeline tổng hợp (retrieval BM25+E5, reader XLM-R SFT, "
        "mô hình quyết định Laya CE/RLCD, policy REINFORCE).",
        "",
        "## Intended use",
        "",
        "Tác tử hỏi đáp trích xuất tiếng Việt trên kho evidence giới hạn đã index; trả lời kèm "
        "trích dẫn hoặc từ chối. Nghiên cứu học thuật/đồ án cuối kỳ — **không** thay thế xác "
        "minh thông tin thế giới mở, tư vấn pháp lý/y tế hay ra quyết định quan trọng.",
        "",
        "## Architecture and training",
        "",
        f"- Reader: `outputs/models/qa_reader` (model hash `{_fmt(reader_hash)}`), base "
        f"FacebookAI/xlm-roberta-base, SFT trên train_strict "
        f"(N={_fmt(strict_rows.get('train_strict'))}), đánh giá `reader_eval.json` "
        f"({'có' if sources.reader_eval else 'CHƯA CÓ'}).",
        f"- Retriever: BM25 (k1={_fmt(sources.cfg['retrieval']['bm25']['k1'])}, "
        f"b={_fmt(sources.cfg['retrieval']['bm25']['b'])}) + E5 "
        f"`{_fmt(sources.cfg['retrieval']['dense']['model'])}`; corpus manifest "
        f"`data/processed/split_manifest.json`.",
        f"- Laya: `outputs/models/laya_ce`, `outputs/models/laya_rlcd` "
        f"(manifest {'có' if sources.stage_manifests else 'CHƯA CÓ'}), regime "
        f"`{'frozen' if sources.cfg['laya'].get('freeze_encoder') is False else _fmt(sources.cfg['laya'].get('freeze_encoder'))}`, "
        f"teacher targets suy từ reward (soft, temperature="
        f"{_fmt(sources.cfg['teacher'].get('temperature'))}).",
        f"- RL policy: `outputs/models/rl_policy` (run_id="
        f"`{_fmt(rl.get('run_id'))}`, updates={_fmt(rl.get('updates_ran'))}, "
        f"val reward={_fmt(rl.get('val_reward'))}, reward terms trong configs/base.yaml).",
        "- **Không phải TypeSafe Jev weights**; không gọi là fine-tune Jev end-to-end.",
        "",
        "## Data and licensing",
        "",
        f"- HF dataset `{_fmt((sources.source_manifest or {}).get('repo_id') or 'taidng/UIT-ViQuAD2.0')}`"
        f" revision `{_fmt((split_manifest.get('source') or {}).get('resolved_revision'))}`; "
        f"strict split theo title-group với số dòng {_fmt(strict_rows)}.",
        f"- Duplicate/leakage + audit âm: `data/processed/split_manifest.json`, "
        f"`data/processed/ambiguous_negatives.json`.",
        f"- `license_status: UNVERIFIED` (card_license="
        f"{_fmt((sources.source_manifest or {}).get('card_license'))}); cấm xuất bản lại raw "
        "dataset/checkpoint khi chưa rà soát.",
        "",
        "## Evaluation",
        "",
    ]
    lines += _table(["Mã", "Split", "N", "Mean reward ↑", "F1 ↑", "Coverage ↑",
                     "Chi phí TB ↓", "Status"],
                    [[row.get("exp_id"), row.get("split"), row.get("n_episodes"),
                      row.get("mean_episode_reward"), row.get("token_f1"), row.get("coverage"),
                      row.get("avg_retrieval_cost"), row.get("status")]
                     for row in rows if row.get("exp_id") in EXP_IDS])
    lines += [
        "",
        "- CI theo cụm title-group chỉ hiển thị khi artifact có cột CI tương ứng; "
        "nhãn/hạn chế contamination: reader SFT chỉ dùng train_strict; test mở một lần sau freeze "
        "(docs/TEST_ACCESS_LOG.md).",
        "",
        "## Resource needs and restrictions",
        "",
        f"- GPU đo được: {_fmt(((sources.gpu or {}).get('gpus') or [{}])[0].get('name'))} "
        f"({_fmt(((sources.gpu or {}).get('gpus') or [{}])[0].get('memory_total_mib'))} MiB); "
        f"peak VRAM train RL={_fmt((sources.rl_report or {}).get('peak_vram_mib'))} MiB; "
        f"RAM={_fmt(((sources.system or {}).get('memory_mib') or {}).get('MemTotal'))} MiB.",
        f"- Ngân sách tìm mặc định: bootstrap {_fmt(sources.cfg['retrieval'].get('initial_bootstrap'))}"
        f" + {_fmt(sources.cfg['retrieval'].get('optional_searches'))} lượt tùy chọn; latency "
        "inference đo trong `timings` của demo CLI (src.inference.cli).",
        "- Phụ thuộc code/license: xem pyproject.toml; không kèm credential.",
        "",
        "## Security, privacy, societal impacts",
        "",
        "- Văn bản tài liệu được coi là đầu vào không tin cậy; không thực thi chỉ dẫn trong văn "
        "bản; không lưu bí mật; luôn kèm trích dẫn hoặc từ chối.",
        "- Mô hình có thể trả lời sai một cách tự tin khi retrieval/reader sai; hiệu năng bị "
        "giới hạn bởi corpus và thiết kế reward.",
        "",
    ]
    return "\n".join(lines)


def render_reproduce(sources: Sources) -> str:
    profile = sources.profile
    return "\n".join([
        "# REPRODUCE — tái lập đánh giá và báo cáo",
        "",
        f"Profile: `{profile}`. Không train lại chỉ để kiểm chứng metric: dùng "
        "`bash scripts/reproduce.sh --run-id <id> --mode eval`.",
        "",
        "```bash",
        "# 0. Môi trường",
        "python3 -m venv .venv && .venv/bin/pip install -e .",
        "bash scripts/preflight.sh",
        "",
        "# 1-3. Dữ liệu và retrieval",
        f".venv/bin/python -m src.pipeline.download --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.audit_split --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.build_retrieval --config configs/{profile}.yaml",
        "",
        "# 4-6. Reader + cache trạng thái + teacher",
        f".venv/bin/python -m src.pipeline.train_reader --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.cache_reader --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.make_teacher --config configs/{profile}.yaml",
        "",
        "# 7-9. Laya (CE, RLCD) + cache prior",
        f".venv/bin/python -m src.pipeline.train_laya --config configs/{profile}.yaml --loss soft-ce",
        f".venv/bin/python -m src.pipeline.train_laya --config configs/{profile}.yaml --loss rlcd",
        f".venv/bin/python -m src.pipeline.cache_laya --config configs/{profile}.yaml",
        "",
        "# 10. REINFORCE",
        f".venv/bin/python -m src.pipeline.train_rl --config configs/{profile}.yaml",
        "",
        "# 11. Baseline B5 (calibration) rồi E00-E07 trên val; test chỉ sau khi khoá lựa chọn",
        f".venv/bin/python -m src.pipeline.train_correctness --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.eval_baselines --config configs/{profile}.yaml "
        "--split val_strict",
        "# (chọn checkpoint B8 tốt nhất trên val — ghi best_selection.json) rồi:",
        f".venv/bin/python -m src.pipeline.select_rl --config configs/{profile}.yaml",
        f".venv/bin/python -m src.pipeline.eval_baselines --config configs/{profile}.yaml "
        "--split test_strict",
        "",
        "# 12. Báo cáo",
        f".venv/bin/python -m src.pipeline.report --config configs/{profile}.yaml",
        "",
        "# Demo suy luận từ artifact đã lưu",
        ".venv/bin/python -m src.inference.cli --question 'Phạm Văn Đồng bắt đầu giữ chức "
        "Thủ tướng CHXHCN Việt Nam từ năm nào?' --config configs/base.yaml --show-trace",
        "",
        "# 13. Kiểm chứng một run đã ghi (hash + subset predictions, không train lại)",
        "bash scripts/reproduce.sh --run-id <run_id> --mode eval",
        "",
        "# 14. Test",
        ".venv/bin/python -m pytest tests/ -q",
        "```",
        "",
        f"- Điều kiện phần cứng: 1 GPU ≥ 20GB VRAM (đo được: "
        f"{_fmt(((sources.gpu or {}).get('gpus') or [{}])[0].get('name'))}), RAM ≥ 12GB, "
        f"disk trống ≥ 20GB.",
        "- Hash để đối chiếu: reports/artifact_manifest.sha256; log registry: "
        "outputs/runs/index.jsonl.",
        "- Nondeterminism: CUDA kernels + thứ tự giảm; eval greedy/no_grad/seed cố định; "
        "so sánh lại bằng tolerance khi verify.",
        "",
    ])


def render_test_access_log(sources: Sources) -> str:
    """Honest access log: what the artifacts can and cannot prove about test usage."""
    rows = []
    for row in sources.registry:
        text = json.dumps(row, ensure_ascii=False)
        if "test_strict" not in text and row.get("split") != "test_strict":
            continue
        touches_labels = (row.get("stage") in ("P7_baselines",)
                          and (row.get("split") == "test_strict"
                               or "test_strict" in text and row.get("stage") == "P7_baselines"))
        if row.get("stage") == "P3_reader_cache":
            touches_labels = "test_strict" in json.dumps(row.get("splits") or {}, ensure_ascii=False)
        rows.append([row.get("ts"), row.get("run_id"), row.get("stage"), row.get("status"),
                     "có" if touches_labels else "không (chỉ tạo/đọc dữ liệu test)",
                     row.get("profile")])
    selection_ts = None
    if sources.selection_path is not None and sources.selection_path.is_file():
        selection_ts = (sources.selection or {}).get("created_utc") \
            or (sources.selection or {}).get("ts") \
            or _dt.datetime.fromtimestamp(
                sources.selection_path.stat().st_mtime,
                tz=_dt.timezone.utc).isoformat(timespec="seconds")
    test_rows = [row for row in rows if row[4] == "có"]
    if not test_rows:
        verdict = ("Chưa có stage nào đọc nhãn test_strict theo registry — test split chưa được "
                   "mở cho đánh giá.")
    elif selection_ts is None:
        verdict = ("**KHÔNG THỂ CHỨNG MINH**: có bản ghi dùng test_strict "
                   f"({test_rows[0][1]}) nhưng không tìm thấy outputs/manifests/best_selection.json, "
                   "nên không chứng minh được test chỉ được mở sau khi khoá lựa chọn. Mọi kết quả "
                   "test phải được coi là exploratory.")
    elif str(selection_ts) <= str(min(row[0] or "" for row in test_rows)):
        verdict = (f"Đã khoá lựa chọn ({selection_ts}) trước bản ghi test sớm nhất "
                   f"({min(row[0] or '' for row in test_rows)}); timestamp lấy từ manifest, "
                   "không chống giả mạo.")
    else:
        verdict = ("**VI PHẠM THỨ TỰ**: best_selection.json "
                   f"({selection_ts}) được ghi SAU bản ghi test đầu tiên; kết quả test không còn "
                   "là held-out sạch.")
    lines = [
        "# TEST ACCESS LOG",
        "",
        "Ghi lại mọi lần artifact chạm vào split `test_strict` (tạo dữ liệu test, cache, hoặc "
        "đánh giá trên nhãn test). Không xoá dòng cũ. Nguồn: outputs/runs/index.jsonl.",
        "",
        f"Kết luận: {verdict}",
        "",
    ]
    lines += _table(["Thời điểm (UTC)", "run_id", "Stage", "Status", "Dùng nhãn test?",
                     "Profile"], rows) if rows else ["(chưa có bản ghi nào chạm test_strict)"]
    lines += [
        "",
        f"- best_selection.json: {'có' if selection_ts else 'CHƯA CÓ'} "
        f"({_fmt(selection_ts)})",
        "- Timestamp lấy từ file do các stage ghi; không có cơ chế chống sửa đổi — đây là mức "
        "chứng minh tốt nhất mà artifact hiện có thể đưa ra.",
        "",
    ]
    return "\n".join(lines)


def render_handoff(sources: Sources, rows: Sequence[Mapping[str, Any]],
                   registry_digest: str | None) -> str:
    statuses = sources.stages
    done = sum(1 for info in statuses.values() if info["status"] == "SUCCEEDED")
    total = len(statuses)
    blocked = [key for key, info in statuses.items()
               if info["status"] in ("FAILED", "INTERRUPTED")]
    if done == total:
        status = "DONE"
    elif done == 0:
        status = "BLOCKED"
    else:
        status = "PARTIAL"
    succeeded = [row for row in rows if row.get("status") == "SUCCEEDED"]
    non_rl = [row for row in succeeded if row.get("exp_id") in
              ("E00", "E01", "E02", "E03", "E04", "E05")]
    best_non_rl = max(non_rl, key=lambda row: row.get("token_f1") or -1, default=None)
    best_rl = _row(rows, "E06", "val_strict")
    lines = [
        "# FINAL HANDOFF",
        "",
        "## Executive status",
        "",
        f"- Trạng thái: **{status}** ({done}/{total} stage SUCCEEDED)",
        f"- Thời điểm sinh: {sources.now}; profile: {sources.profile}",
        f"- Git SHA: {git_sha(REPO_ROOT) or 'không xác định'}; registry sha256: "
        f"{_fmt(registry_digest)} (không gồm dòng P9_report của chính báo cáo)",
        f"- Tiêu chí chọn checkpoint: mean episodic reward trên val (freeze: "
        f"{'có' if sources.selection else 'CHƯA CÓ'})",
        "",
        "## What ran",
        "",
    ]
    lines += _table(["Phía", "Stage", "Trạng thái", "Lệnh", "Bằng chứng", "Artifact"],
                    [[stage.phase, stage.key, source_status["status"],
                      stage.command.format(profile=sources.profile, split="val_strict"),
                      source_status.get("run_id"), ", ".join(stage.outputs)]
                     for stage, source_status in ((stage, statuses[stage.key])
                                                  for stage in STAGES)])
    lines += [
        "",
        "## Model artifacts",
        "",
    ]
    model_entries = [
        ("reader", resolve(sources.cfg, "models") / "qa_reader" / "model.safetensors"),
        ("laya_ce", resolve(sources.cfg, "models") / "laya_ce" / "model.safetensors"),
        ("laya_rlcd", resolve(sources.cfg, "models") / "laya_rlcd" / "model.safetensors"),
        ("policy", resolve(sources.cfg, "models") / "rl_policy" / "policy.safetensors"),
        ("policy_scaler", resolve(sources.cfg, "models") / "rl_policy" / "scaler.json"),
    ]
    lines += _table(["Artifact", "Đường dẫn", "Bytes", "SHA256"],
                    [[name, str(path), path.stat().st_size if path.is_file() else None,
                      sha256_file(path) if path.is_file() else None]
                     for name, path in model_entries])
    lines += [
        "",
        "- Load-back: reader được load lại trong P3 (`reader_inference.json` khớp hash); policy "
        "được load lại trong P7 (`src/eval/policies.load_rl_policy`). Stage báo cáo này "
        "**không** load weight.",
        "",
        "## Results",
        "",
        f"- RL (E06) vs baseline phi-RL tốt nhất theo F1: "
        + (f"E06 F1={_fmt(best_rl.get('token_f1'))} so với {best_non_rl.get('exp_id')} "
           f"F1={_fmt(best_non_rl.get('token_f1'))} trên {best_rl.get('split')} "
           f"(reward {_fmt(best_rl.get('mean_episode_reward'))} vs "
           f"{_fmt(best_non_rl.get('mean_episode_reward'))})"
           if best_rl is not None and best_non_rl is not None else
           "chưa đủ dữ liệu val để so sánh (xem reports/results.csv)"),
        "- Bảng đầy đủ + nguồn: reports/results.csv, reports/results.json, REPORT.md mục 7–8.",
        "- Thất bại/điểm yếu: xem REPORT.md mục 9 và danh sách stage FAILED/INTERRUPTED dưới đây.",
        "",
        "## Run now",
        "",
        "```bash",
        "python3 -m venv .venv && .venv/bin/pip install -e .",
        f".venv/bin/python -m src.pipeline.report --config configs/{sources.profile}.yaml",
        ".venv/bin/python -m src.inference.cli --question '<câu hỏi>' "
        f"--config configs/{sources.profile}.yaml --show-trace",
        "bash scripts/reproduce.sh --run-id <run_id> --mode eval",
        "```",
        "",
        "## Limitations / blockers",
        "",
    ]
    not_run = [f"{stage.phase}/{stage.key}: {statuses[stage.key]['status']} — "
               f"{_fmt(statuses[stage.key].get('reason'))}"
               for stage in STAGES if statuses[stage.key]["status"] != "SUCCEEDED"]
    lines += [f"- {item}" for item in not_run] or ["- (không có)"]
    if blocked:
        lines.append("- Stage FAILED/INTERRUPTED cần chạy lại trước khi tin cậy kết quả: "
                     + ", ".join(blocked))
    lines += [
        "- Lệnh phục hồi: chạy lại đúng stage với `--resume` (xem REPRODUCE.md); run lỗi được "
        "giữ nguyên trong registry.",
        "",
        "## Safety check",
        "",
        f"- License dữ liệu: UNVERIFIED (card_license="
        f"{_fmt((sources.source_manifest or {}).get('card_license'))}); không xuất bản lại "
        "dataset/weight khi chưa rà soát.",
        "- Report và manifest không chứa credential (sinh từ artifact, không nhúng token); "
        "quyền file theo umask mặc định của repo.",
        "- Test access log: docs/TEST_ACCESS_LOG.md; mọi số trong REPORT.md truy về file nguồn "
        "qua cột 'source' trong results.csv và mục Phụ lục.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- manifest/zip

MANIFEST_ROOTS = ("outputs/models", "outputs/predictions", "outputs/manifests", "outputs/runs",
                  "data/processed", "data/scenarios", "data/indexes", "data/cache",
                  "reports", "docs")


def write_manifest(root: Path, manifest_path: Path) -> Path:
    """SHA256 of every artifact under :data:`MANIFEST_ROOTS`; the manifest excludes itself."""
    entries: list[tuple[str, str]] = []
    for relative in MANIFEST_ROOTS:
        base = root / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.name == manifest_path.name:
                continue
            if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                continue
            entries.append((sha256_file(path), str(path.relative_to(root))))
    entries.sort(key=lambda item: item[1])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(f"{digest}  {name}\n" for digest, name in entries),
                             encoding="utf-8")
    return manifest_path


def write_zip(root: Path, zip_path: Path) -> tuple[Path | None, str]:
    """``--zip``: metadata archive only (no weights), refused unless disk headroom is 2x."""
    files: list[Path] = []
    for relative in ZIP_INCLUDE_DIRS:
        base = root / relative
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            if any(part in ZIP_EXCLUDE_NAMES for part in path.parts):
                continue
            if path.suffix in (".zip", ".pyc"):
                continue
            files.append(path)
    files += [root / name for name in ZIP_INCLUDE_FILES if (root / name).is_file()]
    total = sum(path.stat().st_size for path in files)
    free = shutil.disk_usage(root).free
    if free < total * 2 + 512 * 1024 * 1024:
        return None, (f"từ chối --zip: cần ~{total * 2 / 1e6:.0f}MB trống (2x dữ liệu + 512MB "
                      f"margin), chỉ còn {free / 1e6:.0f}MB")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(root)))
    return zip_path, f"{len(files)} tệp, {zip_path.stat().st_size / 1e6:.1f}MB"


# ----------------------------------------------------------------------------------- run

def run(*, cfg: Mapping[str, Any], profile: str, zip_handoff: bool = False,
        now: str | None = None, registry: RunRegistry | None = None) -> dict[str, Any]:
    """Generate every report product; idempotent for a fixed ``now``."""
    now = now or now_iso()
    sources = load_sources(cfg, profile, now)
    registry = registry or RunRegistry(resolve(cfg, "runs") / "index.jsonl")
    run_id = f"p9-report-{profile}-{now.replace(':', '')}"
    registry.append(run_id=run_id, stage=STAGE, status="RUNNING", profile=profile)
    try:
        rows = build_results_rows(sources)
        results_csv = write_results_csv(sources.reports / RESULTS_CSV, rows)
        results_json = write_results_json(sources.reports / RESULTS_JSON, rows, sources)
        failures, failure_note = _failure_cases(sources)
        report_md = sources.reports / REPORT_MD
        report_md.write_text(
            render_report(sources, rows, failures, failure_note, sources.registry_sha),
            encoding="utf-8")
        (sources.reports / REPORT_TXT).write_text(
            markdown_to_plaintext(report_md.read_text(encoding="utf-8")), encoding="utf-8")
        (sources.reports / MODEL_CARD_MD).write_text(render_model_card(sources, rows),
                                                     encoding="utf-8")
        (sources.reports / REPRODUCE_MD).write_text(render_reproduce(sources), encoding="utf-8")
        access_log = REPO_ROOT / TEST_ACCESS_LOG
        access_log.parent.mkdir(parents=True, exist_ok=True)
        access_log.write_text(render_test_access_log(sources), encoding="utf-8")
        (sources.reports / HANDOFF_MD).write_text(
            render_handoff(sources, rows, sources.registry_sha), encoding="utf-8")
        manifest = write_manifest(REPO_ROOT, sources.reports / MANIFEST_FILE)
        artifacts = [results_csv, results_json, report_md, sources.reports / REPORT_TXT,
                     sources.reports / MODEL_CARD_MD, sources.reports / REPRODUCE_MD,
                     access_log, sources.reports / HANDOFF_MD, manifest]
        zip_info: str | None = None
        if zip_handoff:
            zipped, zip_info = write_zip(REPO_ROOT, sources.reports / ZIP_FILE)
            if zipped is not None:
                artifacts.append(zipped)
        registry.record_artifact(run_id, STAGE, artifacts, profile=profile,
                                 n_rows=len(rows), zip=zip_info)
    except Exception as exc:
        registry.append(run_id=run_id, stage=STAGE, status="FAILED", profile=profile,
                        error=f"{type(exc).__name__}: {exc}")
        raise
    n_succeeded = sum(1 for row in rows if row.get("status") == "SUCCEEDED")
    return {"run_id": run_id, "stage": STAGE, "profile": profile,
            "n_rows": len(rows), "n_succeeded": n_succeeded,
            "reports_dir": str(sources.reports), "zip": zip_info,
            "stages": {key: info["status"] for key, info in sources.stages.items()}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.report",
        description="P9: generate REPORT.md/REPORT.txt, results.csv/json, MODEL_CARD.md, "
                    "REPRODUCE.md, FINAL_HANDOFF.md and the artifact manifest from raw artifacts")
    add_profile_args(parser)
    parser.add_argument("--zip", action="store_true",
                        help="also write reports/handoff_metadata.zip (metadata only, "
                             "disk-guarded; off by default)")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    profile = args.profile or cfg["data"]["profile"]
    cfg["data"]["profile"] = profile
    summary = run(cfg=cfg, profile=profile, zip_handoff=args.zip)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    print(f"run_id={summary['run_id']} rows={summary['n_rows']} "
          f"succeeded={summary['n_succeeded']} -> {summary['reports_dir']}")
    skipped = [key for key, status in summary["stages"].items() if status != "SUCCEEDED"]
    if skipped:
        print("stages not SUCCEEDED (ghi NOT_RUN trong báo cáo): " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

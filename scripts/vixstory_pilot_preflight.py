"""CPU-only preflight for the VixSTORY 300/200/246 pilot (issue #30).

This script runs on CPU before any GPU work and answers one question: are the data,
the pinned model snapshot, the runtime dependencies and the machine resources in a
state where the cloud-reader + Laya + two-REINFORCE-checkpoint pilot can start?

What it does
------------
* Reads the *frozen* pilot manifest written by ``scripts/vixstory_pilot_manifest.py``
  and maps its split records to ``train`` / ``dev`` / ``validation`` QID lists. The
  manifest is treated as data, not as an imported API, so a changed record name does
  not break the preflight.
* Independently re-reads the pinned local VixSTORY snapshot (corpus, official train,
  official validation) and checks the manifest against it: row counts, every QID
  present in its split, no duplicates, the three splits disjoint. ``data/test.parquet``
  is never opened; its local presence is only observed.
* Measures the machine now: GPU VRAM via ``nvidia-smi``, system RAM via ``/proc/meminfo``,
  free disk; resolves the cached Laya weight files; probes which runtime
  dependencies are importable (it never imports the heavy ones).
* Estimates, and labels as estimates, the size of one episode's reachable-state tree
  (``src.retrieval.states.reachable_states`` over worst-case divergent rankings) and a
  char/token sample of the split text. The pilot *runtime* is reported as UNKNOWN: the only
  measured runs are ViQuAD medium runs whose sequence lengths differ, so scaling them by
  question count is an explicitly-labelled unsound cross-dataset projection, never a
  completion/timeout promise.

What it never does
------------------
No GPU training, no generation, no model or dataset download, no writes outside the
pilot namespace, and no secret logging: environment values and tokens are never read or
printed. Every number that is not measured here is reported under ``estimated`` with the
basis that produced it, and never as a measurement.

Output: ``preflight.json`` is written *only after the manifest exists*, into the pilot
namespace. Before that, the checks still run and print, but nothing is written. An existing
preflight is never overwritten: a regenerated report is written to the next free
``preflight.v<N>.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import pyarrow.parquet as pq

from src.retrieval.states import reachable_states
from src.utils.hashing import sha256_file, stable_seed
from src.utils.io import atomic_write_json

# --- pinned dataset (VixSTORY revision) -----------------------------------
DATASET_REPO = "ura-hcmut/VixSTORY"
REVISION = "19613082c7ef128647d561472d1e150ca878fb22"
CORPUS_FILE = "corpus/corpus.parquet"
TRAIN_FILE = "data/train.parquet"
VALIDATION_FILE = "data/validation.parquet"
FORBIDDEN_FILE = "data/test.parquet"  # never read
EXPECTED_ROWS = {"corpus": 3462, "train": 2553, "validation": 246}
# Keyed by the same role names as EXPECTED_ROWS / the ``paths`` mapping in ``load_snapshot``; the
# train/validation digests match the pilot manifest's ``source.files``.
EXPECTED_SHA256 = {
    "corpus": "5b1222dcdb0807544c9634b4f3816adc8eea8992ab0b83dffae08dc0db9bbf0a",
    "train": "7e0f41ce2896a66bc5ea4e4150303508fe8aa2787db6800c118e8830eb307dad",
    "validation": "db356de42e18b1b4f3a9150852702c406188bbf95a2b1524b52084f310ca6ab7",
}

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
SNAPSHOT = HF_CACHE / "datasets--ura-hcmut--VixSTORY" / "snapshots" / REVISION

# --- pilot namespace (new outputs only) -----------------------------------
RUN_ID = "vixstory-pilot-300-2026-10-11"
PILOT_DIR = Path("outputs") / "experiments" / RUN_ID
MANIFEST_PATH = PILOT_DIR / "manifest.json"
OUT_PATH = PILOT_DIR / "preflight.json"

# --- model snapshots ------------------------------------------------------
# The reader is a cloud API (no local checkpoint); Laya is the only pinned local model.
MODEL_REPOS = ("convaiinnovations/laya-multilingual",)
WEIGHT_GLOBS = {
    "convaiinnovations/laya-multilingual": ("model.safetensors",),
}
# Pinned snapshot revision: Laya matches the P5 trainer manifests and docs/DECISIONS.md. A cache
# that resolved a different revision is a blocker, not a silent substitution.
MODEL_REVISIONS = {
    "convaiinnovations/laya-multilingual": "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67",
}

# --- dependencies ---------------------------------------------------------
CORE_DEPS = ("torch", "transformers", "accelerate", "sentence_transformers", "pyarrow", "numpy")
PARAM_EFFICIENT_DEPS = ("peft", "trl", "bitsandbytes")

# --- episode / estimation config (configs/base.yaml retrieval block) ------
EPISODE = {"bootstrap_tool": "bm25", "bootstrap_passages": 1,
           "passage_per_search_action": 1, "max_optional_search": 2, "tools": ("bm25", "dense")}
DEFAULT_CHARS_PER_TOKEN = 2.5  # heuristic for Vietnamese subword text; not a tokenizer run

# Reference wall-clock and peak VRAM measured on prior 5,000-question medium runs
# (reports/COMPUTE_AND_ENV.md, P5 trainer manifests). These are *measurements of other
# runs*, used only to scale a projection here; they are not measured for this pilot.
MEASURED_RUNS = {
    "laya_ce": {"questions": 5000, "seconds": 15890.7, "peak_vram_mib": 6139.4},
    "laya_rlcd": {"questions": 5000, "seconds": 16078.6, "peak_vram_mib": 6139.4},
    "reinforce": {"questions": 5000, "seconds": 1525.0},
}

# The only runtime evidence measured on VixSTORY itself: the frozen reader's generation *count*
# from outputs/experiments/2026-10-11-vixstory-reader-eval/. No wall-clock was recorded, so this
# cannot calibrate a stage runtime; it is reported as measured counts and nothing more.
MEASURED_READER_GENERATION = {
    "source": "outputs/experiments/2026-10-11-vixstory-reader-eval",
    "gold_questions_scored": 500,
    "search_boot2_generations": 1707,
    "search_boot2_unique_prompts": 5129,
    "wall_clock_seconds": None,
    "note": "counts only; no wall-clock was recorded, so reader generation does not calibrate the "
            "training-stage projection",
}

# --- pilot protocol -------------------------------------------------------
SPLIT_SIZES = {"train": 300, "dev": 200, "validation": 246}
SPLIT_ALIASES = {
    "train": ("train", "pilot", "pilot_300"),
    "dev": ("dev", "train_dev", "train_dev_200"),
    "validation": ("validation", "val", "validation_246"),
}

# Arm A (cloud reader + Laya + two REINFORCE checkpoints) minimums.
ARM_A_VRAM_MIN_MIB = 12 * 1024
# Host-RAM headroom target. This is a conservative *target*, not a measured stage RSS: the prior
# Laya medium runs peaked at 6.1 GiB VRAM (GPU side) and succeeded on this same 15.9 GiB host, but
# no per-stage host RSS was recorded. So available RAM below the target is a signal to measure
# (page cache is reclaimable), and only *total* RAM below the target is a hard deficit.
ARM_A_RAM_MIN_MIB = 8 * 1024
ARM_A_DISK_MIN_GB = 20.0


def _run(cmd: Sequence[str], timeout: int = 30) -> dict:
    try:
        out = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
        return {"returncode": out.returncode, "stdout": out.stdout.strip(), "stderr": out.stderr.strip()}
    except FileNotFoundError:
        return {"returncode": None, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "stdout": "", "stderr": "timeout"}


# --- manifest (treated as data, never imported) ---------------------------
def extract_split_qids(manifest: Mapping) -> dict[str, list[str]]:
    """Map a manifest's split records to ``{train, dev, validation}`` QID lists.

    Accepts the frozen pilot manifest's ``splits`` records (``pilot_300``,
    ``train_dev_200``, ``validation_246``) and generic ``train`` / ``dev`` /
    ``validation`` names, plus a bare top-level ``qids`` mapping. Raises rather than
    silently accepting a manifest whose splits cannot be identified.
    """
    source = manifest.get("splits")
    if not isinstance(source, Mapping):
        source = manifest.get("qids")
    if not isinstance(source, Mapping):
        raise ValueError("manifest carries neither a 'splits' nor a 'qids' mapping")
    out: dict[str, list[str]] = {}
    for role, aliases in SPLIT_ALIASES.items():
        record = next((source[name] for name in aliases if name in source), None)
        if record is None:
            raise ValueError(f"manifest has no {role} split (looked for {aliases})")
        qids = record.get("qids") if isinstance(record, Mapping) else record
        if not isinstance(qids, (list, tuple)) or not all(isinstance(q, str) for q in qids):
            raise ValueError(f"{role} split must carry a list of string QIDs")
        out[role] = list(qids)
    return out


def verify_splits(qids_by_role: Mapping[str, Sequence[str]], *, train_ids: set[str],
                  validation_ids: set[str], expected_sizes: Mapping[str, int] = SPLIT_SIZES) -> dict:
    """Independently check the manifest QIDs against the pinned local split IDs."""
    checks = {}
    ok = True
    for role, qids in qids_by_role.items():
        universe = validation_ids if role == "validation" else train_ids
        missing = sorted(set(qids) - universe)
        duplicates = len(qids) - len(set(qids))
        size_ok = len(qids) == expected_sizes[role]
        role_ok = not missing and duplicates == 0 and size_ok
        checks[role] = {"n": len(qids), "expected_n": expected_sizes[role], "size_ok": size_ok,
                        "duplicates": duplicates, "missing_from_split": len(missing),
                        "missing_examples": missing[:5], "ok": role_ok}
        ok = ok and role_ok
    overlaps = {
        "train_dev": len(set(qids_by_role["train"]) & set(qids_by_role["dev"])),
        "train_validation": len(set(qids_by_role["train"]) & set(qids_by_role["validation"])),
        "dev_validation": len(set(qids_by_role["dev"]) & set(qids_by_role["validation"])),
    }
    disjoint = not any(overlaps.values())
    return {"splits": checks, "overlaps": overlaps, "all_disjoint": disjoint,
            "official_test_loaded": False, "ok": ok and disjoint}


# --- pinned dataset -------------------------------------------------------
def load_snapshot(snapshot_dir: Path) -> tuple[dict, dict, list, list]:
    """Read only the three pinned non-test tables; report counts, hashes and test presence."""
    paths = {"corpus": snapshot_dir / CORPUS_FILE, "train": snapshot_dir / TRAIN_FILE,
             "validation": snapshot_dir / VALIDATION_FILE}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pinned VixSTORY snapshot is incomplete: missing {missing} in {snapshot_dir}")
    tables = {name: pq.read_table(path).to_pylist() for name, path in paths.items()}
    counts = {name: len(tables[name]) for name in tables}
    digests = {name: sha256_file(path) for name, path in paths.items()}  # hashed once per file
    report = {
        "snapshot": str(snapshot_dir), "repo_id": DATASET_REPO, "revision": REVISION,
        "counts": counts, "expected_counts": dict(EXPECTED_ROWS),
        "counts_ok": {name: counts[name] == EXPECTED_ROWS[name] for name in counts},
        "sha256": digests,
        "sha256_matches_pinned": {name: digests[name] == EXPECTED_SHA256[name] for name in paths},
        "test_split_present_locally": (snapshot_dir / FORBIDDEN_FILE).exists(),
        "test_loaded": False,
    }
    docs = {row["context_id"]: row for row in tables["corpus"]}
    return report, docs, tables["train"], tables["validation"]


# --- machine resources (measured now) -------------------------------------
def gpu_resources(*, nvidia_smi: str | None = None) -> dict:
    smi = nvidia_smi if nvidia_smi is not None else shutil.which("nvidia-smi")
    gpus: list[dict] = []
    if smi:
        query = _run([smi, "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                      "--format=csv,noheader,nounits"])
        for line in query["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({"index": int(parts[0]), "name": parts[1], "memory_total_mib": int(parts[2]),
                             "memory_free_mib": int(parts[3]), "driver": parts[4], "compute_cap": parts[5]})
    torch_info: dict = {"available": None}
    try:  # torch import is CPU-safe and never launches work; guarded so a missing torch is reported
        import torch
        torch_info = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
    except Exception as exc:  # pragma: no cover - depends on the environment
        torch_info = {"available": False, "error": type(exc).__name__}
    return {"nvidia_smi": smi, "gpus": gpus, "present": bool(gpus),
            "vram_total_mib": max((g["memory_total_mib"] for g in gpus), default=0),
            "vram_free_mib": max((g["memory_free_mib"] for g in gpus), default=0), "torch": torch_info}


def system_resources(*, meminfo_path: str = "/proc/meminfo") -> dict:
    """Measure host RAM now, including the reclaimable page cache.

    ``MemAvailable`` already discounts reclaimable cache, so a low ``MemAvailable`` next to a
    large ``mem_reclaimable_mib`` is transient cache pressure, not a hard deficit.
    """
    wanted = ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SReclaimable")
    mem: dict[str, int] = {}
    try:
        for line in Path(meminfo_path).read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            if key in wanted:
                mem[key] = int(rest.strip().split()[0]) // 1024  # MiB
    except OSError:
        pass
    reclaimable = mem.get("Buffers", 0) + mem.get("Cached", 0) + mem.get("SReclaimable", 0)
    return {"hostname": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "mem_total_mib": mem.get("MemTotal", 0), "mem_available_mib": mem.get("MemAvailable", 0),
            "mem_free_mib": mem.get("MemFree", 0), "mem_reclaimable_mib": reclaimable}


def disk_resources(*, paths: Sequence[Path] = (Path.home(),)) -> dict:
    per_path = {}
    for path in paths:
        try:
            usage = shutil.disk_usage(path)
        except OSError:
            continue
        per_path[str(path)] = {"total_gb": round(usage.total / 2**30, 1),
                               "free_gb": round(usage.free / 2**30, 1)}
    return {"paths": per_path, "free_gb": max((v["free_gb"] for v in per_path.values()), default=0.0)}


# --- model snapshots ------------------------------------------------------
def _weight_paths(snap: Path, pattern: str) -> list[Path]:
    """Match a weight glob in a snapshot; a literal name also matches a broken symlink."""
    if any(ch in pattern for ch in "*?["):
        return sorted(snap.glob(pattern))
    candidate = snap / pattern
    return [candidate] if (candidate.is_symlink() or candidate.exists()) else []


def model_snapshot(cache_dir: Path, repo_id: str, weight_globs: Sequence[str]) -> dict:
    """Resolve a cached model's weight files; a broken symlink is reported, not ignored."""
    root = Path(cache_dir) / ("models--" + repo_id.replace("/", "--"))
    snaps = sorted(p for p in (root / "snapshots").glob("*") if p.is_dir()) if (root / "snapshots").is_dir() else []
    resolved: list[dict] = []
    unresolved: list[str] = []
    revision = None
    entries: list[str] = []
    for snap in snaps:
        found: list[dict] = []
        broken: list[str] = []
        for pattern in weight_globs:
            for path in _weight_paths(snap, pattern):
                if path.is_file():  # follows the symlink; broken link -> False
                    found.append({"file": str(path.relative_to(snap)), "bytes": path.stat().st_size})
                elif path.is_symlink() or path.exists():
                    broken.append(str(path.relative_to(snap)))
        if found or broken:
            revision, resolved, unresolved = snap.name, found, broken
            entries = sorted(p.name for p in snap.iterdir())
        if found:  # a snapshot with resolved weights is the one to use
            break
    return {"repo_id": repo_id, "cached": bool(snaps), "snapshot_revision": revision,
            "snapshot_entries": entries, "resolved_weights": resolved, "unresolved_weights": unresolved,
            "total_weight_bytes": sum(w["bytes"] for w in resolved),
            "weights_ok": bool(resolved) and not unresolved}


# --- dependencies (availability probed, never imported) -------------------
def check_dependencies(*, find_spec: Callable[[str], object] = importlib.util.find_spec) -> dict:
    def present(name: str) -> bool:
        try:
            return find_spec(name) is not None
        except (ImportError, ValueError):
            return False

    core = {name: present(name) for name in CORE_DEPS}
    efficient = {name: present(name) for name in PARAM_EFFICIENT_DEPS}
    return {"core": core, "core_ok": all(core.values()),
            "param_efficient": efficient, "param_efficient_ok": all(efficient.values())}


# --- estimates (never measurements) ---------------------------------------
def state_tree_size(corpus_ids: Sequence[str], *,
                    bootstrap_tool: str = EPISODE["bootstrap_tool"],
                    bootstrap_passages: int = EPISODE["bootstrap_passages"],
                    passage_per_search_action: int = EPISODE["passage_per_search_action"],
                    max_optional_search: int = EPISODE["max_optional_search"],
                    tools: Sequence[str] = EPISODE["tools"]) -> int:
    """Worst-case reachable-state count for one episode, via ``reachable_states``.

    The real BM25 and E5 rankings do not exist yet (they are built at P2). To bound the
    tree above, both tools are given maximally divergent synthetic rankings over the real
    corpus IDs, so distinct tool orders never collapse into one state. The real tree can
    only be equal or smaller.
    """
    ids = list(corpus_ids)
    if len(ids) < 8:
        raise ValueError("need at least 8 corpus passages to size the retrieval tree")
    rankings = {tool: (ids if index == 0 else list(reversed(ids))) for index, tool in enumerate(tools)}
    states = reachable_states(rankings, bootstrap_tool=bootstrap_tool, bootstrap_passages=bootstrap_passages,
                              passage_per_search_action=passage_per_search_action,
                              max_optional_search=max_optional_search, tools=list(tools))
    return len(states)


def estimate_states(corpus_ids: Sequence[str], split_sizes: Mapping[str, int] = SPLIT_SIZES) -> dict:
    per_question = state_tree_size(corpus_ids)
    per_split = {role: {"questions": n, "states_upper_bound": per_question * n}
                 for role, n in split_sizes.items()}
    total_questions = sum(split_sizes.values())
    return {"states_per_question_upper_bound": per_question, "per_split": per_split,
            "total_questions": total_questions, "total_states_upper_bound": per_question * total_questions,
            "basis": "reachable_states over divergent worst-case BM25/dense rankings; real P2 rankings "
                     "may merge states, so this is an upper bound, not a measurement"}


def text_sample(rows: Sequence[Mapping], docs: Mapping[str, Mapping], *, sample_n: int, seed: int,
                chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> dict:
    """Deterministic char sample of split text and the largest visible-state prompt."""
    if not rows:
        raise ValueError("no rows to sample")
    sample = sorted(rows, key=lambda r: (stable_seed(seed, r["id"]), r["id"]))[:min(sample_n, len(rows))]
    max_visible = EPISODE["bootstrap_passages"] + EPISODE["passage_per_search_action"] * \
        EPISODE["max_optional_search"]
    question_chars, passage_chars, state_chars = [], [], []
    for row in sample:
        texts = [docs[doc_id]["text"] for doc_id in row["context_ids"]]
        if not texts:
            raise ValueError(f"{row['id']}: no gold passage in the corpus")
        sizes = [len(text) for text in texts]
        question_chars.append(len(row["question"]))
        passage_chars.extend(sizes)
        state_chars.append(len(row["question"]) + max_visible * (sum(sizes) / len(sizes)))

    def stats(values: Sequence[float]) -> dict:
        return {"mean": round(sum(values) / len(values), 1), "max": max(values)}

    state_stats = stats(state_chars)

    def as_tokens(chars: float) -> float:
        return round(chars / chars_per_token, 1)

    return {"sample_n": len(sample), "seed": seed, "visible_passages_max": max_visible,
            "question_chars": stats(question_chars), "gold_passage_chars": stats(passage_chars),
            "max_state_text_chars": state_stats,
            "chars_per_token": chars_per_token,
            "max_state_text_tokens_estimate": {"mean": as_tokens(state_stats["mean"]),
                                               "max": as_tokens(state_stats["max"])}}


def estimate_text(rows_by_role: Mapping[str, Sequence[Mapping]], docs: Mapping[str, Mapping],
                  split_sizes: Mapping[str, int] = SPLIT_SIZES, *, sample_n: int = 40, seed: int = 42,
                  chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> dict:
    per_split = {}
    for role, rows in rows_by_role.items():
        stat = text_sample(rows, docs, sample_n=sample_n, seed=seed, chars_per_token=chars_per_token)
        stat["questions"] = split_sizes[role]
        stat["split_max_state_text_chars_estimate"] = round(stat["max_state_text_chars"]["mean"]
                                                            * split_sizes[role], 0)
        per_split[role] = stat
    total_chars = sum(per_split[role]["split_max_state_text_chars_estimate"] for role in per_split)
    return {"per_split": per_split, "chars_per_token": chars_per_token,
            "total_max_state_text_chars_estimate": round(total_chars, 0),
            "total_max_state_text_tokens_estimate": round(total_chars / chars_per_token, 0),
            "basis": "deterministic sample of split rows x largest reachable state; token count uses a "
                     "char/token heuristic, no tokenizer is run in the CPU preflight"}


def estimate_runtime(split_sizes: Mapping[str, int] = SPLIT_SIZES,
                     measured: Mapping[str, Mapping] = MEASURED_RUNS) -> dict:
    """Report the pilot runtime as UNKNOWN, with the ViQuAD scaling kept as a labelled estimate.

    The reference runs are ViQuAD medium runs (5,000 questions). VixSTORY pairs carry different,
    longer sequence lengths, so scaling them by question count does not transfer across datasets.
    The scaled numbers are therefore kept only as an explicitly-unsound projection and are never a
    completion or timeout promise.
    """
    questions = split_sizes["train"]

    def scale(run: str) -> float:
        ref = measured[run]
        return round(ref["seconds"] * questions / ref["questions"], 1)

    stages = {
        "laya_ce": {"questions": questions, "seconds_estimate": scale("laya_ce")},
        "laya_rlcd": {"questions": questions, "seconds_estimate": scale("laya_rlcd")},
        "reinforce_e06": {"questions": questions, "seconds_estimate": scale("reinforce")},
        "reinforce_e07": {"questions": questions, "seconds_estimate": scale("reinforce")},
    }
    total = sum(stage["seconds_estimate"] for stage in stages.values())
    return {
        "status": "UNKNOWN",
        "unknown_reason": "no VixSTORY stage has a measured wall-clock; the ViQuAD 5,000-question "
                          "runtimes scale only the example count and not the (different, longer) "
                          "VixSTORY sequence lengths",
        "measured_reader_generation": dict(MEASURED_READER_GENERATION),
        "measured_reference": {k: dict(v) for k, v in measured.items()},
        "projection": {"stages": stages, "questions": questions,
                       "total_seconds_estimate": round(total, 1),
                       "total_gpu_hours_estimate": round(total / 3600, 2)},
        "timeout_promise": False,
        "basis": "unsound cross-dataset estimate: previously measured ViQuAD 5,000-question medium "
                 "runs (reports/COMPUTE_AND_ENV.md) scaled linearly by pilot question count; sequence "
                 "lengths differ between datasets, so no completion time is promised",
    }


# --- go / no-go -----------------------------------------------------------
def go_no_go(*, resources: Mapping, models: Mapping[str, Mapping], dependencies: Mapping,
             verification: Mapping, dataset: Mapping, verification_available: bool,
             runtime: Mapping) -> dict:
    """Decision for arm A (cloud reader + Laya + two REINFORCE checkpoints). Arm A separates
    *proven blockers* (NO-GO) from checks that can only be settled by a measurement
    (NEEDS_MEASUREMENT)."""
    gpu = resources["gpu"]
    system = resources["system"]

    a_blockers: list[str] = []
    a_measurements: list[str] = []
    if not gpu["present"]:
        a_blockers.append("no_gpu_visible")
    elif gpu["vram_free_mib"] < ARM_A_VRAM_MIN_MIB:
        a_blockers.append(f"vram_free_below_{ARM_A_VRAM_MIN_MIB}_mib")
    if resources["disk"]["free_gb"] < ARM_A_DISK_MIN_GB:
        a_blockers.append(f"disk_free_below_{ARM_A_DISK_MIN_GB}_gb")
    for repo in MODEL_REPOS:
        model = models[repo]
        if not model["weights_ok"]:
            a_blockers.append(f"weights_unresolved:{repo}")
        elif model["snapshot_revision"] != MODEL_REVISIONS[repo]:
            a_blockers.append(f"weights_revision_mismatch:{repo}")
    for name, present in dependencies["core"].items():
        if not present:
            a_blockers.append(f"dependency_missing:{name}")
    if not all(dataset["counts_ok"].values()):
        a_blockers.append("dataset_counts_mismatch")
    if not all(dataset["sha256_matches_pinned"].values()):
        a_blockers.append("dataset_sha256_mismatch")
    if not verification_available:
        a_blockers.append("manifest_unavailable")
    elif not verification["ok"]:
        a_blockers.append("manifest_verification_failed")

    # Host RAM is not a proven blocker. The prior Laya medium runs succeeded on this same 15.9 GiB
    # host with no per-stage RSS recorded, so available RAM below the target can be reclaimable page
    # cache rather than a deficit: it asks for a measurement, and only *total* RAM below the target
    # is a hard deficit (the machine can never meet the target).
    mem_total, mem_available = system["mem_total_mib"], system["mem_available_mib"]
    ram = {"mem_total_mib": mem_total, "mem_available_mib": mem_available,
           "reclaimable_cache_mib": system.get("mem_reclaimable_mib", 0),
           "headroom_target_mib": ARM_A_RAM_MIN_MIB}
    if mem_total and mem_total < ARM_A_RAM_MIN_MIB:
        ram["classification"] = "hard_deficit"
        a_blockers.append(f"ram_total_below_{ARM_A_RAM_MIN_MIB}_mib")
    elif mem_available < ARM_A_RAM_MIN_MIB:
        ram["classification"] = "transient_cache_pressure"
        a_measurements.append(f"ram_available_below_{ARM_A_RAM_MIN_MIB}_mib_measure_stage_budget")
    else:
        ram["classification"] = "ok"

    decision = "NO-GO" if a_blockers else ("NEEDS_MEASUREMENT" if a_measurements else "GO")
    arm_a = {"decision": decision, "blockers": a_blockers, "measurements_required": a_measurements,
             "ram": ram,
             "stage_memory_budget": {"measured": False,
                                     "measured_peak_vram_mib_per_trainer": 6139.4,
                                     "measured_peak_host_rss_mib": None,
                                     "note": "no per-stage host-RSS budget was recorded for the prior "
                                             "Laya runs, so host RAM cannot be decided from this "
                                             "reading; measure a stage RSS (or re-run with the page "
                                             "cache dropped) before treating low available RAM as a "
                                             "hard limit"},
             "requires": {"vram_free_mib": ARM_A_VRAM_MIN_MIB, "ram_available_mib": ARM_A_RAM_MIN_MIB,
                          "disk_free_gb": ARM_A_DISK_MIN_GB},
             "note": "cloud reader + Laya CE/RLCD + two REINFORCE checkpoints on a single GPU; "
                     "available RAM below the target is likely reclaimable page cache (the prior Laya "
                     "medium runs ran on this same host) and is reported as a measurement required, "
                     "never as a proven infeasibility"}

    return {"cloud_reader_laya_two_rl_checkpoints": arm_a, "runtime_estimate": runtime}


LIMITATIONS = (
    "Runtime, VRAM and memory figures under 'estimated' are analytic projections, not observations "
    "of this pilot; only the resource, snapshot, dependency and dataset fields are measured here.",
    "The pilot runtime is UNKNOWN: the reference medium runs are ViQuAD (5,000 questions) and "
    "VixSTORY sequence lengths differ, so the cross-dataset scaling is an unsound estimate and no "
    "completion timeout is promised. The only VixSTORY runtime evidence is the frozen reader's "
    "generation count, and no wall-clock was recorded for it.",
    "Host RAM below the headroom target is reported as a measurement required, not a proven "
    "infeasibility: no per-stage host-RSS budget was recorded for the prior Laya runs.",
    "The reachable-state count is a worst-case upper bound from divergent synthetic rankings; the real "
    "BM25/E5 rankings are built at P2 and may merge states.",
    "Token counts are a char/token heuristic (no tokenizer is run); they are prompt-cost estimates only.",
    "The official test split is never read; only its local presence is observed.",
    "The manifest is read as JSON data; this script does not import the manifest module.",
)


def build_report(*, manifest_path: Path, snapshot_dir: Path, cache_dir: Path,
                 resources: Mapping | None = None, dependencies: Mapping | None = None,
                 sample_n: int = 40, seed: int = 42) -> dict:
    """Assemble the preflight report from measured inputs and labelled estimates."""
    dataset, docs, train_rows, val_rows = load_snapshot(snapshot_dir)
    resources = resources or {"gpu": gpu_resources(), "system": system_resources(),
                              "disk": disk_resources()}
    dependencies = dependencies or check_dependencies()
    models = {repo: model_snapshot(cache_dir, repo, WEIGHT_GLOBS[repo]) for repo in MODEL_REPOS}
    for repo, model in models.items():
        model["pinned_revision"] = MODEL_REVISIONS[repo]
        model["revision_matches_pin"] = model["snapshot_revision"] == MODEL_REVISIONS[repo]

    verification: dict = {"official_test_loaded": False, "ok": False, "reason": "manifest_absent"}
    manifest_report = {"path": str(manifest_path), "present": False}
    rows_by_role: dict[str, list] = {"train": train_rows, "dev": train_rows, "validation": val_rows}
    if manifest_path.exists():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        qids = extract_split_qids(manifest)
        verification = verify_splits(qids, train_ids={r["id"] for r in train_rows},
                                     validation_ids={r["id"] for r in val_rows})
        wanted = {role: set(role_qids) for role, role_qids in qids.items()}
        rows_by_role = {"train": [r for r in train_rows if r["id"] in wanted["train"]],
                        "dev": [r for r in train_rows if r["id"] in wanted["dev"]],
                        "validation": [r for r in val_rows if r["id"] in wanted["validation"]]}
        manifest_report = {"path": str(manifest_path), "present": True, "run_id": manifest.get("run_id"),
                           "revision": (manifest.get("source") or {}).get("resolved_revision"),
                           "splits": {role: len(role_qids) for role, role_qids in qids.items()},
                           "verification": verification}

    corpus_ids = list(docs)
    runtime = estimate_runtime()
    report = {
        "artifact": "vixstory_pilot_preflight", "schema_version": 1, "run_id": RUN_ID,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "CPU_ONLY", "dataset_loaded": True, "official_test_loaded": False,
        "measured": {"gpu": resources["gpu"], "system": resources["system"], "disk": resources["disk"],
                     "models": models, "dependencies": dependencies, "dataset": dataset},
        "estimated": {"states": estimate_states(corpus_ids),
                      "text": estimate_text(rows_by_role, docs, SPLIT_SIZES, sample_n=sample_n, seed=seed),
                      "runtime": runtime},
        "manifest": manifest_report,
        "go_no_go": go_no_go(resources=resources, models=models, dependencies=dependencies,
                             verification=verification, dataset=dataset,
                             verification_available=manifest_report["present"], runtime=runtime),
        "limitations": list(LIMITATIONS),
    }
    return report


def next_free_path(path: Path) -> Path:
    """Return ``path``, or the first ``<stem>.v<N><suffix>`` that does not exist. Never overwrites."""
    if not path.exists():
        return path
    for version in range(2, 1000):
        candidate = path.with_name(f"{path.stem}.v{version}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"no free version of {path}")


def run(*, out: Path = OUT_PATH, manifest_path: Path = MANIFEST_PATH, snapshot_dir: Path = SNAPSHOT,
        cache_dir: Path = HF_CACHE, resources: Mapping | None = None, dependencies: Mapping | None = None,
        write: bool = True, sample_n: int = 40, seed: int = 42) -> dict:
    """Run the preflight. The report is written only once the manifest exists, and an existing
    preflight is versioned rather than overwritten."""
    report = build_report(manifest_path=manifest_path, snapshot_dir=snapshot_dir, cache_dir=cache_dir,
                          resources=resources, dependencies=dependencies, sample_n=sample_n, seed=seed)
    if write and report["manifest"]["present"]:
        target = next_free_path(out)
        atomic_write_json(target, report)
        report["written"] = str(target)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--cache", type=Path, default=HF_CACHE)
    parser.add_argument("--sample-n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-write", action="store_true", help="run checks without writing preflight.json")
    args = parser.parse_args()
    report = run(out=args.out, manifest_path=args.manifest, snapshot_dir=args.snapshot,
                 cache_dir=args.cache, write=not args.no_write, sample_n=args.sample_n, seed=args.seed)
    decisions = {arm: value["decision"] for arm, value in report["go_no_go"].items()
                 if arm != "runtime_estimate"}
    print(json.dumps({"run_id": report["run_id"], "manifest_present": report["manifest"]["present"],
                      "written": report.get("written"), "decisions": decisions}, ensure_ascii=False, indent=2))
    return 0 if report["go_no_go"]["cloud_reader_laya_two_rl_checkpoints"]["decision"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())

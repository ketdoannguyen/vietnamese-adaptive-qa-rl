# Shared timing provenance for VixSTORY preflight

`scripts/vixstory_pilot_preflight.py` reuses these historical measurements when estimating training cost. They were measured on earlier **5,000-question UIT-ViQuAD2.0 runs**, not on VixSTORY. Cross-dataset projections remain estimates, not measured VixSTORY runtimes.

| Stage | Wall time (s) | Peak allocated VRAM (MiB) | Historical source |
|---|---:|---:|---|
| Laya soft CE | 15890.7 | 6139.4 | `p5-laya-ce-medium-2026-10-09T031230+0000` trainer manifest |
| Laya RLCD | 16078.6 | 6139.4 | `p5-laya-rlcd-medium-2026-10-09T033300+0000` trainer manifest |
| REINFORCE | 1525.0 | Not recorded here | P6 stage duration in the historical run registry |

Measured hardware: one NVIDIA L4. Laya used two epochs and seed 42 in those historical CE/RLCD runs. The VixSTORY preflight independently checks its current hardware and cached model identity.

The retained Laya snapshot pin is `convaiinnovations/laya-multilingual` at `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`, as recorded in the VixSTORY preflight. It is not a ViQuAD-trained checkpoint from `outputs/models/`.

The old ViQuAD checkpoints, logs, manifests and datasets were retired by the approved cleanup. This small shared reference preserves the values already used by VixSTORY; the removed source artifacts are no longer locally available for re-verification. Do not report these figures as a new benchmark or as VixSTORY measurements.

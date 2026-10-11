# COMPUTE_AND_ENV — tài nguyên đo được

Nguồn: outputs/diagnostics/gpu_info.json, system_info.json, outputs/manifests/P5_train_laya_*.json, runs/index.jsonl — không có số nào nhập tay; thiếu artifact thì ghi N/A kèm đường dẫn đã tra.

**Bảng C1 — Phần cứng và runtime (đo tại P0/P5):**

| Mục | Giá trị đo được | Nguồn |
|---|---|---|
| GPU | NVIDIA L4 ×1 (compute 8.9, driver 580.178.04) | outputs/diagnostics/gpu_info.json |
| VRAM tổng / trống lúc đo (MiB) | 23034 / 22564 | outputs/diagnostics/gpu_info.json |
| CUDA / PyTorch | CUDA 13.0 / torch 2.14.1+cu130; bf16 khả dụng: True | outputs/diagnostics/gpu_info.json |
| CPU | Intel(R) Xeon(R) CPU @ 2.20GHz ×4 lõi | outputs/diagnostics/system_info.json |
| RAM tổng / khả dụng (MiB) | 15990 / 12582 | outputs/diagnostics/system_info.json |
| SSD (root) | 77.3 GB tổng / 50.3 GB trống (35.0% đã dùng) | outputs/diagnostics/system_info.json |
| Host / kernel / python | gpu-1-day / 6.8.0-1069-gcp / 3.10.12 | outputs/diagnostics/system_info.json |
| Thư viện huấn luyện (manifest mới nhất: p5-laya-rlcd-medium-2026-10-09T033300+0000) | cuda_runtime=13.0, cudnn=92400, datasets=5.1.0, gymnasium=1.4.0, numpy=2.2.6, pandas=2.3.3, pyarrow=25.0.1, python=3.10.12, rank_bm25=unknown, sentence_transformers=6.1.0, sklearn=1.7.2, torch=2.14.1+cu130, transformers=5.19.0 | outputs/manifests/P5_train_laya_*.json |

**Bảng C2 — VRAM đỉnh và thời lượng do trainer tự đo (từng lần chạy Laya):**

| Lần chạy (run_id) | Regime | Loss | Epochs | Seed | Peak VRAM (MiB) | Walltime (s) | Loss epoch cuối | Manifest |
|---|---|---|---|---|---|---|---|---|
| p5-laya-ce-medium-2026-10-09T031230+0000 | ce | soft-ce | 2 | 42 | 6139.400000 | 15890.700000 | 0.840400 | outputs/manifests/P5_train_laya_ce_medium.json |
| p5-laya-ce-smoke-2026-10-09T005229+0000 | ce | soft-ce | 1 | 42 | 6179.100000 | 27.000000 | 1.420682 | outputs/manifests/P5_train_laya_ce_smoke.json |
| p5-laya-rlcd-medium-2026-10-09T033300+0000 | rlcd | rlcd | 2 | 42 | 6139.400000 | 16078.600000 | 0.905828 | outputs/manifests/P5_train_laya_rlcd_medium.json |

**Bảng C3 — Thời lượng từng stage (đo được: bản ghi RUNNING đầu → dòng kết thúc cuối theo run_id trong runs/index.jsonl):**

| Giai đoạn | Stage | Trạng thái | Giây (đo được) |
|---|---|---|---|
| P1 | download | NOT_RUN | N/A |
| P1 | audit_split | SUCCEEDED | 240.000000 |
| P2 | build_retrieval | SUCCEEDED | 27148.000000 |
| P3 | train_reader | SUCCEEDED | 646.000000 |
| P3 | cache_reader | SUCCEEDED | 1244.000000 |
| P5 | make_teacher | SUCCEEDED | 172.000000 |
| P5 | train_laya_ce | SUCCEEDED | 16002.000000 |
| P5 | train_laya_rlcd | SUCCEEDED | 16084.000000 |
| P5 | cache_laya | SUCCEEDED | 3557.000000 |
| P6 | train_rl | SUCCEEDED | 1525.000000 |
| P6 | select_rl | SUCCEEDED | 0.000000 |
| P8 | train_correctness | SUCCEEDED | 1740.000000 |
| P7 | eval_baselines | SUCCEEDED | 1143.000000 |
| P9 | report | SUCCEEDED | N/A |

- Peak VRAM ở bảng C2 là số trainer tự đo trong tiến trình (allocated), thấp hơn mức thiết bị giữ khi CE ∥ RLCD cùng resident — mức giữ thực tế của cặp đôi đã đo và ghi trong docs/DECISIONS.md (D018/D020) cùng hạn mức 24 GiB.
- Thời gian theo từng lần chạy (kể cả smoke/bị ngắt) và hình loss: REPORT.md Bảng 1b/1c; số phút train là walltime đo từ registry, không tính lại từ log.

# FINAL HANDOFF

## Executive status

- Trạng thái: **PARTIAL** (13/15 stage SUCCEEDED)
- Thời điểm sinh: 2026-10-09T09:18:21+00:00; profile: medium
- Git SHA: b84044cfa4cdaf1af5fa9a1d914a4187d3555629; registry sha256: 7bd4987cdeac86e346e6616772181ad8da111f541bef3a37a81ead270c1e1679 (không gồm dòng P9_report của chính báo cáo)
- Tiêu chí chọn checkpoint: mean episodic reward trên val (freeze: có)

## What ran

| Phía | Stage | Trạng thái | Lệnh | Bằng chứng | Artifact |
|---|---|---|---|---|---|
| P0 | preflight | NOT_RUN | bash scripts/preflight.sh | N/A | outputs/diagnostics/preflight.md, outputs/diagnostics/preflight_verdict.json |
| P1 | download | NOT_RUN | python -m src.pipeline.download --config configs/medium.yaml | N/A | data/raw/source_manifest.json |
| P1 | audit_split | SUCCEEDED | python -m src.pipeline.audit_split --config configs/medium.yaml | p1-audit-split-medium-2026-10-08T165734+0000 | data/processed/split_manifest.json |
| P2 | build_retrieval | SUCCEEDED | python -m src.pipeline.build_retrieval --config configs/medium.yaml | p2-retrieval-medium-2026-10-09T003754+0000 | reports/retrieval_report.json, data/cache/retrieval/ |
| P3 | train_reader | SUCCEEDED | python -m src.pipeline.train_reader --config configs/medium.yaml | p3-reader-medium-2026-10-08T144826+0000 | outputs/models/qa_reader/reader_inference.json, outputs/models/qa_reader/reader_eval.json |
| P3 | cache_reader | SUCCEEDED | python -m src.pipeline.cache_reader --config configs/medium.yaml | p3-reader-cache-medium-2026-10-09T085245+0000 | outputs/models/qa_reader/reader_qa_state_cache_train_strict.parquet, outputs/models/qa_reader/reader_qa_state_cache_val_strict.parquet |
| P5 | make_teacher | SUCCEEDED | python -m src.pipeline.make_teacher --config configs/medium.yaml | p5-teacher-train-medium-2026-10-09T004558+0000 | data/processed/laya_train.jsonl |
| P5 | train_laya_ce | SUCCEEDED | python -m src.pipeline.train_laya --config configs/medium.yaml --loss soft-ce | p5-laya-ce-medium-2026-10-09T031230+0000 | outputs/models/laya_ce/ |
| P5 | train_laya_rlcd | SUCCEEDED | python -m src.pipeline.train_laya --config configs/medium.yaml --loss rlcd | p5-laya-rlcd-medium-2026-10-09T033300+0000 | outputs/models/laya_rlcd/ |
| P5 | cache_laya | SUCCEEDED | python -m src.pipeline.cache_laya --config configs/medium.yaml --regime rlcd | p5-cache-laya-rlcd-medium-2026-10-09T080231+0000 | outputs/models/laya_probs_cache/rlcd/ |
| P6 | train_rl | SUCCEEDED | python -m src.pipeline.train_rl --config configs/medium.yaml | p6-rl-medium-44-2026-10-09T084444+0000 | outputs/models/rl_policy/, reports/rl_training_report.json |
| P6 | select_rl | SUCCEEDED | python -m src.pipeline.select_rl --config configs/medium.yaml | p6-select-medium-2026-10-09T084821+0000 | outputs/manifests/best_selection.json |
| P8 | train_correctness | SUCCEEDED | python -m src.pipeline.train_correctness --config configs/medium.yaml | b5-correctness-medium-42-2026-10-09T080314+0000 | outputs/models/correctness_estimator/, reports/correctness_calibration.md |
| P7 | eval_baselines | SUCCEEDED | python -m src.pipeline.eval_baselines --config configs/medium.yaml --split val_strict | p7-baselines-medium-test-42-2026-10-09T085712+0000 | reports/tables/baseline_results_val.csv, reports/tables/baseline_results_test.csv |
| P9 | report | SUCCEEDED | python -m src.pipeline.report --config configs/medium.yaml | N/A | reports/REPORT.md, reports/results.csv |

## Model artifacts

| Artifact | Đường dẫn | Bytes | SHA256 |
|---|---|---|---|
| reader | /home/nkd/training-code/vietnamese-adaptive-qa-rl/outputs/models/qa_reader/model.safetensors | 1109842416 | 20dee7fafce6ab6728745850ab4ee8bbe880d8505a931c9c6d0823c909b2ac90 |
| laya_ce | /home/nkd/training-code/vietnamese-adaptive-qa-rl/outputs/models/laya_ce/model.safetensors | 643835524 | bcd69a5dddfe4cb5197e7213a314bf88f1f3967c330b50b871c5f820b7f3ecd2 |
| laya_rlcd | /home/nkd/training-code/vietnamese-adaptive-qa-rl/outputs/models/laya_rlcd/model.safetensors | 643835524 | c3b34475256aaee5def0131d1f6fb072e622ed4da7dcbaf415d3dbf9c4ba1895 |
| policy | /home/nkd/training-code/vietnamese-adaptive-qa-rl/outputs/models/rl_policy/policy.safetensors | 86652 | ec32df9e352fb7c699f1485570000c1e5f73f11dea2f27f31d1bdd1f83dd970f |
| policy_scaler | /home/nkd/training-code/vietnamese-adaptive-qa-rl/outputs/models/rl_policy/scaler.json | 1458 | b444ed3cd62f7977d752cb73d2e46a944f615864b56573d5a99e4ff624070c1e |

- Load-back: reader được load lại trong P3 (`reader_inference.json` khớp hash); policy được load lại trong P7 (`src/eval/policies.load_rl_policy`). Stage báo cáo này **không** load weight.

## Results

- RL (E06) vs baseline phi-RL tốt nhất theo F1: E06 F1=0.159219 so với E02 F1=0.330448 trên val_strict (reward -0.489088 vs -1.381596)
- Bảng đầy đủ + nguồn: reports/results.csv, reports/results.json, REPORT.md mục 7–8.
- Thất bại/điểm yếu: xem REPORT.md mục 9 và danh sách stage FAILED/INTERRUPTED dưới đây.

## Run now

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m src.pipeline.report --config configs/medium.yaml
.venv/bin/python -m src.inference.cli --question '<câu hỏi>' --config configs/medium.yaml --show-trace
bash scripts/reproduce.sh --run-id <run_id> --mode eval
```

## Limitations / blockers

- P0/preflight: NOT_RUN — stage không ghi vào runs/index.jsonl (xem outputs/diagnostics)
- P1/download: NOT_RUN — không có bản ghi stage='P1_download' profile='medium' trong runs/index.jsonl
- Lệnh phục hồi: chạy lại đúng stage với `--resume` (xem REPRODUCE.md); run lỗi được giữ nguyên trong registry.

## Safety check

- License dữ liệu: UNVERIFIED (card_license=N/A); không xuất bản lại dataset/weight khi chưa rà soát.
- Report và manifest không chứa credential (sinh từ artifact, không nhúng token); quyền file theo umask mặc định của repo.
- Test access log: docs/TEST_ACCESS_LOG.md; mọi số trong REPORT.md truy về file nguồn qua cột 'source' trong results.csv và mục Phụ lục.

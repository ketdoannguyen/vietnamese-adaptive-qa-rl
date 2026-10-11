# REPRODUCE — tái lập đánh giá và báo cáo

Profile: `medium`. Không train lại chỉ để kiểm chứng metric: dùng `bash scripts/reproduce.sh --run-id <id> --mode eval`.

```bash
# 0. Môi trường
python3 -m venv .venv && .venv/bin/pip install -e .
bash scripts/preflight.sh

# 1-3. Dữ liệu và retrieval
.venv/bin/python -m src.pipeline.download --config configs/medium.yaml
.venv/bin/python -m src.pipeline.audit_split --config configs/medium.yaml
.venv/bin/python -m src.pipeline.build_retrieval --config configs/medium.yaml

# 4-6. Reader + cache trạng thái + teacher
.venv/bin/python -m src.pipeline.train_reader --config configs/medium.yaml
.venv/bin/python -m src.pipeline.cache_reader --config configs/medium.yaml
.venv/bin/python -m src.pipeline.make_teacher --config configs/medium.yaml

# 7-9. Laya (CE, RLCD — 2 epoch theo ngân sách D015) + cache prior theo regime (D019)
.venv/bin/python -m src.pipeline.train_laya --profile medium --config configs/laya_budget_2ep.yaml --loss soft-ce
.venv/bin/python -m src.pipeline.train_laya --profile medium --config configs/laya_budget_2ep.yaml --loss rlcd
.venv/bin/python -m src.pipeline.cache_laya --config configs/medium.yaml --regime rlcd

# 10. REINFORCE — 3 seed đã đăng ký (42/43/44, D015), rồi khoá lựa chọn trên val
.venv/bin/python -m src.pipeline.train_rl --config configs/medium.yaml --seed 42
.venv/bin/python -m src.pipeline.train_rl --config configs/medium.yaml --seed 43
.venv/bin/python -m src.pipeline.train_rl --config configs/medium.yaml --seed 44
.venv/bin/python -m src.pipeline.select_rl --config configs/medium.yaml

# 11. Baseline B5 (--oof-folds 3, D015); E00-E07 trên val; test mở đúng một lần:
#     cache trạng thái test (chỉ dữ liệu, không nhãn) chỉ SAU best_selection.json
.venv/bin/python -m src.pipeline.train_correctness --config configs/medium.yaml --oof-folds 3
.venv/bin/python -m src.pipeline.eval_baselines --config configs/medium.yaml --split val_strict
.venv/bin/python -m src.pipeline.cache_reader --config configs/medium.yaml --splits test_strict
.venv/bin/python -m src.pipeline.eval_baselines --config configs/medium.yaml --split test_strict

# 12. Báo cáo
.venv/bin/python -m src.pipeline.report --config configs/medium.yaml

# Demo suy luận từ artifact đã lưu
.venv/bin/python -m src.inference.cli --question 'Phạm Văn Đồng bắt đầu giữ chức Thủ tướng CHXHCN Việt Nam từ năm nào?' --config configs/base.yaml --show-trace

# 13. Kiểm chứng một run đã ghi (hash + subset predictions, không train lại)
bash scripts/reproduce.sh --run-id <run_id> --mode eval

# 14. Test
.venv/bin/python -m pytest tests/ -q
```

- Điều kiện phần cứng: 1 GPU ≥ 20GB VRAM (đo được: NVIDIA L4), RAM ≥ 12GB, disk trống ≥ 20GB.
- Hash để đối chiếu: reports/artifact_manifest.sha256; log registry: outputs/runs/index.jsonl.
- Nondeterminism: CUDA kernels + thứ tự giảm; eval greedy/no_grad/seed cố định; so sánh lại bằng tolerance khi verify.

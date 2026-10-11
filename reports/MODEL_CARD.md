# MODEL CARD — Vietnamese Adaptive QA RL

## Model identifier

`Vietnamese Adaptive QA RL` — pipeline tổng hợp (retrieval BM25+E5, reader XLM-R SFT, mô hình quyết định Laya CE/RLCD, policy REINFORCE).

## Intended use

Tác tử hỏi đáp trích xuất tiếng Việt trên kho evidence giới hạn đã index; trả lời kèm trích dẫn hoặc từ chối. Nghiên cứu học thuật/đồ án cuối kỳ — **không** thay thế xác minh thông tin thế giới mở, tư vấn pháp lý/y tế hay ra quyết định quan trọng.

## Architecture and training

- Reader: `outputs/models/qa_reader` (model hash `f05b24ee68f2e6bbd43ea2e2ab1e1ab36be12dfd9e6924079867b8eb212f54e8`), base FacebookAI/xlm-roberta-base, SFT trên train_strict (N=N/A), đánh giá `reader_eval.json` (có).
- Retriever: BM25 (k1=1.500000, b=0.750000) + E5 `intfloat/multilingual-e5-small`; corpus manifest `data/processed/split_manifest.json`.
- Laya: `outputs/models/laya_ce`, `outputs/models/laya_rlcd` (manifest có), regime `frozen`, teacher targets suy từ reward (soft, temperature=0.500000).
- RL policy: `outputs/models/rl_policy` (run_id=`p6-rl-medium-43-2026-10-09T084318+0000`, updates=300, val reward=-0.343750, reward terms trong configs/base.yaml).
- **Không phải TypeSafe Jev weights**; không gọi là fine-tune Jev end-to-end.

## Data and licensing

- HF dataset `taidng/UIT-ViQuAD2.0` revision `406f09a45cc106a8f7b7fd0c25078883fe58cb1f`; strict split theo title-group với số dòng {}.
- Duplicate/leakage + audit âm: `data/processed/split_manifest.json`, `data/processed/ambiguous_negatives.json`.
- `license_status: UNVERIFIED` (card_license=N/A); cấm xuất bản lại raw dataset/checkpoint khi chưa rà soát.

## Evaluation

| Mã | Split | N | Mean reward ↑ | F1 ↑ | Coverage ↑ | Chi phí TB ↓ | Status |
|---|---|---|---|---|---|---|---|
| E00 | val_strict | 4399 | -0.713594 | 0.302496 | 0.732212 | 0.100000 | SUCCEEDED |
| E00 | test_strict | 4260 | -0.750563 | 0.324775 | 0.790376 | 0.100000 | SUCCEEDED |
| E01 | val_strict | 4399 | -1.039532 | 0.306042 | 0.827915 | 0.300000 | SUCCEEDED |
| E01 | test_strict | 4260 | -1.069202 | 0.328277 | 0.862676 | 0.300000 | SUCCEEDED |
| E02 | val_strict | 4399 | -1.330802 | 0.316175 | 0.840191 | 0.600000 | SUCCEEDED |
| E02 | test_strict | 4260 | -1.381596 | 0.330448 | 0.873005 | 0.600000 | SUCCEEDED |
| E03 | val_strict | 4399 | -0.726233 | 0.188775 | 0.388725 | 0.321141 | SUCCEEDED |
| E03 | test_strict | 4260 | -0.715094 | 0.206167 | 0.438498 | 0.303075 | SUCCEEDED |
| E04 | val_strict | 4399 | -0.643737 | 0.093587 | 0.212321 | 0.165765 | SUCCEEDED |
| E04 | test_strict | 4260 | -0.586268 | 0.093838 | 0.203521 | 0.164671 | SUCCEEDED |
| E05 | val_strict | 4399 | -0.571334 | 0.000000 | 0.000000 | 0.100000 | SUCCEEDED |
| E05 | test_strict | 4260 | -0.510563 | 0.000000 | 0.000000 | 0.100000 | SUCCEEDED |
| E06 | val_strict | 4399 | -0.489088 | 0.159219 | 0.314390 | 0.100000 | SUCCEEDED |
| E06 | test_strict | 4260 | -0.502629 | 0.174813 | 0.363850 | 0.100000 | SUCCEEDED |
| E07 | val_strict | N/A | N/A | N/A | N/A | N/A | SKIPPED |
| E07 | test_strict | N/A | N/A | N/A | N/A | N/A | SKIPPED |

- CI theo cụm title-group chỉ hiển thị khi artifact có cột CI tương ứng; nhãn/hạn chế contamination: reader SFT chỉ dùng train_strict; test mở một lần sau freeze (docs/TEST_ACCESS_LOG.md).

## Resource needs and restrictions

- GPU đo được: NVIDIA L4 (23034 MiB); peak VRAM train RL=N/A MiB; RAM=15990 MiB.
- Ngân sách tìm mặc định: bootstrap bm25 + 2 lượt tùy chọn; latency inference đo trong `timings` của demo CLI (src.inference.cli).
- Phụ thuộc code/license: xem pyproject.toml; không kèm credential.

## Security, privacy, societal impacts

- Văn bản tài liệu được coi là đầu vào không tin cậy; không thực thi chỉ dẫn trong văn bản; không lưu bí mật; luôn kèm trích dẫn hoặc từ chối.
- Mô hình có thể trả lời sai một cách tự tin khi retrieval/reader sai; hiệu năng bị giới hạn bởi corpus và thiết kế reward.

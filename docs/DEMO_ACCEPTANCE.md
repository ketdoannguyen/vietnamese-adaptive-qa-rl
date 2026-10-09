# DEMO ACCEPTANCE — P9 (spec 02_IMPLEMENTATION/07 §7)

Biên bản chấp nhận của đường demo end-to-end: **3 ca trả lời đúng kèm trích dẫn + 3 ca
từ chối (âm/ambiguous) + 1 vết thất bại thật**. Đây là lần đầu tiên chuỗi chạy lệnh demo
của spec (07 §1) trên artifact đã đóng băng — trước đó đường này chưa từng chạy hết
(xem D025: bug thư mục checkpoint + config thiếu; đã sửa cùng ngày).

Kết luận: **3/3 đúng + 3/3 từ chối đạt yêu cầu spec; 1 vết thất bại giữ nguyên không che**
(ca #7: reader chọn span sai thời kỳ lịch sử; EM=0). Mọi số liệu dưới đây trích từ file
log trong `logs/` — không gõ tay.

## Bối cảnh và tính trung thực của bộ ca

- 7 câu được **chọn từ `val_strict`** (tập đánh giá, không phải test). `test_strict` bị CLI
  từ chối làm corpus demo theo thiết kế; lần mở test duy nhất ghi ở `docs/TEST_ACCESS_LOG.md`.
- Nhãn (từ P7 val run `p7-baselines-medium-val-42-2026-10-09T084929+0000`) chỉ dùng để
  **phân loại ca mẫu** (ca đúng / ca impossible / ca sai); đầu ra demo là **chạy live** qua
  `src.inference.cli` — không dán lại kết quả eval. Bằng chứng nhất quán: demo tái hiện
  đúng đáp án sai của ca #7 và đúng quyết định sát nút của ca #5 như trong eval.
- Demo không bao giờ thấy nhãn (có test `test_no_gold_or_label_field_reaches_the_result`).

## Lệnh đã chạy (spec 07 §1, với `configs/final_inference.yaml`)

```bash
.venv/bin/python -m src.inference.cli --question-file logs/demo_acceptance_questions.txt \
    --config configs/final_inference.yaml --json  > logs/demo_acceptance.jsonl 2> logs/demo_acceptance.err
.venv/bin/python -m src.inference.cli --question-file logs/demo_acceptance_questions.txt \
    --config configs/final_inference.yaml --show-trace > logs/demo_acceptance_trace.log
```

Exit code cả hai lần chạy: 0 (`logs/demo_acceptance_run.log`). Chuỗi nạp là chuỗi medium
đã đóng băng 2026-10-09: reader P3, Laya rlcd P5, policy REINFORCE P6 (canonical
`outputs/models/rl_policy`, winner seed 43). Mọi câu đều đi qua `decision_rule: policy`;
laya_ms > 0 ở run nguội, 0.0 ở run trace (cache `outputs/models/laya_probs_cache_demo`
tái dùng — memoization hoạt động).

## Kết quả 7 ca

| # | Phân loại | qid | Gold | Demo trả lời | Kết quả |
|---|---|---|---|---|---|
| 1 | đúng | train:0001-0001-0001 | Lâm Bá Kiệt | "Lâm Bá Kiệt" + trích dẫn | đúng |
| 2 | đúng | train:0008-0001-0002 | cơ học lượng tử | "cơ học lượng tử" + trích dẫn | đúng |
| 3 | đúng | train:0016-0001-0005 | có thể chơi nhạc bằng việc nhấn vào bàn phím | đúng span + trích dẫn | đúng |
| 4 | âm (impossible) | train:0001-0001-0005 | (không có) | từ chối | đạt |
| 5 | âm (impossible, ambiguous_negative) | train:0008-0001-0006 | (không có) | từ chối | đạt |
| 6 | âm (impossible) | train:0016-0001-0006 | (không có) | từ chối | đạt |
| 7 | thất bại | train:0001-0001-0002 | Thủ tướng | "Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa" | sai (EM=0) |

Cả 7 ca: `retrieval_steps=1` (chỉ bootstrap bm25; policy chọn ANSWER/ABSTAIN ngay —
`tools_used=["bm25"]`, không gọi công cụ tùy chọn), `total_cost=0.1`.

## Trích đoạn log (nguyên văn từ `logs/demo_acceptance_trace.log`)

**Ca đúng #1** — trả lời trích xuất kèm trích dẫn:

```
Câu hỏi: Tên gọi nào được Phạm Văn Đồng sử dụng khi làm Phó chủ nhiệm cơ quan Biện sự xứ tại Quế Lâm?
Theo tài liệu [f3246604359cb2a6515608bd6cdefe8bbf496748620276f4c7266383872c96f3], đáp án là: Lâm Bá Kiệt
  t=0 ANSWER | xác suất chọn hành động: ANSWER=0.560 SEARCH_BM25=0.127 SEARCH_DENSE=0.177 ABSTAIN=0.137
      đã xem=[f32466…] | ứng viên: Lâm Bá Kiệt | hợp lệ: True | chi phí bước: 0.0
```

Ca #2 ("cơ học lượng tử", trích dẫn albert einstein) và #3 ("có thể chơi nhạc bằng việc
nhấn vào bàn phím", trích dẫn dương cầm) cùng cấu trúc, xem log.

**Ca từ chối #4** — đúng thông điệp chuẩn, và ANSWER bị mask (không có ứng viên):

```
Trả lời: Chưa tìm thấy bằng chứng đủ để trả lời trong bộ tài liệu hiện tại.
  t=0 ABSTAIN | xác suất chọn hành động: ANSWER=0.000 SEARCH_BM25=0.168 SEARCH_DENSE=0.274 ABSTAIN=0.558
      ứng viên: (không có ứng viên) | hợp lệ: True
```

**Ca từ chối #5** — ca khó nhất: ANSWER **không** bị mask, policy *tự chọn* ABSTAIN sát nút
(tái hiện đúng hành vi eval):

```
  t=0 ABSTAIN | xác suất chọn hành động: ANSWER=0.365 SEARCH_BM25=0.144 SEARCH_DENSE=0.106 ABSTAIN=0.385
      ứng viên: Albert Einstein | hợp lệ: True
```

Ca #6: ABSTAIN=0.571, ANSWER=0.000 (ANSWER bị mask, cùng dạng #4).

**Vết thất bại #7** — giữ nguyên, kèm chẩn đoán:

```
Câu hỏi: Phạm Văn Đồng giữ chức vụ gì trong bộ máy Nhà nước Cộng hòa Xã hội chủ nghĩa Việt Nam?
Theo tài liệu [f32466…], đáp án là: Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa
  t=0 ANSWER | xác suất chọn hành động: ANSWER=0.468 SEARCH_BM25=0.129 SEARCH_DENSE=0.150 ABSTAIN=0.254
      ứng viên: Thủ tướng Chính phủ Việt Nam Dân chủ Cộng hòa | hợp lệ: True
```

Chẩn đoán: cả hai mốc nằm trong **cùng một passage** (gold "Thủ tướng" ứng với "Thủ tướng
đầu tiên của nước CHXHCNVN từ năm 1976…"; câu sai là chức danh thời Việt Nam Dân chủ Cộng
hòa 1955–1976, cách nhau một câu trong cùng đoạn). Reader chọn span đúng lịch sử nhưng sai
**thời kỳ mà câu hỏi hỏi** — lỗi biên span/thời kỳ điển hình của QA trích xuất. Trên P7 val
ca này có f1=0.333, EM=0; demo tái hiện đúng đáp án đó. Khớp bức tranh định lượng:
`answered_em` = 0.350 (val) / 0.293 (test) trong `reports/tables/baseline_results_*.csv`.

## Thời gian (đo được, run JSONL nguội cache)

`setup_ms = 28094` (một lần cho cả phiên: nạp reader + laya + policy + dựng dense bank 687
docs). Mỗi câu, tính bằng ms:

| # | total_ms | retrieval_ms | reader_ms | laya_ms | policy_ms |
|---|---|---|---|---|---|
| 1 | 6378 | 5755 | 189.9 | 406.9 | 0.72 |
| 2 | 5759 | 5676 | 23.3 | 39.9 | 0.42 |
| 3 | 5736 | 5646 | 24.8 | 42.2 | 0.32 |
| 4 | 5493 | 5410 | 22.5 | 38.7 | 0.30 |
| 5 | 5741 | 5653 | 23.9 | 39.0 | 0.30 |
| 6 | 5621 | 5533 | 24.7 | 41.2 | 0.36 |
| 7 | 5578 | 5490 | 24.4 | 40.8 | 0.39 |

→ thành thục **~5.5–6.4 s/câu**; run trace (cache hit) có laya_ms=0.0, setup_ms=27808.8.

**Phân rã `retrieval_ms`** (đo tách cùng ngày, cùng máy): `BM25Index.build` 687 docs =
**106 ms**, `index.rank` = **5 ms**; vòng đời E5 encoder tính theo câu — nạp lúc encode +
encode 1 truy vấn + unload — ≈ **4.7 s** thành thục (lần đầu 10.2 s; unload ≈ 0.55 s), trên
`cuda` (có thêm kiểm tra HF Hub mỗi lần nạp — cảnh báo unauthenticated trong log). Vậy
`retrieval_ms` bị chi phối bởi vòng đời encoder (quy tắc nạp/xả theo phase để giữ headroom
VRAM), không phải BM25. Chưa tối ưu (ví dụ giữ encoder thường trú) — ngoài phạm vi
acceptance; đây là điểm đầu tiên cần xử lý nếu cần latency production thấp hơn.

## Nguồn (bằng chứng trên đĩa)

| File | Nội dung |
|---|---|
| `logs/demo_acceptance_questions.txt` | 7 câu, đúng thứ tự chạy |
| `logs/demo_acceptance_selection.json` | bộ ca + nhãn chọn từ P7 val (dump pandas của DataFrame lọc) |
| `logs/demo_acceptance.jsonl` | kết quả JSON 7 dòng (run `--json`, đủ trường spec §1) |
| `logs/demo_acceptance_trace.log` | toàn văn run `--show-trace` (nguồn các trích đoạn) |
| `logs/demo_acceptance_run.log` | `JSON_EXIT=0`, `TRACE_EXIT=0` |
| `logs/demo_acceptance.err` | stderr: tiến trình nạp weights + cảnh báo HF Hub |
| `configs/final_inference.yaml` | config của lệnh spec |

Bộ ca được chọn từ predictions val: `outputs/predictions/p7-baselines-medium-val-42-2026-10-09T084929+0000_val_strict.parquet`
join `data/processed/questions_val_strict.parquet`; chi tiết trong `logs/demo_acceptance_selection.json`.

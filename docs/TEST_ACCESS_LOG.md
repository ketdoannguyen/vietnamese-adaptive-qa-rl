# TEST ACCESS LOG

Ghi lại mọi lần artifact chạm vào split `test_strict` (tạo dữ liệu test, cache, hoặc đánh giá trên nhãn test). Không xoá dòng cũ. Nguồn: outputs/runs/index.jsonl.

Kết luận: Đã khoá lựa chọn (2026-10-09T08:48:21+00:00) trước bản ghi test sớm nhất (2026-10-09T08:52:45+00:00); timestamp lấy từ manifest, không chống giả mạo.

| Thời điểm (UTC) | run_id | Stage | Status | Dùng nhãn test? | Profile |
|---|---|---|---|---|---|
| 2026-10-08T12:44:28+00:00 | p1-audit-split-smoke-2026-10-08T124402+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | smoke |
| 2026-10-08T12:45:28+00:00 | p1-audit-split-pilot-2026-10-08T124501+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T12:49:23+00:00 | p2-retrieval-pilot-2026-10-08T124923+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T12:49:23+00:00 | p2-retrieval-pilot-2026-10-08T124923+0000 | P2_retrieval | FAILED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T12:55:28+00:00 | p1-audit-split-smoke-2026-10-08T125501+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | smoke |
| 2026-10-08T13:11:01+00:00 | p1-audit-split-smoke-2026-10-08T131029+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | smoke |
| 2026-10-08T13:11:54+00:00 | p2-retrieval-pilot-2026-10-08T131154+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T13:12:24+00:00 | p2-retrieval-pilot-2026-10-08T131154+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T13:15:35+00:00 | p1-audit-split-pilot-2026-10-08T131507+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T13:25:54+00:00 | p1-audit-split-smoke-2026-10-08T132528+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | smoke |
| 2026-10-08T13:33:33+00:00 | p1-audit-split-smoke-2026-10-08T133306+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | smoke |
| 2026-10-08T13:34:52+00:00 | p1-audit-split-pilot-2026-10-08T133424+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T13:44:53+00:00 | p2-retrieval-pilot-2026-10-08T134453+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T13:45:28+00:00 | p2-retrieval-pilot-2026-10-08T134453+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T14:20:08+00:00 | p1-audit-split-pilot-2026-10-08T141934+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T14:20:09+00:00 | p2-retrieval-pilot-2026-10-08T142009+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T14:21:01+00:00 | p2-retrieval-pilot-2026-10-08T142009+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | pilot |
| 2026-10-08T14:44:14+00:00 | p1-audit-split-medium-2026-10-08T144254+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T14:44:16+00:00 | p2-retrieval-medium-2026-10-08T144416+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T14:48:23+00:00 | p2-retrieval-medium-2026-10-08T144416+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T16:25:38+00:00 | p2-retrieval-medium-2026-10-08T162538+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T16:29:45+00:00 | p2-retrieval-medium-2026-10-08T162538+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T17:00:14+00:00 | p1-audit-split-medium-2026-10-08T165734+0000 | P1_audit_split | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-08T17:03:59+00:00 | p2-retrieval-medium-2026-10-08T170359+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-09T00:26:21+00:00 | p2-retrieval-medium-2026-10-09T002621+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-09T00:28:05+00:00 | p2-retrieval-medium-2026-10-09T002621+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-09T00:37:54+00:00 | p2-retrieval-medium-2026-10-09T003754+0000 | P2_retrieval | RUNNING | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-09T00:38:06+00:00 | p2-retrieval-medium-2026-10-09T003754+0000 | P2_retrieval | SUCCEEDED | không (chỉ tạo/đọc dữ liệu test) | medium |
| 2026-10-09T08:52:45+00:00 | p3-reader-cache-medium-2026-10-09T085245+0000 | P3_reader_cache | RUNNING | có | medium |
| 2026-10-09T08:55:27+00:00 | p3-reader-cache-medium-2026-10-09T085245+0000 | P3_reader_cache | SUCCEEDED | có | medium |
| 2026-10-09T08:57:12+00:00 | p7-baselines-medium-test-42-2026-10-09T085712+0000 | P7_baselines | RUNNING | có | medium |
| 2026-10-09T09:14:32+00:00 | p7-baselines-medium-test-42-2026-10-09T085712+0000 | P7_baselines | SUCCEEDED | có | medium |

- best_selection.json: có (2026-10-09T08:48:21+00:00)
- Timestamp lấy từ file do các stage ghi; không có cơ chế chống sửa đổi — đây là mức chứng minh tốt nhất mà artifact hiện có thể đưa ra.

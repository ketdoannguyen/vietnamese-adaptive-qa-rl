# Kế hoạch: GLiNER ↔ Laya cho Vietnamese Adaptive QA, refactor tối thiểu và rerun có kiểm soát

## Context

Người dùng muốn tích hợp đặc tả trong `Vietnamese_Adaptive_QA_GLiner_Qwen_New_Prompts_Only.zip` vào code training hiện tại, đồng thời làm sạch code, đánh giá trung thực kết quả cũ, cải thiện tốc độ/chất lượng và lưu đủ bằng chứng để viết báo cáo.

**Đã xác nhận với người dùng:**
- GLiNER làm **decision model giống Laya**, không thay QA reader. Hai model chọn `ANSWER / SEARCH_BM25 / SEARCH_DENSE / ABSTAIN` trong cùng hệ QA.
- **Không Qwen** trong lần này: không dependency, config, checkpoint, nhánh code hay placeholder dành cho Qwen.
- **Pilot → full**; mở rộng train từ cap 5.000 lên toàn bộ `train_strict` khoảng 19.795 câu. Giữ danh tính val/test hiện tại.
- Ngân sách tối đa **48 GPU-giờ trên một NVIDIA L4 ~23 GB**. Đây là trần, không phải dự đoán thời gian hoàn thành hoặc cam kết mọi ablation đều chạy được.
- Phiên hiện tại chỉ khảo sát và lập kế hoạch; chưa sửa source, cài package, chạy pytest hoặc training.

**Project thật:** `/home/nkd/training-code/vietnamese-adaptive-qa-rl` — git repo; thư mục cha `/home/nkd/training-code` không phải git repo. Các đường dẫn tương đối bên dưới tính từ project thật.

**Vị trí tài liệu:** bản duyệt hiện tại nằm ở file plan do phiên plan-mode chỉ định. Sau khi được duyệt, lưu kế hoạch thực thi vào project `.claude/plans/2026-10-09-gliner-laya/PLAN.md`, đánh dấu quan hệ thay thế/phạm vi với các plan cũ; không chỉnh sửa hoặc xóa lịch sử plan cũ. Không tự đồng bộ tài liệu/code lên Notion hay dịch vụ ngoài.

---

## 1. Quyết định chính và giới hạn phạm vi

### 1.1 Phương án được đề xuất

**Mở rộng đúng chỗ nối decision prior hiện có**, không xây framework model mới và không sao chép pipeline Laya:

```text
Dữ liệu/split/scenario dùng chung
    ↓
BM25 + E5 → QA reader XLM-R dùng chung, được freeze
    ↓
Reader-state cache → teacher oracle dùng chung → decision dataset
    ↓                                      ↓
Laya CE / RLCD                         GLiNER2.5-Decide SFT
    ↓                                      ↓
          cùng hợp đồng prior 4 xác suất
                       ↓
       masked argmax hoặc cùng REINFORCE MLP
                       ↓
          cùng evaluation và report generator
```

GLiNER chỉ thêm adapter và trainer đặc thù. Reuse `AdaptiveQAEnv`, `RewardCalculator`, teacher, `PolicyNet`, REINFORCE, rollout, metric, split/group bootstrap, config, hashing và run registry.

### 1.2 Không làm

- Không `src/decision_models/`, `src/training/`, `src/evaluation/` song song với các package hiện tại chỉ vì ZIP gợi ý tên như vậy.
- Không plugin registry, abstract base hierarchy, nhiều tầng factory, framework trainer chung, orchestration mới.
- Không tự thay checkpoint GLiNER2.5 bằng GLiNER NER cũ hoặc classifier CLS tự chế nếu upstream không chạy được: ghi `BLOCKED` và chốt lại phạm vi.
- Không PPO/GRPO/DPO, joint reader-policy training, end-to-end RL qua `Classifier.score`, model lớn hơn, QLoRA mặc định, hoặc sửa reward để làm metric đẹp.
- Không đồng thời bật hàng loạt kỹ thuật rồi gọi đó là improvement đã được giải thích.
- Không xóa checkpoint/result cũ để lấy chỗ; không `--force` đè các canonical artifact lịch sử.
- Không hứa “nhiều epoch hơn, nhanh hơn và chính xác hơn” trước khi có pilot. Ba mục tiêu này được đo riêng.

### 1.3 Tiêu chuẩn thành công

1. GLiNER train-save-reload-infer thật, trả đủ 4 điểm từ model thật và đúng thứ tự action.
2. Hai model dùng chung episode/data/reader/reward/mask/evaluator; mọi khác biệt về objective, tuning, precision và trainable parameters được ghi rõ.
3. Không hồi quy legality/provenance/cache isolation; invalid action = 0, đáp án đã trả có provenance hợp lệ = 100%.
4. Có kết quả đầy đủ hoặc trạng thái thất bại/blocked rõ ràng, không bảng rỗng giả thành thành công.
5. Code ít trùng lặp hơn, không thêm pipeline RL thứ hai; baseline Laya vẫn chạy được khi không cài GLiNER extra.
6. Có actual compute accounting, không vượt 48 GPU-h; pilot quyết định số epoch/seed/trial có thể chạy.
7. Không bắt buộc GLiNER thắng Laya để integration được coi là đúng. Kết quả âm vẫn là kết quả nghiên cứu hợp lệ.

---

## 2. Đánh giá hiện trạng dựa trên bằng chứng

### 2.1 ZIP chứa gì?

ZIP chỉ có ba file Markdown, không có implementation/checkpoint/dữ liệu hay benchmark mới:
- `11_CODEX_ADD_GLINER25_MULTI_DECIDE_287M.md`: dùng trong lần này.
- `12_CODEX_ADD_QWEN35_08B_LORA_REINFORCE.md`: loại khỏi scope.
- `README_NEW_ONLY.md`.

Checkpoint đúng: **`fastino/GLiNER2.5-multi-Decide`**, 287M, encoder mDeBERTa-v3-base, kiến trúc boundary, thư viện **`gliner2`**. Không nhầm với package `gliner` hoặc mô hình GLiNER NER. Model card ghi Apache-2.0; benchmark tiếng Anh trên card không phải bằng chứng chất lượng tiếng Việt.

### 2.2 Kết quả cũ: giữ nguyên làm mốc lịch sử

Nguồn: `reports/results.csv`, `reports/tables/baseline_results_{val,test}.csv`, `outputs/models/qa_reader/reader_eval.json`, manifest P5 và `best_selection.json`.

| Nhánh hiện có | Val reward | Test reward | Val token F1 | Test token F1 | Nhận xét |
|---|---:|---:|---:|---:|---|
| Fixed Dense, E02 | -1,330802 | -1,381596 | 0,316175 | 0,330448 | F1 cao hơn nhưng trả giá retrieval cao |
| Laya CE argmax, E04 | -0,643737 | -0,586268 | 0,093587 | 0,093838 | Coverage thấp: val ~21,2% |
| Laya RLCD argmax, E05 | -0,571334 | -0,510563 | 0 | 0 | Luôn abstain; không phải QA tốt |
| Laya RLCD + REINFORCE, E06 | -0,489088 | -0,502629 | 0,159219 | 0,174813 | Trade-off reward/coverage/F1, không thắng mọi metric |
| Laya CE + REINFORCE, E07 | — | — | — | — | SKIPPED, chưa có đối chứng cần thiết |

Lưu ý định nghĩa: `src/eval/reporting.py:1-8` tính EM/F1 chung với abstention = 0. Không so trực tiếp cột này với F1 answerable-only hoặc SQuAD2 score có thưởng cho dự đoán rỗng đúng.

Reader hiện tại:
- Train 3 epoch, 5.237 windows từ 5.000 câu, 489 optimizer steps; phần train được ghi ~285,6 giây, **không phải toàn bộ P3**.
- Oracle-context answerable F1 = **0,475674**; retrieved-context = **0,437405**.
- False-answer rate trên impossible trong retrieved context = **0,841331**.
- Vì reader còn yếu ngay cả oracle context, chỉ đổi decision model không thể tự sửa chất lượng span. Tuy vậy con số oracle-single-context không phải upper bound tuyệt đối của mọi policy đa-context.

Teacher medium có 34.993 states:
- ANSWER 9.643; BM25 228; DENSE 180; ABSTAIN 24.942.
- Hai action search cộng lại ~1,17%. Accuracy đơn thuần rất dễ che giấu majority-class collapse.
- Calibration Laya hiện tại được report `is_held_out: false`; không dùng ECE/Brier fit-in-sample làm bằng chứng generalization.

### 2.3 Điểm mạnh cần giữ lại

- Split title-group, kiểm tra qid/doc/context leakage, reward/provenance constraints.
- Ba lớp cache retrieval → reader-state → decision prior; rollout RL không gọi encoder trực tiếp.
- Run registry, checksum artifact, freeze trước test, báo FAILED/SKIPPED/SUPERSEDED trung thực.
- 9 artifact hash được spot-check khớp trong khảo sát; điều này không thay cho full artifact verification sau triển khai.

### 2.4 Vấn đề cần giải quyết trước rerun

| Vấn đề | Bằng chứng / tác động | Hành động |
|---|---|---|
| Output chung từng bị ba seed ghi đè | D023, P6 bị SUPERSEDED rồi chạy lại tuần tự | Namespace và single-writer guard trước khi train |
| Renderer cắt mất thông tin quyết định | `state_text.py:90-121`; Laya thêm head 92 tokens, state room thực ~675 khi max_len=768 | Renderer mới dùng chung, đo truncation với tokenizer thực |
| Calibration hiện không group-held-out | Laya `split_calibration` chia từng item; không có `calib_data` public trong `finetune` | Tách train-only groups bên ngoài, fit và đánh giá tách biệt |
| Retrieval encode từng query | `build_retrieval.py:245-258` → `retrieval/engine.py:96` | Batch các query thiếu, giữ corpus BM25 per-scenario |
| Identity dense encoder chưa đủ | `retrieval_config_sha` thiếu revision; query encoder không truyền revision như bank | Pin cùng revision và fingerprint cả query/doc encoder |
| Prior cache rewrite toàn bộ mỗi chunk | `laya_cache.py:166-178`, `worlds.py:281-282` | Inference chia batch, flush một lần/split; không shard framework |
| Report in `{}` thay counts | `reports/REPORT.md`, `MODEL_CARD.md` | Sửa generator theo schema split manifest thực |
| Handoff trích seed 44 thay winner 43 | `FINAL_HANDOFF.md` vs `best_selection.json` | Winner lấy từ selection manifest, không latest run |
| Dependency thiếu lock | `pyproject.toml` không khai runtime deps; gliner2 chưa cài | Lock môi trường đã kiểm chứng + extra GLiNER |
| Kết quả pytest cuối chưa được xác nhận | Log cũ thiếu summary; cache từng có lỗi test report determinism | Chạy lại suite sau sửa, không tuyên bố hiện đang pass |

**Truncation được reviewer kiểm tra trên toàn bộ 34.993 states với tokenizer Laya thực:** khoảng 58,5% vượt state room 675; dòng candidate bắt đầu ngoài room ở khoảng 13,4%; depth-2 gần như đều bị truncate. Đây là phép đo chẩn đoán đọc-only, chưa có artifact audit tái lập trong repo: P0 phải xuất script/JSON đo lại. Không suy ra đây là nguyên nhân duy nhất khiến RLCD abstain. Mask vẫn được env áp dụng bên ngoài, nên mất dòng mask trong text không đồng nghĩa sinh illegal action.

### 2.5 Giới hạn khoa học không thể “reset” bằng code

`test_strict` đã được mở ngày 2026-10-09 sau freeze lịch sử. Rerun dùng cùng test phải ghi **reused/exposed holdout**, không gọi untouched confirmatory test. Không resplit dữ liệu đã xem rồi tuyên bố sạch. Tuning chỉ dùng train/validation; chỉ mở lại test sau selection manifest mới và ghi access log. Nghiên cứu confirmatory tương lai cần external/new untouched evaluation riêng, ngoài scope lần này.

---

## 3. Kiến trúc tối thiểu và reuse cụ thể

### 3.1 Hợp đồng decision prior

Giữ API đơn giản theo `src/decision/laya_adapter.py`:

```python
load(checkpoint, ...)
predict_probs(state_texts) -> ndarray[N, 4]
model_hash() -> str
close() -> None
```

- `predict_probs` trả **xác suất chưa mask** theo `ACTION_ORDER`. Không đổi semantics feature hiện tại.
- Mask áp dụng ở policy sampler / masked-argmax, env kiểm tra lần cuối.
- GLiNER lấy đủ bốn raw logits từ scoring API; lưu diagnostic raw scores nếu cần. Không chạy forward hai lần chỉ để lấy cả raw và probs.
- Không tạo logits giả cho Laya. Khi chỉ có probability, ghi `score_type=probability`, raw-logit diagnostic = null/N/A.
- `predict_probs` kiểm tra shape, canonical label mapping, finite, nonnegative và tổng ~1. Không biến lỗi model thành uniform probabilities.
- Một dispatcher nhỏ với nhánh `laya`/`gliner` hoặc bảng tên → loader cố định là đủ; **không dynamic registry**. Import GLiNER lazy để Laya/CPU tests không buộc cài extra.

### 3.2 Refactor đúng chỗ

1. `LayaProbCache` → `DecisionProbCache` trong `src/decision/cache.py`.
2. `cache_laya.py` → `cache_decision.py`; cùng một CLI `--decision-model laya|gliner --variant ...`.
3. `WorldBundle.laya_fn` → `prior_fn`; `merge_laya_fns` → `merge_prior_fns`; tham số `laya_regime` thành decision source/variant rõ ràng.
4. `LayaArgmaxPolicy` thành tên trung tính, vẫn dùng cùng masked argmax.
5. Giữ vector số và thứ tự tensor của RL. Nếu đổi `laya_p_*` thành `prior_p_*`, version feature schema/checkpoint mới rõ ràng; không giả định checkpoint cũ load tương đương chỉ vì cùng width. Replay cũ dùng revision code cũ.
6. `train_rl`, `select_rl`, `eval_baselines`, `inference/cli`, report và scripts chuyển cùng một lượt ở boundary này; không giữ cả API cũ và mới bằng lớp alias lâu dài.
7. Giữ `train_laya.py` và thêm `train_gliner.py`: training backend khác nhau là duplication hợp lý, không ép thành trainer trừu tượng.
8. Experiment rows định nghĩa một lần tại vị trí evaluator hiện có; report consume cùng metadata. Không tạo experiment DSL.

### 3.3 Cache identity và durability

Reuse hashing/io/cache code hiện có, nhưng cache-v2 phải ràng buộc:
- backend + checkpoint/base revision + adapter weights;
- tokenizer/word-splitter identity;
- action order + label descriptions/schema;
- renderer version + deterministic rendered state hash;
- input limits/truncation policy/score transformation/precision;
- calibration artifact hash và group-split hash.

Một inference fingerprint trong metadata đủ; không nhân bản toàn bộ metadata vào từng row. Khóa row vẫn `(qid, state_hash)` trong namespace fingerprint riêng. Reader/teacher/reward/scenario identity nằm ở upstream manifest và được kiểm tra trước dựng world. Khi semantic input đổi mà rendered text giống, vẫn kiểm tra dataset lineage, không dựa vào text hash một mình.

- Cache probability chung chỉ chứa probabilities + identity. Raw GLiNER score diagnostic riêng khi cần calibration/error analysis, không nhét fake `raw_*` vào Laya rows.
- Validate data SHA ở sidecar khi load; cập nhật parquet rồi sidecar theo quy trình atomic, crash giữa hai bước phải detect mismatch, không trả stale rows.
- Inference batched, chỉ persist khi cả split/chunk công việc định trước thành công; trước mắt một flush/split, chấp nhận recompute phần chưa flush khi crash.
- Cache miss trong rollout là lỗi; không gọi live model lén trong RL/eval.

### 3.4 Isolation mà không xây artifact framework mới

Dùng cơ chế `paths` overlay + output argument đã có:

```text
data/experiments/<experiment_id>/...
outputs/experiments/<experiment_id>/
  shared/                  # frozen reader, upstream manifests/cache
  decision/<backend>/<variant>/seed-<s>/
  rl/<backend>/<variant>/seed-<s>/
  selections/
  manifests/
  runs/
reports/experiments/<experiment_id>/
```

Đây là layout mục tiêu, không yêu cầu tạo mọi folder rỗng. Raw download có thể reuse read-only theo hash. Namespace full/pilot/smoke khác nhau, mỗi model/seed có run/config/checkpoint độc lập. Chỉ promote sau selection; không train trực tiếp vào canonical final path. Một lock file độc quyền tại output dir ngăn hai process ghi cùng target; không thêm queue service.

Historical `outputs/models`, `reports`, `data/processed` không bị sửa. Ghi inventory/hash và git SHA, không copy toàn bộ checkpoint thêm nhiều lần trên đĩa chỉ còn khoảng 34 GB.

---

## 4. Dữ liệu, renderer, teacher và calibration

### 4.1 Expand train có kiểm soát

- Khóa mapping qid/title-component → strict split từ baseline. Full = thêm câu còn lại thuộc TRAIN, không chạy một phép random split mới.
- Dùng khoảng 19.795 train-pool; ghi số thực sau audit/loại trùng/scenario generation. Medium 5.000 câu từng tạo 4.999 scenario: tìm và giải thích qid bị loại.
- Giữ danh tính, gold, group mapping và scenario corpus của val/test đã khóa. Nếu implementation không bảo đảm byte identity do serialization, đối chiếu semantic digest và ghi rõ khác biệt; không âm thầm đổi distractors val/test.
- Train full có thể đổi distractor pool so với medium. `scenario_id` một mình không chứng minh cache còn hợp lệ: kiểm tra `corpus_sha`, docs/questions/scenario/config hashes.
- Không mở test labels để tune reader/calibrator. P1 audit disjointness/provenance tách khỏi đánh giá chất lượng test; mọi thao tác chạm test labels ghi access purpose.
- Dùng negative audit hiện có; không tự relabel suspicious negatives thành positive hoặc bỏ val/test khó. Báo clean-vs-ambiguous như code hiện tại.

### 4.2 Một decision dataset canonical, hai converter mỏng

Reuse `make_teacher`, `teacher_row`, `ACTION_ORDER`, `ACTION_CRITERIA`, oracle và renderer.

Mỗi row hoặc metadata sidecar phải giữ: `qid`, `scenario_id`, title-component, depth/history, legal mask, rendered text/state hash, hard action, soft target nếu có, oracle Q/tie/ambiguity provenance, teacher method/version, reader/ranking/reward identity.

- Gold/teacher/Q-values nằm trong private target/metadata, **không vào rendered observation**.
- Candidate answer do reader cung cấp là input hợp lệ; không cấm candidate chỉ vì trùng gold text.
- Group split trước khi expand thành states: mọi state cùng qid/title-component đi cùng partition.
- Duplicate `(qid,state_hash)` được kiểm tra; không dùng forbidden-substring scan thay cho structural leakage tests.
- Label phải legal; oracle nhãn/soft mass cho action illegal phải theo contract được test.

**Label-mode chính mới:** hard action từ teacher, cùng `expected`/tie rule cho hai model. Laya triển khai bằng **`loss='soft-ce'` với one-hot target** hoặc bỏ soft `gold` để parser dùng `expected`; KHÔNG truyền loss `hard-ce` không tồn tại. Manifest dùng `target_mode=hard` để phân biệt với CE soft-target lịch sử.

GLiNER dùng `InputExample`/`Classification` của đúng version pin. Ba fixture phải save→load→inspect round-trip; không tự đoán `true_label` string hay list. Dù cùng hard labels, không gọi native GLiNER loss và Laya CE là cùng công thức nếu chưa kiểm tra implementation.

### 4.3 Renderer v3 chung cho cả hai model

Thay thứ tự, không thêm gold hoặc thông tin env không có:
1. Câu hỏi.
2. Candidate answer, doc/provenance, reader scores/null margin.
3. Budget, action mask, retrieval summary, history.
4. Evidence snippets có bound và thứ tự deterministic.

Reuse nội dung `serialize_state`, không tạo renderer mới cho từng backend. Chi tiết label descriptions thuộc adapter schema, không lặp không cần thiết trong text.

- Candidate/budget/action block phải tồn tại sau tokenization ở **cả hai model**; chỉ đặt lên đầu chưa đủ nếu câu hỏi rất dài: có fixture extreme và public deterministic budget policy.
- Token budget đo bằng encoder input thật, bao gồm schema/head/special tokens. GLiNER `max_len` word cap không được mặc định bằng Laya subword max_length.
- Log overall/depth/class truncation, candidate retained, evidence chars/tokens retained; hai tokenizer có thể giữ độ dài khác nhau, phải khai báo.
- Bump version/hash, rebuild teacher rendered rows + decision cache; reader-state cache không bị invalid chỉ vì đổi renderer.
- Regression v2→v3 với model cũ không được gọi behavior-preserving optimization. Historical v2 so riêng; pilot ablation v2/v3 cho cả model nếu có đủ ngân sách.

### 4.4 Calibration fit và model selection không trộn nhau

- Tách khoảng 10% **train title-components** làm `decision_calibration`, phần còn lại `decision_train`; tỷ lệ/count/group IDs được freeze trước train, không random từng state.
- Teacher dataset vẫn phủ full train pool; số supervised items thực dùng nhỏ hơn pool vì giữ calibration groups, report minh bạch.
- Validation chia group-stable `val_tune` và `val_report` để giảm tự đánh giá trên chính subset chọn checkpoint. Chỉ ~20 title groups nên báo kích thước/uncertainty, không tuyên bố hoàn toàn độc lập do lịch sử reader/val đã từng được sử dụng.
- Train weights trên decision_train. Chọn checkpoint/epoch/hyperparameters trên val_tune theo tiêu chí preregistered. Fit scalar temperature trên decision_calibration, sau đó đo calibration/action metrics trên val_report. Các checkpoint candidate không dùng test.
- Temperature của categorical probabilities giữ argmax, nên đừng hứa temperature sẽ sửa all-abstain argmax; tác dụng chính là chất lượng prior cho RL/confidence. Threshold thay đổi decision chỉ là ablation riêng.
- `P(ANSWER)` ≠ `P(answer correct)`. Giữ correctness estimator B5 riêng; không đưa action Brier/ECE vào cột answer-correctness Brier/ECE.

**Laya API cần dùng đúng:** `eval_data` chỉ đánh giá, không thay internal calibration fit. Đường triển khai ưu tiên wrapper mỏng quanh `laya.train.train_model(..., on_epoch_end=...)` và các hàm public load/parse/calibrate/save để dùng external grouped split, best checkpoint và fit temperature đúng. Không fork training loop, không monkeypatch thêm nhiều hàm. Callback eval phải khôi phục train/eval mode. Nếu version không cho wrapper nhỏ có thể kiểm thử đầy đủ, giữ fixed-epoch protocol và báo limitation; không giả vờ hỗ trợ resume/early stop.

---

## 5. GLiNER feasibility gate trước khi cài hoặc train lớn

Source chính thức đã đọc trong khảo sát:
- Model card đúng checkpoint.
- `gliner2/classification/engine.py`: `Classifier.score/batch_score` chạy dưới inference mode.
- `gliner2/classification/scoring.py`: mapping task→label→raw logit; labels theo schema mã hóa, không dựa thứ tự dict tùy ý; exclusive tasks dùng softmax, non-exclusive dùng sigmoid.
- `gliner2/training/trainer.py`: hỗ trợ boundary code path; `TrainingConfig` hiện có `lora_target_modules`, `num_epochs`, `task_lr`, `gradient_accumulation_steps`, `save_best`, `early_stopping` và `compute_metrics`.

Đây là **source inspection trên upstream đang di chuyển**, chưa là xác nhận wheel/runtime local. Gate phải:
1. Pin package/source commit và HF checkpoint SHA; ghi license, Python/torch/transformers compatibility; không nâng toàn bộ môi trường đang chạy chỉ để ép package mới.
2. Resolve dependency trong môi trường thử riêng trước, không sửa `.venv` baseline. Đưa về một tested lock nếu tương thích; nếu conflict lớn, dừng và nêu lựa chọn rõ, không xây remote-service fallback.
3. Verify imports/signatures thật của `AutoExtractor`, `Classifier`, `ExtractorTrainer`, `TrainingConfig`, `InputExample` và `Classification` trên version pin; mọi tên API trong P4/P5 ràng buộc gate này. Real Vietnamese forward; lấy đủ action names và bốn finite raw scores; so batch vs single, permutation label order, CPU/GPU tolerance.
4. Force schema exclusive `next_action`; không dùng top-label confidence để suy ra vector 4 phần tử. Nếu pinned release không cung cấp all scores, `BLOCKED` cho prior/RL.
5. Inspect module names/trainable parameters của boundary model. `lora_target_modules=['encoder']` chỉ dùng sau chứng minh target match; không chép `lora_targets` từ ví dụ khác version.
6. BF16 trên L4: với GLiNER config ghi **`fp16=False, bf16=True`** nếu supported, không chỉ bật bf16 mà để fp16 mặc định cùng true. AMP Laya đã có; không ghi như kỹ thuật mới chưa tồn tại.
7. LoRA train-save-reload với 20–50 states: finite loss/gradient, có trọng số trainable thay đổi, base-weight behavior đúng, predictions reload gần bằng pre-save theo dtype tolerance. Verify base revision+adapter dependency; không coi adapter vài MB là model hoàn chỉnh không cần base.
8. Checkpoint save/loading và actual resume semantics; nếu không lưu optimizer/scheduler/RNG thì ghi restart-from-checkpoint/restart, không full resume.

Gate fail không được đổi sang NER model khác hay giả score để tiếp tục.

---

## 6. Roadmap thực thi theo dependency và acceptance gates

### P0 — Bảo toàn baseline, contract và môi trường

**Files:** `pyproject.toml`, dependency lock/optional extra phù hợp hiện repo, `src/utils/runs.py`, config overlays; tests registry/report hiện có.

- Check git status và user changes trước mọi edit, không reset. Tạo branch làm việc nếu cần, không commit/push khi chưa được yêu cầu.
- Inventory baseline hashes, unresolved reader revision, data license, hardware, package versions, failed/superseded runs.
- Namespace/lock output, statuses `INTERRUPTED`/`SUPERSEDED` theo contract thực thay vì ghi ngoài validator.
- Tách experiment config `max_gpu_hours=48`, không đổi default global 12 cho mọi run không liên quan.
- GLiNER compatibility + real smoke gate §5; output audit trong namespace mới.
- Thêm một script đọc-only nhỏ `scripts/audit_decision_inputs.py` để đo input/head/token retention bằng tokenizer thật của mỗi backend; JSON audit ghi dưới namespace experiment. Chạy trên v2 làm mốc và chạy lại v3 ở P4; có fixture test, không giữ script thừa không được dùng trong verification.

**Pass:** baseline files không đổi; optional import không phá Laya; artifact/status tests pass; có runtime GLiNER verified hoặc blocker rõ ràng.

### P1 — Refactor boundary và cleanup có bằng chứng

**Files:** `decision/laya_cache.py`, `pipeline/cache_laya.py`, `pipeline/worlds.py`, `eval/policies.py`, caller train/select/eval/inference, tests liên quan.

- Rename/genericize cache + prior routing như §3, behavior-preserving trước khi thêm v3/model mới.
- One experiment row source; report không giữ bảng ID tự phát song song.
- Reuse `artifacts.short_split`, `capture_library_versions` thay các bản copy tương đương.
- Candidate xóa: `src/utils/logging.py`, `qa.reader.best_across_docs`. Trước xóa search toàn repo, entrypoints/scripts/tests/docs; cập nhật test file inventory nếu cần. `best_state_prediction` và `best_prediction_per_question` có semantics khác, không gộp tùy tiện.
- Không gom mọi `_fmt` thành helper chung; không split `report.py` thành hàng chục module chỉ vì dài. Chỉ extract thành phần được dùng lại thực sự.
- `run_p5_gpu_chain.sh` là script phục hồi lịch sử: chưa gọi “dead” nếu còn tài liệu/entrypoint dùng. Chỉ retire sau main DAG thay thế và kiểm chứng callers; vẫn giữ log lịch sử.

**Pass:** stub-Laya predictions/actions/rewards trước–sau giống nhau trên fixture; shape/order RL giữ nguyên; Laya-only smoke không cần GLiNER.

### P2 — Tăng tốc phần shared và chuẩn hóa full data

**Files:** `pipeline/build_retrieval.py`, `retrieval/engine.py`, `retrieval/dense.py`, `pipeline/audit_split.py`, `data/artifacts.py`, cache chung.

- Batch encode query cần tính; `rank_scenario` nhận query vector đã encode, vẫn giữ helper one-query cho inference nếu có caller thật.
- BM25 vẫn tính IDF trên từng scenario corpus, không đổi thành global BM25 để nhanh hơn.
- Pin dense revision thống nhất bank/query/cache config; hash docs/query text/scenario corpus và tokenizer/config phù hợp.
- Compare batched/unbatched embedding/score within tolerance, ranking/action determinism ở tie fixture; nếu ranking đổi vì precision, ghi new retrieval version và rebuild downstream.
- Flush decision cache một lần/split, load một adapter cho công việc cache đang chạy; không bỏ identity check để tránh load.
- Build train full và frozen val/test contract như §4.1. Runtime fixtures smoke/pilot dùng namespace riêng.

**Pass:** no stale cache hit, query encoder calls giảm từ O(n query) sang O(n/batch), cache write count giảm; benchmark actual walltime/RAM/VRAM; full lineage audit pass.

### P3 — Cải thiện và freeze reader dùng chung

**Files:** `qa/train_reader.py`, `pipeline/train_reader.py`, `qa/features.py`, `qa/reader_eval.py`, configs và reader tests.

- Kiểm tra alignment NFC/offset/window/CLS labels bằng fixture tiếng Việt, long context, impossible, multiple gold spans. Không thêm augmentation trước khi nhãn đúng.
- Pin XLM-R base SHA thật; không để `revision=unresolved` trong run mới.
- Reader run mới trên full train; mặc định trần **6 epoch**, có validation/best checkpoint theo đường eval hiện có. So checkpoints 4/6 trong cùng run; chỉ nâng cap lên **8** khi learning curve còn cải thiện và forecast đủ ngân sách, tối đa hai cấu hình LR nếu pilot cho phép. Không phải grid 3×2 bắt buộc.
- Fix `total_steps` floor nếu nó làm bỏ phần epoch: ưu tiên dùng epoch semantics thật của HF Trainer khi không có max_steps; unit test remainder/gradient accumulation, report actual `global_step` và examples seen. Không chỉ đổi `floor` sang `ceil` mà bỏ qua semantics trainer.
- Batch lớn hơn/accum nhỏ hơn trong VRAM; BF16 hiện có được giữ và đo. Dynamic padding/caching features chỉ thêm nếu benchmark có lợi và offset/label tests pass.
- Threshold chọn trên val_tune theo rule hiện tại được làm rõ: tối ưu QA F1 dưới ràng buộc false-answer/coverage; report precision-recall trade-off. Không cải thiện false-answer bằng cách abstain tất cả mà gọi QA tốt hơn.
- Advanced candidate ưu tiên: train-only retrieved hard-negative contexts, gold-in-window handling và answerability balance. Chỉ thử sau base full/epoch, mỗi lần một thay đổi; không dùng test passage/gold để mine.
- Freeze **một reader chung** trước P4. Từ đó chỉ rebuild teacher/prior/RL một lần cho reader đã chọn, tránh cascade sau mỗi trial.

**Pass:** finite learning/overfit gate; metric theo answerability và oracle/retrieved; provenance=100%; reader threshold/hash frozen; có best-vs-last và actual runtime, không chỉ loss.

### P4 — Teacher, renderer v3 và partition manifests

**Files:** `decision/state_text.py`, `decision/schema.py`, `decision/teacher.py` (chỉ khi cần metadata), `pipeline/make_teacher.py`, tests teacher/renderer/leakage.

- Shared renderer v3, audit retention và group-split như §4.
- Teacher rebuild từ reader/rankings/reward mới; giữ private metadata tách input.
- Canonical hard + soft target available, backend export không nhân bản oracle.
- Ghi histogram action/depth/budget/ambiguous, majority baseline, legal-label checks; GLiNER round-trip fixtures.

**Pass:** không group/qid leakage; sample counts giải thích đầy đủ; hashes teacher/config/reader/index; candidate/budget/mask retained theo cả tokenizer.

### P5 — Train decision models và calibration

**Files:** thêm `decision/gliner_adapter.py`, `pipeline/train_gliner.py`; sửa `pipeline/train_laya.py`; thêm helper calibration nhỏ chỉ nếu cả hai backend thực dùng; config experiment.

- Laya giữ library loss `soft-ce`/`rlcd`, target_mode explicit. Wrapper public train_model cho external grouped split + epoch callback; không copy optimizer loop.
- GLiNER dùng native `ExtractorTrainer`; baseline **LoRA r=8, alpha=16, dropout=0.05**, native target module đã Gate0 verify; LR task 2e-4 là điểm khởi đầu cần pilot, không hằng “tốt nhất”.
- Microbatch/accum và token limits từ benchmark, không ép cùng số batch khi tokenizer/model khác. Primary supervised exposure cùng data/epoch-cap và selection opportunities; actual walltime/trainable count/token volume ghi khác biệt.
- Tăng epoch có conditional: pilot cap 2; full target cap **4**, có thể **6** nếu convergence/time budget cho phép; fallback paired cap 2 nếu cost cao. Số này không override forecast gate.
- **Supervised checkpoint selection cố định:** val_tune action macro-F1 cao nhất; tie → uncalibrated categorical NLL thấp hơn; tie tiếp → epoch sớm hơn. Cùng validation groups và selection opportunities cho hai model. Rollout reward/coverage là diagnostic, không tiêu chí đổi winner hậu nghiệm. Khi chọn cấu hình system/deployment, dùng rule reward riêng ở P6. Calibration fit riêng, adapter identity hash sau calibration.
- Laya RLCD nhận target distribution và gọi thêm `soft_ce_loss` trong installed `laya/train.py:499-518`; one-hot là target hợp lệ theo source. Vẫn bắt buộc tiny hard-target RLCD finite-loss/gradient/update test trước LA03. Nếu pinned version không hỗ trợ, đánh dấu LA03 BLOCKED và hỏi trước khi đổi sang soft-target arm; không silent fallback làm mất matched objective.
- Log train/val curves, finite-gradient checks, class confusion, majority-collapse diagnostics, checkpoint epoch, actual examples seen, discarded/failed trials.
- Không training lại chỉ vì test kém.

**Pass:** cả hai save-reload đúng, raw-score GLiNER đúng semantics, calibration holdout, real params update, complete manifests.

### P6 — Shared cache, REINFORCE và matched evaluation

**Files:** `pipeline/cache_decision.py`, `worlds.py`, `train_rl.py`, `select_rl.py`, `eval_baselines.py`, `eval/policies.py`, `eval/reporting.py`, `eval/splitting.py`.

- Cache mọi reachable state cho từng backend/checkpoint; RL consume raw-unmasked prior probabilities cùng numeric features.
- Train REINFORCE chính sách giống nhau: architecture, rewards, action masks, episode IDs, update cap, early-stop rule, entropy, seed schedule.
- Seeds RL **42/43/44** chạy output riêng; mặc định GPU-heavy jobs tuần tự. RL cached có thể CPU để nhường GPU, không bắt chạy GPU cho MLP nhỏ.
- Thay selection subset 64 episodes thành fixed group-stratified validation probe lớn hơn (ví dụ 512 nếu đủ; fingerprint IDs), full val_tune evaluation lúc final selection. Budget eval giống nhau; không rút subset thuận lợi theo model.
- **RL/system selection cố định:** mean episodic reward trên val_tune cao nhất; tie → QA token F1 cao hơn; tie tiếp → mean retrieval cost thấp hơn; tie cuối → checkpoint sớm hơn. Không gọi macro-F1 của teacher là primary system metric. Freeze rule trước pilot, không thay sau nhìn kết quả.
- Report từng seed và mean±SD. Seed được chọn để deploy là hàng riêng, không thay mean-seed headline bằng best seed.
- Baselines B0–B5 rerun trên reader/scenario hiện tại, không lấy số medium cũ trộn vào bảng full mới. B5 cần OOF/correctness rebuild theo reader mới: OOF folds theo train title-components để fit estimator; partition group-disjoint bên trong val_tune cho correctness calibration và threshold selection; val_report chỉ đánh giá. Reuse `group_holdout`/OOF logic hiện có, ghi rõ group counts và không trộn artifact action-calibration với correctness-calibration. Nếu không đủ ngân sách ghi NOT_RUN, không reuse calibrator sai identity.

**Pass:** equal episode digest/fairness block, invalid=0/provenance=100%, no model call inside rollout, seed isolation, all required paired rows có kết quả hoặc status rõ.

### P7 — Freeze, test, report và handoff

**Files:** `select_rl.py`, `eval_baselines.py`, `pipeline/report.py`, `eval/reporting.py`, `scripts/run_all.sh`, `scripts/reproduce.sh`, `docs/TEST_ACCESS_LOG.md`, tests report/run_all/inference.

- Selection manifest per experiment/arm, đủ reader+decision+RL+calibration/data/config identities. Test guard kiểm tra freeze khớp, không chỉ `best_selection.json exists`.
- Một frozen evaluation batch cho các rows đã preregister, không adaptive test selection. Ghi reused-holdout status.
- Report generator sửa counts/winner citation/determinism; export bảng và plots từ structured outputs, không nhập số bằng tay.
- Demo end-to-end dùng cùng session reader/retrieval/decision đã freeze, ít câu validation; latency cold/warm thật, không gọi cached P7 time là online latency.
- Hash environment diagnostics và dependencies; reproduce command và expected artifact; failed/interrupted/skipped runs vẫn trong registry.

**Pass:** báo cáo generated-only đối chiếu predictions, hash manifest, reproducibility instructions; full tests và smoke có exit code/log; compute ledger ≤48h.

---

## 7. Ma trận thí nghiệm và cách phát biểu kết luận

### 7.1 Rows mới — không đổi ý nghĩa ID lịch sử E00–E07

| Nhánh | Decision | Training target/objective | Controller | Ưu tiên |
|---|---|---|---|---|
| LA00 | Laya base | Không SFT | masked argmax | Zero-shot reference nếu pin/load được |
| GL00 | GLiNER2.5-Decide base | Không SFT | masked argmax | Bắt buộc smoke/val |
| LA01 | Laya CE | Hard one-hot, library soft-ce | masked argmax | Chính |
| GL01 | GLiNER SFT LoRA | Cùng hard actions, native loss verified | masked argmax | Chính |
| LA02 | Laya CE frozen | Như LA01 | Shared REINFORCE | Chính |
| GL02 | GLiNER frozen | Như GL01 | Shared REINFORCE | Chính |
| LA03 | Laya RLCD | Cùng target mode hard, RLCD thật | masked argmax | Rerun nhánh hiện có sau main pair |
| LA04 | Laya RLCD frozen | Như LA03 | Shared REINFORCE | Rerun nhánh hiện có |
| B0–B5 | Heuristic/correctness | Dùng shared pipeline mới | Có sẵn | Rerun; status theo thực tế |

- Historical CE soft-target/RLCD-v2 giữ nguyên bảng lịch sử. Rerun hard-target không gọi là “exact reproduction” của objective soft-target cũ.
- Một soft-target bridge CE/RLCD chỉ là ablation thêm nếu cần giải thích lịch sử và đủ compute; không mặc định gấp đôi mọi training run.
- GLiNER LoRA vs Laya full fine-tune là **so sánh hai cấu hình training thực dụng**, không tách thuần ảnh hưởng architecture. Nếu muốn kết luận riêng architecture/adaptation method, thêm matched full-FT GLiNER ablation có ngân sách riêng; không suy luận từ bảng chính.
- Supervised seed tối thiểu 42 cho cả pair; nếu chỉ một supervised seed và ba RL seeds, ghi “conditional on one backbone seed”, không “3-seed end-to-end”. Mở rộng supervised seeds đối xứng khi forecast cho phép.

### 7.2 Câu hỏi nghiên cứu cụ thể

1. Với cùng reader/data/env, GLiNER SFT vs Laya CE: action quality, QA quality, reward-cost, latency, VRAM khác gì?
2. REINFORCE cải thiện gì khi cùng architecture controller nhưng khác frozen prior? So LA02−LA01 và GL02−GL01, không GL02 với LA01 để quy hết cho backbone.
3. RLCD còn collapse khi renderer/data/reader đã sửa? Đây là đo lại, không khẳng định đã sửa nguyên nhân.
4. Tăng data/epoch cải thiện QA hay chỉ action accuracy/abstention? So matched stages, không trộn old-medium và new-full thành model gain.

### 7.3 Ablation nâng accuracy, theo thứ tự và trần

1. Full train + reader best-epoch/threshold: ưu tiên vì reader weakness được đo.
2. Renderer v3 retention: applied shared; pilot v2/v3 paired nếu cần attribution.
3. Class imbalance: plain sampling trước; thử bounded class-balanced sampler hoặc loss weighting **train only**, không cả hai cùng lúc; áp dụng matched protocol hai model. Log effective class mix/duplicate exposure, không oversample validation/calibration.
4. Calibration on/off: rẻ sau có scores; action calibration khác answer correctness.
5. `no-prior` REINFORCE: reuse existing ablation, kiểm tra encoder prior có thực giúp hơn numeric features.
6. GLiNER LoRA rank 8→16 hoặc full-FT, train-only reader hard negatives: chỉ khi còn compute và có hypothesis/error category cụ thể.

Global `max_hparam_trials=8` gồm explicit trial ledger; bounded per-family allocation trước khi chạy. Training cân bằng lớp không phải “chỉ vài phút” vì vẫn phải forward/backward encoder; không đánh giá chi phí như fitting calibrator.

---

## 8. Performance và kế hoạch compute 48 GPU-h

### 8.1 Thứ tự tối ưu

1. Sửa batch-1 query encoding và repeated full parquet flush.
2. Cache upstream đúng identity, tránh rebuild/downloading không cần.
3. Một GPU-heavy model/process mỗi lúc; benchmark throughput end-to-end trước khi nghĩ chạy CE/RLCD song song.
4. Laya thử microbatch 1→2→4→8, điều chỉnh accum để giữ effective batch 8; đo grad-checkpointing on/off với VRAM headroom. AMP đang có sẵn, không coi “bật AMP” là cải tiến mới.
5. GLiNER BF16 nếu verified, batch theo measured length; reader BF16 hiện có.
6. Data preprocessing/tokenization CPU budget (máy ~4 vCPU/16GB RAM), không mặc định workers cao gây RAM/I/O contention.
7. `torch.compile`, fused optimizer, FlashAttention, quantization, custom CUDA không mặc định: chỉ cân nhắc sau profile cho thấy có lợi và model hỗ trợ.

### 8.2 Đo đúng thời gian

- Historical Laya CE/RLCD khoảng 4,4h/job cho 2 epoch medium **khi chạy cạnh tranh GPU**; không cộng hay nhân trực tiếp thành “GPU cost thật” hoặc giả định GLiNER có cùng tốc độ.
- Full train-pool ~3,96× medium questions, nhưng **val/test không 4×**, state depth/token distribution có thể khác; teacher actual state count là số dùng forecast.
- Pilot 300–1.000 states phủ action/depth/length strata; benchmark steady-state sau warmup, bao gồm train/eval/save/calibration overhead. Không đo chỉ 20 state ngắn đầu tiên.
- Forecast `train_steps × measured_seconds/step + eval + save + cache + reserve`, với upper estimate theo length strata; rerun estimate sau mỗi phase.
- Budget tính một L4 theo hợp các interval job sở hữu GPU; khi jobs tuần tự, tổng allocated walltime là kế toán bảo thủ. Không chỉ tính thời gian utilization>0 (sẽ bỏ sót I/O stall trong GPU job). Poll utilization chỉ để chẩn đoán; report tách stage walltime vs device allocation vs CPU-only time.
- Reserve inference/cache/test/failed retries trước khi bắt đầu epoch mới. Stage watchdog có time cap, soft-stop checkpoint/registry nếu supported; interruption không được ghi SUCCEEDED.

### 8.3 Phân bổ trần ban đầu, không phải ETA

| Hạng mục | Phân bổ GPU-h tối đa ban đầu |
|---|---:|
| Compatibility/real smoke + throughput pilots | 2 |
| Shared full retrieval + reader tuning/cache | 6 |
| Full matched pair Laya CE + GLiNER SFT | 22 |
| Laya RLCD rerun + **mọi backend prior cache**/shared RL/B5/eval | 8 |
| Targeted paired ablations/extra seeds | 3 |
| Final online evaluation/report-side measurements | 1 |
| Failure/OOM reserve | 6 |
| **Tổng** | **48** |

Các bucket có thể chuyển theo actuals nhưng không vượt tổng. Teacher/dataset rendering/audit là CPU time được log riêng; GPU reader-state cache thuộc bucket shared 6h, **tất cả** prior probability caches/calibration forwards của Laya và GLiNER thuộc bucket 8h, không chỉ RLCD. Nếu có GPU operation mới trong teacher/audit phải bổ sung forecast trước chạy. Chỉ pilot mới quyết định đủ hay không. Không giả định GLiNER nhanh gấp đôi hoặc “hai epoch cả hai model chắc chắn fit”.

**Thứ tự khi forecast vượt trần:**
1. Bỏ ablation/extra supervised seeds trước, ghi NOT_RUN.
2. Giảm epoch cap cho matched pair theo cùng protocol; giữ best checkpoints và báo actual epochs khác nhau do early stop.
3. Giữ **cả hai backbone trên full data** với ngân sách hợp lý thay vì full một model, medium model kia rồi gọi so sánh công bằng.
4. RLCD/B5 nếu không còn budget ghi NOT_RUN minh bạch và liệt kê phần “rerun toàn bộ” chưa hoàn thành; không vượt 48h để cố đủ checklist.
5. Nếu tối thiểu paired full vẫn không fit, dừng sau pilot và đưa actual forecast để người dùng chọn giảm scope hoặc cấp thêm compute. Không tự hạ model, đổi dữ liệu, hoặc chỉ scale một bên.

More epochs là mục tiêu có điều kiện: reader hướng 6–8; decision hướng 4–6 nếu throughput/convergence cho phép. Không chạy epoch thừa khi validation xấu đi chỉ để đạt số epoch.

---

## 9. Reporting phục vụ viết báo cáo

### 9.1 Một nguồn sự thật, không 15 bản Markdown nhập tay

Reuse run registry và evaluator:

```text
reports/experiments/<id>/
  REPORT.md                  # methods, results, discussion, limitations
  results.csv
  results.json               # cùng schema từ evaluated episode artifacts
  decision_metrics.json      # macro-F1, confusion, action calibration
  training_history.jsonl
  error_analysis.jsonl
  comparison_deltas.csv
  plots/
  artifact_manifest.sha256
  REPRODUCE.md
```

Model card và compute/environment có thể reuse generator hiện có. Không tạo file placeholder nói đã train. New run thất bại vẫn có manifest, command, exit code và reason.

### 9.2 Manifest tối thiểu mỗi run

- experiment/run ID, stage/backend/variant/target_mode, profile, seed và parent run IDs;
- resolved config + code git SHA/dirty-diff digest khi chưa commit;
- dataset revision/license status, split/group/scenario/teacher hashes;
- reader/ranking/renderer/tokenizer/labels/action-order identities;
- model/base/adapter revision, calibrated inference fingerprint;
- optimizer/LR/scheduler, epoch cap/actual, global steps/examples/tokens seen, batch/accum, precision, checkpointing;
- trainable/total params, hardware/dependency versions;
- epoch loss/validation learning curves, selected epoch + selection metric/reason;
- calibration fit/eval partition identities;
- walltime, allocated GPU-h, throughput, max allocated/reserved VRAM, disk bytes;
- status/exit/error, resume semantics, checkpoint files/hash, output paths;
- `test_exposure=reused_holdout`, access/freeze timestamps khi có test.

### 9.3 Metrics bắt buộc

**System QA:** mean reward (primary cost-aware objective), EM/F1 rõ denominator, answerable F1, answer coverage, wrong-answer rate/selective risk, impossible false-answer, abstention quality, BM25/dense calls và cost, action distribution, invalid-action/provenance, clean/ambiguous subgroups.

**Decision:** hard-label accuracy + macro-F1 + per-action precision/recall + confusion, NLL, action Brier/ECE (held-out only), teacher ambiguity/tie counts. Không dùng action accuracy thay QA quality.

**Compute:** seconds/epoch và time-to-target validation quality; train/infer throughput; total cost-to-result bao gồm cache/eval; cold load và warm online p50/p95; peak VRAM; base + adapter + deployment bytes. Cached rollout speed là metric riêng.

**Statistics:** paired episode alignment, paired cluster bootstrap delta theo title component (giữ qid/scenario pairing), 95% CI; seed mean±SD riêng. Chỉ ~20 val/test title groups nên báo limitation. Không gọi per-row CI chồng nhau/không chồng nhau là kiểm định paired difference. Không claim statistically significant nếu chưa có phép kiểm định preregistered và đủ evidence.

**Error taxonomy:** retrieval miss/unreachable, reader span mismatch/null error, premature abstain, wrong answer despite valid provenance, unnecessary search/cost, decision truncation, oracle ambiguity. Chọn ví dụ train/val deterministic, không cherry-pick vài demo đẹp; test examples chỉ sau freeze, không feed ngược tuning.

### 9.4 Handoff

Báo rõ: files changed, commands + exit codes, tests pass/fail/skipped, rows thực chạy, rows NOT_RUN, results/CI, actual compute vs cap, model-selection lineage, dependency/license limitations và bước khoa học nhỏ nhất tiếp theo. Không tự commit/push/upload artifacts.

---

## 10. Test và verification gates

### 10.1 CPU/offline tests (fake model, không cần download)

- GLiNER score mapping đủ bốn labels; đảo label order vẫn đúng; missing/duplicate/NaN/Inf/negative probabilities fail.
- Unmasked prior contract; ≥10 masking fixtures: no candidate, invalid provenance, search budget exhausted, empty ranking, only abstain legal; mask không ảnh hưởng raw-score diagnostics.
- Cache đổi từng identity field không hit; sidecar/data hash mismatch fail; flush frequency; cache-only rollout không gọi model.
- No-gold structural observation, legitimate candidate text allowed, group-disjoint partitions và repeat-qid/depth cases.
- Renderer long Vietnamese/NFC/diacritics; critical block retained theo actual budget; deterministic hash.
- GLiNER serialization round-trip; Laya one-hot vs soft-target exporter; action legal; absent calibration class handling được log, không invent examples.
- Reader windows/offsets/impossible/multiple spans; actual epoch/step counting; callback restores training mode.
- Seed/output race regression, statuses, config root isolation; baseline artifact không đổi.
- Evaluator same episodes, direct-vs-RL paired matrix, action vs answer calibration columns khác nhau.
- Report split counts khớp schema, winner seed đúng selection, rerun cùng inputs/now deterministic, missing metrics=N/A.
- Test-eval refusal trước freeze hoặc khi hash mismatched, no reused stale selection.
- Manifest completeness: mọi artifact khai báo phải tồn tại và hash đúng; thiếu file/sidecar phải fail. Report numeric values khớp aggregate từ episode artifacts, không có nguồn số thứ hai.
- Compute ledger: interval union không double-count overlap, không bỏ allocated GPU time khi utilization=0; projected job vượt cap bị chặn và reserve không bị tiêu âm.

### 10.2 GPU integration tests

1. Pin/load đúng checkpoint; một Vietnamese forward thật.
2. Tiny train 20–50 states: loss/gradient finite, trainable weights delta, save/reload prediction tolerance phù hợp BF16; optimizer state/resume khai đúng.
3. Batch/single parity, LoRA module match count, no gradient through inference scorer.
4. 100–300 episode smoke với real shared reader + GLiNER + existing env/RL, trace candidate/provenance/action budget.
5. Tiny Laya ↔ GLiNER cùng fixture/dataset/metrics schema; no test rows accessed.

### 10.3 Commands verification sau khi triển khai

Các lệnh sau là kế hoạch, **chưa chạy trong phiên này**:

```bash
# Từ project root; dùng interpreter/lock đã chốt sau compatibility gate.
.venv/bin/python -m pytest tests/test_reader.py tests/test_reader_pipeline.py -q
.venv/bin/python -m pytest tests/test_gliner_adapter.py tests/test_train_gliner.py tests/test_decision_cache.py -q
.venv/bin/python -m pytest tests/test_run_all.py tests/test_report.py tests/test_inference_cli.py -q
.venv/bin/python -m pytest -q
# ruff check src tests và formatter check bằng tool đã pin, không auto-format toàn repo.
```

Tên test mới phải được tạo đúng trong implementation; CLI `train_gliner`/`cache_decision` chưa tồn tại, chỉ viết lệnh smoke/pilot/full hoàn chỉnh vào REPRODUCE sau khi parser/config được kiểm thử. Với CLIs hiện có, `--dry-run` chưa chắc read-only (train_laya vẫn ghi registry), nên không dùng bừa trong plan-only.

Pipeline end-to-end sau implementation: smoke → pilot training/save/reload → full shared reader freeze → teacher/prior training → paired RL/validation → selection freeze → gated reused-test evaluation → generated report + artifact verification.

---

## 11. Bản đồ file triển khai tổng hợp

| Nhóm | Files hiện có cần sửa/reuse | Files mới/rename cần thiết |
|---|---|---|
| Decision | `laya_adapter.py`, `state_text.py`, `schema.py` | `gliner_adapter.py`; `laya_cache.py` → `cache.py` |
| Training | `pipeline/train_laya.py`, `make_teacher.py`, `qa/train_reader.py`, `pipeline/train_reader.py` | `pipeline/train_gliner.py`; helper calibration chung chỉ khi thực dùng |
| Pipeline boundary | `worlds.py`, `train_rl.py`, `select_rl.py`, `eval_baselines.py` | `cache_laya.py` → `cache_decision.py` |
| Shared performance | `build_retrieval.py`, `retrieval/engine.py`, `retrieval/dense.py` | Không thêm retrieval framework |
| Evaluation/report | `eval/policies.py`, `reporting.py`, `splitting.py`, `pipeline/report.py` | Không thêm evaluator GLiNER riêng |
| Live inference | `inference/cli.py`, inference config | Backend selector theo cùng adapter |
| Config/repro | `base.yaml`, profile overrides, `run_all.sh`, `reproduce.sh`, `utils/runs.py`, `pyproject.toml` | Một experiment base + pilot/full overlays nếu cần; tested dependency lock/extra |
| Audit/test | Reader/teacher/RL/cache/eval/report/inference/run_all tests hiện có | `scripts/audit_decision_inputs.py`; GLiNER adapter/train tests; cache test rename |

Thứ tự review theo commit-sized chunks: (1) isolation/evidence, (2) no-behavior-change reuse/cleanup, (3) data/performance correctness, (4) renderer/reader protocol, (5) GLiNER training, (6) RL/eval/report, (7) rerun. Không commit tự động; “commit-sized” chỉ nói độ lớn review.

---

## 12. Risks, stop conditions và điều kiện xem xét lại

| Risk | Guard / hành động |
|---|---|
| gliner2 source/docs/wheel khác API | Pin source; inspect + smoke thật; BLOCKED nếu không đủ scores/train/save |
| Model target không match LoRA boundary | Count/module assertions; không fallback model khác |
| Full multi-epoch vượt 48h | Pilot stratified + upper forecast + reserve; bỏ extras, giữ paired comparison |
| Reader tốt lên làm teacher/caches cũ sai | Freeze reader trước teacher; upstream hash invalidation đầy đủ |
| Search-label cực hiếm | Macro/per-class metrics + plain majority baseline; balancing train-only có ablation |
| Truncation gây bất công | Renderer chung, tokenizer-specific retention audit, apply change cho cả pair |
| Calibration/selection leakage | Group partitions, fit-vs-eval distinction, no test tuning |
| Refactor phá lịch sử | Artifact read-only + git revision + regression fixtures; không migration data cũ |
| Disk/RAM/VRAM thiếu | Dry estimate actual bytes; checkpoint retention trong namespace mới; không tự xóa lịch sử |
| “More epochs” làm overfit | Best epoch/held-out metrics + actual samples; stop nếu gain không còn |
| GLiNER không thắng Laya | Report negative result và bottleneck, không hidden tuning/test reruns |

Mở lại quyết định khi: upstream không hỗ trợ critical API; pilot forecast không đủ minimal pair; native loss/score semantics khác đặc tả; license/dependency conflict; new measured label/alignment issue; retention không đạt dù bounded renderer. Các lựa chọn ảnh hưởng scope/budget phải hỏi người dùng, không tự đổi mục tiêu.

---

## 13. Nguồn và trạng thái kiểm chứng

### Nội bộ
- ZIP member `11_CODEX_ADD_GLINER25_MULTI_DECIDE_287M.md` (spec, không executable truth).
- `src/decision/laya_adapter.py`, `laya_cache.py`, `state_text.py`; `src/pipeline/worlds.py`.
- `src/pipeline/train_laya.py`; installed `laya/train.py`, `common.py`, `calibrate.py` trong `.venv` (đã read-only review, pin lại khi implementation).
- `src/qa/train_reader.py`; `src/pipeline/build_retrieval.py`; `src/retrieval/engine.py`.
- `reports/results.csv`, baseline tables, reader_eval/train_report, manifests, `docs/TEST_ACCESS_LOG.md`, `docs/DECISIONS.md`.

### Chính thức bên ngoài (đã đọc, chưa runtime local)
- https://huggingface.co/fastino/GLiNER2.5-multi-Decide
- https://github.com/fastino-ai/GLiNER2
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/classification/engine.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/classification/scoring.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/gliner2/training/trainer.py
- https://raw.githubusercontent.com/fastino-ai/GLiNER2/main/tutorial/10-lora_adapters.md

ZIP dùng đường dẫn `tutorials/...` có URL trả 404; đường dẫn đã xác minh là `tutorial/...`. Context7 có snippets không đồng nhất `lora_targets`/`lora_target_modules`; implementation pinned source và runtime gate có ưu tiên hơn snippet.

**Evidence gate cuối của phiên lập kế hoạch:** khảo sát code/ZIP/results + read-only tokenization/metadata/hash checks đã thực hiện; không có GLiNER training mới, không xác nhận test suite pass, không có speedup/accuracy gain được đo sau refactor. Mọi improvement trong plan là hypothesis có verification gate, không phải kết quả đã đạt.

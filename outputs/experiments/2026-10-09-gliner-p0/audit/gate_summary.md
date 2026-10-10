# GLiNER2 P0 feasibility gate — results (2026-10-09)

Scope: plan `/home/nkd/.claude/plans/cheerful-coalescing-badger.md` §5, GLiNER P0 gate only.
Constraint honored: no training, no datasets, no baseline `.venv` / historical artifact writes,
no commit. All writes confined to `.venv-gliner/` + `outputs/experiments/2026-10-09-gliner-p0/`
+ two `.gitignore` lines.

## 1. Pins, license, compatibility (gate item 1)

- Package: `gliner2[train]==2.0.0` (PyPI; latest release; no `datasets` extra — resolver OK).
- Env: `/home/nkd/training-code/vietnamese-adaptive-qa-rl/.venv-gliner`, Python 3.10.12.
  Full freeze: `env/lock.txt` (53 pkgs). Wheel sha256: `env/wheel_sha256.txt`.
- Key runtime: torch 2.14.1+cu130, transformers 4.57.6, peft 0.21.2, tokenizers 0.22.2,
  accelerate 1.15.0, huggingface-hub 0.36.2, protobuf 7.36.2 (see §2), datasets ABSENT.
- Checkpoint: `fastino/GLiNER2.5-multi-Decide`, revision `e173bca1f0c4217d7a55b3b0c705d5ece8a0218d`,
  lastModified 2026-10-08, license **apache-2.0**, 287,355,159 params, arch `boundary`
  (encoder `deberta-v2`, `Gliner2ForSchemaExtraction`). `model.safetensors` sha256
  `9efe0f88…` (verified equal to HF LFS oid). All file hashes: `env/wheel_sha256.txt`.
- Cross-env constraint: gliner2 2.0.0 requires `transformers<5`; the checkpoint's
  `encoder_config.json` was serialized with transformers 5.17.0 and the baseline repo `.venv`
  has transformers 5.19.0. gliner2 must therefore run only from `.venv-gliner`; any future
  LAYA/GLiNER runner must invoke that interpreter explicitly. Baseline env untouched.

## 2. Resolver / tokenizer finding (the one blocker, resolved without substitution)

First load failed: transformers 4.57.6 cannot parse the checkpoint's 5.x-serialized
`tokenizer_config.json` (`extra_special_tokens` is a LIST, code expects a mapping) →
`AttributeError: 'list' object has no attribute 'keys'`. gliner2 2.0.0 **already ships the
shim** for this exact error (`gliner2/models/base.py::load_extractor_tokenizer` retries with
`extra_special_tokens={}`), but transformers' `except import_protobuf_decode_error():`
(missing protobuf) raised `ImportError` first and masked the AttributeError.

Fix: `uv pip install protobuf==7.36.2` into `.venv-gliner` only. Shim then engages
(UserWarning "legacy list-valued extra_special_tokens … compatibility normalization") and the
tokenizer loads as `DebertaV2TokenizerFast`, len 250112, with all 9 schema tokens as added
tokens (`[P]`=250104 `[L]`=250108 `[C]`=250105 `[E]`=250106 `[R]`=250107 `[EXAMPLE]`=250109
`[OUTPUT]`=250110 `[SEP_STRUCT]`=250102 `[SEP_TEXT]`=250103). No checkpoint file modified;
no model/library substitution; protobuf step added to `scripts/build_env.sh` for reproducibility.
Verbatim failure + mechanism: `logs/tokenizer_failure_before_protobuf.log`.

## 3. Verified API surface (gate item 3) — `audit/gate_smoke.json`

Signatures captured for `AutoExtractor.from_pretrained`, `Classifier.__init__`
(`*, device=None, dtype=None, scorer=None, result_builder=None, decoder=None`),
`Classifier.score` / `batch_score` (both `@torch.inference_mode()`),
`ClassificationConfig`, `ClassificationSchema.single/task`, `ClassificationScores.probability`,
`ExtractorTrainer.__init__(model, config, processor=None, train_data=None, eval_data=None,
compute_metrics=None)`, `TrainingConfig`, `create_classification_example`, `Classification`,
`InputExample`, `TrainingDataset`. Real Vietnamese forwards on both pinned state strings
(Thủ đô/Hà Nội and Bánh trôi nước/Hồ Xuân Hương):

- All 4 action names recovered from the model output equal ACTION_ORDER (name-keyed, not positional).
- 4 finite raw logits exposed; `probability()` == manual softmax (diff 0.0); Σp=1.0.
- Exclusive schema → softmax; non-exclusive control (`.multi()`) → sigmoid, Σp≈2.16 (documents
  the semantics; `.single()` remains the contract for `next_action`).
- Permuted label order: per-label logits identical (diff 0.0) — mapping is by name.
- Repeat determinism on GPU: bitwise equal.
- CPU fp32 vs GPU bf16: argmax equal; max prob deviation 0.0058 (bf16 rounding, expected).

## 4. Schema / scoring contract (gate item 4)

`ClassificationSchema().single("next_action", {4 labels}, instruction=…)` →
`is_exclusive=True`, `min_labels=max_labels=1`, `activation="auto"` → softmax over exactly the
4 labels. All four raw scores are available (`ClassificationScores.tasks["next_action"]`), so
the "BLOCKED if the pinned release cannot provide all scores" clause does not apply — PASS.
Labels/instruction are validated (`(`/`)` and structural tokens `[P] [L] [C] [E] [R]
[DESCRIPTION] [EXAMPLE] [OUTPUT]` are rejected) — schema strings must respect this.

## 5. Boundary model + LoRA targets (gate items 5–6)

- `task_module_names()` → `['classifier', 'boundary_head', 'record_decoder', 'relation_scorer']`.
- `lora_target_modules=['encoder']` proven: `_resolve_targets` yields 72 concrete modules,
  all `encoder.*` (attention q/k/v/dense + intermediate/output dense, 12 layers), all `nn.Linear`.
- `apply_lora(r=8, alpha=16.0, dropout=0.05, targets=['encoder'])` → `PeftModel`, 72 LoRA layers,
  1,327,104 trainable params = 0.4597% of 288,682,263; **only** `.lora_` tensors trainable,
  every non-LoRA base tensor frozen; init identity (prob diff vs base = 0.0).
- TrainingConfig precision semantics: defaults `fp16=True, bf16=False`; `bf16=True` alone
  raises `ValueError: Cannot use both fp16 and bf16`; required call is `fp16=False, bf16=True`.
  GPU bf16 forward on the L4 ran clean (no NaN/inf; eager attention fallback noted by the library).

## 6. Tiny real LoRA train-save-reload (gate item 7) — PASS, run 2026-10-09 18:06–18:08 UTC

First smoke pass proved the adapter save/reload mechanics with no optimizer step
(`training_steps_taken: 0`). The runtime training half was then run as a single ledgered GPU
job: `scripts/run_gate_item7.sh` → `scripts/gate_item7_train_reload.py`, report
`audit/gate_item7.json` (25 checks, all PASS), log `logs/gate_item7_train.log`.

Fixture: 40 states from `data/processed/laya_train.jsonl` only, chosen by fixed stride
(`fixtures/lora_states_train.manifest.json`); guards require `meta.split == "train_strict"`,
`privileged` falsy, and qid absent from both strict TEST and VAL qid sets — measured overlap
0/0. Action mix 33 ABSTAIN / 7 ANSWER / 0 SEARCH_*, which mirrors the source split
(71.3% / 27.6% / 0.7% / 0.5%); SEARCH_* are simply too rare to appear at n=40.

Train: 30 optimizer steps (batch 2, 2 epochs, `max_steps=30`), `fp16=False, bf16=True`
(strict), `strict_training=True`, LoRA r=8 α=16 dropout=0.05 on `encoder`. 144 LoRA tensors /
1,327,104 trainable params (0.4597% of 288,682,263); the optimizer holds LoRA params only at
`task_lr=5e-4`. All 30 losses finite (0.627 → 0.0095, epoch means 0.4599 → 0.2709); all 30
grad norms finite and positive (0.539 … 9.541, clipped at `max_grad_norm=1.0`).

Weight change: all 144 LoRA tensors changed, max |Δ| 6.54e-3, ΔL2 2.974; 663,552 of
1,327,104 LoRA values were exactly zero before training (= every `lora_B`, i.e. the documented
identity init). All 334 frozen base tensors bitwise unchanged (digest
`d3ca71b7…` before and after, max |Δ| 0.0).

Save/reload: trainer wrote `audit/lora_train_run/final/` — `adapter_config.json` +
`adapter_model.safetensors` (5,329,152 B, sha256 `7291f459…`, `peft_type: LORA`, r=8, α=16,
`base_model_name_or_path` = the pinned local model dir). `AutoExtractor.from_pretrained` on the
adapter dir alone fails (`OSError`), confirming the adapter is a delta and not a model.
Fresh-base `PeftModel.from_pretrained` reload → same 72 LoRA module names; **pre-save vs
post-reload logits and probabilities are bitwise equal** (max diff 0.0 over 40 states,
argmax 40/40) — far inside the bf16 tolerances (0.05 logit / 0.01 prob). With
`disable_adapter()` the base argmax is mostly SEARCH_DENSE/SEARCH_BM25; the trained adapter
differs by up to 9.086 logit and flips 33/40 argmaxes, so the adapter is a real behavioural
delta on the pinned base.

Honest scope limit: this gate proves the train→save→reload mechanics, **not** learning
quality. After 30 steps the adapter's argmax is ABSTAIN for all 40 states, i.e. it collapsed
to the 82.5% majority class and did not flip the 7 ANSWER states (epoch-2 mean loss 0.2709 is
consistent with p(ANSWER)≈0.21 on those rows). Any quality claim needs a real training run.

GPU accounting: ledger `audit/gpu_ledger.jsonl`, one interval, 77.051 s, SUCCEEDED (opened
before launch, closed after; precheck enforced `budget_ok` with `max_gpu_hours=2`). Train loop
bounded by `timeout 600`.

## 7. Checkpoint/resume semantics (gate item 8)

Source-verified in pinned `gliner2/training/trainer.py`:
- `_save_checkpoint` (line ~2092) saves adapter-only (PEFT native) or full merged weights; **no**
  `torch.save` of optimizer, scheduler, or RNG state exists anywhere in the trainer.
- `train()` re-creates optimizer/scheduler fresh (lines 1641–1642) and resets
  `self.global_step = 0` (line 1677); `load_checkpoint` (line 2207) restores model weights only.
- Therefore resume semantics = **restart-from-checkpoint (weights only)**: LR schedule, optimizer
  moments, and RNG are not restored. Any multi-stage plan must treat GLiNER fine-tuning as
  single-run-per-stage, or add external LR/step control.

## 8. Resource use

GPU: NVIDIA L4 (23,034 MiB), bf16 capable.

- Smoke (items 1–6, 8): peak VRAM 1,162 MB; forward section ~1 s per run; whole script ~27–29 s.
- Item 7 training run: peak VRAM **9,635 MB allocated / 21,070 MB reserved**; 30 optimizer steps
  in 24.2 s (2.49 samples/s); whole process 75.3 s. The reserved pool nearly fills the L4, so
  batch size or sequence length cannot be raised much before OOM.
- Ledgered GPU time: 77.051 s total (one interval). Far below the 2 h P0 budget.

## Verdict

PASS for gate items 1–8. Item 7 was completed after the initial smoke pass: 40 TRAIN-only
states, 30 real optimizer steps, all LoRA tensors changed, frozen base bitwise unchanged, and
pre-save vs post-reload predictions bitwise equal on a fresh pinned base. Single blocker found
and resolved in-env (protobuf → gliner2's built-in tokenizer shim); no substitution, no
checkpoint edits, no baseline-env change. The item-7 run proves mechanics only — the adapter
collapsed to the majority class, so it carries no learning-quality claim.

Artifacts: `audit/gate_smoke.json`, `audit/gate_item7.json` (item-7 machine report),
`audit/lora_adapter_roundtrip/` (smoke), `audit/lora_train_run/final/` (trained adapter),
`audit/gpu_ledger.jsonl` (GPU intervals), `fixtures/lora_states_train.{jsonl,manifest.json}`,
`env/lock.txt`, `env/wheel_sha256.txt`, `env/python_info.txt`, `logs/*.log`,
`scripts/build_env.sh`, `scripts/gate_smoke.py`, `scripts/make_lora_fixtures.py`,
`scripts/gpu_ledger_cli.py`, `scripts/gate_item7_train_reload.py`, `scripts/run_gate_item7.sh`.

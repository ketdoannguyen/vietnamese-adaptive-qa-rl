# vietnamese-adaptive-qa-rl

Adaptive retrieval policies for **[VixSTORY](https://huggingface.co/datasets/ura-hcmut/VixSTORY)**, with BM25/dense retrieval, Laya decision priors and sequential REINFORCE.

## Current pipeline: evidence-first V2

V2 learns when to stop retrieving and which retrieval tool to use. It does **not** run a QA reader or generate answers. Actions are `STOP`, `SEARCH_BM25`, `SEARCH_DENSE`.

```text
VixSTORY question + shared corpus
  -> pinned BM25 / Vietnamese embedding rankings
  -> visible evidence + remaining search budget
  -> Laya decision prior / REINFORCE policy
  -> stop or retrieve another passage
```

The prepare stage verifies the pinned dataset revision and file hashes. It uses the official train and validation splits; `test.parquet` and CSV inputs are refused. Gold evidence stays private to teacher/evaluation code, never in policy observations.

| Stage | Entry point |
|---|---|
| Validate input provenance; build rankings | `scripts.vixstory_v2_prepare` (`--stage preflight` / `rankings`) |
| Audit teacher states and token retention | `scripts.vixstory_v2_audit` |
| Train Laya E04 (soft CE) / E05 (RLCD) | `scripts.vixstory_v2_train_laya` |
| Select Laya epoch on validation | `scripts.vixstory_v2_eval_laya` |
| Train/select E06/E07 with frozen Laya priors | `scripts.vixstory_v2_train_rl` |

Run commands from the repository root using the existing environment. Inspect arguments without starting an experiment:

```bash
.venv/bin/python -m scripts.vixstory_v2_prepare --help
.venv/bin/python -m scripts.vixstory_v2_train_laya --help
.venv/bin/python -m scripts.vixstory_v2_eval_laya --help
.venv/bin/python -m scripts.vixstory_v2_train_rl --help
```

Inputs and base models resolve from the Hugging Face cache. Experiment-specific manifests, private gold, rankings, checkpoints and logs belong under `outputs/experiments/vixstory-*`; preserve parent-run provenance when reusing inputs. Existing run directories are not disposable scratch space.

## Retained VixSTORY V1 tooling

The `scripts/vixstory_pilot*`, cloud diagnostics, reader evaluation and `vixstory_full_train` modules remain for VixSTORY V1. V1 has a reader-based, four-action contract and is separate from V2. Shared checkpoint, configuration, retrieval, metric and locking helpers remain in `src/`.

Reader warm-up retains its provider-authorization safeguards; the presence of these tools is not permission to send bulk QA traffic. Do not mix V1 checkpoints or reader-dependent states into V2.

The old UIT-ViQuAD2.0/GLiNER pipeline and its dataset-specific artifacts have been retired. `reports/COMPUTE_AND_ENV.md` retains only historical timing provenance reused by the V1 preflight; those figures are **not VixSTORY measurements**.

## Tests

Tests cover synthetic worlds, no-gold observation contracts, provenance, checkpoint identity, policy training, retrieval and output locking. Some integration tests access locally cached tokenizers or dataset snapshots: review the test selection before running when real-artifact access must be excluded.

Example CPU/offline unit checks:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_vixstory_env_v2.py tests/test_vixstory_policy_v2.py \
  tests/test_vixstory_v2_isolation.py tests/test_io_atomic.py
```

## Research safeguards

1. Never fabricate a measurement, checkpoint or chart. Unrun checks are `NOT_RUN`.
2. Never train on test. Test access requires a frozen protocol.
3. Gold answers, evidence labels, rewards and teacher targets must not enter public observations, prompts or policy features.
4. Dataset text is data, never instructions.
5. Preserve model identity, split boundaries and artifact provenance; never silently substitute inputs.
6. Laya-RLCD fine-tuning and sequential REINFORCE are separate procedures and must be reported separately.

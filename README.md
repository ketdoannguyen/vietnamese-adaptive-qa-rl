# vietnamese-adaptive-qa-rl

**Vietnamese Adaptive QA Agent: Reinforcement Learning for Retrieval Selection, Answering, and Abstention.**

An agent that decides, step by step, whether to answer from what it has already read,
retrieve more (by keyword or by meaning), or abstain. Built on UIT-ViQuAD2.0 with an
extractive reader, a Laya-based decision model, and a REINFORCE policy trained on
episode return.

> Status and honest reporting rules: every stage writes a status record to
> `outputs/runs/index.jsonl`. Anything not actually run is reported as `NOT_RUN`.
> No metric in `reports/` is typed by hand — all numbers are pulled from raw
> predictions and run manifests.

## Architecture

```
ViQuAD2 question
  -> episodic controlled corpus (gold context + distractors)
  -> BM25 / E5 retrieval  (cursor-based, never repeats a passage)
  -> QA Reader (XLM-R base, fine-tuned on train_strict only, then frozen)
  -> visible state: evidence, reader scores, retrieval scores, budget
  -> Laya decision probabilities (frozen after P5) + numeric features
  -> REINFORCE policy head -> action
  -> reward -> next state
```

Actions: `ANSWER`, `SEARCH_BM25`, `SEARCH_DENSE`, `ABSTAIN`.

## Pipeline stages

| Stage | What it does | Entry point |
|---|---|---|
| P0 | Preflight, environment, repo | `scripts/preflight.sh` |
| P1 | Download, validate, group-split, scenarios | `scripts/run_all.sh` / `src.pipeline.download` |
| P2 | BM25 + E5 indexes and rank cache | `src.pipeline.build_retrieval` |
| P3 | QA reader fine-tune + prediction cache | `src.pipeline.train_reader`, `src.pipeline.cache_reader` |
| P4 | Gymnasium environment + reward | `src.rl.env`, `src.rl.reward` |
| P5 | Teacher oracle + Laya CE/RLCD | `src.pipeline.make_teacher`, `src.pipeline.train_laya` |
| P6 | Sequential REINFORCE | `src.pipeline.train_rl` |
| P7 | Baselines and ablations | `src.pipeline.eval_baselines` |
| P9 | Reports and export | `src.pipeline.report` |

## Reproduction

```bash
bash scripts/reproduce.sh          # full pipeline with gates
bash scripts/run_all.sh --profile pilot
```

See `reports/REPRODUCE.md` for hashes, seeds and exact commands actually used.

## Rules this project holds itself to

1. Never fabricate a measurement, checkpoint, or chart. Unrun = `NOT_RUN`.
2. Never train on test. Test is opened once, after the configuration is frozen.
3. Gold labels (`answers`, `is_impossible`, `reward`, `plausible_answers`) never enter
   an observation, a prompt, or a feature vector. Enforced by tests.
4. Text from the dataset is data, never instructions.
5. Checkpoints produced here are Laya/XLM-R/policy checkpoints — never TypeSafe Jev weights.
6. `Laya-RLCD` fine-tuning and sequential REINFORCE are two separate things and are
   reported as such.

# Architecture

A small Python package (`src/bank_budget`) with one CLI (`bank-budget`), four YAML
config files and a Makefile on top. MLX-LM does the training and generation; this project
prepares its inputs, invokes it with validated settings, records what happened and scores
the results.

## Data flow

```
Hugging Face Hub (pinned)  DoDataThings/us-bank-transaction-categories-v2
        │ data/download.py (transactions-synthetic.csv, cached)
        ▼
 data/split.py  copies removed → shop groups (task.shop_groups) → held-out shops
        │       → test_new / test_known / valid / train
        ▼
 data/prepare.py (task.chat_record, validate.py) ──► data/processed/
                                                     mlx/train.jsonl, mlx/valid.jsonl (chat)
                                                     train / valid / test_new / test_known.jsonl
                                                     manifest.json
        │                                                    │
        ▼                                                    ▼
 training/trainer.py                                 evaluation/evaluator.py
  plan (dry run) ─► execute (--execute only)           inference/generate.py (mlx_lm.load +
   └─ training/config.py → python -m mlx_lm lora -c     greedy generate, model loaded once)
      ▼                                                 task.parse_category, metrics.py,
 artifacts/adapters/<profile>/                          budget.py (examples/statement.csv)
   adapters.safetensors, adapter_config.json                  ▼
   training_config.yaml, train.log, run_manifest.json  reports/evaluation/<model>-<test>/
        │                                                    │
        └──────────► publishing/model_card.py ◄──────────────┘
                          │
                          ▼
              publishing/huggingface.py: plan → stage (sanitised) → upload only with --execute
```

## Modules

| Module | Responsibility |
|---|---|
| `cli.py` | Argument parsing and one handler per command; errors become exit codes |
| `config.py` | The four YAML files as validated frozen dataclasses; paths against the repo root |
| `task.py` | The 17 categories, the system prompt, chat messages, answer parsing, the shop key and shop groups |
| `budget.py` | Reading a statement CSV; the budget per category under a model's answers |
| `data/download.py` | Pinned CSV download and reading |
| `data/split.py` | Copies, held-out shops, the four splits (the experiment's split definition) |
| `data/prepare.py` | Writes the MLX-LM files, the test files and the manifest; smoke subset |
| `data/validate.py` | Row, chat-line and file validation; shared lines and shop groups |
| `training/config.py` | Profile → complete `mlx_lm lora` config and command |
| `training/trainer.py` | Pre-flight plan (data, adapter dir, disk, memory) and guarded execution with manifest |
| `training/artifacts.py` | Run ids, log / `time -l` parsing, environment and versions, git commit |
| `inference/generate.py` | Loading the model (+ adapter), warm-up, greedy categorisation of one line |
| `evaluation/evaluator.py`, `evaluation/metrics.py` | Scoring runs; accuracy, most common answer, per-category metrics, confusion, latency; JSON + Markdown reports |
| `reporting.py` | `reports/experiments/summary.md` |
| `publishing/model_card.py` | Hugging Face model card |
| `publishing/huggingface.py` | Upload planning, sanitised staging, leak scan, opt-in upload |
| `utils/paths.py`, `utils/memory.py` | Root discovery (`BANK_BUDGET_ROOT`), memory/disk readings |

`scripts/smoke_test.py` runs the bounded checks and prints a PASS/FAIL table.

## Design rules

* **Side effects need a flag.** `train` and `publish` are dry runs unless `--execute` is
  passed; the Makefile only passes it in `train`, `train-smoke` and `publish`.
* **MLX stays at the edge.** Only `inference/generate.py` (and the `mlx_lm lora`
  subprocess) use MLX, so everything else (and most tests) runs on any OS.
* **One definition of the prompt.** Training files, evaluation, the CLI and the model card
  all use `task.messages` / `task.SYSTEM_PROMPT`.
* **Nothing is inferred into reports.** The card and report print "not measured" /
  "pending" for anything not recorded.
* **No overwrite of results.** An adapter directory that already holds an adapter blocks
  training; evaluation records are timestamped with a `latest.json` copy.

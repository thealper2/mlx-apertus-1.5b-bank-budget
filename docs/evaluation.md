# Evaluation

## What is evaluated

| Test | File | Lines |
|---|---|---|
| `test_new` | `data/processed/test_new.jsonl` | 1,700 (100 per category) from shops never seen in training |
| `test_known` | `data/processed/test_known.jsonl` | 1,700 (100 per category), new lines of shops seen in training |
| `month` | `examples/statement.csv` | 31 made-up lines of one month, with amounts |

| Command | Model | Tests |
|---|---|---|
| `make evaluate-base` | base model | all three |
| `make evaluate [ADAPTER=dir]` | base + LoRA adapter (default `publishing.adapter_path`) | all three |
| `make budget [ADAPTER=dir]` | base + adapter | the month, with the budget table |
| `make evaluate-smoke` | base model | 2 lines per category of each test + the month |

Single runs: `uv run bank-budget evaluate --model base|adapter --test test_new|test_known|month
[--adapter DIR] [--smoke] [--max-examples N]`.

## Procedure

The same for both models: the model's chat template with the system prompt
(`task.SYSTEM_PROMPT`) and the line as the user turn, the answer turn opened; greedy
decoding (temperature 0) of at most 8 tokens. Three untimed warm-up generations come
first, so the times per line are fair. The answer is parsed by `task.parse_category`: an
answer that is a category name (any case, trailing dots ignored), otherwise the category
whose name occurs earliest as a whole word, otherwise `other` (wrong).

## Metrics

| Metric | Role |
|---|---|
| **Accuracy** on `test_new` and `test_known` | primary; the gap between them shows how much is memorised shops |
| Lines right in the month; the budget per category as filed vs. true | the practical check |
| Most common answer and its count | shows a model stuck on one category |
| Per-category recall / precision / F1, macro-F1, confusion matrix incl. `other` | diagnostic |
| Answers naming no category, generation failures, median / mean time per line | diagnostic |

## Outputs

`reports/evaluation/<model>-<test>[-smoke]/<timestamp>.json` and `.md`, plus `latest.json` /
`latest.md`, with model and revision, adapter path and run id, test file and its SHA-256,
generation settings, peak memory, environment, all metrics, every line's raw answer and
category, and for the month the budget table. `make report` combines them with the run
manifests into `reports/experiments/summary.md`. The model card uses a record only when
its test file is the canonical one (by SHA-256), its revision is the pinned one and, for the
adapter, its run id is the published run's.

## Measured so far (M1, 8 GB)

| Label | Lines | Right | Most common answer | Median time per line |
|---|---|---|---|---|
| `base-test_new-smoke` | 34 | 3 (8.8%) | Utilities (32×) | 366 ms |
| `base-month` | 31 | 3 (9.7%) | Utilities (28×) | 365 ms |

The full tests and the adapter have not been evaluated yet (only a 2-iteration smoke
adapter exists, which answers nothing usable).

## Limitations

* Both tests come from the same synthetic generator as the training data; real
  statements differ.
* A 95% interval at 1,700 lines is up to about ±2.4 points.
* 100 lines per category: the tests are balanced, real months are not.
* Free-text answers: a correct category phrased unusually ("Food & Dining") counts as
  wrong.

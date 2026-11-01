# Apertus 1.5B bank-statement budget categoriser with MLX

LoRA fine-tuning of **`swiss-ai/Apertus-v1.1-1.5B-Instruct`** to file US bank-statement
lines such as `[debit] SQ *COPPERWICK KETTLE CAFE 1187 DENVER CO` under one of 17 budget
categories — and to add a month of lines up into a budget — entirely on an Apple Silicon
Mac with [MLX-LM](https://github.com/ml-explore/mlx-lm).

The repository is an installable Python package with a CLI, YAML configs, a Makefile,
offline tests and an opt-in Hugging Face publishing workflow, built and tested for a
**MacBook Air M1 with 8 GB** of unified memory.

> **Status.** The pipeline is complete and verified with bounded smoke tests (including
> real 2-iteration training runs at the full batch size). No full training run or full
> evaluation has been performed yet, so there are no quality results in this repository.

## The experiment

| | |
|---|---|
| Base model | `swiss-ai/Apertus-v1.1-1.5B-Instruct` @ `9e9d01154446a645d30f04174cf1515a38058be7` (Apache-2.0 + Apertus Acceptable Use Policy) |
| Data | `DoDataThings/us-bank-transaction-categories-v2` @ `3e8d052cb7b362fb1a04b6064d2331982553ea56` (MIT, synthetic): 68,000 lines → 45,692 distinct |
| Split | 182 whole shop groups held out: train 35,639 / valid 850 / `test_new` 1,700 (never-seen shops) / `test_known` 1,700 |
| Format | system prompt with the 17 categories, the line as the user turn, the category as the answer; loss on the answer only |
| Method | LoRA r 8 / scale 20 on attention + MLP projections of all 16 blocks, AdamW lr 1e-4, 20 warmup + cosine, 1,000 iterations × 16, seed 0 |
| Metrics | accuracy on both tests; lines right and the budget for a month of made-up lines |

Full definition and every implementation decision:
[docs/experiment_specification.md](docs/experiment_specification.md).

## Requirements

* macOS on Apple Silicon (M1 or later) for training, inference and evaluation. Unit tests
  also run on Linux/Intel, without MLX.
* [uv](https://docs.astral.sh/uv/) (`brew install uv`). It installs Python 3.12.
* ≈ 3.5 GB disk for the model cache, plus ≈0.25 GB for a full run with its checkpoints.

### 8 GB

On an 8 GB M1 Air, batch 16 runs out of memory without gradient checkpointing; with it the
peak memory footprint is **5.8 GB** (also for the longest training batch). The default
profile `m1_air_8gb` is therefore the reference recipe with gradient checkpointing and
MLX's memory limited to 5 GB with a 0.5 GB buffer cache (without these a full run stopped
after the first validation pass) — the same updates. One iteration takes ≈15 s, so a full
run takes roughly 4–4.5 hours (extrapolated). Close other apps before `make train`.

## Quick start (no training)

```bash
make install          # .venv from uv.lock; downloads no model
make check-env        # Python, macOS, MLX, memory, disk, configs
make test             # offline unit tests
make data-prepare     # download the pinned CSV (~3.5 MB) and split by shop
make data-validate
make smoke-test       # bounded end-to-end checks; never trains or uploads
make train-dry-run    # show the exact MLX-LM config
```

`make help` lists every target.

## Workflow

```bash
make evaluate-base    # base model: test_new, test_known, the month
make train            # full training (PROFILE=m1_air_8gb): 1,000 iterations of 16
make evaluate         # base + adapter: test_new, test_known, the month
make budget           # the month as a budget table, filed by the adapter
make report           # reports/experiments/summary.md
```

Smoke versions: `make train-smoke`, `make inference-smoke [ADAPTER=…]`, `make evaluate-smoke`.
A passing smoke test says nothing about model quality.

### Inference

```bash
uv run bank-budget categorize --adapter artifacts/adapters/m1_air_8gb \
    --line "[debit] SQ *COPPERWICK KETTLE CAFE 1187 DENVER CO" \
    --line "[credit] QUILLSTONE SOFTWARE PAYROLL PPD ID: 7730021955"
uv run bank-budget evaluate --model adapter --test month   # the budget of examples/statement.csv
```

Every line must start with `[debit]` or `[credit]`. `--max-tokens` (default 8) and
`--output` are available; without `--adapter` the base model answers.

### Model card and Hugging Face

```bash
make card
make publish-dry-run REPO_ID=<you>/<repo>
uv run hf auth login
make publish REPO_ID=<you>/<repo>
```

See [docs/publishing.md](docs/publishing.md).

## Repository layout

```
configs/      data.yaml, training.yaml (reference / m1_air_8gb / smoke), evaluation.yaml, publishing.yaml
src/bank_budget/
  cli.py config.py task.py budget.py logging_config.py reporting.py
  data/        download, split (copies, shops, held-out shops), prepare, validate
  training/    config (MLX-LM options), trainer (plan / execute), artifacts (manifests)
  evaluation/  metrics, evaluator
  inference/   generate (MLX-LM load + greedy generate)
  publishing/  model_card, huggingface
  utils/       paths, memory
examples/     statement.csv (a made-up month, every merchant invented)
scripts/smoke_test.py
tests/        offline pytest suite (+ macOS-only MLX-LM compatibility tests)
docs/         experiment_specification, architecture, data, training, evaluation, publishing, troubleshooting
```

Generated, git-ignored: `data/processed`, `data/smoke`, `artifacts/`, `reports/evaluation`,
`reports/experiments`. See [docs/architecture.md](docs/architecture.md).

## Development

```bash
make lint && make typecheck && make test
```

CI (`.github/workflows/tests.yml`) runs lint, types and tests on Linux without MLX and the
full suite on macOS; it never downloads a model or trains.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md).

## License

Code and `examples/statement.csv`: MIT ([LICENSE](LICENSE)). The base model is Apache-2.0
(its card also asks users to follow the Apertus Acceptable Use Policy); the dataset is
MIT — see [CITATION.cff](CITATION.cff).

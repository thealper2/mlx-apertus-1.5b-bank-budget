# Troubleshooting

Start with `make check-env`; it reports most environment problems in one go.

## Installation and dependencies

**`uv: command not found`** — install uv (`brew install uv`), then `make install`.

**`uv sync --frozen` fails because the lock is out of date** — `pyproject.toml` was edited
without re-locking. Run `uv lock`, review the diff of `uv.lock`, commit it.

**`mlx-lm is not installed` / `MLX-LM needs macOS on Apple Silicon`** — MLX only ships
wheels for Apple Silicon macOS. Elsewhere the package installs without MLX so the unit
tests run, but training, inference and evaluation are unavailable by design.

## Hugging Face access

**`The dataset is not in the local Hugging Face cache…`** — you used `--offline` or
`HF_HUB_OFFLINE=1` before the first download, or the Hub is unreachable. Run
`make data-download` with network access.

**Rate-limit warnings ("unauthenticated requests")** — harmless; `uv run hf auth login`
with a read token raises the limits.

**Disk space** — the base model needs ≈3 GB in the Hugging Face cache, the dataset 3.5 MB,
each adapter checkpoint ≈21 MB (10 for a full `m1_air_8gb` run).

## Data

**`the dataset has N lines, expected 68000` / `… distinct lines … expected 45692`** — the
dataset revision in `configs/data.yaml` was changed. Restore it, or update the expected
counts deliberately (that is a new experiment).

**`… share N shop groups (test_new must be new shops)`** — `task.shop_key`,
`task.shop_groups` or a prepared file was changed after `make data-prepare`. Re-run it and
treat the result as a new experiment.

**`Canonical prepared data not found`** — run `make data-prepare` before `make data-smoke`.

## Training

**`[METAL] Command buffer execution failed: Insufficient Memory`** — on 8 GB, batch 16
needs gradient checkpointing and bounded MLX memory: use `PROFILE=m1_air_8gb` (the
default; `memory_limit_gb: 5.0`, `cache_limit_gb: 0.5`) and close other apps. MLX by
default schedules work up to 1.5 × the GPU working set and keeps freed buffers cached;
on 8 GB either pushes the total past what Metal gives the GPU, typically a few
iterations after the first validation pass. If it still happens, lower
`memory_limit_gb` in a copy of the profile (slower, same results).
If it still fails, copy the profile with `batch_size: 8`, `grad_accumulation_steps: 2`,
`iters: 2000` and the report/eval/save steps doubled (the same lines per update; MLX-LM
averages the two half-batches, which weights tokens slightly differently).

**`… already holds a trained adapter`** — adapter weights and completed runs are never
overwritten (a run that failed before saving any weights does not block a new one). Use
`--adapter-path artifacts/adapters/<new-name>` or move the old directory. For the
disposable smoke adapter: `make clean-artifacts`.

**Training is slow** — ≈12 s per iteration of 16 on an M1 is expected (≈3.5 h for
1,000). Much slower means swapping: check Activity Monitor → Memory.

**A run was stopped** — it is marked `interrupted`; checkpoints
(`0000100_adapters.safetensors`, …) stay in the adapter directory. MLX-LM can resume with
`resume_adapter_file`, but this project does not wire that in; start a new run in a new
directory.

## Inference and evaluation

**`… is not a trained adapter (missing …)`** — train first, or point `ADAPTER=` /
`--adapter` at a directory written by `bank-budget train`.

**Many answers are `other`** — the reply names no category. The base model and the smoke
adapter do this; a trained adapter should not. Check that the system prompt was not
changed (`make data-validate` checks it in the training files).

**The month evaluation refuses the statement** — `examples/statement.csv` needs the
columns `date, description, amount, category`, lines starting with `[debit]` or
`[credit]`, and one of the 17 categories per line.

## Publishing

**`No destination`** — set `repo_id` in `configs/publishing.yaml` or pass `REPO_ID=…`.

**`This is a smoke-test run`** — by design; publish a full training run.

**`Not logged in to Hugging Face`** — `uv run hf auth login` with a write token.

**`Staged files still contain sensitive data`** — a staged text file contains a token-like
string or your home-directory path. Inspect `artifacts/exports/hf_upload/`; nothing was
uploaded.

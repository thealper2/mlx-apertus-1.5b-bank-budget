# Training

LoRA fine-tuning with MLX-LM's own trainer (`python -m mlx_lm lora`). This project
generates its configuration, runs it as a subprocess and records the result.

## What is trained

LoRA of rank 8, scale 20, no dropout, on `q/k/v/o_proj` and the MLP's `up/down_proj`
(Apertus' xIELU MLP has no gate projection) of all 16 blocks: 5,373,952 trainable
parameters (0.356% of 1,510M). Every example is the system prompt, one statement line and
its category; the loss covers only the answer (`mask_prompt`).

## Profiles (`configs/training.yaml`)

| | `reference` | `m1_air_8gb` (default) | `smoke` |
|---|---|---|---|
| Purpose | the recipe | the recipe on an 8 GB Mac | prove the pipeline, measure memory |
| Data | `data/processed/mlx` (35,639 / 850) | same | `data/smoke/mlx` (32 / 16) |
| Iterations × batch | 1,000 × 16 | 1,000 × 16 | 2 × 16 |
| Learning rate | 1e-4: 20 warmup updates from 0, then `cosine_decay(1e-4, 1000, 1e-6)` | same | 1e-4 constant |
| Optimizer | AdamW: betas 0.9 / 0.999, eps 1e-8, weight decay 0.01, no bias correction | same | same |
| LoRA | rank 8, scale 20, dropout 0, 6 projections, 16 blocks | same | same |
| max_seq_length | 512 (examples are 109–155 tokens) | same | same |
| Gradient checkpointing | off | **on** | on |
| MLX memory limit / buffer-cache cap | MLX defaults | **5 GB / 0.5 GB** | 5 GB / 0.5 GB |
| Validation | all 850 lines at iterations 1, 500, 1000 | same | 1 batch |
| Checkpoints | iteration 1000 | every 100 | 2 |
| Seed | 0 | 0 | 0 |
| Output | `artifacts/adapters/reference` | `artifacts/adapters/m1_air_8gb` | `artifacts/adapters/smoke` |

The optimizer settings are MLX's own AdamW defaults, written out so a library change
cannot alter them. MLX-LM joins warmup and cosine at update 21, so the cosine runs for
updates 21–1000 and ends slightly above 1e-6. `m1_air_8gb` differs from `reference` only in
gradient checkpointing (which recomputes activations instead of storing them; the updates
are the same), in MLX's memory limit (5 GB instead of 1.5 × the GPU working set) and
buffer-cache cap (0.5 GB) — MLX-LM then runs through `bank_budget.training.mlx_launch`;
memory only — and in saving a checkpoint every 100 updates.

## Commands

| Command | What happens |
|---|---|
| `make train-config-check` | validates all configs and plans the `reference` profile |
| `make train-dry-run` | full plan + effective MLX-LM config for `PROFILE`; trains nothing |
| `make smoke-test` | bounded checks; plans every profile, never trains |
| `make train-smoke` | real 2-iteration run at batch 16 on the smoke subset |
| `make train` | **full training** (`PROFILE`, default `m1_air_8gb`) |

Without `--execute` the CLI only plans. A plan is blocked when the data are invalid, the
adapter directory already holds an adapter (never overwritten; use `--adapter-path`), or
free disk is low. The model snapshot at the pinned revision is downloaded once (≈3 GB)
and passed to MLX-LM as a local path; the subprocess runs with `HF_HUB_OFFLINE=1`.

## What a run leaves behind

`artifacts/adapters/<profile>/`: `adapters.safetensors` (final),
`NNNNNNN_adapters.safetensors` (checkpoints), `adapter_config.json` (contains the local
snapshot path; sanitised on publish), `training_config.yaml`, `train.log`,
`time_usage.txt` and `run_manifest.json` (run id, status, profile, provenance, train and
validation losses parsed from the log, wall time, peak memory, environment, versions, git
commit). A copy goes to `reports/experiments/<run_id>.json`.

## Measured on the target machine (MacBook Air M1, 8 GB, macOS 27.0)

| Run | Result |
|---|---|
| 2 iterations × 16, **no** gradient checkpointing | **out of memory** (`[METAL] … Insufficient Memory`), 6.08 GB footprint when it stopped |
| `make train-smoke` (2 × 16, checkpointing) | completed; 44.9 s wall time incl. model load and 2 validation passes; **peak memory footprint 5.80 GB**; train loss 9.37 → 9.52 |
| same on the 32 longest training lines (149–155 tokens) | completed; peak 5.81 GB |
| 5 × 16 with checkpointing | completed; 81.7 s; peak 5.88 GB → ≈12.3 s per iteration |
| Base model, one line | ≈0.37 s, 3.2 GB peak MLX memory |
| First full `make train` (no cache cap) | **stopped at iteration 12** with `Insufficient Memory`, 386 s in, right after the first pass over the 850 validation lines (239 s); peak footprint 5.97 GB |
| Same with MLX's buffer cache capped at 1 GB | stopped after iteration 20 (616 s), same error. MLX schedules work up to its memory limit, by default 1.5 × the GPU working set (8.2 GB), more than Metal gives the GPU (5.73 GB) |
| 40 iterations + 2 full validation passes, memory limit 5 GB, cache cap 0.5 GB | **completed**, 1,230 s (validation ≈290–340 s each, ≈14.6 s per iteration), peak footprint 5.89 GB; losses at iterations 10 / 20 (4.829 / 0.637) identical to the failed runs |

From the per-iteration time the 1,000 iterations of `m1_air_8gb` would take roughly
4 hours plus three passes over the validation set (≈5 minutes each); this is an
extrapolation, not a measurement. The first training losses are high (≈9) because the base model expects a
different reply format; the masked targets are the category tokens and the end-of-turn
token.

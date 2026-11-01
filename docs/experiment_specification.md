# Experiment specification

This document is the source of truth for **what the experiment is**. The code, configs and
tests implement it; if they ever disagree with this page, that is a bug.

## 1. Reference definition

| Aspect | Reference |
|---|---|
| Task | One US bank-statement line (`[debit] …` / `[credit] …`) → one of 17 budget categories: Restaurants, Groceries, Shopping, Transportation, Entertainment, Utilities, Subscription, Healthcare, Insurance, Mortgage, Rent, Travel, Education, Personal Care, Transfer, Income, Fees |
| Base model | `swiss-ai/Apertus-v1.1-1.5B-Instruct` (Apache-2.0), bf16 |
| Data | `DoDataThings/us-bank-transaction-categories-v2` (MIT), `transactions-synthetic.csv`: 68,000 synthetic lines |
| Copies | texts filed under two categories dropped; every other text kept once (first occurrence) → 45,692 lines |
| Shop key | lower case; `[debit]`/`[credit]` removed; up to three leading payment wrappers removed (`SQ *`, `TST*`, `PYPL *`, `ACH DEBIT`, … — the list is in `task.py`); cut at `;`, `*`, `#`, `ppd id`, `web id`, `entry descr`; everything from the first digit on removed (names starting with digits keep their head); the first two words of `[a-z&'.]` |
| Shop groups | union of "same first word of the key" and "same first 6 letters of the 4-word key without non-letters" → 964 groups |
| Held-out shops | `random.Random(0)`; per category (in the order above): groups whose most frequent category it is, sorted, shuffled, taken until ≥ 15% of those groups' lines, skipping groups above 30% → 182 groups |
| Splits | `test_new` = 100 lines per category of held-out shops (`DataFrame.sample(random_state=1)`, categories sorted); `test_known` = 100 per category of the other shops (seed 2); `valid` = 50 per category of what remains (seed 3); `train` = the rest, shuffled with seed 4 → 35,639 / 850 / 1,700 / 1,700 |
| Prompt | system: `You sort lines from a US bank statement into budget categories. Each line starts with [debit] (money out) or [credit] (money in). Reply with exactly one category from this list and nothing else: <the 17, comma-separated>.`; user: the line; assistant: the category |
| Training | `mlx_lm lora`: LoRA rank 8, scale 20, dropout 0, all 16 blocks, MLX-LM's default targets for Apertus (`q/k/v/o_proj`, `up/down_proj`); AdamW with MLX's defaults; lr 1e-4, 20 warmup updates, `cosine_decay(1e-4, 1000, 1e-6)`; 1,000 iterations × 16; `mask_prompt`; max_seq_length 512; validation on all of `valid` every 500 iterations; seed 0 |
| Evaluation | greedy, at most 8 new tokens, the same system prompt for base and adapter, 3 warm-up generations; answer = the category the reply is, else the earliest category name in it as a whole word, else wrong; accuracy on `test_new` and `test_known`, the most common answer, median time per line |
| The month | 31 statement lines of one month: lines right and the budget (sum per category) as filed vs. true |
| Output | LoRA adapter |

### Licenses

| Component | License | Notes |
|---|---|---|
| `swiss-ai/Apertus-v1.1-1.5B-Instruct` | Apache-2.0 | Its model card also asks users to follow the Apertus Acceptable Use Policy |
| `DoDataThings/us-bank-transaction-categories-v2` | MIT | Synthetic data |
| `examples/statement.csv` | this repository (MIT) | Written for this project; every merchant is invented |
| This repository's code | MIT | See `LICENSE` |

## 2. Verified properties

Checked on the target machine, not assumed:

* **Split.** The held-out shop groups (182), the shop group of every one of the 45,692
  lines, and the lines, categories and order of all four splits are identical to an
  independent re-derivation of the definition above from the pinned CSV.
* **Shop key.** Identical to the re-derivation for all 45,692 lines, with 2 and 4 words.
* **Prompt and parser.** The system prompt, the chat messages and the answer parser agree
  with the definition (parser on cases covering exact names, case, dots, several names,
  partial words and empty answers).
* **Trainable parameters.** 5,373,952 (0.356% of 1,510M), the count MLX-LM reports for the
  default targets; the explicit key list selects exactly these.
* **Data.** No line in two splits; no shop group of `test_new` in `train` or `valid`. The
  month shares no line with the dataset; only generic bank operations (`Zelle`, `ATM`,
  `MONTHLY`) share a first word with dataset lines.
* **Lengths.** Training examples are 109–155 tokens with the chat template; 512 never cuts.

## 3. Preserved, reorganised and changed

| Area | Status | Detail |
|---|---|---|
| Model, data, copies, shop key, groups, held-out shops, splits, prompt, LoRA, optimizer, schedule, iterations, batch size, masking, validation, seed, decoding, parser, metrics | Preserved | `configs/`, `task.py`, `data/split.py`; tests pin the values |
| Model revision, data revision | **Pinned (new)** | `9e9d0115…` and `3e8d052c…` (latest on 2026-10-30) |
| LoRA targets, optimizer settings | Made explicit | MLX-LM's defaults for Apertus and MLX's AdamW defaults written into the config (betas 0.9/0.999, eps 1e-8, weight decay 0.01, no bias correction) |
| Hardware profile | Added | `m1_air_8gb`: the same recipe with gradient checkpointing, MLX's memory limit at 5 GB and its buffer cache at 0.5 GB (identical updates; measured as required on 8 GB) |
| The month | Replaced | A new month written for this project (`examples/statement.csv`), every merchant invented |
| Smoke subset, manifests, validation, extra metrics (per-category metrics, confusion matrix, macro-F1) | Added | Accuracy stays the headline metric |
| Release | Adapter only | The LoRA adapter (MLX-LM format) with model card; no fused or converted model |
| Workflow | Implemented | Python package + CLI + Makefile; every step runs from a terminal |

## 4. Assumptions and limitations

* All lines are synthetic, from one generator; real statements differ.
* "Shop" is a heuristic key; it merges some different shops on purpose and can still split
  one shop written in very different ways.
* The month is one small, hand-made example.
* Full training on the M1 MacBook Air 8 GB has not been run yet; see `docs/training.md`.

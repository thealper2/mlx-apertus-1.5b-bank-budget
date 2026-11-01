# Data

## Source

[`DoDataThings/us-bank-transaction-categories-v2`](https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2)
(MIT), pinned to revision `3e8d052cb7b362fb1a04b6064d2331982553ea56`: one CSV
(`transactions-synthetic.csv`, ≈3.5 MB) of **68,000 synthetic** US bank-statement lines
with `description` and `category`. Every description starts with `[debit]` or `[credit]`;
the lines imitate the formats of several banks and payment services (`SQ *`, `TST*`,
`PYPL *`, store numbers, addresses, ACH ids).

## Splits (`data/split.py`)

1. **Copies.** 68,000 lines hold 45,692 distinct texts once texts filed under two different
   categories are dropped; every other text is kept once.
2. **Shops.** `task.shop_key` strips the sign, payment wrappers, ids and everything from the
   first digit on (`[debit] SQ *MOSSY ROCK BAKERY 0451 BOULDER CO` → `mossy rock`).
   `task.shop_groups` puts two lines in one group when their keys share the first word or
   the first 6 letters without spaces (`TRADERJOES` / `TRADER JOES`): 964 groups. It merges
   somewhat too much on purpose, so a held-out shop is really new.
3. **Held-out shops.** `random.Random(0)`; per category, shop groups whose most frequent
   category it is are shuffled and held out until they hold 15% of those lines (skipping
   any group above 30%): 182 groups.
4. **Tests.** `test_new`: 100 lines per category from held-out shops; `test_known`: 100 per
   category of other lines of the remaining shops.
5. **Training.** The rest of the remaining shops: 50 per category for `valid`, the other
   35,639 lines for `train` (shuffled with seed 4).

| Split | Lines | Purpose |
|---|---|---|
| train | 35,639 | MLX-LM training (`mlx/train.jsonl`) |
| valid | 850 | MLX-LM validation loss (`mlx/valid.jsonl`) |
| test_new | 1,700 | shops never seen in training |
| test_known | 1,700 | new lines of shops seen in training |

Seeds and sizes are in `configs/data.yaml`. No line is in two splits, and no shop group of
`test_new` is in `train` or `valid` (checked by `make data-validate`).

## Files

| File | Format |
|---|---|
| `data/processed/mlx/{train,valid}.jsonl` | `{"messages": [system prompt, user = line, assistant = category]}` |
| `data/processed/{train,valid,test_new,test_known}.jsonl` | `{"line", "category", "shop"}` |
| `data/processed/manifest.json` | provenance, counts, labels, overlap, SHA-256 of every file |

The training lines are 109–155 tokens long with the chat template (median 123).

## The month (`examples/statement.csv`)

31 lines of October 2026 for one made-up person, written for this project: every merchant
name is invented and none of the lines is in the dataset. Columns `date, description,
amount, category` (amount negative for money out). `make budget` files them and sums the
budget per category; see `docs/evaluation.md`.

## Commands

| Command | What happens |
|---|---|
| `make data-download` | download or reuse the CSV; print line counts |
| `make data-prepare` | split and write `data/processed/` |
| `make data-inspect` | lines, shops, categories, credits, line lengths per split |
| `make data-validate` | formats, the system prompt, counts, shared lines and shops |
| `make data-smoke` | `data/smoke/`: 32 training lines, 16 validation lines, 2 lines per category of each test |

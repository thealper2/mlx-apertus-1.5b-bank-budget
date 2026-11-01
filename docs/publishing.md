# Publishing to Hugging Face

Publishing is opt-in. Nothing is uploaded by installation, tests, smoke checks, training,
evaluation or `make card`.

## What a release contains

| File | Content |
|---|---|
| `README.md` | model card |
| `adapters.safetensors`, `adapter_config.json` | LoRA in MLX-LM format (~21 MB) |
| `run_manifest.json` | provenance: configuration, versions, losses, timings |
| `training_config.yaml` | the exact `mlx_lm lora` configuration (local paths removed) |

The release is only an adapter: `mlx_lm.load` downloads the base model
`swiss-ai/Apertus-v1.1-1.5B-Instruct` at the pinned revision and applies it. Checkpoints,
training logs, data, reports, caches and `.env` stay local.

## 1. Log in (once)

```bash
uv run hf auth login
```

Use a token with **write** access. It is stored by `huggingface_hub`; the project never
reads it from YAML, never prints it and needs it only for the final upload.

## 2. Choose the destination and the adapter

Set `repo_id` in `configs/publishing.yaml` or pass `REPO_ID=…`. `private: true` (default)
creates a private repository. `adapter_path` selects the trained adapter (default
`artifacts/adapters/m1_air_8gb`; override with `ADAPTER=…`). `adapter_license` sets the
card's license field (default `apache-2.0`, the base model's license); the base model's
card also asks users to follow the Apertus Acceptable Use Policy, which the model card
repeats. Confirm both yourself.

## 3. Model card

```bash
make card
```

Writes `artifacts/exports/model_card/README.md` from the configs, the data manifest, the
run manifest and the latest `base-*` / `adapter-*` evaluation records. A record is used only
when it evaluated the canonical test file (by SHA-256), with the pinned model revision, and
— for the adapter — with this exact run.

## 4. Dry run

```bash
make publish-dry-run REPO_ID=<namespace>/<name>
```

Checks without network access: valid repository id; the adapter directory holds
`adapters.safetensors`, `adapter_config.json` and a `run_manifest.json` with status
`completed` from a non-smoke profile; the model card exists; only the listed files (plus
the card) are selected. It then stages everything in `artifacts/exports/hf_upload/` with
local paths removed from JSON/YAML files, and scans every staged text file for token-like
strings and home-directory paths.

## 5. Upload

```bash
make publish REPO_ID=<namespace>/<name>
```

Same checks and staging, verifies the login, creates the repository if needed and uploads
the staged folder in one commit.

## Using a release

The model card's "How to use" section has the complete code: download the repository
with `snapshot_download`, `mlx_lm.load(base_model, revision=…, adapter_path=folder)`, then
the system prompt, the line as the user turn and greedy `generate` with at most 8 tokens.

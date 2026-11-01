# Bank-statement lines → 17 budget categories: LoRA fine-tuning of Apertus 1.5B with MLX.
# Run `make help` for the list of targets. Nothing here trains or uploads unless you
# call `make train`, `make train-smoke` or `make publish` yourself.

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

UV      ?= uv
RUN     := $(UV) run --frozen
CLI     := $(RUN) bank-budget
PROFILE ?= m1_air_8gb
ADAPTER ?=
REPO_ID ?=

ADAPTER_ARG := $(if $(ADAPTER),--adapter $(ADAPTER),)
REPO_ARG    := $(if $(REPO_ID),--repo-id $(REPO_ID),)

.PHONY: help install check-env lint format typecheck test \
        data-inspect data-download data-prepare data-validate data-smoke \
        train-config-check train-dry-run smoke-test train-smoke train \
        inference-smoke evaluate evaluate-base budget evaluate-smoke report \
        card publish-dry-run publish clean-cache clean-artifacts

## General
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [PROFILE=...] [ADAPTER=...] [REPO_ID=...]\n"} \
	     /^## / {printf "\n%s\n", substr($$0, 4)} \
	     /^[a-zA-Z0-9_-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Defaults: PROFILE=$(PROFILE) (training.yaml profile), ADAPTER=publishing.adapter_path"

## Environment
install: ## Create .venv and install the package + dev tools from uv.lock (no model download)
	$(UV) sync --frozen

check-env: ## Check Python, macOS, MLX, packages, memory, disk and configs
	$(CLI) check-env

lint: ## Ruff lint + format check
	$(RUN) ruff check src tests scripts
	$(RUN) ruff format --check src tests scripts

format: ## Apply Ruff formatting and safe fixes
	$(RUN) ruff check --fix src tests scripts
	$(RUN) ruff format src tests scripts

typecheck: ## Run mypy on the package
	$(RUN) mypy

test: ## Offline unit tests (no network, model, dataset or training)
	$(RUN) pytest

## Data
data-download: ## Download (or reuse the cached) pinned CSV (~3.5 MB) and count its lines
	$(CLI) data download

data-prepare: ## Split by shop and write data/processed (MLX-LM files + two tests)
	$(CLI) data prepare

data-inspect: ## Summary statistics of the prepared splits
	$(CLI) data inspect

data-validate: ## Validate data/processed (formats, sizes, shared lines, held-out shops)
	$(CLI) data validate

data-smoke: ## Write the tiny data/smoke subset (canonical data untouched)
	$(CLI) data smoke
	$(CLI) data validate --smoke

## Training
train-config-check: ## Validate all configs and plan the reference training profile
	$(CLI) config-check
	$(CLI) train --profile reference

train-dry-run: ## Show the effective MLX-LM config for PROFILE without training
	$(CLI) train --profile $(PROFILE)

smoke-test: ## Bounded pipeline checks; never trains, never uploads
	$(RUN) python scripts/smoke_test.py

train-smoke: data-smoke ## Tiny REAL training: 2 iterations of 16 (~5.8 GB, ~1 min after download)
	$(CLI) train --profile smoke --execute

train: ## FULL training with PROFILE (default m1_air_8gb: 1,000 steps of 16, ~4-4.5 h on an M1)
	$(CLI) train --profile $(PROFILE) --execute

## Inference and evaluation
inference-smoke: ## File three example lines with the base model (+ADAPTER)
	$(CLI) categorize $(ADAPTER_ARG) \
	  --line "[debit] SQ *COPPERWICK KETTLE CAFE 1187 DENVER CO" \
	  --line "[debit] CEDARLINE HOME LOANS MTG PYMT" \
	  --line "[credit] QUILLSTONE SOFTWARE PAYROLL PPD ID: 7730021955"

evaluate: ## Evaluate base + adapter (ADAPTER or publishing.adapter_path): both tests + the month
	$(CLI) evaluate --model adapter --test test_new $(ADAPTER_ARG)
	$(CLI) evaluate --model adapter --test test_known $(ADAPTER_ARG)
	$(CLI) evaluate --model adapter --test month $(ADAPTER_ARG)

evaluate-base: ## Evaluate the base model: both tests (1,700 lines each) + the month
	$(CLI) evaluate --model base --test test_new
	$(CLI) evaluate --model base --test test_known
	$(CLI) evaluate --model base --test month

budget: ## The month (examples/statement.csv) as a budget, filed by base + adapter (ADAPTER)
	$(CLI) evaluate --model adapter --test month $(ADAPTER_ARG)

evaluate-smoke: data-smoke ## Base model on 2 lines per category of each test + the month
	$(CLI) evaluate --model base --test test_new --smoke
	$(CLI) evaluate --model base --test test_known --smoke
	$(CLI) evaluate --model base --test month

report: ## Summarise recorded runs and evaluations in reports/experiments/summary.md
	$(CLI) report

## Publication
card: ## Generate the Hugging Face model card locally (no upload)
	$(CLI) card $(ADAPTER_ARG) $(REPO_ARG)

publish-dry-run: card ## Check and stage the upload; uploads nothing
	$(CLI) publish $(ADAPTER_ARG) $(REPO_ARG)

publish: card ## UPLOAD the adapter + card to REPO_ID (or publishing.repo_id)
	$(CLI) publish --execute $(ADAPTER_ARG) $(REPO_ARG)

## Maintenance
clean-cache: ## Remove Python/tool caches inside this repository only
	find src tests scripts -type d -name __pycache__ -prune -exec rm -r {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache

clean-artifacts: ## Remove disposable outputs: smoke data/adapter, upload staging (keeps real adapters + reports)
	rm -rf data/smoke artifacts/adapters/smoke artifacts/exports/hf_upload

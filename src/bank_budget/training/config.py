"""Translate a validated training profile into an MLX-LM ``lora`` config + command."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

from bank_budget.config import TrainingProfile

# Files MLX-LM needs to load the model (mirrors mlx_lm.utils' own download patterns).
MODEL_FILE_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]


def mlx_lora_config(profile: TrainingProfile, model: str, adapter_path: str) -> dict[str, Any]:
    """The complete option set passed to ``mlx_lm lora -c``.

    Every option is spelled out so MLX-LM defaults never fill in silently.
    ``model`` is a Hub id or the local snapshot directory of the pinned revision.

    Optimizer: every AdamW setting is written out (betas, eps, weight decay, bias
    correction), so MLX's defaults never apply implicitly. The schedule is counted in
    optimizer updates: with ``warmup_steps`` a linear warmup from 0, then MLX's
    ``cosine_decay(learning_rate, decay_steps, end_learning_rate)``; MLX-LM joins the two
    at update ``warmup_steps + 1``.
    """
    optimizer_options: dict[str, Any] = {
        "betas": [profile.adam_beta1, profile.adam_beta2],
        "eps": profile.adam_eps,
        "bias_correction": profile.bias_correction,
    }
    if profile.optimizer == "adamw":
        optimizer_options["weight_decay"] = profile.weight_decay
    lr_schedule = None
    if profile.lr_schedule == "cosine":
        lr_schedule = {
            "name": "cosine_decay",
            "arguments": [profile.learning_rate, profile.decay_steps, profile.end_learning_rate],
            "warmup": profile.warmup_steps,
            "warmup_init": 0.0,
        }
    config: dict[str, Any] = {
        "model": model,
        "train": True,
        "test": False,
        "data": profile.data_dir,
        "adapter_path": adapter_path,
        "fine_tune_type": profile.fine_tune_type,
        "optimizer": profile.optimizer,
        "optimizer_config": {profile.optimizer: optimizer_options},
        "learning_rate": profile.learning_rate,
        "iters": profile.iters,
        "batch_size": profile.batch_size,
        "grad_accumulation_steps": profile.grad_accumulation_steps,
        "max_seq_length": profile.max_seq_length,
        "num_layers": profile.num_layers,
        "lora_parameters": {
            "rank": profile.lora.rank,
            "scale": profile.lora.scale,
            "dropout": profile.lora.dropout,
            "keys": list(profile.lora.keys),
        },
        "mask_prompt": profile.mask_prompt,
        "grad_checkpoint": profile.grad_checkpoint,
        "steps_per_report": profile.steps_per_report,
        "steps_per_eval": profile.steps_per_eval,
        "val_batches": profile.val_batches,
        "save_every": profile.save_every,
        "seed": profile.seed,
    }
    if lr_schedule is not None:
        config["lr_schedule"] = lr_schedule
    return config


def write_mlx_config(config: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def training_command(
    config_path: Path,
    usage_path: Path | None = None,
    cache_limit_gb: float | None = None,
    memory_limit_gb: float | None = None,
) -> list[str]:
    """``python -m mlx_lm lora -c <config>``, wrapped in ``/usr/bin/time -l`` when available.

    With a memory or cache limit MLX-LM runs through :mod:`bank_budget.training.mlx_launch`,
    which sets them first (memory only; the training is unchanged). ``time -l`` (macOS)
    records wall time and peak memory footprint to ``usage_path``.
    """
    if cache_limit_gb is None and memory_limit_gb is None:
        cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    else:
        cmd = [sys.executable, "-m", "bank_budget.training.mlx_launch"]
        if memory_limit_gb is not None:
            cmd += ["--memory-limit-gb", f"{memory_limit_gb:g}"]
        if cache_limit_gb is not None:
            cmd += ["--cache-limit-gb", f"{cache_limit_gb:g}"]
        cmd += ["-c", str(config_path)]
    time_bin = shutil.which("time", path="/usr/bin")
    if usage_path is not None and time_bin and sys.platform == "darwin":
        cmd = [time_bin, "-l", "-o", str(usage_path), *cmd]
    return cmd

"""Guarded MLX-LM training entry point.

``plan_training`` performs every check without side effects (it is the dry run).
``execute_training`` runs the plan; it is only reached through an explicit
``--execute`` flag and never from tests, smoke checks or other commands.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bank_budget.config import ProjectConfig, TrainingProfile
from bank_budget.data.validate import chat_line_problems, validate_jsonl
from bank_budget.training.artifacts import (
    RUN_MANIFEST,
    TIME_USAGE,
    TRAIN_LOG,
    TRAINING_CONFIG_COPY,
    adapter_problems,
    environment_info,
    git_commit,
    new_run_id,
    parse_time_usage,
    parse_training_log,
    read_json,
    utc_now,
    write_json,
)
from bank_budget.training.config import (
    MODEL_FILE_PATTERNS,
    mlx_lora_config,
    training_command,
    write_mlx_config,
)
from bank_budget.utils.memory import (
    estimate_training_memory_gb,
    free_disk_gb,
    memory_snapshot,
    training_memory_warnings,
)
from bank_budget.utils.paths import display, resolve

log = logging.getLogger(__name__)

MODEL_WEIGHTS_GB = 3.02  # bf16 safetensors of the pinned model revision (3,020,063,528 bytes)
VOCAB_SIZE = 131_072
_CHARS_PER_TOKEN = 2.5  # deliberately pessimistic for bank lines full of ids and codes
_TEMPLATE_TOKENS = 16  # Apertus chat template markers around system, user and answer


class TrainingBlocked(RuntimeError):
    """The training plan failed a pre-flight check."""


@dataclass
class TrainingPlan:
    profile: TrainingProfile
    adapter_dir: Path
    data_dir: Path
    train_examples: int
    valid_examples: int
    estimated_seq_length: int
    estimated_peak_gb: float
    mlx_config: dict[str, Any]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _longest_example_chars(path: Path) -> int:
    longest = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            longest = max(longest, sum(len(m["content"]) for m in obj["messages"]))
    return longest


def plan_training(
    project: ProjectConfig, profile_name: str, adapter_path: str | None = None
) -> TrainingPlan:
    """Validate everything a training run needs, without changing anything on disk."""
    root = project.root
    profile = project.training.profile(profile_name)
    safety = project.training.safety
    data_dir = resolve(profile.data_dir, root)
    adapter_dir = resolve(adapter_path or profile.adapter_path, root)
    problems: list[str] = []
    warnings: list[str] = []

    counts = {"train": 0, "valid": 0}
    for split in ("train", "valid"):
        report = validate_jsonl(data_dir / f"{split}.jsonl", chat_line_problems)
        counts[split] = report.lines
        if not report.ok:
            problems.append(
                f"{display(report.path, root)}: {report.error_count} problem(s), e.g. "
                f"{'; '.join(report.errors[:3])}"
            )
    if counts["train"] and counts["train"] < profile.batch_size:
        problems.append(f"train split has {counts['train']} rows < batch_size {profile.batch_size}")
    if counts["valid"] and counts["valid"] < profile.batch_size:
        problems.append(f"valid split has {counts['valid']} rows < batch_size {profile.batch_size}")
    if counts["valid"] and counts["valid"] < profile.val_batches * profile.batch_size:
        warnings.append(
            f"val_batches x batch_size = {profile.val_batches * profile.batch_size} exceeds the "
            f"{counts['valid']} validation rows; MLX-LM will evaluate fewer batches."
        )

    # A finished run, or any adapter weights, are never overwritten. A run that failed
    # before saving weights leaves only its logs (its manifest is also kept in
    # reports/experiments/), so its directory can be reused.
    previous = read_json(adapter_dir / RUN_MANIFEST) or {}
    existing = sorted(p.name for p in adapter_dir.glob("*adapters.safetensors"))
    if previous.get("status") == "completed":
        existing.append(RUN_MANIFEST)
    if existing:
        problems.append(
            f"{display(adapter_dir, root)} already holds a trained adapter "
            f"({', '.join(existing)}). "
            "Previous adapters are never overwritten: pass --adapter-path <new dir> or move it"
            + (
                " (the disposable smoke adapter is removed by `make clean-artifacts`)."
                if profile.kind == "smoke"
                else "."
            )
        )

    disk = free_disk_gb(adapter_dir)
    if disk < safety.min_free_disk_gb:
        problems.append(f"Only {disk:.1f} GB free disk; need >= {safety.min_free_disk_gb} GB")

    seq = profile.max_seq_length
    train_file = data_dir / "train.jsonl"
    if not problems and train_file.is_file():
        chars = _longest_example_chars(train_file)
        seq = min(profile.max_seq_length, int(chars / _CHARS_PER_TOKEN) + _TEMPLATE_TOKENS)
    estimate = estimate_training_memory_gb(MODEL_WEIGHTS_GB, profile.batch_size, seq, VOCAB_SIZE)
    mem = memory_snapshot()
    warnings += training_memory_warnings(
        mem.total_gb, estimate, profile.max_seq_length, safety.warn_seq_length_8gb
    )

    adapter_rel = display(adapter_dir, root)
    config = mlx_lora_config(profile, project.training.model.repo_id, adapter_rel)
    return TrainingPlan(
        profile=profile,
        adapter_dir=adapter_dir,
        data_dir=data_dir,
        train_examples=counts["train"],
        valid_examples=counts["valid"],
        estimated_seq_length=seq,
        estimated_peak_gb=round(estimate, 1),
        mlx_config=config,
        problems=problems,
        warnings=warnings,
    )


def describe_plan(plan: TrainingPlan, project: ProjectConfig) -> str:
    """Human-readable summary of a plan (used for dry runs and before real runs)."""
    p = plan.profile
    m = project.training.model
    lines = [
        f"Training profile : {p.name} ({p.kind})",
        f"Base model       : {m.repo_id} @ {m.revision}",
        f"Data             : {display(plan.data_dir, project.root)} "
        f"(train {plan.train_examples}, valid {plan.valid_examples})",
        f"Adapter output   : {display(plan.adapter_dir, project.root)}",
        f"Method           : {p.fine_tune_type} on {', '.join(p.lora.keys)} of "
        f"{'all' if p.num_layers == -1 else f'the last {p.num_layers}'} blocks, rank "
        f"{p.lora.rank}, scale {p.lora.scale}, dropout {p.lora.dropout}",
        f"Optimisation     : {p.optimizer} (betas {p.adam_beta1:g}/{p.adam_beta2:g}, eps "
        f"{p.adam_eps:g}, weight decay {p.weight_decay:g}, bias correction "
        f"{p.bias_correction}) lr {p.learning_rate:g}, {p.lr_schedule}"
        f"{f' with {p.warmup_steps} warmup updates' if p.warmup_steps else ''}"
        + (
            f" then cosine_decay({p.learning_rate:g}, {p.decay_steps}, {p.end_learning_rate:g})"
            if p.lr_schedule == "cosine"
            else ""
        )
        + f"; {p.iters} iters "
        f"x batch {p.batch_size} (accumulate {p.grad_accumulation_steps}) = "
        f"{p.optimizer_updates} updates of {p.examples_per_update} examples",
        f"Sequence         : max_seq_length {p.max_seq_length}, mask_prompt {p.mask_prompt}, "
        f"grad_checkpoint {p.grad_checkpoint}, MLX memory limit "
        f"{'default' if p.memory_limit_gb is None else f'{p.memory_limit_gb:g} GB'}, cache cap "
        f"{'none' if p.cache_limit_gb is None else f'{p.cache_limit_gb:g} GB'}",
        f"Memory (rough)   : ~{plan.estimated_peak_gb} GB peak estimate, not a guarantee",
        "Effective mlx_lm lora config:",
        *("  " + ln for ln in json.dumps(plan.mlx_config, indent=2).splitlines()),
    ]
    lines += [f"WARNING: {w}" for w in plan.warnings]
    lines += [f"BLOCKED: {pr}" for pr in plan.problems]
    return "\n".join(lines)


def _download_model(repo_id: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    log.info("Resolving model snapshot %s @ %s (downloads ~3 GB once)", repo_id, revision)
    return Path(snapshot_download(repo_id, revision=revision, allow_patterns=MODEL_FILE_PATTERNS))


def execute_training(plan: TrainingPlan, project: ProjectConfig) -> dict[str, Any]:
    """Run MLX-LM training for an approved plan and write the run manifest."""
    if not plan.ok:
        raise TrainingBlocked("; ".join(plan.problems))
    root = project.root
    profile = plan.profile
    if profile.kind == "full":
        log.warning(
            "Starting FULL training (%s): %d iterations. This can take a long time and "
            "most of the machine's memory. Ctrl-C stops it; the run is then marked interrupted.",
            profile.name,
            profile.iters,
        )
    model = project.training.model
    snapshot = _download_model(model.repo_id, model.revision)
    adapter_dir = plan.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    config = dict(plan.mlx_config, model=str(snapshot))
    config_path = adapter_dir / TRAINING_CONFIG_COPY
    write_mlx_config(config, config_path)
    usage_path = adapter_dir / TIME_USAGE
    log_path = adapter_dir / TRAIN_LOG
    cmd = training_command(config_path, usage_path, profile.cache_limit_gb, profile.memory_limit_gb)
    run_id = new_run_id(profile.name)
    started = utc_now()
    log.info("Run %s: %s", run_id, " ".join(cmd))

    env = dict(
        os.environ,
        NO_COLOR="1",
        TERM="dumb",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_PROGRESS_BARS="1",
        PYTHONUNBUFFERED="1",
    )
    status, exit_code = "failed", None
    with log_path.open("w", encoding="utf-8") as log_fh:
        proc = subprocess.Popen(
            cmd,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log_fh.write(line)
            exit_code = proc.wait()
            status = "completed" if exit_code == 0 else "failed"
        except KeyboardInterrupt:
            proc.terminate()
            exit_code = proc.wait(timeout=30)
            status = "interrupted"

    parsed = parse_training_log(log_path.read_text(encoding="utf-8"))
    usage = parse_time_usage(usage_path.read_text()) if usage_path.is_file() else {}
    missing = adapter_problems(adapter_dir)
    if status == "completed" and missing:
        status = "failed"
    manifest = {
        "run_id": run_id,
        "status": status,
        "exit_code": exit_code,
        "started_at": started,
        "finished_at": utc_now(),
        "profile": profile.to_dict(),
        "base_model": {"repo_id": model.repo_id, "revision": model.revision},
        "dataset": _dataset_provenance(project),
        "mlx_lora_config": dict(config, model=f"{model.repo_id}@{model.revision}"),
        "artifacts": {
            "adapter_dir": display(adapter_dir, root),
            "train_log": display(log_path, root),
            "missing": missing,
        },
        "metrics": {
            "train_loss": parsed["train_loss"],
            "val_loss": parsed["val_loss"],
            "final_train_loss": parsed["train_loss"][-1]["loss"] if parsed["train_loss"] else None,
            "final_val_loss": parsed["val_loss"][-1]["loss"] if parsed["val_loss"] else None,
            "trainable_params_m": parsed.get("trainable_params_m"),
            "total_params_m": parsed.get("total_params_m"),
            "wall_time_s": usage.get("wall_time_s"),
            "peak_memory_gb": usage.get("peak_memory_gb"),
        },
        "environment": environment_info(),
        "git_commit": git_commit(root),
    }
    write_json(adapter_dir / RUN_MANIFEST, manifest)
    write_json(root / "reports" / "experiments" / f"{run_id}.json", manifest)
    log.info(
        "Run %s %s (exit code %s). Manifest: %s",
        run_id,
        status,
        exit_code,
        display(adapter_dir / RUN_MANIFEST, root),
    )
    return manifest


def _dataset_provenance(project: ProjectConfig) -> dict[str, Any]:
    d = project.data.dataset
    out: dict[str, Any] = {"repo_id": d.repo_id, "revision": d.revision, "file": d.file}
    manifest = project.data.output_dir / "manifest.json"
    if manifest.is_file():
        out["prepared_files"] = json.loads(manifest.read_text(encoding="utf-8")).get("files")
    return out

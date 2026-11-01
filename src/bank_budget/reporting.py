"""Experiment summary built only from recorded run manifests and evaluation records."""

from __future__ import annotations

from pathlib import Path

from bank_budget.config import ProjectConfig
from bank_budget.evaluation.evaluator import load_latest
from bank_budget.training.artifacts import RUN_MANIFEST, read_json, utc_now
from bank_budget.utils.paths import resolve

LABELS = tuple(
    f"{model}-{test}{suffix}"
    for suffix in ("", "-smoke")
    for model in ("base", "adapter")
    for test in ("test_new", "test_known", *(("month",) if not suffix else ()))
)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_experiment_report(project: ProjectConfig) -> str:
    """Markdown summary of every training profile's run and the latest evaluations."""
    m = project.training.model
    d = project.data.dataset
    lines = [
        "# Experiment report",
        "",
        f"Generated {utc_now()} from recorded outputs only.",
        "",
        f"- Base model: `{m.repo_id}` @ `{m.revision}`",
        f"- Dataset: `{d.repo_id}` @ `{d.revision}`",
        "",
        "## Training runs",
        "",
        "| Profile | Status | Run id | Iters | Final train loss | Wall time | Peak memory |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, profile in sorted(project.training.profiles.items()):
        run = read_json(resolve(profile.adapter_path, project.root) / RUN_MANIFEST)
        if run is None:
            lines.append(f"| {name} | not run | – | {profile.iters} | – | – | – |")
            continue
        mt = run.get("metrics", {})

        def fmt(key: str, suffix: str = "", mt: dict = mt) -> str:
            value = mt.get(key)
            return "n/a" if value is None else f"{value}{suffix}"

        lines.append(
            f"| {name} | {run.get('status')} | {run.get('run_id')} | {profile.iters} | "
            f"{fmt('final_train_loss')} | {fmt('wall_time_s', ' s')} | "
            f"{fmt('peak_memory_gb', ' GB')} |"
        )
    lines += [
        "",
        "## Evaluations (latest per label)",
        "",
        "| Label | Adapter run | Lines | Accuracy | Most common answer | Median time | Date |",
        "|---|---|---|---|---|---|---|",
    ]
    for label in LABELS:
        ev = load_latest(project.evaluation.output_dir, label)
        if ev is None:
            lines.append(f"| {label} | – | – | not evaluated | – | – | – |")
            continue
        mt = ev["metrics"]
        run_id = (ev.get("adapter") or {}).get("run_id") or "–"
        top = mt["most_common_answer"]
        lines.append(
            f"| {label} | {run_id} | {mt['examples']} | {mt['correct']} "
            f"({_pct(mt['accuracy'])}) | {top['category']} ({top['count']}) | "
            f"{mt['median_latency_ms']} ms | {ev['evaluated_at'][:10]} |"
        )
    lines += [
        "",
        "Smoke labels use 2 lines per category and a 2-step adapter; they only show that the "
        "pipeline runs and say nothing about model quality.",
    ]
    return "\n".join(lines) + "\n"


def write_experiment_report(project: ProjectConfig) -> Path:
    path = project.root / "reports" / "experiments" / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_experiment_report(project), encoding="utf-8")
    return path

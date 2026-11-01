"""Evaluation runner: file every test line and score the answers.

The model is loaded once by the caller and passed in as a ``categorize`` callable, so the
scoring logic is testable without MLX and never reloads weights per line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bank_budget.evaluation.metrics import aggregate
from bank_budget.inference.generate import Generation
from bank_budget.task import CATEGORIES, NO_CATEGORY
from bank_budget.training.artifacts import read_json, utc_now, write_json

log = logging.getLogger(__name__)

CategorizeFn = Callable[[str], Generation]
LATEST = "latest.json"


def evaluate_lines(
    rows: Sequence[dict[str, Any]], categorize: CategorizeFn
) -> list[dict[str, Any]]:
    """File and score every line in order (deterministic for greedy decoding)."""
    results = []
    for i, row in enumerate(rows, start=1):
        try:
            gen: Generation | None = categorize(row["line"])
            error = None
        except Exception as exc:  # one failure must not abort the run
            log.warning("Generation failed for line %d: %s", i, exc)
            gen, error = None, f"{type(exc).__name__}: {exc}"
        prediction = gen.category if gen else NO_CATEGORY
        results.append(
            {
                "line": row["line"],
                "category": row["category"],
                **({"shop": row["shop"]} if "shop" in row else {}),
                "raw_output": gen.raw if gen else None,
                "prediction": prediction,
                "correct": prediction == row["category"],
                "generation_error": error,
                "latency_ms": round(1000 * gen.latency_s, 1) if gen else None,
            }
        )
        if i % 200 == 0 or i == len(rows):
            log.info(
                "%d/%d evaluated, %d correct", i, len(rows), sum(r["correct"] for r in results)
            )
    return results


def build_result(label: str, results: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Machine-readable evaluation record."""
    return {
        "status": "measured",
        "label": label,
        "evaluated_at": utc_now(),
        **meta,
        "metrics": aggregate(results),
        "examples": results,
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Readable report for one evaluation record."""
    m = result["metrics"]

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{100 * v:.1f}%"

    def na(v: Any, suffix: str = "") -> str:
        return "n/a" if v is None else f"{v}{suffix}"

    meta = result.get("adapter") or {}
    adapter = f"`{meta.get('path')}` (run {meta.get('run_id')})" if meta else "none (base model)"
    top = m["most_common_answer"]
    cols = [*CATEGORIES, NO_CATEGORY]
    lines = [
        f"# Evaluation: {result['label']}",
        "",
        f"- Model: `{(result.get('model') or {}).get('repo_id')}` @ "
        f"`{((result.get('model') or {}).get('revision') or 'main')[:12]}`",
        f"- Adapter: {adapter}",
        f"- Test: {result.get('test')} — `{result.get('test_file')}` ({m['examples']} lines)",
        f"- Evaluated: {result['evaluated_at']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {m['correct']} / {m['examples']} ({pct(m['accuracy'])}) |",
        f"| Macro-F1 | {na(m['macro_f1'])} |",
        f"| Most common answer | {top['category']} ({top['count']} times) |",
        f"| Answers naming no category | {m['no_category_answers']} |",
        f"| Generation failures | {m['generation_failures']} |",
        f"| Median / mean time per line | {na(m['median_latency_ms'], ' ms')} / "
        f"{na(m['mean_latency_ms'], ' ms')} |",
        "",
        "| Category | Support | Recall | Precision | F1 |",
        "|---|---|---|---|---|",
        *(
            f"| {c} | {v['support']} | {pct(v['recall'])} | {pct(v['precision'])} | {na(v['f1'])} |"
            for c, v in m["per_category"].items()
        ),
        "",
        "Confusion matrix (rows = true category, columns = answer):",
        "",
        "| | " + " | ".join(cols) + " |",
        "|---" * (len(cols) + 1) + "|",
        *(
            f"| {c} | " + " | ".join(str(row[k]) for k in cols) + " |"
            for c, row in m["confusion"].items()
        ),
    ]
    if result.get("budget"):
        from bank_budget.budget import render_budget

        lines += ["", render_budget(result["budget"])]
    return "\n".join(lines) + "\n"


def save_result(result: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write ``<label>/<timestamp>.json|.md`` and refresh ``<label>/latest.json``."""
    folder = output_dir / result["label"]
    stamp = result["evaluated_at"].replace(":", "").replace("-", "").replace("+0000", "Z")
    json_path = folder / f"{stamp}.json"
    md_path = folder / f"{stamp}.md"
    write_json(json_path, result)
    md_path.write_text(render_markdown(result), encoding="utf-8")
    write_json(folder / LATEST, result)
    (folder / "latest.md").write_text(render_markdown(result), encoding="utf-8")
    return json_path, md_path


def load_latest(output_dir: Path, label: str) -> dict[str, Any] | None:
    """Most recent evaluation record for ``label``, or None if never evaluated."""
    return read_json(output_dir / label / LATEST)

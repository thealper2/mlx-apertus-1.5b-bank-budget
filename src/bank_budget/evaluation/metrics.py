"""Classification metrics for categorised statement lines."""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from typing import Any

from bank_budget.task import CATEGORIES, NO_CATEGORY


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy, the most common answer, per-category metrics, confusion, latency."""
    n = len(results)
    correct = sum(r["correct"] for r in results)
    columns = [*CATEGORIES, NO_CATEGORY]
    confusion = {c: dict.fromkeys(columns, 0) for c in CATEGORIES}
    for r in results:
        confusion[r["category"]][r["prediction"]] += 1
    per_category: dict[str, dict[str, Any]] = {}
    f1s: list[float] = []
    for c in CATEGORIES:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        predicted = sum(confusion[g][c] for g in CATEGORIES)
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else (0.0 if support else None)
        )
        if f1 is not None:
            f1s.append(f1)
        per_category[c] = {
            "support": support,
            "recall": None if recall is None else round(recall, 4),
            "precision": None if precision is None else round(precision, 4),
            "f1": None if f1 is None else round(f1, 4),
        }
    answers = Counter(r["prediction"] for r in results)
    top = answers.most_common(1)[0] if answers else (None, 0)
    latencies = [r["latency_ms"] for r in results if r.get("latency_ms") is not None]
    return {
        "examples": n,
        "correct": correct,
        "accuracy": _rate(correct, n),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "most_common_answer": {"category": top[0], "count": top[1]},
        "answers": dict(answers),
        "no_category_answers": answers.get(NO_CATEGORY, 0),
        "generation_failures": sum(bool(r.get("generation_error")) for r in results),
        "per_category": per_category,
        "confusion": confusion,
        "median_latency_ms": round(statistics.median(latencies), 1) if latencies else None,
        "mean_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }

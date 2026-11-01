"""A month of statement lines → a budget: money per category, as filed by a model.

The statement is a CSV with ``date, description, amount, category`` (amount negative for
money out, ``category`` = how a person would file the line). The budget sums the amounts
per category under the model's answers and under the true categories.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from bank_budget.task import CATEGORIES

STATEMENT_COLUMNS = ("date", "description", "amount", "category")


def read_statement(path: Path) -> list[dict[str, Any]]:
    """Statement rows ``{date, line, amount, category}`` in file order."""
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != STATEMENT_COLUMNS:
            raise ValueError(f"{path.name}: columns must be {', '.join(STATEMENT_COLUMNS)}")
        rows = []
        for i, r in enumerate(reader, start=2):
            if r["category"] not in CATEGORIES:
                raise ValueError(f"{path.name} line {i}: unknown category {r['category']!r}")
            if not r["description"].startswith(("[debit]", "[credit]")):
                raise ValueError(f"{path.name} line {i}: must start with [debit] or [credit]")
            rows.append(
                {
                    "date": r["date"],
                    "line": r["description"],
                    "amount": float(r["amount"]),
                    "category": r["category"],
                }
            )
    return rows


def budget_table(rows: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """Sums per category under the predictions and under the true categories."""
    filed: dict[str, float] = defaultdict(float)
    true: dict[str, float] = defaultdict(float)
    for row, pred in zip(rows, predictions, strict=True):
        filed[pred] += row["amount"]
        true[row["category"]] += row["amount"]
    categories = sorted(set(filed) | set(true), key=lambda c: (true.get(c, 0.0), c))
    return {
        "lines": len(rows),
        "lines_right": sum(p == r["category"] for r, p in zip(rows, predictions, strict=True)),
        "categories": [
            {
                "category": c,
                "filed": round(filed.get(c, 0.0), 2),
                "true": round(true.get(c, 0.0), 2),
            }
            for c in categories
        ],
        "money_in": round(sum(r["amount"] for r in rows if r["amount"] > 0), 2),
        "money_out": round(sum(r["amount"] for r in rows if r["amount"] < 0), 2),
    }


def render_budget(table: dict[str, Any]) -> str:
    """Markdown table of a budget."""
    lines = [
        f"Budget: {table['lines_right']} / {table['lines']} lines filed right.",
        "",
        "| Category | As filed by the model | True |",
        "|---|---:|---:|",
        *(f"| {c['category']} | {c['filed']:.2f} | {c['true']:.2f} |" for c in table["categories"]),
        f"| **Money in** | {table['money_in']:.2f} | {table['money_in']:.2f} |",
        f"| **Money out** | {table['money_out']:.2f} | {table['money_out']:.2f} |",
    ]
    return "\n".join(lines)

from __future__ import annotations

from bank_budget.budget import budget_table
from bank_budget.evaluation.evaluator import (
    build_result,
    evaluate_lines,
    render_markdown,
    save_result,
)
from bank_budget.evaluation.metrics import aggregate
from bank_budget.inference.generate import Generation
from bank_budget.task import parse_category


def _rows() -> list[dict]:
    cats = ["Rent", "Rent", "Fees", "Income"]
    return [{"line": f"[debit] L{i}", "category": c, "shop": f"w:{i}"} for i, c in enumerate(cats)]


def test_aggregate_counts_and_most_common_answer():
    results = [
        {"category": "Rent", "prediction": "Rent", "correct": True, "latency_ms": 10.0},
        {"category": "Rent", "prediction": "Utilities", "correct": False, "latency_ms": 30.0},
        {"category": "Fees", "prediction": "Utilities", "correct": False, "latency_ms": 20.0},
        {"category": "Income", "prediction": "other", "correct": False, "latency_ms": None},
    ]
    m = aggregate(results)
    assert (m["correct"], m["accuracy"], m["no_category_answers"]) == (1, 0.25, 1)
    assert m["most_common_answer"] == {"category": "Utilities", "count": 2}
    assert m["confusion"]["Income"]["other"] == 1
    assert m["per_category"]["Rent"] == {
        "support": 2,
        "recall": 0.5,
        "precision": 1.0,
        "f1": 0.6667,
    }
    assert m["per_category"]["Travel"]["support"] == 0 and m["per_category"]["Travel"]["f1"] is None
    assert (m["median_latency_ms"], m["mean_latency_ms"]) == (20.0, 20.0)


def test_evaluate_lines_scores_and_survives_failures():
    rows = _rows()
    gold = {r["line"]: r["category"] for r in rows}

    def oracle(line: str) -> Generation:
        return Generation(gold[line].lower(), parse_category(gold[line].lower()), 0.01)

    good = evaluate_lines(rows, oracle)
    assert all(r["correct"] for r in good) and good[0]["shop"] == "w:0"

    def broken(line: str) -> Generation:
        raise RuntimeError("boom")

    bad = aggregate(evaluate_lines(rows, broken))
    assert bad["generation_failures"] == 4 and bad["no_category_answers"] == 4


def test_reports_include_the_budget(tmp_path):
    rows = [{"line": "[debit] A", "category": "Rent", "amount": -100.0, "date": "d"}]
    results = evaluate_lines(rows, lambda line: Generation("Rent", "Rent", 0.0))
    result = build_result("adapter-month", results, {"budget": budget_table(rows, ["Rent"])})
    json_path, md_path = save_result(result, tmp_path)
    text = md_path.read_text()
    assert json_path.is_file() and (tmp_path / "adapter-month" / "latest.json").is_file()
    assert "| Accuracy | 1 / 1 (100.0%) |" in text and "| Rent | -100.00 | -100.00 |" in text
    assert "Confusion matrix" in render_markdown(result)

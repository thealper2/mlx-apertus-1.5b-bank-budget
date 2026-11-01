"""Bounded end-to-end pipeline checks (``make smoke-test``).

Never trains and never uploads: the training entry point is replaced by a function that
raises, and the publication check stages a synthetic adapter without network access.
The only network use is downloading the ~3.5 MB dataset if it is not cached. Pass
``--with-model`` to also load the base model and file one line of the example month.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
import tempfile
import traceback
from collections.abc import Callable
from pathlib import Path

from bank_budget.budget import budget_table, read_statement
from bank_budget.cli import validate_prepared
from bank_budget.config import load_project_config
from bank_budget.data.download import load_lines
from bank_budget.data.prepare import prepare_dataset, prepare_smoke_subset
from bank_budget.data.validate import read_jsonl
from bank_budget.evaluation.evaluator import evaluate_lines
from bank_budget.evaluation.metrics import aggregate
from bank_budget.inference.generate import Generation
from bank_budget.logging_config import setup_logging
from bank_budget.publishing import huggingface
from bank_budget.publishing.model_card import (
    REQUIRED_SECTIONS,
    collect_inputs,
    parse_front_matter,
    write_card,
)
from bank_budget.task import NO_CATEGORY, parse_category
from bank_budget.training import trainer
from bank_budget.training.artifacts import write_json

Result = tuple[str, str]  # (status, note)


def _forbid_training(*_: object, **__: object) -> None:
    raise AssertionError("smoke test attempted to start training")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-model", action="store_true", help="also load the base model (~3.2 GB RAM)"
    )
    args = parser.parse_args()
    setup_logging()
    trainer.execute_training = _forbid_training  # type: ignore[assignment]
    project = load_project_config()
    root = project.root
    results: list[tuple[str, str, str]] = []

    def check(name: str, fn: Callable[[], Result]) -> None:
        try:
            status, note = fn()
        except Exception as exc:
            traceback.print_exc()
            status, note = "FAIL", f"{type(exc).__name__}: {exc}"
        results.append((name, status, note))
        print(f"[{status}] {name}: {note}", flush=True)

    def data() -> Result:
        d = project.data
        if not (d.output_dir / "manifest.json").is_file():
            prepare_dataset(load_lines(d.dataset), d, root)
            note = "prepared canonical data"
        else:
            note = "canonical prepared data present"
        prepare_smoke_subset(d, root)
        problems = validate_prepared(project, smoke=False)
        problems += validate_prepared(project, smoke=True)
        if problems:
            return "FAIL", "; ".join(problems[:3])
        return "PASS", note + "; files valid, no shared lines, test_new shops never in training"

    def dry_runs() -> Result:
        notes = []
        for name in sorted(project.training.profiles):
            plan = trainer.plan_training(project, name)
            data_problems = [p for p in plan.problems if "already holds" not in p]
            if data_problems:
                return "FAIL", f"{name}: {data_problems[0]}"
            notes.append(f"{name}={plan.profile.optimizer_updates} updates")
        return "PASS", ", ".join(notes) + " (planned only)"

    def evaluation() -> Result:
        rows = read_jsonl(project.evaluation.smoke_test_files["test_new"])
        labels = {r["line"]: r["category"] for r in rows}

        def oracle(line: str) -> Generation:
            answer = f"{labels[line].upper()}."
            return Generation(answer, parse_category(answer), 0.0)

        def broken(line: str) -> Generation:
            return Generation("I can help with that!", NO_CATEGORY, 0.0)

        good = aggregate(evaluate_lines(rows, oracle))
        bad = aggregate(evaluate_lines(rows, broken))
        if (
            good["correct"] != len(rows)
            or bad["correct"]
            or bad["no_category_answers"] != len(rows)
        ):
            return "FAIL", f"oracle={good['correct']} broken={bad['correct']}"
        month = read_statement(project.evaluation.statement_file)
        table = budget_table(month, [r["category"] for r in month])
        if table["lines_right"] != len(month) or any(
            c["filed"] != c["true"] for c in table["categories"]
        ):
            return "FAIL", "budget of the true categories does not add up"
        return "PASS", (
            f"oracle {len(rows)}/{len(rows)}, unparseable 0; month of {len(month)} lines adds "
            f"up (in {table['money_in']:.2f}, out {table['money_out']:.2f}); no model"
        )

    def card() -> Result:
        inputs = collect_inputs(project)
        out = write_card(inputs, project.publishing.model_card_path)
        text = out.read_text()
        meta = parse_front_matter(text)
        missing = [s for s in REQUIRED_SECTIONS if s not in text]
        if missing or meta["base_model"] != project.training.model.repo_id:
            return "FAIL", f"missing sections {missing}"
        return "PASS", f"{len(text):,} chars, metadata valid, status line present"

    def publish() -> Result:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = Path(tmp) / "adapter"
            adapter.mkdir()
            (adapter / "adapters.safetensors").write_bytes(b"\0" * 16)
            write_json(
                adapter / "adapter_config.json",
                {"model": str(Path.home() / "cache" / "snapshot"), "num_layers": -1},
            )
            write_json(
                adapter / "run_manifest.json",
                {"run_id": "synthetic", "status": "completed", "profile": {"kind": "full"}},
            )
            staging = root / "artifacts" / "exports" / "smoke_test_upload"
            pub = dataclasses.replace(project.publishing, staging_dir=staging)
            proj = dataclasses.replace(project, publishing=pub)
            plan = huggingface.plan_upload(proj, "smoke-test/not-a-real-repo", adapter)
            if not plan.ok:
                return "FAIL", "; ".join(plan.problems)
            try:
                url = huggingface.publish(plan, proj, execute=False)
                staged = json.loads((staging / "adapter_config.json").read_text())
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        if url is not None or staged["model"] != project.training.model.repo_id:
            return "FAIL", "dry run uploaded or did not sanitize local paths"
        return "PASS", f"{len(plan.files)} files staged, local paths removed, nothing uploaded"

    def model() -> Result:
        if not args.with_model:
            return "SKIP", "pass --with-model (or run `make inference-smoke`)"
        from bank_budget.inference.generate import categorize, load_model, peak_memory_gb

        m = project.training.model
        mdl, tok = load_model(m.repo_id, m.revision)
        line = read_statement(project.evaluation.statement_file)[2]["line"]
        gen = categorize(mdl, tok, line, project.evaluation.max_tokens)
        return "PASS", (
            f"{line!r}: answered {gen.raw!r} -> {gen.category} in {gen.latency_s}s, "
            f"peak {peak_memory_gb()} GB"
        )

    check("data pipeline", data)
    check("training dry runs", dry_runs)
    check("evaluation pipeline", evaluation)
    check("model card", card)
    check("publish dry run", publish)
    check("model load + categorize", model)

    print("\n| Check | Status | Notes |\n|---|---|---|")
    for name, status, note in results:
        print(f"| {name} | {status} | {note} |")
    return 1 if any(status == "FAIL" for _, status, _ in results) else 0


if __name__ == "__main__":
    sys.exit(main())

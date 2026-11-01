from __future__ import annotations

import re

import pytest

from bank_budget.budget import budget_table, read_statement
from bank_budget.data.prepare import sha256_file
from bank_budget.data.validate import read_jsonl
from bank_budget.evaluation.evaluator import build_result, evaluate_lines, save_result
from bank_budget.inference.generate import Generation
from bank_budget.publishing.model_card import (
    PENDING,
    REQUIRED_SECTIONS,
    collect_inputs,
    parse_front_matter,
    render_card,
    training_status,
    write_card,
)
from bank_budget.task import SYSTEM_PROMPT
from bank_budget.training.artifacts import RUN_MANIFEST, write_json

RUN_ID = "m1_air_8gb-20260101-000000"


def _manifest(kind: str = "full") -> dict:
    return {
        "run_id": RUN_ID,
        "status": "completed",
        "profile": {
            "name": "m1_air_8gb",
            "kind": kind,
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "bias_correction": False,
            "learning_rate": 1e-4,
            "warmup_steps": 20,
            "decay_steps": 1000,
            "end_learning_rate": 1e-6,
            "iters": 1000,
            "batch_size": 16,
            "grad_accumulation_steps": 1,
            "max_seq_length": 512,
            "num_layers": 16,
            "lora": {"rank": 8, "scale": 20.0, "dropout": 0.0, "keys": ["self_attn.q_proj"]},
            "mask_prompt": True,
            "grad_checkpoint": True,
            "memory_limit_gb": 5.0,
            "cache_limit_gb": 0.5,
            "seed": 0,
        },
        "metrics": {
            "final_train_loss": 0.0123,
            "final_val_loss": 0.0456,
            "trainable_params_m": 5.374,
            "total_params_m": 1510.019,
            "wall_time_s": 9999.0,
            "peak_memory_gb": 5.8,
        },
        "environment": {
            "chip": "Apple M1",
            "memory_gb": 8,
            "os_version": "27.0",
            "python": "3.12.15",
            "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"},
        },
    }


def _record(project, model: str, test: str, run_id: str | None, correct: int) -> None:
    if test == "month":
        path = project.evaluation.statement_file
        rows = read_statement(path)
    else:
        path = project.evaluation.test_files[test]
        rows = read_jsonl(path)
    answers = iter(r["category"] if i < correct else "other" for i, r in enumerate(rows))

    def fake(line: str) -> Generation:
        a = next(answers)
        return Generation(a, a, 0.05)

    results = evaluate_lines(rows, fake)
    meta = {
        "model": {
            "repo_id": project.training.model.repo_id,
            "revision": project.training.model.revision,
        },
        "adapter": {"run_id": run_id} if run_id else None,
        "test_file_sha256": sha256_file(path),
    }
    if test == "month":
        meta["budget"] = budget_table(rows, [r["prediction"] for r in results])
    save_result(build_result(f"{model}-{test}", results, meta), project.evaluation.output_dir)


def test_untrained_card_has_all_sections_and_no_numbers(project):
    card = render_card(collect_inputs(project))
    for section in REQUIRED_SECTIONS:
        assert section in card
    assert "**Training status: not trained.**" in card
    rows = [ln for ln in card.splitlines() if PENDING in ln]
    assert len(rows) == 3 and not any(re.search(r"\d+\.\d%", ln) for ln in rows)
    assert "| Final train loss | not measured |" in card
    assert "| Hardware / OS | not recorded (no training run yet) |" in card


def test_metadata_is_valid_and_derived_from_config(project):
    meta = parse_front_matter(render_card(collect_inputs(project)))
    assert meta["base_model"] == "swiss-ai/Apertus-v1.1-1.5B-Instruct"
    assert meta["datasets"] == ["DoDataThings/us-bank-transaction-categories-v2"]
    assert meta["library_name"] == "mlx" and meta["license"] == "apache-2.0"


def test_card_states_prompt_policy_and_adapter_only(project):
    card = render_card(collect_inputs(project))
    assert SYSTEM_PROMPT in card and "Acceptable Use Policy" in card
    assert "contains only a LoRA adapter" in card and "synthetic" in card


def test_usage_example_is_valid_python(project):
    card = render_card(collect_inputs(project, repo_id="me/bank-budget"))
    usage = card.split("## How to use")[1].split("## Limitations")[0]
    code = re.findall(r"```python\n(.*?)```", usage, re.S)[0]
    compile(code, "usage", "exec")
    assert 'snapshot_download("me/bank-budget")' in code
    assert f'revision="{project.training.model.revision}"' in code
    assert SYSTEM_PROMPT in code and "transformers" not in code


def test_trained_card_reports_recorded_values_only(prepared_project):
    p = prepared_project
    write_json(p.publishing.adapter_path / RUN_MANIFEST, _manifest())
    _record(p, "base", "test_new", None, correct=3)
    _record(p, "adapter", "test_new", RUN_ID, correct=30)
    _record(p, "adapter", "month", RUN_ID, correct=31)
    card = render_card(collect_inputs(p))
    assert "**Training status: completed.**" in card
    assert "| Final validation loss | 0.0456 |" in card
    assert "| Peak memory footprint | 5.8 GB |" in card
    assert "3 / 34 (8.8%)" in card and "30 / 34 (88.2%)" in card and "31 / 31 (100.0%)" in card
    assert "other (31×)" in card  # the base model's most common answer
    assert "Budget: 31 / 31 lines filed right." in card


def test_evaluations_of_other_runs_or_files_are_ignored(prepared_project):
    p = prepared_project
    write_json(p.publishing.adapter_path / RUN_MANIFEST, _manifest())
    _record(p, "adapter", "test_new", "other-run", correct=34)
    assert collect_inputs(p).adapter_evals["test_new"] is None
    (p.evaluation.statement_file).write_text(
        p.evaluation.statement_file.read_text().replace("-8.40", "-8.41")
    )
    _record(p, "adapter", "month", RUN_ID, correct=31)
    assert collect_inputs(p).adapter_evals["month"] is not None
    p.evaluation.statement_file.write_text(
        p.evaluation.statement_file.read_text().replace("-8.41", "-8.40")
    )
    assert collect_inputs(p).adapter_evals["month"] is None  # the month file changed


def test_training_status_values():
    assert training_status(None) == "not trained"
    assert training_status(_manifest("smoke")) == "smoke test only"
    assert training_status({**_manifest(), "status": "interrupted"}) == "interrupted"


def test_write_card_never_overwrites_project_readme(project):
    with pytest.raises(ValueError, match="Refusing"):
        write_card(collect_inputs(project), project.root / "README.md")

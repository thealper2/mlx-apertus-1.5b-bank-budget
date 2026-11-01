from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from bank_budget.budget import budget_table, read_statement, render_budget
from bank_budget.cli import validate_prepared
from bank_budget.config import SplitConfig
from bank_budget.data.prepare import prepare_dataset
from bank_budget.data.split import deduplicate, held_out_groups, make_splits
from bank_budget.data.validate import chat_line_problems, read_jsonl, row_problems
from bank_budget.task import CATEGORIES, SYSTEM_PROMPT
from conftest import SPLIT, fake_lines

CFG = SplitConfig(**SPLIT)


def test_deduplicate_keeps_one_copy_and_drops_ambiguous_texts():
    frame = deduplicate(fake_lines())
    assert len(frame) == 340
    assert frame.description.is_unique
    assert "[debit] TWOFACED THING" not in set(frame.description)


def test_held_out_shops_are_whole_and_per_category():
    parts = make_splits(fake_lines(), CFG)
    held = parts["held_out_groups"]
    assert len(held) == len(CATEGORIES)  # one of 4 shops (25%) per category
    frame = parts["unique"]
    assert held == held_out_groups(frame, 0.25, 0.5, 0)  # deterministic
    assert held != held_out_groups(frame, 0.25, 0.5, 1)


def test_splits_keep_test_new_shops_out_of_training():
    parts = make_splits(fake_lines(), CFG)
    train_shops = set(parts["train"].group) | set(parts["valid"].group)
    assert not set(parts["test_new"].group) & train_shops
    assert set(parts["test_known"].group) <= set(parts["train"].group)
    for name, n in (("test_new", 2), ("test_known", 2), ("valid", 1)):
        assert parts[name].category.value_counts().to_dict() == dict.fromkeys(CATEGORIES, n)
    all_lines = [set(parts[n].description) for n in ("train", "valid", "test_new", "test_known")]
    assert sum(map(len, all_lines)) == len(set().union(*all_lines))  # no line in two splits
    again = make_splits(fake_lines(), CFG)
    assert list(again["train"].description) == list(parts["train"].description)


def test_prepare_writes_mlx_and_test_files(project):
    manifest = prepare_dataset(fake_lines(), project.data, project.root)
    assert manifest["rows"] == {"train": 204, "valid": 17, "test_new": 34, "test_known": 34}
    assert manifest["lines"]["held_out_shop_groups"] == 17
    assert manifest["overlap"]["train/test_new"] == {"lines": 0, "shops": 0}
    chat = read_jsonl(project.data.mlx_dir / "train.jsonl")[0]
    assert chat_line_problems(chat) == [] and chat["messages"][0]["content"] == SYSTEM_PROMPT
    rows = read_jsonl(project.data.output_dir / "train.jsonl")
    assert [r["line"] for r in rows] == [
        c["messages"][1]["content"] for c in read_jsonl(project.data.mlx_dir / "train.jsonl")
    ]
    assert set(json.loads((project.data.output_dir / "manifest.json").read_text())["files"]) == {
        "train",
        "valid",
        "test_new",
        "test_known",
        "mlx_train",
        "mlx_valid",
    }


def test_prepare_refuses_unexpected_counts(project):
    data = dataclasses.replace(project.data, expected_unique_lines=999)
    with pytest.raises(ValueError, match="expected 999"):
        prepare_dataset(fake_lines(), data, project.root)
    with pytest.raises(ValueError, match="lines, expected"):
        prepare_dataset(fake_lines().head(10), project.data, project.root)


def test_dry_run_writes_nothing(project):
    prepare_dataset(fake_lines(), project.data, project.root, dry_run=True)
    assert not project.data.output_dir.exists()


def test_smoke_subset(prepared_project):
    d = prepared_project.data
    assert len(read_jsonl(d.smoke_mlx_dir / "train.jsonl")) == 32
    test = read_jsonl(d.smoke_dir / "test_new.jsonl")
    assert len(test) == 17 and len({r["category"] for r in test}) == 17  # 1 per category


def test_prepared_data_validates(prepared_project):
    assert validate_prepared(prepared_project, smoke=False) == []
    assert validate_prepared(prepared_project, smoke=True) == []


def test_validation_catches_shared_lines_and_shops(prepared_project):
    out = prepared_project.data.output_dir
    test_new = read_jsonl(out / "test_new.jsonl")
    train = read_jsonl(out / "train.jsonl")
    leaked = [{**train[0], "shop": test_new[0]["shop"]}, *train[1:]]
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in leaked))
    assert any("shop groups" in p for p in validate_prepared(prepared_project, smoke=False))
    copied = [{**train[0], "line": test_new[0]["line"]}, *train[1:]]
    (out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in copied))
    assert any("identical lines" in p for p in validate_prepared(prepared_project, smoke=False))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"line": "SQ *SHOP"}, "[debit] or [credit]"),
        ({"category": "Food"}, "17 categories"),
        ({"shop": ""}, "shop"),
    ],
)
def test_row_problems(change, message):
    row = {"line": "[debit] X", "category": "Fees", "shop": "w:x"}
    assert row_problems(row) == []
    assert any(message in p for p in row_problems({**row, **change}))


def test_chat_line_problems_require_the_system_prompt():
    rec = {
        "messages": [
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "[debit] X"},
            {"role": "assistant", "content": "Fees"},
        ]
    }
    assert any("system prompt" in p for p in chat_line_problems(rec))


def test_statement_and_budget(tmp_path: Path):
    rows = read_statement(Path(__file__).resolve().parents[1] / "examples" / "statement.csv")
    assert len(rows) == 31 and all(r["category"] in CATEGORIES for r in rows)
    preds = [r["category"] for r in rows]
    preds[1] = "Rent"  # the mortgage payment filed as rent
    table = budget_table(rows, preds)
    by = {c["category"]: c for c in table["categories"]}
    assert table["lines_right"] == 30
    assert by["Rent"] == {"category": "Rent", "filed": -1612.45, "true": 0.0}
    assert by["Mortgage"]["filed"] == 0.0 and by["Mortgage"]["true"] == -1612.45
    assert table["money_in"] == 7900.0
    assert "| Rent | -1612.45 | 0.00 |" in render_budget(table)
    bad = tmp_path / "bad.csv"
    bad.write_text("date,description,amount,category\n2026-10-01,[debit] X,-1.0,Food\n")
    with pytest.raises(ValueError, match="unknown category"):
        read_statement(bad)

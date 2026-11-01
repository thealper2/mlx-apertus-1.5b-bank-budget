"""Shared fixtures: a throw-away project root with a small synthetic dataset.

Nothing here touches the network, the dataset, a model or training.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from bank_budget.config import ProjectConfig, SplitConfig, load_project_config
from bank_budget.task import CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[1]
SHOPS_PER_CATEGORY = 4
LINES_PER_SHOP = 5
SPLIT = {
    "holdout_share": 0.25,
    "holdout_max_group_share": 0.5,
    "holdout_seed": 0,
    "test_per_category": 2,
    "test_new_seed": 1,
    "test_known_seed": 2,
    "valid_per_category": 1,
    "valid_seed": 3,
    "train_shuffle_seed": 4,
}


def _name(i: int) -> str:
    """A unique letters-only shop name (unique first word and first 6 letters)."""
    return "".join(chr(97 + (i // 26**k) % 26) for k in range(3)).upper() + "VILLE"


def fake_lines() -> Any:
    """Dataset lines: 4 shops x 5 lines per category, one exact copy, one ambiguous text."""
    import pandas as pd

    rows = []
    for c, category in enumerate(CATEGORIES):
        sign = "[credit]" if category == "Income" else "[debit]"
        for s in range(SHOPS_PER_CATEGORY):
            name = _name(c * SHOPS_PER_CATEGORY + s)
            for n in range(LINES_PER_SHOP):
                rows.append(
                    {"description": f"{sign} {name} STORE #{100 + n}", "category": category}
                )
    rows.append(dict(rows[0]))  # an exact copy: kept once
    rows.append({"description": "[debit] TWOFACED THING", "category": "Fees"})
    rows.append({"description": "[debit] TWOFACED THING", "category": "Rent"})  # dropped
    return pd.DataFrame(rows)


def split_counts() -> tuple[int, dict[str, int]]:
    """(distinct lines, rows per split) of the fake dataset under the test split config."""
    from bank_budget.data.split import make_splits

    parts = make_splits(fake_lines(), SplitConfig(**SPLIT))
    return len(parts["unique"]), {
        n: len(parts[n]) for n in ("train", "valid", "test_new", "test_known")
    }


def make_project(root: Path) -> ProjectConfig:
    """Copy the real configs into ``root`` with small sizes, then load them."""
    cfg_dir = root / "configs"
    shutil.copytree(REPO_ROOT / "configs", cfg_dir)
    (root / "pyproject.toml").write_text("[project]\nname = 'tmp'\n")
    (root / "examples").mkdir()
    shutil.copy(REPO_ROOT / "examples" / "statement.csv", root / "examples" / "statement.csv")
    data = yaml.safe_load((cfg_dir / "data.yaml").read_text())
    unique, rows = split_counts()
    data["dataset"]["expected_lines"] = len(fake_lines())
    data["split"] = dict(SPLIT)
    data["expected_unique_lines"] = unique
    data["expected_rows"] = rows
    data["smoke"].update(train_size=32, valid_size=16, test_per_category=1)
    (cfg_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    return load_project_config(root=root)


@pytest.fixture(autouse=True)
def _reset_logging():
    """CLI tests attach log handlers to pytest's captured streams; drop them afterwards."""
    import logging

    yield
    for handler in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(handler)


@pytest.fixture
def project(tmp_path: Path) -> ProjectConfig:
    return make_project(tmp_path)


@pytest.fixture
def prepared_project(tmp_path: Path) -> ProjectConfig:
    """Project with canonical + smoke data prepared from the fake dataset."""
    from bank_budget.data.prepare import prepare_dataset, prepare_smoke_subset

    proj = make_project(tmp_path)
    prepare_dataset(fake_lines(), proj.data, proj.root)
    prepare_smoke_subset(proj.data, proj.root)
    return proj


@pytest.fixture
def repo_root_env(monkeypatch: pytest.MonkeyPatch):
    """Point the CLI at a temporary project root."""

    def _set(root: Path) -> None:
        monkeypatch.setenv("BANK_BUDGET_ROOT", str(root))

    return _set

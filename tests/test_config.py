from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bank_budget.config import (
    ConfigError,
    load_project_config,
    parse_data_config,
    parse_evaluation_config,
    parse_training_config,
    read_yaml,
)
from conftest import REPO_ROOT


def test_real_configs_keep_experiment_identity():
    cfg = load_project_config(root=REPO_ROOT)
    assert cfg.training.model.repo_id == "swiss-ai/Apertus-v1.1-1.5B-Instruct"
    assert cfg.training.model.revision == "9e9d01154446a645d30f04174cf1515a38058be7"
    d = cfg.data
    assert d.dataset.repo_id == "DoDataThings/us-bank-transaction-categories-v2"
    assert d.dataset.revision == "3e8d052cb7b362fb1a04b6064d2331982553ea56"
    assert (d.dataset.file, d.dataset.expected_lines, d.expected_unique_lines) == (
        "transactions-synthetic.csv",
        68000,
        45692,
    )
    assert d.expected_rows == {"train": 35639, "valid": 850, "test_new": 1700, "test_known": 1700}
    s = d.split
    assert (s.holdout_share, s.holdout_max_group_share, s.holdout_seed) == (0.15, 0.30, 0)
    assert (s.test_per_category, s.test_new_seed, s.test_known_seed) == (100, 1, 2)
    assert (s.valid_per_category, s.valid_seed, s.train_shuffle_seed) == (50, 3, 4)
    ev = cfg.evaluation
    assert (ev.max_tokens, ev.warmup_generations) == (8, 3)


def test_reference_profile_is_the_recipe():
    p = load_project_config(root=REPO_ROOT).training.profile("reference")
    assert (p.iters, p.batch_size, p.grad_accumulation_steps, p.seed) == (1000, 16, 1, 0)
    assert (p.optimizer, p.weight_decay, p.adam_beta1, p.adam_beta2, p.adam_eps) == (
        "adamw",
        0.01,
        0.9,
        0.999,
        1e-8,
    )
    assert p.bias_correction is False  # MLX's own AdamW default, written out
    assert (p.learning_rate, p.lr_schedule, p.warmup_steps) == (1e-4, "cosine", 20)
    assert (p.decay_steps, p.end_learning_rate) == (1000, 1e-6)
    assert (p.lora.rank, p.lora.scale, p.lora.dropout, p.num_layers) == (8, 20.0, 0.0, 16)
    assert p.lora.keys == [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    assert (p.max_seq_length, p.mask_prompt, p.grad_checkpoint) == (512, True, False)
    assert (p.steps_per_report, p.steps_per_eval, p.val_batches, p.save_every) == (
        10,
        500,
        -1,
        1000,
    )


def test_m1_profile_only_changes_memory_settings():
    t = load_project_config(root=REPO_ROOT).training
    ref, m1 = t.profile("reference").to_dict(), t.profile("m1_air_8gb").to_dict()
    differ = {k for k in ref if ref[k] != m1[k]}
    assert differ == {
        "name",
        "adapter_path",
        "grad_checkpoint",
        "memory_limit_gb",
        "cache_limit_gb",
        "save_every",
    }
    assert m1["grad_checkpoint"] is True
    assert (m1["memory_limit_gb"], m1["cache_limit_gb"]) == (5.0, 0.5)


def _training_raw() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs/training.yaml").read_text())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("lora_alpha", 16, "unknown key"),
        ("batch_size", "16", "batch_size must be an integer"),
        ("lr_schedule", "linear", "lr_schedule must be one of"),
        ("warmup_steps", 1000, "warmup_steps must be smaller"),
        ("end_learning_rate", 1.0, "end_learning_rate must be in"),
        ("adam_beta2", 1.5, "adam_beta2 must be in"),
        ("bias_correction", "no", "true or false"),
        ("num_layers", 0, "num_layers must be > 0 or -1"),
    ],
)
def test_invalid_profile_values(key, value, message):
    raw = _training_raw()
    raw["profiles"]["reference"][key] = value
    with pytest.raises(ConfigError, match=message):
        parse_training_config(raw)


def test_lora_needs_keys_and_smoke_stays_small():
    raw = _training_raw()
    del raw["profiles"]["reference"]["lora"]["keys"]
    with pytest.raises(ConfigError, match="missing required key 'keys'"):
        parse_training_config(raw)
    raw = _training_raw()
    raw["profiles"]["smoke"]["iters"] = 100
    with pytest.raises(ConfigError, match="smoke profile has iters=100"):
        parse_training_config(raw)


def test_data_config_errors(tmp_path: Path):
    raw = yaml.safe_load((REPO_ROOT / "configs/data.yaml").read_text())
    with pytest.raises(ConfigError, match="must map exactly"):
        parse_data_config({**raw, "expected_rows": {"train": 1}}, tmp_path)
    with pytest.raises(ConfigError, match=r"relative \.csv path"):
        parse_data_config({**raw, "dataset": {**raw["dataset"], "file": "/x.csv"}}, tmp_path)
    with pytest.raises(ConfigError, match="inconsistent"):
        parse_data_config({**raw, "expected_unique_lines": 100}, tmp_path)
    with pytest.raises(ConfigError, match=r"holdout_share must be in"):
        parse_data_config({**raw, "split": {**raw["split"], "holdout_share": 1.5}}, tmp_path)
    with pytest.raises(ConfigError, match=r"smoke\.dir must differ"):
        parse_data_config({**raw, "smoke": {**raw["smoke"], "dir": "data/processed"}}, tmp_path)


def test_evaluation_config_errors(tmp_path: Path):
    raw = yaml.safe_load((REPO_ROOT / "configs/evaluation.yaml").read_text())
    with pytest.raises(ConfigError, match="must map exactly"):
        parse_evaluation_config({**raw, "test_files": {"test_new": "x"}}, tmp_path)
    with pytest.raises(ConfigError, match="unknown key"):
        parse_evaluation_config({**raw, "generation": {"max_tokens": 8, "temp": 0}}, tmp_path)
    with pytest.raises(ConfigError, match="must stay small"):
        parse_evaluation_config(
            {**raw, "limits": {**raw["limits"], "smoke_max_examples": 99}}, tmp_path
        )


def test_missing_and_invalid_files(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        read_yaml(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [1, 2")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        read_yaml(bad)


def test_publishing_defaults_are_safe(project):
    assert project.publishing.repo_id is None and project.publishing.private
    assert project.publishing.adapter_license == "apache-2.0"
    assert project.publishing.adapter_path.name == "m1_air_8gb"

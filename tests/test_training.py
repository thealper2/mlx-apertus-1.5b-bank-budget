from __future__ import annotations

import pytest

from bank_budget.training import trainer
from bank_budget.training.artifacts import parse_time_usage, parse_training_log
from bank_budget.training.config import mlx_lora_config, training_command

# Captured from this project's real `mlx_lm lora` 0.32.2 smoke run (ANSI codes included).
REAL_LOG = (
    "Trainable parameters: 0.356% (5.374M/1510.019M)\n"
    "     1    \x1b[1mval\x1b[0m \x1b[1m10.553\x1b[0m    5.37s\n"
    "     1    \x1b[1m9.370 ▼\x1b[0m    \x1b[1m    4\x1b[0m      0.1k\n"
    "     2    \x1b[1mval\x1b[0m \x1b[1m12.434\x1b[0m    5.28s\n"
    "     2    \x1b[1m9.518 ▲\x1b[0m    \x1b[1m    4\x1b[0m      0.1k\n"
)


def test_parse_training_log_and_usage():
    out = parse_training_log(REAL_LOG)
    assert [p["loss"] for p in out["train_loss"]] == [9.37, 9.518]
    assert [p["loss"] for p in out["val_loss"]] == [10.553, 12.434]
    assert (out["trainable_params_m"], out["total_params_m"]) == (5.374, 1510.019)
    text = "       44.86 real         2.14 user\n          5797252240  peak memory footprint\n"
    assert parse_time_usage(text) == {"wall_time_s": 44.86, "peak_memory_gb": 5.8}


def test_reference_config_matches_the_recipe(project):
    cfg = mlx_lora_config(project.training.profile("reference"), "m", "a")
    assert cfg["optimizer"] == "adamw"
    assert cfg["optimizer_config"] == {
        "adamw": {
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "bias_correction": False,
            "weight_decay": 0.01,
        }
    }
    assert cfg["lr_schedule"] == {
        "name": "cosine_decay",
        "arguments": [1e-4, 1000, 1e-6],
        "warmup": 20,
        "warmup_init": 0.0,
    }
    assert cfg["lora_parameters"] == {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
    }
    assert (cfg["iters"], cfg["batch_size"], cfg["num_layers"], cfg["seed"]) == (1000, 16, 16, 0)
    assert (cfg["steps_per_eval"], cfg["val_batches"], cfg["max_seq_length"]) == (500, -1, 512)
    assert cfg["mask_prompt"] and not cfg["grad_checkpoint"] and cfg["train"] and not cfg["test"]


def test_smoke_profile_has_no_schedule(project):
    assert "lr_schedule" not in mlx_lora_config(project.training.profile("smoke"), "m", "a")


def test_training_command(tmp_path):
    assert training_command(tmp_path / "c.yaml")[-5:] == [
        "-m",
        "mlx_lm",
        "lora",
        "-c",
        str(tmp_path / "c.yaml"),
    ]
    capped = training_command(tmp_path / "c.yaml", cache_limit_gb=0.5, memory_limit_gb=5.0)
    assert capped[-8:] == [
        "-m",
        "bank_budget.training.mlx_launch",
        "--memory-limit-gb",
        "5",
        "--cache-limit-gb",
        "0.5",
        "-c",
        str(tmp_path / "c.yaml"),
    ]


@pytest.mark.macos
def test_launcher_caps_the_cache_and_hands_over_the_arguments(monkeypatch):
    mx = pytest.importorskip("mlx.core", reason="needs MLX (Apple Silicon)")
    lora = pytest.importorskip("mlx_lm.lora")
    import sys

    from bank_budget.training import mlx_launch

    seen = {}
    monkeypatch.setattr(lora, "main", lambda: seen.update(argv=list(sys.argv)))
    limits = []
    monkeypatch.setattr(mx, "set_memory_limit", lambda n: limits.append(("memory", n)))
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: limits.append(("cache", n)))
    mlx_launch.main(["--memory-limit-gb", "5", "--cache-limit-gb", "1.5", "-c", "cfg.yaml"])
    assert limits == [("memory", 5_000_000_000), ("cache", 1_500_000_000)]
    assert seen["argv"] == ["mlx_lm.lora", "-c", "cfg.yaml"]


def test_plan_is_side_effect_free(project):
    plan = trainer.plan_training(project, "reference")
    assert not plan.ok and not plan.adapter_dir.exists()


def test_plan_ok_and_existing_adapter_blocked(prepared_project):
    plan = trainer.plan_training(prepared_project, "smoke")
    assert plan.ok and (plan.train_examples, plan.valid_examples) == (32, 16)
    assert "cosine_decay" not in trainer.describe_plan(plan, prepared_project)
    plan.adapter_dir.mkdir(parents=True)
    (plan.adapter_dir / "adapters.safetensors").write_bytes(b"x")
    blocked = trainer.plan_training(prepared_project, "smoke")
    assert not blocked.ok and "never overwritten" in blocked.problems[0]
    with pytest.raises(trainer.TrainingBlocked):
        trainer.execute_training(blocked, prepared_project)


def test_plan_describes_the_schedule(prepared_project):
    text = trainer.describe_plan(
        trainer.plan_training(prepared_project, "m1_air_8gb"), prepared_project
    )
    assert "then cosine_decay(0.0001, 1000, 1e-06)" in text and "bias correction False" in text


@pytest.mark.macos
def test_generated_options_are_understood_by_installed_mlx_lm(project):
    """Guards against MLX-LM changes: every option must exist and the schedule must build."""
    lora = pytest.importorskip("mlx_lm.lora", reason="needs mlx-lm (Apple Silicon)")
    from mlx_lm.tuner.utils import build_schedule

    known = {a.dest for a in lora.build_parser()._actions} | set(lora.CONFIG_DEFAULTS)
    for name in project.training.profiles:
        cfg = mlx_lora_config(project.training.profile(name), "m", "a")
        assert set(cfg) <= known, set(cfg) - known
    schedule = build_schedule(
        mlx_lora_config(project.training.profile("reference"), "m", "a")["lr_schedule"]
    )
    assert float(schedule(0)) == 0.0
    assert float(schedule(10)) == pytest.approx(0.5e-4)
    assert float(schedule(20)) == pytest.approx(1e-4)
    assert float(schedule(521)) == pytest.approx(1e-6 + 0.5 * (1e-4 - 1e-6), rel=1e-4)
    assert float(schedule(1021)) == pytest.approx(1e-6)


@pytest.mark.macos
def test_mlx_adamw_accepts_the_optimizer_options(project):
    optim = pytest.importorskip("mlx.optimizers", reason="needs MLX (Apple Silicon)")
    cfg = mlx_lora_config(project.training.profile("reference"), "m", "a")
    opt = optim.AdamW(learning_rate=1e-4, **cfg["optimizer_config"]["adamw"])
    assert (opt.weight_decay, opt.bias_correction, opt.eps) == (0.01, False, 1e-8)


def test_parse_time_usage_with_a_decimal_comma():
    text = (
        "      386,31 real         9,92 user        17,80 sys\n"
        "          5974576392  peak memory footprint\n"
    )
    assert parse_time_usage(text) == {"wall_time_s": 386.31, "peak_memory_gb": 5.97}


def test_a_failed_run_without_weights_does_not_block_a_new_one(prepared_project):
    from bank_budget.training.artifacts import RUN_MANIFEST, write_json

    plan = trainer.plan_training(prepared_project, "smoke")
    write_json(plan.adapter_dir / RUN_MANIFEST, {"status": "failed"})
    (plan.adapter_dir / "train.log").write_text("Insufficient Memory\n")
    assert trainer.plan_training(prepared_project, "smoke").ok
    (plan.adapter_dir / "0000100_adapters.safetensors").write_bytes(b"x")  # a checkpoint
    assert not trainer.plan_training(prepared_project, "smoke").ok
    (plan.adapter_dir / "0000100_adapters.safetensors").unlink()
    write_json(plan.adapter_dir / RUN_MANIFEST, {"status": "completed"})
    assert not trainer.plan_training(prepared_project, "smoke").ok

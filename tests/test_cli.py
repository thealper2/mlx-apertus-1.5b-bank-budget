from __future__ import annotations

import json
from pathlib import Path

import pytest

from bank_budget import cli
from bank_budget.publishing import huggingface
from bank_budget.training import trainer
from bank_budget.training.artifacts import RUN_MANIFEST, write_json


def _forbid(*_args, **_kwargs):
    raise AssertionError("must not be called")


def test_parser_validates_arguments():
    parse = cli.build_parser().parse_args
    with pytest.raises(SystemExit):
        parse(["train"])
    with pytest.raises(SystemExit):
        parse(["categorize"])  # needs at least one --line
    with pytest.raises(SystemExit):
        parse(["categorize", "--line", "x", "--max-tokens", "0"])
    with pytest.raises(SystemExit):
        parse(["evaluate", "--model", "finetuned", "--test", "test_new"])
    with pytest.raises(SystemExit):
        parse(["evaluate", "--model", "base", "--test", "test"])
    args = parse(["categorize", "--line", "a", "--line", "b"])
    assert args.line == ["a", "b"]


def test_config_check(project, repo_root_env, capsys):
    repo_root_env(project.root)
    assert cli.main(["config-check"]) == 0
    out = capsys.readouterr().out
    assert "swiss-ai/Apertus-v1.1-1.5B-Instruct" in out and "Configuration OK" in out


def test_unknown_profile_is_a_config_error(prepared_project, repo_root_env):
    repo_root_env(prepared_project.root)
    assert cli.main(["train", "--profile", "huge"]) == 2


def test_train_without_execute_is_a_dry_run(prepared_project, repo_root_env, monkeypatch, capsys):
    repo_root_env(prepared_project.root)
    monkeypatch.setattr(trainer, "execute_training", _forbid)
    assert cli.main(["train", "--profile", "reference"]) == 0
    out = capsys.readouterr().out
    assert "Dry run only" in out and '"iters": 1000' in out and "cosine_decay" in out
    assert not (prepared_project.root / "artifacts/adapters/reference").exists()


def test_train_blocked_when_data_missing(project, repo_root_env, monkeypatch):
    repo_root_env(project.root)
    monkeypatch.setattr(trainer, "execute_training", _forbid)
    assert cli.main(["train", "--profile", "smoke", "--execute"]) == 1


def test_execute_flag_reaches_training(prepared_project, repo_root_env, monkeypatch):
    repo_root_env(prepared_project.root)
    calls = []
    monkeypatch.setattr(
        trainer,
        "execute_training",
        lambda plan, project: calls.append(plan.profile.name) or {"status": "completed"},
    )
    assert cli.main(["train", "--profile", "smoke", "--execute"]) == 0
    assert calls == ["smoke"]


def test_data_commands(prepared_project, repo_root_env, capsys):
    repo_root_env(prepared_project.root)
    assert cli.main(["data", "validate"]) == 0
    assert cli.main(["data", "validate", "--smoke"]) == 0
    assert cli.main(["data", "inspect"]) == 0
    out = capsys.readouterr().out
    assert "Data valid" in out and '"lines": 204' in out


def test_evaluate_needs_an_adapter_for_adapter_mode(prepared_project, repo_root_env):
    repo_root_env(prepared_project.root)
    assert cli.main(["evaluate", "--model", "adapter", "--test", "month"]) == 1


def _fake_adapter(root: Path, kind: str = "full", status: str = "completed") -> Path:
    adapter = root / "artifacts/adapters/m1_air_8gb"
    adapter.mkdir(parents=True)
    (adapter / "adapters.safetensors").write_bytes(b"\0" * 8)
    write_json(
        adapter / "adapter_config.json",
        {
            "model": "/Users/someone/.cache/huggingface/hub/x",
            "num_layers": 16,
            "lora_parameters": {"rank": 8},
        },
    )
    write_json(
        adapter / RUN_MANIFEST, {"run_id": "r1", "status": status, "profile": {"kind": kind}}
    )
    (adapter / ".env").write_text("HF_TOKEN=hf_" + "a" * 30)
    return adapter


def test_publish_requires_destination(prepared_project, repo_root_env, capsys):
    repo_root_env(prepared_project.root)
    _fake_adapter(prepared_project.root)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish"]) == 1
    assert "No destination" in capsys.readouterr().out


def test_publish_dry_run_stages_without_uploading(
    prepared_project, repo_root_env, monkeypatch, capsys
):
    repo_root_env(prepared_project.root)
    _fake_adapter(prepared_project.root)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", _forbid)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish", "--repo-id", "me/bank-budget"]) == 0
    assert "Dry run OK" in capsys.readouterr().out
    staged = prepared_project.publishing.staging_dir
    assert sorted(p.name for p in staged.iterdir()) == [
        "README.md",
        "adapter_config.json",
        "adapters.safetensors",
        "run_manifest.json",
    ]
    config = json.loads((staged / "adapter_config.json").read_text())
    assert config["model"] == "swiss-ai/Apertus-v1.1-1.5B-Instruct"  # local path removed


def test_publish_execute_requires_login(prepared_project, repo_root_env, monkeypatch):
    repo_root_env(prepared_project.root)
    _fake_adapter(prepared_project.root)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    monkeypatch.setattr(huggingface_hub, "HfApi", _forbid)
    assert cli.main(["card"]) == 0
    assert cli.main(["publish", "--repo-id", "me/bank-budget", "--execute"]) == 1


@pytest.mark.parametrize(("kind", "status"), [("smoke", "completed"), ("full", "failed")])
def test_smoke_or_failed_adapters_are_not_publishable(
    prepared_project, repo_root_env, kind, status
):
    repo_root_env(prepared_project.root)
    _fake_adapter(prepared_project.root, kind=kind, status=status)
    assert cli.main(["card"]) == 0
    assert not huggingface.plan_upload(prepared_project, "me/bank-budget").ok


def test_leak_scanner_finds_tokens_and_home_paths(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"t": "hf_" + "b" * 30, "p": str(Path.home() / "x")}))
    assert len(huggingface.scan_for_leaks([f])) == 2


def test_report_without_runs(prepared_project, repo_root_env, capsys):
    repo_root_env(prepared_project.root)
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "| m1_air_8gb | not run |" in out
    assert "| adapter-test_new | – | – | not evaluated |" in out

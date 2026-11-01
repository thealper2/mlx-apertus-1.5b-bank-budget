"""Command-line interface: ``bank-budget <command>``.

Commands that can be expensive or have outside effects need an explicit flag:
``train --execute`` starts training, ``publish --execute`` uploads. Without the flag
both are dry runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
from collections import Counter
from importlib import util as import_util
from pathlib import Path
from typing import Any

from bank_budget import __version__
from bank_budget.config import (
    SPLITS,
    TEST_SPLITS,
    ConfigError,
    ProjectConfig,
    load_project_config,
)
from bank_budget.logging_config import setup_logging
from bank_budget.utils.paths import display, resolve

log = logging.getLogger("bank_budget")


# --------------------------------------------------------------------------- environment


def cmd_check_env(project: ProjectConfig, args: argparse.Namespace) -> int:
    """Report Python/OS/hardware/package/resource status; never loads a model."""
    from bank_budget.training.artifacts import environment_info
    from bank_budget.utils.memory import free_disk_gb, memory_snapshot

    env = environment_info()
    mem = memory_snapshot()
    apple_silicon = env["system"] == "Darwin" and env["machine"] == "arm64"
    disk = free_disk_gb(project.root)
    rows: list[tuple[str, str, bool | None]] = [
        ("Python", env["python"], sys.version_info >= (3, 11)),
        ("OS", f"{env['system']} {env['os_version']}", env["system"] == "Darwin"),
        ("Apple Silicon", str(env.get("chip") or env["machine"]), apple_silicon),
        ("Unified memory", f"{mem.total_gb:.1f} GB" if mem.total_gb else "unknown", None),
        (
            "Approx. available",
            f"{mem.available_gb:.1f} GB (changes constantly)" if mem.available_gb else "unknown",
            None,
        ),
        ("Free disk", f"{disk:.1f} GB", disk >= project.training.safety.min_free_disk_gb),
    ]
    required = {"huggingface-hub", "pandas", "pyyaml"} | (
        {"mlx", "mlx-lm"} if apple_silicon else set()
    )
    for pkg, version in env["packages"].items():
        rows.append((pkg, version or "not installed", bool(version) if pkg in required else None))
    mlx_device = None
    if apple_silicon and import_util.find_spec("mlx") is not None:
        try:
            import mlx.core as mx

            mlx_device = str(mx.default_device())
        except Exception as exc:  # report, do not crash the check
            mlx_device = f"error: {exc}"
    rows.append(
        (
            "MLX device",
            mlx_device or "unavailable",
            None if not apple_silicon else bool(mlx_device and mlx_device.startswith("Device(gpu")),
        )
    )
    rows.append(("Configuration", "valid (all four files)", True))
    token_present = bool(os.environ.get("HF_TOKEN"))
    try:
        from huggingface_hub import get_token

        token_present = token_present or get_token() is not None
    except ImportError:
        pass
    rows.append(
        (
            "Hugging Face login",
            "found (needed only for publishing)"
            if token_present
            else "none (fine: only `make publish` needs it)",
            None,
        )
    )
    failed = False
    for name, value, ok in rows:
        mark = "OK  " if ok else ("FAIL" if ok is False else "INFO")
        failed |= ok is False
        print(f"[{mark}] {name:<18} {value}")
    print("\nMemory figures are approximate; MLX's usable share changes while it runs.")
    if mem.total_gb and mem.total_gb <= 8.5:
        print("8 GB machine detected: use PROFILE=m1_air_8gb for full training.")
    return 1 if failed else 0


def cmd_config_check(project: ProjectConfig, args: argparse.Namespace) -> int:
    """Print the validated configuration (loading it already validated it)."""
    t = project.training
    d = project.data.dataset
    rows = ", ".join(f"{k} {v}" for k, v in project.data.expected_rows.items())
    print(f"Model   : {t.model.repo_id} @ {t.model.revision}")
    print(f"Dataset : {d.repo_id} @ {d.revision} ({rows})")
    names = [args.profile] if args.profile else sorted(t.profiles)
    for name in names:
        p = t.profile(name)
        print(
            f"Profile {name:<10}: kind={p.kind} iters={p.iters} batch={p.batch_size} "
            f"accum={p.grad_accumulation_steps} lr={p.learning_rate:g} ({p.lr_schedule}, "
            f"warmup {p.warmup_steps}) rank={p.lora.rank} scale={p.lora.scale} "
            f"layers={p.num_layers} -> {p.adapter_path}"
        )
    print("Configuration OK")
    return 0


# --------------------------------------------------------------------------- data


def _lines(project: ProjectConfig, offline: bool) -> Any:
    from bank_budget.data.download import load_lines

    return load_lines(project.data.dataset, local_files_only=offline)


def cmd_data_download(project: ProjectConfig, args: argparse.Namespace) -> int:
    lines = _lines(project, args.offline)
    print(
        f"{project.data.dataset.repo_id}: {len(lines):,} lines, "
        f"{lines.description.nunique():,} distinct, {lines.category.nunique()} categories"
    )
    return 0


def cmd_data_inspect(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.data.validate import read_jsonl

    summary = {}
    for name in SPLITS:
        path = project.data.output_dir / f"{name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"{display(path, project.root)} not found; run `make data-prepare`"
            )
        rows = read_jsonl(path)
        chars = sorted(len(r["line"]) for r in rows)
        summary[name] = {
            "lines": len(rows),
            "shops": len({r["shop"] for r in rows}),
            "categories": dict(sorted(Counter(r["category"] for r in rows).items())),
            "credit_lines": sum(r["line"].startswith("[credit]") for r in rows),
            "line_chars": {"min": chars[0], "median": statistics.median(chars), "max": chars[-1]},
            "example": rows[0],
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_data_prepare(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.data.prepare import prepare_dataset

    manifest = prepare_dataset(
        _lines(project, args.offline), project.data, project.root, dry_run=args.dry_run
    )
    print(json.dumps({k: manifest[k] for k in ("lines", "rows", "overlap")}, indent=2))
    for name, info in manifest["files"].items():
        print(f"{name:<11} {info['lines']:>6} lines -> {info['path']}")
    return 0


def cmd_data_smoke(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.data.prepare import prepare_smoke_subset

    manifest = prepare_smoke_subset(project.data, project.root)
    for name, info in manifest["files"].items():
        print(f"{name:<11} {info['lines']:>3} lines -> {info['path']}")
    return 0


def validate_prepared(project: ProjectConfig, smoke: bool) -> list[str]:
    """Validate prepared files; returns a list of problems (empty = valid)."""
    from bank_budget.data.validate import (
        chat_line_problems,
        overlap_report,
        read_jsonl,
        row_problems,
        validate_jsonl,
    )

    d = project.data
    mlx_dir, out_dir = (d.smoke_mlx_dir, d.smoke_dir) if smoke else (d.mlx_dir, d.output_dir)
    reports = [
        validate_jsonl(mlx_dir / "train.jsonl", chat_line_problems),
        validate_jsonl(mlx_dir / "valid.jsonl", chat_line_problems),
    ]
    row_splits = TEST_SPLITS if smoke else SPLITS
    reports += [validate_jsonl(out_dir / f"{s}.jsonl", row_problems) for s in row_splits]
    problems = []
    for r in reports:
        status = "OK" if r.ok else f"{r.error_count} problem(s)"
        print(f"{display(r.path, project.root):<36} {r.lines:>6} lines  {status}")
        problems += [f"{display(r.path, project.root)} {e}" for e in r.errors]
    if problems:
        return problems
    if (mlx_dir / "test.jsonl").exists():
        problems.append(
            f"{display(mlx_dir / 'test.jsonl', project.root)} must not exist: MLX-LM would "
            "use it as an extra split"
        )
    chats = {
        name: [
            {"line": c["messages"][1]["content"], "category": c["messages"][2]["content"]}
            for c in read_jsonl(mlx_dir / f"{name}.jsonl")
        ]
        for name in ("train", "valid")
    }
    tests = {name: read_jsonl(out_dir / f"{name}.jsonl") for name in TEST_SPLITS}
    lines = {name: {r["line"] for r in rows} for name, rows in {**chats, **tests}.items()}
    for a, b in [("train", t) for t in TEST_SPLITS] + [("valid", t) for t in TEST_SPLITS]:
        if shared := len(lines[a] & lines[b]):
            problems.append(f"{shared} identical lines shared by {a} and {b}")
    if smoke:
        return problems
    rows = {name: read_jsonl(out_dir / f"{name}.jsonl") for name in SPLITS}
    if [r["line"] for r in rows["train"]] != [r["line"] for r in chats["train"]]:
        problems.append("mlx/train.jsonl and train.jsonl hold different lines")
    overlap = overlap_report(rows)
    print(f"Overlap between splits (identical lines, shared shop groups): {overlap}")
    for pair, v in overlap.items():
        if v["lines"]:
            problems.append(f"{pair} share {v['lines']} identical lines")
        if "test_new" in pair and v["shops"]:
            problems.append(f"{pair} share {v['shops']} shop groups (test_new must be new shops)")
    sizes = {name: len(r) for name, r in rows.items()}
    if sizes != d.expected_rows:
        problems.append(f"sizes {sizes} differ from expected {d.expected_rows}")
    return problems


def cmd_data_validate(project: ProjectConfig, args: argparse.Namespace) -> int:
    problems = validate_prepared(project, args.smoke)
    for p in problems:
        print(f"PROBLEM: {p}")
    print("Data valid" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


# --------------------------------------------------------------------------- training


def cmd_train(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.training.trainer import describe_plan, execute_training, plan_training

    plan = plan_training(project, args.profile, args.adapter_path)
    print(describe_plan(plan, project), flush=True)
    if not plan.ok:
        print("\nTraining blocked; fix the problems above.")
        return 1
    if not args.execute:
        print(
            "\nDry run only: no model was downloaded and nothing was trained. "
            "Add --execute to train."
        )
        return 0
    manifest = execute_training(plan, project)
    return 0 if manifest["status"] == "completed" else 1


# --------------------------------------------------------------------------- inference


def _adapter_arg(project: ProjectConfig, value: str | None) -> Path | None:
    return resolve(value, project.root) if value else None


def cmd_categorize(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.inference.generate import categorize, load_model, peak_memory_gb

    m = project.training.model
    adapter = _adapter_arg(project, args.adapter)
    model, tok = load_model(m.repo_id, m.revision, adapter)
    results = []
    for line in args.line:
        gen = categorize(model, tok, line, args.max_tokens)
        results.append(
            {
                "line": line,
                "raw_output": gen.raw,
                "category": gen.category,
                "latency_ms": round(1000 * gen.latency_s, 1),
            }
        )
    out = {
        "model": f"{m.repo_id}@{m.revision}",
        "adapter": display(adapter, project.root) if adapter else None,
        "results": results,
        "peak_memory_gb": peak_memory_gb(),
    }
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        path = resolve(args.output, project.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def _resolve_adapter(project: ProjectConfig, args: argparse.Namespace) -> tuple[Any, Any]:
    """(adapter path or None, adapter metadata or None) for ``--model``."""
    from bank_budget.inference.generate import check_adapter_dir
    from bank_budget.training.artifacts import RUN_MANIFEST, read_json

    if args.model != "adapter":
        return None, None
    adapter = _adapter_arg(project, args.adapter) or project.publishing.adapter_path
    check_adapter_dir(adapter)
    run = read_json(adapter / RUN_MANIFEST) or {}
    return adapter, {
        "path": display(adapter, project.root),
        "run_id": run.get("run_id"),
        "profile": run.get("profile", {}).get("name"),
    }


def cmd_evaluate(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.budget import budget_table, read_statement, render_budget
    from bank_budget.data.prepare import sha256_file
    from bank_budget.data.validate import read_jsonl
    from bank_budget.evaluation.evaluator import build_result, evaluate_lines, save_result
    from bank_budget.inference.generate import (
        Generation,
        categorize,
        load_model,
        peak_memory_gb,
        warm_up,
    )
    from bank_budget.training.artifacts import environment_info

    ev = project.evaluation
    if args.test == "month":
        test_file = ev.statement_file
        if not test_file.is_file():
            raise FileNotFoundError(f"{display(test_file, project.root)} not found")
        rows: list[dict[str, Any]] = read_statement(test_file)
        description = "a month of made-up statement lines"
    else:
        test_file = (ev.smoke_test_files if args.smoke else ev.test_files)[args.test]
        if not test_file.is_file():
            raise FileNotFoundError(
                f"{display(test_file, project.root)} not found; run "
                f"`make {'data-smoke' if args.smoke else 'data-prepare'}`"
            )
        limit = args.max_examples or (ev.smoke_max_examples if args.smoke else ev.max_examples)
        rows = read_jsonl(test_file)
        rows = rows[:limit] if limit else rows
        description = {
            "test_new": "lines of shops never seen in training",
            "test_known": "new lines of shops seen in training",
        }[args.test]
    adapter, adapter_meta = _resolve_adapter(project, args)
    smoke = args.smoke and args.test != "month"
    label = f"{args.model}-{args.test}" + ("-smoke" if smoke else "")
    m = project.training.model
    log.info("Evaluating %s on %d lines", label, len(rows))
    model, tok = load_model(m.repo_id, m.revision, adapter)
    warm_up(model, tok, ev.warmup_generations)

    def categorize_one(line: str) -> Generation:
        return categorize(model, tok, line, ev.max_tokens)

    results = evaluate_lines(rows, categorize_one)
    meta: dict[str, Any] = {
        "model": {"repo_id": m.repo_id, "revision": m.revision},
        "adapter": adapter_meta,
        "test": f"{args.test} ({description})",
        "test_file": display(test_file, project.root),
        "test_file_sha256": sha256_file(test_file),
        "generation": {
            "max_tokens": ev.max_tokens,
            "sampler": "greedy (temperature 0)",
            "warmup_generations": ev.warmup_generations,
        },
    }
    if args.test == "month":
        meta["budget"] = budget_table(rows, [r["prediction"] for r in results])
    meta["peak_memory_gb"] = peak_memory_gb()
    meta["environment"] = environment_info()
    result = build_result(label, results, meta)
    json_path, md_path = save_result(result, ev.output_dir)
    mt = result["metrics"]
    top = mt["most_common_answer"]
    print(
        f"{label}: {mt['correct']}/{mt['examples']} right ({mt['accuracy']}), most common "
        f"answer {top['category']} ({top['count']}x), median {mt['median_latency_ms']} ms per line"
    )
    if args.test == "month":
        print(render_budget(meta["budget"]))
    print(f"Results: {display(json_path, project.root)}, {display(md_path, project.root)}")
    return 0


def cmd_report(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.reporting import write_experiment_report

    path = write_experiment_report(project)
    print(path.read_text())
    print(f"Written to {display(path, project.root)}")
    return 0


# --------------------------------------------------------------------------- publishing


def cmd_card(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.publishing.model_card import collect_inputs, training_status, write_card

    inputs = collect_inputs(project, _adapter_arg(project, args.adapter), args.repo_id)
    out = resolve(args.output, project.root) if args.output else project.publishing.model_card_path
    write_card(inputs, out)
    print(
        f"Model card written to {display(out, project.root)} "
        f"(training status: {training_status(inputs.run)}; base evaluations: "
        f"{sum(v is not None for v in inputs.base_evals.values())}/3; adapter evaluations: "
        f"{sum(v is not None for v in inputs.adapter_evals.values())}/3)"
    )
    return 0


def cmd_publish(project: ProjectConfig, args: argparse.Namespace) -> int:
    from bank_budget.publishing.huggingface import (
        PublishBlocked,
        describe_upload,
        plan_upload,
        publish,
    )

    plan = plan_upload(project, args.repo_id, _adapter_arg(project, args.adapter))
    print(describe_upload(plan, project))
    if not plan.ok:
        print("\nPublication blocked; nothing was uploaded.")
        return 1
    try:
        url = publish(plan, project, execute=args.execute)
    except PublishBlocked as exc:
        print(f"BLOCKED: {exc}")
        return 1
    print(f"Uploaded: {url}" if url else "\nDry run OK: staged locally, nothing uploaded.")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bank-budget",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("--config-dir", type=Path, help="alternative configs/ directory")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check-env", help="check Python, macOS, MLX, packages, memory, disk")
    p = sub.add_parser("config-check", help="validate configs and print training profiles")
    p.add_argument("--profile")

    data = sub.add_parser("data", help="dataset commands").add_subparsers(
        dest="data_command", required=True
    )
    for name, text in (
        ("download", "download (or reuse the cached) dataset file and count its lines"),
        ("prepare", "split by shop and write the MLX-LM and test files"),
    ):
        p = data.add_parser(name, help=text)
        p.add_argument("--offline", action="store_true", help="use the local cache only")
        if name == "prepare":
            p.add_argument("--dry-run", action="store_true", help="compute stats, write nothing")
    data.add_parser("inspect", help="summary statistics of the prepared splits")
    data.add_parser("smoke", help="write the tiny smoke subset (canonical data untouched)")
    p = data.add_parser("validate", help="validate prepared data files")
    p.add_argument("--smoke", action="store_true", help="validate the smoke subset instead")

    p = sub.add_parser("train", help="plan (default) or run MLX-LM LoRA training")
    p.add_argument("--profile", required=True, help="training profile from training.yaml")
    p.add_argument("--adapter-path", help="override the profile's adapter output directory")
    p.add_argument(
        "--execute", action="store_true", help="actually train (without it this is a dry run)"
    )

    p = sub.add_parser("categorize", help="file one or more statement lines")
    p.add_argument("--line", action="append", required=True, help="a line; repeatable")
    p.add_argument("--adapter", help="adapter directory; omit to use the base model")
    p.add_argument("--max-tokens", type=_positive_int, default=8)
    p.add_argument("--output", help="also write the result as JSON to this file")

    p = sub.add_parser("evaluate", help="evaluate the base model or base + adapter")
    p.add_argument("--model", choices=("base", "adapter"), required=True)
    p.add_argument("--test", choices=(*TEST_SPLITS, "month"), required=True)
    p.add_argument("--adapter", help="adapter directory (default: publishing.adapter_path)")
    p.add_argument("--smoke", action="store_true", help="tiny deterministic subset")
    p.add_argument("--max-examples", type=_positive_int)

    sub.add_parser("report", help="summarise recorded training runs and evaluations")

    p = sub.add_parser("card", help="generate the Hugging Face model card locally")
    p.add_argument("--adapter")
    p.add_argument("--repo-id")
    p.add_argument("--output", help="output path (default: publishing.model_card_path)")

    p = sub.add_parser("publish", help="dry run (default) or upload the adapter to the Hub")
    p.add_argument("--repo-id", help="override publishing.repo_id")
    p.add_argument("--adapter")
    p.add_argument("--execute", action="store_true", help="actually upload")
    return parser


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


_COMMANDS = {
    "check-env": cmd_check_env,
    "config-check": cmd_config_check,
    "train": cmd_train,
    "categorize": cmd_categorize,
    "evaluate": cmd_evaluate,
    "report": cmd_report,
    "card": cmd_card,
    "publish": cmd_publish,
}
_DATA_COMMANDS = {
    "download": cmd_data_download,
    "inspect": cmd_data_inspect,
    "prepare": cmd_data_prepare,
    "smoke": cmd_data_smoke,
    "validate": cmd_data_validate,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        project = load_project_config(config_dir=args.config_dir)
        handler = (
            _DATA_COMMANDS[args.data_command] if args.command == "data" else _COMMANDS[args.command]
        )
        return handler(project, args)
    except (ConfigError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 2
    except (RuntimeError, ValueError) as exc:  # data, training, inference, publishing errors
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.error("Interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Generate the Hugging Face model card (README.md) from configs and recorded outputs.

Everything in the card comes from configuration files, the data manifest, the training
run manifest and saved evaluation records. Values that were never measured are written
as "not measured" / "pending"; nothing is estimated or filled in.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bank_budget.budget import render_budget
from bank_budget.config import ProjectConfig
from bank_budget.evaluation.evaluator import load_latest
from bank_budget.task import CATEGORIES, SYSTEM_PROMPT
from bank_budget.training.artifacts import RUN_MANIFEST, read_json
from bank_budget.utils.paths import display

NOT_MEASURED = "not measured"
PENDING = "pending: not evaluated yet"
REQUIRED_SECTIONS = (
    "## Model overview",
    "## Model details",
    "## Intended use",
    "## Training data",
    "## Training procedure",
    "## Evaluation",
    "## How to use",
    "## Limitations and risks",
    "## License and attribution",
    "## Reproducibility",
)


class _Recorded(dict):
    """Manifest section where absent keys read as "not recorded" instead of failing."""

    def __missing__(self, key: str) -> str:
        return "not recorded"


EVAL_LABELS = ("test_new", "test_known", "month")


@dataclass(frozen=True)
class CardInputs:
    project: ProjectConfig
    adapter_dir: Path
    repo_id: str | None
    run: dict[str, Any] | None
    data_manifest: dict[str, Any] | None
    base_evals: dict[str, dict[str, Any] | None]
    adapter_evals: dict[str, dict[str, Any] | None]


def training_status(run: dict[str, Any] | None) -> str:
    """``not trained`` | ``smoke test only`` | ``completed`` | ``failed`` | ``interrupted``."""
    if run is None:
        return "not trained"
    if run.get("profile", {}).get("kind") == "smoke" and run.get("status") == "completed":
        return "smoke test only"
    return str(run.get("status", "unknown"))


def collect_inputs(
    project: ProjectConfig, adapter_dir: Path | None = None, repo_id: str | None = None
) -> CardInputs:
    """Gather all recorded outputs the card is built from.

    A test record is used only when it evaluated the canonical file (the SHA-256 in the
    data manifest; for the month, the configured statement file) with the configured model
    revision; an adapter record only when it also came from this exact adapter run.
    """
    from bank_budget.data.prepare import sha256_file

    pub = project.publishing
    adapter_dir = adapter_dir or pub.adapter_path
    run = read_json(adapter_dir / RUN_MANIFEST)
    manifest = read_json(project.data.output_dir / "manifest.json")
    files = (manifest or {}).get("files", {})
    statement = project.evaluation.statement_file
    expected_sha = {
        "test_new": (files.get("test_new") or {}).get("sha256"),
        "test_known": (files.get("test_known") or {}).get("sha256"),
        "month": sha256_file(statement) if statement.is_file() else None,
    }
    out_dir = project.evaluation.output_dir
    revision = project.training.model.revision

    def usable(ev: dict[str, Any] | None, test: str, adapter: bool) -> dict[str, Any] | None:
        if not ev or not expected_sha[test] or ev.get("test_file_sha256") != expected_sha[test]:
            return None
        if (ev.get("model") or {}).get("revision") != revision:
            return None
        if adapter and (
            run is None or (ev.get("adapter") or {}).get("run_id") != run.get("run_id")
        ):
            return None
        return ev

    base = {t: usable(load_latest(out_dir, f"base-{t}"), t, False) for t in EVAL_LABELS}
    tuned = {t: usable(load_latest(out_dir, f"adapter-{t}"), t, True) for t in EVAL_LABELS}
    return CardInputs(project, adapter_dir, repo_id or pub.repo_id, run, manifest, base, tuned)


def card_metadata(inputs: CardInputs) -> dict[str, Any]:
    """Hugging Face YAML front matter, derived from configuration only."""
    p = inputs.project
    return {
        "base_model": p.training.model.repo_id,
        "library_name": "mlx",
        "license": p.publishing.adapter_license,
        "datasets": [p.data.dataset.repo_id],
        "language": ["en"],
        "pipeline_tag": "text-classification",
        "tags": ["mlx", "lora", "apertus", "finance", "budgeting", "text-classification"],
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, dict):
        return ", ".join(
            f"{k} {v:,}" if isinstance(v, int) else f"{k} {v}" for k, v in value.items()
        )
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _pct(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{100 * value:.1f}%"


def _cell(ev: dict[str, Any] | None) -> str:
    if ev is None:
        return PENDING
    m = ev["metrics"]
    return f"{m['correct']:,} / {m['examples']:,} ({_pct(m['accuracy'])})"


def _top(ev: dict[str, Any] | None) -> str:
    if ev is None:
        return "–"
    top = ev["metrics"]["most_common_answer"]
    return f"{top['category']} ({top['count']:,}×)"


def render_card(inputs: CardInputs) -> str:
    """Full README.md for the publication repository."""
    p = inputs.project
    model = p.training.model
    ds = p.data.dataset
    d = p.data
    run = inputs.run
    status = training_status(run)
    if run and run.get("profile"):
        profile = _Recorded(run["profile"])
        profile_name = str(profile["name"])
    else:
        profile = _Recorded(p.training.profile("m1_air_8gb").to_dict())
        profile_name = "m1_air_8gb (planned)"
    lora = _Recorded(profile.get("lora") or {})
    metrics = (run or {}).get("metrics", {})
    env = (run or {}).get("environment", {})
    pkgs = env.get("packages", {})
    manifest = inputs.data_manifest or {}
    lines_info = manifest.get("lines") or {}
    repo = inputs.repo_id or "<your-namespace>/<adapter-repo>"
    categories = ", ".join(CATEGORIES)
    raw_keys = lora.get("keys")
    keys: list[str] = list(raw_keys) if isinstance(raw_keys, list) else []
    n_test = d.expected_rows["test_new"]

    sections: list[str] = []
    add = sections.append
    add("---\n" + yaml.safe_dump(card_metadata(inputs), sort_keys=False) + "---")
    add(
        "# Apertus 1.5B bank-statement budget categoriser (MLX LoRA adapter)\n\n"
        f"A LoRA adapter for [`{model.repo_id}`](https://huggingface.co/{model.repo_id}) that "
        "files one US bank-statement line under one of 17 budget categories: "
        f"{categories}. Add up a month of filed lines and you have a budget. Trained and run "
        "with Apple MLX (`mlx-lm`).\n\n"
        f"**Training status: {status}.**"
        + (
            ""
            if status == "completed"
            else " The numbers below are only filled in from recorded runs; anything not yet run "
            "is marked as pending or not measured."
        )
    )
    add(
        "## Model overview\n\n"
        "| | |\n|---|---|\n"
        "| Task | one statement line (`[debit] …` / `[credit] …`) → one of 17 categories, "
        "answered as text |\n"
        f"| Base model | `{model.repo_id}` (revision `{model.revision}`) |\n"
        "| Architecture | Apertus (`apertus`), 16 blocks, hidden size 2,048, xIELU MLP |\n"
        f"| Fine-tuning | LoRA on {', '.join(f'`{k}`' for k in keys) or 'not recorded'} of all "
        "16 blocks |\n"
        "| Framework | MLX / MLX-LM (Apple Silicon, Metal) |\n"
        f"| Project status | experimental / educational; training status: {status} |"
    )
    add(
        "## Model details\n\n"
        f"- **Base model:** `{model.repo_id}` at commit `{model.revision}`, bf16, license "
        f"`{model.license}` (Swiss AI Initiative; its model card also asks users to follow the "
        "Apertus Acceptable Use Policy).\n"
        "- **This repository contains only a LoRA adapter** (`adapters.safetensors` + "
        "`adapter_config.json`, MLX-LM format). It is not a standalone model and must be "
        "loaded on top of the exact base model above.\n"
        f"- **Trainable parameters:** {_fmt(metrics.get('trainable_params_m'), 'M')} of "
        f"{_fmt(metrics.get('total_params_m'), 'M')}.\n"
        f"- **Software:** mlx {_fmt(pkgs.get('mlx'))}, mlx-lm {_fmt(pkgs.get('mlx-lm'))}."
    )
    add(
        "## Intended use\n\n"
        "Sorting the lines of a personal US bank or card statement into budget categories, "
        "to sum a month's spending per category, for experiments and teaching. A person "
        "should look over the result: a budget is only as good as its least obvious lines.\n\n"
        "Not for: credit, lending, fraud or tax decisions; statements in other countries, "
        "languages or currencies without checking; business bookkeeping that needs a chart "
        "of accounts."
    )
    add(
        "## Training data\n\n"
        f"- **Dataset:** [`{ds.repo_id}`](https://huggingface.co/datasets/{ds.repo_id}), "
        f"revision `{ds.revision}`, license `{ds.license}`: {ds.expected_lines:,} **synthetic** "
        "US bank-statement lines in 17 categories, made from merchant names in the formats of "
        "several banks and payment services.\n"
        f"- **Copies removed:** texts with two different categories dropped, every other text "
        f"kept once: {_fmt(lines_info.get('distinct'))} distinct lines.\n"
        "- **Shops held out:** every line gets a shop group (payment wrappers such as `SQ *`, "
        "`TST*`, `PYPL *`, store numbers and addresses removed; lines sharing the first word or "
        "the first 6 letters are one shop). Per category, random shop groups are held out until "
        f"they hold about {100 * d.split.holdout_share:.0f}% of its lines: "
        f"{_fmt(lines_info.get('held_out_shop_groups'))} of "
        f"{_fmt(lines_info.get('shop_groups'))} groups.\n"
        f"- **Splits:** {_fmt(manifest.get('rows'))}. `test_new` = "
        f"{d.split.test_per_category} lines per category from held-out shops; `test_known` = "
        f"{d.split.test_per_category} per category of new lines from the other shops; `valid` = "
        f"{d.split.valid_per_category} per category (validation loss only).\n"
        f"- **Overlap:** {_fmt((manifest.get('overlap') or {}).get('train/test_new'))} between "
        "train and `test_new`; no identical line in two splits.\n"
        "- **Format:** MLX-LM chat JSONL — the system prompt below, the line as the user turn, "
        "the category as the answer; loss on the answer only.\n\n"
        f"```text\n{SYSTEM_PROMPT}\n```\n\n"
        "- **Limitations:** synthetic lines; real statements have other merchants, formats and "
        "abbreviations."
    )
    add(
        "## Training procedure\n\n"
        "| Setting | Value |\n|---|---|\n"
        f"| Profile | `{profile_name}` |\n"
        f"| LoRA | rank {lora['rank']}, scale {lora['scale']}, dropout {lora['dropout']}, "
        f"{profile['num_layers']} blocks |\n"
        f"| Optimizer | {profile['optimizer']}, betas {profile['adam_beta1']}/"
        f"{profile['adam_beta2']}, eps {profile['adam_eps']}, weight decay "
        f"{profile['weight_decay']}, bias correction {profile['bias_correction']} |\n"
        f"| Learning rate | {profile['learning_rate']}: {profile['warmup_steps']} warmup updates, "
        f"then cosine decay over {profile['decay_steps']} updates to "
        f"{profile['end_learning_rate']} |\n"
        f"| Steps | {profile['iters']} × batch {profile['batch_size']} (accumulation "
        f"{profile['grad_accumulation_steps']}) |\n"
        f"| Max sequence length | {profile['max_seq_length']} |\n"
        f"| Prompt masking / gradient checkpointing | {profile['mask_prompt']} / "
        f"{profile['grad_checkpoint']} |\n"
        f"| Seed | {profile['seed']} |\n"
        f"| Final train loss | {_fmt(metrics.get('final_train_loss'))} |\n"
        f"| Final validation loss | {_fmt(metrics.get('final_val_loss'))} |\n"
        f"| Wall time | {_fmt(metrics.get('wall_time_s'), ' s')} |\n"
        f"| Peak memory footprint | {_fmt(metrics.get('peak_memory_gb'), ' GB')} |\n"
        + (
            f"| Hardware | {env.get('chip')}, {env.get('memory_gb')} GB unified memory |\n"
            f"| OS / Python | macOS {env.get('os_version')} / Python {env.get('python')} |\n"
            if env
            else "| Hardware / OS | not recorded (no training run yet) |\n"
        )
        + f"| Run id | {_fmt((run or {}).get('run_id'))} |"
    )
    month = inputs.adapter_evals.get("month")
    add(
        "## Evaluation\n\n"
        "Greedy decoding, at most "
        f"{p.evaluation.max_tokens} new tokens, the same system prompt for the base model and "
        "the adapter. The answer is the category it names (exactly, or the earliest category "
        "name in it); an answer without one counts as wrong.\n\n"
        "| Test | Base model | Base + this adapter | Most common answer (base / adapter) |\n"
        "|---|---|---|---|\n"
        + "\n".join(
            f"| {name} | {_cell(inputs.base_evals.get(t))} | {_cell(inputs.adapter_evals.get(t))} "
            f"| {_top(inputs.base_evals.get(t))} / {_top(inputs.adapter_evals.get(t))} |"
            for t, name in (
                ("test_new", f"{n_test:,} lines of shops never seen in training"),
                ("test_known", f"{n_test:,} new lines of shops seen in training"),
                ("month", "a month of made-up statement lines (31)"),
            )
        )
        + "\n\n"
        f"A 95% interval at {n_test:,} lines is up to about "
        f"±{100 * 1.96 * (0.25 / n_test) ** 0.5:.1f} percentage points. Both tests come from the "
        "same synthetic generator as the training data."
        + (
            "\n\nThe month as filed by this adapter:\n\n" + render_budget(month["budget"])
            if month and month.get("budget")
            else ""
        )
    )
    add(
        "## How to use\n\n"
        "Requires macOS on Apple Silicon.\n\n"
        '```bash\npip install "mlx-lm>=0.32.2,<0.33" huggingface_hub\n```\n\n'
        "```python\n"
        "from huggingface_hub import snapshot_download\n"
        "from mlx_lm import generate, load\n"
        "from mlx_lm.sample_utils import make_sampler\n\n"
        f'SYSTEM = "{SYSTEM_PROMPT}"\n\n'
        f'adapter_dir = snapshot_download("{repo}")\n'
        f'model, tokenizer = load("{model.repo_id}", revision="{model.revision}",\n'
        "                        adapter_path=adapter_dir)\n\n"
        'line = "[debit] SQ *COPPERWICK KETTLE CAFE 1187 DENVER CO"\n'
        "prompt = tokenizer.apply_chat_template(\n"
        '    [{"role": "system", "content": SYSTEM}, {"role": "user", "content": line}],\n'
        "    add_generation_prompt=True, tokenize=False)\n"
        "print(generate(model, tokenizer, prompt, "
        f"max_tokens={p.evaluation.max_tokens}, sampler=make_sampler(temp=0.0)))\n"
        "```\n\n"
        "Use exactly this system prompt and start every line with `[debit]` or `[credit]`: "
        "that is how the adapter was trained. `mlx_lm.load` applies the adapter from a local "
        "folder, which is why the repository is downloaded first."
    )
    add(
        "## Limitations and risks\n\n"
        "- Lines of shops it has never seen are clearly harder than new lines of known shops; "
        "check the two test rows above.\n"
        "- Ambiguous lines (a pharmacy that also sells groceries, a transfer that is really "
        "rent) get one category; a person may file them differently.\n"
        "- Trained and tested on synthetic lines; accuracy on real statements is unknown.\n"
        "- The answer is free text; always map it to an allowed category and handle answers "
        "without one.\n"
        "- US formats and English only.\n"
        "- **Do not use it for decisions about people** (credit, lending, fraud)."
    )
    add(
        "## License and attribution\n\n"
        f"- **Adapter:** `{p.publishing.adapter_license}` (as configured by the publisher).\n"
        f"- **Base model:** `{model.repo_id}`, Swiss AI Initiative, license `{model.license}`; "
        "users are asked to follow the Apertus Acceptable Use Policy.\n"
        f"- **Dataset:** `{ds.repo_id}`, license `{ds.license}` (synthetic).\n"
        "- The code, dataset and model licenses are independent; check each before reuse."
    )
    add(
        "## Reproducibility\n\n"
        "The project that produced this adapter keeps every setting in `configs/` and drives "
        "everything through `make`:\n\n"
        "```bash\n"
        "make install\n"
        "make data-prepare\n"
        "make data-validate\n"
        "make evaluate-base\n"
        f"make train PROFILE={profile_name.split()[0]}\n"
        "make evaluate\n"
        "make card\n"
        "make publish-dry-run\n"
        "```\n\n"
        f"Artifacts: adapter in `{display(inputs.adapter_dir, p.root)}` with `{RUN_MANIFEST}` "
        "(configuration, versions, losses, timings), evaluation records under "
        f"`{display(p.evaluation.output_dir, p.root)}/`."
    )
    return "\n\n".join(sections) + "\n"


def write_card(inputs: CardInputs, output: Path) -> Path:
    """Write the card to ``output`` (refuses to write over a hand-written README)."""
    if output.name == "README.md" and output.parent.resolve() == inputs.project.root.resolve():
        raise ValueError("Refusing to overwrite the project's own README.md with a model card")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_card(inputs), encoding="utf-8")
    return output


def parse_front_matter(card: str) -> dict[str, Any]:
    """Parse the YAML metadata block of a card (raises ValueError when absent/invalid)."""
    if not card.startswith("---\n"):
        raise ValueError("card does not start with YAML front matter")
    end = card.find("\n---", 4)
    if end < 0:
        raise ValueError("unterminated YAML front matter")
    data = yaml.safe_load(card[4:end])
    if not isinstance(data, dict):
        raise ValueError("front matter is not a mapping")
    return data

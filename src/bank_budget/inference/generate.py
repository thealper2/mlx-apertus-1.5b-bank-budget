"""Load the base model (optionally with a LoRA adapter) and file statement lines.

MLX imports happen inside the functions so the rest of the package (and the unit tests)
work on machines without MLX.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bank_budget.task import parse_category, render_generation_prompt

log = logging.getLogger(__name__)
WARMUP_LINE = "[debit] warm up"


class InferenceUnavailable(RuntimeError):
    """MLX / MLX-LM cannot be used on this machine or the artifacts are missing."""


@dataclass(frozen=True)
class Generation:
    raw: str
    category: str  # one of the 17 categories or "other"
    latency_s: float


def require_mlx() -> None:
    """Fail early with a clear message when MLX-LM cannot run here."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise InferenceUnavailable("MLX-LM needs macOS on Apple Silicon (arm64).")
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise InferenceUnavailable("mlx-lm is not installed; run `make install`.") from exc


def check_adapter_dir(adapter_path: Path) -> None:
    """Raise when ``adapter_path`` does not hold an MLX-LM adapter."""
    missing = [
        n
        for n in ("adapters.safetensors", "adapter_config.json")
        if not (adapter_path / n).is_file()
    ]
    if missing:
        raise InferenceUnavailable(
            f"{adapter_path} is not a trained adapter (missing {', '.join(missing)}). "
            "Train one first, or run without an adapter to use the base model."
        )


def load_model(
    repo_id: str, revision: str | None, adapter_path: Path | None = None
) -> tuple[Any, Any]:
    """Load model + tokenizer once. With ``adapter_path`` the LoRA layers are applied."""
    require_mlx()
    if adapter_path is not None:
        check_adapter_dir(adapter_path)
    from mlx_lm import load

    log.info(
        "Loading %s @ %s%s",
        repo_id,
        (revision or "main")[:12],
        f" + adapter {adapter_path}" if adapter_path else " (base model, no adapter)",
    )
    loaded = load(
        repo_id, revision=revision, adapter_path=str(adapter_path) if adapter_path else None
    )
    return loaded[0], loaded[1]


def _generate(model: Any, tokenizer: Any, line: str, max_tokens: int) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    prompt = render_generation_prompt(tokenizer, line)
    return str(
        generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
    )


def warm_up(model: Any, tokenizer: Any, times: int) -> None:
    """A few untimed generations, so the first timed line does not pay for GPU warm-up."""
    for _ in range(times):
        _generate(model, tokenizer, WARMUP_LINE, max_tokens=4)


def categorize(model: Any, tokenizer: Any, line: str, max_tokens: int) -> Generation:
    """Greedy-decode the model's answer for one line and read the category from it."""
    start = time.perf_counter()
    raw = _generate(model, tokenizer, line, max_tokens)
    latency = time.perf_counter() - start
    return Generation(raw=raw, category=parse_category(raw), latency_s=round(latency, 4))


def peak_memory_gb() -> float | None:
    """Peak MLX memory of this process so far, if MLX is available."""
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return round(mx.get_peak_memory() / 1e9, 2)

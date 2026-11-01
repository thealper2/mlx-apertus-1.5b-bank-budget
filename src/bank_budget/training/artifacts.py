"""Run manifests, environment provenance, adapter checks and training-log parsing."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

RUN_MANIFEST = "run_manifest.json"
TRAINING_CONFIG_COPY = "training_config.yaml"
TRAIN_LOG = "train.log"
TIME_USAGE = "time_usage.txt"
ADAPTER_FILES = ("adapters.safetensors", "adapter_config.json")
TRACKED_PACKAGES = (
    "mlx",
    "mlx-metal",
    "mlx-lm",
    "huggingface-hub",
    "transformers",
    "pandas",
    "numpy",
    "pyyaml",
)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# MLX-LM 0.32 report lines: "  10    1.234 ▼    1,234    12.3k" and "  10    val 1.234    0.52s"
_TRAIN_LINE = re.compile(r"^\s*(\d+)\s+(\d+\.\d+)\s+[▼▲]\s+([\d,]+)\s+([\d.]+)k\s*$")
_VAL_LINE = re.compile(r"^\s*(\d+)\s+val\s+(\d+\.\d+)\s+([\d.]+)s\s*$")
_TRAINABLE = re.compile(r"Trainable parameters:\s*([\d.]+)%\s*\(([\d.]+)M/([\d.]+)M\)")


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_run_id(label: str) -> str:
    """``<label>-YYYYmmdd-HHMMSS`` in UTC."""
    return f"{label}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that define the training/inference stack."""
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _sysctl(key: str) -> str | None:
    try:
        res = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=5, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() or None


def environment_info() -> dict[str, Any]:
    """Python, OS and hardware description for manifests (no user-identifying data)."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0] or platform.release(),
        "packages": package_versions(),
    }
    if platform.system() == "Darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        mem = _sysctl("hw.memsize")
        info["memory_gb"] = round(int(mem) / 1024**3) if mem and mem.isdigit() else None
    return info


def git_commit(root: Path) -> str | None:
    """Current commit hash (with ``-dirty`` when the tree has changes), if in a git repo."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def adapter_problems(adapter_dir: Path) -> list[str]:
    """Missing or empty adapter files; empty list when the adapter looks complete."""
    problems = []
    for name in ADAPTER_FILES:
        path = adapter_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif path.stat().st_size == 0:
            problems.append(f"{name} is empty")
    return problems


def parse_training_log(text: str) -> dict[str, Any]:
    """Extract loss curves and trainable-parameter info from MLX-LM's console output.

    Returns empty lists when nothing could be parsed (e.g. a changed output format);
    callers must treat missing values as unavailable, never as zero.
    """
    clean = _ANSI.sub("", text).replace("\r", "\n")
    train, val = [], []
    for line in clean.splitlines():
        if m := _TRAIN_LINE.match(line):
            train.append(
                {
                    "iter": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "tokens_per_s": float(m.group(3).replace(",", "")),
                }
            )
        elif m := _VAL_LINE.match(line):
            val.append({"iter": int(m.group(1)), "loss": float(m.group(2))})
    out: dict[str, Any] = {"train_loss": train, "val_loss": val}
    if m := _TRAINABLE.search(clean):
        out["trainable_percent"] = float(m.group(1))
        out["trainable_params_m"] = float(m.group(2))
        out["total_params_m"] = float(m.group(3))
    return out


def parse_time_usage(text: str) -> dict[str, float]:
    """Wall time and peak memory from macOS ``/usr/bin/time -l`` output."""
    out: dict[str, float] = {}
    # The decimal separator follows the locale: "386.31 real" or "386,31 real".
    if m := re.search(r"(\d+(?:[.,]\d+)?)\s+real", text):
        out["wall_time_s"] = float(m.group(1).replace(",", "."))
    if m := re.search(r"(\d+)\s+peak memory footprint", text):
        out["peak_memory_gb"] = round(int(m.group(1)) / 1e9, 2)
    elif m := re.search(r"(\d+)\s+maximum resident set size", text):
        out["max_rss_gb"] = round(int(m.group(1)) / 1e9, 2)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any] | None:
    """Parsed JSON object, or None when the file does not exist."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None

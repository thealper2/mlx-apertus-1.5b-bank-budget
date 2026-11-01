"""Write the MLX-LM training files and the held-out test files.

Outputs (canonical, ``data/processed``)::

    mlx/train.jsonl, mlx/valid.jsonl     chat examples for mlx_lm lora (system, line, category)
    train.jsonl, valid.jsonl             the same lines as {line, category, shop}
    test_new.jsonl, test_known.jsonl     {line, category, shop}: the two tests
    manifest.json                        provenance, counts, shop overlap, file hashes

The split itself is defined in :mod:`bank_budget.data.split`.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bank_budget.config import SPLITS, TEST_SPLITS, DataConfig
from bank_budget.data.split import make_splits, to_rows
from bank_budget.data.validate import overlap_report, read_jsonl, row_problems
from bank_budget.task import CATEGORIES, chat_record
from bank_budget.utils.paths import display

log = logging.getLogger(__name__)

MANIFEST = "manifest.json"


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL atomically (temp file + rename). Returns the line count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def prepare_dataset(
    lines: Any, cfg: DataConfig, root: Path, dry_run: bool = False
) -> dict[str, Any]:
    """Split the dataset lines, validate the result and write everything."""
    if len(lines) != cfg.dataset.expected_lines:
        raise ValueError(
            f"the dataset has {len(lines)} lines, expected {cfg.dataset.expected_lines}; "
            "check the revision"
        )
    parts = make_splits(lines, cfg.split)
    if len(parts["unique"]) != cfg.expected_unique_lines:
        raise ValueError(
            f"{len(parts['unique'])} distinct lines after removing copies, expected "
            f"{cfg.expected_unique_lines}"
        )
    splits = {name: to_rows(parts[name]) for name in SPLITS}
    for name in SPLITS:
        bad = [i for i, r in enumerate(splits[name]) if row_problems(r)]
        if bad:
            raise ValueError(
                f"{name}: {len(bad)} invalid rows, e.g. row {bad[0]}: "
                f"{row_problems(splits[name][bad[0]])}"
            )
        if len(splits[name]) != cfg.expected_rows[name]:
            raise ValueError(
                f"{name} has {len(splits[name])} lines, expected {cfg.expected_rows[name]}; "
                "check the revision and data.yaml"
            )
    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "repo_id": cfg.dataset.repo_id,
            "revision": cfg.dataset.revision,
            "license": cfg.dataset.license,
        },
        "lines": {
            "published": len(lines),
            "distinct": len(parts["unique"]),
            "shop_groups": int(parts["unique"].group.nunique()),
            "held_out_shop_groups": len(parts["held_out_groups"]),
        },
        "rows": {n: len(splits[n]) for n in SPLITS},
        "labels": {
            n: {c: sum(r["category"] == c for r in splits[n]) for c in CATEGORIES} for n in SPLITS
        },
        "overlap": overlap_report(splits),
        "format": {
            "mlx": "MLX-LM chat JSONL: system prompt, user = line, assistant = category",
            "rows": "JSONL {line, category, shop}",
        },
        "files": {},
    }
    if dry_run:
        log.info("Dry run: nothing written")
        return manifest
    targets: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for name in SPLITS:
        targets[name] = (cfg.output_dir / f"{name}.jsonl", splits[name])
    for name in ("train", "valid"):
        records = [chat_record(r["line"], r["category"]) for r in splits[name]]
        targets[f"mlx_{name}"] = (cfg.mlx_dir / f"{name}.jsonl", records)
    for path, records in targets.values():
        write_jsonl(path, records)
    manifest["files"] = {
        name: {"path": display(path, root), "sha256": sha256_file(path), "lines": len(records)}
        for name, (path, records) in targets.items()
    }
    write_json(cfg.output_dir / MANIFEST, manifest)
    return manifest


def first_per_category(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """The first ``n`` rows of every category, in file order."""
    seen: Counter[str] = Counter()
    out = []
    for r in rows:
        if seen[r["category"]] < n:
            seen[r["category"]] += 1
            out.append(r)
    return out


def prepare_smoke_subset(cfg: DataConfig, root: Path) -> dict[str, Any]:
    """The head of the canonical files, written to the smoke dirs."""
    sources = {
        "mlx_train": cfg.mlx_dir / "train.jsonl",
        "mlx_valid": cfg.mlx_dir / "valid.jsonl",
        **{name: cfg.output_dir / f"{name}.jsonl" for name in TEST_SPLITS},
    }
    missing = [str(p) for p in sources.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Canonical prepared data not found ({', '.join(missing)}). Run `make data-prepare`."
        )
    s = cfg.smoke
    subsets = {
        "mlx_train": (
            cfg.smoke_mlx_dir / "train.jsonl",
            read_jsonl(sources["mlx_train"])[: s.train_size],
        ),
        "mlx_valid": (
            cfg.smoke_mlx_dir / "valid.jsonl",
            read_jsonl(sources["mlx_valid"])[: s.valid_size],
        ),
    }
    for name in TEST_SPLITS:
        rows = first_per_category(read_jsonl(sources[name]), s.test_per_category)
        subsets[name] = (cfg.smoke_dir / f"{name}.jsonl", rows)
    files = {}
    for name, (target, rows) in subsets.items():
        write_jsonl(target, rows)
        files[name] = {"path": display(target, root), "lines": len(rows)}
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "smoke-subset",
        "derived_from": display(cfg.output_dir / MANIFEST, root),
        "files": files,
    }
    write_json(cfg.smoke_dir / MANIFEST, manifest)
    return manifest


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

"""Row-level and file-level validation of the prepared data."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bank_budget.task import CATEGORIES, SYSTEM_PROMPT

MAX_REPORTED_ERRORS = 20
_SIGNS = ("[debit]", "[credit]")


def row_problems(row: Any) -> list[str]:
    """Problems with one ``{line, category, shop}`` row; empty when usable."""
    if not isinstance(row, dict):
        return [f"row is {type(row).__name__}, expected an object"]
    problems = []
    line = row.get("line")
    if not isinstance(line, str) or not line.strip():
        problems.append("'line' must be a non-empty string")
    elif not line.startswith(_SIGNS):
        problems.append("'line' must start with [debit] or [credit]")
    if row.get("category") not in CATEGORIES:
        problems.append(f"'category' {row.get('category')!r} is not one of the 17 categories")
    if not isinstance(row.get("shop"), str) or not row["shop"]:
        problems.append("'shop' must be a non-empty string")
    return problems


def chat_line_problems(obj: Any) -> list[str]:
    """Problems with one MLX-LM chat line: system prompt, line, category."""
    if not isinstance(obj, dict) or not isinstance(obj.get("messages"), list):
        return ["line must be an object with a 'messages' list"]
    msgs = obj["messages"]
    if len(msgs) != 3:
        return [f"expected 3 messages (system, user, assistant), found {len(msgs)}"]
    problems = []
    for msg, role in zip(msgs, ("system", "user", "assistant"), strict=True):
        if not isinstance(msg, dict) or msg.get("role") != role:
            problems.append(f"expected a '{role}' message")
        elif not isinstance(msg.get("content"), str) or not msg["content"].strip():
            problems.append(f"'{role}' content must be a non-empty string")
    if problems:
        return problems
    if msgs[0]["content"] != SYSTEM_PROMPT:
        problems.append("system message differs from the task's system prompt")
    if not msgs[1]["content"].startswith(_SIGNS):
        problems.append("user content must start with [debit] or [credit]")
    if msgs[2]["content"] not in CATEGORIES:
        problems.append(f"assistant content {msgs[2]['content']!r} is not a category")
    return problems


@dataclass
class FileReport:
    path: Path
    lines: int = 0
    errors: list[str] = field(default_factory=list)
    error_count: int = 0

    @property
    def ok(self) -> bool:
        return self.error_count == 0 and self.lines > 0

    def add(self, line_no: int, problem: str) -> None:
        self.error_count += 1
        if len(self.errors) < MAX_REPORTED_ERRORS:
            self.errors.append(f"line {line_no}: {problem}")


def validate_jsonl(path: Path, check: Callable[[Any], list[str]]) -> FileReport:
    """Validate every line of a JSONL file with ``check``."""
    report = FileReport(path)
    if not path.is_file():
        report.add(0, "file does not exist")
        return report
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            report.lines += 1
            if not line.strip():
                report.add(line_no, "blank line")
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                report.add(line_no, f"invalid JSON ({exc.msg})")
                continue
            for problem in check(obj):
                report.add(line_no, problem)
    if report.lines == 0:
        report.add(0, "file is empty")
    return report


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of objects."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def overlap_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, int]]:
    """Identical lines and shared shop groups between each pair of splits."""
    names = list(splits)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            lines = {r["line"] for r in splits[a]} & {r["line"] for r in splits[b]}
            shops = {r["shop"] for r in splits[a]} & {r["shop"] for r in splits[b]}
            out[f"{a}/{b}"] = {"lines": len(lines), "shops": len(shops)}
    return out

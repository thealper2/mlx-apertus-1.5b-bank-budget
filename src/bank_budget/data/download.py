"""Fetch the dataset's CSV file from the Hugging Face Hub (cached, revision-pinned)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bank_budget.config import DatasetSource

log = logging.getLogger(__name__)
COLUMNS = ("description", "category")


class DataAccessError(RuntimeError):
    """The dataset could not be downloaded or read."""


def download_file(source: DatasetSource, local_files_only: bool = False) -> Path:
    """Local path of the dataset file, downloading it only if not cached (~3.5 MB)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=source.repo_id,
            filename=source.file,
            repo_type="dataset",
            revision=source.revision,
            local_files_only=local_files_only,
        )
    except LocalEntryNotFoundError as exc:
        raise DataAccessError(
            "The dataset is not in the local Hugging Face cache and downloading is disabled "
            "or the Hub is unreachable. Run `make data-download` with network access."
        ) from exc
    except EntryNotFoundError as exc:
        raise DataAccessError(f"{source.file} not found in {source.repo_id}") from exc
    except Exception as exc:  # network / HTTP errors from the Hub client
        raise DataAccessError(f"Could not download {source.repo_id}: {exc}") from exc
    return Path(path)


def read_lines(path: Path) -> Any:
    """The ``description`` and ``category`` columns as a pandas DataFrame, in file order."""
    import pandas as pd

    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise DataAccessError(f"Could not read {path.name}: {exc}") from exc
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise DataAccessError(f"{path.name} lacks column(s) {missing}")
    return frame[list(COLUMNS)]


def load_lines(source: DatasetSource, local_files_only: bool = False) -> Any:
    """All dataset lines (as published, duplicates included)."""
    log.info("Dataset %s @ %s", source.repo_id, (source.revision or "main")[:12])
    return read_lines(download_file(source, local_files_only))

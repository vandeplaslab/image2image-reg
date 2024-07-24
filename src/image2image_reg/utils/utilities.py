"""Utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from shutil import rmtree

from loguru import logger

if ty.TYPE_CHECKING:
    pass


def _safe_delete(file_: Path) -> None:
    if not file_.exists():
        return
    if file_.is_dir():
        try:
            rmtree(file_)
            logger.trace(f"Deleted directory {file_}.")
        except Exception as e:
            logger.error(f"Could not delete {file_}. {e}")
    else:
        try:
            file_.unlink()
            logger.trace(f"Deleted file {file_}.")
        except Exception as e:
            logger.error(f"Could not delete {file_}. {e}")

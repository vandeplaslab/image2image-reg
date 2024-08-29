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


def make_new_name(
    src_name: str, ref_name: str | None = None, project_name: str | None = None, suffix: str = ".ome.tiff"
) -> str:
    """Make a new name for a file based on a reference name."""
    src_name = src_name.replace("_registered", "")
    if ref_name:
        ref_name = ref_name.replace("_registered", "")
        src_name = src_name.replace(ref_name, "")  # ensure that name doesn't contain reference name
    if ref_name and ref_name != src_name:
        new_name = f"{src_name}_to_{ref_name}"
    else:
        new_name = f"{src_name}"
    if project_name:
        new_name = f"{project_name}_{new_name}"
    new_name += f"_registered{suffix}"
    return new_name

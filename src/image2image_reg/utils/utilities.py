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
    if src_name.startswith("_to_"):
        src_name = src_name[4:]
    if ref_name and ref_name != src_name:
        new_name = f"{src_name}_to_{ref_name}"
    else:
        new_name = f"{src_name}"
    if project_name:
        new_name = f"{project_name}_{new_name}"
    new_name += f"_registered{suffix}"
    if new_name.startswith("_to_"):
        new_name = new_name[4:]
    return new_name


def print_versions() -> None:
    """Print versions."""
    from koyo.utilities import get_version, is_installed

    logger.info(f"image2image-io version: {get_version('image2image-io')}")
    logger.info(f"image2image-reg version: {get_version('image2image-reg')}")
    if is_installed("valis-wsi"):
        logger.info(f"valis version: {get_version('valis-wsi')}")

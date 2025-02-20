"""Utilities."""

from __future__ import annotations

import typing as ty
from copy import deepcopy
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


def update_kwargs_on_channel_names(search_names: list[str], **kwargs: ty.Any) -> tuple[bool, dict]:
    """Update keyword arguments based on input."""
    if "channel_names" not in kwargs:
        return False, kwargs
    search_names = [se.lower() for se in search_names]
    channel_names_ = deepcopy(kwargs["channel_names"])
    channel_names_ = [ch.lower() for ch in channel_names_]
    indices = []
    # perform basic check, if search name has same name as channel name
    for ch in search_names:
        if ch in channel_names_:
            indices.append(channel_names_.index(ch))
    # alternatively, check whether search name is part of channel name - this can of course produce a number of
    # false positives...
    if not indices:
        for chn in channel_names_:
            for ch in search_names:
                if ch in chn:
                    indices.append(channel_names_.index(chn))
    if indices:
        kwargs["channel_indices"] = indices
        return True, kwargs
    return False, kwargs

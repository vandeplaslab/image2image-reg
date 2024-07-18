"""Click utilities."""

from __future__ import annotations

import sys

import click
from koyo.click import cli_parse_paths_sort
from koyo.typing import PathLike
from loguru import logger

# declare common options
ALLOW_EXTRA_ARGS = {"help_option_names": ["-h", "--help"], "ignore_unknown_options": True, "allow_extra_args": True}
overwrite_ = click.option(
    "-W",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=None,
    show_default=True,
)
as_uint8_ = click.option(
    "-u/-U",
    "--as_uint8/--no_as_uint8",
    help="Downcast the image data format to uint8 which will substantially reduce the size of the files (unless it's"
    " already in uint8...).",
    is_flag=True,
    default=None,
    show_default=True,
)
original_size_ = click.option(
    "-o/-O",
    "--original_size/--no_original_size",
    help="Write images in their original size after applying transformations.",
    is_flag=True,
    default=True,
    show_default=True,
)
remove_merged_ = click.option(
    "-g/-G",
    "--remove_merged/--no_remove_merged",
    help="Remove written images that have been merged.",
    is_flag=True,
    default=True,
    show_default=True,
)
write_merged_ = click.option(
    "-m/-M",
    "--write_merged/--no_write_merged",
    help="Write merge images. Nothing will happen if merge modalities have not been specified.",
    is_flag=True,
    default=True,
    show_default=True,
)
write_not_registered_ = click.option(
    "-n/-N",
    "--write_not_registered/--no_write_not_registered",
    help="Write not-registered images.",
    is_flag=True,
    default=False,
    show_default=True,
)
write_registered_ = click.option(
    "-r/-R",
    "--write_registered/--no_write_registered",
    help="Write registered images.",
    is_flag=True,
    default=True,
    show_default=True,
)
fmt_ = click.option(
    "-f",
    "--fmt",
    help="Output format.",
    type=click.Choice(["ome-tiff"], case_sensitive=False),
    default="ome-tiff",
    show_default=True,
    required=False,
)
write_ = click.option(
    "-w/-W",
    "--write/--no_write",
    help="Write images to disk.",
    is_flag=True,
    default=True,
    show_default=True,
)
n_parallel_ = click.option(
    "-j",
    "--n_parallel",
    help="How many actions should be taken simultaneously.",
    type=click.IntRange(1, 24, clamp=True),
    show_default=True,
    default=1,
)
parallel_mode_ = click.option(
    "-P",
    "--parallel_mode",
    help="Parallel mode.",
    type=click.Choice(["inner", "outer"], case_sensitive=False),
    default="outer",
    show_default=True,
    required=False,
)
project_path_multi_ = click.option(
    "-p",
    "--project_dir",
    help="Path to the project directory. It usually ends in .i2reg extension.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
project_path_single_ = click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .i2reg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)


# noinspection PyUnusedLocal
def arg_split_bbox(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [int(arg.strip()) for arg in value.split(",")]
    assert len(args) == 4, "Bounding box must have 4 values"
    return args


def get_preprocessing(preprocessing: str | None, affine: str | None = None) -> Preprocessing | None:
    """Get a pre-processing object."""
    from image2image_reg.models import Preprocessing

    if preprocessing in ["dark", "fluorescence"]:
        pre = Preprocessing.fluorescence()
    elif preprocessing in ["light", "brightfield"]:
        pre = Preprocessing.brightfield()
    elif preprocessing in ["basic"]:
        pre = Preprocessing.basic()
    else:
        pre = None
    if pre and affine:
        if affine and not Path(affine).suffix == ".json":
            raise ValueError("Affine must be a JSON file.")
        # will be automatically converted to a numpy array
        pre.affine = affine  # type: ignore[assignment]
    return pre


def set_logger(verbosity: float, no_color: bool, log: PathLike | None = None) -> None:
    """Setup logger."""
    from koyo.logging import get_loguru_config, set_loguru_env, set_loguru_log

    level = verbosity * 10
    level, fmt, colorize, enqueue = get_loguru_config(level, no_color=no_color)  # type: ignore[assignment]
    set_loguru_env(fmt, level, colorize, enqueue)  # type: ignore[arg-type]
    set_loguru_log(level=level.upper(), no_color=no_color, logger=logger)  # type: ignore[attr-defined]
    logger.enable("image2image_reg")
    logger.enable("image2image_io")
    logger.enable("koyo")
    # override koyo logger
    set_loguru_log(level=level.upper(), no_color=no_color)  # type: ignore[attr-defined]
    logger.debug(f"Activated logger with level '{level}'.")
    if log:
        set_loguru_log(
            log,
            level=level.upper(),  # type: ignore[attr-defined]
            enqueue=enqueue,
            colorize=False,
            no_color=True,
            catch=True,
            diagnose=True,
            logger=logger,
            remove=False,
        )
        logger.trace(f"Command: {' '.join(sys.argv)}")

"""Click utilities."""

from __future__ import annotations

import sys
import typing as ty
from pathlib import Path

import click
from koyo.click import cli_parse_paths_sort
from koyo.typing import PathLike
from loguru import logger

if ty.TYPE_CHECKING:
    from image2image_reg.models import Preprocessing

# declare common options
ALLOW_EXTRA_ARGS = {"help_option_names": ["-h", "--help"], "ignore_unknown_options": True, "allow_extra_args": True}
overwrite_ = click.option(
    "-O",
    "--overwrite",
    help="Overwrite existing data.",
    is_flag=True,
    default=False,
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
    default=False,
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
write_attached_ = click.option(
    "-a/-A",
    "--write_attached/--no_write_attached",
    help="Write attached modalities.",
    is_flag=True,
    default=False,
    show_default=True,
)
write_attached_shapes_ = click.option(
    "write_attached_shapes",
    "--was/--no_was",
    help="Write attached shapes. If the argument is used, it will overwrite the `write_attached` option.",
    is_flag=True,
    default=None,
    show_default=True,
)
write_attached_points_ = click.option(
    "write_attached_points",
    "--wap/--no_wap",
    help="Write attached points. If the argument is used, it will overwrite the `write_attached` option.",
    is_flag=True,
    default=None,
    show_default=True,
)
write_attached_images_ = click.option(
    "write_attached_images",
    "--wai/--no_wai",
    help="Write attached images. If the argument is used, it will overwrite the `write_attached` option.",
    is_flag=True,
    default=None,
    show_default=True,
)
write_registered_ = click.option(
    "-r/-R",
    "--write_registered/--no_write_registered",
    help="Write registered images.",
    is_flag=True,
    default=False,
    show_default=True,
)
write_ = click.option(
    "-w/-W",
    "--write/--no_write",
    help="Write images to disk.",
    is_flag=True,
    default=True,
    show_default=True,
)
rename_ = click.option(
    "--rename/--no_rename",
    help="Rename modalities (e.g. imageA -> imageA_to_imageB).",
    is_flag=True,
    default=False,
    show_default=True,
)
clip_ = click.option(
    "--clip",
    help="Clip points/shapes outside of the image area.",
    type=click.Choice(["ignore", "clip", "part-remove", "remove"], case_sensitive=False),
    default="ignore",
    show_default=True,
    required=False,
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
modality_multi_ = click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=True,
    required=True,
)
modality_single_ = click.option(
    "-n",
    "--name",
    help="Name of the image (modality).",
    type=click.STRING,
    show_default=True,
    required=True,
)
image_ = click.option(
    "-i",
    "--image",
    help="Path to the image(s) that should be co-registered.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
files_ = click.option(
    "-f",
    "--files",
    help="Path to the image, shape (e.g. GeoJSON) or point files.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
output_dir_ = click.option(
    "-o",
    "--output_dir",
    help="Output directory where the project should be saved to.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
output_dir_current_ = click.option(
    "-o",
    "--output_dir",
    help="Output directory where the project should be saved to.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    default=".",
    show_default=True,
    required=True,
)
pixel_size_opt_ = click.option(
    "-s",
    "--pixel_size",
    help="Pixel size in micrometers.",
    type=click.FLOAT,
    show_default=True,
    required=False,
    default=None,
)
attach_to_ = click.option(
    "-a",
    "--attach_to",
    help="Name of the modality to which the attachment should be added.",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
attach_image_ = click.option(
    "-i",
    "--image",
    help="Path to image file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
attach_points_ = click.option(
    "-f",
    "--file",
    help="Path to GeoJSON file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
attach_shapes_ = click.option(
    "-f",
    "--file",
    help="Path to GeoJSON file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)


# noinspection PyUnusedLocal
def arg_split_bbox(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [int(arg.strip()) for arg in value.split(",")]
    assert len(args) == 4, "Bounding box must have 4 values"
    return args


def get_preprocessing(
    preprocessing: str | None, affine: str | None = None, method: str | None = None
) -> Preprocessing | None:
    """Get a pre-processing object."""
    from image2image_reg.models import Preprocessing

    if preprocessing in ["dark", "fluorescence"]:
        pre = Preprocessing.fluorescence()
    elif preprocessing in ["light", "brightfield"]:
        pre = Preprocessing.brightfield()
    elif preprocessing in ["basic"]:
        pre = Preprocessing.basic()
    elif preprocessing in ["postaf"]:
        pre = Preprocessing.postaf()
    elif preprocessing in ["pas"]:
        pre = Preprocessing.pas()
    elif preprocessing in ["he"]:
        pre = Preprocessing.he()
    elif preprocessing in ["mip"]:
        pre = Preprocessing.dapi()
    else:
        pre = None
    if pre and affine:
        if affine and not Path(affine).suffix == ".json":
            raise ValueError("Affine must be a JSON file.")
        # will be automatically converted to a numpy array
        pre.affine = affine  # type: ignore[assignment]
    if method == "auto":
        if preprocessing in ["dark", "fluorescence"]:
            pre.method = "MaxIntensityProjection"
        elif preprocessing in ["light", "brightfield"]:
            pre.method = "ColorfulStandardizer"
    elif method is not None:
        pre.method = method
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

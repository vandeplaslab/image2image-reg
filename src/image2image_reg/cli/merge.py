"""Merge command."""

from __future__ import annotations

import typing as ty

import click
from image2image_io.cli._common import arg_split_bbox, as_uint8_, fmt_, overwrite_
from image2image_io.enums import WriterMode
from koyo.click import Parameter, arg_parse_framelist_multi, cli_parse_paths_sort, print_parameters
from koyo.timer import MeasureTimer
from loguru import logger


@overwrite_
@as_uint8_
@fmt_
@click.option(
    "-C",
    "--channel_ids",
    type=click.STRING,
    default=None,
    help="Specify channel ids in the format: 1,2,4-6. You can provide multiple. If you are providing any, make sure to"
    " provide one for each file you are trying to merge.",
    callback=arg_parse_framelist_multi,
    show_default=True,
    multiple=True,
    required=False,
)
@click.option(
    "-b",
    "--crop_bbox",
    help="Bound box to be used for cropping of the image(s). It must be supplied in the format: x,y,width,height and"
    " be in PIXEL units. It will throw an error if fewer or more than 4 values are supplied.",
    type=click.UNPROCESSED,
    show_default=True,
    required=False,
    callback=arg_split_bbox,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to images to be merged.",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    show_default=True,
    required=True,
)
@click.option(
    "-p",
    "--path",
    help="Path to images to be merged.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name of the merged image.",
    type=click.STRING,
    show_default=True,
    required=True,
)
@click.command("merge")
def merge(
    name: str,
    path: ty.Sequence[str],
    output_dir: str,
    crop_bbox: tuple[int, int, int, int] | None,
    channel_ids: ty.Sequence[tuple] | None,
    fmt: WriterMode,
    as_uint8: bool | None,
    overwrite: bool,
) -> None:
    """Export images."""
    merge_runner(name, path, output_dir, crop_bbox, channel_ids, fmt, as_uint8, overwrite)


def merge_runner(
    name: str,
    paths: ty.Sequence[str],
    output_dir: str,
    crop_bbox: tuple[int, int, int, int] | None,
    channel_ids: ty.Sequence[tuple] | None,
    fmt: WriterMode = "ome-tiff",
    as_uint8: bool | None = False,
    overwrite: bool = False,
) -> None:
    """Register images."""
    from image2image_reg.workflows.merge import merge as merge_images

    print_parameters(
        Parameter("Name", "-n/--name", name),
        Parameter("Image paths", "-p/--path", paths),
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Crop bounding box", "-b/--crop_bbox", crop_bbox),
        Parameter("Channel ids", "-C/--channel_ids", channel_ids),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    if channel_ids:
        if len(channel_ids) != len(paths):
            raise ValueError("Number of channel ids must match number of images.")

    with MeasureTimer() as timer:
        merge_images(
            name, list(paths), output_dir, crop_bbox, fmt, as_uint8, channel_ids=channel_ids, overwrite=overwrite
        )
    logger.info(f"Finished processing project in {timer()}.")

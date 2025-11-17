"""Merge images."""

from __future__ import annotations

from pathlib import Path

from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_reg.models import BoundingBox


def merge(
    name: str,
    paths: list[PathLike],
    output_dir: PathLike,
    crop_bbox: tuple[int, int, int, int] | None,
    fmt: str = "ome-tiff",
    as_uint8: bool | None = None,
    channel_ids: list[int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Merge multiple images."""
    from image2image_io.models.merge import MergeImages
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers.merge_tiff_writer import MergeOmeTiffWriter
    # from image2image_reg.wrapper import ImageWrapper

    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir)

    crop_bbox = BoundingBox(*crop_bbox) if crop_bbox else None  # type: ignore[assignment]

    reader_names = []
    pixel_sizes = []
    channel_names = []
    image_shapes = []
    with MeasureTimer() as timer:
        for path_ in paths:
            path = Path(path_)
            reader = get_simple_reader(path)
            reader_names.append(reader.name)
            pixel_sizes.append(reader.resolution)
            channel_names.append(reader.channel_names)
            image_shapes.append(reader.image_shape)
    logger.info(f"Loaded {len(reader_names)} images in {timer()}.")
    image_shape = image_shapes[0]
    crop_mask = crop_bbox.to_mask(image_shape) if crop_bbox else None  # type: ignore[attr-defined]
    if crop_bbox:
        crop_bbox.to_file(name, output_dir, image_shape)  # type: ignore[attr-defined]
    output_path = output_dir / f"{name}_merged-registered"
    merge_obj = MergeImages(paths, pixel_sizes, channel_names=channel_names)
    writer = MergeOmeTiffWriter(merge_obj, crop_mask=crop_mask)
    with MeasureTimer() as timer:
        writer.merge_write_image_by_plane(
            output_path.name,
            reader_names,
            output_dir=output_dir,
            as_uint8=as_uint8,
            channel_ids=channel_ids,
        )
    logger.info(f"Merged images in {timer()}.")
    return path

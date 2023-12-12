"""Merge images."""
from __future__ import annotations

from pathlib import Path

from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_wsireg.models import Modality


def merge(name: str, paths: list[PathLike], output_dir: PathLike, fmt: str) -> Path:
    """Merge multiple images."""
    from image2image_io.readers.merge import MergeImages
    from image2image_io.writers.merge_tiff_writer import MergeOmeTiffWriter

    from image2image_wsireg.wrapper import ImageWrapper

    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir)

    with MeasureTimer() as timer:
        sub_images = []
        pixel_sizes = []
        channel_names = []
        for path_ in paths:
            path = Path(path_)
            modality = Modality(name=path.name, path=path)
            sub_images.append(modality.name)
            wrapper = ImageWrapper(modality)
            pixel_sizes.append(wrapper.reader.resolution)
            channel_names.append(wrapper.reader.channel_names)
    logger.info(f"Loaded {len(sub_images)} images in {timer()}.")

    output_path = output_dir / f"{name}_merged-registered"
    merge_obj = MergeImages(paths, pixel_sizes, channel_names=channel_names)
    writer = MergeOmeTiffWriter(merge_obj)
    with MeasureTimer() as timer:
        writer.merge_write_image_by_plane(output_path.name, sub_images, output_dir=output_dir)
    logger.info(f"Merged images in {timer()}.")
    return path

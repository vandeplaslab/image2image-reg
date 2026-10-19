"""Memory-bounded resampling of 2D readers into a physical output grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from image2image_io.readers import BaseReader


def resample_region(
    reader: BaseReader,
    transform: sitk.Transform,
    channel: int,
    *,
    origin: tuple[float, float],
    spacing: tuple[float, float],
    region: tuple[int, int, int, int],
    max_source_bytes: int = 64 * 1024 * 1024,
) -> np.ndarray:
    """Resample a YX region using an output-to-input physical XY transform.

    ``region`` is (y, x, height, width); ``origin`` and ``spacing`` use XY order.
    Source crops include the linear interpolation halo. Nonlinear bounds use all
    output sample positions, rather than corners, to preserve curved interiors.
    """
    y, x, height, width = region
    if min(height, width) <= 0 or min(spacing) <= 0 or max_source_bytes <= 0:
        message = "Region size, spacing and memory limit must be positive."
        raise ValueError(message)
    dtype = np.dtype(reader.dtype)
    output_origin = (origin[0] + x * spacing[0], origin[1] + y * spacing[1])
    field = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat64,
        [width, height],
        output_origin,
        spacing,
    )
    coords = sitk.GetArrayFromImage(field)
    coords[..., 0] += output_origin[0] + np.arange(width) * spacing[0]
    coords[..., 1] += output_origin[1] + np.arange(height)[:, None] * spacing[1]
    coords /= reader.resolution
    if not np.all(np.isfinite(coords)):
        message = "Transformation produced non-finite source coordinates."
        raise ValueError(message)
    low = np.floor(coords.min(axis=(0, 1))).astype(np.int64) - 1
    high = np.ceil(coords.max(axis=(0, 1))).astype(np.int64) + 2
    source_height, source_width = reader.image_shape
    left, top = max(0, int(low[0])), max(0, int(low[1]))
    right, bottom = min(source_width, int(high[0])), min(source_height, int(high[1]))
    if right <= left or bottom <= top:
        return np.zeros((height, width), dtype=dtype)
    # Account for both the materialized NumPy crop and its SimpleITK copy.
    if (right - left) * (bottom - top) * dtype.itemsize * 2 > max_source_bytes:
        if height == width == 1:
            message = "Source interpolation footprint exceeds the memory limit."
            raise MemoryError(message)
        result = np.empty((height, width), dtype=dtype)
        if height >= width:
            split = height // 2
            regions = [(y, x, split, width), (y + split, x, height - split, width)]
        else:
            split = width // 2
            regions = [(y, x, height, split), (y, x + split, height, width - split)]
        del coords, field
        for part_y, part_x, part_h, part_w in regions:
            result[part_y - y : part_y - y + part_h, part_x - x : part_x - x + part_w] = resample_region(
                reader,
                transform,
                channel,
                origin=origin,
                spacing=spacing,
                region=(part_y, part_x, part_h, part_w),
                max_source_bytes=max_source_bytes,
            )
        return result
    pixels = reader.get_channel(channel, split_rgb=True)[top:bottom, left:right]
    if hasattr(pixels, "compute"):
        pixels = pixels.compute()
    image = sitk.GetImageFromArray(np.asarray(pixels))
    image.SetSpacing((reader.resolution, reader.resolution))
    image.SetOrigin((left * reader.resolution, top * reader.resolution))
    resampled = sitk.Resample(
        image,
        [width, height],
        transform,
        sitk.sitkLinear,
        output_origin,
        spacing,
        [1.0, 0.0, 0.0, 1.0],
        0.0,
        image.GetPixelID(),
    )
    return sitk.GetArrayFromImage(resampled)

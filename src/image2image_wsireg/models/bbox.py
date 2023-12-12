"""Bounding box module."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from loguru import logger


class BoundingBox:
    """Bounding box named tuple."""

    x: int
    y: int
    width: int
    height: int

    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Return dict."""
        if as_wsireg:
            return {"X:": self.x, "Y": self.y, "Width": self.width, "Height": self.height}
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def to_mask(self, image_shape: tuple[int, int], dtype: type = bool, value: bool | int = True) -> np.ndarray:
        """Return mask."""
        mask: np.ndarray = np.zeros(image_shape, dtype=dtype)
        if self.x + self.width > image_shape[1]:
            self.width = image_shape[1] - self.x
            logger.trace(f"Bounding box width exceeds image width. Setting width to {self.width}")
        if self.y + self.height > image_shape[0]:
            self.height = image_shape[0] - self.y
            logger.trace(f"Bounding box height exceeds image height. Setting height to {self.height}")
        mask[self.y : self.y + self.height, self.x : self.x + self.width] = value
        return mask

    def to_sitk_image(self, image_shape: tuple[int, int], pixel_size: float = 1.0) -> sitk.Image:
        """Return image."""
        mask = self.to_mask(image_shape, dtype=np.uint8, value=255)
        image = sitk.GetImageFromArray(mask, isVector=False)
        image.SetSpacing((pixel_size, pixel_size))  # type: ignore[no-untyped-call]
        return image


def _transform_to_bbox(mask_bbox: tuple[int, int, int, int] | list[int]) -> BoundingBox:
    """Transform to bounding box."""
    return BoundingBox(*mask_bbox)

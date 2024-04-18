"""Bounding box module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from koyo.typing import PathLike
from loguru import logger


class MaskMixin:
    """Mask mixin."""

    mask_type: str

    def to_mask(self, image_shape: tuple[int, int], dtype: type = bool, value: bool | int = True) -> np.ndarray:
        """Return mask."""
        raise NotImplementedError("Must implement method")

    def to_sitk_image(self, image_shape: tuple[int, int], pixel_size: float = 1.0) -> sitk.Image:
        """Return image."""
        mask = self.to_mask(image_shape, dtype=np.uint8, value=255)
        image = sitk.GetImageFromArray(mask, isVector=False)
        image.SetSpacing((pixel_size, pixel_size))  # type: ignore[no-untyped-call]
        return image

    def to_file(self, name: str, output_dir: PathLike, image_shape: tuple[int, int]) -> Path:
        """Save bounding box to file."""
        from tifffile import imwrite

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mask = self.to_mask(image_shape, dtype=np.uint8, value=255)
        imwrite(output_dir / f"{name}_{self.mask_type}.tiff", mask, compression="deflate")
        return output_dir / f"{name}_{self.mask_type}.tiff"


class Polygon(MaskMixin):
    """Polygon where data is in yx format."""

    xy: np.ndarray
    mask_type: str = "polygon"

    def __init__(self, xy: np.ndarray):
        self.xy = xy

    def to_dict(self, as_wsireg: bool = False) -> list:
        """Return dict."""
        return self.xy.tolist()  # type: ignore[no-any-return]

    def to_mask(self, image_shape: tuple[int, int], dtype: type = bool, value: bool | int = True) -> np.ndarray:
        """Return mask."""
        import cv2

        dtype = np.uint8
        mask = np.zeros(image_shape, dtype=dtype)
        mask = cv2.fillPoly(mask, pts=[self.xy.astype(np.int32)], color=np.iinfo(dtype).max)
        return mask


def _transform_to_polygon(v: np.ndarray) -> Polygon:
    """Transform to bounding box."""
    if v is None:
        return None
    if isinstance(v, list):
        v = np.array(v)
        return Polygon(v)
    elif isinstance(v, Polygon):
        return v
    return Polygon(v)


class BoundingBox(MaskMixin):
    """Bounding box named tuple."""

    x: int
    y: int
    width: int
    height: int
    mask_type: str = "bbox"

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


def _transform_to_bbox(v: tuple[int, int, int, int] | list[int]) -> BoundingBox:
    """Transform to bounding box."""
    if v is None:
        return None
    if isinstance(v, dict):
        return BoundingBox(**v)
    elif isinstance(v, (list, tuple)):
        v = list(v)
        assert len(v) == 4, "Bounding box must have 4 values"
        return BoundingBox(*v)
    elif isinstance(v, BoundingBox):
        return v
    return BoundingBox(*v)

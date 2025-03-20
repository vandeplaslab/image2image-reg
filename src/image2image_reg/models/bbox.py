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


def _transform_to_polygon(v: np.ndarray) -> Polygon:
    """Transform to bounding box."""
    if v is None:
        return None
    if isinstance(v, list):
        assert len(v) > 0, "Polygon must have at least 1 value"
        if isinstance(v[0], (list, np.ndarray)):
            v = [np.array(v_) for v_ in v]
        else:
            v = [np.array(v)]
        return Polygon(v)
    elif isinstance(v, Polygon):
        return v
    return Polygon(v)


class Polygon(MaskMixin):
    """Polygon where data is in yx format."""

    xy: np.ndarray
    mask_type: str = "polygon"

    def __init__(self, xy: np.ndarray | list[np.ndarray]):
        self.xy = xy if isinstance(xy, list) else [xy]

    def to_dict(self, as_wsireg: bool = False) -> list:
        """Return dict."""
        if isinstance(self.xy, list):
            return [xy.tolist() for xy in self.xy]
        return self.xy.tolist()  # type: ignore[no-any-return]

    def to_mask(self, image_shape: tuple[int, int], dtype: type = bool, value: bool | int = True) -> np.ndarray:
        """Return mask."""
        import cv2

        dtype = np.uint8
        mask = np.zeros(image_shape, dtype=dtype)
        if isinstance(self.xy, list):
            for xy in self.xy:
                mask = cv2.fillPoly(mask, pts=[xy.astype(np.int32)], color=np.iinfo(dtype).max)
            return mask
        mask = cv2.fillPoly(mask, pts=[self.xy.astype(np.int32)], color=np.iinfo(dtype).max)
        return mask

    def as_str(self) -> str:
        """Stringify bounding box."""
        return f"Polygon({len(self.xy)})"


class BoundingBox(MaskMixin):
    """Bounding box named tuple."""

    mask_type: str = "bbox"

    def __init__(self, x: int | list[int], y: int | list[int], width: int | list[int], height: int | list[int]):
        self.x = x if isinstance(x, list) else [x]
        self.y = y if isinstance(y, list) else [y]
        self.width = width if isinstance(width, list) else [width]
        self.height = height if isinstance(height, list) else [height]
        assert len(self.x) == len(self.y) == len(self.width) == len(self.height), "Bounding box must have 4 values"

    def __repr__(self) -> str:
        """Repr."""
        return f"{self.__class__.__name__}<(x={self.x}, y={self.y}, width={self.width}, height={self.height}>"

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Return dict."""
        if as_wsireg:
            return {"X:": self.x, "Y": self.y, "Width": self.width, "Height": self.height}
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    def to_mask(self, image_shape: tuple[int, int], dtype: type = bool, value: bool | int = True) -> np.ndarray:
        """Return mask."""
        mask: np.ndarray = np.zeros(image_shape, dtype=dtype)
        for index in range(len(self.x)):
            mask = self._draw_bbox(mask, index, image_shape, value)
        return mask

    def _draw_bbox(self, mask: np.ndarray, index: int, image_shape: tuple[int, int], value: bool | int) -> np.ndarray:
        """Draw bounding box."""
        if self.x[index] + self.width[index] > image_shape[1]:
            self.width[index] = image_shape[1] - self.x[index]
            logger.trace(f"Bounding box width exceeds image width. Setting width to {self.width[index]}")
        if self.y[index] + self.height[index] > image_shape[0]:
            self.height[index] = image_shape[0] - self.y[index]
            logger.trace(f"Bounding box height exceeds image height. Setting height to {self.height[index]}")
        mask[self.y[index] : self.y[index] + self.height[index], self.x[index] : self.x[index] + self.width[index]] = (
            value
        )
        return mask

    def as_str(self) -> str:
        """Stringify bounding box."""
        if len(self.x) == 1:
            return f"Bbox({self.x[0]}, {self.y[0]}, {self.width[0]}, {self.height[0]})"
        return f"Bbox({self.x}, {self.y}, {self.width}, {self.height})"


def _transform_to_bbox(v: tuple[int, int, int, int] | list[int]) -> BoundingBox:
    """Transform to bounding box."""
    if v is None:
        return None
    if isinstance(v, dict):
        return BoundingBox(**v)
    elif isinstance(v, (list, tuple)):
        if isinstance(v, tuple):
            v = list(v)
        assert len(v) > 0, "Bounding box must have at least 1 value"
        if isinstance(v[0], (int, float)):
            return BoundingBox(*v)
        x = [v_[0] for v_ in v]
        y = [v_[1] for v_ in v]
        width = [v_[2] for v_ in v]
        height = [v_[3] for v_ in v]
        return BoundingBox(x, y, width, height)
    elif isinstance(v, BoundingBox):
        return v
    return BoundingBox(*v)

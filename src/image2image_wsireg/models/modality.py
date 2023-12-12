"""Modality."""
import typing as ty

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from pydantic import BaseModel, validator

from image2image_wsireg.enums import ArrayLike
from image2image_wsireg.models.bbox import BoundingBox
from image2image_wsireg.models.preprocessing import Preprocessing


class Modality(BaseModel):
    """Modality."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    name: str
    path: ty.Union[PathLike, np.ndarray, da.core.Array, zarr.Array]
    preprocessing: ty.Optional[Preprocessing] = None
    channel_names: ty.Optional[list[str]] = None
    channel_colors: ty.Optional[list[str]] = None
    pixel_size: float = 1.0
    mask: ty.Optional[ty.Union[PathLike, np.ndarray]] = None
    mask_bbox: ty.Optional[BoundingBox] = None
    output_pixel_size: ty.Optional[tuple[float, float]] = None

    @validator("mask_bbox", pre=True)
    def _validate_bbox(cls, v) -> BoundingBox:
        if isinstance(v, dict):
            return BoundingBox(**v)
        elif isinstance(v, (list, tuple)):
            v = list(v)
            assert len(v) == 4, "Bounding box must have 4 values"
            return BoundingBox(*v)
        elif isinstance(v, BoundingBox):
            return v
        return None

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Convert to dict."""
        data = self.dict(exclude_none=True, exclude_defaults=False)
        if data.get("preprocessing"):
            if isinstance(data["preprocessing"], Preprocessing):
                data["preprocessing"] = data["preprocessing"].to_dict(as_wsireg)
        if isinstance(data["path"], ArrayLike):
            data["path"] = "ArrayLike"
        if data.get("mask"):
            if isinstance(data["mask"], ArrayLike):
                data["mask"] = "ArrayLike"
        if data.get("mask_bbox"):
            data["mask_bbox"] = data["mask_bbox"].to_dict()
        if as_wsireg:
            if data.get("path"):
                data["image_filepath"] = data.pop("path")
            if data.get("pixel_size"):
                data["image_res"] = data.pop("pixel_size")
            if data.get("output_pixel_size"):
                data["output_res"] = data.pop("output_pixel_size")
        return data

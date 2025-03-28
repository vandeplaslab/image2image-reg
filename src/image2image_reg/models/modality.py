"""Modality."""

import typing as ty

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from pydantic import BaseModel, Field, field_validator

from image2image_reg.enums import ArrayLike
from image2image_reg.models.export import Export
from image2image_reg.models.preprocessing import Preprocessing

# TODO: move mask/mask_bbox/mask_polygon/transform_mask to pre-processing
if ty.TYPE_CHECKING:
    from image2image_reg.wrapper import ImageWrapper


class Modality(BaseModel):
    """Modality."""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        validate_assignment = True

    name: str
    path: ty.Union[PathLike, np.ndarray, da.core.Array, zarr.Array]
    preprocessing: ty.Optional[Preprocessing] = None
    export: ty.Optional[Export] = None
    channel_names: ty.Optional[list[str]] = None
    channel_colors: ty.Optional[list[str]] = None
    pixel_size: float = 1.0
    output_pixel_size: ty.Optional[tuple[float, float]] = None
    reader_kws: ty.Optional[dict[str, ty.Any]] = Field(None)

    @field_validator("output_pixel_size", mode="before")
    @classmethod
    def _validate_output_pixel_size(cls, value) -> ty.Optional[tuple[float, float]]:
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("output_pixel_size should be a tuple of 2 floats")
            return float(value[0]), float(value[1])
        if isinstance(value, (int, float)):
            return float(value), float(value)
        return None

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Convert to dict."""
        data = self.model_dump(exclude_none=True, exclude_defaults=False)
        if data.get("preprocessing"):
            if isinstance(data["preprocessing"], Preprocessing):
                data["preprocessing"] = data["preprocessing"].to_dict(as_wsireg)
            else:
                pre = {}
                for k, v in data["preprocessing"].items():
                    pre[k] = v.to_dict(as_wsireg) if hasattr(v, "to_dict") else v
                data["preprocessing"] = pre
        if data.get("export"):
            if isinstance(data["export"], Export):
                data["export"] = data["export"].to_dict()
        if isinstance(data["path"], ArrayLike):
            data["path"] = "ArrayLike"

        # if export for wsireg, let's remove all extra components and rename few attributes
        if as_wsireg:
            if data.get("path"):
                data["image_filepath"] = data.pop("path")
            if data.get("pixel_size"):
                data["image_res"] = data.pop("pixel_size")
            if data.get("output_pixel_size"):
                data["output_res"] = data.pop("output_pixel_size")
            if data.get("export"):
                data.pop("export")
        return data

    def to_wrapper(self) -> "ImageWrapper":
        """Convert to ImageWrapper."""
        from image2image_reg.wrapper import ImageWrapper

        return ImageWrapper(self)

    def is_masked(self) -> bool:
        """Return if masked."""
        return self.preprocessing.is_masked()

    def is_cropped(self) -> bool:
        """Return if masked."""
        return self.preprocessing.is_cropped()

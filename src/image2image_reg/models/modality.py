"""Modality."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from copy import deepcopy

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from pydantic import BaseModel, ConfigDict, Field, field_validator

from image2image_reg.models.export import Export
from image2image_reg.models.preprocessing import Preprocessing

# TODO: move mask/mask_bbox/mask_polygon/transform_mask to pre-processing
if ty.TYPE_CHECKING:
    from image2image_reg.wrapper import ImageWrapper


class Modality(BaseModel):
    """Modality."""

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    name: str
    path: PathLike | np.ndarray | da.Array | zarr.Array
    preprocessing: Preprocessing | None = None
    export: Export | None = None
    channel_names: list[str] | None = None
    channel_colors: list[str] | None = None
    pixel_size: float = 1.0
    output_pixel_size: tuple[float, float] | None = None
    reader_kws: dict[str, ty.Any] | None = Field(None)

    @field_validator("output_pixel_size", mode="before")
    @classmethod
    def _validate_output_pixel_size(cls, value) -> tuple[float, float] | None:
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
        if data.get("export") and isinstance(data["export"], Export):
            data["export"] = data["export"].to_dict()
        if not isinstance(data["path"], (str, Path)):
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

    def to_wrapper(self) -> ImageWrapper:
        """Convert to ImageWrapper."""
        from image2image_reg.wrapper import ImageWrapper

        return ImageWrapper(self)

    def is_masked(self) -> bool:
        """Return if masked."""
        return self.preprocessing.is_masked()

    def is_cropped(self) -> bool:
        """Return if masked."""
        return self.preprocessing.is_cropped()

    def auto_set_preprocessing(self, preprocessing_func: ty.Callable[..., Preprocessing], **kwargs) -> None:
        """Auto set preprocessing."""
        preprocessing = deepcopy(self.preprocessing)
        self.preprocessing = preprocessing_func(channel_names=self.channel_names, **kwargs)
        if preprocessing:
            self.preprocessing.mask = preprocessing.mask
            self.preprocessing.mask_bbox = preprocessing.mask_bbox
            self.preprocessing.mask_polygon = preprocessing.mask_polygon
            self.preprocessing.crop_bbox = preprocessing.crop_bbox
            self.preprocessing.crop_polygon = preprocessing.crop_polygon
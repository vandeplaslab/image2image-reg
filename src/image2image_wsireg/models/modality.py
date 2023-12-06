"""Modality."""
import typing as ty

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from pydantic import BaseModel

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
    output_pixel_size: ty.Optional[tuple[float, float]] = None

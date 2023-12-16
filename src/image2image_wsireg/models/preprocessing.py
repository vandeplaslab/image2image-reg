"""Preprocessing parameters for image2image registration."""
import typing as ty
from enum import Enum
from pathlib import Path

import numpy as np
from koyo.json import read_json_data
from pydantic import BaseModel, validator

from image2image_wsireg.enums import CoordinateFlip, ImageType
from image2image_wsireg.models.bbox import BoundingBox, _transform_to_bbox


def _index_to_list(ch_indices: ty.Union[int, list[int]]) -> list[int]:
    """Convert index to list."""
    if isinstance(ch_indices, int):
        ch_indices = [ch_indices]
    return ch_indices


def _transform_custom_proc(
    custom_procs: ty.Union[list[ty.Callable], tuple[ty.Callable, ...]]
) -> dict[str, ty.Callable]:
    """Transform custom processing."""
    return {f"custom processing {str(idx+1).zfill(2)}": proc for idx, proc in enumerate(custom_procs)}


class Preprocessing(BaseModel):
    """Preprocessing parameter model.

    Attributes
    ----------
    image_type: ImageType
        Whether image is dark or light background. Light background images are intensity inverted
        by default
    max_intensity_projection: bool
        Perform max intensity projection number of channels > 1.
    contrast_enhance: bool
        Enhance contrast of image
    channel_indices: list of int or int
        Channel indicies to use for registration, 0-index, so ch_indices = 0, pulls the first channel
    as_uint8: bool
        Whether to byte scale registration image data for memory saving
    invert_intensity: bool
        invert the intensity of an image
    rotate_counter_clockwise: int, float
        Rotate image counter-clockwise by degrees, can be positive or negative (cw rot)
    flip: CoordinateFlip, default: None
        flip coordinates, "v" = vertical flip, "h" = horizontal flip
    crop_to_bbox: bool
        Convert a binary mask to a bounding box and crop to this area
    crop_bbox: tuple or list of 4 ints
        supply a pre-computed list of bbox info of form x,y,width,height
    downsample: int
        Downsampling by integer factor, i.e., downsampling = 3, downsamples image 3x
    use_mask: bool
        Whether to use mask in elastix registration. At times it is better to use the mask to find a cropping area
        then use the mask during the registration process as errors are frequent
    custom_processing: callable
        Custom intensity preprocessing functions in a dict like {"my_custom_process: custom_func} that will be applied
        to the image. Must take in an sitk.Image and return an sitk.Image
    """

    class Config:
        """Pydantic config."""

        use_enum_names = True
        arbitrary_types_allowed = True

    # intensity preprocessing
    image_type: ImageType = ImageType.DARK
    max_intensity_projection: bool = True
    channel_indices: ty.Optional[list[int]] = None
    as_uint8: bool = True
    contrast_enhance: bool = False
    invert_intensity: bool = False
    custom_processing: ty.Optional[dict[str, ty.Callable]] = None

    # spatial preprocessing
    affine: ty.Optional[np.ndarray] = None
    rotate_counter_clockwise: float = 0
    flip: ty.Optional[CoordinateFlip] = None
    crop_to_bbox: bool = False
    crop_bbox: ty.Optional[BoundingBox] = None
    downsample: int = 1
    use_mask: bool = True

    def __init__(self, **kwargs: ty.Any):
        if "max_int_proj" in kwargs:
            kwargs["max_intensity_projection"] = kwargs.pop("max_int_proj")
        if "ch_indices" in kwargs:
            kwargs["channel_indices"] = kwargs.pop("ch_indices")
        if "rotate_cc" in kwargs:
            kwargs["rotate_counter_clockwise"] = kwargs.pop("rotate_cc")
        if "crop_to_mask_bbox" in kwargs:
            kwargs["crop_to_bbox"] = kwargs.pop("crop_to_mask_bbox")
        if "mask_bbox" in kwargs:
            kwargs["crop_bbox"] = kwargs.pop("mask_bbox")
        if "downsampling" in kwargs:
            kwargs["downsample"] = kwargs.pop("downsampling")
        super().__init__(**kwargs)

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Return dict."""
        data = self.dict(exclude_none=True, exclude_defaults=True)
        if data.get("affine"):
            data["affine"] = data["affine"].tolist()
        if data.get("crop_bbox"):
            data["crop_bbox"] = data["crop_bbox"].to_dict(as_wsireg)
        if as_wsireg:
            if data.get("channel_indices"):
                data["ch_indices"] = data.pop("channel_indices")
            if data.get("crop_bbox"):
                data["mask_bbox"] = data.pop("crop_bbox")
            if data.get("downsample"):
                data["downsampling"] = data.pop("downsample")
            if data.get("rotate_counter_clockwise"):
                data["rotate_cc"] = data.pop("rotate_counter_clockwise")
            if data.get("max_intensity_projection"):
                data["max_int_proj"] = data.pop("max_intensity_projection")
        return data

    @classmethod
    def basic(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(image_type=ImageType.DARK, as_uint8=True, max_int_proj=True)

    @classmethod
    def fluorescence(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=ImageType.DARK,
            as_uint8=True,
            max_int_proj=True,
            contrast_enhance=True,
        )

    @classmethod
    def brightfield(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=ImageType.LIGHT,
            as_uint8=True,
            max_int_proj=False,
            invert_intensity=True,
        )

    @validator("crop_bbox", pre=True)
    def _make_bbox(cls, v):
        if v is None:
            return None
        return _transform_to_bbox(v)

    @validator("channel_indices", pre=True)
    def _make_ch_list(cls, v):
        return _index_to_list(v)

    @validator("custom_processing", pre=True)
    def _check_custom_prepro(cls, v):
        if isinstance(v, (list, tuple)):
            return _transform_custom_proc(v)
        return v

    @validator("affine", pre=True)
    def _check_affine(cls, v):
        if v is not None:
            if isinstance(v, (str, Path)):
                v = read_json_data(Path(v))
            v = np.asarray(v)
            assert v.ndim == 2, "affine must be 2D"
            assert v.shape[0] == v.shape[1], "affine must be square"
            assert v.shape[0] == 3, "affine must be 3x3"
        return v

    def dict(self, **kwargs: ty.Any) -> dict:
        """Convert to dict."""
        output = super().dict(**kwargs)
        for k, v in output.items():
            if isinstance(v, Enum):
                output[k] = v.value
        return output

"""Preprocessing parameters for image2image registration."""

import typing as ty
from enum import Enum
from pathlib import Path

import numpy as np
from koyo.json import read_json_data
from koyo.typing import PathLike
from pydantic import BaseModel, validator

from image2image_reg.enums import ArrayLike, CoordinateFlip, ImageType
from image2image_reg.models.bbox import BoundingBox, Polygon, _transform_to_bbox, _transform_to_polygon


def _index_to_list(ch_indices: ty.Union[int, list[int]]) -> list[int]:
    """Convert index to list."""
    if isinstance(ch_indices, (int, str)):
        ch_indices = [ch_indices]
    return ch_indices


def _transform_custom_proc(
    custom_procs: ty.Union[list[ty.Callable], tuple[ty.Callable, ...]],
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
        Channel indicies to use for registration, 0-index, so ch_indices = 0, pulls the first channel. It's ignored
        if image is RGB.
    channel_names: list of str or str
        Channel names to use for registration, if channel_indices is not supplied, this will be used.
        If channel_indices is supplied, this will be ignored. It's ignored if image is RGB.
    as_uint8: bool
        Whether to byte scale registration image data for memory saving
    invert_intensity: bool
        invert the intensity of an image
    affine: np.ndarray
        Affine transformation matrix (3x3) to apply to image. Slightly broken so please avoid for the time being.
    rotate_counter_clockwise: int, float
        Rotate image counter-clockwise by degrees, can be positive or negative (cw rot)
    flip: CoordinateFlip, default: None
        flip coordinates, "v" = vertical flip, "h" = horizontal flip
    translate_x: int
        Translate image in x direction (specify in physical units, eg. microns)
    translate_y: int
        Translate image in y direction (specify in physical units, eg. microns)
    use_crop: bool
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
        validate_assignment = True

    # intensity preprocessing
    image_type: ImageType = ImageType.DARK
    max_intensity_projection: bool = True
    equalize_histogram: bool = False
    contrast_enhance: bool = False
    invert_intensity: bool = False
    channel_indices: ty.Optional[list[int]] = None
    channel_names: ty.Optional[list[str]] = None
    as_uint8: bool = True
    custom_processing: ty.Optional[dict[str, ty.Callable]] = None

    # spatial preprocessing
    affine: ty.Optional[np.ndarray] = None
    rotate_counter_clockwise: float = 0
    flip: ty.Optional[CoordinateFlip] = None
    translate_x: int = 0
    translate_y: int = 0
    downsample: int = 1

    # crop pre-processing
    use_crop: bool = False
    crop_bbox: ty.Optional[BoundingBox] = None
    crop_polygon: ty.Optional[Polygon] = None

    # mask pre-processing
    transform_mask: bool = False
    use_mask: bool = True
    mask: ty.Optional[ty.Union[PathLike, np.ndarray]] = None
    mask_bbox: ty.Optional[BoundingBox] = None
    mask_polygon: ty.Optional[Polygon] = None

    # valis-only
    method: ty.Optional[str] = None

    def __init__(self, **kwargs: ty.Any):
        if "max_int_proj" in kwargs:
            kwargs["max_intensity_projection"] = kwargs.pop("max_int_proj")
        if "ch_indices" in kwargs:
            kwargs["channel_indices"] = kwargs.pop("ch_indices")
        if "rotate_cc" in kwargs:
            kwargs["rotate_counter_clockwise"] = kwargs.pop("rotate_cc")
        if "crop_to_mask_bbox" in kwargs:
            kwargs["use_crop"] = kwargs.pop("crop_to_mask_bbox")
        # if "mask_bbox" in kwargs:
        #     kwargs["crop_bbox"] = kwargs.pop("mask_bbox")
        if "downsampling" in kwargs:
            kwargs["downsample"] = kwargs.pop("downsampling")
        super().__init__(**kwargs)

    def is_cropped(self) -> bool:
        """Return if cropped."""
        return self.use_crop and (self.crop_bbox is not None or self.crop_polygon is not None)

    def is_masked(self) -> bool:
        """Return if masked."""
        return self.use_mask and (self.mask is not None or self.mask_bbox is not None or self.mask_polygon is not None)

    def as_str(self) -> tuple[str, str]:
        """Create nice formatting based on pre-processing."""
        text = f"{self.image_type.value}; "
        tooltip = f"Image type: {self.image_type.value}\n"
        if self.max_intensity_projection:
            text += "MIP; "
            tooltip += "Max intensity projection\n"
        if self.equalize_histogram:
            text += "equalize; "
            tooltip += "Histogram equalization\n"
        if self.contrast_enhance:
            text += "enhance; "
            tooltip += "Contrast enhancement\n"
        if self.invert_intensity:
            text += "invert"
            tooltip += "Invert intensity\n"
        if text.endswith("; "):
            text = text[:-2]
        text += "\n"
        if self.channel_indices:
            text += f"ids: {self.channel_indices}\n"
            tooltip += f"Channel indices: {self.channel_indices}\n"
        if self.flip:
            text += f"flip-{self.flip.value}; "
            tooltip += f"Flip: {self.flip.value}\n"
        if self.translate_x or self.translate_y:
            text += f"translate({self.translate_x}, {self.translate_y}); "
            tooltip += f"Translate: ({self.translate_x}, {self.translate_y})\n"
        if self.rotate_counter_clockwise:
            text += f"{self.rotate_counter_clockwise}°"
            tooltip += f"Rotate: {self.rotate_counter_clockwise}°\n"
        if text.endswith("; "):
            text = text[:-2]
        if not text.endswith("\n"):
            text += "\n"
        if self.downsample > 1:
            text += f"x{self.downsample} downsample"
            tooltip += f"Downsample: x{self.downsample}\n"
        if self.is_masked():
            text += "mask was specified"
            tooltip += "Using mask during registration\n"
        if self.is_cropped():
            text += "crop was specified"
            tooltip += "Cropping image\n"
        if tooltip.endswith("\n"):
            tooltip = tooltip[:-1]
        return text, tooltip

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Return dict."""
        data = self.dict(exclude_none=True, exclude_defaults=True)
        if data.get("affine"):
            data["affine"] = data["affine"].tolist()
        if data.get("crop_bbox") and hasattr(data["crop_bbox"], "to_dict"):
            data["crop_bbox"] = data["crop_bbox"].to_dict(as_wsireg)
        if data.get("crop_polygon") and hasattr(data["crop_polygon"], "to_dict"):
            data["crop_polygon"] = data["crop_polygon"].to_dict(as_wsireg)
        if data.get("mask"):
            if isinstance(data["mask"], ArrayLike):
                data["mask"] = "ArrayLike"
        if data.get("mask_bbox") and hasattr(data["mask_bbox"], "to_dict"):
            data["mask_bbox"] = data["mask_bbox"].to_dict(as_wsireg)
        if data.get("mask_polygon") and hasattr(data["mask_polygon"], "to_dict"):
            data["mask_polygon"] = data["mask_polygon"].to_dict(as_wsireg)
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
            for key in [
                "use_crop",
                "crop_polygon",
                "translate_x",
                "translate_y",
                "channel_names",
                "use_mask",
                "mask",
                "mask_bbox",
                "mask_polygon",
                "transform_mask",
            ]:
                if data.get(key):
                    data.pop(key)
        return data

    @classmethod
    def basic(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(image_type=ImageType.DARK, as_uint8=True, max_intensity_projection=True)

    @classmethod
    def fluorescence(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=ImageType.DARK,
            as_uint8=True,
            max_intensity_projection=True,
            contrast_enhance=True,
        )

    @classmethod
    def brightfield(cls) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=ImageType.LIGHT,
            as_uint8=True,
            max_intensity_projection=False,
            invert_intensity=True,
        )

    @validator("mask_bbox", "crop_bbox", pre=True)
    def _validate_bbox(cls, v) -> ty.Optional[BoundingBox]:
        return _transform_to_bbox(v)

    @validator("mask_polygon", "crop_polygon", pre=True)
    def _validate_polygon(cls, v) -> ty.Optional[Polygon]:
        return _transform_to_polygon(v)

    @validator("channel_indices", "channel_names", pre=True)
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

    @validator("rotate_counter_clockwise", pre=True)
    def _validate_rotate_counter_clockwise(cls, v):
        if v == 360:
            v = 0
        if v > 360:
            v = v % 360
        return v

    def dict(self, **kwargs: ty.Any) -> dict:
        """Convert to dict."""
        output = super().dict(**kwargs)
        for k, v in output.items():
            if isinstance(v, Enum):
                output[k] = v.value
            elif isinstance(v, (BoundingBox, Polygon)):
                output[k] = v.to_dict()
        return output

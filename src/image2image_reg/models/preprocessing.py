"""Preprocessing parameters for image2image registration."""

import typing as ty
from enum import Enum
from pathlib import Path

import numpy as np
from koyo.json import read_json_data
from koyo.typing import PathLike
from pydantic import BaseModel, field_validator

from image2image_reg.enums import ArrayLike, CoordinateFlip, ImageType
from image2image_reg.models.bbox import BoundingBox, Polygon, _transform_to_bbox, _transform_to_polygon
from image2image_reg.utils.utilities import update_kwargs_on_channel_names


def _index_to_list(ch_indices: ty.Union[int, list[int]]) -> list[int]:
    """Convert index to list."""
    if isinstance(ch_indices, (int, str)):
        ch_indices = [ch_indices]
    return ch_indices


def _transform_custom_proc(
    custom_procs: ty.Union[list[ty.Callable], tuple[ty.Callable, ...]],
) -> dict[str, ty.Callable]:
    """Transform custom processing."""
    return {f"custom processing {str(idx + 1).zfill(2)}": proc for idx, proc in enumerate(custom_procs)}


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
    # method: ty.Optional[ty.Union[str, tuple[str, dict]]] = None

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

    def to_valis(self) -> tuple[str, dict]:
        """Return valis."""
        if self.method == "I2RegPreprocessor":
            return self.method, self.model_dump()
        elif self.method == "ColorfulStandardizer":
            return "ColorfulStandardizer", {"c": 0.2, "h": 0}
        elif self.method in ["mip", "MaxIntensityProjection"]:
            return "MaxIntensityProjection", {"channel_names": self.channel_names}
        elif self.method == "ChannelGetter":
            return "ChannelGetter", {"channel": "dapi"}
        return self.method, {}

    def as_str(self, valis: bool = False) -> tuple[str, str]:
        """Create nice formatting based on pre-processing."""
        text = f"{self.image_type.value}; "
        tooltip = f"Image type: {self.image_type.value}\n"
        if valis and self.method:
            text += f"{self.method}; "
            tooltip += f"Method: {self.method}\n"
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
        if self.flip and not valis:
            text += f"flip-{self.flip.value}; "
            tooltip += f"Flip: {self.flip.value}\n"
        if (self.translate_x or self.translate_y) and not valis:
            text += f"translate({self.translate_x}, {self.translate_y}); "
            tooltip += f"Translate: ({self.translate_x}, {self.translate_y})\n"
        if self.rotate_counter_clockwise and not valis:
            text += f"{self.rotate_counter_clockwise}°"
            tooltip += f"Rotate: {self.rotate_counter_clockwise}°\n"
        if text.endswith("; "):
            text = text[:-2]
        if not text.endswith("\n"):
            text += "\n"
        if self.downsample > 1 and not valis:
            text += f"x{self.downsample} downsample\n"
            tooltip += f"Downsample: x{self.downsample}\n"
        if self.is_masked():
            text += "masked"
            if self.mask_bbox:
                text += f" ({self.mask_bbox.as_str()})"
            if self.mask_polygon:
                text += f" ({self.mask_polygon.as_str()})"
            text += "\n"
            tooltip += "Using mask during registration\n"
        if self.is_cropped():
            text += "cropped"
            if self.crop_bbox:
                text += f" ({self.crop_bbox.as_str()})"
            if self.crop_polygon:
                text += f" ({self.crop_polygon.as_str()})"
            tooltip += "Cropping image\n"
        if tooltip.endswith("\n"):
            tooltip = tooltip[:-1]
        return text, tooltip

    def to_dict(self, as_wsireg: bool = False) -> dict:
        """Return dict."""
        data = self.model_dump(exclude_none=True, exclude_defaults=True)
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

    def update_from_another(self, preprocessing: "Preprocessing") -> "Preprocessing":
        """Update from another preprocessing."""
        for key, value in preprocessing.model_dump().items():
            if value is not None:
                setattr(self, key, value)
        return self

    def select_channel(self, channel_id: ty.Optional[int] = None, channel_name: ty.Optional[str] = None) -> None:
        """Select channel."""
        if channel_name is not None:
            if channel_name in self.channel_names:
                channel_id = self.channel_indices.index(channel_name)
        if channel_id is not None:
            self.channel_indices = [channel_id]

    def select_channels(self, channel_names: list[str]) -> None:
        """Select channels."""
        channel_indices = []
        for channel_name in channel_names:
            if channel_name in self.channel_names:
                channel_indices.append(self.channel_names.index(channel_name))
        if channel_indices:
            self.channel_indices = channel_indices

    @classmethod
    def basic(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.DARK,
        as_uint8: bool = True,
        max_intensity_projection: bool = True,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @classmethod
    def fluorescence(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.DARK,
        as_uint8: bool = True,
        max_intensity_projection: bool = True,
        contrast_enhance: bool = True,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            contrast_enhance=contrast_enhance,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @classmethod
    def postaf(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.DARK,
        as_uint8: bool = True,
        max_intensity_projection: bool = True,
        which: ty.Literal["any", "brightfield", "egfp"] = "any",
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        changed = False
        if which in ["any", "brightfield"]:
            changed, kwargs = update_kwargs_on_channel_names(["bright", "brightfield"], **kwargs)
            if changed:
                kwargs["invert_intensity"] = True
                kwargs["equalize_histogram"] = True
                kwargs["contrast_enhance"] = False

        if (which == "any" and not changed) or which == "egfp":
            changed, kwargs = update_kwargs_on_channel_names(["egfp"], **kwargs)
            if changed:
                kwargs["invert_intensity"] = False
                kwargs["equalize_histogram"] = True
                kwargs["contrast_enhance"] = False

        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @classmethod
    def dapi(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.DARK,
        as_uint8: bool = True,
        max_intensity_projection: bool = True,
        equalize_histogram: bool = True,
        contrast_enhance: bool = False,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        _, kwargs = update_kwargs_on_channel_names(["dapi"], **kwargs)
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            method="I2RegPreprocessor" if valis else None,
            equalize_histogram=equalize_histogram,
            contrast_enhance=contrast_enhance,
            **kwargs,
        )

    @classmethod
    def brightfield(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.LIGHT,
        as_uint8: bool = True,
        max_intensity_projection: bool = False,
        invert_intensity: bool = True,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            invert_intensity=invert_intensity,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @classmethod
    def he(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.LIGHT,
        as_uint8: bool = True,
        max_intensity_projection: bool = False,
        invert_intensity: bool = True,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            invert_intensity=invert_intensity,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @classmethod
    def pas(
        cls,
        valis: bool = False,
        image_type: ImageType = ImageType.LIGHT,
        as_uint8: bool = True,
        max_intensity_projection: bool = False,
        invert_intensity: bool = True,
        equalize_histogram: bool = True,
        **kwargs: ty.Any,
    ) -> "Preprocessing":
        """Basic image preprocessing."""
        return cls(
            image_type=image_type,
            as_uint8=as_uint8,
            max_intensity_projection=max_intensity_projection,
            invert_intensity=invert_intensity,
            equalize_histogram=equalize_histogram,
            method="I2RegPreprocessor" if valis else None,
            **kwargs,
        )

    @field_validator("mask_bbox", "crop_bbox", mode="before")
    @classmethod
    def _validate_bbox(cls, v) -> ty.Optional[BoundingBox]:
        return _transform_to_bbox(v)

    @field_validator("mask_polygon", "crop_polygon", mode="before")
    @classmethod
    def _validate_polygon(cls, v) -> ty.Optional[Polygon]:
        return _transform_to_polygon(v)

    @field_validator("channel_indices", "channel_names", mode="before")
    @classmethod
    def _make_ch_list(cls, v):
        return _index_to_list(v)

    @field_validator("custom_processing", mode="before")
    @classmethod
    def _check_custom_prepro(cls, v):
        if isinstance(v, (list, tuple)):
            return _transform_custom_proc(v)
        return v

    @field_validator("affine", mode="before")
    @classmethod
    def _check_affine(cls, v):
        if v is not None:
            if isinstance(v, (str, Path)):
                v = read_json_data(Path(v))
            v = np.asarray(v)
            assert v.ndim == 2, "affine must be 2D"
            assert v.shape[0] == v.shape[1], "affine must be square"
            assert v.shape[0] == 3, "affine must be 3x3"
        return v

    @field_validator("rotate_counter_clockwise", mode="before")
    @classmethod
    def _validate_rotate_counter_clockwise(cls, v):
        if v == 360:
            v = 0
        if v > 360:
            v = v % 360
        return v

    def model_dump(self, **kwargs: ty.Any) -> dict:
        """Convert to dict."""
        output = super().model_dump(**kwargs)
        for k, v in output.items():
            if isinstance(v, Enum):
                output[k] = v.value
            elif isinstance(v, (BoundingBox, Polygon)):
                output[k] = v.to_dict()
        return output

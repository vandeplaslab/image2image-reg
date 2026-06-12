"""Image wrapper."""

from __future__ import annotations

import math
import typing as ty
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from image2image_io.readers import BaseReader, ShapesReader, get_simple_reader
from koyo.json import read_json_data, write_json_data
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_reg._typing import OnError
from image2image_reg.constants import DEFAULT_MAX_REGISTRATION_PIXELS
from image2image_reg.models import BoundingBox, Modality, Polygon, Preprocessing
from image2image_reg.preprocessing.convert import sitk_image_to_itk_image


def filename_with_suffix(filename: Path, extra: str, suffix: str) -> Path:
    """Return path that includes extra and suffix."""
    return filename.parent / f"{filename.stem.replace('.ome', '')}_{extra}{suffix}"


def normalize_max_registration_pixels(max_registration_pixels: int | None) -> int | None:
    """Normalize the maximum registration pixel count."""
    if max_registration_pixels is None or max_registration_pixels <= 0:
        return None
    return int(max_registration_pixels)


def registration_cache_extra(max_registration_pixels: int | None) -> str | None:
    """Return cache suffix for a registration pixel cap."""
    max_registration_pixels = normalize_max_registration_pixels(max_registration_pixels)
    if max_registration_pixels is None:
        return None
    return f"regpx={max_registration_pixels}"


def _reader_channel_count(reader: BaseReader) -> int | None:
    channel_names = getattr(reader, "channel_names", None)
    if isinstance(channel_names, list | tuple) and len(channel_names) > 1:
        return len(channel_names)
    n_channels = getattr(reader, "n_channels", None)
    if isinstance(n_channels, int) and n_channels > 1:
        return n_channels
    return None


def _spatial_axes_from_reader_shape(shape: tuple[int, ...], reader: BaseReader) -> tuple[int, int] | None:
    image_shape = getattr(reader, "image_shape", None)
    if not isinstance(image_shape, tuple | list) or len(image_shape) != 2:
        return None
    reference_height, reference_width = (int(image_shape[0]), int(image_shape[1]))
    if reference_height <= 0 or reference_width <= 0:
        return None

    best_axes: tuple[int, int] | None = None
    best_score = float("inf")
    for first_axis in range(len(shape)):
        for second_axis in range(first_axis + 1, len(shape)):
            height, width = shape[first_axis], shape[second_axis]
            if height <= 0 or width <= 0:
                continue
            height_scale = height / reference_height
            width_scale = width / reference_width
            score = abs(math.log(height_scale / width_scale))
            if height > reference_height * 1.1:
                score += 1.0
            if width > reference_width * 1.1:
                score += 1.0
            if score < best_score:
                best_axes = (first_axis, second_axis)
                best_score = score
    return best_axes if best_score < 0.25 else None


def _spatial_axes(array: ty.Any, reader: BaseReader) -> tuple[int, int]:
    shape = tuple(int(i) for i in array.shape)
    if len(shape) < 2:
        raise ValueError(f"Expected at least two image dimensions, got {shape}.")  # noqa: TRY003
    if len(shape) == 2:
        return 0, 1

    if len(shape) == 3:
        is_rgb = bool(reader.is_rgb)
        if is_rgb and shape[0] in (3, 4) and shape[-1] not in (3, 4):
            return 1, 2
        if is_rgb and shape[-1] in (3, 4) and shape[0] not in (3, 4):
            return 0, 1

        n_channels = _reader_channel_count(reader)
        if n_channels is not None:
            channel_axes = [idx for idx, size in enumerate(shape) if size == n_channels]
            if len(channel_axes) == 1:
                channel_axis = channel_axes[0]
                return tuple(idx for idx in range(3) if idx != channel_axis)  # type: ignore[return-value]

    reader_axes = _spatial_axes_from_reader_shape(shape, reader)
    if reader_axes is not None:
        return reader_axes

    if len(shape) == 3:
        if shape[-1] in (3, 4):
            return 0, 1
        return 1, 2
    return len(shape) - 2, len(shape) - 1


def _spatial_shape(array: ty.Any, reader: BaseReader) -> tuple[int, int]:
    shape = tuple(int(i) for i in array.shape)
    first_axis, second_axis = _spatial_axes(array, reader)
    return shape[first_axis], shape[second_axis]


def _scale_for_pyramid(reader: BaseReader, pyramid_index: int, shape: tuple[int, int]) -> float:
    if hasattr(reader, "scale_for_pyramid"):
        scale = reader.scale_for_pyramid(pyramid_index)
        if isinstance(scale, tuple | list):
            return float(scale[0])
        return float(scale)
    image_shape = tuple(int(i) for i in reader.image_shape)
    if not image_shape:
        return float(reader.resolution or 1.0)
    reference_height = image_shape[0]
    scale_factor = reference_height / shape[0]
    return float(reader.resolution or 1.0) * scale_factor


def _slice_spatial_axes(array: ty.Any, factor: int, reader: BaseReader) -> ty.Any:
    if factor <= 1:
        return array
    slices: list[slice] = [slice(None)] * array.ndim
    for axis in _spatial_axes(array, reader):
        slices[axis] = slice(None, None, factor)
    return array[tuple(slices)]


def _normalize_channel_axis_for_preprocessing(array: ty.Any, reader: BaseReader) -> ty.Any:
    if array.ndim != 3:
        return array
    spatial_axes = _spatial_axes(array, reader)
    channel_axes = [axis for axis in range(array.ndim) if axis not in spatial_axes]
    if len(channel_axes) != 1:
        return array
    channel_axis = channel_axes[0]
    target_axis = 2 if reader.is_rgb else 0
    if channel_axis == target_axis:
        return array
    return np.moveaxis(array, channel_axis, target_axis)


def _resample_mask_to_image(mask: sitk.Image | None, image: sitk.Image) -> sitk.Image | None:
    if mask is None or mask.GetSize() == image.GetSize():
        return mask
    transform = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    return sitk.Resample(mask, image, transform, sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def _set_registration_spacing(image: sitk.Image, pixel_size: float) -> None:
    """Set registration spacing while preserving non-spatial stack spacing."""
    spacing = [float(pixel_size)] * image.GetDimension()
    if image.GetDimension() > 2:
        spacing[2:] = [1.0] * (image.GetDimension() - 2)
    image.SetSpacing(spacing)  # type: ignore[no-untyped-call]


class ImageWrapper:
    """Wrapper around the image class to add additional functionality."""

    _reader: BaseReader | None = None

    def __init__(
        self,
        modality: Modality,
        preprocessing: Preprocessing | None = None,
        preview: bool = False,
        quick: bool = False,
        quiet: bool = False,
        on_error: OnError = "raise",
    ):
        self.modality = modality
        self.preprocessing = preprocessing
        self.preview = preview
        self.quick = quick
        self.quiet = quiet
        self.on_error = on_error

        self.image: sitk.Image | None = None
        self._mask: sitk.Image | None = None
        self.initial_transforms: list[dict] = []
        self.original_size_transform: dict | None = None

    @property
    def reader(self) -> BaseReader | None:
        """Lazy reader."""
        if self._reader is None:
            try:
                reader_kws = self.modality.reader_kws or {}
                self._reader: BaseReader = get_simple_reader(
                    self.modality.path,
                    init_pyramid=False,
                    quick=self.quick,
                    quiet=self.quiet,
                    scene_index=reader_kws.get("scene_index", None),
                )
            except Exception as exc:
                if self.on_error == "raise":
                    raise
                if self.on_error == "warn":
                    logger.warning(f"Failed to initialize reader for {self.modality.name}: {exc}")
                self._reader = None
        return self._reader

    @property
    def mask(self) -> sitk.Image | None:
        """Lazy mask."""
        preprocessing = self.preprocessing
        if preprocessing is None:
            preprocessing = self.modality.preprocessing

        if self._mask is None and preprocessing and preprocessing.use_mask:
            if preprocessing.mask is not None:
                self._mask = self.read_mask(preprocessing.mask)
            elif preprocessing.mask_bbox is not None:
                self._mask = self.make_bbox_mask(preprocessing.mask_bbox)
            elif preprocessing.mask_polygon is not None:
                self._mask = self.make_bbox_mask(preprocessing.mask_polygon)
        return self._mask

    @property
    def name(self) -> str:
        """Name of the modality."""
        return self.modality.name

    def sitk_to_itk(self, inplace: bool = False) -> tuple[sitk.Image, sitk.Image | None]:
        """Convert SimpleITK image to ITK image."""
        image = sitk_image_to_itk_image(self.image)
        mask = sitk_image_to_itk_image(self.mask) if self.mask else None
        if inplace:
            self.image = image
            self._mask = mask
        return image, mask

    def release_image_data(self) -> None:
        """Release loaded image and mask data held by the wrapper."""
        self.image = None
        self._mask = None

    def _get_capped_image(self, max_registration_pixels: int | None) -> tuple[ty.Any, float]:
        """Return a registration image array and matching pixel size."""
        if self.reader is None:
            raise ValueError(f"Could not initialize reader for {self.modality.name}.")
        max_registration_pixels = normalize_max_registration_pixels(max_registration_pixels)

        image = self.reader.pyramid[0]
        image_shape = _spatial_shape(image, self.reader)
        pixel_size = float(self.reader.resolution or self.modality.pixel_size or 1.0)
        if max_registration_pixels is None:
            return image, pixel_size

        selected_index = 0
        selected_shape = image_shape
        for idx, candidate in enumerate(self.reader.pyramid):
            candidate_shape = _spatial_shape(candidate, self.reader)
            if candidate_shape[0] * candidate_shape[1] <= max_registration_pixels:
                image = candidate
                selected_index = idx
                selected_shape = candidate_shape
                pixel_size = _scale_for_pyramid(self.reader, idx, candidate_shape)
                break
        else:
            image = self.reader.pyramid[-1]
            selected_index = len(self.reader.pyramid) - 1
            selected_shape = _spatial_shape(image, self.reader)
            pixel_size = _scale_for_pyramid(self.reader, selected_index, selected_shape)

        n_pixels = selected_shape[0] * selected_shape[1]
        if n_pixels > max_registration_pixels:
            factor = math.ceil(math.sqrt(n_pixels / max_registration_pixels))
            image = _slice_spatial_axes(image, factor, self.reader)
            selected_shape = _spatial_shape(image, self.reader)
            pixel_size *= factor

        if selected_index != 0 or selected_shape != image_shape:
            logger.info(
                f"Capped registration image for {self.modality.name} to {selected_shape} "
                f"at {pixel_size:.4f} pixel size.",
            )
        return image, pixel_size

    @staticmethod
    def preprocessing_hash(modality: Modality, preprocessing: Preprocessing | None = None) -> str:
        """Hash of the preprocessing parameters."""
        if preprocessing is None and modality.preprocessing is None:
            return hash_parameters(n_in_hash=6)
        if preprocessing:
            return hash_parameters(n_in_hash=6, **preprocessing.dict())
        return hash_parameters(n_in_hash=6, **modality.preprocessing.dict())

    @staticmethod
    def get_cache_path(
        modality: Modality,
        cache_dir: PathLike,
        extra: str | None = None,
        preprocessing: Preprocessing | None = None,
    ) -> Path:
        """Get the cache path."""
        cache_dir = Path(cache_dir)
        preprocessing_hash = ImageWrapper.preprocessing_hash(modality, preprocessing)
        filename = f"{modality.name}_hash={preprocessing_hash}.tiff"
        if extra:
            filename = f"{modality.name}-{extra}_hash={preprocessing_hash}.tiff"
        return cache_dir / filename

    def check_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None) -> bool:
        """Check the image cache."""
        filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)
        return bool(use_cache and filename.exists())

    def save_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None) -> None:
        """Save the image to the cache."""
        # TODO: this should do partial saves (e.g. image, mask, pre-processing and initial)
        if not use_cache or not self.image:
            return
        with MeasureTimer() as timer:
            # write image
            filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)
            sitk.WriteImage(self.image, str(filename), useCompression=True)
            self.write_thumbnail(self.image, filename.with_suffix(".png"))
            # write pre-processing parameters
            write_json_data(filename_with_suffix(filename, "initial", ".json"), self.initial_transforms or None)
            if self.modality.preprocessing:
                write_json_data(
                    filename_with_suffix(filename, "preprocessing", ".json"),
                    self.modality.preprocessing.dict(),
                )
            if self.preprocessing:
                write_json_data(
                    filename_with_suffix(filename, "preprocessing_override", ".json"),
                    self.preprocessing.dict(),
                )
            if self.original_size_transform:
                write_json_data(
                    filename_with_suffix(filename, "original_size_transform", ".json"),
                    self.original_size_transform,
                )
            # write mask
            if self.mask is not None:
                sitk.WriteImage(self.mask, str(filename_with_suffix(filename, "mask", ".tiff")), useCompression=True)
                self.write_thumbnail(self.mask, filename_with_suffix(filename, "mask", ".png"))
        logger.trace(f"Saved image to cache: {filename} for {self.modality.name} in {timer()}")

    @staticmethod
    def write_thumbnail(image: sitk.Image, filename: PathLike, size: int = 512) -> None:
        """Write thumbnail."""
        from image2image_reg.utils.preprocessing import create_thumbnail

        thumbnail = create_thumbnail(image, size)
        sitk.WriteImage(thumbnail, str(filename), useCompression=True)

    def load_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None) -> None:
        """Load data from cache."""
        if not use_cache:
            return
        with MeasureTimer() as timer:
            filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)
            if filename.exists():
                self.image = sitk.ReadImage(str(filename))
                self.initial_transforms = read_json_data(filename_with_suffix(filename, "initial", ".json"))
            logger.trace(f"Loaded image from cache: {filename} for {self.modality.name}")
            if filename_with_suffix(filename, "preprocessing", ".json").exists():
                self.preprocessing = Preprocessing(
                    **read_json_data(filename_with_suffix(filename, "preprocessing", ".json")),
                )
            if filename_with_suffix(filename, "original_size_transform", ".json").exists():
                self.original_size_transform = read_json_data(
                    filename_with_suffix(filename, "original_size_transform", ".json"),
                )
            if filename_with_suffix(filename, "mask", ".tiff").exists():
                self._mask = sitk.ReadImage(str(filename_with_suffix(filename, "mask", ".tiff")))
        logger.trace(f"Loaded data from cache in {timer()}")

    @classmethod
    def load_initial_transform(cls, modality: Modality, cache_dir: PathLike) -> list[dict] | None:
        """Load original size transform metadata."""
        filename = cls.get_cache_path(modality, cache_dir)
        output_filename = filename_with_suffix(filename, "initial", ".json")
        transforms = []
        if output_filename.exists():
            data = read_json_data(output_filename)
            data = [data] if isinstance(data, dict) else data
            if data:
                transforms.extend(data)
        return transforms or None

    @classmethod
    def load_original_size_transform(cls, modality: Modality, cache_dir: PathLike) -> list[dict] | None:
        """Load original size transform metadata."""
        filename = cls.get_cache_path(modality, cache_dir)
        output_filename = filename_with_suffix(filename, "original_size_transform", ".json")
        transforms = []
        if output_filename.exists():
            data = read_json_data(output_filename)
            data = [data] if isinstance(data, dict) else data
            if data:
                transforms.extend(data)
        return transforms or None

    def preprocess(self, max_registration_pixels: int | None = DEFAULT_MAX_REGISTRATION_PIXELS) -> None:
        """Pre-process image."""
        from image2image_reg.utils.preprocessing import convert_and_cast, preprocess, preprocess_dask_array

        preprocessing = self.preprocessing or self.modality.preprocessing
        transform_mask = bool(preprocessing and preprocessing.transform_mask)

        # retrieve the best registration image before materializing the array
        image, pixel_size = self._get_capped_image(max_registration_pixels)
        channel_names = self.reader.channel_names

        # pre-process image
        with MeasureTimer() as timer:
            logger.trace(f"Pre-processing image {self.modality.name} with {preprocessing}...")
            image = _normalize_channel_axis_for_preprocessing(image, self.reader)
            image = preprocess_dask_array(
                image,
                channel_names,
                is_rgb=self.reader.is_rgb,
                preprocessing=preprocessing,
            )
            logger.trace(f"Initialized from dask array in {timer()}")
            # convert and cast
            image = convert_and_cast(image, preprocessing)
            logger.trace(f"Converted and cast image in {timer(since_last=True)}")
            _set_registration_spacing(image, pixel_size)

            # if mask is not going to be transformed, then we don't need to retrieve it at this moment in time
            self.image = image
            mask = _resample_mask_to_image(self.mask, image) if transform_mask else None
            # set image
            if preprocessing:
                self.image, mask, self.initial_transforms, self.original_size_transform = preprocess(
                    image,
                    mask,
                    preprocessing,
                    pixel_size,
                    self.reader.is_rgb,
                    self.initial_transforms,
                    transform_mask=transform_mask,
                )
            else:
                self.image, mask, self.original_size_transform = image, mask, None
            # mask was not transformed so let's now retrieve it if it actually exists
            if not transform_mask:
                mask = _resample_mask_to_image(self.mask, self.image) if self.image is not None else self.mask
            # overwrite existing mask
            self._mask = mask
            logger.trace(f"Pre-processed image in {timer(since_last=True)}")

    def read_mask(self, mask: str | Path | sitk.Image | np.ndarray) -> sitk.Image:
        """Read a mask from geoJSON or a binary image.

        Parameters
        ----------
        mask: path to image/geoJSON or image
            Data to be used to make the mask, can be a path to a geoJSON
            or an image file, or a if an np.ndarray, used directly.

        Returns
        -------
        mask: sitk.Image
            Mask image with spacing/size of `reg_image`
        """
        pixel_size = self.modality.pixel_size
        if isinstance(mask, np.ndarray):
            mask = sitk.GetImageFromArray(mask)
            logger.trace(f"Loaded mask from array for {self.modality.name}")
        elif isinstance(mask, str | Path):
            if Path(mask).suffix.lower() == ".geojson":
                image_shape = self.reader.image_shape
                # pixel_size = self.reader.resolution
                mask_shapes = ShapesReader(mask)
                mask = mask_shapes.to_mask_alt(image_shape[::-1], with_index=False)
                mask = sitk.GetImageFromArray(mask)
                logger.trace(f"Loaded mask from GeoJSON for {self.modality.name}")
            else:
                mask = sitk.ReadImage(mask)
                logger.trace(f"Loaded mask from image for {self.modality.name}")
        elif isinstance(mask, sitk.Image):
            mask = mask
            logger.trace(f"Loaded mask from image for {self.modality.name}")
        else:
            raise TypeError(f"Unknown mask type: {type(mask)}")
        mask.SetSpacing((pixel_size, pixel_size))  # type: ignore[no-untyped-call]
        return mask

    def make_bbox_mask(
        self,
        bbox: BoundingBox | Polygon,
        pixel_size: float | None = None,
        image_shape: tuple[int, int] | None = None,
    ) -> sitk.Image:
        """Make mask from bounding box."""
        if image_shape is None:
            # should be height, width
            image_shape = self.image.GetSize()[::-1] if self.image else self.reader.image_shape
        if pixel_size is None:
            pixel_size = self.image.GetSpacing()[0] if self.image else self.modality.pixel_size
        mask = bbox.to_sitk_image(image_shape, pixel_size)
        mask.SetSpacing((pixel_size, pixel_size))  # type: ignore[no-untyped-call]
        kind = "bbox" if isinstance(bbox, BoundingBox) else "polygon"
        logger.trace(
            f"Loaded mask from {kind} for {self.modality.name} with shape: {image_shape} "
            f"and pixel size: {pixel_size:.2f}",
        )
        return mask

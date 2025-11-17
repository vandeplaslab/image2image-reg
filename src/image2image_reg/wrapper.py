"""Image wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from image2image_io.readers import BaseReader, ShapesReader, get_simple_reader
from koyo.json import read_json_data, write_json_data
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_reg.models import BoundingBox, Modality, Polygon, Preprocessing
from image2image_reg.preprocessing.convert import sitk_image_to_itk_image


def filename_with_suffix(filename: Path, extra: str, suffix: str) -> Path:
    """Return path that includes extra and suffix."""
    return filename.parent / f"{filename.stem.replace('.ome', '')}_{extra}{suffix}"


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
        raise_on_error: bool = True,
    ):
        self.modality = modality
        self.preprocessing = preprocessing
        self.preview = preview
        self.quick = quick
        self.quiet = quiet
        self.raise_on_error = raise_on_error

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
            except Exception as e:
                if self.raise_on_error:
                    raise e
                self._reader = None
        return self._reader

    @property
    def mask(self) -> sitk.Image | None:
        """Lazy mask."""
        preprocessing = self.preprocessing
        if preprocessing is None:
            preprocessing = self.modality.preprocessing

        if self._mask is None and preprocessing:
            if preprocessing.use_mask:
                if self.modality.preprocessing.mask is not None:
                    self._mask = self.read_mask(self.modality.preprocessing.mask)
                if self.modality.preprocessing.mask_bbox is not None:
                    self._mask = self.make_bbox_mask(self.modality.preprocessing.mask_bbox)
                elif self.modality.preprocessing.mask_polygon is not None:
                    self._mask = self.make_bbox_mask(self.modality.preprocessing.mask_polygon)
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
        modality: Modality, cache_dir: PathLike, extra: str | None = None, preprocessing: Preprocessing | None = None
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
        if use_cache and filename.exists():
            return True
        return False

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
                    filename_with_suffix(filename, "preprocessing", ".json"), self.modality.preprocessing.dict()
                )
            if self.preprocessing:
                write_json_data(
                    filename_with_suffix(filename, "preprocessing_override", ".json"), self.preprocessing.dict()
                )
            if self.original_size_transform:
                write_json_data(
                    filename_with_suffix(filename, "original_size_transform", ".json"), self.original_size_transform
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
                    **read_json_data(filename_with_suffix(filename, "preprocessing", ".json"))
                )
            if filename_with_suffix(filename, "original_size_transform", ".json").exists():
                self.original_size_transform = read_json_data(
                    filename_with_suffix(filename, "original_size_transform", ".json")
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

    def preprocess(self) -> None:
        """Pre-process image."""
        from image2image_reg.utils.preprocessing import convert_and_cast, preprocess, preprocess_dask_array

        preprocessing = self.preprocessing or self.modality.preprocessing

        # retrieve first array in the pyramid i.e. the highest resolution
        image = self.reader.pyramid[0]
        channel_names = self.reader.channel_names

        # pre-process image
        with MeasureTimer() as timer:
            logger.trace(f"Pre-processing image {self.modality.name} with {preprocessing}...")
            image = preprocess_dask_array(image, channel_names, preprocessing=preprocessing)
            logger.trace(f"Initialized from dask array in {timer()}")
            # convert and cast
            image = convert_and_cast(image, preprocessing)
            logger.trace(f"Converted and cast image in {timer(since_last=True)}")

            # if mask is not going to be transformed, then we don't need to retrieve it at this moment in time
            mask = self.mask if self.modality.preprocessing.transform_mask else None
            # set image
            if preprocessing:
                self.image, mask, self.initial_transforms, self.original_size_transform = preprocess(
                    image,
                    mask,
                    preprocessing,
                    self.reader.resolution,
                    self.reader.is_rgb,
                    self.initial_transforms,
                    transform_mask=self.modality.preprocessing.transform_mask,
                )
            else:
                self.image, mask, self.original_size_transform = image, mask, None
            # mask was not transformed so let's now retrieve it if it actually exists
            if not self.modality.preprocessing.transform_mask:
                mask = self.mask
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
        elif isinstance(mask, (str, Path)):
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
            raise ValueError(f"Unknown mask type: {type(mask)}")
        mask.SetSpacing((pixel_size, pixel_size))  # type: ignore[no-untyped-call]
        return mask

    def make_bbox_mask(
        self, bbox: BoundingBox | Polygon, pixel_size: float | None = None, image_shape: tuple[int, int] | None = None
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
            f"Loaded mask from {kind} for {self.modality.name} with shape: {image_shape} and pixel size: {pixel_size:.2f}"
        )
        return mask

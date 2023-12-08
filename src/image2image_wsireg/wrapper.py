"""Image wrapper."""
from __future__ import annotations

from pathlib import Path

import SimpleITK as sitk
from image2image_io._reader import get_simple_reader
from image2image_io.readers import BaseReader
from koyo.json import read_json_data, write_json_data
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image_wsireg.models import Modality, Preprocessing
from image2image_wsireg.utils.convert import sitk_image_to_itk_image


def filename_with_suffix(filename: Path, extra: str, suffix: str) -> Path:
    """Return path that includes extra and suffix."""
    return filename.parent / f"{filename.stem.replace('.ome', '')}_{extra}{suffix}"


class ImageWrapper:
    """Wrapper around the image class to add additional functionality."""

    def __init__(self, modality: Modality, preprocessing: Preprocessing | None = None):
        self.modality = modality
        self.preprocessing = preprocessing
        # TODO: this won't work with arrays
        self.reader: BaseReader = get_simple_reader(modality.path, init_pyramid=False)
        if modality.channel_names:
            self.reader._channel_names = modality.channel_names
        if modality.channel_colors:
            self.reader._channel_colors = modality.channel_colors

        self.image: sitk.Image | None = None
        self.mask: sitk.Image | None = None
        self.initial_transforms: list[dict] = []
        self.original_size_transform: dict | None = None

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
            self.mask = mask
        return image, mask

    @staticmethod
    def preprocessing_hash(modality: Modality, preprocessing: Preprocessing | None = None) -> str:
        """Hash of the preprocessing parameters."""
        if preprocessing is None or modality.preprocessing is None:
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
        filename = f"{modality.name}_hash={preprocessing_hash}.ome.tiff"
        if extra:
            filename = f"{modality.name}-{extra}_hash={preprocessing_hash}.ome.tiff"
        return cache_dir / filename

    def check_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None) -> bool:
        """Check the image cache."""
        filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)
        if use_cache and filename.exists():
            return True
        return False

    def save_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None) -> None:
        """Save the image to the cache."""
        if not use_cache or not self.image:
            return
        with MeasureTimer() as timer:
            # write image
            filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)
            sitk.WriteImage(self.image, str(filename), useCompression=True)
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
            if self.mask:
                sitk.WriteImage(
                    self.mask, str(filename_with_suffix(filename, "mask", ".ome.tiff")), useCompression=True
                )
        logger.trace(f"Saved image to cache: {filename} for {self.modality.name} in {timer()}")

    def load_cache(self, cache_dir: PathLike, use_cache: bool = True, extra: str | None = None):
        """Load data from cache."""
        if not use_cache:
            return
        filename = self.get_cache_path(self.modality, cache_dir, extra=extra, preprocessing=self.preprocessing)

        if filename.exists():
            self.image = sitk.ReadImage(str(filename))
            self.initial_transforms = read_json_data(filename_with_suffix(filename, "initial", ".json"))
            if filename_with_suffix(filename, "preprocessing", ".json").exists():
                self.preprocessing = Preprocessing(
                    **read_json_data(filename_with_suffix(filename, "preprocessing", ".json"))
                )
            if filename_with_suffix(filename, "original_size_transform", ".json").exists():
                self.original_size_transform = read_json_data(
                    filename_with_suffix(filename, "original_size_transform", ".json")
                )
            if filename_with_suffix(filename, "mask", ".ome.tiff").exists():
                self.mask = sitk.ReadImage(str(filename_with_suffix(filename, "mask", ".ome.tiff")))

    @classmethod
    def load_original_size_transform(cls, modality: Modality, cache_dir: PathLike) -> dict | None:
        """Load original size transform metadata."""
        filename = cls.get_cache_path(modality, cache_dir)
        if filename_with_suffix(filename, "original_size_transform", ".json").exists():
            return read_json_data(filename_with_suffix(filename, "original_size_transform", ".json"))
        return None

    def preprocess(self) -> None:
        """Pre-process image."""
        from image2image_wsireg.utils.preprocessing import convert_and_cast, preprocess, preprocess_dask_array

        # retrieve first array in the pyramid i.e. the highest resolution
        image = self.reader.pyramid[0]
        # pre-process image
        image = preprocess_dask_array(image, self.preprocessing)
        # convert and cast
        image = convert_and_cast(image, self.preprocessing)

        mask = None
        preprocessing = self.preprocessing or self.modality.preprocessing
        # set image
        if preprocessing:
            self.image, self.mask, self.initial_transforms, self.original_size_transform = preprocess(
                image, mask, preprocessing, self.reader.resolution, self.reader.is_rgb, self.initial_transforms
            )
        else:
            self.image, self.mask, self.original_size_transform = image, mask, None

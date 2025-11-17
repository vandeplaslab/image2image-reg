"""Mixin class for preprocessing."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from image2image_reg.preprocessing.convert import numpy_to_sitk_image, numpy_view_to_sitk_image, sitk_image_to_numpy


class PreprocessorMixin:
    """Mixin class."""

    array: np.ndarray | sitk.Image
    pixel_size: float

    def set_array(self, array: sitk.Image | np.ndarray, pixel_size: float) -> None:
        """Set array."""
        self.array = array
        self.pixel_size = pixel_size

    @property
    def shape(self) -> tuple[int, ...]:
        """Get image shape."""
        if isinstance(self.array, np.ndarray):
            shape = self.array.shape
        else:
            shape = numpy_view_to_sitk_image(self.array).shape
        return shape

    @property
    def original_spacing(self) -> tuple[float, ...]:
        """Get image spacing."""
        shape = self.shape
        ndim = len(shape)
        if ndim == 2:
            return self.pixel_size, self.pixel_size
        else:
            return (self.pixel_size, self.pixel_size, 1) if self.is_rgb else (1, self.pixel_size, self.pixel_size)

    @property
    def spacing(self) -> tuple[float, float]:
        """Get image spacing."""
        return self.pixel_size, self.pixel_size

    @property
    def is_rgb(self) -> bool:
        """Check if image is RGB."""
        from image2image_io.utils.utilities import guess_rgb

        return guess_rgb(self.shape)

    @property
    def is_multi_channel(self) -> bool:
        """Check if image is multichannel."""
        return not self.is_rgb and not self.is_single_channel

    @property
    def is_single_channel(self) -> bool:
        """Check if image is single-channel."""
        return len(self.shape) == 2

    def to_sitk(self) -> sitk.Image:
        """Convert to SimpleITK filter."""
        if isinstance(self.array, sitk.Image):
            self.array.SetSpacing(self.original_spacing)  # type: ignore[no-untyped-call]
            return self.array
        return numpy_to_sitk_image(self.array)

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        if isinstance(self.array, np.ndarray):
            return self.array
        return sitk_image_to_numpy(self.array)

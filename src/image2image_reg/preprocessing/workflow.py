"""Pre-processing workflow."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from image2image_reg.preprocessing.mixin import PreprocessorMixin
from image2image_reg.preprocessing.step import Preprocessor, get_preprocessor


class Workflow(PreprocessorMixin):
    """Pre-processing workflow."""

    def __init__(self, array: np.ndarray | sitk.Image, pixel_size: float, steps: list[str | Preprocessor]):
        self.array = array
        self.pixel_size = pixel_size
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate steps."""
        for step in self.steps:
            step_ = step
            if isinstance(step, str):
                step: Preprocessor = get_preprocessor(step)  # type: ignore[no-redef]
            if not isinstance(step, Preprocessor):
                raise ValueError(f"Invalid step: '{step_}'")

    def run(self, to_array: bool = False) -> np.ndarray | sitk.Image:
        """Run workflow."""
        array = self.array
        for step in self.steps:
            if isinstance(step, str):
                step = get_preprocessor(step)
            assert isinstance(step, Preprocessor), f"Invalid step: '{step}'"
            array = step(array, self.pixel_size)
        self.array = array
        if to_array:
            return self.to_array()
        return self.array

    def create_mask(self) -> np.ndarray:
        """Create binary mask for the image."""
        raise NotImplementedError("Must implement method")

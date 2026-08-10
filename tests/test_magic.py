"""Tests for interactive FFT initial-overlap helpers."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.elastix.magic import (
    MAGIC_MAX_REGISTRATION_PIXELS,
    _resample_to_common_grid,
    estimate_translation,
)
from image2image_reg.models import Modality, Preprocessing


def _feature_image() -> np.ndarray:
    """Create an asymmetric image with enough detail for phase correlation."""
    image = np.zeros((256, 256), dtype=np.float32)
    cv2.circle(image, (75, 150), 35, 0.8, -1)
    cv2.rectangle(image, (160, 45), (220, 90), 0.5, -1)
    cv2.line(image, (45, 35), (135, 55), 1.0, 5)
    return image


def _install_fake_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    images: dict[str, np.ndarray],
    spacing: float = 2.5,
) -> list[Any]:
    """Install an ImageWrapper that applies its saved physical translation."""
    wrappers = []

    class FakeWrapper:
        def __init__(self, modality: Modality, preprocessing: Preprocessing) -> None:
            self.modality = modality
            self.preprocessing = preprocessing
            self.image = None
            wrappers.append(self)

        def preprocess(self, max_registration_pixels: int) -> None:
            self.max_registration_pixels = max_registration_pixels
            matrix = np.float32(
                [
                    [1, 0, self.preprocessing.translate_x / spacing],
                    [0, 1, self.preprocessing.translate_y / spacing],
                ],
            )
            image = cv2.warpAffine(
                images[self.modality.name],
                matrix,
                images[self.modality.name].shape[::-1],
                flags=cv2.INTER_LINEAR,
            )
            self.image = sitk.GetImageFromArray(image)
            self.image.SetSpacing((spacing, spacing))

    monkeypatch.setattr("image2image_reg.elastix.magic.ImageWrapper", FakeWrapper)
    return wrappers


def test_estimate_translation_uses_snapshots_and_returns_physical_shift(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ignore moving x/y while preserving the target frame and returning its inverse shift."""
    wrappers = _install_fake_wrapper(monkeypatch, {"moving": _feature_image(), "fixed": _feature_image()})
    moving_preprocessing = Preprocessing.basic(translate_x=100, translate_y=-50)
    fixed_preprocessing = Preprocessing.basic(translate_x=25, translate_y=-15)
    moving = Modality(name="moving", path="moving.tiff", preprocessing=moving_preprocessing)
    fixed = Modality(name="fixed", path="fixed.tiff", preprocessing=fixed_preprocessing)

    result = estimate_translation(moving, fixed, moving_preprocessing, fixed_preprocessing)
    assert result == pytest.approx((-25, 15), abs=0.5)
    assert moving_preprocessing.translate_x == 100
    assert moving_preprocessing.translate_y == -50
    assert fixed_preprocessing.translate_x == 25
    assert fixed_preprocessing.translate_y == -15
    assert wrappers[0].preprocessing.translate_x == 0
    assert wrappers[0].preprocessing.translate_y == 0
    assert wrappers[1].preprocessing.translate_x == 25
    assert wrappers[1].preprocessing.translate_y == -15
    assert all(wrapper.max_registration_pixels == MAGIC_MAX_REGISTRATION_PIXELS for wrapper in wrappers)


def test_estimate_translation_propagates_requested_pixel_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the supplied cap for both preprocessing and the shared FFT grid."""
    wrappers = _install_fake_wrapper(monkeypatch, {"moving": _feature_image(), "fixed": _feature_image()})
    preprocessing = Preprocessing.basic()
    moving = Modality(name="moving", path="moving.tiff", preprocessing=preprocessing)
    fixed = Modality(name="fixed", path="fixed.tiff", preprocessing=preprocessing)

    estimate_translation(moving, fixed, preprocessing, preprocessing, max_registration_pixels=50_000)

    assert [wrapper.max_registration_pixels for wrapper in wrappers] == [50_000, 50_000]


def test_common_fft_grid_honors_pixel_cap() -> None:
    """Avoid creating a large shared grid after capped preprocessing."""
    image = sitk.GetImageFromArray(np.zeros((800, 1_200), dtype=np.float32))
    moving, fixed, _spacing = _resample_to_common_grid(image, image, max_pixels=100_000)

    assert moving.size <= 100_000
    assert fixed.shape == moving.shape


@pytest.mark.parametrize(
    "moving_image,fixed_image",
    [
        (np.zeros((256, 256), dtype=np.float32), np.zeros((256, 256), dtype=np.float32)),
        (
            np.random.default_rng(1).random((256, 256), dtype=np.float32),
            np.random.default_rng(2).random((256, 256), dtype=np.float32),
        ),
    ],
)
def test_estimate_translation_rejects_ambiguous_images(
    monkeypatch: pytest.MonkeyPatch,
    moving_image: np.ndarray,
    fixed_image: np.ndarray,
) -> None:
    """Reject flat and unrelated images instead of returning a speculative shift."""
    _install_fake_wrapper(monkeypatch, {"moving": moving_image, "fixed": fixed_image})
    preprocessing = Preprocessing.basic()
    moving = Modality(name="moving", path="moving.tiff", preprocessing=preprocessing)
    fixed = Modality(name="fixed", path="fixed.tiff", preprocessing=preprocessing)

    with pytest.raises(ValueError, match="Magic could not"):
        estimate_translation(moving, fixed, preprocessing, preprocessing)

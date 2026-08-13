"""Interactive initial-overlap helpers."""

from __future__ import annotations

import typing as ty
from copy import deepcopy

import cv2
import numpy as np
import SimpleITK as sitk

from image2image_reg.wrapper import ImageWrapper

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality, Preprocessing


MAGIC_MAX_REGISTRATION_PIXELS = 1_000_000
"""Maximum pixels used by interactive translation estimation."""

MAGIC_MIN_PHASE_RESPONSE = 0.1
"""Minimum phase-correlation response accepted for an automatic overlap."""

_ERROR_NOT_2D = "Auto-translation estimation requires two-dimensional images."
_ERROR_NO_DATA = "Auto-translation estimation could not read usable image data."
_ERROR_NO_DETAIL = "Auto-translation estimation could not find enough image detail to estimate an overlap."
_ERROR_PREPROCESS = "Auto-translation estimation could not preprocess both images."
_ERROR_LOW_CONFIDENCE = (
    "Auto-translation estimation could not confidently estimate an overlap. Try different"
    " preprocessing or align manually."
)


def _physical_bounds(image: sitk.Image) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical lower and upper image bounds."""
    size_x, size_y = image.GetSize()
    corners = np.asarray(
        [image.TransformIndexToPhysicalPoint((x, y)) for x in (0, size_x - 1) for y in (0, size_y - 1)],
        dtype=float,
    )
    return corners.min(axis=0), corners.max(axis=0)


def _resample_to_common_grid(
    moving_image: sitk.Image,
    fixed_image: sitk.Image,
    max_pixels: int,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """Resample preprocessed images to one capped physical-coordinate grid."""
    if moving_image.GetDimension() != 2 or fixed_image.GetDimension() != 2:
        raise ValueError(_ERROR_NOT_2D)

    moving_min, moving_max = _physical_bounds(moving_image)
    fixed_min, fixed_max = _physical_bounds(fixed_image)
    lower = np.minimum(moving_min, fixed_min)
    upper = np.maximum(moving_max, fixed_max)
    spacing = np.maximum(np.asarray(moving_image.GetSpacing()), np.asarray(fixed_image.GetSpacing()))
    size = np.ceil((upper - lower) / spacing).astype(int) + 1
    n_pixels = int(np.prod(size))
    if n_pixels > max_pixels:
        spacing *= np.ceil(np.sqrt(n_pixels / max_pixels))
        size = np.ceil((upper - lower) / spacing).astype(int) + 1

    reference = sitk.Image([int(size[0]), int(size[1])], sitk.sitkFloat32)
    reference.SetOrigin(tuple(lower))
    reference.SetSpacing(tuple(spacing))
    identity = sitk.Transform(2, sitk.sitkIdentity)
    moving = sitk.Resample(moving_image, reference, identity, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    fixed = sitk.Resample(fixed_image, reference, identity, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    return (
        sitk.GetArrayFromImage(moving).astype(np.float32, copy=False),
        sitk.GetArrayFromImage(fixed).astype(np.float32, copy=False),
        (float(spacing[0]), float(spacing[1])),
    )


def _phase_features(image: np.ndarray) -> np.ndarray:
    """Build normalized, windowed edge features for phase correlation."""
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        raise ValueError(_ERROR_NO_DATA)

    lower, upper = np.percentile(finite, (1, 99))
    if upper <= lower:
        raise ValueError(_ERROR_NO_DETAIL)
    normalized = np.clip((image - lower) / (upper - lower), 0.0, 1.0).astype(np.float32)
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    features = cv2.magnitude(gradient_x, gradient_y)
    std = float(features.std())
    if std < 1e-6:
        raise ValueError(_ERROR_NO_DETAIL)
    features = (features - features.mean()) / std
    window = np.outer(np.hanning(features.shape[0]), np.hanning(features.shape[1])).astype(np.float32)
    return features * window


def estimate_translation(
    moving_modality: Modality,
    fixed_modality: Modality,
    moving_preprocessing: Preprocessing,
    fixed_preprocessing: Preprocessing,
    *,
    max_registration_pixels: int = MAGIC_MAX_REGISTRATION_PIXELS,
) -> tuple[float, float]:
    """Estimate the physical translation that aligns a moving image to a fixed image.

    The moving image's current translation is intentionally ignored so the returned
    values can replace it. All other moving settings and all fixed preprocessing
    settings are preserved. A fast FFT phase-correlation heuristic is used instead
    of iterative Elastix registration. The returned x/y uses the preprocessing
    transform convention, which is the inverse of OpenCV's measured image shift.
    """
    moving_preprocessing = deepcopy(moving_preprocessing)
    moving_preprocessing.translate_x = 0
    moving_preprocessing.translate_y = 0
    fixed_preprocessing = deepcopy(fixed_preprocessing)

    moving_wrapper = ImageWrapper(moving_modality, moving_preprocessing)
    fixed_wrapper = ImageWrapper(fixed_modality, fixed_preprocessing)
    moving_wrapper.preprocess(max_registration_pixels=max_registration_pixels)
    fixed_wrapper.preprocess(max_registration_pixels=max_registration_pixels)
    if moving_wrapper.image is None or fixed_wrapper.image is None:
        raise ValueError(_ERROR_PREPROCESS)

    moving, fixed, spacing = _resample_to_common_grid(
        moving_wrapper.image,
        fixed_wrapper.image,
        max_registration_pixels,
    )
    shift, response = cv2.phaseCorrelate(_phase_features(moving), _phase_features(fixed))
    if response < MAGIC_MIN_PHASE_RESPONSE:
        raise ValueError(_ERROR_LOW_CONFIDENCE)
    return float(-shift[0] * spacing[0]), float(-shift[1] * spacing[1])

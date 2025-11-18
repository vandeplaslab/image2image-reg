"""Convert."""

from __future__ import annotations

import itk
import numpy as np
import SimpleITK as sitk


def itk_image_to_sitk_image(image: itk.ImageBase) -> sitk.Image:
    """Convert ITK image to SITK image."""
    origin = tuple(image.GetOrigin())
    spacing = tuple(image.GetSpacing())
    direction = itk.GetArrayFromMatrix(image.GetDirection()).flatten()
    image = sitk.GetImageFromArray(itk.GetArrayFromImage(image), isVector=image.GetNumberOfComponentsPerPixel() > 1)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    return image  # type: ignore[no-any-return]


def sitk_image_to_itk_image(image: sitk.Image, cast_to_float32: bool = False) -> itk.ImageBase:
    """Convert SITK image to ITK image."""
    origin = image.GetOrigin()  # type: ignore[no-untyped-call]
    spacing = image.GetSpacing()  # type: ignore[no-untyped-call]
    is_vector = image.GetNumberOfComponentsPerPixel() > 1  # type: ignore[no-untyped-call]
    if cast_to_float32:
        image = sitk.Cast(image, sitk.sitkFloat32)  # type: ignore[no-untyped-call]
        image = sitk.GetArrayFromImage(image)  # type: ignore[no-untyped-call,assignment]
    else:
        image = sitk.GetArrayFromImage(image)  # type: ignore[no-untyped-call,assignment]

    image = itk.GetImageFromArray(image, is_vector=is_vector)
    image.SetOrigin(origin)  # type: ignore[no-untyped-call]
    image.SetSpacing(spacing)  # type: ignore[no-untyped-call]
    return image


def sitk_image_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SITK image to numpy array."""
    return sitk.GetArrayFromImage(image)


def numpy_view_to_sitk_image(image: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """Convert numpy array to SITK image."""
    ndim = image.ndim
    image = sitk.GetArrayViewFromImage(image)  # type: ignore[no-untyped-call,assignment,arg-type]
    image.SetSpacing((resolution,) * ndim)  # type: ignore[no-untyped-call,attr-defined]
    return image


def numpy_to_sitk_image(image: np.ndarray, resolution: float = 1.0) -> sitk.Image:
    """Convert numpy array to SITK image."""
    ndim = image.ndim
    image = sitk.GetImageFromArray(image)  # type: ignore[no-untyped-call,assignment]
    image.SetSpacing((resolution,) * ndim)  # type: ignore[no-untyped-call,attr-defined]
    return image  # type: ignore[no-any-return]

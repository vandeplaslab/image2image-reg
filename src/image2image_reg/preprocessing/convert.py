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
    return image


def sitk_image_to_itk_image(image: sitk.Image, cast_to_float32=False) -> itk.ImageBase:
    """Convert SITK image to ITK image."""
    origin = image.GetOrigin()
    spacing = image.GetSpacing()
    # direction = image.GetDirection()
    is_vector = image.GetNumberOfComponentsPerPixel() > 1
    if cast_to_float32:
        image = sitk.Cast(image, sitk.sitkFloat32)
        image = sitk.GetArrayFromImage(image)
    else:
        image = sitk.GetArrayFromImage(image)

    image = itk.GetImageFromArray(image, is_vector=is_vector)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    return image


def sitk_image_to_numpy(image: sitk.Image) -> np.ndarray:
    """Convert SITK image to numpy array."""
    return sitk.GetArrayFromImage(image)


def numpy_view_to_sitk_image(image: sitk.Image) -> np.ndarray:
    """Convert numpy array to SITK image."""
    return sitk.GetArrayViewFromImage(image)


def numpy_to_sitk_image(image: np.ndarray) -> sitk.Image:
    """Convert numpy array to SITK image."""
    return sitk.GetImageFromArray(image)

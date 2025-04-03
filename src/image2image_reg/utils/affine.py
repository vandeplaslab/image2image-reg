"""Affine transformation methods."""

from __future__ import annotations

import numpy as np


def combined_transform(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
    rotation_angle: float | int = 0,
    translation: tuple[float, float] = (0, 0),
    flip_lr: bool = False,
) -> np.ndarray:
    """Combined transform.

    Transformations are performed in the following order:
    - translation along x/y-axis
    - rotation around the center point
    - horizontal flip
    """
    image_size = np.asarray(image_size)  # type: ignore[assignment]
    image_spacing = np.asarray(image_spacing)  # type: ignore[assignment]
    translation = np.asarray(translation)  # type: ignore[assignment]
    tran = centered_translation_transform(translation)
    rot = centered_rotation_transform(image_size, image_spacing, rotation_angle)
    flip = np.eye(3)
    if flip_lr:
        flip = centered_horizontal_flip(image_size, image_spacing)
    return tran @ rot @ flip  # type: ignore[no-any-return]


def centered_translation_transform(
    translation: tuple[float, float],
) -> np.ndarray:
    """Centered translation transform."""
    transform = np.eye(3)
    transform[:2, 2] = np.asarray(translation)
    return transform


def centered_rotation_transform(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
    rotation_angle: float | int,
) -> np.ndarray:
    """Centered rotation transform."""
    angle = np.deg2rad(rotation_angle)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)
    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]

    return rot_mat


def centered_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
    direction: str,
) -> np.ndarray:
    """Centered flip transform."""
    angle = np.deg2rad(0)

    sina = np.sin(angle)
    cosa = np.cos(angle)

    # build rot mat
    rot_mat = np.eye(3)

    rot_mat[0, 0] = cosa
    rot_mat[1, 1] = cosa
    rot_mat[1, 0] = sina
    rot_mat[0, 1] = -sina

    if direction.lower() == "vertical":
        rot_mat[0, 0] = -1
    else:
        rot_mat[1, 1] = -1

    # recenter transform
    center_point = np.multiply(image_size, image_spacing) / 2
    center_point = np.append(center_point, 0)
    translation = center_point - np.dot(rot_mat, center_point)
    rot_mat[:2, 2] = translation[:2]
    return rot_mat


def centered_vertical_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
) -> np.ndarray:
    """Centered vertical flip transform."""
    return centered_flip(image_size, image_spacing, "vertical")


def centered_horizontal_flip(
    image_size: tuple[int, int],
    image_spacing: tuple[float, float],
) -> np.ndarray:
    """Centered horizontal flip transform."""
    return centered_flip(image_size, image_spacing, "horizontal")

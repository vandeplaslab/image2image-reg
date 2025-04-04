"""Transformation utilities."""

from __future__ import annotations

import json
import math
import typing as ty
from copy import deepcopy
from pathlib import Path

import itk
import numpy as np
import SimpleITK as sitk

from image2image_reg.elastix.registration_utils import json_to_pmap_dict
from image2image_reg.elastix.transform_sequence import Transform
from image2image_reg.elastix.transformation_map import BASE_AFFINE_TRANSFORM, BASE_RIGID_TRANSFORM
from image2image_reg.enums import ELX_TO_ITK_INTERPOLATORS
from image2image_reg.preprocessing.convert import itk_image_to_sitk_image, sitk_image_to_itk_image


def resample(
    image: sitk.Image, transform: Transform, image_shape: tuple[int, int], inverse: bool = False
) -> sitk.Image:
    """Resample image for specified transform."""
    resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
    resampler.SetOutputOrigin(transform.output_origin)  # type: ignore[no-untyped-call]
    resampler.SetOutputDirection(transform.output_direction)  # type: ignore[no-untyped-call]
    resampler.SetSize(image_shape)  # type: ignore[no-untyped-call]
    resampler.SetOutputSpacing(transform.output_spacing)  # type: ignore[no-untyped-call]

    interpolator = ELX_TO_ITK_INTERPOLATORS[transform.resample_interpolator]
    resampler.SetInterpolator(interpolator)  # type: ignore[no-untyped-call]
    resampler.SetTransform(  #  type: ignore[no-untyped-call]
        transform.final_transform if not inverse else transform.final_transform.GetInverse(),
    )
    return resampler.Execute(image)  # type: ignore[no-untyped-call]


def compute_affine_bound_for_image(image: sitk.Image, affine: np.ndarray) -> tuple[float, float]:
    """Compute the bounds of an image after an affine transformation."""
    w, h = image.GetSize()[0:2]  # type: ignore[no-untyped-call]
    return compute_affine_bound((h, w), affine)


def compute_affine_bound(
    shape: tuple[int, int], affine: np.ndarray, spacing: float = 1
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute affine bounds."""
    w, h = shape
    # w = w * spacing
    # h = h * spacing
    # # Top-left, Top-right, Bottom-left, Bottom-right
    # corners = np.array(
    #     [
    #         [0, 0, 1],  # Adding 1 for homogeneous coordinates (x, y, 1)
    #         [w, 0, 1],
    #         [0, h, 1],
    #         [w, h, 1],
    #     ]
    # )
    # # Apply affine transformation to corners
    # transformed_corners = np.dot(corners, affine.T)  # Transpose matrix to match shapes
    #
    # # Extracting x and y coordinates
    # x_coords, y_coords = transformed_corners[:, 0], transformed_corners[:, 1]
    #
    # # Calculate bounds
    # min_x, max_x = np.min(x_coords), np.max(x_coords)
    # min_y, max_y = np.min(y_coords), np.max(y_coords)
    #
    # # Calculate new  width and height
    # new_width = max(w, int(math.ceil(max_x - min_x)))
    # new_height = max(h, int(math.ceil(max_y - min_y)))

    angle = calculate_rotation_angle(affine)
    new_width, new_height = compute_rotation_bounds(shape, angle)
    new_width = max(w, new_width)
    new_height = max(h, new_height)

    affine_ = affine[:2, :] / spacing  # not really valid but we only care about the tx/ty
    new_width += abs(affine_[1, 2])  # tx
    new_height += abs(affine_[0, 2])  # ty
    return (new_width, new_height), (new_width / 2, new_height / 2)


def calculate_rotation_angle(affine: np.ndarray) -> float:
    """Calculate rotation angle."""
    affine = np.linalg.inv(affine)
    affine_ = affine[:2, :]
    a, b, _, c, d, _ = affine_.flatten()

    # The singular values are the square root of the eigenvalues
    # of the matrix times its transpose, M M*
    # Computing trace and determinant of M M*
    trace = a**2 + b**2 + c**2 + d**2
    det = (a * d - b * c) ** 2

    delta = trace**2 / 4 - det
    if delta < 1e-12:
        delta = 0

    l1 = math.sqrt(trace / 2 + math.sqrt(delta))
    y, x = c / l1, a / l1
    return math.atan2(y, x) * 180 / math.pi


def calculate_center_of_rotation(
    affine: np.ndarray, shape: tuple[int, int], spacing: tuple[float, float]
) -> tuple[float, float]:
    """Calculate the center of rotation based on the affine matrix, image shape, and pixel spacing.

    Parameters
    ----------
    affine  : np.ndarray
        The 3x3 affine transformation matrix.
    shape : tuple[int, int]
        The (width, height) of the image.
    spacing : tuple[float, float]
        The (x, y) pixel spacing.

    Returns
    -------
    - A tuple (x, y) representing the center of rotation in the transformed space.
    """
    w, h = shape
    s_x, s_y = spacing

    # Calculate the center of the image in original space, adjusted for pixel spacing
    center_original = np.array([(w * s_x) / 2, (h * s_y) / 2, 1])

    # Apply the affine transformation to this center
    center_transformed = np.dot(affine, center_original)

    return center_transformed[0], center_transformed[1]


def affine_to_itk_affine(
    affine: np.ndarray,
    image_shape: tuple[int, int],
    spacing: float = 1.0,
    inverse: bool = False,
) -> dict:
    """Convert affine matrix (yx, um) to ITK affine matrix.

    The assumption is that the affine matrix is provided in numpy ordering (e.g. from napari) and values are in um.
    """
    # TODO change the origin so that we don't have to make the image bigger

    assert affine.shape == (3, 3), "affine matrix must be 3x3"
    if inverse:
        affine = np.linalg.inv(affine)

    tform = deepcopy(BASE_AFFINE_TRANSFORM)
    tform["Spacing"] = [str(spacing), str(spacing)]
    # compute new image shape
    (bound_w, bound_h), (_origin_x, _origin_y) = compute_affine_bound(image_shape, affine, spacing)  # width, height

    # calculate rotation center point
    # center_of_rot = calculate_center_of_rotation(affine, image_shape, (spacing, spacing))
    # center_of_rot = center_of_rot[0] * spacing, center_of_rot[0] * spacing
    # w, h = image_shape
    # center_of_rot = ((w - 1) / 2, (h - 1) / 2)
    # tform["CenterOfRotationPoint"] = [str(center_of_rot[0]), str(center_of_rot[1])]

    # adjust for pixel spacing
    tform["Size"] = [str(math.ceil(bound_w)), str(math.ceil(bound_h))]

    # extract affine parameters
    affine_ = affine[:2, :]
    tform["TransformParameters"] = [
        affine_[1, 1],
        affine_[1, 0],
        affine_[0, 1],
        affine_[0, 0],
        affine_[1, 2],  # tx
        affine_[0, 2],  # ty
    ]
    return tform


def affine_to_itk_affine2(
    affine: np.ndarray,
    image_shape: tuple[int, int],
    spacing: float = 1.0,
    inverse: bool = False,
) -> dict:
    """Convert affine matrix (yx, um) to ITK affine matrix.

    The assumption is that the affine matrix is provided in numpy ordering (e.g. from napari) and values are in um.
    """
    # TODO change the origin so that we don't have to make the image bigger

    assert affine.shape == (3, 3), "affine matrix must be 3x3"
    if inverse:
        affine = np.linalg.inv(affine)

    tform = deepcopy(BASE_AFFINE_TRANSFORM)
    tform["Spacing"] = [str(spacing), str(spacing)]
    # compute new image shape
    (bound_w, bound_h), (origin_x, origin_y) = compute_affine_bound(image_shape, affine, spacing)  # width, height

    # calculate rotation center point
    # center_of_rot = calculate_center_of_rotation(affine, image_shape, (spacing, spacing))
    # center_of_rot = image.TransformContinuousIndexToPhysicalPoint(
    #     ((bound_w - 1) / 2, (bound_h - 1) / 2),
    # )  # type: ignore[no-untyped-call]
    # w, h = image_shape
    # center_of_rot = ((w - 1) / 2, (h - 1) / 2)
    (rot_x_phy, rot_y_phy) = ((bound_w - 1) / 2, (bound_h - 1) / 2)
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]

    # adjust for pixel spacing
    tform["Size"] = [str(math.ceil(bound_w)), str(math.ceil(bound_h))]
    # tform["Origin"] = [str(origin_x), str(origin_y)]

    # Extract parameters from the 3x3 matrix
    a = affine[0, 0]
    b = affine[0, 1]
    tx = affine[0, 2]
    c = affine[1, 0]
    d = affine[1, 1]
    ty = affine[1, 2]

    # extract affine parameters
    tform["TransformParameters"] = [a, b, ty, c, d, tx]
    return tform


def prepare_tform_dict(tform_dict: dict, shape_tform: bool = False) -> dict:
    """Prepare the transformation dictionary for use in SimpleElastix."""
    transforms_out_dict = {}
    for k, v in tform_dict.items():
        if k == "initial":
            transforms_out_dict["initial"] = v
        else:
            transforms = []
            for tform in v:
                if "invert" in list(tform.keys()):
                    if not shape_tform:
                        transforms.append(tform["image"])
                    else:
                        transforms.append(tform["invert"])
                else:
                    transforms.append(tform)
            transforms_out_dict[k] = transforms
    return transforms_out_dict


def transform_2d_image_itkelx(
    image: sitk.Image, transformation_maps: list, writer: str = "sitk", **_zarr_kwargs: ty.Any
):
    """Transform 2D images with multiple models.

    Return the transformed image or write the transformed image to disk as a .tif file.
    Multichannel or multicomponent images (RGB) have to be transformed a single channel at a time
    This function takes care of performing those transformations and reconstructing the image in the same
    data type as the input.

    Parameters
    ----------
    image : SimpleITK.Image
        Image to be transformed
    transformation_maps : list
        list of SimpleElastix ParameterMaps to used for transformation
    writer : str
        Type of writer to use. Currently only supports "sitk" (default) and "zarr".

    Returns
    -------
    Transformed SimpleITK.Image.
    """
    if transformation_maps is not None:
        tfx = itk.TransformixFilter.New()

        # TODO: add mask cropping here later

        #     print("mask cropping")
        #     tmap = sitk.ReadParameterFile(transformation_maps[0])
        #     x_min = int(float(tmap["MinimumX"][0]))
        #     x_max = int(float(tmap["MaximumX"][0]))
        #     y_min = int(float(tmap["MinimumY"][0]))
        #     y_max = int(float(tmap["MaximumY"][0]))
        #     image = image[x_min:x_max, y_min:y_max]
        #     origin = np.repeat(0, len(image.GetSize()))
        #     image.SetOrigin(tuple([int(i) for i in origin]))

        # else:
        transform_pobj = itk.ParameterObject.New()
        for idx, tmap in enumerate(transformation_maps):
            if isinstance(tmap, str):
                tmap = sitk.ReadParameterFile(tmap)

            if idx == 0:
                tmap["InitialTransformParametersFileName"] = ("NoInitialTransform",)
                transform_pobj.AddParameterMap(tmap)
            else:
                tmap["InitialTransformParametersFileName"] = ("NoInitialTransform",)
                transform_pobj.AddParameterMap(tmap)
        tfx.SetTransformParameterObject(transform_pobj)
        tfx.LogToConsoleOn()
        tfx.LogToFileOff()
    else:
        tfx = None

    # if tfx is None:
    #     xy_final_size = np.array(image.GetSize(), dtype=np.uint32)
    # else:
    #     xy_final_size = np.array(
    #         transformation_maps[-1]["Size"], dtype=np.uint32
    #     )

    if writer == "sitk" or writer is None:
        return transform_image_itkelx_to_sitk(image, tfx)
    elif writer == "zarr":
        return
    else:
        raise ValueError(f"writer type {writer} not recognized")


def transform_image_to_sitk(image, tfx):
    """Transform image to SimpleITK format."""
    # manage transformation/casting if data is multichannel or RGB
    # data is always returned in the same PixelIDType as it is entered

    pixel_id = image.GetPixelID()
    if tfx is not None:
        if pixel_id in list(range(1, 13)) and image.GetDepth() == 0:
            tfx.SetMovingImage(image)
            image = tfx.Execute()
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

        elif pixel_id in list(range(1, 13)) and image.GetDepth() > 0:
            images = []
            for chan in range(image.GetDepth()):
                tfx.SetMovingImage(image[:, :, chan])
                images.append(sitk.Cast(tfx.Execute(), pixel_id))  # type: ignore[no-untyped-call]
            image = sitk.JoinSeries(images)  # type: ignore[no-untyped-call]
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

        elif pixel_id > 12:
            images = []
            for idx in range(image.GetNumberOfComponentsPerPixel()):
                im = sitk.VectorIndexSelectionCast(image, idx)  # type: ignore[no-untyped-call]
                pixel_id_nonvec = im.GetPixelID()
                tfx.SetMovingImage(im)
                images.append(sitk.Cast(tfx.Execute(), pixel_id_nonvec))  # type: ignore[no-untyped-call]
                del im

            image = sitk.Compose(images)  # type: ignore[no-untyped-call]
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

    return image


def transform_image_itkelx_to_sitk(image, tfx):
    """Transform image to SimpleITK format."""
    # manage transformation/casting if data is multichannel or RGB
    # data is always returned in the same PixelIDType as it is entered

    pixel_id = image.GetPixelID()
    if tfx is not None:
        if pixel_id in list(range(1, 13)) and image.GetDepth() == 0:
            image = sitk_image_to_itk_image(image, cast_to_float32=True)
            tfx.SetMovingImage(image)
            tfx.UpdateLargestPossibleRegion()
            image = tfx.GetOutput()
            image = itk_image_to_sitk_image(image)
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

        elif pixel_id in list(range(1, 13)) and image.GetDepth() > 0:
            images = []
            for chan in range(image.GetDepth()):
                image = sitk_image_to_itk_image(image[:, :, chan], cast_to_float32=True)
                tfx.SetMovingImage(image)
                tfx.UpdateLargestPossibleRegion()
                image = tfx.GetOutput()
                image = itk_image_to_sitk_image(image)
                image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]
                images.append(image)
            image = sitk.JoinSeries(images)  # type: ignore[no-untyped-call]
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

        elif pixel_id > 12:
            images = []
            for idx in range(image.GetNumberOfComponentsPerPixel()):
                im = sitk.VectorIndexSelectionCast(image, idx)  # type: ignore[no-untyped-call]
                pixel_id_nonvec = im.GetPixelID()
                im = sitk_image_to_itk_image(im, cast_to_float32=True)
                tfx.SetMovingImage(im)
                tfx.UpdateLargestPossibleRegion()
                im = tfx.GetOutput()
                im = itk_image_to_sitk_image(im)
                im = sitk.Cast(im, pixel_id_nonvec)  # type: ignore[no-untyped-call]
                images.append(im)
                del im

            image = sitk.Compose(images)  # type: ignore[no-untyped-call]
            image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]

    return image


def compute_rotation_bounds_for_image(image: sitk.Image, angle_deg: float = 30) -> tuple[float, float]:
    """Compute the bounds of an image after by an angle.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated angle
    angle_deg : float
        angle of rotation in degrees, rotates counter-clockwise if positive

    Returns
    -------
    tuple of the rotated image's size in x and y

    """
    w, h = image.GetSize()[0:2]  # type: ignore[no-untyped-call]
    return compute_rotation_bounds((h, w), angle_deg=angle_deg)


def compute_rotation_bounds(shape: tuple[int, int], angle_deg: float = 0) -> tuple[float, float]:
    """Compute rotation bounds."""
    h, w = shape
    theta = np.radians(angle_deg)
    c, s = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    bound_w = (h * s) + (w * c)
    bound_h = (h * c) + (w * s)
    return bound_w, bound_h


def generate_rigid_rotation_transform(image: sitk.Image, spacing: float, angle_deg: float) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    angle_deg : float
        angle of rotation in degrees, rotates counter-clockwise if positive.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w_px, bound_h_px = compute_rotation_bounds_for_image(image, angle_deg=angle_deg)
    # calculate rotation center point
    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w_px - 1) / 2, (bound_h_px - 1) / 2),
    )  # type: ignore[no-untyped-call]

    size = image.GetSize()
    c_x, c_y = (size[0] - 1) / 2, (size[1] - 1) / 2  # type: ignore[no-untyped-call]
    c_x_phy, c_y_phy = image.TransformContinuousIndexToPhysicalPoint((c_x, c_y))  # type: ignore[no-untyped-call]
    translation_x_phy = rot_x_phy - c_x_phy
    translation_y_phy = rot_y_phy - c_y_phy

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(math.ceil(bound_w_px)), str(math.ceil(bound_h_px))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [
        str(np.radians(angle_deg)),
        str(-1 * translation_x_phy),
        str(-1 * translation_y_phy),
    ]
    return tform


# def generate_rigid_translation_transform(
#     image: sitk.Image,
#     spacing: float,
#     translation_x_phy: float,
#     translation_y_phy: float,
#     size_x_phy: int,
#     size_y_phy: int,
# ) -> dict:
#     """Generate a SimpleElastix transformation parameter Map to translate and crop an image.
#
#     Parameters
#     ----------
#     image : sitk.Image
#         The input image.
#     spacing : float
#         The desired spacing of the output image.
#     translation_x_phy : float
#         The translation in the x-direction in physical units.
#     translation_y_phy : float
#         The translation in the y-direction in physical units.
#     size_x_phy : int
#         The width of the output image in physical units.
#     size_y_phy : int
#         The height of the output image in physical units.
#
#     Returns
#     -------
#     dict
#         SimpleITK.ParameterMap of translation transformation (EulerTransform)
#     """
#     tform = deepcopy(BASE_RIGID_TRANSFORM)
#     inv_spacing = 1 / spacing
#     image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
#     bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)
#
#     (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
#         ((bound_w - 1) / 2, (bound_h - 1) / 2),
#     )  # type: ignore[no-untyped-call]
#     size_x_px, size_y_px = size_x_phy * inv_spacing, size_y_phy * inv_spacing
#
#     tform["Spacing"] = [str(spacing), str(spacing)]
#     tform["Size"] = [str(math.ceil(size_x_px)), str(math.ceil(size_y_px))]
#     tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
#     tform["TransformParameters"] = [
#         str(0),
#         str(translation_x_phy - size_x_phy / 2),
#         str(translation_y_phy - size_y_phy / 2),
#     ]
#     return tform  # Corrected return variable


def generate_rigid_translation_transform(
    image: sitk.Image,
    spacing: float,
    translation_x_phy: float,
    translation_y_phy: float,
    size_x_phy: int,
    size_y_phy: int,
) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    inv_spacing = 1 / spacing
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)

    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w - 1) / 2, (bound_h - 1) / 2),
    )  # type: ignore[no-untyped-call]
    size_x_px, size_y_px = size_x_phy * inv_spacing, size_y_phy * inv_spacing

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(math.ceil(size_x_px)), str(math.ceil(size_y_px))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [str(0), str(int(translation_x_phy)), str(int(translation_y_phy))]
    return tform


def generate_rigid_translation_transform_alt4(
    image: sitk.Image,
    spacing: float,
    translation_x_phy: float,
    translation_y_phy: float,
    size_x_phy: int,
    size_y_phy: int,
) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    inv_spacing = 1 / spacing
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)

    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w - 1) / 2, (bound_h - 1) / 2),
    )  # type: ignore[no-untyped-call]
    size_x_px, size_y_px = size_x_phy * inv_spacing, size_y_phy * inv_spacing

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(math.ceil(size_x_px)), str(math.ceil(size_y_px))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [str(0), str(int(translation_x_phy)), str(int(translation_y_phy))]
    return tform


# def generate_rigid_translation_transform(
#     image: sitk.Image,
#     spacing: float,
#     translation_x_phy: float,
#     translation_y_phy: float,
#     size_x_phy: int,
#     size_y_phy: int,
# ) -> dict:
#     """Generate a SimpleElastix transformation parameter Map to rotate image by angle.
#
#     Returns
#     -------
#     SimpleITK.ParameterMap of rotation transformation (EulerTransform)
#     """
#     tform = deepcopy(BASE_RIGID_TRANSFORM)
#     image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
#     bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)
#
#     (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
#         ((bound_w - 1) / 2, (bound_h - 1) / 2),
#     )  # type: ignore[no-untyped-call]
#     (size_x_px, size_y_px) = image.TransformPhysicalPointToContinuousIndex((float(size_x_phy), float(size_y_phy)))
#     size_x_px, size_y_px = size_x_phy, size_y_px
#     (translation_x_phy, translation_y_phy) = image.TransformPhysicalPointToContinuousIndex(
#         (float(translation_x_phy), float(translation_y_phy))
#     )
#     (size_x_phy, size_y_phy) = image.TransformContinuousIndexToPhysicalPoint((float(size_x_px), float(size_y_px)))
#
#     tform["Spacing"] = [str(spacing), str(spacing)]
#     tform["Size"] = [str(math.ceil(size_x_px)), str(math.ceil(size_y_px))]
#     tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
#     tform["TransformParameters"] = [
#         str(0),
#         str(int(translation_x_phy + size_x_phy)),
#         str(int(translation_y_phy + size_y_phy)),
#     ]
#     return tform


def generate_rigid_translation_transform_alt3(
    image: sitk.Image,
    spacing: float,
    translation_x_px: float,
    translation_y_px: float,
    size_x_px: int,
    size_y_px: int,
) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w_px, bound_h_px = compute_rotation_bounds_for_image(image, angle_deg=0)

    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w_px - 1) / 2, (bound_h_px - 1) / 2),
    )  # type: ignore[no-untyped-call]
    (translation_x_phy, translation_y_phy) = image.TransformPhysicalPointToContinuousIndex(
        (float(translation_x_px), float(translation_y_px))
    )
    rot_x_phy += translation_x_phy
    rot_y_phy += translation_y_phy
    translation_x_phy = translation_x_px - size_x_px  # * spacing
    translation_y_phy = translation_y_px - size_y_px  # * spacing
    # (size_x_px, size_y_px) = image.TransformPhysicalPointToContinuousIndex((size_x_phy, size_y_phy))
    # size_x_px, size_y_px = size_x_px / spacing, size_y_px / spacing
    size_x_px = size_x_px / 2**4
    size_y_px = size_y_px / 2**4
    print(translation_x_phy, translation_y_phy, size_y_px, size_x_px)
    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(math.ceil(size_x_px)), str(math.ceil(size_y_px))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [str(0), str(int(translation_x_phy)), str(int(translation_y_phy))]
    return tform


def generate_rigid_translation_transform_alt2(
    image: sitk.Image,
    spacing: float,
    translation_x_px: float,
    translation_y_px: float,
) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    w, h = image.GetSize()  # type: ignore[no-untyped-call]
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w_px, bound_h_px = compute_rotation_bounds_for_image(image, angle_deg=0)

    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w_px - 1) / 2, (bound_h_px - 1) / 2),
    )  # type: ignore[no-untyped-call]
    (
        translation_x_phy,
        translation_y_phy,
    ) = image.TransformContinuousIndexToPhysicalPoint(
        (float(translation_x_px), float(translation_y_px)),
    )  # type: ignore[no-untyped-call]
    translation_x_phy, translation_y_phy = translation_x_px, translation_y_px
    # c_x, c_y = (image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2

    w = int(math.ceil(max(w - translation_x_px, bound_w_px)) * 1)
    h = int(math.ceil(max(h - translation_y_px, bound_h_px)) * 1)

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(w), str(h)]
    # tform["Origin"] = [str(int(w)), str(int(h))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [
        str(0),
        str(translation_x_phy),
        str(translation_y_phy),
    ]
    return tform


def generate_rigid_translation_transform_alt(
    image: sitk.Image,
    spacing: float,
    translation_x_phy: float,
    translation_y_phy: float,
) -> dict:
    """Generate a SimpleElastix transformation parameter Map to rotate image by angle.

    Returns
    -------
    SimpleITK.ParameterMap of rotation transformation (EulerTransform)
    """
    w, h = image.GetSize()  # type: ignore[no-untyped-call]
    tform = deepcopy(BASE_RIGID_TRANSFORM)
    image.SetSpacing((spacing, spacing))  # type: ignore[no-untyped-call]
    bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)

    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(
        ((bound_w - 1) / 2, (bound_h - 1) / 2),
    )  # type: ignore[no-untyped-call]
    (translation_x, translation_y) = image.TransformPhysicalPointToContinuousIndex(
        (translation_x_phy, translation_y_phy),
    )  # type: ignore[no-untyped-call]
    # (
    #     translation_x_phy,
    #     translation_y_phy,
    # ) = image.TransformContinuousIndexToPhysicalPoint(
    #     (float(translation_x), float(translation_y)),
    # )  # type: ignore[no-untyped-call]
    # translation_x_phy, translation_y_phy = translation_x, translation_y
    # c_x, c_y = (image.GetSize()[0] - 1) / 2, (image.GetSize()[1] - 1) / 2

    # print("X", w, translation_x, bound_w)
    # print("Y", h, translation_y, bound_h)
    w = int(math.ceil(w - translation_x) * 1)
    h = int(math.ceil(h - translation_y) * 1)
    # w = int(math.ceil(max(w - translation_x, bound_w)) * 1)
    # h = int(math.ceil(max(h - translation_y, bound_h)) * 1)

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(w), str(h)]
    # tform["Origin"] = [str(int(w // 2)), str(int(h // 2))]
    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    tform["TransformParameters"] = [
        str(0),
        str(translation_x_phy),
        str(translation_y_phy),
    ]
    return tform


def generate_rigid_original_transform(original_size: tuple[int, int], crop_transform: dict) -> dict:
    """Generate a SimpleElastix transformation to return a cropped image to its original size."""
    crop_transform["Size"] = [str(original_size[0]), str(original_size[1])]
    tform_params = [float(t) for t in crop_transform["TransformParameters"]]
    crop_transform["TransformParameters"] = [str(0), str(tform_params[1] * -1), str(tform_params[2] * -1)]
    return crop_transform


def generate_affine_flip_transform(image: sitk.Image, spacing: float, flip: str = "h") -> dict:
    """Generate a SimpleElastix transformation parameter Map to horizontally or vertically flip image.

    Parameters
    ----------
    image : sitk.Image
        SimpleITK image that will be rotated
    spacing : float
        Physical spacing of the SimpleITK image
    flip : str
        "h" or "v" for horizontal or vertical flipping, respectively

    Returns
    -------
    SimpleITK.ParameterMap of flipping transformation (AffineTransform)

    """
    tform = deepcopy(BASE_AFFINE_TRANSFORM)
    image.SetSpacing((spacing, spacing))
    bound_w, bound_h = compute_rotation_bounds_for_image(image, angle_deg=0)
    (rot_x_phy, rot_y_phy) = image.TransformContinuousIndexToPhysicalPoint(((bound_w - 1) / 2, (bound_h - 1) / 2))

    tform["Spacing"] = [str(spacing), str(spacing)]
    tform["Size"] = [str(int(bound_w)), str(int(bound_h))]

    tform["CenterOfRotationPoint"] = [str(rot_x_phy), str(rot_y_phy)]
    if flip == "h":
        tform_params = ["-1", "0", "0", "1", "0", "0"]
    elif flip == "v":
        tform_params = ["1", "0", "0", "-1", "0", "0"]

    tform["TransformParameters"] = tform_params

    return tform


def make_composite_itk(itk_transforms: list[Transform]) -> sitk.CompositeTransform:
    """Make composite transform."""
    itk_composite = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
    for t in itk_transforms:
        itk_composite.AddTransform(t.itk_transform)  # type: ignore[no-untyped-call]
    return itk_composite


def collate_wsireg_transforms(parameter_data):
    """Collate WsiReg transforms."""
    if isinstance(parameter_data, str) and Path(parameter_data).suffix == ".json":
        parameter_data = json.load(open(parameter_data))

    parameter_data_list = []
    for k, v in parameter_data.items():
        if k == "initial":
            if isinstance(v, dict):
                parameter_data_list.append([v])
            elif isinstance(v, list):
                for init_tform in v:
                    parameter_data_list.append([init_tform])
        else:
            sub_tform = []
            if isinstance(v, dict):
                sub_tform.append(v)
            elif isinstance(v, list):
                sub_tform += v
            sub_tform = sub_tform[::-1]
            parameter_data_list.append(sub_tform)

    flat_pmap_list = [item for sublist in parameter_data_list for item in sublist]

    if all(isinstance(t, dict) for t in flat_pmap_list):
        flat_pmap_list = [Transform(t) for t in flat_pmap_list]

    return flat_pmap_list


def wsireg_transforms_to_itk_composite(parameter_data):
    """Wsireg transforms to ITK composite."""
    reg_transforms = collate_wsireg_transforms(parameter_data)
    composite_tform = make_composite_itk(reg_transforms)

    return composite_tform, reg_transforms


def prepare_wsireg_transform_data(transform_data: str | dict | None) -> tuple:
    """Prepare WsiReg transform data."""
    if isinstance(transform_data, str):
        transform_data = json_to_pmap_dict(transform_data)

    if transform_data is not None:
        (
            composite_transform,
            itk_transforms,
        ) = wsireg_transforms_to_itk_composite(transform_data)
        final_tform = itk_transforms[-1]
    return composite_transform, itk_transforms, final_tform


def identity_elx_transform(
    image_size: tuple[int, int],
    image_spacing: tuple[int, int] | tuple[float, float],
) -> dict:
    """Create identity elastix transform."""
    identity = deepcopy(BASE_RIGID_TRANSFORM)
    identity.update({"Size": [str(i) for i in image_size]})
    identity.update({"Spacing": [str(i) for i in image_spacing]})
    return identity


def wsireg_transforms_to_resampler(final_tform: Transform) -> sitk.ResampleImageFilter:
    """Create resmapler."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(final_tform.output_origin)
    resampler.SetSize(final_tform.output_size)
    resampler.SetOutputDirection(final_tform.output_direction)
    resampler.SetOutputSpacing(final_tform.output_spacing)

    interpolator = ELX_TO_ITK_INTERPOLATORS.get(final_tform.resample_interpolator)
    resampler.SetInterpolator(interpolator)
    return resampler


def sitk_transform_image(image: sitk.Image, final_tform: Transform, composite_transform: Transform) -> sitk.Image:
    """Transform image."""
    resampler = wsireg_transforms_to_resampler(final_tform)
    resampler.SetTransform(composite_transform)
    image = resampler.Execute(image)
    return image


def transform_plane(image: sitk.Image, final_transform: Transform, composite_transform: Transform) -> sitk.Image:
    """Transform image."""
    return sitk_transform_image(image, final_transform, composite_transform)

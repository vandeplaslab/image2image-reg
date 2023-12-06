"""Pre-process dask array."""
from __future__ import annotations

import cv2
import dask.array as da
import numpy as np
import SimpleITK as sitk
from image2image_io.readers.utilities import grayscale, guess_rgb
from loguru import logger

from image2image_wsireg.enums import ImageType
from image2image_wsireg.models import Preprocessing
from image2image_wsireg.models.preprocessing import BoundingBox
from image2image_wsireg.utils.transformation import (
    gen_affine_transform_flip,
    gen_rig_to_original,
    gen_rigid_tform_rot,
    gen_rigid_translation,
    prepare_wsireg_transform_data,
    transform_plane,
)


def preprocess_dask_array(
    array: da.core.Array,  # type: ignore[name-defined]
    preprocessing: Preprocessing | None = None,
) -> sitk.Image:
    """Pre-process dask array."""
    is_rgb = guess_rgb(array.shape)
    if is_rgb:
        if preprocessing:
            array_out = np.asarray(grayscale(array, is_interleaved=is_rgb))
            array_out = sitk.GetImageFromArray(array_out)
        else:
            array_out = np.asarray(array)
            array_out = sitk.GetImageFromArray(array_out, isVector=True)

    elif len(array.shape) == 2:
        array_out = sitk.GetImageFromArray(np.asarray(array))
    else:
        if preprocessing:
            if preprocessing.ch_indices and len(array.shape) > 2:
                chs = list(preprocessing.ch_indices)
                array = array[chs, :, :]
        array_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))
    return array_out


def convert_and_cast(image: sitk.Image, preprocessing: Preprocessing | None = None) -> sitk.Image:
    """Covert image to uint8 if specified in preprocessing."""
    if preprocessing is not None and preprocessing.as_uint8 is True and image.GetPixelID() != sitk.sitkUInt8:
        image = sitk.RescaleIntensity(image)
        image = sitk.Cast(image, sitk.sitkUInt8)
    return image


def read_preprocess_array(
    array: np.ndarray | da.Array, preprocessing: Preprocessing, force_rgb: bool | None = None
) -> sitk.Image:
    """Read np.array, zarr.Array, or dask.array image into memory with preprocessing for registration."""
    is_interleaved = guess_rgb(array.shape)
    is_rgb = is_interleaved if not force_rgb else force_rgb

    if is_rgb:
        if preprocessing:
            image_out = np.asarray(grayscale(array, is_interleaved=is_interleaved))
            image_out = sitk.GetImageFromArray(image_out)
        else:
            image_out = np.asarray(array)
            if not is_interleaved:
                image_out = np.rollaxis(image_out, 0, 3)
            image_out = sitk.GetImageFromArray(image_out, isVector=True)

    elif len(array.shape) == 2:
        image_out = sitk.GetImageFromArray(np.asarray(array))

    else:
        if preprocessing:
            if preprocessing.ch_indices and len(array.shape) > 2:
                chs = list(preprocessing.ch_indices)
                array = array[chs, :, :]
        image_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))
    return image_out


def sitk_vect_to_gs(image: sitk.Image) -> sitk.Image:
    """
    converts simpleITK RGB image to greyscale using cv2.

    Parameters
    ----------
    image
        SimpleITK image.

    Returns
    -------
        Greyscale SimpleITK image
    """
    image = sitk.GetArrayFromImage(image)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    return sitk.GetImageFromArray(image, isVector=False)


def sitk_max_int_proj(image: sitk.Image) -> sitk.Image:
    """
    Finds maximum intensity projection of multi-channel SimpleITK image.

    Parameters
    ----------
    image
        multichannel impleITK image


    Returns
    -------
    SimpleITK image
    """
    # check if there are 3 dimensions (XYC)
    if len(image.GetSize()) == 3:
        return sitk.MaximumProjection(image, 2)[:, :, 0]
    else:
        logger.warning("Cannot perform maximum intensity project on single channel image")
        return image


def sitk_inv_int(image: sitk.Image) -> sitk.Image:
    """Inverts intensity of images for registration, useful for alignment of bright field and fluorescence images."""
    return sitk.InvertIntensity(image)


def contrast_enhance(image: sitk.Image) -> sitk.Image:
    """Enhance contrast of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.convertScaleAbs(image, alpha=7, beta=1)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def preprocess_intensity(
    image: sitk.Image, preprocessing: Preprocessing, pixel_size: float, is_rgb: bool
) -> sitk.Image:
    """Preprocess image intensity data to single channel image."""
    if preprocessing.image_type == ImageType.DARK:
        preprocessing.invert_intensity = False
    elif preprocessing.image_type == ImageType.LIGHT:
        preprocessing.max_int_proj = False
        preprocessing.contrast_enhance = False
        if is_rgb:
            preprocessing.invert_intensity = True

    if preprocessing.max_int_proj:
        image = sitk_max_int_proj(image)

    if preprocessing.contrast_enhance:
        image = contrast_enhance(image)

    if preprocessing.invert_intensity:
        image = sitk_inv_int(image)

    if preprocessing.custom_processing:
        for k, v in preprocessing.custom_processing.items():
            logger.trace(f"Performing preprocessing step: {k}")
            image = v(image)
    image.SetSpacing((pixel_size, pixel_size))
    return image


def compute_mask_to_bbox(mask: sitk.Image, mask_padding: int = 100) -> BoundingBox:
    """Calculate bbox of mask."""
    mask.SetSpacing((1, 1))
    mask_size = mask.GetSize()
    mask = sitk.Threshold(mask, 1, 255)
    mask = sitk.ConnectedComponent(mask)

    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.SetBackgroundValue(0)
    label_stats.ComputePerimeterOff()
    label_stats.ComputeFeretDiameterOff()
    label_stats.ComputeOrientedBoundingBoxOff()
    label_stats.Execute(mask)

    bb_points = []
    for label in label_stats.GetLabels():
        x1, y1, xw, yh = label_stats.GetBoundingBox(label)
        x2, y2 = x1 + xw, y1 + yh
        lab_points = np.asarray([[x1, y1], [x2, y2]])
        bb_points.append(lab_points)

    bb_points = np.concatenate(bb_points)
    x_min = np.min(bb_points[:, 0])
    y_min = np.min(bb_points[:, 1])
    x_max = np.max(bb_points[:, 0])
    y_max = np.max(bb_points[:, 1])

    if (x_min - mask_padding) < 0:
        x_min = 0
    else:
        x_min -= mask_padding

    if (y_min - mask_padding) < 0:
        y_min = 0
    else:
        y_min -= mask_padding

    if (x_max + mask_padding) > mask_size[0]:
        x_max = mask_size[0]
    else:
        x_max += mask_padding

    if (y_max + mask_padding) > mask_size[1]:
        y_max = mask_size[1]
    else:
        y_max += mask_padding

    x_width = x_max - x_min
    y_height = y_max - y_min
    return BoundingBox(x_min, y_min, x_width, y_height)


def preprocess_reg_image_spatial(
    image: sitk.Image,
    preprocessing: Preprocessing,
    pixel_size: float,
    mask: sitk.Image | None = None,
    imported_transforms=None,
) -> tuple[sitk.Image, sitk.Image | None, list[dict], tuple]:
    """
    Spatial preprocessing of the reg_image.

    Returns
    -------
    image: sitk.Image
        Spatially pre-processed image ready for registration
    mask: sitk.Image, optional
        Spatially pre-processed mask ready for registration
    transforms: list of transforms
        List of pre-initial transformations.
    original_size_transform: tuple, optional
        Transform to return the image to its original size.
    """
    transforms = []
    original_size = image.GetSize()

    if preprocessing.downsample > 1:
        logger.trace(f"Performing downsampling by factor: {preprocessing.downsample}")
        image.SetSpacing((pixel_size, pixel_size))
        image = sitk.Shrink(image, (preprocessing.downsample, preprocessing.downsample))

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = sitk.Shrink(mask(preprocessing.downsample, preprocessing.downsample))
        pixel_size = image.GetSpacing()[0]

    # apply affine transformation
    if preprocessing.affine is not None:
        logger.trace("Applying affine transformation")
        affine = preprocessing.affine
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [affine]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)

    # rotate counter-clockwise
    if float(preprocessing.rotate_counter_clockwise) != 0.0:
        logger.trace(f"Rotating counter-clockwise {preprocessing.rotate_counter_clockwise}")
        rot_tform = gen_rigid_tform_rot(image, pixel_size, preprocessing.rotate_counter_clockwise)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [rot_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
        transforms.append(rot_tform)

    # flip image
    if preprocessing.flip:
        logger.trace(f"Flipping image {preprocessing.flip.value}")
        flip_tform = gen_affine_transform_flip(image, pixel_size, preprocessing.flip.value)

        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [flip_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)

        transforms.append(flip_tform)

    if mask and preprocessing.crop_to_mask_bbox:
        logger.trace("computing mask bounding box")
        if preprocessing.mask_bbox is None:
            mask_bbox = compute_mask_to_bbox(mask)
            preprocessing.mask_bbox = mask_bbox

    original_size_transform = None
    if preprocessing.mask_bbox:
        logger.trace("cropping to mask")
        translation_transform = gen_rigid_translation(
            image,
            pixel_size,
            preprocessing.mask_bbox.X,
            preprocessing.mask_bbox.Y,
            preprocessing.mask_bbox.WIDTH,
            preprocessing.mask_bbox.HEIGHT,
        )

        (
            composite_transform,
            _,
            final_tform,
        ) = prepare_wsireg_transform_data({"initial": [translation_transform]})

        image = transform_plane(image, final_tform, composite_transform)
        original_size_transform = gen_rig_to_original(original_size, deepcopy(translation_transform))

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
        transforms.append(translation_transform)

    return image, mask, transforms, original_size_transform


def preprocess(
    image: sitk.Image,
    mask: sitk.Image | None,
    preprocessing: Preprocessing,
    pixel_size: float,
    is_rgb: bool,
    transforms: list,
) -> sitk.Image:
    """Run full intensity and spatial preprocessing."""
    # intensity based pre-processing
    image = preprocess_intensity(image, preprocessing, pixel_size, is_rgb)

    # ensure that intensity-based pre-processing resulted in a single-channel image
    if image.GetDepth() >= 1:
        raise ValueError("preprocessing did not result in a single image plane\nmulti-channel or 3D image return")
    if image.GetNumberOfComponentsPerPixel() > 1:
        raise ValueError(
            "preprocessing did not result in a single image plane\nmulti-component / RGB(A) image returned"
        )

    # spatial pre-processing
    image, mask, transforms, original_size_transform = preprocess_reg_image_spatial(
        image, preprocessing, pixel_size, mask, transforms
    )
    return image, mask, transforms, original_size_transform

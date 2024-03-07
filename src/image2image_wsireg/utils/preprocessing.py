"""Pre-process dask array."""
from __future__ import annotations

from copy import deepcopy

import cv2
import dask.array as da
import numpy as np
import SimpleITK as sitk
from image2image_io.readers.utilities import grayscale, guess_rgb
from loguru import logger

from image2image_wsireg.enums import ImageType
from image2image_wsireg.models import Preprocessing
from image2image_wsireg.models.bbox import BoundingBox
from image2image_wsireg.utils.transformation import (
    affine_to_itk_affine,
    generate_affine_flip_transform,
    generate_rigid_original_transform,
    generate_rigid_rotation_transform,
    generate_rigid_translation_transform,
    generate_rigid_translation_transform_alt,
    prepare_wsireg_transform_data,
    transform_plane,
)


def get_channel_indices_from_names(channel_names: list[str], channel_names_to_select: list[str]) -> list[int]:
    """Get channel indices from channel names."""
    channel_indices = []
    for channel_name in channel_names_to_select:
        if channel_name in channel_names:
            channel_indices.append(channel_names.index(channel_name))
        else:
            logger.warning(f"Channel name {channel_name} not found in the list of channel names")
    return channel_indices


def sort_indices(channel_indices: list[int]) -> list[int]:
    """Sort channel indices and remove duplicates."""
    return np.unique(channel_indices).tolist()


def preprocess_dask_array(
    array: da.core.Array,
    channel_names: list[str],
    preprocessing: Preprocessing | None = None,
) -> sitk.Image:
    """Pre-process dask array."""
    is_rgb = guess_rgb(array.shape)
    # perform max intensity projection
    if is_rgb:
        if preprocessing:
            array_out = np.asarray(grayscale(array, is_interleaved=is_rgb))
            array_out = sitk.GetImageFromArray(array_out)  # type: ignore[assignment]
        else:
            array_out = np.asarray(array)
            array_out = sitk.GetImageFromArray(array_out, isVector=True)  # type: ignore[assignment]
    # no need to pre-process
    elif len(array.shape) == 2:
        array_out = sitk.GetImageFromArray(np.asarray(array))  # type: ignore[assignment]
    else:
        # select channels
        if preprocessing:
            channel_indices = None
            if len(array.shape) > 2 and preprocessing.channel_indices is not None:
                channel_indices = list(preprocessing.channel_indices)
            elif len(array.shape) > 2 and preprocessing.channel_names is not None:
                channel_indices = get_channel_indices_from_names(channel_names, preprocessing.channel_names)
            if channel_indices:
                array = array[sort_indices(channel_indices), :, :]
        array_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))  # type: ignore[assignment]
    return array_out


def convert_and_cast(image: sitk.Image, preprocessing: Preprocessing | None = None) -> sitk.Image:
    """Covert image to uint8 if specified in preprocessing."""
    if preprocessing is not None and preprocessing.as_uint8 and image.GetPixelID() != sitk.sitkUInt8:
        image = sitk.RescaleIntensity(image)
        image = sitk.Cast(image, sitk.sitkUInt8)
    return image


def sitk_vect_to_gs(image: sitk.Image) -> sitk.Image:
    """Converts simpleITK RGB image to greyscale using cv2.

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
    """Finds maximum intensity projection of multi-channel SimpleITK image.

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
        preprocessing.max_intensity_projection = False
        preprocessing.contrast_enhance = False
        if is_rgb:
            preprocessing.invert_intensity = True

    if preprocessing.max_intensity_projection:
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
    transform_mask: bool = True,
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
    transform_mask : bool
        If True, mask will be transformed using the same transformation parameters as the image.
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
        affine_tform = preprocessing.affine
        if isinstance(affine_tform, np.ndarray):
            affine_tform = affine_to_itk_affine(image, affine_tform, original_size, pixel_size, True)
        transforms.append(affine_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [affine_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            if transform_mask:
                mask = transform_plane(mask, final_tform, composite_transform)

    # translate x/y image
    if preprocessing.translate_x or preprocessing.translate_y:
        logger.trace(f"Transforming image by translation: {preprocessing.translate_x}, {preprocessing.translate_y}")
        translation_transform = generate_rigid_translation_transform_alt(
            image, pixel_size, preprocessing.translate_x, preprocessing.translate_y
        )
        transforms.append(translation_transform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [translation_transform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            if transform_mask:
                mask.SetSpacing((pixel_size, pixel_size))
                mask = transform_plane(mask, final_tform, composite_transform)

    # rotate counter-clockwise
    if float(preprocessing.rotate_counter_clockwise) != 0.0:
        logger.trace(f"Rotating counter-clockwise {preprocessing.rotate_counter_clockwise}")
        rot_tform = generate_rigid_rotation_transform(image, pixel_size, preprocessing.rotate_counter_clockwise)
        transforms.append(rot_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [rot_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            if transform_mask:
                mask.SetSpacing((pixel_size, pixel_size))
                mask = transform_plane(mask, final_tform, composite_transform)

    # flip image
    if preprocessing.flip:
        logger.trace(f"Flipping image {preprocessing.flip.value}")
        flip_tform = generate_affine_flip_transform(image, pixel_size, preprocessing.flip.value)
        transforms.append(flip_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [flip_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None:
            if transform_mask:
                mask.SetSpacing((pixel_size, pixel_size))
                mask = transform_plane(mask, final_tform, composite_transform)

    # crop to bbox
    if mask and preprocessing.crop_to_bbox:
        logger.trace("Computing mask bounding box")
        if preprocessing.crop_bbox is None:
            mask_bbox = compute_mask_to_bbox(mask)
            preprocessing.crop_bbox = mask_bbox

    original_size_transform = None
    if preprocessing.crop_bbox:
        logger.trace("Cropping to mask")
        translation_transform = generate_rigid_translation_transform(
            image,
            pixel_size,
            preprocessing.crop_bbox.x,
            preprocessing.crop_bbox.y,
            preprocessing.crop_bbox.width,
            preprocessing.crop_bbox.height,
        )
        transforms.append(translation_transform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [translation_transform]})

        image = transform_plane(image, final_tform, composite_transform)
        original_size_transform = generate_rigid_original_transform(original_size, deepcopy(translation_transform))

        if mask is not None:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)

    return image, mask, transforms, original_size_transform


def preprocess(
    image: sitk.Image,
    mask: sitk.Image | None,
    preprocessing: Preprocessing,
    pixel_size: float,
    is_rgb: bool,
    transforms: list,
    transform_mask: bool = True,
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
        image,
        preprocessing,
        pixel_size,
        mask,
        transforms,
        transform_mask=transform_mask,
    )
    return image, mask, transforms, original_size_transform


def create_thumbnail(input_image: sitk.Image, max_thumbnail_size: int = 512) -> sitk.Image:
    """
    Creates a thumbnail from a SimpleITK.Image.

    Parameters
    ----------
    - input_image: The input SimpleITK.Image object.
    - max_thumbnail_size: The maximum size of the thumbnail's longest dimension.

    Returns
    -------
    - A SimpleITK.Image object representing the thumbnail.
    """
    # Get the original size and spacing
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()

    # Calculate the aspect ratio and new size while maintaining the aspect ratio
    aspect_ratio = original_size[1] / original_size[0]
    if original_size[0] > original_size[1]:
        new_size = [max_thumbnail_size, int(max_thumbnail_size * aspect_ratio)]
    else:
        new_size = [int(max_thumbnail_size / aspect_ratio), max_thumbnail_size]

    # Calculate new spacing to maintain the aspect ratio
    new_spacing = [
        original_spacing[0] * (original_size[0] / new_size[0]),
        original_spacing[1] * (original_size[1] / new_size[1]),
    ]

    # Resample the image to the new size
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    thumbnail = resampler.Execute(input_image)

    return thumbnail

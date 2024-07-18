"""Pre-process dask array."""

from __future__ import annotations

from copy import deepcopy

import cv2
import dask.array as da
import numpy as np
import SimpleITK as sitk
from image2image_io.readers.utilities import grayscale, guess_rgb
from koyo.timer import MeasureTimer
from loguru import logger

from image2image_reg.enums import ImageType
from image2image_reg.models import Preprocessing
from image2image_reg.models.bbox import BoundingBox
from image2image_reg.utils.transformation import (
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
    with MeasureTimer() as timer:
        # perform max intensity projection
        if is_rgb:
            if preprocessing:
                array_out = np.asarray(grayscale(array, is_interleaved=is_rgb))
                array_out = sitk.GetImageFromArray(array_out)  # type: ignore[assignment]
                logger.trace(f"Converted RGB to greyscale in {timer()}")
            else:
                array_out = np.asarray(array)
                array_out = sitk.GetImageFromArray(array_out, isVector=True)  # type: ignore[assignment]
                logger.trace(f"Converted RGB to SimpleITK image in {timer()}")
        # no need to pre-process
        elif len(array.shape) == 2:
            array_out = sitk.GetImageFromArray(np.asarray(array))  # type: ignore[assignment]
            logger.trace(f"Converted 2D array to SimpleITK image in {timer()}")
        else:
            # select channels
            if preprocessing:
                channel_indices = None
                if len(array.shape) > 2 and preprocessing.channel_indices is not None:
                    channel_indices = list(preprocessing.channel_indices)
                elif len(array.shape) > 2 and preprocessing.channel_names is not None:
                    channel_indices = get_channel_indices_from_names(channel_names, preprocessing.channel_names)
                logger.trace(f"Pre-processing dask array with {channel_indices} channels.")
                if channel_indices:
                    array = array[sort_indices(channel_indices), :, :]
            array_out = sitk.GetImageFromArray(np.squeeze(np.asarray(array)))  # type: ignore[assignment]
            logger.trace(f"Pre-processed dask array in {timer()}")
    return array_out


def convert_and_cast(image: sitk.Image, preprocessing: Preprocessing | None = None) -> sitk.Image:
    """Covert image to uint8 if specified in preprocessing."""
    with MeasureTimer() as timer:
        if preprocessing is not None and preprocessing.as_uint8 and image.GetPixelID() != sitk.sitkUInt8:
            image = sitk.RescaleIntensity(image)
            image = sitk.Cast(image, sitk.sitkUInt8)
        logger.trace(f"Converted image to uint8 in {timer()}")
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
        multichannel SimpleITK image

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


def sitk_mean_int_proj(image: sitk.Image) -> sitk.Image:
    """Finds maximum intensity projection of multi-channel SimpleITK image.

    Parameters
    ----------
    image
        multichannel SimpleITK image

    Returns
    -------
    SimpleITK image
    """
    # check if there are 3 dimensions (XYC)
    if len(image.GetSize()) == 3:
        return sitk.MeanProjection(image, 2)[:, :, 0]
    else:
        logger.warning("Cannot perform maximum intensity project on single channel image")
        return image


def sitk_inv_int(image: sitk.Image, mask_zeros: bool = True) -> sitk.Image:
    """Inverts intensity of images for registration, useful for alignment of bright field and fluorescence images."""
    mask = None
    if mask_zeros:
        image = sitk.GetArrayFromImage(image)
        mask = image < 1
        image = sitk.GetImageFromArray(image)
    image = sitk.InvertIntensity(image)
    if mask is not None and mask_zeros:
        image = sitk.GetArrayFromImage(image)
        image[mask] = 0
        image = sitk.GetImageFromArray(image)
    return image


def contrast_enhance(image: sitk.Image, alpha: float = 7, beta: float = 1) -> sitk.Image:
    """Enhance contrast of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def equalize_histogram(image: sitk.Image) -> sitk.Image:
    """Equalize histogram of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image).astype(np.uint8)
    image = cv2.equalizeHist(image)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def clip_intensity(image: sitk.Image, min_val: int = 0, max_val: int = 255) -> sitk.Image:
    """Clip intensity of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    mask_below = image < min_val
    image[mask_below] = 0
    mask_above = image > max_val
    image[mask_above] = max_val
    # image = np.clip(image, min_val, max_val)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def equalize_clahe(image: sitk.Image) -> sitk.Image:
    """Equalize histogram of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    image = clahe.apply(image)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def equalize_adapthist(image: sitk.Image) -> sitk.Image:
    """Equalize histogram of image."""
    from skimage import exposure

    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image).astype(np.uint8)
    image = exposure.equalize_adapthist(image)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def norm_and_adjust(image: sitk.Image) -> sitk.Image:
    """Normalize and brightness adjustment."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image).astype(np.uint8)
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    image = cv2.add(image, 50)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def median_filter(image: sitk.Image) -> sitk.Image:
    """Equalize histogram of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.medianBlur(image, 5)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def remove_background_rolling_ball(image: sitk.Image) -> sitk.Image:
    """Remove background using rolling ball algorithm."""
    from skimage.restoration import rolling_ball

    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    background = rolling_ball(image)
    image = image - background
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def remove_background_black_tophat(image: sitk.Image) -> sitk.Image:
    """Remove background using black tophat algorithm."""
    # use openCV
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def remove_background_white_tophat(image: sitk.Image) -> sitk.Image:
    """Remove background using white tophat algorithm."""
    # use openCV
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


# def test_filter(image: sitk.Image, brightness_q: float = 0.99) -> sitk.Image:
#     from skimage import exposure
#
#     spacing = image.GetSpacing()
#     image = sitk.GetArrayFromImage(image)
#     image, _ = calc_background_color_dist(image, brightness_q=brightness_q)
#     image = exposure.rescale_intensity(image, in_range="image", out_range=(0, 1))
#     image = exposure.equalize_adapthist(image)
#     image = exposure.rescale_intensity(image, in_range="image", out_range=(0, 255)).astype(np.uint8)
#     image = sitk.GetImageFromArray(image)
#     image.SetSpacing(spacing)
#     return image
#
#
# def calc_background_color_dist(
#     img: np.ndarray, brightness_q: float = 0.99, mask: np.ndarray | None = None
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Create mask that only covers tissue
#
#     #. Find background pixel (most luminescent)
#     #. Convert image to CAM16-UCS
#     #. Calculate distance between each pixel and background pixel
#     #. Threshold on distance (i.e. higher distance = different color)
#
#     Returns
#     -------
#     cam_d : float
#         Distance from background color
#     cam : float
#         CAM16UCS image
#
#     """
#     import colour
#
#     eps = np.finfo("float").eps
#     with colour.utilities.suppress_warnings(colour_usage_warnings=True):
#         if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
#             cam = colour.convert(img / 255 + eps, "sRGB", "CAM16UCS")
#         else:
#             cam = colour.convert(img + eps, "sRGB", "CAM16UCS")
#
#     if mask is None:
#         brightest_thresh = np.quantile(cam[..., 0], brightness_q)
#     else:
#         brightest_thresh = np.quantile(cam[..., 0][mask > 0], brightness_q)
#
#     brightest_idx = np.where(cam[..., 0] >= brightest_thresh)
#     brightest_pixels = cam[brightest_idx]
#     bright_cam = brightest_pixels.mean(axis=0)
#     cam_d = np.sqrt(np.sum((cam - bright_cam) ** 2, axis=2))
#     return cam_d, cam
def bilateral_filter(image: sitk.Image) -> sitk.Image:
    """Equalize histogram of image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.bilateralFilter(image, 9, 75, 75)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def nlmeans_filter(image: sitk.Image) -> sitk.Image:
    """Non-local means denoising.."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def background_subtract(image: sitk.Image) -> sitk.Image:
    """Subtract background from image."""
    spacing = image.GetSpacing()
    image = sitk.GetArrayFromImage(image)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


def preprocess_intensity(
    image: sitk.Image, preprocessing: Preprocessing, pixel_size: float, is_rgb: bool
) -> sitk.Image:
    """Preprocess image intensity data to single channel image."""
    with MeasureTimer() as timer:
        convert_and_cast(image, preprocessing)
        if preprocessing.max_intensity_projection:
            image = sitk_max_int_proj(image)
            logger.trace(f"Maximum intensity projection applied in {timer(since_last=True)}")
        if image.GetDepth() > 1:
            logger.warning("Image has more than one channel, mean intensity projection will be used")
            image = sitk_mean_int_proj(image)
            logger.trace(f"Mean intensity projection applied in {timer(since_last=True)}")
        if preprocessing.equalize_histogram:
            # image = equalize_histogram(image)
            image = equalize_clahe(image)
            logger.trace(f"Equalized histogram applied in {timer(since_last=True)}")
        # image = remove_background_black_tophat(image)
        # logger.trace(f"Background removed in {timer(since_last=True)}")
        if preprocessing.contrast_enhance:
            image = contrast_enhance(image)
            logger.trace(f"Contrast enhancement applied in {timer(since_last=True)}")
        if preprocessing.invert_intensity:
            image = sitk_inv_int(image)
            logger.trace(f"Inverted intensity in {timer(since_last=True)}")
        if preprocessing.custom_processing:
            for k, v in preprocessing.custom_processing.items():
                logger.trace(f"Performing preprocessing step: {k} in {timer(since_last=True)}")
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
    spatial: bool = True,
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
            logger.trace("Downsampled mask")
        pixel_size = image.GetSpacing()[0]

    # apply affine transformation
    if preprocessing.affine is not None and spatial:
        logger.trace("Applying affine transformation")
        affine_tform = preprocessing.affine
        if isinstance(affine_tform, np.ndarray):
            affine_tform = affine_to_itk_affine(affine_tform, original_size, pixel_size, True)
        transforms.append(affine_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [affine_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None and transform_mask:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
            logger.trace("Applied affine transform to mask")

    # flip image
    if preprocessing.flip and spatial:
        logger.trace(f"Flipping image {preprocessing.flip.value}")
        flip_tform = generate_affine_flip_transform(image, pixel_size, preprocessing.flip.value)
        transforms.append(flip_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [flip_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None and transform_mask:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
            logger.trace("Flipped mask")

    # rotate counter-clockwise
    if float(preprocessing.rotate_counter_clockwise) != 0.0 and spatial:
        logger.trace(f"Rotating counter-clockwise {preprocessing.rotate_counter_clockwise}")
        rot_tform = generate_rigid_rotation_transform(image, pixel_size, preprocessing.rotate_counter_clockwise)
        transforms.append(rot_tform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [rot_tform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None and transform_mask:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
            logger.trace("Rotated mask")

    # translate x/y image
    if (preprocessing.translate_x or preprocessing.translate_y) and spatial:
        logger.trace(f"Transforming image by translation: {preprocessing.translate_x}, {preprocessing.translate_y}")
        translation_transform = generate_rigid_translation_transform_alt(
            image, pixel_size, preprocessing.translate_x, preprocessing.translate_y
        )
        transforms.append(translation_transform)
        composite_transform, _, final_tform = prepare_wsireg_transform_data({"initial": [translation_transform]})
        image = transform_plane(image, final_tform, composite_transform)

        if mask is not None and transform_mask:
            mask.SetSpacing((pixel_size, pixel_size))
            mask = transform_plane(mask, final_tform, composite_transform)
            logger.trace("Translated mask")

    # crop to bbox
    if mask and preprocessing.use_crop:
        logger.trace("Computing mask bounding box")
        if preprocessing.crop_bbox is None:
            mask_bbox = compute_mask_to_bbox(mask)
            preprocessing.crop_bbox = mask_bbox
            logger.trace(f"Computed mask bounding box: {mask_bbox}")

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
    check: bool = False,
    spatial: bool = True,
) -> tuple[sitk.Image, sitk.Image, list, list]:
    """Run full intensity and spatial preprocessing."""
    if check:
        if preprocessing.image_type == ImageType.DARK:
            preprocessing.invert_intensity = False
        elif preprocessing.image_type == ImageType.LIGHT:
            preprocessing.max_intensity_projection = False
            preprocessing.contrast_enhance = False
            if is_rgb:
                preprocessing.invert_intensity = True
    if preprocessing.equalize_histogram or preprocessing.contrast_enhance:
        preprocessing.as_uint8 = True

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
        image, preprocessing, pixel_size, mask, transforms, transform_mask=transform_mask, spatial=spatial
    )
    return image, mask, transforms, original_size_transform


def create_thumbnail(input_image: sitk.Image, max_thumbnail_size: int = 512) -> sitk.Image:
    """
    Creates a thumbnail from a SimpleITK.Image.

    Parameters
    ----------
    input_image:
        The input SimpleITK.Image object.
    max_thumbnail_size:
        The maximum size of the thumbnail's longest dimension.

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


def preprocess_preview(
    image: np.ndarray,
    is_rgb: bool,
    resolution: float,
    preprocessing: Preprocessing,
    initial_transforms: list | None = None,
    transform_mask: bool = False,
    spatial: bool = True,
) -> np.ndarray:
    """Complete pre-processing."""
    channel_names = preprocessing.channel_names

    image = preprocess_dask_array(image, channel_names, preprocessing)
    # convert and cast
    image = convert_and_cast(image, preprocessing)

    # if mask is not going to be transformed, then we don't need to retrieve it at this moment in time
    # set image
    image, mask, initial_transforms, original_size_transform = preprocess(
        image,
        None,
        preprocessing,
        resolution,
        is_rgb,
        initial_transforms,
        transform_mask=transform_mask,
        check=False,
        spatial=spatial,
    )
    return sitk.GetArrayFromImage(image)

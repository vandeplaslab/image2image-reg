"""Estimate affine transforms from MALDI IMS data to post IMS autofluorescence."""

from __future__ import annotations

import math
import typing as ty
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from image2image_io.readers import get_simple_reader
from koyo.typing import PathLike
from skimage import filters
from skimage.io import imread

NumberPair = tuple[float, float]
Shape2D = tuple[int, int]
IMS_SHAPE_COUNT_MESSAGE = "ims_shape must contain two values in rows,columns order."
IMS_SHAPE_MINIMUM_MESSAGE = "ims_shape values must be greater than one."
FOOTPRINT_NOT_DETECTED_MESSAGE = "Could not detect an IMS ablation footprint in the postAF image."
MARKERS_NOT_DETECTED_MESSAGE = "Could not find enough ablation marker pixels to estimate an IMS footprint."
EMPTY_RESPONSE_MESSAGE = "PostAF ablation response image is empty."
MISSING_PIXEL_SIZE_MESSAGE = "pixel_size must be supplied when estimating from in-memory image data."
IMS_MARKERS_NOT_DETECTED_MESSAGE = "Could not detect enough IMS foreground pixels to estimate its footprint."


@dataclass(slots=True)
class IMS2PostAFAffineResult:
    """IMS to postAF affine estimation result."""

    matrix_yx_um: np.ndarray
    matrix_yx_px: np.ndarray
    confidence: float
    postaf_corners_yx_px: np.ndarray
    diagnostics: dict[str, ty.Any]

    def to_dict(self) -> dict[str, ty.Any]:
        """Return a JSON serializable representation."""
        data = asdict(self)
        data["matrix_yx_um"] = self.matrix_yx_um.tolist()
        data["matrix_yx_px"] = self.matrix_yx_px.tolist()
        data["postaf_corners_yx_px"] = self.postaf_corners_yx_px.tolist()
        return data


@dataclass(slots=True)
class IMS2PostAFPreview:
    """Low-resolution IMS to postAF preview images."""

    postaf: np.ndarray
    ims: np.ndarray
    overlay: np.ndarray


@dataclass(slots=True)
class _ImageData:
    array: np.ndarray
    pixel_size_yx: NumberPair
    channel_axis: int | None = None

    @property
    def spatial_shape(self) -> Shape2D:
        if self.array.ndim == 2:
            return ty.cast(Shape2D, self.array.shape)
        if self.channel_axis == 0:
            return ty.cast(Shape2D, self.array.shape[1:3])
        return ty.cast(Shape2D, self.array.shape[:2])


def create_ims_postaf_preview(
    postaf_image: PathLike | np.ndarray | sitk.Image,
    ims_image: PathLike | np.ndarray | sitk.Image,
    matrix_yx_px: np.ndarray,
    postaf_pixel_size: float | NumberPair | None = None,
    ims_pixel_size: float | NumberPair | None = None,
    max_analysis_size: int = 4096,
) -> IMS2PostAFPreview:
    """Create preview images of transformed IMS signal over the postAF image.

    Parameters
    ----------
    postaf_image
        Post IMS autofluorescence image or path.
    ims_image
        IMS intensity image or path.
    matrix_yx_px
        Affine matrix mapping IMS preview pixels to postAF preview pixels, in yx order.
    postaf_pixel_size
        PostAF pixel size override, required for in-memory arrays.
    ims_pixel_size
        IMS pixel size override, required for in-memory arrays.
    max_analysis_size
        Maximum long-axis size used when reading image paths for preview.

    Returns
    -------
    IMS2PostAFPreview
        RGB uint8 postAF, transformed IMS, and overlay preview images.
    """
    postaf_data = _read_image_data(
        postaf_image,
        pixel_size=postaf_pixel_size,
        max_analysis_size=max_analysis_size,
    )
    ims_data = _read_image_data(
        ims_image,
        pixel_size=ims_pixel_size,
        max_analysis_size=max_analysis_size,
    )
    postaf_gray = _to_gray_uint8(postaf_data.array, channel_axis=postaf_data.channel_axis)
    ims_gray = _to_gray_uint8(ims_data.array, channel_axis=ims_data.channel_axis)
    warped_ims = cv2.warpAffine(
        ims_gray,
        _yx_to_xy_affine(matrix_yx_px),
        (postaf_gray.shape[1], postaf_gray.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    postaf_channel = postaf_gray
    ims_channel = warped_ims
    postaf_preview = np.dstack([postaf_channel, postaf_channel, postaf_channel])
    ims_preview = np.zeros((*ims_channel.shape, 3), dtype=np.uint8)
    ims_preview[..., 1] = ims_channel
    overlay = np.zeros((*postaf_channel.shape, 3), dtype=np.uint8)
    overlay[..., 0] = np.maximum(postaf_channel, (postaf_channel * 0.35).astype(np.uint8))
    overlay[..., 1] = np.maximum((postaf_channel * 0.35).astype(np.uint8), ims_channel)
    overlay[..., 2] = (postaf_channel * 0.35).astype(np.uint8)
    return IMS2PostAFPreview(postaf=postaf_preview, ims=ims_preview, overlay=overlay)


def estimate_ims_to_postaf_affine(
    postaf_image: PathLike | np.ndarray | sitk.Image,
    ims_image: PathLike | np.ndarray | sitk.Image,
    postaf_pixel_size: float | NumberPair | None = None,
    ims_pixel_size: float | NumberPair | None = None,
    ims_shape: Shape2D | None = None,
    min_confidence: float = 0.05,
    max_analysis_size: int = 4096,
) -> IMS2PostAFAffineResult:
    """Estimate an affine transform from MALDI IMS pixels to postAF image coordinates.

    Parameters
    ----------
    postaf_image
        Post IMS autofluorescence image or path.
    ims_image
        IMS intensity image or path. Non-background IMS pixels are used to estimate the source acquisition footprint.
    postaf_pixel_size
        PostAF pixel size in micrometers, in yx order if two values are supplied. If ``postaf_image`` is a path, this
        value is inferred from the image reader unless supplied.
    ims_pixel_size
        IMS pixel size in micrometers, in yx order if two values are supplied. If ``ims_image`` is a path, this value is
        inferred from the image reader unless supplied.
    ims_shape
        Optional expected IMS shape as ``(rows, columns)``. This is validated against the loaded IMS image but is not
        used as the acquisition footprint.
    min_confidence
        Minimum confidence required before returning a result.
    max_analysis_size
        Maximum long-axis size used when reading image paths for analysis.

    Returns
    -------
    IMS2PostAFAffineResult
        Affine matrices and detection diagnostics.
    """
    postaf_data = _read_image_data(
        postaf_image,
        pixel_size=postaf_pixel_size,
        max_analysis_size=max_analysis_size,
    )
    ims_data = _read_image_data(
        ims_image,
        pixel_size=ims_pixel_size,
        max_analysis_size=max_analysis_size,
    )
    if ims_shape is not None:
        _validate_expected_shape(ims_shape, ims_data.spatial_shape)

    postaf_gray = _to_gray_uint8(postaf_data.array, channel_axis=postaf_data.channel_axis)
    ims_gray = _to_gray_uint8(ims_data.array, channel_axis=ims_data.channel_axis)
    source_corners_yx_px, source_area_px = _detect_source_ims_footprint(ims_gray)
    response = _enhance_ablation_marks(postaf_gray)
    corners_yx_px, confidence, diagnostics = _detect_ims_footprint(
        response=response,
        postaf_shape=postaf_gray.shape,
        postaf_pixel_size_yx=postaf_data.pixel_size_yx,
        ims_footprint_corners_yx_px=source_corners_yx_px,
        ims_footprint_area_px=source_area_px,
        ims_pixel_size_yx=ims_data.pixel_size_yx,
    )
    if confidence < min_confidence:
        msg = f"Detected IMS footprint confidence {confidence:.3f} is below minimum {min_confidence:.3f}."
        raise ValueError(msg)

    matrix_yx_px = _estimate_yx_affine(source_corners_yx_px, corners_yx_px)
    diagnostics["refined"] = False
    diagnostics["ims_source_area_px"] = float(source_area_px)
    diagnostics["ims_source_corners_yx_px"] = source_corners_yx_px.tolist()
    matrix_yx_px, diagnostics = _refine_with_ims_image(
        matrix_yx_px=matrix_yx_px,
        postaf_gray=postaf_gray,
        ims_gray=ims_gray,
        diagnostics=diagnostics,
    )

    matrix_yx_um = _pixel_affine_to_micron_affine(matrix_yx_px, postaf_data.pixel_size_yx, ims_data.pixel_size_yx)
    return IMS2PostAFAffineResult(
        matrix_yx_um=matrix_yx_um,
        matrix_yx_px=matrix_yx_px,
        confidence=confidence,
        postaf_corners_yx_px=corners_yx_px,
        diagnostics=diagnostics,
    )


def _read_image_data(
    image: PathLike | np.ndarray | sitk.Image,
    pixel_size: float | NumberPair | None,
    max_analysis_size: int,
) -> _ImageData:
    if isinstance(image, np.ndarray):
        if pixel_size is None:
            raise ValueError(MISSING_PIXEL_SIZE_MESSAGE)
        return _ImageData(image, _normalize_pair(pixel_size, "pixel_size"), _infer_channel_axis(image))
    if isinstance(image, sitk.Image):
        pixel_size = pixel_size or image.GetSpacing()[::-1]
        array = sitk.GetArrayFromImage(image)
        return _ImageData(array, _normalize_pair(pixel_size, "pixel_size"), _infer_channel_axis(array))
    reader = get_simple_reader(Path(image), init_pyramid=False, auto_pyramid=False, quick=False)
    array, inferred_pixel_size = reader.get_thumbnail(max_analysis_size)
    channel_axis, _n_channels = reader.get_channel_axis_and_n_channels(array.shape)
    pixel_size = pixel_size or inferred_pixel_size
    return _ImageData(
        _materialize_reader_array(Path(image), array), _normalize_pair(pixel_size, "pixel_size"), channel_axis
    )


def _materialize_reader_array(path: Path, array: ty.Any) -> np.ndarray:
    try:
        return np.asarray(array)
    except ValueError as exc:
        if "closed file" not in str(exc):
            raise
        return np.asarray(imread(path))


def _infer_channel_axis(image: np.ndarray) -> int | None:
    if image.ndim != 3:
        return None
    if image.shape[-1] in {3, 4} and image.shape[0] > 4:
        return 2
    return 0


def _normalize_pair(value: float | NumberPair, name: str) -> NumberPair:
    if isinstance(value, int | float):
        value = (float(value), float(value))
    if len(value) != 2:
        msg = f"{name} must be a scalar or a pair of values."
        raise ValueError(msg)
    y_value, x_value = float(value[0]), float(value[1])
    if y_value <= 0 or x_value <= 0:
        msg = f"{name} values must be greater than zero."
        raise ValueError(msg)
    return y_value, x_value


def _normalize_shape(value: Shape2D) -> Shape2D:
    if len(value) != 2:
        raise ValueError(IMS_SHAPE_COUNT_MESSAGE)
    rows, cols = int(value[0]), int(value[1])
    if rows < 2 or cols < 2:
        raise ValueError(IMS_SHAPE_MINIMUM_MESSAGE)
    return rows, cols


def _validate_expected_shape(expected_shape: Shape2D, image_shape: Shape2D) -> None:
    expected_shape = _normalize_shape(expected_shape)
    if expected_shape != image_shape[:2]:
        msg = f"ims_shape {expected_shape} does not match loaded IMS image shape {image_shape[:2]}."
        raise ValueError(msg)


def _to_gray_uint8(image: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
    image = np.squeeze(np.asarray(image))
    if image.ndim == 3:
        if channel_axis == 0:
            image = np.nanmax(image, axis=0)
        elif channel_axis == 2 and image.shape[-1] in {3, 4}:
            code = cv2.COLOR_RGBA2GRAY if image.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
            image = cv2.cvtColor(image, code)
        elif channel_axis == 2:
            image = np.nanmax(image, axis=-1)
        elif image.shape[-1] in {3, 4}:
            code = cv2.COLOR_RGBA2GRAY if image.shape[-1] == 4 else cv2.COLOR_RGB2GRAY
            image = cv2.cvtColor(image, code)
        else:
            image = np.nanmax(image, axis=0)
    if image.ndim != 2:
        msg = f"Expected a 2D, RGB, RGBA, or channel stack image. Got shape {image.shape}."
        raise ValueError(msg)
    image = image.astype(np.float32, copy=False)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [1, 99])
    if math.isclose(float(lo), float(hi)):
        return np.zeros(image.shape, dtype=np.uint8)
    image = np.clip((image - lo) / (hi - lo), 0, 1)
    return (image * 255).astype(np.uint8)


def _enhance_ablation_marks(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    image = clahe.apply(image)
    smooth = cv2.GaussianBlur(image, (0, 0), sigmaX=4.0, sigmaY=4.0)
    response = cv2.absdiff(image, smooth)
    response = cv2.GaussianBlur(response, (3, 3), sigmaX=0)
    return cv2.normalize(response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


def _detect_ims_footprint(
    response: np.ndarray,
    postaf_shape: Shape2D,
    postaf_pixel_size_yx: NumberPair,
    ims_footprint_corners_yx_px: np.ndarray,
    ims_footprint_area_px: float,
    ims_pixel_size_yx: NumberPair,
) -> tuple[np.ndarray, float, dict[str, ty.Any]]:
    threshold = _response_threshold(response)
    high_response_mask = (response >= threshold).astype(np.uint8)
    step_px = min(
        ims_pixel_size_yx[0] / postaf_pixel_size_yx[0],
        ims_pixel_size_yx[1] / postaf_pixel_size_yx[1],
    )
    kernel_size = _odd_int(max(5, min(61, round(step_px * 1.5))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    connected_mask = cv2.morphologyEx(high_response_mask, cv2.MORPH_CLOSE, kernel)
    connected_mask = cv2.dilate(connected_mask, kernel, iterations=1)

    contours, _hierarchy = cv2.findContours(connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(FOOTPRINT_NOT_DETECTED_MESSAGE)

    expected_area = _expected_footprint_area_px(ims_footprint_area_px, ims_pixel_size_yx, postaf_pixel_size_yx)
    expected_geometry = _footprint_geometry(ims_footprint_corners_yx_px, ims_pixel_size_yx)
    best: tuple[float, float, np.ndarray, dict[str, ty.Any]] | None = None
    for contour in contours:
        contour_mask = np.zeros(postaf_shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], contourIdx=-1, color=1, thickness=-1)
        y_coords, x_coords = np.nonzero((high_response_mask > 0) & (contour_mask > 0))
        if len(x_coords) < 8:
            continue
        points_xy = np.column_stack([x_coords, y_coords]).astype(np.float32)
        corners_yx, rect_area = _minimum_area_corners_yx(points_xy)
        if rect_area <= 0:
            continue
        geometry = _footprint_geometry(corners_yx, postaf_pixel_size_yx)
        area_ratio = rect_area / expected_area if expected_area > 0 else 0
        area_score = min(area_ratio, 1 / area_ratio) if area_ratio > 0 else 0
        width_score = _ratio_score(geometry["width"], expected_geometry["width"])
        height_score = _ratio_score(geometry["height"], expected_geometry["height"])
        aspect_score = _ratio_score(geometry["aspect"], expected_geometry["aspect"])
        orientation_score = _orientation_score(geometry["angle"], expected_geometry["angle"])
        axis_score = (0.35 * width_score) + (0.35 * height_score) + (0.20 * aspect_score) + (0.10 * orientation_score)
        density = len(x_coords) / max(rect_area, 1)
        density_score = min(1.0, density / 0.015)
        inside_response = float(np.mean(response[y_coords, x_coords]))
        global_response = float(np.mean(response[response > 0])) if np.any(response > 0) else 1.0
        response_score = min(1.0, inside_response / max(global_response, 1.0))
        confidence = float(
            np.clip(
                (0.45 * axis_score) + (0.25 * area_score) + (0.15 * density_score) + (0.15 * response_score),
                0,
                1,
            )
        )
        diagnostics = {
            "threshold": int(threshold),
            "morphology_kernel_size": int(kernel_size),
            "expected_area_px": float(expected_area),
            "detected_area_px": float(rect_area),
            "area_ratio": float(area_ratio),
            "expected_width_um": float(expected_geometry["width"]),
            "expected_height_um": float(expected_geometry["height"]),
            "expected_aspect": float(expected_geometry["aspect"]),
            "expected_angle": float(expected_geometry["angle"]),
            "detected_width_um": float(geometry["width"]),
            "detected_height_um": float(geometry["height"]),
            "detected_aspect": float(geometry["aspect"]),
            "detected_angle": float(geometry["angle"]),
            "density": float(density),
            "axis_score": float(axis_score),
            "area_score": float(area_score),
            "width_score": float(width_score),
            "height_score": float(height_score),
            "aspect_score": float(aspect_score),
            "orientation_score": float(orientation_score),
            "density_score": float(density_score),
            "response_score": float(response_score),
            "n_contours": len(contours),
        }
        if best is None or confidence > best[0]:
            best = confidence, rect_area, corners_yx, diagnostics

    if best is None:
        raise ValueError(MARKERS_NOT_DETECTED_MESSAGE)
    confidence, _rect_area, corners_yx, diagnostics = best
    return corners_yx, confidence, diagnostics


def _response_threshold(response: np.ndarray) -> int:
    nonzero = response[response > 0]
    if len(nonzero) == 0:
        raise ValueError(EMPTY_RESPONSE_MESSAGE)
    otsu = filters.threshold_otsu(nonzero)
    percentile = np.percentile(nonzero, 90)
    return int(max(otsu, percentile))


def _minimum_area_corners_yx(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    rect = cv2.minAreaRect(points_xy)
    (width, height) = rect[1]
    box_xy = cv2.boxPoints(rect)
    ordered_xy = _order_corners_xy(box_xy)
    corners_yx = ordered_xy[:, ::-1]
    return corners_yx.astype(np.float64), float(width * height)


def _order_corners_xy(corners_xy: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners_xy, dtype=np.float64)
    ordered = np.zeros((4, 2), dtype=np.float64)
    sums = corners.sum(axis=1)
    diffs = corners[:, 0] - corners[:, 1]
    ordered[0] = corners[np.argmin(sums)]
    ordered[2] = corners[np.argmax(sums)]
    ordered[1] = corners[np.argmin(diffs)]
    ordered[3] = corners[np.argmax(diffs)]
    return ordered[[0, 3, 1, 2]]


def _footprint_geometry(corners_yx: np.ndarray, pixel_size_yx: NumberPair) -> dict[str, float]:
    width = _physical_distance(corners_yx[0], corners_yx[1], pixel_size_yx)
    height = _physical_distance(corners_yx[0], corners_yx[2], pixel_size_yx)
    aspect = width / max(height, np.finfo(float).eps)
    vector = corners_yx[1] - corners_yx[0]
    angle = math.degrees(math.atan2(vector[0] * pixel_size_yx[0], vector[1] * pixel_size_yx[1]))
    return {"width": width, "height": height, "aspect": aspect, "angle": angle}


def _physical_distance(start_yx: np.ndarray, end_yx: np.ndarray, pixel_size_yx: NumberPair) -> float:
    delta = (end_yx - start_yx) * np.asarray(pixel_size_yx)
    return float(np.linalg.norm(delta))


def _ratio_score(value: float, expected: float) -> float:
    if value <= 0 or expected <= 0:
        return 0.0
    ratio = value / expected
    return float(min(ratio, 1 / ratio))


def _orientation_score(angle: float, expected_angle: float) -> float:
    delta = abs(((angle - expected_angle + 90) % 180) - 90)
    return float(max(0.0, math.cos(math.radians(delta))) ** 4)


def _detect_source_ims_footprint(ims_gray: np.ndarray) -> tuple[np.ndarray, float]:
    nonzero = ims_gray[ims_gray > 0]
    if len(nonzero) == 0:
        raise ValueError(IMS_MARKERS_NOT_DETECTED_MESSAGE)
    threshold = max(1, int(filters.threshold_otsu(nonzero) * 0.5))
    mask = (ims_gray > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) < 8:
        raise ValueError(IMS_MARKERS_NOT_DETECTED_MESSAGE)
    y_min, y_max = float(np.min(y_coords)), float(np.max(y_coords))
    x_min, x_max = float(np.min(x_coords)), float(np.max(x_coords))
    corners_yx = np.asarray(
        [
            [y_min, x_min],
            [y_min, x_max],
            [y_max, x_min],
            [y_max, x_max],
        ],
        dtype=np.float64,
    )
    return corners_yx, max((y_max - y_min) * (x_max - x_min), 1.0)


def _expected_footprint_area_px(
    ims_footprint_area_px: float,
    ims_pixel_size_yx: NumberPair,
    postaf_pixel_size_yx: NumberPair,
) -> float:
    ims_area_um = ims_footprint_area_px * ims_pixel_size_yx[0] * ims_pixel_size_yx[1]
    postaf_pixel_area_um = postaf_pixel_size_yx[0] * postaf_pixel_size_yx[1]
    return max(ims_area_um / postaf_pixel_area_um, 1.0)


def _estimate_yx_affine(source_yx: np.ndarray, target_yx: np.ndarray) -> np.ndarray:
    source = np.column_stack([source_yx, np.ones(len(source_yx), dtype=np.float64)])
    params, _residuals, _rank, _singular = np.linalg.lstsq(source, target_yx, rcond=None)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :] = params.T
    return matrix


def _pixel_affine_to_micron_affine(
    matrix_yx_px: np.ndarray,
    postaf_pixel_size_yx: NumberPair,
    ims_pixel_size_yx: NumberPair,
) -> np.ndarray:
    postaf_scale = np.diag([postaf_pixel_size_yx[0], postaf_pixel_size_yx[1], 1.0])
    ims_scale_inv = np.diag([1 / ims_pixel_size_yx[0], 1 / ims_pixel_size_yx[1], 1.0])
    return postaf_scale @ matrix_yx_px @ ims_scale_inv


def _refine_with_ims_image(
    matrix_yx_px: np.ndarray,
    postaf_gray: np.ndarray,
    ims_gray: np.ndarray,
    diagnostics: dict[str, ty.Any],
) -> tuple[np.ndarray, dict[str, ty.Any]]:
    if ims_gray.shape[0] < 2 or ims_gray.shape[1] < 2:
        diagnostics["refinement_error"] = "IMS image is too small."
        return matrix_yx_px, diagnostics
    try:
        matrix_xy = _yx_to_xy_affine(matrix_yx_px)
        warped_ims = cv2.warpAffine(
            ims_gray,
            matrix_xy,
            (postaf_gray.shape[1], postaf_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        template = _normalize_for_ecc(postaf_gray)
        moving = _normalize_for_ecc(warped_ims)
        warp_xy = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        correlation, delta_xy = cv2.findTransformECC(template, moving, warp_xy, cv2.MOTION_TRANSLATION, criteria)
    except cv2.error as exc:
        diagnostics["refinement_error"] = str(exc)
        return matrix_yx_px, diagnostics
    if not np.isfinite(correlation) or correlation < 0.05:
        diagnostics["refinement_correlation"] = float(correlation) if np.isfinite(correlation) else None
        diagnostics["refinement_error"] = "ECC correlation was too low."
        return matrix_yx_px, diagnostics
    delta_yx = _xy_to_yx_affine(delta_xy)
    refined = delta_yx @ matrix_yx_px
    diagnostics["refined"] = True
    diagnostics["refinement_correlation"] = float(correlation)
    return refined, diagnostics


def _normalize_for_ecc(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    if float(np.max(image)) > float(np.min(image)):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def _yx_to_xy_affine(matrix_yx: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [matrix_yx[1, 1], matrix_yx[1, 0], matrix_yx[1, 2]],
            [matrix_yx[0, 1], matrix_yx[0, 0], matrix_yx[0, 2]],
        ],
        dtype=np.float32,
    )


def _xy_to_yx_affine(matrix_xy: np.ndarray) -> np.ndarray:
    matrix_yx = np.eye(3, dtype=np.float64)
    matrix_yx[0, 0] = matrix_xy[1, 1]
    matrix_yx[0, 1] = matrix_xy[1, 0]
    matrix_yx[0, 2] = matrix_xy[1, 2]
    matrix_yx[1, 0] = matrix_xy[0, 1]
    matrix_yx[1, 1] = matrix_xy[0, 0]
    matrix_yx[1, 2] = matrix_xy[0, 2]
    return matrix_yx


def _odd_int(value: float) -> int:
    value = max(3, round(value))
    return value if value % 2 else value + 1

from __future__ import annotations

import typing as ty

import numpy as np
import SimpleITK as sitk

SITK_TO_NP_DTYPE = {
    0: np.int8,
    1: np.uint8,
    2: np.int16,
    3: np.uint16,
    4: np.int32,
    5: np.uint32,
    6: np.int64,
    7: np.uint64,
    8: np.float32,
    9: np.float64,
    10: np.complex64,
    11: np.complex64,
    12: np.int8,
    13: np.uint8,
    14: np.int16,
    15: np.int16,
    16: np.int32,
    17: np.int32,
    18: np.int64,
    19: np.int64,
    20: np.float32,
    21: np.float64,
    22: np.uint8,
    23: np.uint16,
    24: np.uint32,
    25: np.uint64,
}
NUMERIC_ELX_PARAMETERS = {
    "CenterOfRotationPoint": np.float64,
    "DefaultPixelValue": np.float64,
    "Direction": np.float64,
    "FixedImageDimension": np.int64,
    "Index": np.int64,
    "MovingImageDimension": np.int64,
    "NumberOfParameters": np.int64,
    "Origin": np.float64,
    "Size": np.int64,
    "Spacing": np.float64,
    "TransformParameters": np.float64,
}
ELX_LINEAR_TRANSFORMS = ["AffineTransform", "EulerTransform", "SimilarityTransform"]
ELX_TO_ITK_INTERPOLATORS: dict[str, ty.Any] = {
    "FinalNearestNeighborInterpolator": sitk.sitkNearestNeighbor,
    "FinalLinearInterpolator": sitk.sitkLinear,
    "FinalBSplineInterpolator": sitk.sitkBSpline,
}

"""Enums."""

from __future__ import annotations

import typing as ty
from enum import Enum

if ty.TYPE_CHECKING:
    import dask.array as da
    import numpy as np
    import zarr


class ImageType(str, Enum):
    """Set the photometric interpretation of the image
    * "FL": background is black (fluorescence)
    * "BF": Background is white (brightfield).
    """

    DARK = "FL"
    LIGHT = "BF"


class CoordinateFlip(str, Enum):
    """Coordinate flip options
    * "h" : horizontal flip
    * "v" : vertical flip.
    """

    HORIZONTAL = "h"
    VERTICAL = "v"


class BackgroundSubtractType(str, Enum):
    """Background subtraction mode."""

    NONE = "none"
    SHARP = "sharp"
    SMOOTH = "smooth"
    BLACKHAT = "blackhat"
    TOPHAT = "tophat"


ProcessingDefaults = ty.Literal["none", "basic", "light", "dark", "he", "pas", "postaf", "mip"]
PreprocessingOptionsWithNone = ty.get_args(ProcessingDefaults)
PreprocessingOptions = PreprocessingOptionsWithNone[1::]

NetworkTypes = ty.Literal[
    "random",
    "kk",
    "planar",
    "spring",
    "spectral",
    "shell",
    "circular",
    "spiral",
    "arf",
]
ArrayLike = ty.Union["np.ndarray", "da.core.Array", "zarr.Array"]
WriterMode = ty.Literal["sitk", "ome-zarr", "ome-tiff", "ome-tiff-by-plane", "ome-tiff-by-tile"]
ValisPreprocessingMethod = ty.Literal[
    "auto",
    # short-name
    "cs",
    "lum",
    "he",
    "mip",
    "i2r",
    # long-name
    "OD",
    "ChannelGetter",
    "ColorfulStandardizer",  # cs
    "Luminosity",  # lum
    "BgColorDistance",
    "StainFlattener",
    "Gray",
    "HEDeconvolution",
    "NoProcessing",
    "HEPreprocessing",  # he
    "MaxIntensityProjection",  # mip
    "I2RegPreprocessor",  # i2r
]
ValisDetectorMethod = ty.Literal[
    # long-name
    "vgg",
    "orb_vgg",
    "boost",
    "latch",
    "daisy",
    "kaze",
    "akaze",
    "brisk",
    "orb",
    "skcensure",
    "skdaisy",
    # "super_point",
    "sensitive_vgg",
    "very_sensitive_vgg",
    # short-name
    "svgg",
    "vsvgg",
]
ValisMatcherMethod = ty.Literal["ransac", "gms"]  # , "super_point", "super_glue"]
ValisInterpolation = ty.Literal["linear", "bicubic"]
ValisCrop = ty.Literal[True, False, "overlap", "reference"]

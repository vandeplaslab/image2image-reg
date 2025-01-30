"""Utility functions for image processing and visualization."""

from __future__ import annotations

import re
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted

if ty.TYPE_CHECKING:
    from valis.registration import Valis


def get_slide_path(registrar: Valis, name: str) -> Path:
    """Find slide path by it's name."""
    slide = registrar.get_slide(name)
    return Path(slide.src_f)


def get_image_files(img_dir: PathLike, ordered: bool = False) -> list[Path]:
    """Get images filenames in img_dir.

    If imgs_ordered is True, then this ensures the returned list is sorted
    properly. Otherwise, the list is sorted lexicographically.

    Parameters
    ----------
    img_dir : str
        Path to directory containing the images.

    ordered: bool, optional
        Whether the order of images already known. If True, the file
        names should start with ascending numbers, with the first image file
        having the smallest number, and the last image file having the largest
        number. If False (the default), the order of images will be determined
        by ordering a distance matrix.

    Returns
    -------
        If `ordered` is True, then this ensures the returned list is sorted
        properly. Otherwise, the list is sorted lexicographically.

    """
    from image2image_io.readers import SUPPORTED_IMAGE_FORMATS

    img_dir = Path(img_dir)
    img_list = []
    for fmt in SUPPORTED_IMAGE_FORMATS:
        img_list.extend(list(img_dir.glob(f"*.{fmt}")))

    # remove duplicate entries
    img_list = list(set(img_list))

    if ordered:
        img_list = natsorted(img_list)
    else:
        img_list.sort()
    return img_list


def order_distance_matrix(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster distance matrix and sort.

    Leaf sorting is accomplished using optimal leaf ordering (Bar-Joseph 2001)

    Parmaters
    ---------
    distance: ndarray
        (N, N) Symmetric distance matrix for N samples

    Returns
    -------
    sorted_d :ndarray
        (N, N) array Distance matrix sorted using optimal leaf ordering

    ordered_leaves : ndarray
        (1, N) array containing the leaves of dendrogram found during
        hierarchical clustering

    optimal_z : ndarray
        ordered linkage matrix

    """
    from fastcluster import linkage
    from scipy.cluster.hierarchy import leaves_list, optimal_leaf_ordering
    from scipy.spatial.distance import squareform

    d = distance.copy()
    sq_d = squareform(d)
    z = linkage(sq_d, "single", preserve_input=True)

    optimal_z = optimal_leaf_ordering(z, sq_d)
    ordered_leaves = leaves_list(optimal_z)

    sorted_d = d[ordered_leaves, :]
    sorted_d = sorted_d[:, ordered_leaves]

    return sorted_d, ordered_leaves, optimal_z


def get_max_image_dimensions(img_list: list[np.ndarray]) -> tuple[int, int]:
    """Find the maximum width and height of all images.

    Parameters
    ----------
    img_list : list
        List of images

    Returns
    -------
    max_wh : tuple
        Maximum width and height of all images

    """
    shapes = [img.shape[0:2] for img in img_list]
    all_w, all_h = list(zip(*shapes))
    max_wh = (max(all_w), max(all_h))
    return max_wh


def get_micro_registration_dimension(registrar: Valis, fraction: float = 0.125, max_size: int = 3000) -> int:
    """Get the size of the micro registration image."""
    try:
        img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
        min_max_size = np.min([np.max(d) for d in img_dims])
        micro_reg_size = np.floor(min_max_size * fraction).astype(int)
    except Exception:  # type: ignore
        micro_reg_size = max_size
        logger.error("Failed to calculate micro registration size.")
    return micro_reg_size


def get_image_name(filename: PathLike) -> str:
    """To get an object's name, remove image type extension from filename."""
    filename = str(filename)
    if re.search(r"\.", filename) is None:
        # Extension already removed
        return filename

    filename = Path(filename).name
    if filename.endswith(".ome.tiff") or filename.endswith(".ome.tif"):
        back_slice_idx = 2
    else:
        back_slice_idx = 1
    img_name = "".join([".".join(filename.split(".")[:-back_slice_idx])])
    return img_name


def get_preprocessor(preprocessor: str | type) -> type:
    """Get pre-processor."""
    import valis.preprocessing as pre_valis

    import image2image_reg.valis.preprocessing as pre_wsireg

    if isinstance(preprocessor, str):
        if hasattr(pre_wsireg, preprocessor):
            preprocessor = getattr(pre_wsireg, preprocessor)
        elif hasattr(pre_valis, preprocessor):
            preprocessor = getattr(pre_valis, preprocessor)
        else:
            raise ValueError(f"Preprocessor {preprocessor} not found.")
    return preprocessor


def get_preprocessing_for_path(path: PathLike) -> list[str, dict]:
    """Get preprocessing kws for specified image."""
    from image2image_io.config import CONFIG
    from image2image_io.readers import get_simple_reader

    with CONFIG.temporary_overwrite(only_last_pyramid=True, init_pyramid=False):
        reader = get_simple_reader(path)
        if reader.is_rgb:
            kws = ["ColorfulStandardizer", {"c": 0.2, "h": 0}]
        else:
            kws = ["MaxIntensityProjection", {"channel_names": reader.channel_names}]
    return kws


def get_feature_detector_str(feature_detector: str) -> str:
    """Get feature detector."""
    available = {
        "vgg": "VggFD",
        "orb_vgg": "OrbVggFD",
        "boost": "BoostFD",
        "latch": "LatchFD",
        "daisy": "DaisyFD",
        "kaze": "KazeFD",
        "akaze": "AkazeFD",
        "brisk": "BriskFD",
        "orb": "OrbFD",
        "skcensure": "CensureVggFD",
        "skdaisy": "DaisyFD",
        "super_point": "SuperPointFD",
        # custom
        "sensitive_vgg": "SensitiveVggFD",
        "svgg": "SensitiveVggFD",
        "very_sensitive_vgg": "VerySensitiveVggFD",
        "vsvgg": "VerySensitiveVggFD",
    }
    feature_detector = feature_detector.lower()
    all_available = list(available.values()) + list(available.keys())
    if feature_detector not in all_available:
        raise ValueError(f"Feature detector {feature_detector} not found. Please one of use: {all_available}")
    return available[feature_detector] if feature_detector in available else feature_detector


def get_feature_detector(feature_detector: str) -> type:
    """Get feature detector object."""
    import valis.feature_detectors as fd_valis

    import image2image_reg.valis.detect as fd_wsireg

    feature_detector = get_feature_detector_str(feature_detector)
    if isinstance(feature_detector, str):
        if hasattr(fd_wsireg, feature_detector):
            feature_detector = getattr(fd_wsireg, feature_detector)
        elif hasattr(fd_valis, feature_detector):
            feature_detector = getattr(fd_valis, feature_detector)
        else:
            raise ValueError(f"Feature detector {feature_detector} not found.")
    return feature_detector


def get_feature_matcher_str(feature_matcher: str) -> str:
    """Standardize feature matcher."""
    feature_matcher = feature_matcher.lower()
    available = {
        "ransac": ("Matcher", "RANSAC"),
        "gms": ("Matcher", "GMS"),
        "super_point": ("SuperPointMatcher", None),
        "super_glue": ("SuperGlueMatcher", None),
    }
    if feature_matcher not in available:
        raise ValueError(f"Feature matcher {feature_matcher} not found. Please one of use: {list(available.keys())}")
    return available[feature_matcher] if feature_matcher in available else feature_matcher


def get_feature_matcher(feature_matcher: str) -> type:
    """Get feature detector object."""
    import valis.feature_matcher as fm_valis

    (feature_matcher, method) = get_feature_matcher_str(feature_matcher)
    if isinstance(feature_matcher, str):
        if hasattr(fm_valis, feature_matcher):
            feature_matcher = getattr(fm_valis, feature_matcher)
        else:
            raise ValueError(f"Feature detector {feature_matcher} not found.")
    if method:
        feature_matcher = feature_matcher(match_filter_method=method)
    else:
        feature_matcher = feature_matcher()
    return feature_matcher


def get_valis_registrar(project_name: str, output_dir: PathLike, init_jvm: bool = False) -> ty.Any:
    """Get Valis registrar if it's available."""
    # initialize java
    if init_jvm:
        from valis import registration

        registration.init_jvm()

    registrar = None
    output_dir = Path(output_dir)
    registrar_path = output_dir / project_name / "data" / f"{project_name}_registrar.pickle"
    if registrar_path.exists():
        import pickle

        with open(registrar_path, "rb") as f:
            registrar = pickle.load(f)
    return registrar


def get_valis_registrar_alt(project_dir: PathLike, name: str, init_jvm: bool = False) -> ty.Any:
    """Get Valis registrar if it's available."""
    # initialize java
    if init_jvm:
        from valis import registration

        registration.init_jvm()

    registrar = None
    registrar_path = get_registrar_path(project_dir, name)
    if registrar_path.exists():
        import pickle

        with open(registrar_path, "rb") as f:
            registrar = pickle.load(f)
    return registrar


def get_registrar_path(project_dir: PathLike, name: str) -> Path:
    """Get registrar path."""
    project_dir = Path(project_dir) / "data"
    path = project_dir / f"{name}_registrar.pickle"
    if not path.exists():
        paths = list(project_dir.glob("*.pickle"))
        if len(paths) == 1:
            path = paths[0]
    return path


def update_registrar_paths(registrar: ty.Any, project_dir: PathLike) -> None:
    """Update registrar paths."""
    project_dir = Path(project_dir)
    data_dir = Path(registrar.data_dir)
    if data_dir != project_dir / "data":
        registrar.data_dir = str(project_dir / "data")
        registrar.set_dst_paths()
        for slide_obj in registrar.slide_dict.values():
            slide_obj.update_results_img_paths()

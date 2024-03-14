"""Utility functions for image processing and visualization."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from natsort import natsorted


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

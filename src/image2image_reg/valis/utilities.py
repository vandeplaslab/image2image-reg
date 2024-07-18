"""Utility functions for image processing and visualization."""

from __future__ import annotations

import re
import typing as ty
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted

from image2image_reg.enums import ValisCrop, ValisInterpolation

if ty.TYPE_CHECKING:
    from valis.registration import Valis


def transform_points(registrar: Valis, source_path: PathLike, x: np.ndarray, y: np.ndarray, crop: str = "reference"):
    """Transform points."""
    from valis.valtils import get_name

    slide_src = registrar.get_slide(get_name(str(source_path)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")
    xy_transformed = slide_src.warp_xy(np.c_[x, y], crop=crop, non_rigid=True)
    return xy_transformed[:, 0], xy_transformed[:, 1]


def transform_points_df(
    registrar: Valis, source_path: PathLike, df: pd.DataFrame, crop: str = "reference"
) -> pd.DataFrame:
    """Transform points in a dataframe.

    Parameters
    ----------
    registrar : Valis
        Valis object
    source_path : PathLike
        Path to source slide
    df : pd.DataFrame
        Dataframe with x and y columns
    crop : str, optional
        Crop method, by default "reference"
    """
    return _transform_points_df(registrar, source_path, df, "x", "y", crop=crop)


def transform_vertices_df(
    registrar: Valis, source_path: PathLike, df: pd.DataFrame, crop: str = "reference"
) -> pd.DataFrame:
    """Transform points in a dataframe.

    Parameters
    ----------
    registrar : Valis
        Valis object
    source_path : PathLike
        Path to source slide
    df : pd.DataFrame
        Dataframe with x and y columns
    crop : str, optional
        Crop method, by default "reference"
    """
    return _transform_points_df(registrar, source_path, df, "vertex_", "vertex_y", crop=crop)


def _transform_points_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    crop: str = "reference",
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError("Dataframe must have x and y columns.")
    x = df[x_key].values
    y = df[y_key].values
    x, y = transform_points(registrar, source_path, x, y, crop=crop)
    if f"{x_key}_transformed" in df.columns:
        df.drop(columns=[f"{x_key}_transformed"], inplace=True)
    if f"{y_key}_transformed" in df.columns:
        df.drop(columns=[f"{y_key}_transformed"], inplace=True)
    df.insert(df.columns.get_loc(x_key), f"{x_key}_transformed", x)
    df.insert(df.columns.get_loc(y_key), f"{y_key}_transformed", y)
    return df


def get_slide_path(registrar: Valis, name: str) -> Path:
    """Find slide path by it's name."""
    slide = registrar.get_slide(name)
    return Path(slide.src_f)


def transform_attached_image(
    registrar: Valis,
    source_path: PathLike,
    paths_to_register: list[PathLike],
    output_dir: PathLike,
    interp_method: str | ValisInterpolation = "bicubic",
    crop: str | bool | ValisCrop = "reference",
    pyramid: int = 0,
) -> list[Path]:
    """Transform valis image."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from valis.slide_tools import vips2numpy
    from valis.valtils import get_name

    output_dir = Path(output_dir)

    # get reference slide and source slide
    slide_ref = None
    if registrar.reference_img_f:
        slide_ref = registrar.get_ref_slide()
        if slide_ref and not Path(slide_ref.src_f).exists():
            raise ValueError(f"Reference slide {slide_ref.src_f} does not exist.")
    else:
        crop = False
        logger.warning("No reference image found. Disabling cropping.")
    slide_src = registrar.get_slide(get_name(str(source_path)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")

    files = []
    for path in paths_to_register:
        reader = get_simple_reader(path)
        output_filename = output_dir / reader.path.name
        if output_filename.exists():
            continue

        # warp image
        warped = slide_src.warp_slide(level=pyramid, interp_method=interp_method, crop=crop, src_f=str(path))
        if not isinstance(warped, np.ndarray):
            warped = vips2numpy(warped)

        # move channel axis to first axis if it's not RGB
        if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not reader.is_rgb:
            warped = np.moveaxis(warped, 2, 0)

        exported = write_ome_tiff_from_array(
            output_filename,
            None,
            warped,
            resolution=slide_ref.resolution if slide_ref else slide_src.resolution,
            channel_names=reader.channel_names,
        )
        files.append(exported)
    return files


def transform_registered_image(
    registrar: Valis,
    output_dir: PathLike,
    interp_method: str | ValisInterpolation = "bicubic",
    crop: str | bool | ValisCrop = "reference",
    non_rigid_reg: bool = True,
    pyramid: int = 0,
) -> list[Path]:
    """Transform valis image."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from valis.slide_tools import vips2numpy

    output_dir = Path(output_dir)
    # export images to OME-TIFFs
    slide_ref = None
    if registrar.reference_img_f:
        slide_ref = registrar.get_ref_slide()
        if slide_ref and not Path(slide_ref.src_f).exists():
            raise ValueError(f"Reference slide {slide_ref.src_f} does not exist.")
    else:
        crop = False
        logger.warning("No reference image found. Disabling cropping.")

    files = []
    for slide_obj in registrar.slide_dict.values():
        if not Path(slide_obj.src_f).exists():
            raise ValueError(f"Slide {slide_ref.src_f} does not exist.")

        reader = get_simple_reader(slide_obj.src_f)
        output_filename = output_dir / reader.path.name
        if output_filename.exists():
            continue

        warped = slide_obj.warp_slide(level=pyramid, interp_method=interp_method, crop=crop, non_rigid=non_rigid_reg)
        if not isinstance(warped, np.ndarray):
            warped = vips2numpy(warped)

        # ensure that RGB remains RGB but AF remain AF
        if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not reader.is_rgb:
            warped = np.moveaxis(warped, 2, 0)

        exported = write_ome_tiff_from_array(
            output_filename,
            None,
            warped,
            resolution=slide_ref.resolution if slide_ref else slide_obj.resolution,
            channel_names=reader.channel_names,
        )
        files.append(exported)
    return files


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

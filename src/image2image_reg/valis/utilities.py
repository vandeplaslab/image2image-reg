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
from tqdm import tqdm

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
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    crop: str = "reference",
    x_key: str = "x",
    y_key: str = "y",
    suffix: str = "_transformed",
    replace: bool = False,
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
    return _transform_points_df(registrar, source_path, df, x_key, y_key, crop=crop, suffix=suffix, replace=replace)


def transform_vertices_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    crop: str = "reference",
    x_key: str = "vertex_x",
    y_key: str = "vertex_y",
    suffix: str = "_transformed",
    replace: bool = False,
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
    return _transform_points_df(registrar, source_path, df, x_key, y_key, crop=crop, suffix=suffix, replace=replace)


def _transform_points_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    crop: str = "reference",
    suffix: str = "_transformed",
    replace: bool = False,
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Dataframe must have '{x_key}' and '{y_key}' columns.")
    if replace and suffix == "_transformed":
        suffix = "_original"

    x = df[x_key].values
    y = df[y_key].values
    x, y = transform_points(registrar, source_path, x, y, crop=crop)
    if f"{x_key}{suffix}" in df.columns:
        df.drop(columns=[f"{x_key}{suffix}"], inplace=True)
    if f"{y_key}{suffix}" in df.columns:
        df.drop(columns=[f"{y_key}{suffix}"], inplace=True)
    if replace:
        df.insert(max(0, df.columns.get_loc(x_key) - 1), f"{x_key}{suffix}", df[x_key])
        df.insert(max(0, df.columns.get_loc(y_key) - 1), f"{y_key}{suffix}", df[y_key])
        df[x_key] = x
        df[y_key] = y
    else:
        df.insert(df.columns.get_loc(x_key), f"{x_key}{suffix}", x)
        df.insert(df.columns.get_loc(y_key), f"{y_key}{suffix}", y)
    return df


def get_slide_path(registrar: Valis, name: str) -> Path:
    """Find slide path by it's name."""
    slide = registrar.get_slide(name)
    return Path(slide.src_f)


def transform_registered_image(
    registrar: Valis,
    output_dir: PathLike,
    interp_method: str | ValisInterpolation = "bicubic",
    crop: str | bool | ValisCrop = "reference",
    non_rigid_reg: bool = True,
    pyramid: int = 0,
    as_uint8: bool = False,
    tile_size: int = 512,
    overwrite: bool = False,
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
        logger.trace(f"Transforming {slide_obj.name}...")
        if not Path(slide_obj.src_f).exists():
            raise ValueError(f"Slide {slide_ref.src_f} does not exist.")

        reader = get_simple_reader(slide_obj.src_f, init_pyramid=False)
        output_filename = output_dir / reader.path.name
        if output_filename.exists() and not overwrite:
            logger.trace(f"File {output_filename} already exists. Moving on...")
            continue

        # warp image and if necessary, convert to numpy
        warped = slide_obj.warp_slide(level=pyramid, interp_method=interp_method, crop=crop, non_rigid=non_rigid_reg)
        if not isinstance(warped, np.ndarray):
            warped = vips2numpy(warped)

        # ensure that RGB remains RGB but AF remain AF
        if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not reader.is_rgb:
            warped = np.moveaxis(warped, 2, 0)

        # write to disk
        exported = write_ome_tiff_from_array(
            output_filename,
            None,
            warped,
            resolution=slide_ref.resolution if slide_ref else slide_obj.resolution,
            channel_names=reader.channel_names,
            as_uint8=as_uint8,
            tile_size=tile_size,
        )
        files.append(exported)
    return files


def transform_attached_image(
    registrar: Valis,
    source_path: PathLike,
    paths_to_register: list[PathLike],
    output_dir: PathLike,
    interp_method: str | ValisInterpolation = "bicubic",
    crop: str | bool | ValisCrop = "reference",
    pyramid: int = 0,
    as_uint8: bool = False,
    tile_size: int = 512,
    overwrite: bool = False,
) -> list[Path]:
    """Transform valis image."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from valis.slide_tools import vips2numpy
    from valis.valtils import get_name

    output_dir = Path(output_dir)

    logger.trace(f"Transforming attached images for {source_path}...")
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
    for path in tqdm(paths_to_register, desc="Transforming attached images"):
        logger.trace(f"Transforming {path} to {source_path}...")
        reader = get_simple_reader(path)
        output_filename = output_dir / reader.path.name
        if output_filename.exists() and not overwrite:
            logger.trace(f"File {output_filename} already exists. Moving on...")
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
            as_uint8=as_uint8,
            tile_size=tile_size,
        )
        files.append(exported)
    return files


def transform_attached_points(
    registrar: Valis,
    attach_to: PathLike,
    output_dir: PathLike,
    paths: list[PathLike],
    pixel_size: float,
    crop: str | bool | ValisCrop = "reference",
    suffix: str = "_previous",
    overwrite: bool = False,
) -> list[Path]:
    """Transform attached points."""
    from image2image_io.readers.points_reader import read_points
    from image2image_io.readers.utilities import get_column_name
    from valis.valtils import get_name

    slide_src = registrar.get_slide(get_name(str(attach_to)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")

    inv_pixel_size = 1 / pixel_size
    is_in_px = pixel_size == 1.0
    paths_ = []
    for path in tqdm(paths, desc="Transforming attached points"):
        # read data
        path = Path(path)
        output_path = output_dir / path.name
        if output_path.exists() and not overwrite:
            logger.trace(f"File {output_path} already exists. Moving on...")
            continue

        df = read_points(path, return_df=True)
        x_key = get_column_name(df, ["x", "x_location", "x_centroid", "x:x", "vertex_x"])
        y_key = get_column_name(df, ["y", "y_location", "y_centroid", "y:y", "vertex_y"])
        if x_key not in df.columns or y_key not in df.columns:
            raise ValueError(f"Invalid columns: {df.columns}")

        # Valis operates in index units so we need to convert from physical units to index units explicitly
        if not is_in_px:
            df[x_key] = df[x_key] * inv_pixel_size
            df[y_key] = df[y_key] * inv_pixel_size

        # transform the points
        df_transformed = transform_points_df(
            registrar,
            slide_src.src_f,
            df,
            x_key=x_key,
            y_key=y_key,
            crop=crop,
            replace=True,
            suffix=suffix,
        )
        if not is_in_px:
            df_transformed[x_key] = df_transformed[x_key] * pixel_size
            df_transformed[y_key] = df_transformed[y_key] * pixel_size

        if path.suffix in [".csv", ".txt", ".tsv"]:
            sep = {"csv": ",", "txt": "\t", "tsv": "\t"}[path.suffix[1:]]
            df_transformed.to_csv(output_path, index=False, sep=sep)
        elif path.suffix == ".parquet":
            df_transformed.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Invalid file extension: {path.suffix}")
        paths_.append(output_path)
    return paths_


def transform_attached_shapes(
    registrar: Valis,
    attach_to: PathLike,
    output_dir: PathLike,
    paths: list[PathLike],
    pixel_size: float,
    crop: str | bool | ValisCrop = "reference",
    overwrite: bool = False,
) -> list[Path]:
    """Transform attached shapes."""
    raise NotImplementedError("Must implement method")


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
        # custom
        "sensitive_vgg": "SensitiveVggFD",
        "svgg": "SensitiveVggFD",
        "very_sensitive_vgg": "VerySensitiveVggFD",
        "vsvgg": "VerySensitiveVggFD",
    }
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
    output_dir = Path(project_dir)
    registrar_path = project_dir / "data" / f"{name}_registrar.pickle"
    if registrar_path.exists():
        import pickle

        with open(registrar_path, "rb") as f:
            registrar = pickle.load(f)
    return registrar

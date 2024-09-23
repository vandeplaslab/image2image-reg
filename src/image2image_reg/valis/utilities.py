"""Utility functions for image processing and visualization."""

from __future__ import annotations

import re
import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from tqdm import tqdm

from image2image_reg.enums import ValisCrop, ValisInterpolation
from image2image_reg.utils.utilities import make_new_name

if ty.TYPE_CHECKING:
    from valis.registration import Slide, Valis


def warp_xy_non_rigid(xy, dxdy, displacement_shape_rc=None):
    from scipy.interpolate import RectBivariateSpline

    single_pt = xy.ndim == 1
    if single_pt:
        xy = np.array([xy])

    if displacement_shape_rc is None:
        displacement_shape_rc = dxdy[0].shape

    bbox = [0, displacement_shape_rc[0], 0, displacement_shape_rc[1]]
    grid_r = np.arange(displacement_shape_rc[0])
    grid_c = np.arange(displacement_shape_rc[1])

    interp_dx = RectBivariateSpline(grid_r, grid_c, dxdy[0], bbox=bbox)
    interp_dy = RectBivariateSpline(grid_r, grid_c, dxdy[1], bbox=bbox)

    nr_x = xy[:, 0] + interp_dx(xy[:, 1], xy[:, 0], grid=False)
    nr_y = xy[:, 1] + interp_dy(xy[:, 1], xy[:, 0], grid=False)

    nr_xy = np.dstack([nr_x, nr_y])[0]
    if single_pt:
        nr_xy = nr_xy[0]

    return nr_xy


def _warp_xy_numpy(
    xy,
    M=None,
    transformation_src_shape_rc=None,
    transformation_dst_shape_rc=None,
    src_shape_rc=None,
    dst_shape_rc=None,
    bk_dxdy=None,
    fwd_dxdy=None,
):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """
    from valis.warp_tools import get_inverse_field, get_warp_scaling_factors, warp_xy_rigid

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(
        transformation_src_shape_rc=transformation_src_shape_rc,
        transformation_dst_shape_rc=transformation_dst_shape_rc,
        src_shape_rc=src_shape_rc,
        dst_shape_rc=dst_shape_rc,
        bk_dxdy=bk_dxdy,
        fwd_dxdy=fwd_dxdy,
    )
    if src_sxy is not None:
        in_src_xy = xy / src_sxy
    else:
        in_src_xy = xy

    if M is not None:
        rigid_xy = warp_xy_rigid(in_src_xy, M).astype(float)
        if not do_non_rigid:
            if dst_sxy is not None:
                return rigid_xy * dst_sxy
            else:
                return rigid_xy
    else:
        rigid_xy = in_src_xy

    if displacement_sxy is not None:
        # displacement was found on scaled version of the rigidly registered image.
        # So move points into new displacement field
        rigid_xy *= displacement_sxy

    if bk_dxdy is not None and fwd_dxdy is None:
        fwd_dxdy = get_inverse_field(bk_dxdy)

    nonrigid_xy = warp_xy_non_rigid(rigid_xy, dxdy=fwd_dxdy, displacement_shape_rc=displacement_shape_rc)

    if dst_sxy is not None:
        nonrigid_xy *= dst_sxy

    return nonrigid_xy


def warp_xy(
    xy,
    M=None,
    transformation_src_shape_rc=None,
    transformation_dst_shape_rc=None,
    src_shape_rc=None,
    dst_shape_rc=None,
    bk_dxdy=None,
    fwd_dxdy=None,
    pt_buffer=100,
):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray, pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray, pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        If `bk_dxdy` or `fwd_dxdy` are pyvips.Image object, then
        pt_buffer` determines the size of the window around the point used to
        get the local displacements.


    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """
    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    warped_xy = _warp_xy_numpy(
        xy,
        M,
        transformation_src_shape_rc=transformation_src_shape_rc,
        transformation_dst_shape_rc=transformation_dst_shape_rc,
        src_shape_rc=src_shape_rc,
        dst_shape_rc=dst_shape_rc,
        bk_dxdy=bk_dxdy,
        fwd_dxdy=fwd_dxdy,
    )
    return warped_xy


def slide_warp_xy(slide: Slide, xy, M=None, slide_level=0, pt_level=0, non_rigid=True, crop=True):
    """Overloaded warping method."""
    from valis.registration import CROP_OVERLAP, CROP_REF

    if M is None:
        M = slide.M

    if np.issubdtype(type(pt_level), np.integer):
        pt_dim_rc = slide.slide_dimensions_wh[pt_level][::-1]
    else:
        pt_dim_rc = np.array(pt_level)

    if np.issubdtype(type(slide_level), np.integer):
        if slide_level != 0:
            if np.issubdtype(type(slide_level), np.integer):
                aligned_slide_shape = slide.val_obj.get_aligned_slide_shape(slide_level)
            else:
                aligned_slide_shape = np.array(slide_level)
        else:
            aligned_slide_shape = slide.aligned_slide_shape_rc
    else:
        aligned_slide_shape = np.array(slide_level)

    if non_rigid:
        fwd_dxdy = slide.fwd_dxdy
    else:
        fwd_dxdy = None

    warped_xy = warp_xy(
        xy,
        M=M,
        transformation_src_shape_rc=slide.processed_img_shape_rc,
        transformation_dst_shape_rc=slide.reg_img_shape_rc,
        src_shape_rc=pt_dim_rc,
        dst_shape_rc=aligned_slide_shape,
        fwd_dxdy=fwd_dxdy,
    )

    crop_method = slide.get_crop_method(crop)
    if crop_method is not False:
        if crop_method == CROP_REF:
            ref_slide = slide.val_obj.get_ref_slide()
            if isinstance(slide_level, int):
                scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[slide_level][::-1]
            else:
                if len(slide_level) == 2:
                    scaled_aligned_shape_rc = slide_level
        elif crop_method == CROP_OVERLAP:
            scaled_aligned_shape_rc = aligned_slide_shape

        crop_bbox_xywh, _ = slide.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
        warped_xy -= crop_bbox_xywh[0:2]
    return warped_xy


def _transform_points(
    slide_src: Slide,
    x: np.ndarray,
    y: np.ndarray,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform points."""
    xy_transformed = slide_src.warp_xy(np.c_[x, y], crop=crop, non_rigid=non_rigid)
    return xy_transformed[:, 0], xy_transformed[:, 1]


def transform_points(
    registrar: Valis,
    source_path: PathLike,
    x: np.ndarray,
    y: np.ndarray,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform points."""
    from valis.valtils import get_name

    # retrieve slide
    slide_src = registrar.get_slide(get_name(str(source_path)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")
    return _transform_points(slide_src, x, y, crop=crop, non_rigid=non_rigid, silent=silent)


def transform_points_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    x_key: str = "x",
    y_key: str = "y",
    suffix: str = "_transformed",
    replace: bool = False,
    silent: bool = False,
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
    non_rigid : bool, optional
        Whether to use non-rigid registration, by default True
    x_key : str, optional
        X column key, by default "x"
    y_key : str, optional
        Y column key, by default "y"
    suffix : str, optional
        Suffix to add to the transformed columns, by default "_transformed"
    replace : bool, optional
        Whether to replace the original columns with the transformed ones, by default False
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    return _transform_points_df(
        registrar,
        source_path,
        df,
        x_key,
        y_key,
        crop=crop,
        non_rigid=non_rigid,
        suffix=suffix,
        replace=replace,
        silent=silent,
    )


def _transform_points_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    suffix: str = "_transformed",
    replace: bool = False,
    silent: bool = False,
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Dataframe must have '{x_key}' and '{y_key}' columns.")
    if replace and suffix == "_transformed":
        suffix = "_original"

    x = df[x_key].values
    y = df[y_key].values
    x, y = transform_points(registrar, source_path, x, y, crop=crop, non_rigid=non_rigid, silent=silent)
    if f"{x_key}{suffix}" in df.columns:
        df.drop(columns=[f"{x_key}{suffix}"], inplace=True)
    if f"{y_key}{suffix}" in df.columns:
        df.drop(columns=[f"{y_key}{suffix}"], inplace=True)
    if replace:
        df.insert(max(0, df.columns.get_loc(x_key)), f"{x_key}{suffix}", df[x_key])
        df.insert(max(0, df.columns.get_loc(y_key)), f"{y_key}{suffix}", df[y_key])
        df[x_key] = x
        df[y_key] = y
    else:
        df.insert(df.columns.get_loc(x_key), f"{x_key}{suffix}", x)
        df.insert(df.columns.get_loc(y_key), f"{y_key}{suffix}", y)
    return df


def transform_vertices_df(
    registrar: Valis,
    source_path: PathLike,
    df: pd.DataFrame,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    x_key: str = "vertex_x",
    y_key: str = "vertex_y",
    suffix: str = "_transformed",
    replace: bool = False,
    silent: bool = False,
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
    non_rigid : bool, optional
        Whether to use non-rigid registration, by default True
    x_key : str, optional
        X column key, by default "vertex_x"
    y_key : str, optional
        Y column key, by default "vertex_y"
    suffix : str, optional
        Suffix to add to the transformed columns, by default "_transformed"
    replace : bool, optional
        Whether to replace the original columns with the transformed ones, by default False
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    return _transform_points_df(
        registrar,
        source_path,
        df,
        x_key,
        y_key,
        crop=crop,
        non_rigid=non_rigid,
        suffix=suffix,
        replace=replace,
        silent=silent,
    )


def get_slide_path(registrar: Valis, name: str) -> Path:
    """Find slide path by it's name."""
    slide = registrar.get_slide(name)
    return Path(slide.src_f)


def transform_registered_image(
    registrar: Valis,
    output_dir: PathLike,
    interp_method: str | ValisInterpolation = "bicubic",
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    pyramid: int = 0,
    as_uint8: bool | None = None,
    rename: bool = True,
    tile_size: int = 512,
    path_to_name_map: dict[Path, str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Transform valis image."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from valis.slide_tools import vips2numpy

    output_dir = Path(output_dir)

    # export images to OME-TIFFs
    ref_slide = None
    if registrar.reference_img_f:
        ref_slide = registrar.get_ref_slide()
        if ref_slide and not Path(ref_slide.src_f).exists():
            raise ValueError(f"Reference slide {ref_slide.src_f} does not exist.")
    else:
        crop = False
        logger.warning("No reference image found. Disabling cropping.")

    if path_to_name_map is None:
        path_to_name_map = {}
    ref_name = ""
    if ref_slide:
        ref_name = path_to_name_map.get(Path(ref_slide.src_f), ref_slide.name)

    files = []
    for slide_obj in registrar.slide_dict.values():
        logger.trace(f"Transforming {slide_obj.name}...")
        if not Path(slide_obj.src_f).exists():
            raise ValueError(f"Slide {ref_slide.src_f} does not exist.")

        reader = get_simple_reader(slide_obj.src_f, init_pyramid=False)
        # renaming involves naming the file such as:
        # <source_name>_to_<reference_name>.ome.tiff
        if rename and path_to_name_map:
            slide_name = path_to_name_map.get(Path(slide_obj.src_f), slide_obj.name)
            filename = make_new_name(slide_name, ref_name)
        else:
            filename = reader.name
        output_filename = output_dir / filename
        if output_filename.exists() and not overwrite:
            logger.trace(f"File {output_filename} already exists. Moving on...")
            continue

        # warp image and if necessary, convert to numpy
        warped = slide_obj.warp_slide(level=pyramid, interp_method=interp_method, crop=crop, non_rigid=non_rigid)
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
            resolution=ref_slide.resolution if ref_slide else slide_obj.resolution,
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
    non_rigid: bool = True,
    pyramid: int = 0,
    as_uint8: bool | None = None,
    rename: bool = True,
    tile_size: int = 512,
    path_to_name_map: dict[Path, str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Transform valis image."""
    from image2image_io.readers import get_simple_reader
    from image2image_io.writers import write_ome_tiff_from_array
    from valis.slide_io import get_slide_reader
    from valis.slide_tools import vips2numpy
    from valis.valtils import get_name

    output_dir = Path(output_dir)

    logger.trace(f"Transforming attached images for {source_path}...")
    # get reference slide and source slide
    ref_slide = None
    if registrar.reference_img_f:
        ref_slide = registrar.get_ref_slide()
        if ref_slide and not Path(ref_slide.src_f).exists():
            raise ValueError(f"Reference slide {ref_slide.src_f} does not exist.")
    else:
        crop = False
        logger.warning("No reference image found. Disabling cropping.")

    if path_to_name_map is None:
        path_to_name_map = {}
    ref_name = ""
    if ref_slide:
        ref_name = path_to_name_map.get(Path(ref_slide.src_f), ref_slide.name)

    slide_src = registrar.get_slide(get_name(str(source_path)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")

    files = []
    for path in tqdm(paths_to_register, desc="Transforming attached images"):
        logger.trace(f"Transforming {path} to {source_path}...")
        reader = get_simple_reader(path)
        # renaming involves naming the file such as:
        # <source_name>_to_<reference_name>.ome.tiff
        if rename and path_to_name_map:
            slide_name = path_to_name_map.get(reader.path, reader.name)
            filename = make_new_name(slide_name, ref_name)
        else:
            filename = reader.name
        output_filename = output_dir / filename
        if output_filename.exists() and not overwrite:
            logger.trace(f"File {output_filename} already exists. Moving on...")
            continue

        reader_cls = get_slide_reader(str(path), series=0)
        if reader_cls is None:
            logger.error(f"Could not find reader for {path}. Skipping...")
            continue

        # warp image
        warped = slide_src.warp_slide(
            level=pyramid,
            interp_method=interp_method,
            crop=crop,
            src_f=str(path),
            non_rigid=non_rigid,
            reader=reader_cls(str(path), series=0),
        )
        if not isinstance(warped, np.ndarray):
            warped = vips2numpy(warped)

        # move channel axis to first axis if it's not RGB
        if warped.ndim == 3 and np.argmin(warped.shape) == 2 and not reader.is_rgb:
            warped = np.moveaxis(warped, 2, 0)

        exported = write_ome_tiff_from_array(
            output_filename,
            None,
            warped,
            resolution=ref_slide.resolution if ref_slide else slide_src.resolution,
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
    source_pixel_size: float,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
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

    is_px = source_pixel_size != 1.0  # if it's not 1.0, then we need to transform the points
    if not is_px:
        source_pixel_size = slide_src.resolution
    inv_source_pixel_size = 1 / source_pixel_size
    target_pixel_size = source_pixel_size
    if registrar.reference_img_f:
        ref_slide: Slide = registrar.get_ref_slide()
        target_pixel_size = ref_slide.resolution

    paths_ = []
    with tqdm(paths) as progress_bar:
        for path in paths:
            path = Path(path)
            progress_bar.set_description(
                f"Transforming {path.name} points to {slide_src.name} (is={is_px} as={is_px};"
                f" s={source_pixel_size:.3f}; s-inv={inv_source_pixel_size:.3f}; t={target_pixel_size:.3f})"
            )
            # read data
            output_path = output_dir / path.name
            if output_path.exists() and not overwrite:
                logger.trace(f"File {output_path} already exists. Moving on...")
                continue

            df = read_points(path, return_df=True)
            x_key = get_column_name(df, ["x", "x_location", "x_centroid", "x:x", "vertex_x"])
            y_key = get_column_name(df, ["y", "y_location", "y_centroid", "y:y", "vertex_y"])
            if x_key not in df.columns or y_key not in df.columns:
                raise ValueError(f"Invalid columns: {df.columns}")
            # change dtype to float64
            df[x_key] = df[x_key].astype(np.float64)
            df[y_key] = df[y_key].astype(np.float64)

            # Valis operates in index units, so we need to convert from physical units to index units explicitly
            df[x_key], df[y_key] = _transform_original_from_um_to_px(df[x_key], df[y_key], is_px, source_pixel_size)

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
                non_rigid=non_rigid,
            )
            df_transformed[x_key], df_transformed[y_key] = _transform_transformed_from_px_to_um(
                df_transformed[x_key], df_transformed[y_key], is_px, target_pixel_size
            )

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
    source_pixel_size: float,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    overwrite: bool = False,
) -> list[Path]:
    """Transform attached shapes."""
    import json

    from image2image_io.readers.shapes_reader import ShapesReader
    from valis.valtils import get_name

    slide_src = registrar.get_slide(get_name(str(attach_to)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")

    is_px = source_pixel_size != 1.0  # if it's not 1.0, then we need to transform the points
    if not is_px:
        source_pixel_size = slide_src.resolution
    target_pixel_size = source_pixel_size
    if registrar.reference_img_f:
        ref_slide: Slide = registrar.get_ref_slide()
        target_pixel_size = ref_slide.resolution

    paths_ = []
    for path in tqdm(paths, desc="Transforming attached points"):
        # read data
        path = Path(path)
        output_path = output_dir / path.name
        if output_path.exists() and not overwrite:
            logger.trace(f"File {output_path} already exists. Moving on...")
            continue

        reader = ShapesReader(path)
        geojson_data = deepcopy(reader.geojson_data)
        if isinstance(geojson_data, list):
            if "type" in geojson_data[0] and geojson_data[0]["type"] == "Feature":
                geojson_data = _transform_geojson_features(
                    geojson_data,
                    slide_src,
                    is_px=is_px,
                    as_px=is_px,
                    source_pixel_size=source_pixel_size,
                    target_pixel_size=target_pixel_size,
                    crop=crop,
                    non_rigid=non_rigid,
                )
            else:
                raise ValueError("Invalid GeoJSON data.")

        with open(output_path, "w") as f:
            json.dump(geojson_data, f, indent=1)
        paths_.append(output_path)
    return paths_


def _transform_geojson_features(
    geojson_data: list[dict],
    slide_src: Slide,
    is_px: bool,
    as_px: bool,
    source_pixel_size: float = 1.0,
    target_pixel_size: float = 1.0,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
) -> list[dict]:
    inv_source_pixel_size = 1 / source_pixel_size

    # iterate over features and depending on the geometry type, transform the coordinates
    result = []
    for feature in geojson_data:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            x, y = geometry["coordinates"]
            x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
            x, y = _transform_points(slide_src, x, y, crop=crop, non_rigid=non_rigid)
            x, y = _transform_transformed_from_px_to_um(x, y, as_px, target_pixel_size)
            geometry["coordinates"] = [x[0], y[0]]
        elif geometry["type"] == "Polygon":
            for i, ring in enumerate(
                tqdm(geometry["coordinates"], desc="Transforming Polygon", leave=False, mininterval=1, disable=True)
            ):
                x, y = np.array(ring).T
                x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
                x, y = _transform_points(slide_src, x, y, crop=crop, non_rigid=non_rigid)
                x, y = _transform_transformed_from_px_to_um(x, y, as_px, target_pixel_size)
                geometry["coordinates"][i] = np.c_[x, y].tolist()
        elif geometry["type"] == "MultiPolygon":
            for j, polygon in enumerate(geometry["coordinates"]):
                for i, ring in enumerate(tqdm(polygon, desc="Transforming MultiPolygon", leave=False, mininterval=1)):
                    x, y = np.array(ring).T
                    x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
                    x, y = _transform_points(slide_src, x, y, crop=crop, non_rigid=non_rigid)
                    x, y = _transform_transformed_from_px_to_um(x, y, as_px, target_pixel_size)
                    geometry["coordinates"][j][i] = np.c_[x, y].tolist()
        result.append(feature)
    return result


def _transform_original_from_um_to_px(
    x: np.ndarray, y: np.ndarray, is_px: bool, source_pixel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    inv_source_pixel_size = 1 / source_pixel_size
    if is_px:  # no need to transform since it's already in pixel coordinates
        return x, y
    # convert from um to pixel by multiplying by the inverse of the pixel size
    return x * inv_source_pixel_size, y * inv_source_pixel_size


def _transform_transformed_from_px_to_um(
    x: np.ndarray, y: np.ndarray, as_px: bool, target_pixel_size: float
) -> tuple[np.ndarray, np.ndarray]:
    if as_px:  # no need to transform since it's already in pixel coordinates
        return x, y
    # convert from px to um by multiplying by the pixel size
    return x * target_pixel_size, y * target_pixel_size


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
    registrar_path = Path(project_dir) / "data" / f"{name}_registrar.pickle"
    if registrar_path.exists():
        import pickle

        with open(registrar_path, "rb") as f:
            registrar = pickle.load(f)
    return registrar


def update_registrar_paths(registrar: ty.Any, project_dir: PathLike):
    """Update registrar paths."""
    project_dir = Path(project_dir)
    data_dir = Path(registrar.data_dir)
    if data_dir != project_dir / "data":
        registrar.data_dir = str(project_dir / "data")
        registrar.set_dst_paths()
        for slide_obj in registrar.slide_dict.values():
            slide_obj.update_results_img_paths()

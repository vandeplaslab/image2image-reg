from __future__ import annotations

import typing as ty
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike
from loguru import logger
from tqdm import tqdm

from image2image_reg.enums import ValisCrop, ValisInterpolation
from image2image_reg.utils.transform_utils import (
    _cleanup_transform_coordinate_image,
    _convert_df_to_geojson,
    _convert_geojson_to_df,
    _filter_transform_coordinate_image,
    _prepare_transform_coordinate_image,
    _replace_column,
    _transform_original_from_um_to_px,
    _transform_transformed_from_px_to_um,
)
from image2image_reg.utils.utilities import make_new_name

if ty.TYPE_CHECKING:
    from valis.registration import Slide, Valis


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


def _transform_points(
    slide_src: Slide,
    x: np.ndarray,
    y: np.ndarray,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform points."""
    # xy_transformed = slide_src.warp_xy(np.c_[x, y], crop=crop, non_rigid=non_rigid)
    slide_ref = slide_src.val_obj.get_ref_slide()
    if slide_ref is None:
        xy_transformed = slide_src.warp_xy(np.c_[x, y], crop=crop, non_rigid=non_rigid)
    else:
        xy_transformed = slide_src.warp_xy_from_to(np.c_[x, y], slide_ref, non_rigid=non_rigid)
    return xy_transformed[:, 0], xy_transformed[:, 1]


def transform_points_as_image(
    registrar: Valis,
    source_path: PathLike,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Transform points."""
    from valis.valtils import get_name

    # retrieve slide
    slide_src = registrar.get_slide(get_name(str(source_path)))
    if not Path(slide_src.src_f).exists():
        raise ValueError(f"Source slide {slide_src.src_f} does not exist.")
    return _transform_points_as_image(slide_src, x, y, df, crop=crop, non_rigid=non_rigid, silent=silent)


def _transform_points_as_image(
    slide_src: Slide,
    x: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    height, width = slide_src.slide_dimensions_wh[0][::-1]
    image_of_index, index_of_coords, x, y = _prepare_transform_coordinate_image(height, width, x, y)
    image_of_index_transformed, _ = _transform_coordinate_image(
        slide_src, image_of_index, crop=crop, non_rigid=non_rigid
    )
    new_x, new_y, _ = _cleanup_transform_coordinate_image(image_of_index_transformed, index_of_coords)
    return new_x, new_y, df


def _transform_coordinate_image(
    slide_src: Slide,
    image_of_index: np.ndarray,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    slide_ref = slide_src.val_obj.get_ref_slide()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if slide_ref:
            image_of_index_ = slide_src.warp_img_from_to(
                image_of_index, slide_ref, non_rigid=non_rigid, interp_method="nearest"
            )
        else:
            image_of_index_ = slide_src.warp_img(
                image_of_index, crop=crop, non_rigid=non_rigid, interp_method="nearest"
            )
    return image_of_index_, image_of_index_.flatten()


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
    as_image: bool = False,
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
    as_image : bool, optional
        Whether the points are in image coordinates, by default False
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
        as_image=as_image,
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
    as_image: bool = False,
    silent: bool = False,
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Dataframe must have '{x_key}' and '{y_key}' columns.")
    if replace and suffix == "_transformed":
        suffix = "_original"

    x = df[x_key].values
    y = df[y_key].values
    if as_image:
        x, y, df = transform_points_as_image(
            registrar, source_path, x, y, df, crop=crop, non_rigid=non_rigid, silent=silent
        )
    else:
        x, y = transform_points(registrar, source_path, x, y, crop=crop, non_rigid=non_rigid, silent=silent)
    df = _replace_column(df, x, y, x_key, y_key, suffix, replace)
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
        try:
            warped = slide_src.warp_slide(
                level=pyramid,
                interp_method=interp_method,
                crop=crop,
                src_f=str(path),
                non_rigid=non_rigid,
                reader=reader_cls(str(path), series=0),
            )
        except TypeError:
            warped = slide_src.warp_slide(
                level=pyramid,
                interp_method=interp_method,
                crop=crop,
                src_f=str(path),
                non_rigid=non_rigid,
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
    as_image: bool = False,
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
            df.loc[:, x_key] = df[x_key].astype(np.float64)
            df.loc[:, y_key] = df[y_key].astype(np.float64)

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
                as_image=as_image,
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
    as_image: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Transform attached shapes."""
    from image2image_io.readers.shapes_reader import ShapesReader
    from koyo.json import write_json_data
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
                if as_image:
                    geojson_data = _transform_geojson_features_as_image(
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

        write_json_data(output_path, geojson_data, compress=True, check_existing=False)
        paths_.append(output_path)
    return paths_


def _transform_geojson_features_as_image(
    geojson_data: list[dict],
    slide_src: Slide,
    is_px: bool,
    as_px: bool,
    source_pixel_size: float = 1.0,
    target_pixel_size: float = 1.0,
    crop: str | bool | ValisCrop = "reference",
    non_rigid: bool = True,
) -> list[dict]:
    df, n_to_prop = _convert_geojson_to_df(geojson_data, is_px, source_pixel_size)
    x, y, df = _transform_points_as_image(slide_src, df.x.values, df.y.values, df, crop=crop, non_rigid=non_rigid)
    height, width = slide_src.slide_dimensions_wh[0][::-1]
    x, y, df = _filter_transform_coordinate_image(height, width, x, y, df)
    return _convert_df_to_geojson(df, x, y, as_px, target_pixel_size, n_to_prop=n_to_prop)


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
    # inv_source_pixel_size = 1 / source_pixel_size

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

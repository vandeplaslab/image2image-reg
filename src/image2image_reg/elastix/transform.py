from __future__ import annotations

import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike
from koyo.utilities import random_chunks
from loguru import logger
from tqdm import tqdm

from image2image_reg.elastix.transform_sequence import TransformSequence
from image2image_reg.utils.transform_utils import (
    MULTIPLIER,
    ClipFunc,
    _cleanup_transform_coordinate_image,
    _clip_noop,
    _convert_df_to_geojson,
    _convert_geojson_to_df,
    _filter_transform_coordinate_image,
    _get_clip_func,
    _prepare_transform_coordinate_image,
    _replace_column,
    _transform_original_from_um_to_px,
    _transform_transformed_from_px_to_um,
)

if ty.TYPE_CHECKING:
    from image2image_reg.wrapper import ImageWrapper


def transform_points(
    seq: TransformSequence,
    x: np.ndarray,
    y: np.ndarray,
    in_px: bool = False,
    as_px: bool = False,
    source_pixel_size: float = 1.0,
    clip_func: ClipFunc = _clip_noop,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform points.

    Parameters
    ----------
    seq : TransformSequence
        Transform sequence
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    in_px : bool, optional
        Whether input coordinates are in pixels or physical units, by default False
    as_px : bool, optional
        Whether to return coordinates in pixels or physical units, by default False
    source_pixel_size : float, optional
        Pixel size of the source image, by default 1.0
    clip_func : callable, optional
        Fix points that are outside of the image, by default _clip_noop
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    transformed_xy = seq.transform_points(
        np.c_[x, y], is_px=in_px, as_px=as_px, source_pixel_size=source_pixel_size, silent=silent
    )
    tx, ty, mask = clip_func(x=transformed_xy[:, 0], y=transformed_xy[:, 1])
    return tx, ty


def transform_points_as_image(
    seq: TransformSequence,
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
    df: pd.DataFrame,
    in_px: bool = False,
    as_px: bool = False,
    source_pixel_size: float = 1.0,
    silent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform points.

    Parameters
    ----------
    seq : TransformSequence
        Transform sequence
    x : np.ndarray
        X coordinates
    y : np.ndarray
        Y coordinates
    height: int,
        Height of the image
    width: int,
        Width of the image
    df : pd.DataFrame
        Dataframe with x and y columns
    in_px : bool, optional
        Whether input coordinates are in pixels or physical units, by default False
    as_px : bool, optional
        Whether to return coordinates in pixels or physical units, by default False
    source_pixel_size : float, optional
        Pixel size of the source image, by default 1.0
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    target_pixel_size = seq.output_spacing[0] * MULTIPLIER
    x, y = _transform_original_from_um_to_px(x, y, in_px, source_pixel_size)
    indices = np.arange(len(x))
    image_of_index, index_of_coords, _, _ = _prepare_transform_coordinate_image(height, width, x, y)
    image_of_index_transformed, _ = _transform_coordinate_image(seq, image_of_index)
    transformed_x, transformed_y, failed_mask = _cleanup_transform_coordinate_image(
        image_of_index_transformed, index_of_coords
    )

    # indices = indices[failed_indices]
    n_failed = failed_mask.sum()
    failed_indices = indices[failed_mask]
    max_iter = 0
    while n_failed > 0 and max_iter < 10:
        logger.warning(f"Failed to transform {n_failed:,} points. Iteration {max_iter + 1}.")
        n_failed_ = 0
        failed_mask_ = np.full(len(x), False)
        for failed_subset in tqdm(
            random_chunks(failed_indices.copy(), n_tasks=4), desc="Retrying failed points...", leave=False, total=4
        ):
            failed_subset = np.sort(failed_subset)  # sort since they were randomly sub-sampled
            image_of_index, index_of_coords, _, _ = _prepare_transform_coordinate_image(
                height, width, x[failed_subset], y[failed_subset]
            )
            image_of_index_transformed, _ = _transform_coordinate_image(seq, image_of_index)
            transformed_x_, transformed_y_, failed_mask_in_subset = _cleanup_transform_coordinate_image(
                image_of_index_transformed, index_of_coords
            )
            if transformed_x_.size == 0:
                logger.error(f"Failed to transform {len(failed_subset):,} points - but are no improvements...")
                n_failed = 0

            failed_mask_[failed_subset[failed_mask_in_subset]] = True
            transformed_x[failed_subset] = transformed_x_
            transformed_y[failed_subset] = transformed_y_
            n_failed_ += failed_mask_in_subset.sum()
        failed_indices = indices[failed_mask_]
        n_failed = n_failed_
        max_iter += 1

        # logger.warning(f"Failed to transform {n_failed:,} points. Iteration {max_iter}.")
        # indices_before_all = indices[failed_mask]
        # # only retain the failed points (with True)
        # failed_mask_ = np.full(n_failed, True)
        # # failed_mask_ = failed_mask[indices_before_all]
        # image_of_index, index_of_coords, _, _ = _prepare_transform_coordinate_image(
        #     height, width, x[indices_before_all], y[indices_before_all]
        # )
        # image_of_index_transformed, _ = _transform_coordinate_image(seq, image_of_index)
        # transformed_x_, transformed_y_, failed_mask = _cleanup_transform_coordinate_image(
        #     image_of_index_transformed, index_of_coords
        # )
        # # get the difference between the before and after masks, anything that was in the 'before' should be updated
        # # and anything that wasn't, should be left as is
        # # get values that are true and false
        # failed_fixed_mask = failed_mask & ~failed_mask_
        # # get the indices of the fixed points
        # indices_to_fix = indices_before_all[failed_fixed_mask]
        # # get the values of the fixed points
        # transformed_x[indices_to_fix] = transformed_x_[failed_fixed_mask]
        # transformed_y[indices_to_fix] = transformed_y_[failed_fixed_mask]
        # indices = indices_before_all
        # n_failed = failed_mask.sum()
        # max_iter += 1
    transformed_x, transformed_y = _transform_transformed_from_px_to_um(
        transformed_x, transformed_y, as_px, target_pixel_size
    )
    return transformed_x, transformed_y


def _transform_coordinate_image(
    seq: TransformSequence, image_of_index: np.ndarray, scale: tuple[float, float] = (1, 1)
) -> tuple[np.ndarray, np.ndarray]:
    # convert image to Image
    import SimpleITK as sitk

    # set output spacing
    seq.set_output_spacing((0.163, 0.163))

    image_of_index = sitk.GetImageFromArray(image_of_index)
    image_of_index.SetSpacing((scale[0] / MULTIPLIER, scale[1] / MULTIPLIER))
    image_of_index_ = sitk.GetArrayFromImage(seq(image_of_index))
    return image_of_index_, image_of_index_.flatten()


def transform_points_df(
    seq: TransformSequence,
    df: pd.DataFrame,
    in_px: bool = False,
    as_px: bool = False,
    x_key: str = "x",
    y_key: str = "y",
    suffix: str = "_transformed",
    replace: bool = False,
    source_pixel_size: float = 1.0,
    as_image: bool = False,
    image_shape: tuple[int, int] = (0, 0),
    silent: bool = False,
    clip_func: ClipFunc = _clip_noop,
) -> pd.DataFrame:
    """Transform points in a dataframe.

    Parameters
    ----------
    seq : TransformSequence
        Transform sequence
    df : pd.DataFrame
        Dataframe with x and y columns
    in_px : bool, optional
        Whether input coordinates are in pixels or physical units, by default False
    as_px : bool, optional
        Whether to return coordinates in pixels or physical units, by default False
    x_key : str, optional
        X column key, by default "x"
    y_key : str, optional
        Y column key, by default "y"
    suffix : str, optional
        Suffix to add to the transformed columns, by default "_transformed"
    replace : bool, optional
        Whether to replace the original columns with the transformed ones, by default False
    source_pixel_size : float, optional
        Pixel size of the source image, by default 1.0
    as_image : bool, optional
        Whether to treat the points as an image, by default False
    image_shape : tuple[int, int], optional
        Shape of the input image, only used if `as_image` is True, by default (0, 0)
    silent : bool, optional
        Whether to show progress bar, by default False
    clip_func : callable, optional
        Clip function, by default _clip_noop
    """
    return _transform_points_df(
        seq,
        df,
        x_key,
        y_key,
        in_px=in_px,
        as_px=as_px,
        suffix=suffix,
        replace=replace,
        source_pixel_size=source_pixel_size,
        silent=silent,
        as_image=as_image,
        image_shape=image_shape,
        clip_func=clip_func,
    )


def transform_vertices_df(
    seq: TransformSequence,
    df: pd.DataFrame,
    in_px: bool = False,
    as_px: bool = False,
    x_key: str = "vertex_x",
    y_key: str = "vertex_y",
    suffix: str = "_transformed",
    replace: bool = False,
    source_pixel_size: float = 1.0,
    silent: bool = False,
) -> pd.DataFrame:
    """Transform points in a dataframe.

    Parameters
    ----------
    seq : TransformSequence
        Transform sequence
    df : pd.DataFrame
        Dataframe with x and y columns
    in_px : bool, optional
        Whether input coordinates are in pixels or physical units, by default False
    as_px : bool, optional
        Whether to return coordinates in pixels or physical units, by default False
    x_key : str, optional
        X column key, by default "vertex_x"
    y_key : str, optional
        Y column key, by default "vertex_y"
    suffix : str, optional
        Suffix to add to the transformed columns, by default "_transformed"
    replace : bool, optional
        Whether to replace the original columns with the transformed ones, by default False
    source_pixel_size : float, optional
        Pixel size of the source image, by default 1.0
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    return _transform_points_df(
        seq,
        df,
        x_key,
        y_key,
        in_px=in_px,
        as_px=as_px,
        suffix=suffix,
        replace=replace,
        source_pixel_size=source_pixel_size,
    )


def _transform_points_df(
    seq: TransformSequence,
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    in_px: bool = False,
    as_px: bool = False,
    suffix: str = "_transformed",
    replace: bool = False,
    source_pixel_size: float = 1.0,
    as_image: bool = False,
    image_shape: tuple[int, int] = (0, 0),
    silent: bool = False,
    clip_func: ClipFunc = _clip_noop,
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Dataframe must have '{x_key}' and '{y_key}' columns.")
    if replace and suffix == "_transformed":
        suffix = "_original"

    x = df[x_key].values
    y = df[y_key].values
    if as_image:
        height, width = image_shape
        x, y = transform_points_as_image(
            seq,
            x,
            y,
            height,
            width,
            df=df,
            in_px=in_px,
            as_px=as_px,
            source_pixel_size=source_pixel_size,
            silent=silent,
        )
    else:
        x, y = transform_points(seq, x, y, in_px=in_px, as_px=as_px, source_pixel_size=source_pixel_size, silent=silent)
        x, y, mask = clip_func(x, y)
        if mask is not None:
            df = df[mask]
    if len(df) == len(x):
        df = _replace_column(df, x, y, x_key, y_key, suffix, replace)
    return df


def transform_attached_point(
    transform_sequence: TransformSequence,
    path: PathLike,
    source_pixel_size: float,
    output_path: PathLike,
    silent: bool = False,
    as_image: bool = False,
    image_shape: tuple[int, int] = (0, 0),
    clip: str = "ignore",
) -> Path:
    """Transform points data."""
    from image2image_io.readers.points_reader import read_points
    from image2image_io.readers.shapes_reader import get_shape_columns

    pd.options.mode.chained_assignment = None
    is_in_px = source_pixel_size != 1.0

    # read data
    path = Path(path)
    x_key, y_key, group_by = get_shape_columns(path)
    df = read_points(path, return_df=True)
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Invalid columns: {df.columns}")

    if as_image:
        height, width = image_shape
        if height == 0 or width == 0:
            raise ValueError("Invalid image shape.")

    # don't remove all points if data is not actually a shape in the form of points
    if not group_by and clip == "remove":
        clip = "part-remove"
    clip_func = _get_clip_func(transform_sequence, clip, as_px=is_in_px)
    need_to_split = len(df[group_by].unique()) != len(df) if group_by else False
    if (group_by and need_to_split) and not as_image:
        df_transformed = []
        n_removed = 0
        for _group, df_group in tqdm(df.groupby(group_by), desc="Transforming groups", leave=False, mininterval=1):
            df_group_transformed = transform_points_df(
                transform_sequence,
                df_group.copy(),
                in_px=is_in_px,
                as_px=is_in_px,
                x_key=x_key,
                y_key=y_key,
                replace=True,
                source_pixel_size=source_pixel_size,
                silent=True,
                as_image=as_image,
                image_shape=image_shape,
                clip_func=clip_func,
            )
            if len(df_group_transformed) > 0:
                df_transformed.append(df_group_transformed)
            else:
                n_removed += 1
        if n_removed > 0:
            logger.warning(f"Removed {n_removed:,} groups with no points - {len(df_transformed)} were kept.")
        if df_transformed:
            df_transformed = pd.concat(df_transformed)
            df_transformed = remove_invalid_points(df_transformed, group_by)
        else:
            df_transformed = pd.DataFrame()
    else:
        df_transformed = transform_points_df(
            transform_sequence,
            df.copy(),
            in_px=is_in_px,
            as_px=is_in_px,
            x_key=x_key,
            y_key=y_key,
            replace=True,
            source_pixel_size=source_pixel_size,
            silent=silent,
            as_image=as_image,
            image_shape=image_shape,
            clip_func=clip_func,
        )

    if path.suffix in [".csv", ".txt", ".tsv"]:
        sep = {"csv": ",", "txt": "\t", "tsv": "\t"}[path.suffix[1:]]
        df_transformed.to_csv(output_path, index=False, sep=sep)
    elif path.suffix == ".parquet":
        df_transformed.to_parquet(output_path, index=False)
    return Path(output_path)


def remove_invalid_points(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """Remove invalid points (e.g. have fewer than 2 points)."""
    if group_by not in df.columns:
        raise ValueError(f"Invalid columns: {df.columns}")
    # remove any groups that have fewer than two entries by using group_by
    df = df.groupby(group_by).filter(lambda x: len(x) > 2)
    return df


def transform_attached_shape(
    transform_sequence: TransformSequence,
    path: PathLike,
    source_pixel_size: float,
    output_path: PathLike,
    silent: bool = False,
    mode: ty.Literal["dataframe", "image", "geojson"] = "geojson",
    image_shape: tuple[int, int] = (0, 0),
    clip: str = "ignore",
) -> Path:
    """Transform points data."""
    from image2image_io.readers.shapes_reader import ShapesReader
    from koyo.json import write_json_data

    # if value is equal to 1.0, then the coordinates are in pixels
    is_in_px = source_pixel_size != 1.0

    _get_clip_func(transform_sequence, clip)
    if mode == "image":
        height, width = image_shape
        if height == 0 or width == 0:
            raise ValueError("Invalid image shape.")

    reader = ShapesReader(path)
    geojson_data = deepcopy(reader.geojson_data)
    if isinstance(geojson_data, list):
        if "type" in geojson_data[0] and geojson_data[0]["type"] == "Feature":
            if transform_sequence is not None:
                if mode == "image":
                    geojson_data = _transform_geojson_features_as_image(
                        geojson_data,
                        transform_sequence,
                        image_shape=image_shape,
                        is_px=is_in_px,
                        as_px=is_in_px,
                        source_pixel_size=source_pixel_size,
                    )
                elif mode == "dataframe":
                    geojson_data = _transform_geojson_features_as_df(
                        geojson_data,
                        transform_sequence,
                        is_px=is_in_px,
                        as_px=is_in_px,
                        clip=clip,
                        source_pixel_size=source_pixel_size,
                    )
                elif mode == "geojson":
                    geojson_data = _transform_geojson_features(
                        geojson_data,
                        transform_sequence,
                        in_px=is_in_px,
                        as_px=is_in_px,
                        source_pixel_size=source_pixel_size,
                        silent=silent,
                        clip=clip,
                    )
        else:
            raise ValueError("Invalid GeoJSON data.")
    write_json_data(output_path, geojson_data, compress=True, check_existing=False)
    return Path(output_path)


def _transform_geojson_features_as_image(
    geojson_data: list[dict],
    transform_sequence: TransformSequence,
    image_shape: tuple[int, int],
    is_px: bool,
    as_px: bool,
    source_pixel_size: float = 1.0,
) -> list[dict]:
    target_pixel_size = transform_sequence.output_spacing[0]

    height, width = image_shape
    df, n_to_prop = _convert_geojson_to_df(geojson_data, is_px, source_pixel_size)
    x, y = transform_points_as_image(
        transform_sequence,
        df.x.values,
        df.y.values,
        height,
        width,
        df=df,
        in_px=is_px,
        as_px=as_px,
        source_pixel_size=source_pixel_size,
    )
    x, y, df = _filter_transform_coordinate_image(*image_shape, x, y, df)
    return _convert_df_to_geojson(df, as_px, target_pixel_size, n_to_prop=n_to_prop)


def _transform_geojson_features_as_df(
    geojson_data: list[dict],
    transform_sequence: TransformSequence,
    is_px: bool,
    as_px: bool,
    clip: str = "ignore",
    source_pixel_size: float = 1.0,
) -> list[dict]:
    target_pixel_size = transform_sequence.output_spacing[0]

    df, n_to_prop = _convert_geojson_to_df(geojson_data, is_px, source_pixel_size)
    if df.unique_index.nunique() > 1:
        group_by = "unique_index"
    else:
        group_by = "outer"
    # don't remove all points if data is not actually a shape in the form of points
    if not group_by and clip == "remove":
        clip = "part-remove"
    clip_func = _get_clip_func(transform_sequence, clip, as_px=as_px)
    df_transformed = []
    counter = 0
    to_remove = []
    for index, (_group, df_group) in enumerate(
        tqdm(df.groupby(group_by), desc="Transforming groups", leave=False, mininterval=1)
    ):
        df_group_transformed = transform_points_df(
            transform_sequence,
            df_group.copy(),
            in_px=is_px,
            as_px=as_px,
            replace=True,
            source_pixel_size=source_pixel_size,
            silent=True,
            as_image=False,
            clip_func=clip_func,
        )
        if len(df_group_transformed) > 0:
            df_transformed.append(df_group_transformed)
        else:
            counter += 1
            to_remove.append(index)
    if counter > 0:
        logger.warning(f"Removed {counter:,} groups with no points - {len(df_transformed)} were kept.")
    n_to_prop = _renumber_props(n_to_prop, to_remove)
    if df_transformed:
        df_transformed = pd.concat(df_transformed)
        df_transformed = remove_invalid_points(df_transformed, group_by)
        return _convert_df_to_geojson(df_transformed, as_px, target_pixel_size, n_to_prop=n_to_prop)
    return []


def _renumber_props(n_to_prop: dict[int, dict], to_remove: list[int]) -> dict[int, dict]:
    """Renumber props."""
    if len(n_to_prop) > 1:
        for index in to_remove:
            n_to_prop.pop(index)
        n_to_prop_ = {}
        for i, (index, props) in enumerate(n_to_prop.items()):
            n_to_prop_[i] = props
        n_to_prop = n_to_prop_
    return n_to_prop


def _transform_geojson_features(
    geojson_data: list[dict],
    transform_sequence: TransformSequence,
    in_px: bool,
    as_px: bool,
    source_pixel_size: float = 1.0,
    clip: str = "ignore",
    silent: bool = False,
) -> list[dict]:
    clip_func = _get_clip_func(transform_sequence, clip, as_px=as_px)

    result = []
    for feature in geojson_data:
        # for feature in tqdm(geojson_data, desc="Transforming Features", leave=False, mininterval=1):
        geometry = feature["geometry"]

        # convert points
        if geometry["type"] == "Point":
            x, y = geometry["coordinates"]
            x, y = transform_points(
                transform_sequence,
                [x],
                [y],
                in_px=in_px,
                as_px=as_px,
                source_pixel_size=source_pixel_size,
                clip_func=clip_func,
            )
            geometry["coordinates"] = [round(x[0], 3), round(y[0], 3)]
        # convert multi-points
        elif geometry["type"] == "MultiPoint":
            xy = np.asarray(geometry["coordinates"])
            x, y = xy[:, 0], xy[:, 1]
            x, y = transform_points(
                transform_sequence,
                x,
                y,
                in_px=in_px,
                as_px=as_px,
                source_pixel_size=source_pixel_size,
                clip_func=clip_func,
            )
            geometry["coordinates"] = np.round(np.c_[x, y], 3).tolist()

        elif geometry["type"] == "Polygon":
            for i, ring in enumerate(
                tqdm(geometry["coordinates"], desc="Transforming Polygon", leave=False, mininterval=1, disable=True)
            ):
                x, y = np.array(ring).T
                x, y = transform_points(
                    transform_sequence,
                    x,
                    y,
                    in_px=in_px,
                    as_px=as_px,
                    silent=False,
                    source_pixel_size=source_pixel_size,
                    clip_func=clip_func,
                )
                if x.size == 0:
                    continue
                geometry["coordinates"][i] = np.round(np.c_[x, y], 3).tolist()
        elif geometry["type"] == "MultiPolygon":
            for j, polygon in enumerate(geometry["coordinates"]):
                for i, ring in enumerate(tqdm(polygon, desc="Transforming MultiPolygon", leave=False, mininterval=1)):
                    x, y = np.array(ring).T
                    x, y = transform_points(
                        transform_sequence,
                        x,
                        y,
                        in_px=in_px,
                        as_px=as_px,
                        silent=True,
                        source_pixel_size=source_pixel_size,
                        clip_func=clip_func,
                    )
                    if x.size == 0:
                        continue
                    geometry["coordinates"][j][i] = np.round(np.c_[x, y], 3).tolist()
        result.append(feature)
    return result


def transform_images_for_pyramid(
    wrapper: ImageWrapper,
    transformation_sequence: TransformSequence | None,
    pyramid: int = -1,
    channel_ids: list[int] | None = None,
) -> np.ndarray:
    """Transform all images."""
    import SimpleITK as sitk

    reader = wrapper.reader
    channel_axis, n_channels = reader.get_channel_axis_and_n_channels()
    channel_axis = channel_axis or 0
    if transformation_sequence is None:
        return np.asarray(reader.pyramid[pyramid])

    transformed = []
    channel_ids = channel_ids or reader.channel_ids
    for channel_index in channel_ids:
        image = np.squeeze(reader.get_channel(channel_index, pyramid, split_rgb=True))
        image = sitk.GetImageFromArray(image)
        image.SetSpacing(reader.scale_for_pyramid(pyramid))
        transformed.append(sitk.GetArrayFromImage(transformation_sequence(image)))
    return np.stack(transformed, axis=channel_axis)

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from koyo.typing import PathLike
from tqdm import tqdm

from image2image_reg.models import TransformSequence


def transform_points(
    seq: TransformSequence,
    x: np.ndarray,
    y: np.ndarray,
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
    in_px : bool, optional
        Whether input coordinates are in pixels or physical units, by default False
    as_px : bool, optional
        Whether to return coordinates in pixels or physical units, by default False
    source_pixel_size : float, optional
        Pixel size of the source image, by default 1.0
    silent : bool, optional
        Whether to show progress bar, by default False
    """
    transformed_xy = seq.transform_points(
        np.c_[x, y], is_px=in_px, px=as_px, source_pixel_size=source_pixel_size, silent=silent
    )
    return transformed_xy[:, 0], transformed_xy[:, 1]


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
        X column key, by default "x"
    y_key : str, optional
        Y column key, by default "y"
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
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Dataframe must have '{x_key}' and '{y_key}' columns.")
    if replace and suffix == "_transformed":
        suffix = "_original"

    x = df[x_key].values
    y = df[y_key].values
    x, y = transform_points(seq, x, y, in_px=in_px, as_px=as_px, source_pixel_size=source_pixel_size)

    # remove transformed columns if they exist
    if f"{x_key}{suffix}" in df.columns:
        df.drop(columns=[f"{x_key}{suffix}"], inplace=True)
    if f"{y_key}{suffix}" in df.columns:
        df.drop(columns=[f"{y_key}{suffix}"], inplace=True)
    # put data in place
    if replace:
        df.insert(max(0, df.columns.get_loc(x_key) - 1), f"{x_key}{suffix}", df[x_key])
        df.insert(max(0, df.columns.get_loc(y_key) - 1), f"{y_key}{suffix}", df[y_key])
        df[x_key] = x
        df[y_key] = y
    else:
        df.insert(df.columns.get_loc(x_key), f"{x_key}{suffix}", x)
        df.insert(df.columns.get_loc(y_key), f"{y_key}{suffix}", y)
    return df


def transform_attached_point(
    transform_sequence: TransformSequence, path: PathLike, pixel_size: float, output_path: PathLike
) -> Path:
    """Transform points data."""
    from image2image_io.readers.points_reader import read_points
    from image2image_io.readers.utilities import get_column_name

    is_in_px = pixel_size == 1.0

    # read data
    path = Path(path)
    df = read_points(path, return_df=True)
    x_key = get_column_name(df, ["x", "x_location", "x_centroid", "x:x", "vertex_x"])
    y_key = get_column_name(df, ["y", "y_location", "y_centroid", "y:y", "vertex_y"])
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError(f"Invalid columns: {df.columns}")

    df_transformed = transform_points_df(
        transform_sequence,
        df.copy(),
        in_px=is_in_px,
        as_px=is_in_px,
        x_key=x_key,
        y_key=y_key,
        replace=True,
    )
    if path.suffix in [".csv", ".txt", ".tsv"]:
        sep = {"csv": ",", "txt": "\t", "tsv": "\t"}[path.suffix[1:]]
        df_transformed.to_csv(output_path, index=False, sep=sep)
    elif path.suffix == ".parquet":
        df_transformed.to_parquet(output_path, index=False)
    return Path(output_path)


def transform_attached_shape(
    transform_sequence: TransformSequence, path: PathLike, pixel_size: float, output_path: PathLike
) -> Path:
    """Transform points data."""
    import json

    from image2image_io.readers.shapes_reader import ShapesReader

    is_in_px = pixel_size == 1.0
    reader = ShapesReader(path)
    geojson_data = deepcopy(reader.geojson_data)
    if isinstance(geojson_data, list):
        if "type" in geojson_data[0] and geojson_data[0]["type"] == "Feature":
            geojson_data = _transform_geojson_features(geojson_data, transform_sequence, in_px=is_in_px, as_px=is_in_px)
        else:
            raise ValueError("Invalid GeoJSON data.")

    with open(output_path, "w") as f:
        json.dump(geojson_data, f, indent=1)


def _transform_geojson_features(
    geojson_data: list[dict], transform_sequence: TransformSequence, in_px: bool, as_px: bool
) -> list[dict]:
    result = []
    for feature in geojson_data:
        geometry = feature["geometry"]
        if geometry["type"] == "Point":
            x, y = geometry["coordinates"]
            x, y = transform_points(transform_sequence, [x], [y], in_px=in_px, as_px=as_px)
            geometry["coordinates"] = [x[0], y[0]]
        elif geometry["type"] == "Polygon":
            for i, ring in enumerate(
                tqdm(geometry["coordinates"], desc="Transforming Polygon", leave=False, miniters=500)
            ):
                x, y = np.array(ring).T
                x, y = transform_points(transform_sequence, x, y, in_px=in_px, as_px=as_px, silent=True)
                geometry["coordinates"][i] = np.c_[x, y].tolist()
        elif geometry["type"] == "MultiPolygon":
            for j, polygon in enumerate(geometry["coordinates"]):
                for i, ring in enumerate(tqdm(polygon, desc="Transforming MultiPolygon", leave=False, miniters=500)):
                    x, y = np.array(ring).T
                    x, y = transform_points(transform_sequence, x, y, in_px=in_px, as_px=as_px, silent=True)
                    geometry["coordinates"][j][i] = np.c_[x, y].tolist()
        result.append(feature)
    return result

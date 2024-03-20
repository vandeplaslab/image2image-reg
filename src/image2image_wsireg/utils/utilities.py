"""Utilities."""

from __future__ import annotations

import typing as ty

import numpy as np
import pandas as pd

if ty.TYPE_CHECKING:
    from image2image_wsireg.models.transform_sequence import TransformSequence


def transform_points(
    seq: TransformSequence,
    x: np.ndarray,
    y: np.ndarray,
    in_px: bool = False,
    as_px: bool = False,
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
    """
    transformed_xy = seq.transform_points(np.c_[x, y], is_px=in_px, px=as_px)
    return transformed_xy[:, 0], transformed_xy[:, 1]


def transform_points_df(
    seq: TransformSequence, df: pd.DataFrame, in_px: bool = False, as_px: bool = False
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
    """
    return _transform_points_df(seq, df, "x", "y", in_px=in_px, as_px=as_px)


def transform_vertices_df(
    seq: TransformSequence, df: pd.DataFrame, in_px: bool = False, as_px: bool = False
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
    """
    return _transform_points_df(seq, df, "vertex_", "vertex_y", in_px=in_px, as_px=as_px)


def _transform_points_df(
    seq: TransformSequence,
    df: pd.DataFrame,
    x_key: str = "x",
    y_key: str = "y",
    in_px: bool = False,
    as_px: bool = False,
) -> pd.DataFrame:
    if x_key not in df.columns or y_key not in df.columns:
        raise ValueError("Dataframe must have x and y columns.")
    x = df[x_key].values
    y = df[y_key].values
    x, y = transform_points(seq, x, y, in_px=in_px, as_px=as_px)
    if f"{x_key}_transformed" in df.columns:
        df.drop(columns=[f"{x_key}_transformed"], inplace=True)
    if f"{y_key}_transformed" in df.columns:
        df.drop(columns=[f"{y_key}_transformed"], inplace=True)
    df.insert(df.columns.get_loc(x_key), f"{x_key}_transformed", x)
    df.insert(df.columns.get_loc(y_key), f"{y_key}_transformed", y)
    return df

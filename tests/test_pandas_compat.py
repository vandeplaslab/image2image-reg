"""Regression tests for pandas 2.x/3.x compatibility."""

import numpy as np
import pandas as pd

from image2image_reg.elastix.transform import remove_invalid_points
from image2image_reg.utils.transform_utils import _filter_transform_coordinate_image, _replace_column


def test_replace_column_on_slice_preserves_original_dataframe():
    df = pd.DataFrame(
        {
            "group": [1, 1, 2],
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0],
        }
    )

    sliced = df.loc[df["group"] == 1]
    replaced = _replace_column(sliced, np.array([5.0, 6.0]), np.array([15.0, 16.0]), replace=True)

    assert list(replaced.columns) == ["group", "x_transformed", "x", "y_transformed", "y"]
    assert replaced["x"].tolist() == [5.0, 6.0]
    assert replaced["y"].tolist() == [15.0, 16.0]
    assert df["x"].tolist() == [0.0, 1.0, 2.0]
    assert df["y"].tolist() == [10.0, 11.0, 12.0]


def test_filter_transform_coordinate_image_returns_reindexed_dataframe():
    df = pd.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}, index=[10, 11])

    new_x, new_y, filtered = _filter_transform_coordinate_image(
        height=100,
        width=100,
        x=np.array([0.5, 1.5]),
        y=np.array([0.5, 1.5]),
        df=df,
    )

    assert np.array_equal(new_x, np.array([0.5, 1.5]))
    assert np.array_equal(new_y, np.array([0.5, 1.5]))
    assert filtered.index.tolist() == [0, 1]


def test_remove_invalid_points_returns_owned_dataframe():
    df = pd.DataFrame(
        {
            "shape_id": ["keep", "keep", "keep", "drop", "drop"],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [5.0, 6.0, 7.0, 8.0, 9.0],
        }
    )

    filtered = remove_invalid_points(df, "shape_id")
    updated = _replace_column(filtered, np.array([10.0, 11.0, 12.0]), np.array([20.0, 21.0, 22.0]), replace=False)

    assert filtered["shape_id"].tolist() == ["keep", "keep", "keep"]
    assert updated["x_transformed"].tolist() == [10.0, 11.0, 12.0]
    assert updated["y_transformed"].tolist() == [20.0, 21.0, 22.0]

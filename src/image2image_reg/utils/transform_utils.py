"""Transform utilities."""

from __future__ import annotations

import typing as ty
from functools import partial

import numpy as np
import polars as pl
from koyo.utilities import find_nearest_index_batch
from loguru import logger
from tqdm import tqdm

if ty.TYPE_CHECKING:
    from image2image_reg.elastix.transform_sequence import TransformSequence

MULTIPLIER = 1

ClipFunc = ty.Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, ty.Optional[np.ndarray]]]


def _prepare_transform_coordinate_image(
    height: int,
    width: int,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # get coordinates of the image
    x = np.round(x * MULTIPLIER).astype(int)
    y = np.round(y * MULTIPLIER).astype(int)
    index_of_coords = np.arange(len(x)) + 1

    # crop so that only the coordinates within the slide are kept
    width = width * MULTIPLIER
    x[x < 0] = 0
    x[x >= width] = width - 1
    height = height * MULTIPLIER
    y[y < 0] = 0
    y[y >= height] = height - 1

    # let's create a image  of the same shape as the source image
    # image_of_index = np.full(shape[::-1], np.nan, dtype=np.float32)
    image_of_index = np.zeros((height, width), dtype=np.int32)
    image_of_index[y, x] = index_of_coords
    return image_of_index, index_of_coords, x, y


def _update_transform_coordinate_image(
    image_of_index: np.ndarray,
    height: int,
    width: int,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_of_index[image_of_index > 0] = 0
    # get coordinates of the image
    x = np.round(x * MULTIPLIER).astype(int)
    y = np.round(y * MULTIPLIER).astype(int)
    index_of_coords = np.arange(len(x)) + 1

    # crop so that only the coordinates within the slide are kept
    width = width * MULTIPLIER
    x[x < 0] = 0
    x[x >= width] = width - 1
    height = height * MULTIPLIER
    y[y < 0] = 0
    y[y >= height] = height - 1

    image_of_index[y, x] = index_of_coords
    return image_of_index, index_of_coords, x, y


def _filter_transform_coordinate_image(
    height: int,
    width: int,
    x: np.ndarray,
    y: np.ndarray,
    df: pl.DataFrame,
    fraction: float = 0.15,
    x_key: str = "x",
    y_key: str = "y",
) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
    frac_width = width * MULTIPLIER * fraction
    frac_height = height * MULTIPLIER * fraction

    diff_x = np.abs(df[x_key].to_numpy() - x)
    max_shift_x = np.max(diff_x)
    # if point shifts by more than 50% then it's likely very fishy!
    if max_shift_x > frac_width:
        logger.error(f"Warning: x shift is more than {fraction:.1%} of the slide width - {max_shift_x:.1f}")
        # mask = diff_x < frac_width
        # x = x[mask]
        # y = y[mask]
        # df = df.loc[mask]

    diff_y = np.abs(df[y_key].to_numpy() - y)
    max_shift_y = np.max(diff_y)
    if max_shift_y > frac_height:
        logger.error(f"Warning: y shift is more than {fraction:.1%} of the slide height - {max_shift_y:.1f}")
        # mask = diff_y < frac_height
        # x = x[mask]
        # y = y[mask]
        # df = df.loc[mask]
    return x, y, df.clone()


def _cleanup_transform_coordinate_image(
    image_of_index: np.ndarray,
    index_of_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(image_of_index)
    values = image_of_index[yy, xx]
    sort = np.argsort(values)
    values = values[sort]
    xx = xx[sort] - 1
    yy = yy[sort] - 1
    # find nearest indices
    indices = find_nearest_index_batch(values, index_of_coords)
    # find indices of elements that we failed to find
    failed_mask = np.ones(len(values), dtype=bool) if indices.size == 0 else values[indices] != index_of_coords
    new_x = xx[indices]
    new_x = new_x + np.random.uniform(-0.3, 0.3, len(indices))
    new_y = yy[indices]
    new_y = new_y + np.random.uniform(-0.3, 0.3, len(indices))
    return new_x / MULTIPLIER, new_y / MULTIPLIER, failed_mask


def _convert_geojson_to_df(
    geojson_data: list[dict],
    is_px: bool,
    source_pixel_size: float = 1.0,
) -> tuple[pl.DataFrame, dict[int, dict]]:
    """Convert GeoJSON data so that it can be transformed back to GeoJSON."""
    # types: pt = Point; pg = Polygon; mp = MultiPolygon
    data = []  # columns: x, y, unique_index, inner, outer, type
    n_to_prop = {}
    n = 1
    for feature in geojson_data:
        geometry = feature["geometry"]
        # convert points
        if geometry["type"] == "Point":
            x, y = geometry["coordinates"]
            x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
            # x, y, index, inner, outer, type
            data.append([x, y, n, 0, 0, "pt"])
            n_to_prop[n] = {
                "props": feature.get("properties", {}),
                "id": feature.get("id", None),
            }
            n += 1

        # convert multi-points
        elif geometry["type"] == "MultiPoint":
            for x, y in geometry["coordinates"]:
                x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
                # x, y, index, inner, outer, type
                data.append([x, y, n, 0, 0, "mpt"])
            n_to_prop[n] = {
                "props": feature.get("properties", {}),
                "id": feature.get("id", None),
            }
            n += 1

        # convert polygons
        elif geometry["type"] == "Polygon":
            for outer, ring in enumerate(geometry["coordinates"]):
                for x, y in ring:
                    x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
                    # x, y, index, inner, outer, type
                    data.append([x, y, n, outer, 0, "pg"])
                n_to_prop[n] = {
                    "props": feature.get("properties", {}),
                    "id": feature.get("id", None),
                }
                n += 1

        # convert multi-polygons
        elif geometry["type"] == "MultiPolygon":
            for outer, polygon in enumerate(geometry["coordinates"]):
                for inner, ring in enumerate(polygon):
                    for x, y in ring:
                        x, y = _transform_original_from_um_to_px(x, y, is_px, source_pixel_size)
                        # x, y, index, inner, outer, type
                        data.append([x, y, n, outer, inner, "mp"])
                    n_to_prop[n] = {
                        "props": feature.get("properties", {}),
                        "id": feature.get("id", None),
                    }
    if data:
        df = pl.DataFrame(data, schema=["x", "y", "unique_index", "outer", "inner", "type"], orient="row")
    else:
        df = pl.DataFrame(
            schema={
                "x": pl.Float64,
                "y": pl.Float64,
                "unique_index": pl.Int64,
                "outer": pl.Int64,
                "inner": pl.Int64,
                "type": pl.String,
            }
        )
    return df, n_to_prop


def _convert_df_to_geojson(
    df: pl.DataFrame,
    as_px: bool,
    target_pixel_size: float,
    n_to_prop: dict[int, dict] | None = None,
) -> list[dict]:
    """Convert DataFrame back to GeoJSON."""
    from shapely.geometry import Polygon, mapping
    from shapely.validation import make_valid

    if n_to_prop is None:
        n_to_prop = {}

    def _normalize_group_name(group: ty.Any) -> ty.Any:
        if isinstance(group, tuple) and len(group) == 1:
            return group[0]
        return group

    failed = False
    geojson_data, geojson_data_fixed = [], []
    for unique_index, dff in tqdm(
        df.group_by("unique_index", maintain_order=True),
        desc="Converting to GeoJSON",
        leave=False,
    ):
        unique_index = _normalize_group_name(unique_index)
        props = n_to_prop.get(unique_index, {})
        prop = props.get("props", {})
        id_ = props.get("id", None)
        kws = {"id": id_} if id_ is not None else {}
        geometry = {}
        geometry_type = dff.item(0, "type")
        # point
        if geometry_type == "pt":
            x_, y_ = _transform_transformed_from_px_to_um(
                dff["x"].to_numpy(),
                dff["y"].to_numpy(),
                as_px,
                target_pixel_size,
            )
            geometry["type"] = "Point"
            geometry["coordinates"] = [round(x_[0], 2), round(y_[0], 2)]

        # multi-point
        elif geometry_type == "mpt":
            geometry["type"] = "MultiPoint"
            geometry["coordinates"] = []
            x_, y_ = _transform_transformed_from_px_to_um(
                dff["x"].to_numpy(),
                dff["y"].to_numpy(),
                as_px,
                target_pixel_size,
            )
            for x, y in zip(x_, y_):
                geometry["coordinates"].append([round(x, 2), round(y, 2)])

        # polygon
        elif geometry_type == "pg":
            geometry["type"] = "Polygon"
            geometry["coordinates"] = []

            for row in dff.iter_rows(named=True):
                x_, y_ = _transform_transformed_from_px_to_um(row["x"], row["y"], as_px, target_pixel_size)
                # prepare the geometry
                if row["outer"] == 0 and len(geometry["coordinates"]) == 0:
                    geometry["coordinates"].append([])
                geometry["coordinates"][row["outer"]].append([round(x_, 2), round(y_, 2)])

        # multi-polygon
        elif geometry_type == "mp":
            geometry["type"] = "MultiPolygon"
            geometry["coordinates"] = []
            while len(geometry["coordinates"]) < int(dff["outer"].max()) + 1:
                geometry["coordinates"].append([])
            for row in dff.iter_rows(named=True):
                x_, y_ = _transform_transformed_from_px_to_um(row["x"], row["y"], as_px, target_pixel_size)
                while len(geometry["coordinates"][row["outer"]]) <= row["inner"]:
                    geometry["coordinates"][row["outer"]].append([])
                geometry["coordinates"][row["outer"]][row["inner"]].append([round(x_, 2), round(y_, 2)])

        # skip broken geometries
        if not geometry or not geometry.get("coordinates"):
            continue
        if not failed:
            try:
                for shape in geometry["coordinates"]:
                    polygon = make_valid(Polygon(shape))
                    geometry = mapping(polygon)
                    geojson_data_fixed.append({"type": "Feature", **kws, "geometry": geometry, "properties": prop})
            except Exception:
                failed = True
                logger.warning("Failed to automatically fix shape to GeoJSON.")
        geojson_data.append({"type": "Feature", **kws, "geometry": geometry, "properties": prop})
    return geojson_data if failed else geojson_data_fixed


def _clip_noop(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    return x, y, None


def _clip_points_outside_of_image(
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Clip points outside of image dimensions, by setting them to the nearest edge.

    Coordinates should be in pixel coordinates.
    """
    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)
    return x, y, None


def _remove_points_outside_of_image(
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove points outside of image dimensions.

    Coordinates should be in pixel coordinates.
    """
    mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[mask], y[mask], mask


def _remove_all_points_if_outside_of_image(
    x: np.ndarray,
    y: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Remove points outside of image dimensions.

    Coordinates should be in pixel coordinates.
    """
    mask = (x >= 0) & (x <= width) & (y >= 0) & (y <= height)
    if not mask.all():
        return np.array([]), np.array([]), np.zeros_like(x, dtype=bool)
    return x[mask], y[mask], mask


def _get_clip_func(transform_sequence: TransformSequence, clip: str, as_px: bool = True) -> ClipFunc:
    width, height = transform_sequence.output_size
    if not as_px:
        height = height * transform_sequence.resolution
        width = width * transform_sequence.resolution
    if clip == "ignore":
        return _clip_noop
    if clip == "clip":
        return partial(_clip_points_outside_of_image, height=height, width=width)
    if clip == "remove":
        return partial(_remove_all_points_if_outside_of_image, height=height, width=width)
    if clip == "part-remove":
        return partial(_remove_points_outside_of_image, height=height, width=width)
    raise ValueError(f"Invalid clip value: {clip}")


def _transform_original_from_um_to_px(
    x: np.ndarray,
    y: np.ndarray,
    is_px: bool,
    source_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    inv_source_pixel_size = 1 / source_pixel_size
    if is_px:  # no need to transform since it's already in pixel coordinates
        return x, y
    # convert from um to pixel by multiplying by the inverse of the pixel size
    return x * inv_source_pixel_size, y * inv_source_pixel_size


def _transform_transformed_from_px_to_um(
    x: np.ndarray,
    y: np.ndarray,
    as_px: bool,
    target_pixel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if as_px:  # no need to transform since it's already in pixel coordinates
        return x, y
    # convert from px to um by multiplying by the pixel size
    return x * target_pixel_size, y * target_pixel_size


def _replace_column(
    df: pl.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    x_key: str = "x",
    y_key: str = "y",
    suffix: str = "_transformed",
    replace: bool = False,
) -> pl.DataFrame:
    df = df.clone()
    # remove transformed columns if they exist
    if f"{x_key}{suffix}" in df.columns:
        df = df.drop(f"{x_key}{suffix}")
    if f"{y_key}{suffix}" in df.columns:
        df = df.drop(f"{y_key}{suffix}")
    # put data in place
    if replace:
        df.insert_column(max(0, df.columns.index(x_key)), pl.Series(f"{x_key}{suffix}", df[x_key].to_numpy().copy()))
        df.insert_column(max(0, df.columns.index(y_key)), pl.Series(f"{y_key}{suffix}", df[y_key].to_numpy().copy()))
        df = df.with_columns(pl.Series(x_key, x), pl.Series(y_key, y))
    else:
        df.insert_column(df.columns.index(x_key), pl.Series(f"{x_key}{suffix}", x))
        df.insert_column(df.columns.index(y_key), pl.Series(f"{y_key}{suffix}", y))
    return df

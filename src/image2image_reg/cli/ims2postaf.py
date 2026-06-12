"""IMS to postAF registration commands."""

from __future__ import annotations

import json
import typing as ty
from pathlib import Path

import click
import cv2
import numpy as np
from click_groups import GroupedGroup
from koyo.click import Parameter, print_parameters
from loguru import logger

SHAPE_VALUE_COUNT_MESSAGE = "Expected two values in rows,columns order."
SHAPE_INTEGER_MESSAGE = "Shape values must be integers."
SHAPE_MINIMUM_MESSAGE = "Shape values must be greater than one."
PIXEL_SIZE_VALUE_COUNT_MESSAGE = "Expected one scalar value or two yx values."
PIXEL_SIZE_NUMERIC_MESSAGE = "Pixel size values must be numeric."
PIXEL_SIZE_POSITIVE_MESSAGE = "Pixel size values must be greater than zero."


def parse_shape(ctx: click.Context, param: click.Parameter, value: str | None) -> tuple[int, int] | None:
    """Parse a two-value image shape."""
    del ctx, param
    if value is None:
        return None
    value = value.lower().replace("x", ",")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise click.BadParameter(SHAPE_VALUE_COUNT_MESSAGE)
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise click.BadParameter(SHAPE_INTEGER_MESSAGE) from exc
    if rows < 2 or cols < 2:
        raise click.BadParameter(SHAPE_MINIMUM_MESSAGE)
    return rows, cols


def parse_pixel_size(
    ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> float | tuple[float, float] | None:
    """Parse scalar or yx pixel size values."""
    del ctx, param
    if value is None:
        return None
    value = value.lower().replace("x", ",")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) not in {1, 2}:
        raise click.BadParameter(PIXEL_SIZE_VALUE_COUNT_MESSAGE)
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise click.BadParameter(PIXEL_SIZE_NUMERIC_MESSAGE) from exc
    if any(part <= 0 for part in values):
        raise click.BadParameter(PIXEL_SIZE_POSITIVE_MESSAGE)
    return values[0] if len(values) == 1 else ty.cast(tuple[float, float], values)


@click.group("ims2postaf", cls=GroupedGroup)
def ims2postaf() -> None:
    """IMS to postAF registration."""


@click.option(
    "--output_diagnostics",
    help="Optional path where detailed diagnostics should be written as JSON.",
    type=click.Path(exists=False, resolve_path=True, file_okay=True, dir_okay=False),
    default=None,
    show_default=True,
)
@click.option(
    "--output_preview",
    help=(
        "Optional preview output path or prefix. Writes _preview_postaf.png, _preview_ims.png, "
        "and _preview_overlay.png."
    ),
    type=click.Path(exists=False, resolve_path=True, file_okay=True, dir_okay=False),
    default=None,
    show_default=True,
)
@click.option(
    "--output_affine",
    help="Path where the yx-micron affine JSON should be written.",
    type=click.Path(exists=False, resolve_path=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--min_confidence",
    help="Minimum footprint detection confidence required for success.",
    type=click.FloatRange(0, 1, clamp=True),
    default=0.05,
    show_default=True,
)
@click.option(
    "--ims_image",
    help="IMS intensity image used to estimate the MALDI acquisition footprint.",
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--ims_pixel_size",
    help="IMS pixel size in micrometers. Provide scalar or y,x. Inferred from the reader when omitted.",
    type=click.STRING,
    callback=parse_pixel_size,
    default=None,
    required=False,
    show_default=True,
)
@click.option(
    "--ims_shape",
    help="Optional expected IMS image shape in rows,columns order.",
    type=click.STRING,
    callback=parse_shape,
    default=None,
    required=False,
    show_default=True,
)
@click.option(
    "--postaf_pixel_size",
    help="PostAF pixel size in micrometers. Provide scalar or y,x. Inferred from the reader when omitted.",
    type=click.STRING,
    callback=parse_pixel_size,
    default=None,
    required=False,
    show_default=True,
)
@click.option(
    "--postaf_image",
    help="Path to the post IMS autofluorescence image.",
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    required=True,
)
@ims2postaf.command("estimate_affine")
def estimate_affine_cmd(
    postaf_image: str,
    ims_image: str,
    postaf_pixel_size: float | tuple[float, float] | None,
    ims_shape: tuple[int, int] | None,
    ims_pixel_size: float | tuple[float, float] | None,
    min_confidence: float,
    output_affine: str,
    output_preview: str | None,
    output_diagnostics: str | None,
) -> None:
    """Estimate an affine from MALDI IMS pixels to postAF coordinates."""
    estimate_affine_runner(
        postaf_image=postaf_image,
        postaf_pixel_size=postaf_pixel_size,
        ims_shape=ims_shape,
        ims_pixel_size=ims_pixel_size,
        ims_image=ims_image,
        min_confidence=min_confidence,
        output_affine=output_affine,
        output_preview=output_preview,
        output_diagnostics=output_diagnostics,
    )


def estimate_affine_runner(
    postaf_image: str,
    ims_image: str,
    output_affine: str,
    postaf_pixel_size: float | tuple[float, float] | None = None,
    ims_shape: tuple[int, int] | None = None,
    ims_pixel_size: float | tuple[float, float] | None = None,
    min_confidence: float = 0.05,
    output_preview: str | None = None,
    output_diagnostics: str | None = None,
) -> None:
    """Estimate and write an IMS to postAF affine matrix."""
    from image2image_reg.workflows.ims2postaf import create_ims_postaf_preview, estimate_ims_to_postaf_affine

    print_parameters(
        Parameter("PostAF image", "--postaf_image", postaf_image),
        Parameter("PostAF pixel size", "--postaf_pixel_size", postaf_pixel_size),
        Parameter("IMS shape", "--ims_shape", ims_shape),
        Parameter("IMS pixel size", "--ims_pixel_size", ims_pixel_size),
        Parameter("IMS image", "--ims_image", ims_image),
        Parameter("Minimum confidence", "--min_confidence", min_confidence),
        Parameter("Output affine", "--output_affine", output_affine),
        Parameter("Output preview", "--output_preview", output_preview),
        Parameter("Output diagnostics", "--output_diagnostics", output_diagnostics),
    )
    result = estimate_ims_to_postaf_affine(
        postaf_image=postaf_image,
        ims_image=ims_image,
        postaf_pixel_size=postaf_pixel_size,
        ims_shape=ims_shape,
        ims_pixel_size=ims_pixel_size,
        min_confidence=min_confidence,
    )

    output_affine_path = Path(output_affine)
    output_affine_path.parent.mkdir(exist_ok=True, parents=True)
    with output_affine_path.open("w", encoding="utf-8") as handle:
        json.dump(result.matrix_yx_um.tolist(), handle, indent=2)
    logger.info(f"Wrote IMS to postAF affine matrix to {output_affine_path}")

    preview_paths = _preview_paths(output_affine_path, output_preview)
    for path in preview_paths.values():
        path.parent.mkdir(exist_ok=True, parents=True)
    previews = create_ims_postaf_preview(
        postaf_image=postaf_image,
        ims_image=ims_image,
        matrix_yx_px=result.matrix_yx_px,
        postaf_pixel_size=postaf_pixel_size,
        ims_pixel_size=ims_pixel_size,
    )
    _write_rgb_preview(preview_paths["postaf"], previews.postaf)
    _write_rgb_preview(preview_paths["ims"], previews.ims)
    _write_rgb_preview(preview_paths["overlay"], previews.overlay)
    logger.info(f"Wrote IMS to postAF preview images to {preview_paths['overlay'].parent}")

    output_diagnostics_path = (
        Path(output_diagnostics)
        if output_diagnostics
        else output_affine_path.with_name(f"{output_affine_path.stem}_diagnostics.json")
    )
    output_diagnostics_path.parent.mkdir(exist_ok=True, parents=True)
    with output_diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    logger.info(f"Wrote IMS to postAF diagnostics to {output_diagnostics_path}")


def _preview_paths(output_affine_path: Path, output_preview: str | None) -> dict[str, Path]:
    if output_preview:
        base_path = Path(output_preview)
        if base_path.suffix:
            base_path = base_path.with_suffix("")
    else:
        base_path = output_affine_path.with_suffix("")
    return {
        "postaf": base_path.with_name(f"{base_path.name}_preview_postaf.png"),
        "ims": base_path.with_name(f"{base_path.name}_preview_ims.png"),
        "overlay": base_path.with_name(f"{base_path.name}_preview_overlay.png"),
    }


def _write_rgb_preview(path: Path, image: np.ndarray) -> None:
    """Write an RGB preview image."""
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

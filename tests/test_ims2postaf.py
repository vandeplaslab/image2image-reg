"""Test IMS to postAF affine estimation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from skimage.io import imsave

from image2image_reg.cli.ims2postaf import ims2postaf, parse_pixel_size, parse_shape
from image2image_reg.models import Preprocessing
from image2image_reg.workflows.ims2postaf import (
    estimate_ims_to_postaf_affine,
)


def _synthetic_postaf(
    ims_shape: tuple[int, int] = (30, 42),
    ims_pixel_size: float = 10.0,
    postaf_shape: tuple[int, int] = (520, 680),
    angle_degrees: float = 8.0,
    translation_yx: tuple[float, float] = (130.0, 165.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10)
    ims = np.zeros(ims_shape, dtype=np.uint8)
    yy, xx = np.indices(ims_shape)
    mask = (((yy - 14) / 12) ** 2 + ((xx - 21) / 18) ** 2) <= 1
    mask &= ~((yy < 10) & (xx > 24))
    mask |= ((yy > 18) & (yy < 25) & (xx > 5) & (xx < 15))
    ims[mask] = 180

    postaf = rng.normal(28, 3, size=postaf_shape).astype(np.float32)
    angle = math.radians(angle_degrees)
    scale = ims_pixel_size
    matrix_yx_px = np.asarray(
        [
            [math.cos(angle) * scale, -math.sin(angle) * scale, translation_yx[0]],
            [math.sin(angle) * scale, math.cos(angle) * scale, translation_yx[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    for row, col in np.column_stack(np.nonzero(ims)):
        y_coord, x_coord, _one = matrix_yx_px @ np.asarray([row, col, 1.0])
        cv2.circle(postaf, (round(x_coord), round(y_coord)), 2, 160, -1)

    postaf = cv2.GaussianBlur(postaf, (3, 3), sigmaX=0)
    return ims, np.clip(postaf, 0, 255).astype(np.uint8), matrix_yx_px


def _draw_ablation_from_matrix(
    postaf: np.ndarray,
    ims: np.ndarray,
    matrix_yx_px: np.ndarray,
    intensity: int,
    radius: int = 2,
) -> None:
    for row, col in np.column_stack(np.nonzero(ims)):
        y_coord, x_coord, _one = matrix_yx_px @ np.asarray([row, col, 1.0])
        cv2.circle(postaf, (round(x_coord), round(y_coord)), radius, intensity, -1)


def test_estimate_ims_to_postaf_affine_recovers_synthetic_grid() -> None:
    """Test affine estimation on a synthetic ablation grid."""
    ims_shape = (24, 36)
    ims, postaf, expected_matrix_px = _synthetic_postaf(ims_shape=ims_shape)

    result = estimate_ims_to_postaf_affine(
        postaf,
        ims,
        postaf_pixel_size=1.0,
        ims_pixel_size=10.0,
    )

    source = np.column_stack([np.nonzero(ims)[0], np.nonzero(ims)[1], np.ones(np.count_nonzero(ims))]).T
    expected_corners = (expected_matrix_px @ source).T[:, :2]
    detected_corners = (result.matrix_yx_px @ source).T[:, :2]

    assert result.confidence > 0.35
    assert np.sqrt(np.mean((expected_corners - detected_corners) ** 2)) < 6.0
    assert result.matrix_yx_um.shape == (3, 3)
    assert result.matrix_yx_px.shape == (3, 3)


def test_estimate_ims_to_postaf_affine_prefers_ims_axis_aligned_candidate() -> None:
    """Test wrong-orientation candidates do not win over the IMS acquisition axis."""
    ims, _postaf, expected_matrix_px = _synthetic_postaf(angle_degrees=0.0, translation_yx=(140.0, 120.0))
    rng = np.random.default_rng(12)
    postaf = rng.normal(28, 3, size=(520, 680)).astype(np.float32)
    _draw_ablation_from_matrix(postaf, ims, expected_matrix_px, intensity=140)

    wrong_angle = math.radians(90.0)
    wrong_matrix_px = np.asarray(
        [
            [math.cos(wrong_angle) * 10.0, -math.sin(wrong_angle) * 10.0, 95.0],
            [math.sin(wrong_angle) * 10.0, math.cos(wrong_angle) * 10.0, 520.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _draw_ablation_from_matrix(postaf, ims, wrong_matrix_px, intensity=220)
    postaf = cv2.GaussianBlur(postaf, (3, 3), sigmaX=0)

    result = estimate_ims_to_postaf_affine(
        np.clip(postaf, 0, 255).astype(np.uint8),
        ims,
        postaf_pixel_size=1.0,
        ims_pixel_size=10.0,
    )

    source = np.column_stack([np.nonzero(ims)[0], np.nonzero(ims)[1], np.ones(np.count_nonzero(ims))]).T
    expected_corners = (expected_matrix_px @ source).T[:, :2]
    detected_corners = (result.matrix_yx_px @ source).T[:, :2]

    assert np.sqrt(np.mean((expected_corners - detected_corners) ** 2)) < 12.0
    assert result.diagnostics["orientation_score"] > 0.9


def test_estimate_ims_to_postaf_affine_constrains_target_size() -> None:
    """Test oversized postAF responses do not scale the IMS footprint."""
    ims, _postaf, expected_matrix_px = _synthetic_postaf(angle_degrees=0.0, translation_yx=(140.0, 120.0))
    rng = np.random.default_rng(13)
    postaf = rng.normal(28, 3, size=(520, 680)).astype(np.float32)
    _draw_ablation_from_matrix(postaf, ims, expected_matrix_px, intensity=180)
    y_coords, x_coords = np.nonzero(postaf > 100)
    postaf[np.clip(y_coords + 28, 0, postaf.shape[0] - 1), x_coords] = 170
    postaf[np.clip(y_coords - 28, 0, postaf.shape[0] - 1), x_coords] = 170
    postaf = cv2.GaussianBlur(postaf, (3, 3), sigmaX=0)

    result = estimate_ims_to_postaf_affine(
        np.clip(postaf, 0, 255).astype(np.uint8),
        ims,
        postaf_pixel_size=1.0,
        ims_pixel_size=10.0,
    )

    assert result.diagnostics["detected_height_um"] > result.diagnostics["expected_height_um"]
    assert result.diagnostics["corrected_height_um"] == pytest.approx(result.diagnostics["expected_height_um"])


def test_estimate_ims_to_postaf_affine_handles_cyx_ims_file(tmp_path: Path) -> None:
    """Test CYX IMS files are projected over the channel axis."""
    ims, postaf, expected_matrix_px = _synthetic_postaf()
    ims_cyx = np.zeros((24, *ims.shape), dtype=np.uint8)
    ims_cyx[17] = ims
    ims_path = tmp_path / "ims_cyx.tiff"
    tifffile.imwrite(ims_path, ims_cyx)

    result = estimate_ims_to_postaf_affine(
        postaf,
        ims_path,
        postaf_pixel_size=1.0,
        ims_pixel_size=10.0,
    )

    source = np.column_stack([np.nonzero(ims)[0], np.nonzero(ims)[1], np.ones(np.count_nonzero(ims))]).T
    expected_corners = (expected_matrix_px @ source).T[:, :2]
    detected_corners = (result.matrix_yx_px @ source).T[:, :2]

    assert result.confidence > 0.35
    assert np.sqrt(np.mean((expected_corners - detected_corners) ** 2)) < 12.0


def test_estimate_ims_to_postaf_affine_rejects_empty_postaf() -> None:
    """Test empty postAF images fail with a controlled error."""
    with pytest.raises(ValueError, match=r"empty|Could not detect"):
        estimate_ims_to_postaf_affine(
            np.zeros((100, 100), dtype=np.uint8),
            np.ones((10, 10), dtype=np.uint8),
            postaf_pixel_size=1.0,
            ims_pixel_size=10.0,
        )


def test_ims2postaf_cli_writes_affine_json(tmp_path: Path) -> None:
    """Test CLI output can be used as a preprocessing affine."""
    ims, postaf, _expected_matrix_px = _synthetic_postaf()
    ims_path = tmp_path / "ims.tiff"
    postaf_path = tmp_path / "postaf.tiff"
    output_affine = tmp_path / "ims_to_postaf.json"
    output_preview_postaf = tmp_path / "ims_to_postaf_preview_postaf.png"
    output_preview_ims = tmp_path / "ims_to_postaf_preview_ims.png"
    output_preview_overlay = tmp_path / "ims_to_postaf_preview_overlay.png"
    output_diagnostics = tmp_path / "ims_to_postaf_diagnostics.json"
    imsave(ims_path, ims)
    imsave(postaf_path, postaf)

    runner = CliRunner()
    result = runner.invoke(
        ims2postaf,
        [
            "estimate_affine",
            "--postaf_image",
            str(postaf_path),
            "--ims_image",
            str(ims_path),
            "--output_affine",
            str(output_affine),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_affine.exists()
    assert output_preview_postaf.exists()
    assert output_preview_ims.exists()
    assert output_preview_overlay.exists()
    assert output_diagnostics.exists()
    for preview_path in [output_preview_postaf, output_preview_ims, output_preview_overlay]:
        preview = cv2.imread(str(preview_path), cv2.IMREAD_COLOR)
        assert preview is not None
        assert preview.shape[:2] == postaf.shape
        assert np.max(preview) > 0
    matrix = json.loads(output_affine.read_text())
    prepro = Preprocessing(affine=matrix)
    assert prepro.affine.shape == (3, 3)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10", 10.0),
        ("10,20", (10.0, 20.0)),
        ("10x20", (10.0, 20.0)),
    ],
)
def test_parse_pixel_size(value: str, expected: float | tuple[float, float]) -> None:
    """Test IMS to postAF pixel size parser."""
    assert parse_pixel_size(None, None, value) == expected


@pytest.mark.parametrize(("value", "expected"), [("24,36", (24, 36)), ("24x36", (24, 36))])
def test_parse_shape(value: str, expected: tuple[int, int]) -> None:
    """Test IMS shape parser."""
    assert parse_shape(None, None, value) == expected

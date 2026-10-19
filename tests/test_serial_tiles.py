"""Compare bounded source reads with full-image physical resampling."""

import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.elastix.tiles import resample_region


class Reader:
    """Record the size of each source read independently of output tiling."""

    def __init__(self) -> None:
        yy, xx = np.mgrid[:91, :107]
        self.data = np.stack([yy + xx, yy * 2 + xx], axis=0).astype(np.float32)
        self.dtype = self.data.dtype
        self.image_shape = self.data.shape[1:]
        self.resolution = 0.7
        self.reads: list[int] = []

    def get_channel(self, channel: int, split_rgb: bool = True) -> object:
        """Return an array-like channel recording every requested source region."""
        reader = self

        class Channel:
            def __getitem__(self, region: tuple[slice, slice]) -> np.ndarray:
                result = reader.data[channel][region]
                reader.reads.append(result.nbytes * 2)
                return result

        return Channel()


@pytest.mark.parametrize("kind", ["identity", "translation", "rotation", "affine", "deformable"])
def test_resample_matches_whole_image(kind: str) -> None:
    reader = Reader()
    transform = sitk.AffineTransform(2)
    if kind == "translation":
        transform.SetTranslation((3.2, -1.1))
    elif kind == "rotation":
        transform.Rotate(0, 1, 0.18)
        transform.SetCenter((30, 30))
    elif kind == "affine":
        transform.SetMatrix([1.15, 0.18, -0.1, 0.85])
    elif kind == "deformable":
        image = sitk.Image([110, 100], sitk.sitkFloat64)
        transform = sitk.BSplineTransformInitializer(image, [4, 4])
        params = np.asarray(transform.GetParameters())
        params[:] = np.sin(np.arange(len(params)) / 4) * 3
        transform.SetParameters(params)
    origin, spacing, size = (-2.3, 1.4), (0.9, 0.9), (83, 71)
    for channel in range(2):
        image = sitk.GetImageFromArray(reader.data[channel])
        image.SetSpacing((0.7, 0.7))
        expected = sitk.GetArrayFromImage(sitk.Resample(image, size, transform, sitk.sitkLinear, origin, spacing))
        actual = np.zeros(expected.shape, dtype=reader.dtype)
        for y in range(0, size[1], 19):
            for x in range(0, size[0], 23):
                h, w = min(19, size[1] - y), min(23, size[0] - x)
                actual[y : y + h, x : x + w] = resample_region(
                    reader,
                    transform,
                    channel,
                    origin=origin,
                    spacing=spacing,
                    region=(y, x, h, w),
                    max_source_bytes=2048,
                )
        np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert max(reader.reads) <= 2048


def test_outside_source_does_not_read() -> None:
    reader = Reader()
    result = resample_region(
        reader, sitk.Transform(2, sitk.sitkIdentity), 0, origin=(1000, 1000), spacing=(1, 1), region=(0, 0, 10, 10)
    )
    assert not reader.reads
    assert not result.any()

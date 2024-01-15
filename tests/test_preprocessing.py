"""Pre-processing tests."""
from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from image2image_wsireg.preprocessing.convert import numpy_to_sitk_image
from image2image_wsireg.preprocessing.step import (
    BackgroundColorDistancePreprocessor,
    ContrastEnhancePreprocessor,
    DownsamplePreprocessor,
    GrayPreprocessor,
    HandEDeconvolutionPreprocessor,
    InvertIntensityPreprocessor,
    LuminosityPreprocessor,
    MaximumIntensityProcessor,
    Preprocessor,
    StainFlattenerPreprocessor,
)


def twod_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


def rgb_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


def multichannel_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(4, 100, 100), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


@pytest.mark.parametrize("array", [twod_array(), twod_array(True)])
def test_pre_twod(array):
    pre = Preprocessor(array, pixel_size=1)
    assert pre.is_single_channel, "Single channel image not detected"
    assert not pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply()


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_pre_rgb(array):
    pre = Preprocessor(array, pixel_size=1)
    assert pre.is_rgb, "RGB image not detected"
    assert not pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply()


@pytest.mark.parametrize("array", [multichannel_array(), multichannel_array(True)])
def test_pre_multichannel(array):
    pre = Preprocessor(array, pixel_size=1)
    assert not pre.is_rgb, "RGB image not detected"
    assert pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply()


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_luminosity(array):
    pre = LuminosityPreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_gray(array):
    pre = GrayPreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_background(array):
    pre = BackgroundColorDistancePreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [twod_array(), twod_array(True)])
def test_invert(array):
    pre = InvertIntensityPreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), multichannel_array()])
def test_maximum_projection(array):
    pre = MaximumIntensityProcessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_hande(array):
    pre = HandEDeconvolutionPreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [twod_array(), twod_array(True)])
def test_contrast_enhance(array):
    pre = ContrastEnhancePreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_stain_flattener(array):
    pre = StainFlattenerPreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [twod_array(), twod_array(True)])
def test_downsample_2d(array):
    pre = DownsamplePreprocessor(array, pixel_size=1)
    image = pre.apply()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(to_array=True, factor=2)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (50, 50), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"

"""Pre-processing tests."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.preprocessing.convert import numpy_to_sitk_image
from image2image_reg.preprocessing.step import (
    PREPROCESSOR_REGISTER,
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
    get_preprocessor,
)
from image2image_reg.preprocessing.workflow import Workflow


def single_channel_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(100, 100), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


def rgb_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


def multi_channel_array(as_sitk: bool = False) -> np.ndarray | sitk.Image:
    """Get NumPy array."""
    array = np.random.randint(0, 255, size=(4, 100, 100), dtype=np.uint8)
    return numpy_to_sitk_image(array) if as_sitk else array


def test_pre_attr_set():
    for pre_cls in PREPROCESSOR_REGISTER.values():
        assert pre_cls.allow_multi_channel is not None, "allow_multi_channel not set"
        assert pre_cls.allow_rgb is not None, "allow_rgb not set"
        assert pre_cls.allow_single_channel is not None, "allow_twod not set"


def test_get_preprocessor():
    for key in PREPROCESSOR_REGISTER:
        pre = get_preprocessor(key)
        assert pre is not None, f"Preprocessor not found for '{key}'"


@pytest.mark.parametrize("array", [single_channel_array(), single_channel_array(True)])
def test_pre_twod(array):
    pre = Preprocessor()
    pre.set_array(array, 1)
    assert pre.is_single_channel, "Single channel image not detected"
    assert not pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply(array, 1)


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_pre_rgb(array):
    pre = Preprocessor()
    pre.set_array(array, 1)
    assert pre.is_rgb, "RGB image not detected"
    assert not pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply(array, 1)


@pytest.mark.parametrize("array", [multi_channel_array(), multi_channel_array(True)])
def test_pre_multichannel(array):
    pre = Preprocessor()
    pre.set_array(array, 1)
    assert not pre.is_rgb, "RGB image not detected"
    assert pre.is_multi_channel, "Multichannel image detected as RGB"
    numpy_array = pre.to_array()
    assert isinstance(numpy_array, np.ndarray), "Conversion to NumPy failed"
    sitk_image = pre.to_sitk()
    assert isinstance(sitk_image, sitk.Image), "Conversion to SimpleITK failed"
    with pytest.raises(NotImplementedError):
        pre.apply(array, 1)


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_luminosity(array):
    pre = LuminosityPreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_gray(array):
    pre = GrayPreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_background(array):
    pre = BackgroundColorDistancePreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [single_channel_array(), single_channel_array(True)])
def test_invert(array):
    pre = InvertIntensityPreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), multi_channel_array()])
def test_maximum_projection(array):
    pre = MaximumIntensityProcessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_hande(array):
    pre = HandEDeconvolutionPreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [single_channel_array(), single_channel_array(True)])
def test_contrast_enhance(array):
    pre = ContrastEnhancePreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_stain_flattener(array):
    pre = StainFlattenerPreprocessor()
    image = pre.apply(array, 1)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (100, 100), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [single_channel_array(), single_channel_array(True)])
def test_downsample_2d(array):
    pre = DownsamplePreprocessor()
    image = pre.apply(array, 1, factor=2)
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = pre.apply(array, 1, to_array=True, factor=2)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.shape == (50, 50), "Shape mismatch"
    assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [rgb_array(), rgb_array(True)])
def test_all_rgb(array):
    for pre in PREPROCESSOR_REGISTER.values():
        if pre.allow_rgb:
            image = pre.apply(array, 1)
            assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
            array_ = pre.apply(array, 1, to_array=True)
            assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
            assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [single_channel_array(), single_channel_array(True)])
def test_all_single_channel(array):
    for pre in PREPROCESSOR_REGISTER.values():
        if pre.allow_single_channel:
            image = pre.apply(array, 1)
            assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
            array_ = pre.apply(array, 1, to_array=True)
            assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
            assert array_.dtype == np.uint8, "Incorrect dtype"


@pytest.mark.parametrize("array", [multi_channel_array(), single_channel_array(True)])
def test_all_multi_channel(array):
    for pre in PREPROCESSOR_REGISTER.values():
        if pre.allow_multi_channel:
            image = pre.apply(array, 1)
            assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
            array_ = pre.apply(array, 1, to_array=True)
            assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
            assert array_.dtype == np.uint8, "Incorrect dtype"


def test_workflow():
    wf = Workflow(single_channel_array(), 1, ("contrast_enhance",))
    image = wf.run()
    assert isinstance(image, sitk.Image), "Conversion to SimpleITK failed"
    array_ = wf.run(to_array=True)
    assert isinstance(array_, np.ndarray), "Conversion to NumPy failed"
    assert array_.dtype == np.uint8, "Incorrect dtype"

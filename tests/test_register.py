"""Test registration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.elastix.registration import _elx_lineparser
from image2image_reg.elastix.registration_utils import register_2d_images
from image2image_reg.models import Preprocessing
from image2image_reg.utils._test import get_test_file
from image2image_reg.workflows import ElastixReg
from image2image_reg.wrapper import ImageWrapper


def _make_ellipse_project(
    tmp_path: Path,
    with_mask: bool = False,
    with_initial: bool = False,
    with_bbox: bool = False,
    with_through: bool = False,
) -> ElastixReg:
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    mask = get_test_file("ellipse_mask.tiff") if with_mask else None

    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    obj.set_logger()
    pre = Preprocessing.basic()
    if with_initial:
        pre.affine = np.asarray([[1, 0, 550], [0, 1, 500], [0, 0, 1]])
    mask_bbox = None
    if with_bbox:
        mask_bbox = (500, 500, 2000, 2000)
    obj.add_modality("source", source, preprocessing=pre)
    obj.add_modality("target", target, mask=mask, preprocessing=Preprocessing.basic(), mask_bbox=mask_bbox)
    if with_through:
        obj.add_modality("through", source, preprocessing=Preprocessing.basic())
    if with_through:
        obj.add_registration_path("through", "target", transform=["rigid"])
        obj.add_registration_path("source", "target", through="through", transform=["rigid"])
    else:
        obj.add_registration_path("source", "target", transform=["rigid"])
    return obj


def test_preprocess(tmp_path):
    obj = _make_ellipse_project(tmp_path)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) != 0, "No images written."


def test_preprocess_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, True)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) != 0, "No images written."


def test_register(tmp_path):
    obj = _make_ellipse_project(tmp_path)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


def test_register_with_registration_pixel_cap(tmp_path):
    obj = _make_ellipse_project(tmp_path)
    obj.register(max_registration_pixels=1_000_000)
    assert obj.is_registered, "Registration failed."
    width, height = obj.preprocessed_cache["image_sizes"]["source"]
    assert width * height <= 1_000_000
    assert (width, height) != (2048, 2048)


def test_register_through(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_through=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


def test_register_with_initial(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


def test_register_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_mask=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


def test_register_with_bbox_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_bbox=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


def test_register_with_initial_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_mask=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


def test_register_through_with_initial(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_through=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


def test_register_with_initial_with_bbox_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_bbox=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


def test_add_modality_rejects_mask_conflicts(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    mask = get_test_file("ellipse_mask.tiff")
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)

    with pytest.raises(ValueError, match="Mask can only be specified"):
        obj.add_modality("source", source, preprocessing=Preprocessing.basic(), mask=mask, mask_bbox=(0, 0, 10, 10))


def test_auto_add_modality_rejects_preprocessing_mask_conflicts(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    mask = get_test_file("ellipse_mask.tiff")
    preprocessing = Preprocessing(mask_polygon=[[0, 0], [0, 10], [10, 10], [10, 0]])
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)

    with pytest.raises(ValueError, match="Mask can only be specified"):
        obj.auto_add_modality("source", source, preprocessing=preprocessing, mask=mask)


def test_elastix_parameter_line_parser_skips_blank_and_comment_lines():
    assert _elx_lineparser("") == (None, None)
    assert _elx_lineparser("# comment") == (None, None)
    assert _elx_lineparser("// comment") == (None, None)


def test_elastix_parameter_line_parser_parses_values():
    assert _elx_lineparser('(Transform "EulerTransform")') == ("Transform", ["EulerTransform"])
    assert _elx_lineparser("(NumberOfResolutions 4)") == ("NumberOfResolutions", ["4"])


def test_elastix_parameter_line_parser_rejects_malformed_lines():
    with pytest.raises(ValueError, match="Malformed Elastix parameter line"):
        _elx_lineparser("(Transform)")


def test_register_2d_images_releases_wrapper_image_data(tmp_path):
    source = MagicMock(spec=ImageWrapper)
    source.name = "source"
    source.image = sitk.GetImageFromArray(np.ones((8, 8), dtype=np.uint8))
    source.mask = None
    target = MagicMock(spec=ImageWrapper)
    target.name = "target"
    target.image = sitk.GetImageFromArray(np.ones((8, 8), dtype=np.uint8))
    target.mask = None

    transform_parameters = MagicMock()
    transform_parameters.GetNumberOfParameterMaps.return_value = 1
    transform_parameters.GetParameterMap.return_value = {"Transform": ["EulerTransform"]}
    registrar = MagicMock()
    registrar.GetTransformParameterObject.return_value = transform_parameters
    parameter_object = MagicMock()

    with (
        patch("image2image_reg.elastix.registration_utils.sitk_image_to_itk_image", side_effect=["moving", "fixed"]),
        patch("image2image_reg.elastix.registration_utils.itk") as itk_mock,
    ):
        itk_mock.ElastixRegistrationMethod.New.return_value = registrar
        itk_mock.ParameterObject.New.return_value = parameter_object

        transforms = register_2d_images(
            source,
            target,
            [{"Transform": ["EulerTransform"]}],
            tmp_path,
            max_registration_pixels=100,
        )

    assert transforms == [{"Transform": ["EulerTransform"]}]
    source.release_image_data.assert_called_once_with()
    target.release_image_data.assert_called_once_with()
    itk_mock.ElastixRegistrationMethod.New.assert_called_once_with("moving", "fixed")
    registrar.SetMovingImage.assert_not_called()
    registrar.SetFixedImage.assert_not_called()

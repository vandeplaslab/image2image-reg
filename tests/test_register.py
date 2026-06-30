"""Test registration."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, patch

import itk
import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.elastix.registration import _elx_lineparser
from image2image_reg.elastix.registration_utils import register_2d_images
from image2image_reg.elastix.transform_sequence import Transform, TransformSequence
from image2image_reg.elastix.transformation_map import BASE_TRANSLATION_TRANSFORM
from image2image_reg.models import Modality, Preprocessing
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


def _make_registration_wrapper(name: str, array: np.ndarray) -> ImageWrapper:
    wrapper = MagicMock(spec=ImageWrapper)
    wrapper.name = name
    wrapper.image = sitk.GetImageFromArray(array)
    wrapper.image.SetSpacing((1.0, 1.0))  # type: ignore[union-attr]
    wrapper.mask = None
    return wrapper


def _make_translated_registration_images() -> tuple[np.ndarray, np.ndarray]:
    size = 96
    y, x = np.mgrid[:size, :size]
    target = (
        180 * np.exp(-(((x - 45) ** 2) / (2 * 12**2) + ((y - 52) ** 2) / (2 * 10**2)))
        + 80 * np.exp(-(((x - 65) ** 2) / (2 * 5**2) + ((y - 28) ** 2) / (2 * 6**2)))
    )
    target = (target / target.max() * 255).astype(np.uint8)
    source = np.zeros_like(target)
    source[: size - 5, 8:size] = target[5:size, : size - 8]
    return source, target


def _translation_registration_parameters() -> dict[str, list[str]]:
    parameter_map = {
        key: list(value)
        for key, value in itk.ParameterObject.New().GetDefaultParameterMap("translation").items()
    }
    parameter_map.update(
        {
            "AutomaticTransformInitialization": ["false"],
            "ImageSampler": ["Full"],
            "MaximumNumberOfIterations": ["80"],
            "NewSamplesEveryIteration": ["false"],
            "NumberOfResolutions": ["2"],
            "WriteTransformParametersEachResolution": ["false"],
        },
    )
    return parameter_map


def _translation_from_transform(transform: dict[str, list[str]]) -> np.ndarray:
    return np.asarray(transform["TransformParameters"], dtype=float)


def _mean_absolute_registration_error(
    moving: np.ndarray,
    fixed: np.ndarray,
    translation: np.ndarray,
) -> float:
    moving_image = sitk.GetImageFromArray(moving)
    fixed_image = sitk.GetImageFromArray(fixed)
    moving_image.SetSpacing((1.0, 1.0))
    fixed_image.SetSpacing((1.0, 1.0))
    transform = sitk.TranslationTransform(2, translation.tolist())
    resampled = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0, moving_image.GetPixelID())
    return float(np.mean(np.abs(sitk.GetArrayFromImage(resampled).astype(float) - fixed.astype(float))))


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
    config = obj._get_config(registered=True)
    assert config["registration_image_sizes"]["source"] == obj.preprocessed_cache["image_sizes"]["source"]
    assert config["registration_pixel_cap_factors"]["source"] > 1


def _make_transform_sequence(spacing: float = 4.0, size: int = 5) -> TransformSequence:
    transform = deepcopy(BASE_TRANSLATION_TRANSFORM)
    transform["TransformParameters"] = ["8", "-5"]
    transform["Spacing"] = [str(spacing), str(spacing)]
    transform["Size"] = [str(size), str(size)]
    return TransformSequence([Transform(transform)], transform_sequence_index=[0])


def test_prepare_registered_transform_removes_registration_pixel_cap(tmp_path):
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    obj.modalities = {
        "source": Modality(name="source", path="source.tiff"),
        "target": Modality(name="target", path="target.tiff"),
    }
    obj.registration_paths = {"source": ["target"]}
    obj.transformations = {"source": {"full-transform-seq": _make_transform_sequence()}}
    obj.preprocessed_cache["image_spacing"]["target"] = (4.0, 4.0)
    obj.preprocessed_cache["image_sizes"]["target"] = (5, 5)
    obj.preprocessed_cache["registration_pixel_cap_factors"]["target"] = 4.0

    _, transform_seq, _ = obj._prepare_registered_transform("source")
    transform = transform_seq.transforms[-1].elastix_transform

    assert transform["Spacing"] == ["1.0", "1.0"]
    assert transform["Size"] == ["20", "20"]
    assert transform["TransformParameters"] == ["8", "-5"]


def test_prepare_registered_transform_keeps_output_pixel_size_precedence(tmp_path):
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    obj.modalities = {
        "source": Modality(name="source", path="source.tiff", output_pixel_size=(2.0, 2.0)),
        "target": Modality(name="target", path="target.tiff"),
    }
    obj.registration_paths = {"source": ["target"]}
    obj.transformations = {"source": {"full-transform-seq": _make_transform_sequence()}}
    obj.preprocessed_cache["image_spacing"]["target"] = (4.0, 4.0)
    obj.preprocessed_cache["image_sizes"]["target"] = (5, 5)
    obj.preprocessed_cache["registration_pixel_cap_factors"]["target"] = 4.0

    _, transform_seq, _ = obj._prepare_registered_transform("source")
    transform = transform_seq.transforms[-1].elastix_transform

    assert transform["Spacing"] == ["2.0", "2.0"]
    assert transform["Size"] == ["10", "10"]
    assert transform["TransformParameters"] == ["8", "-5"]


def test_prepare_not_registered_transform_creates_uncapped_identity_grid(tmp_path):
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    obj.modalities = {"target": Modality(name="target", path="target.tiff")}
    obj.preprocessed_cache["image_spacing"]["target"] = (4.0, 4.0)
    obj.preprocessed_cache["image_sizes"]["target"] = (5, 5)
    obj.preprocessed_cache["registration_pixel_cap_factors"]["target"] = 4.0

    _, transform_seq, _ = obj._prepare_not_registered_transform("target")
    assert transform_seq is not None
    transform = transform_seq.transforms[-1].elastix_transform

    assert transform["Spacing"] == ["1.0", "1.0"]
    assert transform["Size"] == ["20", "20"]


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
    source = _make_registration_wrapper("source", np.ones((8, 8), dtype=np.uint8))
    target = _make_registration_wrapper("target", np.ones((8, 8), dtype=np.uint8))

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
    itk_mock.ElastixRegistrationMethod.New.assert_called_once_with("fixed", "moving")
    registrar.SetMovingImage.assert_not_called()
    registrar.SetFixedImage.assert_not_called()


def test_register_2d_images_preserves_moving_fixed_direction_with_cap(tmp_path):
    source_image, target_image = _make_translated_registration_images()
    parameters = _translation_registration_parameters()

    uncapped = register_2d_images(
        _make_registration_wrapper("source", source_image),
        _make_registration_wrapper("target", target_image),
        [parameters],
        tmp_path / "uncapped",
        max_registration_pixels=0,
    )
    capped = register_2d_images(
        _make_registration_wrapper("source", source_image),
        _make_registration_wrapper("target", target_image),
        [parameters],
        tmp_path / "capped",
        max_registration_pixels=1_000,
    )

    uncapped_translation = _translation_from_transform(uncapped[-1])
    capped_translation = _translation_from_transform(capped[-1])

    assert uncapped_translation == pytest.approx([8, -5], abs=0.75)
    assert capped_translation == pytest.approx([8, -5], abs=1.25)
    assert np.sign(capped_translation).tolist() == np.sign(uncapped_translation).tolist()

    unregistered_error = float(np.mean(np.abs(source_image.astype(float) - target_image.astype(float))))
    uncapped_error = _mean_absolute_registration_error(source_image, target_image, uncapped_translation)
    capped_error = _mean_absolute_registration_error(source_image, target_image, capped_translation)
    assert uncapped_error < unregistered_error * 0.5
    assert capped_error < unregistered_error * 0.6


def test_coregister_images_keeps_source_as_moving_and_target_as_fixed(tmp_path):
    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    source = MagicMock(spec=ImageWrapper)
    source.name = "source"
    source.initial_transforms = []
    target = MagicMock(spec=ImageWrapper)
    target.name = "target"
    target.original_size_transform = None
    transform = deepcopy(BASE_TRANSLATION_TRANSFORM)
    transform["TransformParameters"] = ["8", "-5"]
    transform["Size"] = ["96", "96"]
    transform["Spacing"] = ["1", "1"]

    with patch("image2image_reg.elastix.registration_utils.register_2d_images", return_value=[transform]) as register:
        obj._coregister_images(
            source,
            target,
            ["rigid"],
            tmp_path,
            max_registration_pixels=1_000,
    )

    assert register.call_args.args[:2] == (source, target)
    assert register.call_args.kwargs["max_registration_pixels"] is None

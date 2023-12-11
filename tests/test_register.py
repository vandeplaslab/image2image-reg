"""Test registration."""
from pathlib import Path

from image2image_wsireg.models import Preprocessing
from image2image_wsireg.utils._test import get_test_file
from image2image_wsireg.workflows import WsiReg2d


def _make_project(tmp_path: Path, with_mask: bool = False) -> WsiReg2d:
    source = get_test_file("moving_image_8bit.tiff")
    target = get_test_file("target_image_8bit.tiff")
    mask = get_test_file("mask_image.tiff") if with_mask else None

    obj = WsiReg2d(name="test.wsireg", output_dir=tmp_path, cache=True, merge=True)
    obj.set_logger()
    obj.add_modality("source", source, preprocessing=Preprocessing.basic())
    obj.add_modality("target", target, mask=mask, preprocessing=Preprocessing.basic())
    obj.add_registration_path("source", "target", transform=["rigid"])
    return obj


def test_preprocess(tmp_path):
    obj = _make_project(tmp_path)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) == 2, "No images written."


def test_preprocess_with_mask(tmp_path):
    obj = _make_project(tmp_path, True)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) == 3, "No images written."


def test_register(tmp_path):
    obj = _make_project(tmp_path)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write_images()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


def test_register_with_mask(tmp_path):
    obj = _make_project(tmp_path)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write_images()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."

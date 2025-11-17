"""Test registration."""

from pathlib import Path

import numpy as np
import pytest

from image2image_reg.models import Preprocessing
from image2image_reg.utils._test import get_test_file
from image2image_reg.workflows import ElastixReg


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


@pytest.mark.xfail(reason="need to fix")
def test_preprocess(tmp_path):
    obj = _make_ellipse_project(tmp_path)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_preprocess_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, True)
    obj.preprocess()
    assert not obj.is_registered, "Registration failed."
    assert len(list(obj.cache_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register(tmp_path):
    obj = _make_ellipse_project(tmp_path)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_through(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_through=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_with_initial(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_mask=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_with_bbox_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_bbox=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_with_initial_with_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_mask=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) == 2, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_through_with_initial(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_through=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."


@pytest.mark.xfail(reason="need to fix")
def test_register_with_initial_with_bbox_mask(tmp_path):
    obj = _make_ellipse_project(tmp_path, with_bbox=True, with_initial=True)
    obj.register()
    assert obj.is_registered, "Registration failed."
    obj.write()
    assert len(list(obj.image_dir.glob("*.tiff"))) != 0, "No images written."

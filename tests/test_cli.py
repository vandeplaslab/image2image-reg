"""Test CLI."""

import os

import pytest

from image2image_reg.utils._test import get_test_file
from image2image_reg.workflows import ElastixReg


@pytest.mark.xfail(reason="need to fix")
def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("i2reg --help")
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_init(tmp_path):
    """Test CLI init."""
    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert (tmp / "test.wsireg").exists(), "No config file created."
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_add_images_path_attachment(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    polygon = get_test_file("polygons.geojson")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0

    # add shapes
    exit_status = os.system(f"i2reg --debug elastix add-shape -p '{path!s}' -a source -n shape -f {polygon!s}")
    assert exit_status == 0

    # add merge modalities
    exit_status = os.system(f"i2reg --debug elastix add-merge -p '{path!s}' -n merge -m source -m target")
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_add_images_override_preprocessing(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0

    # add paths
    exit_status = os.system(
        f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid -S light -P dark"
    )
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_add_images_with_affine(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    affine = get_test_file("ellipse_affine.json")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic -A {affine!s}"
    )
    assert exit_status == 0
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_add_images_path_mask(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    mask = get_test_file("ellipse_mask.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic -m {mask!s}"
    )
    assert exit_status == 0

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_add_images_path_mask_bbox(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic -b 0,0,1000,1000"
    )
    assert exit_status == 0

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0

    obj = ElastixReg.from_path(path)
    modality = obj.modalities["target"]
    assert modality.preprocessing.mask_bbox is not None, "No mask bbox found."
    assert modality.preprocessing.mask_bbox.x == 0
    assert modality.preprocessing.mask_bbox.y == 0
    assert modality.preprocessing.mask_bbox.width == 1000
    assert modality.preprocessing.mask_bbox.height == 1000


@pytest.mark.xfail(reason="need to fix")
def test_cli_merge(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg merge -n test -o '{tmp!s}' -p '{source!s}' -p '{target!s}'")
    assert list(tmp.glob("*.ome.tiff")), "No merged images found."
    assert exit_status == 0


@pytest.mark.xfail(reason="need to fix")
def test_cli_merge_with_crop(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg merge -n test -o '{tmp!s}' -p '{source!s}' -p '{target!s}' -b 512,512,512,512")
    assert list(tmp.glob("*.ome.tiff")), "No merged images found."
    assert exit_status == 0

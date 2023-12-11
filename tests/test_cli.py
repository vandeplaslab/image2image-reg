"""Test CLI."""
import os

from image2image_wsireg.utils._test import get_test_file


def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("iwsireg --help")
    assert exit_status == 0


def test_cli_init(tmp_path):
    """Test CLI init."""
    tmp = tmp_path
    exit_status = os.system(f"iwsireg new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert (tmp / "test.wsireg").exists(), "No config file created."
    assert exit_status == 0


def test_cli_add_images_path_attachment(tmp_path):
    """Test CLI init."""
    source = get_test_file("moving_image_8bit.tiff")
    target = get_test_file("target_image_8bit.tiff")
    polygon = get_test_file("polygons.geojson")

    tmp = tmp_path
    exit_status = os.system(f"iwsireg new -n test.wsireg -o '{tmp!s}' --cache --merge")
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."
    assert exit_status == 0

    # add images
    exit_status = os.system(f"iwsireg --debug add-image -p '{path!s}' -n source -i '{source!s}' -P none")
    assert exit_status == 0
    exit_status = os.system(f"iwsireg --debug add-image -p '{path!s}' -n target -i '{target!s}' -P none")
    assert exit_status == 0

    # add paths
    exit_status = os.system(f"iwsireg --debug add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0

    # add shapes
    exit_status = os.system(f"iwsireg --debug add-shape -p '{path!s}' -a source -n shape -s {polygon!s}")
    assert exit_status == 0

    # add merge modalities
    exit_status = os.system(f"iwsireg --debug add-merge -p '{path!s}' -n merge -m source -m target")
    assert exit_status == 0

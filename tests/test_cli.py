"""Test CLI."""
import os

from image2image_wsireg.utils._test import get_test_files


def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("iwsireg --help")
    assert exit_status == 0


def test_cli_init(tmp_path):
    """Test CLI init."""
    tmp = tmp_path / "test-init"
    tmp.mkdir()
    exit_status = os.system(f"iwsireg new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert (tmp / "test.wsireg").exists(), "No config file created."
    assert exit_status == 0


def test_cli_add_images(tmp_path):
    """Test CLI init."""
    images = get_test_files("*.tiff")
    assert len(images) >= 2, "Not enough images for testing."
    source = images[0]
    target = images[1]

    tmp = tmp_path / "test-add-images"
    tmp.mkdir()
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

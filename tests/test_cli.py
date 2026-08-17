"""Test CLI."""

import os
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from image2image_reg.cli.elastix import (
    _export,
    _register,
    add_modality_runner,
    export_cmd,
    parse_max_registration_pixels,
    register_cmd,
)
from image2image_reg.cli.merge import arg_parse_channel_ids
from image2image_reg.cli.valis import cli_parse_method
from image2image_reg.utils._test import get_test_file
from image2image_reg.workflows import ElastixReg


def test_cli_merge_channel_ids_parser():
    """Test merge channel id parser."""
    command = click.Command("merge")
    option = click.Option(["--channel_ids"])
    ctx = click.Context(command)

    assert arg_parse_channel_ids(ctx, option, ("1,2,4-6", "0")) == ((1, 2, 4, 5, 6), (0,))

    with pytest.raises(click.BadParameter):
        arg_parse_channel_ids(ctx, option, ("4-2",))


def test_cli_valis_preprocessing_aliases():
    """Test Valis pre-processing shortcut expansion."""
    assert cli_parse_method(None, None, ["cs", "mip", "CustomProcessor"]) == [
        "ColorfulStandardizer",
        "MaxIntensityProjection",
        "CustomProcessor",
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("off", 0),
        ("100000000", 100_000_000),
        ("100M", 100_000_000),
        ("15m", 15_000_000),
        ("1B", 1_000_000_000),
        ("1.5M", 1_500_000),
        ("250 k px", 250_000),
    ],
)
def test_cli_max_registration_pixels_parser(value, expected):
    """Test compact pixel cap parser."""
    command = click.Command("register")
    option = click.Option(["--max-registration-pixels"])
    ctx = click.Context(command)

    assert parse_max_registration_pixels(ctx, option, value) == expected


def test_cli_max_registration_pixels_parser_rejects_invalid_values():
    """Test compact pixel cap parser rejects invalid values."""
    command = click.Command("register")
    option = click.Option(["--max-registration-pixels"])
    ctx = click.Context(command)

    with pytest.raises(click.BadParameter):
        parse_max_registration_pixels(ctx, option, "1T")


def test_cli_add_image_rejects_mismatched_masks():
    """Test repeatable add-image options reject mismatched lengths."""
    with pytest.raises(ValueError, match="Number of mask values"):
        add_modality_runner(
            project_dir="unused.wsireg",
            names=["source", "target"],
            paths=["source.tiff", "target.tiff"],
            masks=["mask.tiff"],
            preprocessings=["basic"],
        )


def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("i2reg --help")
    assert exit_status == 0


@pytest.mark.parametrize("command", [register_cmd, export_cmd])
def test_cli_elastix_export_commands_expose_native_resolution(command):
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--native_resolution" in result.output


@pytest.mark.parametrize("runner", [_register, _export])
def test_cli_elastix_export_commands_propagate_native_resolution(runner):
    obj = MagicMock()
    obj.is_registered = True

    with patch("image2image_reg.workflows.elastix.ElastixReg.from_path", return_value=obj):
        if runner is _register:
            runner(
                "project.wsireg",
                histogram_match=False,
                max_registration_pixels=None,
                write_images=True,
                fmt="ome-tiff",
                write_registered=True,
                write_not_registered=False,
                write_attached=False,
                write_merged=False,
                remove_merged=False,
                original_size=True,
                as_uint8=False,
                native_resolution=True,
            )
        else:
            runner(
                "project.wsireg",
                fmt="ome-tiff",
                write_registered=True,
                write_not_registered=False,
                write_attached_images=False,
                write_attached_shapes=False,
                write_attached_points=False,
                write_merged=False,
                remove_merged=False,
                original_size=True,
                as_uint8=False,
                native_resolution=True,
            )

    assert obj.write.call_args.kwargs["write_native_resolution"] is True


def test_cli_init(tmp_path):
    """Test CLI init."""
    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert (tmp / "test.wsireg").exists(), "No config file created."
    assert exit_status == 0


def test_cli_add_images_path_attachment(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    polygon = get_test_file("polygons.geojson")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert exit_status == 0, "Failed to create config file."
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0, "Failed to add source image."
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0, "Failed to add target image."

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0, "Failed to add registration path."

    # add shapes
    exit_status = os.system(f"i2reg --debug elastix attach-shape -p '{path!s}' -a source -n shape -f {polygon!s}")
    assert exit_status == 0, "Failed to attach shape."

    # add merge modalities
    exit_status = os.system(f"i2reg --debug elastix add-merge -p '{path!s}' -n merge -m source -m target")
    assert exit_status == 0, "Failed to add merge modality."


def test_cli_add_images_override_preprocessing(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert exit_status == 0, "Failed to create config file."
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0, "Failed to add source image."
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0, "Failed to add target image."

    # add paths
    exit_status = os.system(
        f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid -S light -P dark",
    )
    assert exit_status == 0, "Failed to add registration path."


#
#
def test_cli_add_images_with_affine(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    affine = get_test_file("ellipse_affine.json")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert exit_status == 0, "Failed to create config file."
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."

    # add images
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic -A {affine!s}",
    )
    assert exit_status == 0, "Failed to add source image."
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic")
    assert exit_status == 0, "Failed to add target image."
    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0, "Failed to add registration path."


def test_cli_add_images_path_mask(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")
    mask = get_test_file("ellipse_mask.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert exit_status == 0, "Failed to create config file."
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0, "Failed to add source image."
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic -m {mask!s}",
    )
    assert exit_status == 0, "Failed to add target image."

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0, "Failed to add registration path."


def test_cli_add_images_path_mask_bbox(tmp_path):
    """Test CLI init."""
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg elastix new -n test.wsireg -o '{tmp!s}' --cache --merge")
    assert exit_status == 0, "Failed to create config file."
    path = tmp / "test.wsireg"
    assert path.exists(), "No config file created."

    # add images
    exit_status = os.system(f"i2reg --debug elastix add-image -p '{path!s}' -n source -i '{source!s}' -P basic")
    assert exit_status == 0, "Failed to add source image."
    exit_status = os.system(
        f"i2reg --debug elastix add-image -p '{path!s}' -n target -i '{target!s}' -P basic -b 0,0,1000,1000",
    )
    assert exit_status == 0, "Failed to add target image."

    # add paths
    exit_status = os.system(f"i2reg --debug elastix add-path -p '{path!s}' -s source -t target -R rigid")
    assert exit_status == 0, "Failed to add registration path."

    obj = ElastixReg.from_path(path)
    modality = obj.modalities["target"]
    assert modality.preprocessing.mask_bbox is not None, "No mask bbox found."
    assert modality.preprocessing.mask_bbox.x == [0]
    assert modality.preprocessing.mask_bbox.y == [0]
    assert modality.preprocessing.mask_bbox.width == [1000]
    assert modality.preprocessing.mask_bbox.height == [1000]


def test_cli_merge(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg merge -n test -o '{tmp!s}' -p '{source!s}' -p '{target!s}'")
    assert exit_status == 0, "Merge command failed."
    assert list(tmp.glob("*.ome.tiff")), "No merged images found."


def test_cli_merge_with_crop(tmp_path):
    source = get_test_file("ellipse_moving.tiff")
    target = get_test_file("ellipse_target.tiff")

    tmp = tmp_path
    exit_status = os.system(f"i2reg merge -n test -o '{tmp!s}' -p '{source!s}' -p '{target!s}' -b 512,512,512,512")
    assert exit_status == 0, "Merge command failed."
    assert list(tmp.glob("*.ome.tiff")), "No merged images found."

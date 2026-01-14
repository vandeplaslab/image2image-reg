"""Test bbox."""

import numpy as np
import pytest
from pydantic import ValidationError

from image2image_reg.enums import CoordinateFlip, ImageType, BackgroundSubtractType
from image2image_reg.models import BoundingBox, Export, Modality, Polygon, Preprocessing


def test_modality():
    modality = Modality(
        name="test",
        path=np.random.random((100, 100)),
        pixel_size=0.5,
        output_pixel_size=(1.0, 1.0),
    )
    assert modality.name == "test", "Name should be test"
    assert isinstance(modality.path, np.ndarray), "Path should be np.ndarray"
    assert modality.pixel_size == 0.5, "Pixel size should be 0.5"
    assert modality.output_pixel_size == (1.0, 1.0), "Output pixel size should be (1.0, 1.0)"

    data = modality.to_dict()
    assert data["name"] == "test", "Name should be test"
    assert data["pixel_size"] == 0.5, "Pixel size should be 0.5"
    assert data["output_pixel_size"] == (1.0, 1.0), "Output pixel size should be (1.0, 1.0)"


def test_polygon():
    poly = Polygon(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
    assert isinstance(poly.xy, list), "xy should be a list"
    assert len(poly.xy) == 1, "Length should be 1"
    assert poly.xy[0].shape == (4, 2), "Shape should be (4, 2)"

    mask = poly.to_mask((100, 100))
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 255, "Max value should be 255"

    mask = poly.to_mask((150, 100))
    assert mask.shape == (150, 100), "Shape should be (150, 100)"
    assert mask.max() == 255, "Max value should be 255"

    image = poly.to_sitk_image((100, 100), pixel_size=1.0)
    assert image.GetSize() == (100, 100), "Size should be (100, 100)"
    assert image.GetSpacing() == (1.0, 1.0), "Spacing should be (1.0, 1.0)"
    assert image.GetPixelID() == 1, "PixelID should be 1 (uint8)"

    data = poly.to_dict()
    assert isinstance(data, list), "data should be a list"


def test_polygon_list():
    poly = Polygon(
        [
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]]) + 10,
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]]) + 20,
        ],
    )
    assert isinstance(poly.xy, list), "xy should be a list"
    assert len(poly.xy) == 3, "Length should be 3"

    mask = poly.to_mask((100, 100))
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 255, "Max value should be 255"

    mask = poly.to_mask((150, 100))
    assert mask.shape == (150, 100), "Shape should be (150, 100)"
    assert mask.max() == 255, "Max value should be 255"

    image = poly.to_sitk_image((100, 100), pixel_size=1.0)
    assert image.GetSize() == (100, 100), "Size should be (100, 100)"
    assert image.GetSpacing() == (1.0, 1.0), "Spacing should be (1.0, 1.0)"
    assert image.GetPixelID() == 1, "PixelID should be 1 (uint8)"

    data = poly.to_dict()
    assert isinstance(data, list), "xy should be a list"
    assert len(data) == 3, "Length should be 3"
    assert isinstance(data[0], list), "xy should be a list"


def test_bbox():
    bbox = BoundingBox(0, 0, 10, 10)
    assert bbox.x == [0], "X should be 0"
    assert bbox.y == [0], "Y should be 0"
    assert bbox.width == [10], "WIDTH should be 10"
    assert bbox.height == [10], "HEIGHT should be 10"

    mask = bbox.to_mask((100, 100))
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 1, "Max value should be 1"

    mask = bbox.to_mask((100, 100), dtype=int, value=255)
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 255, "Max value should be 255"

    image = bbox.to_sitk_image((100, 100), pixel_size=1.0)
    assert image.GetSize() == (100, 100), "Size should be (100, 100)"
    assert image.GetSpacing() == (1.0, 1.0), "Spacing should be (1.0, 1.0)"
    assert image.GetPixelID() == 1, "PixelID should be 1 (uint8)"


def test_bbox_list():
    bbox = BoundingBox([0, 5], [0, 5], [10, 2], [10, 2])
    assert bbox.x == [0, 5], "X should be 0"
    assert bbox.y == [0, 5], "Y should be 0"
    assert bbox.width == [10, 2], "WIDTH should be 10"
    assert bbox.height == [10, 2], "HEIGHT should be 10"

    mask = bbox.to_mask((100, 100))
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 1, "Max value should be 1"

    mask = bbox.to_mask((100, 100), dtype=int, value=255)
    assert mask.shape == (100, 100), "Shape should be (100, 100)"
    assert mask.max() == 255, "Max value should be 255"

    image = bbox.to_sitk_image((100, 100), pixel_size=1.0)
    assert image.GetSize() == (100, 100), "Size should be (100, 100)"
    assert image.GetSpacing() == (1.0, 1.0), "Spacing should be (1.0, 1.0)"
    assert image.GetPixelID() == 1, "PixelID should be 1 (uint8)"


def test_export():
    export = Export(as_uint8=True, channel_ids=[0, 1, 2])
    assert export.as_uint8 is True, "as_uint8 should be True"
    assert export.channel_ids == [0, 1, 2], "channel_ids should be [0, 1, 2]"

    data = {"as_uint8": True, "channel_ids": [0, 1, 2]}
    export = Export(**data)
    assert export.as_uint8 is True, "as_uint8 should be True"
    assert export.channel_ids == [0, 1, 2], "channel_ids should be [0, 1, 2]"


def test_prepro():
    prepro = Preprocessing(translate_x=0, translate_y=0, rotate_counter_clockwise=0)
    # validate image_type
    prepro.image_type = "BF"
    assert isinstance(prepro.image_type, ImageType), "ImageType should be ImageType"
    assert prepro.image_type == ImageType.LIGHT, "image_type should be Dark"
    prepro.image_type = ImageType.LIGHT
    assert prepro.image_type == ImageType.LIGHT, "image_type should be Light"
    # validate flip
    assert prepro.flip is None, "flip should be None"
    prepro.flip = "horz"
    assert prepro.flip == CoordinateFlip.HORIZONTAL
    assert isinstance(prepro.flip, CoordinateFlip), "flip should be CoordinateFlip"
    prepro.flip = "v"
    assert isinstance(prepro.flip, CoordinateFlip), "flip should be CoordinateFlip"
    assert prepro.flip == CoordinateFlip.VERTICAL, "flip should be Vertical"
    prepro.background_subtract = "sharp"
    assert isinstance(prepro.background_subtract, BackgroundSubtractType), (
        "background_subtract should be BackgroundSubtractType"
    )
    assert prepro.background_subtract == "sharp", "background_subtract should be sharp"
    prepro.background_subtract = BackgroundSubtractType.SMOOTH
    assert prepro.background_subtract == "smooth", "background_subtract should be smooth"
    prepro.background_subtract = None
    assert prepro.background_subtract == "none", "background_subtract should be none"

    # change translate
    prepro.translate_x = 50
    assert prepro.translate_x == 50, "translate_x should be 50"
    prepro.translate_y = 50
    assert prepro.translate_y == 50, "translate_y should be 50"
    # change rotate
    prepro.rotate_counter_clockwise = 360
    assert prepro.rotate_counter_clockwise == 0, "rotate_counter_clockwise should be 0"
    prepro.rotate_counter_clockwise = 360.0
    assert prepro.rotate_counter_clockwise == 0, "rotate_counter_clockwise should be 0"
    # validate affine
    with pytest.raises(ValidationError):
        prepro.affine = np.eye(2)
    prepro.affine = np.eye(3)
    assert isinstance(prepro.affine, np.ndarray), "affine should be np.ndarray"

    # check config
    text, tooltip = prepro.as_str()
    assert isinstance(text, str), "text should be a str"
    assert isinstance(tooltip, str), "tooltip should be a str"

    dump = prepro.model_dump()
    assert isinstance(dump, dict), "dump should be a dict"
    assert isinstance(dump["image_type"], str), "field should be str"
    assert isinstance(dump["flip"], str), "field should be str"

    wsireg = prepro.to_dict(as_wsireg=True)
    assert isinstance(wsireg, dict), "wsireg should be a dict"
    for k in ["channel_indices", "crop_bbox", "downsample", "rotate_counter_clockwise", "translate_x", "translate_y"]:
        assert k not in wsireg, f"{k} should be in wsireg"


def test_prepro_with_mask():
    prepro = Preprocessing(use_mask=True, mask_bbox=[0, 0, 100, 100])
    assert prepro.mask_bbox is not None, "mask is None"
    config = prepro.to_dict()
    assert config, "Config is missing"
    assert not isinstance(config["mask_bbox"], BoundingBox), "mask_bbox should not be a bbox"

    prepro = Preprocessing(use_crop=True, mask_polygon=[[100, 0], [200, 0], [400, 0]])
    assert prepro.mask_polygon is not None, "mask is None"
    assert prepro.is_masked(), "is_masked should be True"
    config = prepro.to_dict()
    assert config, "Config is missing"
    assert not isinstance(config["mask_polygon"], Polygon), "mask_polygon should not be a bbox"


def test_prepro_with_crop():
    prepro = Preprocessing(use_crop=True, crop_bbox=[0, 0, 100, 100])
    assert prepro.crop_bbox is not None, "crop is None"
    config = prepro.to_dict()
    assert config, "Config is missing"
    assert not isinstance(config["crop_bbox"], BoundingBox), "crop_bbox should not be a bbox"

    prepro = Preprocessing(use_crop=True, crop_polygon=[[100, 0], [200, 0], [400, 0]])
    assert prepro.crop_polygon is not None, "crop is None"
    assert prepro.is_cropped(), "is_cropped should be True"
    config = prepro.to_dict()
    assert config, "Config is missing"
    assert not isinstance(config["crop_polygon"], Polygon), "crop_polygon should not be a bbox"


def test_prepro_channel_indices():
    prepro = Preprocessing(channel_names=["dapi", "egfp", "mcherry"], channel_indices=[0, 1, 2])
    prepro.select_channel(channel_name="egfp")
    assert prepro.channel_indices == [1], "channel_indices should be [1]"
    prepro.select_channel(channel_name="mcherry")
    assert prepro.channel_indices == [2], "channel_indices should be [2]"
    prepro.select_channel(channel_name="dapi")
    assert prepro.channel_indices == [0], "channel_indices should be [0]"

    prepro.select_channels(["dapi", "mcherry"])
    assert prepro.channel_indices == [0, 2], "channel_indices should be [0, 2]"


def test_prepro_defaults():
    # some dark field modalities
    prepro = Preprocessing.fluorescence()
    assert prepro.image_type == ImageType.DARK, "image_type should be DARK"
    prepro = Preprocessing.dapi()
    assert prepro.image_type == ImageType.DARK, "image_type should be DARK"
    prepro = Preprocessing.postaf()
    assert prepro.image_type == ImageType.DARK, "image_type should be DARK"

    # some brightfield modalities
    prepro = Preprocessing.brightfield()
    assert prepro.image_type == ImageType.LIGHT, "image_type should be LIGHT"
    prepro = Preprocessing.pas()
    assert prepro.image_type == ImageType.LIGHT, "image_type should be LIGHT"
    prepro = Preprocessing.he()
    assert prepro.image_type == ImageType.LIGHT, "image_type should be LIGHT"

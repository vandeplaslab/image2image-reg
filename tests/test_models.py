"""Test bbox."""
import numpy as np
import pytest
from pydantic import ValidationError

from image2image_reg.models import BoundingBox, Export, Polygon, Preprocessing


def test_polygon():
    poly = Polygon(np.array([[0, 0], [0, 10], [10, 10], [10, 0]]))
    assert poly.yx.shape == (4, 2), "Shape should be (4, 2)"

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


def test_bbox():
    bbox = BoundingBox(0, 0, 10, 10)
    assert bbox.x == 0, "X should be 0"
    assert bbox.y == 0, "Y should be 0"
    assert bbox.width == 10, "WIDTH should be 10"
    assert bbox.height == 10, "HEIGHT should be 10"

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
    prepro.translate_x = 50
    assert prepro.translate_x == 50, "translate_x should be 50"
    prepro.translate_y = 50
    assert prepro.translate_y == 50, "translate_y should be 50"

    with pytest.raises(ValidationError):
        prepro.affine = np.eye(2)

    prepro.rotate_counter_clockwise = 360
    assert prepro.rotate_counter_clockwise == 0, "rotate_counter_clockwise should be 0"
    prepro.rotate_counter_clockwise = 360.0
    assert prepro.rotate_counter_clockwise == 0, "rotate_counter_clockwise should be 0"

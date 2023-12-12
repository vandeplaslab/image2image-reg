"""Test bbox."""
from image2image_wsireg.models import BoundingBox


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

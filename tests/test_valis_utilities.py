"""Test Valis utility helpers."""

import pytest

from image2image_reg.valis.utilities import get_feature_detector_str


def test_feature_detector_aliases_return_canonical_names():
    """Test short detector aliases map to canonical Valis class names."""
    assert get_feature_detector_str("vgg") == "VggFD"
    assert get_feature_detector_str("svgg") == "SensitiveVggFD"


def test_feature_detector_accepts_canonical_names_case_insensitively():
    """Test canonical Valis detector names are accepted."""
    assert get_feature_detector_str("VggFD") == "VggFD"
    assert get_feature_detector_str("vggfd") == "VggFD"


def test_feature_detector_rejects_unknown_names():
    """Test unknown detector names raise a clear error."""
    with pytest.raises(ValueError, match="Feature detector unknown not found"):
        get_feature_detector_str("unknown")

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import SimpleITK as sitk

from image2image_reg.models import Modality, Preprocessing
from image2image_reg.wrapper import ImageWrapper


class TestImageWrapper:
    @pytest.fixture
    def mock_modality(self):
        modality = MagicMock(spec=Modality)
        modality.name = "test_modality"
        modality.pixel_size = 1.0
        modality.preprocessing = MagicMock(spec=Preprocessing)
        modality.preprocessing.mask = None
        modality.preprocessing.mask_bbox = None
        modality.preprocessing.mask_polygon = None
        modality.preprocessing.use_mask = False
        return modality

    @pytest.fixture
    def wrapper(self, mock_modality):
        return ImageWrapper(modality=mock_modality)

    def test_init(self, wrapper, mock_modality):
        assert wrapper.modality == mock_modality
        assert wrapper.image is None
        assert wrapper._mask is None

    def test_sitk_to_itk(self, wrapper):
        # Create a dummy SimpleITK image
        image = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        wrapper.image = image

        with patch("image2image_reg.wrapper.sitk_image_to_itk_image") as mock_convert:
            mock_convert.return_value = "itk_image"
            itk_image, mask = wrapper.sitk_to_itk()

            assert itk_image == "itk_image"
            assert mask is None
            mock_convert.assert_called_with(image)

    def test_read_mask_image(self, wrapper):
        # Test reading mask from SimpleITK image
        mask_image = sitk.GetImageFromArray(np.zeros((10, 10), dtype=np.uint8))
        result = wrapper.read_mask(mask_image)
        assert isinstance(result, sitk.Image)

    def test_read_mask_array(self, wrapper):
        # Test reading mask from numpy array
        mask_array = np.zeros((10, 10), dtype=np.uint8)
        result = wrapper.read_mask(mask_array)
        assert isinstance(result, sitk.Image)

    def test_read_mask_invalid(self, wrapper):
        # Test reading mask from invalid type
        with pytest.raises(TypeError, match="Unknown mask type"):
            wrapper.read_mask(123)

    def test_preprocess_intensity_optimization(self):
        # Verify that preprocess_intensity runs without error using the new numpy optimizations
        from image2image_reg.utils.preprocessing import preprocess_intensity

        # Create a small random image
        image = sitk.GetImageFromArray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        image.SetSpacing((1.0, 1.0))

        preprocessing = MagicMock(spec=Preprocessing)
        preprocessing.max_intensity_projection = False
        # Enabling flags that trigger the new numpy functions
        preprocessing.equalize_histogram = True
        preprocessing.contrast_enhance = True
        preprocessing.invert_intensity = True
        preprocessing.as_uint8 = True
        preprocessing.custom_processing = None

        # Should run without error
        result = preprocess_intensity(image, preprocessing, pixel_size=1.0, is_rgb=False)
        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (100, 100)

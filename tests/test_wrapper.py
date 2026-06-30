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
        image = sitk.GetImageFromArray(np.random.default_rng().integers(0, 255, (100, 100), dtype=np.uint8))
        image.SetSpacing((1.0, 1.0))

        preprocessing = MagicMock(spec=Preprocessing)
        preprocessing.max_intensity_projection = False
        # Enabling flags that trigger the new numpy functions
        preprocessing.equalize_histogram = True
        preprocessing.contrast_enhance = True
        preprocessing.invert_intensity = True
        preprocessing.as_uint8 = True
        preprocessing.custom_processing = None
        preprocessing.background_subtract = False

        # Should run without error
        result = preprocess_intensity(image, preprocessing, pixel_size=1.0, is_rgb=False)
        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (100, 100)

    def test_mask_uses_preprocessing_override(self, mock_modality):
        mask_array = np.ones((8, 8), dtype=np.uint8)
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(
            modality=mock_modality,
            preprocessing=Preprocessing(use_mask=True, mask=mask_array),
        )

        mask = wrapper.mask

        assert isinstance(mask, sitk.Image)
        assert mask.GetSize() == (8, 8)

    def test_preprocess_uses_override_transform_mask(self, mock_modality):
        mask_array = np.ones((8, 8), dtype=np.uint8)
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(
            modality=mock_modality,
            preprocessing=Preprocessing(use_mask=True, mask=mask_array, transform_mask=True),
        )
        reader = MagicMock()
        reader.pyramid = [np.ones((8, 8), dtype=np.uint8)]
        reader.channel_names = []
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (8, 8)
        wrapper._reader = reader

        wrapper.preprocess()

        assert isinstance(wrapper.image, sitk.Image)
        assert isinstance(wrapper.mask, sitk.Image)

    def test_preprocess_uses_capped_pyramid_level(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [
            np.ones((20, 20), dtype=np.uint8),
            np.ones((8, 8), dtype=np.uint8),
        ]
        reader.channel_names = []
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 20)
        reader.scale_for_pyramid.return_value = (2.5, 2.5)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=100)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetSize() == (8, 8)
        assert wrapper.image.GetSpacing() == (2.5, 2.5)
        assert wrapper.registration_pixel_cap_factor == 2.5

    def test_preprocess_strides_when_pyramid_level_exceeds_cap(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((20, 20), dtype=np.uint8)]
        reader.channel_names = []
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 20)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=25)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetSize() == (5, 5)
        assert wrapper.image.GetSpacing() == (4.0, 4.0)
        assert wrapper.registration_pixel_cap_factor == 4.0

    def test_preprocess_disabled_cap_keeps_base_level(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [
            np.ones((20, 20), dtype=np.uint8),
            np.ones((8, 8), dtype=np.uint8),
        ]
        reader.channel_names = []
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 20)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=0)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetSize() == (20, 20)
        assert wrapper.image.GetSpacing() == (1.0, 1.0)
        assert wrapper.registration_pixel_cap_factor == 1.0

    def test_preprocess_sets_stack_spacing_for_channel_first_image(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((3, 20, 20), dtype=np.uint8)]
        reader.channel_names = []
        reader.resolution = 2.0
        reader.is_rgb = False
        reader.image_shape = (20, 20)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=0)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetDimension() == 3
        assert wrapper.image.GetSpacing() == (2.0, 2.0, 1.0)

    def test_get_capped_image_preserves_rgb_channel_axis(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((20, 20, 3), dtype=np.uint8)]
        reader.channel_names = ["R", "G", "B"]
        reader.resolution = 1.0
        reader.is_rgb = True
        reader.image_shape = (20, 20)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        image, pixel_size = wrapper._get_capped_image(max_registration_pixels=25)

        assert image.shape == (5, 5, 3)
        assert pixel_size == 4.0

    def test_preprocess_handles_rgb_channel_first_image(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((3, 20, 20), dtype=np.uint8)]
        reader.channel_names = ["R", "G", "B"]
        reader.resolution = 1.0
        reader.is_rgb = True
        reader.image_shape = (20, 20)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=25)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetDimension() == 2
        assert wrapper.image.GetNumberOfComponentsPerPixel() == 3
        assert wrapper.image.GetSize() == (5, 5)
        assert wrapper.image.GetSpacing() == (4.0, 4.0)

    def test_get_capped_image_preserves_channel_first_axis(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((3, 20, 20), dtype=np.uint8)]
        reader.channel_names = ["C1", "C2", "C3"]
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 20)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        image, pixel_size = wrapper._get_capped_image(max_registration_pixels=25)

        assert image.shape == (3, 5, 5)
        assert pixel_size == 4.0

    def test_get_capped_image_preserves_channel_last_axis(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((20, 16, 5), dtype=np.uint8)]
        reader.channel_names = ["C1", "C2", "C3", "C4", "C5"]
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 16)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        image, pixel_size = wrapper._get_capped_image(max_registration_pixels=20)

        assert image.shape == (5, 4, 5)
        assert pixel_size == 4.0

    def test_preprocess_handles_channel_last_image(self, mock_modality):
        mock_modality.preprocessing = None
        wrapper = ImageWrapper(modality=mock_modality)
        reader = MagicMock()
        reader.pyramid = [np.ones((20, 16, 5), dtype=np.uint8)]
        reader.channel_names = ["C1", "C2", "C3", "C4", "C5"]
        reader.resolution = 1.0
        reader.is_rgb = False
        reader.image_shape = (20, 16)
        reader.scale_for_pyramid.return_value = (1.0, 1.0)
        wrapper._reader = reader

        wrapper.preprocess(max_registration_pixels=20)

        assert isinstance(wrapper.image, sitk.Image)
        assert wrapper.image.GetDimension() == 3
        assert wrapper.image.GetSize() == (4, 5, 5)
        assert wrapper.image.GetSpacing() == (4.0, 4.0, 1.0)

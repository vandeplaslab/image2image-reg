"""Pre-processing utilities."""

import numpy as np
from image2image_io.readers.utilities import grayscale
from skimage import exposure
from valis import slide_io
from valis.preprocessing import ImageProcesser, create_tissue_mask_from_multichannel, create_tissue_mask_from_rgb


class NoProcessing(ImageProcesser):
    """No processing"""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)
        return tissue_mask

    def process_image(self, *args, **kwargs):
        return self.image


class HEPreprocessing(ImageProcesser):
    """HE Pre-processing"""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)
        return tissue_mask

    def process_image(self, *args, **kwargs):
        # turn into grayscale
        image = np.asarray(grayscale(self.image, is_interleaved=True))
        # invert intensities so dark is light and light is dark
        image = 255 - image
        return image


class MaxIntensityProjection(ImageProcesser):
    """Select channel from image"""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)
        return tissue_mask

    def process_image(
        self,
        channel_names: tuple[str, ...] = (),
        channel_ids: tuple[int, ...] = (),
        adaptive_eq=True,
        *args,
        **kwaargs,
    ):
        reader_cls = slide_io.get_slide_reader(self.src_f, series=self.series)
        reader = reader_cls(self.src_f)

        if not channel_names and not channel_ids:
            channel_names = reader.metadata.channel_names
            channel_ids = list(range(len(channel_names)))
        if not channel_ids:
            channel_ids = []
            for channel_name in channel_names:
                channel_ids.append(reader.get_channel_index(channel_name))
        channel_ids = set(channel_ids)
        if not channel_ids:
            raise ValueError("No channels were specified.")

        image = []
        for channel_id in channel_ids:
            if self.image is None:
                image.append(reader.get_channel(channel=channel_id, level=self.level, series=self.series).astype(float))
            else:
                image.append(self.image[..., channel_id])
        # maximum intensity projection
        if len(image) > 1:
            image = np.dstack(image)
            image = np.max(image, axis=2)
        else:
            image = image[0]

        chnl = exposure.rescale_intensity(image, in_range="image", out_range=(0.0, 1.0))
        if adaptive_eq:
            chnl = exposure.equalize_adapthist(chnl)
        chnl = exposure.rescale_intensity(chnl, in_range="image", out_range=(0, 255)).astype(np.uint8)
        return chnl

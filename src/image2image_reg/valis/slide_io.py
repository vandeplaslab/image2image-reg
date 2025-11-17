"""Slide reader using image2image as backend."""

import os

import numpy as np
from image2image_io.config import CONFIG
from image2image_io.readers import BaseReader, get_simple_reader
from valis.slide_io import MICRON_UNIT, MetaData, SlideReader
from valis.slide_tools import numpy2vips


class Image2ImageSlideReader(SlideReader):
    """Slide reader with appropriate interface."""

    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f, *args, **kwargs)
        self.metadata = self.create_metadata()

    def create_metadata(self) -> MetaData:
        CONFIG.split_rgb = False

        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        slide_meta = MetaData(meta_name, "image2image")
        reader = get_simple_reader(self.src_f, quick=True, init_pyramid=False, auto_pyramid=False)

        slide_meta.is_rgb = reader.is_rgb
        slide_meta.channel_names = reader.channel_names
        slide_meta.n_channels = reader.n_channels
        slide_meta.pixel_physical_size_xyu = [reader.resolution, reader.resolution, MICRON_UNIT]
        slide_meta.slide_dimensions = self._get_slide_dimensions(reader)

        # f_extension = get_slide_extension(self.src_f)
        # if f_extension in BF_READABLE_FORMATS:
        #     with hide_stdout():
        #         bf_reader = BioFormatsSlideReader(self.src_f)
        #
        #     slide_meta.original_xml = bf_reader.metadata.original_xml
        #     slide_meta.bf_datatype = bf_reader.metadata.bf_datatype
        reader.close()
        return slide_meta

    def slide2vips(self, xywh=None, **kwargs):
        img = self.slide2image(xywh=xywh, **kwargs)
        return numpy2vips(img)

    def slide2image(self, xywh=None, *args, **kwargs):
        # get highest resolution image
        reader = get_simple_reader(self.src_f, quick=True, init_pyramid=False, auto_pyramid=False)
        is_rgb = reader.is_rgb
        channel_axis, _ = reader.get_channel_axis_and_n_channels()
        # retrieve highest resolution image
        img = reader.pyramid[0]
        reader.close()
        if xywh is not None:
            xywh = np.array(xywh)
            start_c, start_r = xywh[0:2]
            end_c, end_r = xywh[0:2] + xywh[2:]
            if is_rgb:
                img = img[start_r:end_r, start_c:end_c]
            else:
                if channel_axis == 0:
                    img = img[:, start_r:end_r, start_c:end_c]
                elif channel_axis == 1:
                    img = img[start_r:end_r, :, start_c:end_c]
                elif channel_axis == 2:
                    img = img[start_r:end_r, start_c:end_c, :]
        return img.compute()

    def _get_slide_dimensions(self, reader: BaseReader, *args, **kwargs):  # type: ignore
        channel_axis, _ = reader.get_channel_axis_and_n_channels()
        slide_dimensions = []
        for array in reader.get_channel_pyramid(0):
            if array.ndim == 2:
                slide_dimensions.append(array.shape)
            else:
                if channel_axis == 0:
                    slide_dimensions.append(array.shape[1::])
                else:
                    slide_dimensions.append(array.shape[0:-1])
        return np.asarray(slide_dimensions)

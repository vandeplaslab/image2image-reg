"""Pre-processing utilities."""

from __future__ import annotations

import colour
import numpy as np
from image2image_io.readers.utilities import grayscale
from scipy import ndimage, spatial
from skimage import exposure, filters, measure
from valis import slide_io
from valis.preprocessing import (
    ImageProcesser,
    combine_masks_by_hysteresis,
    create_tissue_mask_from_multichannel,
    create_tissue_mask_from_rgb,
    mask2contours,
    mask2covexhull,
    rgb2jch,
)

from image2image_reg.enums import ImageType


def clean_mask(mask, img, rel_min_size=0.001):
    """Remove small objects, regions that are not very colorful (relativey), and retangularly shaped objects."""
    fg_labeled = measure.label(mask)
    fg_regions = measure.regionprops(fg_labeled)

    if len(fg_regions) == 1:
        return mask

    jch = rgb2jch(img, cspace="JzAzBz")
    c = exposure.rescale_intensity(jch[..., 1], out_range=(0, 1))
    colorfulness_img = np.zeros(mask.shape)

    for _i, r in enumerate(fg_regions):
        # Fill in contours that are touching border
        r0, c0, r1, c1 = r.bbox
        r_filled_img = r.image_filled.copy()
        if r0 == 0:
            # Touching top
            lr = np.where(r.image_filled[0, :])[0]
            r_filled_img[0, min(lr) : max(lr)] = 255

        if r1 == mask.shape[0]:
            # Touching bottom
            lr = np.where(r.image_filled[-1, :])[0]
            r_filled_img[-1, min(lr) : max(lr)] = 255

        if c0 == 0:
            tb = np.where(r.image_filled[:, 0])[0]
            # Touchng left border
            r_filled_img[min(tb) : max(tb), 0] = 255

        if c1 == mask.shape[1]:
            # Touchng right border
            tb = np.where(r.image_filled[:, -1])[0]
            r_filled_img[min(tb) : max(tb), -1] = 255

        r_filled_img = ndimage.binary_fill_holes(r_filled_img)
        colorfulness_img[r.slice][r_filled_img] = np.max(c[r.slice][r_filled_img])

    color_thresh = filters.threshold_otsu(colorfulness_img[mask > 0])
    color_mask = colorfulness_img > color_thresh
    mask_list = [mask.astype(bool), color_mask]

    feature_mask = combine_masks_by_hysteresis(mask_list)
    if feature_mask.max() == 0:
        feature_mask = np.sum(np.dstack(mask_list), axis=2)
        feature_thresh = len(mask_list) // 2
        feature_mask[feature_mask <= feature_thresh] = 0
        feature_mask[feature_mask != 0] = 255

    features_labeled = measure.label(feature_mask)
    feature_regions = measure.regionprops(features_labeled)

    if len(feature_regions) == 1:
        return feature_mask

    region_sizes = np.array([r.area for r in feature_regions])
    min_abs_size = int(rel_min_size * np.multiply(*mask.shape[0:2]))  # *kernel_size
    keep_region_idx = np.where(region_sizes > min_abs_size)[0]
    if len(keep_region_idx) == 0:
        biggest_idx = np.argmax([r.area for r in fg_regions])
        keep_region_idx = [biggest_idx]

    # Get final regions
    fg_mask = np.zeros(mask.shape[0:2], np.uint8)
    for _i, rid in enumerate(keep_region_idx):
        r = feature_regions[rid]
        fg_mask[r.slice][r.image_filled] = 255

    return fg_mask


def jc_dist(img, cspace="IHLS", p=99, metric="euclidean"):
    """
    Cacluate distance between backround and each pixel
    using a luminosity and colofulness/saturation in a polar colorspace.

    Parameters
    ----------
    img : np.ndarray
        RGB image

    cspace: str
        Name of colorspace to use for calculation

    p: int
        Percentile used to determine background values, i.e.
        background pixels have a luminosity greather 99% of other
        pixels. Needs to be between 0-100

    metric: str
        Name of distance metric. Passed to `scipy.spatial.distance.cdist`

    Returns
    -------
    jc_dist : np.ndarray
        Color distance between backround and each pixel

    """
    if cspace.upper() == "IHLS":
        hys = colour.models.RGB_to_IHLS(img)  # Hue, luminance, saturation/colorfulness
        j = hys[..., 1]
        c = hys[..., 2]

    else:
        jch = rgb2jch(img, cspace=cspace)
        j = jch[..., 0]
        c = jch[..., 1]

    j01 = exposure.rescale_intensity(j, out_range=(0, 1))
    c01 = exposure.rescale_intensity(c, out_range=(0, 1))
    jc01 = np.dstack([j01, c01]).reshape((-1, 2))

    bg_j = np.percentile(j01, p)
    bg_c = np.percentile(c01, 100 - p)

    jc_dist_img = spatial.distance.cdist(jc01, np.array([[bg_j, bg_c]]), metric=metric).reshape(img.shape[0:2])

    return jc_dist_img


def create_tissue_mask_with_jc_dist(img):
    """
    Create tissue mask using JC distance from background.

    Parameters
    ----------
    img : np.ndarray
        RGB image

    Returns
    -------
    mask : np.ndarray
        Mask covering tissue

    chull_mask : np.ndarray
        Mask created by drawing a convex hull around each region in
        `mask`

    """
    assert img.ndim == 3, "`img` needs to be RGB image"
    jc_dist_img = jc_dist(img, metric="chebyshev")
    jc_dist_img[np.isnan(jc_dist_img)] = np.nanmax(jc_dist_img)

    jc_t, _ = filters.threshold_multiotsu(jc_dist_img)
    jc_mask = 255 * (jc_dist_img > jc_t).astype(np.uint8)
    jc_dist_img = exposure.equalize_adapthist(exposure.rescale_intensity(jc_dist_img, out_range=(0, 1)))

    img_edges = filters.scharr(jc_dist_img)
    p_t = filters.threshold_otsu(img_edges)
    edges_mask = 255 * (img_edges > p_t).astype(np.uint8)

    temp_mask = edges_mask.copy()
    temp_mask[jc_mask == 0] = 0
    temp_mask = mask2contours(temp_mask, 3)

    mask = clean_mask(mask=temp_mask, img=img)
    chull_mask = mask2covexhull(mask)

    return mask, chull_mask


class NoProcessing(ImageProcesser):
    """No processing."""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)
        return tissue_mask

    def process_image(self, *args, **kwargs):
        return self.image


class HEPreprocessing(ImageProcesser):
    """HE Pre-processing."""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self) -> np.ndarray:
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)
        return tissue_mask

    def process_image(self, *args, **kwargs):
        # turn into grayscale
        image = np.asarray(grayscale(self.image, is_interleaved=True))
        # mask out background
        mask = image < 1
        # invert intensities so dark is light and light is dark
        image = 255 - image
        # apply mask
        image[mask] = 0
        return image


class MaxIntensityProjection(ImageProcesser):
    """Select channel from image."""

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
        **kwargs,
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


class I2RegPreprocessor(ImageProcesser):
    """Select channel from image."""

    def create_mask(self) -> np.ndarray:
        """Create tissue mask."""
        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)
        return tissue_mask

    def process_image(
        self,
        image_type: ImageType = ImageType.DARK,
        max_intensity_projection: bool = True,
        equalize_histogram: bool = False,
        contrast_enhance: bool = False,
        invert_intensity: bool = False,
        channel_indices: list[int] | None = None,
        channel_names: list[str] | None = None,
        as_uint8: bool = True,
        *args,
        **kwargs,
    ):
        from image2image_io.readers import get_simple_reader

        from image2image_reg.models import Preprocessing
        from image2image_reg.utils.preprocessing import preprocess_preview

        pre = Preprocessing(
            image_type=image_type,
            max_intensity_projection=max_intensity_projection,
            equalize_histogram=equalize_histogram,
            contrast_enhance=contrast_enhance,
            invert_intensity=invert_intensity,
            channel_indices=channel_indices,
            channel_names=channel_names,
            as_uint8=as_uint8,
        )
        is_rgb = get_simple_reader(self.src_f, init_pyramid=False, auto_pyramid=False).is_rgb
        return preprocess_preview(self.image, is_rgb, 1.0, preprocessing=pre, spatial=False, valis=True)


class OD(ImageProcesser):
    """Convert the image from RGB to optical density (OD) and calculate pixel norms."""

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level, series=series, *args, **kwargs)

    def create_mask(self):
        _, mask = create_tissue_mask_with_jc_dist(self.image)

        return mask

    def process_image(self, adaptive_eq=False, p=95, *args, **kwargs):
        """Calculate norm of the OD image."""
        eps = np.finfo("float").eps
        img01 = self.image / 255
        od = -np.log10(img01 + eps)
        od_norm = np.mean(od, axis=2)
        upper_p = np.percentile(od_norm, p)
        lower_p = 0
        od_clipped = np.clip(od_norm, lower_p, upper_p)

        if adaptive_eq:
            od_clipped = exposure.equalize_adapthist(exposure.rescale_intensity(od_clipped, out_range=(0, 1)))
        processed = exposure.rescale_intensity(od_clipped, out_range=np.uint8)
        return processed

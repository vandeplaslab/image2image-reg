"""Visuals."""

from __future__ import annotations

import colour
import numba as nb
import numpy as np
from image2image_io.utils.utilities import get_shape_of_image, guess_rgb
from scipy.spatial import distance
from skimage import exposure
from skimage.color import lab2rgb, rgb2gray, rgb2lab


@nb.njit(fastmath=True, cache=True)
def blend_colors(img: np.ndarray, colors: np.ndarray, scale_by: str):
    """Color an image by blending.

    Parameters
    ----------
    img : ndarray
        Image containing the raw data (float 32)

    colors : ndarray
        Colors for each channel in `img`

    scale_by : int
        How to scale each channel. "image" will scale the channel
        by the maximum value in the image. "channel" will scale
        the channel by the maximum in the channel

    Returns
    -------
    blended_img : ndarray
        A colored version of `img`

    """
    if len(colors) > 1:
        n_channel_colors = colors.shape[1]
    else:
        n_channel_colors = len(colors)

    if img.ndim > 2:
        r, c, nc = img.shape[:3]
    else:
        nc = 1
        r, c = img.shape[2]

    eps = 1.0000000000000001e-15
    sum_img = img.sum(axis=2) + eps
    if scale_by == "image":
        img_max = img.max()

    blended_img = np.zeros((r, c, n_channel_colors))
    for i in range(nc):
        # relative image is how bright the channel will be
        if scale_by != "image":
            channel_max = img[..., i].max()
            relative_img = img[..., i] / channel_max
        else:
            relative_img = img[..., i] / img_max

        # blending is how to weight the mix of colors, similar to an alpha channel
        blending = img[..., i] / sum_img
        for j in range(colors.shape[1]):
            channel_color = colors[i, j]
            blended_img[..., j] += channel_color * relative_img * blending

    return blended_img


def jzazbz_cmap(luminosity: float = 0.012, colorfulness: float = 0.02, max_h: float = 260) -> np.ndarray:
    """
    Get colormap based on JzAzBz colorspace, which has good hue linearity.
    Already preceptually uniform.

    Parameters
    ----------
    luminosity :  float, optional

    colorfulness : float, optional

    max_h : int, optional

    """
    h = np.deg2rad(np.arange(0, 360))
    a = colorfulness * np.cos(h)
    b = colorfulness * np.sin(h)
    j = np.repeat(luminosity, len(h))

    jzazbz = np.dstack([j, a, b])
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jzazbz, "JzAzBz", "sRGB")

    rgb = np.clip(rgb, 0, 1)[0]
    if max_h != 360:
        rgb = rgb[0:max_h]
    return rgb


def color_multichannel(
    multichannel_img, marker_colors, rescale_channels=False, normalize_by="image", cspace="Hunter Lab"
):
    """Color a multichannel image to view as RGB.

    Parameters
    ----------
    multichannel_img : ndarray
        Image to color

    marker_colors : ndarray
        sRGB colors for each channel.

    rescale_channels : bool
        If True, then each channel will be scaled between 0 and 1 before
        building the composite RGB image. This will allow markers to 'pop'
        in areas where they are expressed in isolation, but can also make
        it appear more marker is expressed than there really is.

    normalize_by : str, optionaal
        "image" will produce an image where all values are scaled between
        0 and the highest intensity in the composite image. This will produce
        an image where one can see the expression of each marker relative to
        the others, making it easier to compare marker expression levels.

        "channel" will first scale the intensity of each channel, and then
        blend all of the channels together. This will allow one to see the
        relative expression of each marker, but won't allow one to directly
        compare the expression of markers across channels.

    cspace : str
        Colorspace in which `marker_colors` will be blended.
        JzAzBz, Hunter Lab, and sRGB all work well. But, see
        colour.COLOURSPACE_MODELS for other possible colorspaces

    Returns
    -------
    rgb : ndarray
        An sRGB version of `multichannel_img`

    """
    if rescale_channels:
        multichannel_img = np.dstack(
            [
                exposure.rescale_intensity(multichannel_img[..., i].astype(float), in_range="image", out_range=(0, 1))
                for i in range(multichannel_img.shape[2])
            ]
        )

    is_srgb = cspace.lower() == "srgb"
    is_srgb_01 = True
    if 1 < marker_colors.max() <= 255 and np.issubdtype(marker_colors.dtype, np.integer):
        srgb_01 = marker_colors / 255
        is_srgb_01 = False

    else:
        srgb_01 = marker_colors
    eps = np.finfo("float").eps
    if not is_srgb:
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            cspace_colors = colour.convert(srgb_01 + eps, "sRGB", cspace)
    else:
        cspace_colors = srgb_01

    blended_img = blend_colors(multichannel_img, cspace_colors, normalize_by)
    if not is_srgb:
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            srgb_blended = colour.convert(blended_img + eps, cspace, "sRGB") - 2 * eps
    else:
        srgb_blended = blended_img

    srgb_blended = np.clip(srgb_blended, 0, 1)
    if not is_srgb_01:
        srgb_blended = (255 * srgb_blended).astype(marker_colors.dtype)
    return srgb_blended


def get_n_colors(rgb: np.ndarray, n: int) -> np.ndarray:
    """
    Pick n most different colors in rgb. Differences based of rgb values in the CAM16UCS colorspace
    Based on https://larssonjohan.com/post/2016-10-30-farthest-points/.
    """
    n_clrs = rgb.shape[0]
    if n_clrs < n:
        n_full_rep = n // n_clrs
        n_extra = n % n_clrs

        all_colors = np.vstack([*[rgb] * n_full_rep, rgb[0:n_extra]])
        assert all_colors.shape[0] == n

        np.random.shuffle(all_colors)

        return all_colors

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < rgb.max() <= 255 and np.issubdtype(rgb.dtype, np.integer):
            cam = colour.convert(rgb / 255, "sRGB", "CAM16UCS")
        else:
            cam = colour.convert(rgb, "sRGB", "CAM16UCS")

    sq_D = distance.cdist(cam, cam)
    max_D = sq_D.max()
    most_dif_2Didx = np.where(sq_D == max_D)  # 2 most different colors
    most_dif_img1 = most_dif_2Didx[0][0]
    most_dif_img2 = most_dif_2Didx[1][0]
    rgb_idx = [most_dif_img1, most_dif_img2]

    possible_idx = list(range(sq_D.shape[0]))
    possible_idx.remove(most_dif_img1)
    possible_idx.remove(most_dif_img2)

    for _new_color in range(2, n):
        max_d_idx = np.argmax([np.min(sq_D[i, rgb_idx]) for i in possible_idx])
        new_rgb_idx = possible_idx[max_d_idx]
        rgb_idx.append(new_rgb_idx)
        possible_idx.remove(new_rgb_idx)
    return rgb[rgb_idx]


def get_shape(img: np.ndarray) -> np.ndarray:
    """Get shape of image (row, col, nchannels).

    Parameters
    ----------
    img : numpy.array, pyvips.Image
        Image to get shape of

    Returns
    -------
    shape_rc : numpy.array
        Number of rows and columns and channels in the image

    """
    shape_rc = np.array(img.shape[0:2])
    ndim = img.shape[2] if img.ndim > 2 else 1
    shape = np.array([*shape_rc, ndim])
    return shape


def prepare_images_for_overlap(images: list[np.ndarray]) -> list[np.ndarray]:
    """Prepare images for overlap."""
    from koyo.image import clip_hotspots

    grey_images = []
    for img in images:
        _, channel_axis, shape = get_shape_of_image(img)
        is_rgb = guess_rgb(img.shape)
        if is_rgb:
            grey_images.append(rgb2gray(img))
        elif channel_axis is None:
            grey_images.append(img)
        elif channel_axis == 2:
            grey_images.append(img[:, :, 0])
        elif channel_axis == 0:
            grey_images.append(img[0, :, :])
        else:
            grey_images.append(np.max(img, axis=channel_axis))

    # normalize the images
    for i in range(len(grey_images)):
        grey_image = clip_hotspots(grey_images[i])
        grey_images[i] = exposure.rescale_intensity(grey_image, out_range=(0, 1)) * 255
    return grey_images


def create_overlap_img(images: list[np.ndarray], cmap: np.ndarray | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Parameters
    ----------
    images : list
        list of single channel images to create overlap from
    cmap : ndarray, optional
        colormap to use for coloring the images

    Returns
    -------
    blended : ndarray
        Overlap of images in `img_list` colored by `cmap`
    """
    if cmap is None:
        cmap = jzazbz_cmap()

    n_imgs = len(images)
    color_list = get_n_colors(cmap, n_imgs)
    color_list = [rgb2lab(np.array([[clr]])) * 255 for clr in color_list]

    grey_images = prepare_images_for_overlap(images)

    eps = np.finfo("float").eps
    sum_img = np.full(get_shape(grey_images[0])[0:2], eps)
    blended_img = np.zeros((*sum_img.shape, 3))

    max_v = 0
    for i in range(len(grey_images)):
        sum_img += grey_images[i]
        max_v = max(max_v, grey_images[i].max())

    for i in range(len(grey_images)):
        weight = grey_images[i] / sum_img
        lab_clr = color_list[i]
        blended_img += lab_clr * np.dstack([(grey_images[i] / 255) / max_v * weight] * 3)

    blended_img = lab2rgb(blended_img)
    overlap_img = np.asarray(blended_img)
    overlap_img = exposure.rescale_intensity(overlap_img, out_range=(0, 255)).astype(np.uint8)
    return overlap_img, grey_images


def crop_overlap_img(image: np.ndarray, frac: float = 0.25) -> np.ndarray:
    """Crop overlap image."""
    # find centroid point in the image and add padding around the image, ensuring  that it captures ~128 pixels
    # but does not go outside of the image
    r, c, _ = image.shape
    size = int(min(r, c) * frac)
    r_center, c_center = r // 2, c // 2
    r_start = max(0, r_center - size // 2)
    r_end = min(r, r_center + size // 2)
    c_start = max(0, c_center - size // 2)
    c_end = min(c, c_center + size // 2)
    return image[r_start:r_end, c_start:c_end, :]

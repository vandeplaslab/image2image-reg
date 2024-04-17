"""Visuals."""

import colour
import numba as nb
import numpy as np
from scipy.spatial import distance
from skimage import exposure


def get_n_colors(rgb, n):
    """
    Pick n most different colors in rgb. Differences based of rgb values in the CAM16UCS colorspace
    Based on https://larssonjohan.com/post/2016-10-30-farthest-points/.
    """
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


def jzazbz_cmap(luminosity=0.012, colorfulness=0.02, max_h=260):
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

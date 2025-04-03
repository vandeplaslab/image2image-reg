"""Utility functions for preprocessing images."""

from __future__ import annotations

import typing as ty

import colour
import numpy as np
import SimpleITK as sitk
from skimage import exposure, measure


def get_luminosity(img: np.ndarray) -> np.ndarray:
    """Get luminosity of an RGB image.

    Converts and RGB image to the CAM16-UCS colorspace, extracts the luminosity, and then scales it between 0-255

    Parameters
    ----------
    img : ndarray
        RGB image

    Returns
    -------
    lum : ndarray
        CAM16-UCS luminosity

    """
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img / 255, "sRGB", "CAM16UCS")
        else:
            cam = colour.convert(img, "sRGB", "CAM16UCS")
    return exposure.rescale_intensity(cam[..., 0], in_range=(0, 1), out_range=(0, 255))


def normalize_he(img: np.ndarray, intensity: int = 240, alpha: int = 1, beta: float = 0.15) -> np.ndarray:
    """Normalize staining appearence of H&E stained images.

    Parameters
    ----------
    img : ndarray
        2D RGB image to be transformed, np.array<height, width, ch>.
    intensity : int, optional
        The transmitted light intensity. The default value is ``240``.
    alpha : int, optional
        This value is used to get the alpha(th) and (100-alpha)(th) percentile
        as robust approximations of the intensity histogram min and max values.
        The default value, found empirically, is ``1``.
    beta : float, optional
        Threshold value used to remove the pixels with a low OD for stability reasons.
        The default value, found empirically, is ``0.15``.

    Returns
    -------
    normalized_stains_conc : ndarray
        The normalized stains vector, np.array<2, im_height*im_width>.

    """
    max_conc_ref = np.array([1.9705, 1.0308])

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    opt_density = -np.log((img.astype(np.float32) + 1) / intensity)

    # remove transparent pixels
    opt_density_hat = opt_density[~np.any(opt_density < beta, axis=1)]

    # compute eigenvectors
    _, eigvecs = np.linalg.eigh(np.cov(opt_density_hat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    t_hat = opt_density_hat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)

    v_min = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if v_min[0] > v_max[0]:
        h_e_vector = np.array((v_min[:, 0], v_max[:, 0])).T
    else:
        h_e_vector = np.array((v_max[:, 0], v_min[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    y = np.reshape(opt_density, (-1, 3)).T

    # determine concentrations of the individual stains
    stains_conc = np.linalg.lstsq(h_e_vector, y, rcond=None)[0]

    # normalize stains concentrations
    max_conc = np.array([np.percentile(stains_conc[0, :], 99), np.percentile(stains_conc[1, :], 99)])
    tmp = np.divide(max_conc, max_conc_ref)
    return np.divide(stains_conc, tmp[:, np.newaxis])


def deconvolution_he(
    img: np.ndarray, concentrations: np.ndarray, stain: ty.Literal["hem", "eos"] = "hem", intensity: int = 240
) -> np.ndarray:
    """Unmix the hematoxylin or eosin channel based on their respective normalized concentrations.

    Parameters
    ----------
    img : ndarray
        2D RGB image to be transformed, np.array<height, width, ch>.
    concentrations : ndarray
        The normalized stains vector, np.array<2, im_height*im_width>.
    stain : str
        Either ``hem`` for the hematoxylin stain or ``eos`` for the eosin one.
    intensity : int, optional
        The transmitted light intensity. The default value is ``240``.

    Returns
    -------
    out : ndarray
        2D image with a single channel corresponding to the desired stain, np.array<height, width>.

    """
    # define height and width of image
    h, w, _ = img.shape

    # unmix hematoxylin or eosin
    if stain == "hem":
        out = np.multiply(intensity, concentrations[0, :])
    elif stain == "eos":
        out = np.multiply(intensity, concentrations[1, :])
    else:
        raise ValueError(f"Stain '{stain}' is unknown.")

    np.clip(out, a_min=0, a_max=255, out=out)
    out = np.reshape(out, (h, w)).astype(np.uint8)
    return out


def calc_background_color_dist(
    img: np.ndarray, brightness_q: float = 0.99, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Create mask that only covers tissue.

    #. Find background pixel (most luminescent)
    #. Convert image to CAM16-UCS
    #. Calculate distance between each pixel and background pixel
    #. Threshold on distance (i.e. higher distance = different color)

    Returns
    -------
    cam_d : float
        Distance from background color
    cam : float
        CAM16UCS image
    """
    img = exposure.rescale_intensity(img, in_range="image", out_range=(0, 1))
    # eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        cam = colour.convert(img, "sRGB", "CAM16UCS")

    if mask is None:
        brightest_thresh = np.quantile(cam[..., 0], brightness_q)
    else:
        brightest_thresh = np.quantile(cam[..., 0][mask > 0], brightness_q)

    brightest_idx = np.where(cam[..., 0] >= brightest_thresh)
    brightest_pixels = cam[brightest_idx]
    bright_cam = brightest_pixels.mean(axis=0)
    cam_d = np.sqrt(np.sum((cam - bright_cam) ** 2, axis=2))

    return cam_d, cam


def rgb2jab(rgb: np.ndarray, cspace: str = "CAM16UCS") -> np.ndarray:
    eps = np.finfo("float").eps
    if np.issubdtype(rgb.dtype, np.integer) and rgb.max() > 1:
        rgb01 = rgb / 255.0
    else:
        rgb01 = rgb

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        jab = colour.convert(rgb01 + eps, "sRGB", cspace)
    return jab


def jab2rgb(jab: np.ndarray, cspace: str = "CAM16UCS") -> np.ndarray:
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jab + eps, cspace, "sRGB")
    return rgb


def estimate_k(x, max_k: int = 100, step_size: int = 10):
    if max_k <= 10:
        step_size = 1

    # Create initial cluster list
    potential_c = np.arange(0, max_k, step=step_size)
    # potential_c = np.linspace(2, max_k, n_steps).astype(int)
    if potential_c[-1] != max_k:
        potential_c = np.hstack([potential_c, max_k])
    potential_c[0] = 2
    potential_c = np.unique(potential_c[potential_c > 1])

    almost_done = False
    done = False
    best_k = 2
    best_clst = None
    k_step = step_size
    while not done:
        inertia_list = []
        nc = []
        clst_list = []

        for i in potential_c:
            try:
                clusterer = cluster.MiniBatchKMeans(n_clusters=i, n_init=3)
                clusterer.fit(x)

            except Exception:
                continue
            inertia_list.append(clusterer.inertia_)
            nc.append(i)
            clst_list.append(clusterer)

        inertia_list = np.array(inertia_list)

        dy = np.diff(inertia_list)
        intertia_t = thresh_unimodal(dy, int(np.max(potential_c)))
        best_k_idx = np.where(dy >= intertia_t)[0][0] + 1
        best_k = potential_c[best_k_idx]
        best_clst = clst_list[best_k_idx]
        if almost_done:
            done = True
            break

        next_k_range = np.clip([best_k - k_step // 2, best_k + k_step // 2], 2, max_k)
        kd = np.diff(next_k_range)[0]
        if kd == 0:
            break
        if kd <= 10:
            k_step = 1
            almost_done = True
        else:
            k_step = step_size
        potential_c = np.arange(next_k_range[0], next_k_range[1], k_step)
    return best_k, best_clst


def thresh_unimodal(x: np.ndarray, bins: int = 256) -> int:
    """
    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf.

    To threshold
    :param px_vals:
    :param bins:
    :return:
    """
    from scipy import stats
    from shapely import LineString

    # Threshold unimodal distribution
    skew = stats.skew(x)
    # Find line from peak to tail
    if skew >= 0:
        counts, bin_edges = np.histogram(x, bins=bins)
    else:
        # Tail is to the left, so reverse values to use this method, which assumes tail is on the right
        counts, bin_edges = np.histogram(-x, bins=bins)

    bin_width = bin_edges[1] - bin_edges[0]
    midpoints = bin_edges[0:-1] + bin_width / 2
    hist_line = LineString(np.column_stack([midpoints, counts]))

    peak_bin = np.argmax(counts)
    last_non_zero = np.where(counts > 0)[0][-1]
    if last_non_zero == len(counts) - 1:
        min_bin = last_non_zero
    else:
        min_bin = last_non_zero + 1

    peak_x, min_bin_x = midpoints[peak_bin], midpoints[min_bin]
    peak_y, min_bin_y = counts[peak_bin], counts[min_bin]

    peak_m = (peak_y - min_bin_y) / (peak_x - min_bin_x + np.finfo(float).resolution)
    peak_b = peak_y - peak_m * peak_x
    perp_m = -peak_m + np.finfo(float).resolution
    n_v = len(midpoints)
    d = [-1] * n_v
    all_xi = [-1] * n_v

    for i in range(n_v):
        x1 = midpoints[i]
        if x1 < peak_x:
            continue
        y1 = peak_m * x1 + peak_b
        perp_b = y1 - perp_m * x1
        y2 = 0
        x2 = -perp_b / (perp_m)

        perp_line_obj = LineString([[x1, y1], [x2, y2]])
        if not perp_line_obj.is_valid or not hist_line.is_valid:
            print("perpline is valid", perp_line_obj.is_valid, "hist line is valid", hist_line.is_valid)
            print("perpline xy1, xy2", [x1, y1], [x2, y2], "m=", perp_m)

        intersection = perp_line_obj.intersection(hist_line)
        if intersection.is_empty:
            # No intersection
            continue
        if intersection.geom_type == "MultiPoint":
            all_x, all_y = LineString(intersection.geoms).xy
            xi = all_x[-1]
            yi = all_y[-1]
        elif intersection.geom_type == "Point":
            xi, yi = intersection.xy
            xi = xi[0]
            yi = yi[0]
        d[i] = np.sqrt((xi - x1) ** 2 + (yi - y1) ** 2)
        all_xi[i] = xi

    max_d_idx = np.argmax(d)
    t = all_xi[max_d_idx]

    if skew < 0:
        t *= -1

    return t


def rgb255_to_rgb1(rgb_img: np.ndarray) -> np.ndarray:
    if np.issubdtype(rgb_img.dtype, np.integer) or rgb_img.max() > 1:
        rgb01 = rgb_img / 255.0
    else:
        rgb01 = rgb_img
    return rgb01


def rgb2od(rgb_img: np.ndarray) -> np.ndarray:
    eps = np.finfo("float").eps
    rgb01 = rgb255_to_rgb1(rgb_img)
    od = -np.log10(rgb01 + eps)
    od[od < 0] = 0
    return od


def stainmat2decon(stain_mat_srgb255: np.ndarray) -> np.ndarray:
    od_mat = rgb2od(stain_mat_srgb255)

    eps = np.finfo("float").eps
    M = od_mat / np.linalg.norm(od_mat + eps, axis=1, keepdims=True)
    M[np.isnan(M)] = 0
    D = np.linalg.pinv(M)
    return D


def deconvolve_img(rgb_img: np.ndarray, D) -> np.ndarray:
    od_img = rgb2od(rgb_img)
    deconvolved_img = np.dot(od_img, D)
    deconvolved_img[deconvolved_img < 0] = 0
    return deconvolved_img


def standardize_colorfulness(img, c: float = 0.2, h: int = 0):
    """Give image constant colorfulness and hue.

    Image is converted to cylindrical CAM-16UCS assigned a constant
    hue and colorfulness, and then coverted back to RGB.

    Parameters
    ----------
    img : ndarray
        Image to be processed
    c : float
        Colorfulness
    h : float
        Hue, in radians (-pi to pi)

    Returns
    -------
    rgb2 : ndarray
        `img` with constant hue and colorfulness

    """
    # Convert to CAM16 #
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img / 255 + eps, "sRGB", "CAM16UCS")
        else:
            cam = colour.convert(img + eps, "sRGB", "CAM16UCS")

    lum = cam[..., 0]
    cc = np.full_like(lum, c)
    hc = np.full_like(lum, h)
    new_a, new_b = cc * np.cos(hc), cc * np.sin(hc)
    new_cam = np.dstack([lum, new_a + eps, new_b + eps])
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb2 = colour.convert(new_cam, "CAM16UCS", "sRGB")
        rgb2 -= eps

    rgb2 = (np.clip(rgb2, 0, 1) * 255).astype(np.uint8)
    return rgb2


def create_edges_mask(labeled_img):
    """Create two masks, one with objects not touching image bordersa second with objects that do touch the border."""
    unique_v = np.unique(labeled_img)
    unique_v = unique_v[unique_v != 0]
    if len(unique_v) == 1:
        labeled_img = measure.label(labeled_img)

    img_regions = measure.regionprops(labeled_img)
    inner_mask = np.zeros(labeled_img.shape, dtype=np.uint8)
    edges_mask = np.zeros(labeled_img.shape, dtype=np.uint8)
    for regn in img_regions:
        on_border_idx = np.where(
            (regn.coords[:, 0] == 0)
            | (regn.coords[:, 0] == labeled_img.shape[0] - 1)
            | (regn.coords[:, 1] == 0)
            | (regn.coords[:, 1] == labeled_img.shape[1] - 1)
        )[0]
        if len(on_border_idx) == 0:
            inner_mask[regn.coords[:, 0], regn.coords[:, 1]] = 255
        else:
            edges_mask[regn.coords[:, 0], regn.coords[:, 1]] = 255
    return inner_mask, edges_mask


def rescale_to_uint8(array: sitk.Image, min_value: int = 0, max_value: int = 255):
    """Rescale image."""
    array = sitk.Cast(array, sitk.sitkFloat32)
    array = sitk.RescaleIntensity(array, min_value, max_value)
    array = sitk.Cast(array, sitk.sitkUInt8)
    return array

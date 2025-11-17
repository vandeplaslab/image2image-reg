"""Pre-processing step."""

from __future__ import annotations

import typing as ty
import warnings

import cv2
import numpy as np
import SimpleITK as sitk
from skimage import exposure

from image2image_reg.preprocessing.convert import numpy_to_sitk_image, sitk_image_to_numpy
from image2image_reg.preprocessing.mixin import PreprocessorMixin
from image2image_reg.preprocessing.utilities import (
    calc_background_color_dist,
    deconvolution_he,
    deconvolve_img,
    estimate_k,
    get_luminosity,
    jab2rgb,
    normalize_he,
    rescale_to_uint8,
    rgb2jab,
    stainmat2decon,
    standardize_colorfulness,
)

if ty.TYPE_CHECKING:
    from sklearn.cluster import MiniBatchKMeans


PREPROCESSOR_REGISTER: dict[str, Preprocessor] = {}


def register(name: str, **kwargs: ty.Any) -> ty.Callable[[type[Preprocessor]], type[Preprocessor]]:
    """Decorate a class to register a name for it, optionally with a set of associated initialization parameters.

    Parameters
    ----------
    name : str
        The name to register the filter under.
    **kwargs
        Keyword arguments forwarded to the decorated class's initialization method

    Returns
    -------
    function
        A decorating function which will carry out the registration
        process on the decorated class.
    """

    def _wrap(cls: type[Preprocessor]) -> type[Preprocessor]:
        PREPROCESSOR_REGISTER[name] = cls(**kwargs)
        return cls

    return _wrap


def get_preprocessor(name: str) -> Preprocessor:
    """Get preprocessor class."""
    if name not in PREPROCESSOR_REGISTER:
        raise ValueError(f"Invalid pre-processing step: '{name}'")
    return PREPROCESSOR_REGISTER[name]


class Preprocessor(PreprocessorMixin):
    """Pre-processing step."""

    allow_rgb: bool
    allow_single_channel: bool
    allow_multi_channel: bool

    array: np.ndarray | sitk.Image
    pixel_size: float

    def __init__(self, **kwargs: ty.Any):
        self.kws = kwargs

    def __call__(self, array: np.ndarray | sitk.Image, pixel_size: float, **kwargs: ty.Any) -> sitk.Image:
        """Apply step to image."""
        if not kwargs:
            kwargs = self.kws
        # temporarily set the array
        self.array = array
        self.pixel_size = pixel_size
        # apply the step
        self.array = self._apply(**kwargs)
        return self.array

    def apply(
        self, array: np.ndarray | sitk.Image, pixel_size: float, to_array: bool = False, **kwargs: ty.Any
    ) -> sitk.Image | np.ndarray:
        """Apply step to image."""
        res = self(array, pixel_size, **kwargs)
        if to_array:
            return sitk_image_to_numpy(res)
        return res

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        raise NotImplementedError


@register("color_standardizer")
class ColorStandardizerPreprocessor(Preprocessor):
    """Color standardization."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    def _apply(
        self, c: float = 0.2, invert: bool = True, adaptive_equalize: bool = False, **kwargs: ty.Any
    ) -> sitk.Image:
        if self.is_rgb:
            array = self.to_array()
            std_rgb = standardize_colorfulness(array, c)
            std_g = GrayPreprocessor().apply(std_rgb, self.pixel_size, to_array=True)
            if invert:
                std_g = 255 - std_g

            if adaptive_equalize:
                std_g = exposure.equalize_adapthist(std_g / 255)

            array = exposure.rescale_intensity(std_g, in_range="image", out_range=(0, 255)).astype(np.uint8)
            array = numpy_to_sitk_image(array)  # type: ignore[assignment]
            array.SetSpacing(self.spacing)  # type: ignore[attr-defined]
            return array  # type: ignore[return-value]
        return self.to_sitk()


@register("luminosity")
class LuminosityPreprocessor(Preprocessor):
    """Luminosity."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        if self.is_rgb:
            image = self.to_array()
            inv_lum = 255 - get_luminosity(image)
            image = exposure.rescale_intensity(inv_lum, out_range=(0, 255)).astype(np.uint8)
            array = numpy_to_sitk_image(image)
            array.SetSpacing(self.spacing)  # type: ignore[no-untyped-call]
        else:
            array = self.to_sitk()
            warnings.warn("Cannot compute luminosity for non-RGB image.", stacklevel=2)
        return array


@register("background_color_distance")
class BackgroundColorDistancePreprocessor(Preprocessor):
    """Background color distance."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    def _apply(self, brightness_q: float = 0.99, **kwargs: ty.Any) -> sitk.Image:
        if self.is_rgb:
            array = self.to_array()
            array, _ = calc_background_color_dist(array, brightness_q=brightness_q)
            array = exposure.rescale_intensity(array, in_range="image", out_range=(0, 1))
            array = exposure.equalize_adapthist(array)
            array = exposure.rescale_intensity(array, in_range="image", out_range=(0, 255)).astype(np.uint8)
            array = numpy_to_sitk_image(array)  # type: ignore[assignment]
            array.SetSpacing(self.spacing)  # type: ignore[attr-defined]
        else:
            array = self.to_sitk()  # type: ignore[assignment]
            warnings.warn("Cannot compute background color distance for non-RGB image.", stacklevel=2)
        return array  # type: ignore[return-value]


@register("stain_flattener")
class StainFlattenerPreprocessor(Preprocessor):
    """Stain flattening."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    model: MiniBatchKMeans

    def _apply(
        self, n_colors: int = 100, q: int = 95, adaptive_equalize: float = True, max_colors: int = 100, **kwargs: ty.Any
    ) -> sitk.Image:
        if self.is_rgb:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.preprocessing import StandardScaler

            img_to_cluster = rgb2jab(self.to_array())

            ss = StandardScaler()
            x = ss.fit_transform(img_to_cluster.reshape(-1, img_to_cluster.shape[2]))
            if n_colors > 0:
                self.n_colors = n_colors
                model = MiniBatchKMeans(n_clusters=n_colors, reassignment_ratio=0, n_init=3)
                model.fit(x)
            else:
                k, model = estimate_k(x, max_k=max_colors)
                self.n_colors = k

            self.model = model
            stain_rgb = jab2rgb(ss.inverse_transform(model.cluster_centers_))
            stain_rgb = np.clip(stain_rgb, 0, 1)

            stain_rgb = 255 * stain_rgb
            stain_rgb = np.clip(stain_rgb, 0, 255)
            stain_rgb = np.unique(stain_rgb, axis=0)
            decon = stainmat2decon(stain_rgb)
            deconvolved = deconvolve_img(self.to_array(), decon)

            d_flat = deconvolved.reshape(-1, deconvolved.shape[2])
            dmax = np.percentile(d_flat, q, axis=0) + np.finfo("float").eps
            for i in range(deconvolved.shape[2]):
                deconvolved[..., i] = np.clip(deconvolved[..., i], 0, dmax[i])
                deconvolved[..., i] /= dmax[i]
            array = deconvolved.mean(axis=2)
            array = numpy_to_sitk_image(array)
            array.SetSpacing(self.spacing)
            array = rescale_to_uint8(array)
        else:
            array = self.to_sitk()
            warnings.warn("Cannot compute stain flattening for non-RGB image.", stacklevel=2)
        return array  # type: ignore[no-any-return]


@register("gray")
class GrayPreprocessor(Preprocessor):
    """Convert to grayscale."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        if self.is_rgb:
            image = self.to_array()
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            array = numpy_to_sitk_image(image)
            array.SetSpacing(self.spacing)  # type: ignore[no-untyped-call]
        else:
            array = self.to_sitk()
            warnings.warn("Cannot convert to grayscale for non-RGB image.", stacklevel=2)
        return array


@register("he_deconvolution")
class HandEDeconvolutionPreprocessor(Preprocessor):
    """Normalize staining appearance of hematoxylin and eosin (H&E) stained image and get the HE deconvolution image.

    Reference
    ---------
    A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009.
    """

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = False

    def _apply(
        self,
        intensity: int = 240,
        alpha: int = 1,
        beta: float = 0.15,
        stain: ty.Literal["hem", "eos"] = "hem",
        **kwargs: ty.Any,
    ) -> sitk.Image:
        if self.is_rgb:
            image = self.to_array()
            concentrations = normalize_he(image, intensity=intensity, alpha=alpha, beta=beta)
            image = deconvolution_he(image, intensity=intensity, concentrations=concentrations, stain=stain)
            image = numpy_to_sitk_image(image)  # type: ignore[assignment]
            image.SetSpacing(self.spacing)  # type: ignore[attr-defined]
            return image  # type: ignore[return-value]
        return self.to_sitk()


@register("contrast_enhance")
class ContrastEnhancePreprocessor(Preprocessor):
    """Contrast enhancement."""

    allow_rgb = True
    allow_single_channel = True
    allow_multi_channel = True

    def _apply(self, alpha: int = 7, beta: float = 1, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_array()
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = sitk.GetImageFromArray(image)  # type: ignore[assignment]
        image.SetSpacing(self.original_spacing)  # type: ignore[attr-defined]
        return image  # type: ignore[return-value]


@register("max_intensity_projection")
class MaximumIntensityProcessor(Preprocessor):
    """Maximum intensity projection of a multichannel SimpleITK image."""

    allow_rgb = True
    allow_single_channel = False
    allow_multi_channel = True

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        if self.is_multi_channel:
            image = sitk.MaximumProjection(image, 2)[:, :, 0]  # type: ignore[no-untyped-call]
        elif self.is_rgb:
            image = sitk.MaximumProjection(image, 0)[0, :, :]  # type: ignore[no-untyped-call]
        return image


@register("invert_intensity")
class InvertIntensityPreprocessor(Preprocessor):
    """Invert intensity."""

    allow_rgb = True
    allow_single_channel = True
    allow_multi_channel = True

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        return sitk.InvertIntensity(image)  # type: ignore[no-untyped-call, no-any-return]


@register("downsample")
class DownsamplePreprocessor(Preprocessor):
    """Downsample image."""

    allow_rgb = True
    allow_single_channel = True
    allow_multi_channel = True

    def _apply(self, factor: int = 1, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        if factor > 1:
            return sitk.Shrink(  # type: ignore[no-any-return]
                image,
                [int(factor)] * image.GetDimension(),  # type: ignore[no-untyped-call]
            )
        return image

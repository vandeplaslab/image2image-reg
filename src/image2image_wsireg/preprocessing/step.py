"""Pre-processing step."""
from __future__ import annotations

import typing as ty

import cv2
import numpy as np
import SimpleITK as sitk
from skimage import exposure

from image2image_wsireg.preprocessing.convert import numpy_to_sitk_image, numpy_view_to_sitk_image, sitk_image_to_numpy
from image2image_wsireg.preprocessing.utilities import (
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


class Preprocessor:
    """Pre-processing step."""

    def __init__(self, array: np.ndarray | sitk.Image, pixel_size: float, level: int = 0):
        self.array = array
        self.level = level
        self.pixel_size = float(pixel_size)

    def __call__(self, **kwargs: ty.Any) -> sitk.Image | np.ndarray:
        """Apply step to image."""
        return self.apply(**kwargs)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get image shape."""
        if isinstance(self.array, np.ndarray):
            shape = self.array.shape
        else:
            shape = numpy_view_to_sitk_image(self.array).shape
        return shape

    @property
    def original_spacing(self) -> tuple[float, ...]:
        """Get image spacing."""
        shape = self.shape
        ndim = len(shape)
        if ndim == 2:
            return self.pixel_size, self.pixel_size
        else:
            return (self.pixel_size, self.pixel_size, 1) if self.is_rgb else (1, self.pixel_size, self.pixel_size)

    @property
    def spacing(self) -> tuple[float, float]:
        """Get image spacing."""
        return self.pixel_size, self.pixel_size

    @property
    def is_rgb(self) -> bool:
        """Check if image is RGB."""
        from image2image_io.readers.utilities import guess_rgb

        return guess_rgb(self.shape)

    @property
    def is_multi_channel(self) -> bool:
        """Check if image is multichannel."""
        return not self.is_rgb and not self.is_single_channel

    @property
    def is_single_channel(self) -> bool:
        """Check if image is single-channel."""
        return len(self.shape) == 2

    def to_sitk(self) -> sitk.Image:
        """Convert to SimpleITK filter."""
        if isinstance(self.array, sitk.Image):
            self.array.SetSpacing(self.original_spacing)  # type: ignore[no-untyped-call]
            return self.array
        return numpy_to_sitk_image(self.array)

    def to_array(self) -> np.ndarray:
        """Convert to NumPy array."""
        if isinstance(self.array, np.ndarray):
            return self.array
        return sitk_image_to_numpy(self.array)

    def apply(self, to_array: bool = False, **kwargs: ty.Any) -> sitk.Image | np.ndarray:
        """Apply step to image."""
        array = self._apply(**kwargs)
        if to_array:
            return sitk_image_to_numpy(array)
        return array

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        raise NotImplementedError


class ColorStandardizerPreprocessor(Preprocessor):
    """Color standardization."""

    def _apply(
        self, c: float = 0.2, invert: bool = True, adaptive_equalize: bool = False, **kwargs: ty.Any
    ) -> sitk.Image:
        if self.is_rgb:
            array = self.to_array()
            std_rgb = standardize_colorfulness(array, c)
            std_g = GrayPreprocessor(std_rgb, pixel_size=self.pixel_size).apply(to_array=True)
            if invert:
                std_g = 255 - std_g

            if adaptive_equalize:
                std_g = exposure.equalize_adapthist(std_g / 255)

            array = exposure.rescale_intensity(std_g, in_range="image", out_range=(0, 255)).astype(np.uint8)
            array = numpy_to_sitk_image(array)  # type: ignore[assignment]
            array.SetSpacing(self.spacing)  # type: ignore[attr-defined]
            return array  # type: ignore[return-value]
        return self.to_sitk()


class LuminosityPreprocessor(Preprocessor):
    """Luminosity."""

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_array()
        inv_lum = 255 - get_luminosity(image)
        image = exposure.rescale_intensity(inv_lum, out_range=(0, 255)).astype(np.uint8)
        image = numpy_to_sitk_image(image)  # type: ignore[assignment]
        image.SetSpacing(self.spacing)  # type: ignore[attr-defined]
        return image  # type: ignore[return-value]


class BackgroundColorDistancePreprocessor(Preprocessor):
    """Background color distance."""

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
        return array  # type: ignore[return-value]


class StainFlattenerPreprocessor(Preprocessor):
    """Stain flattening."""

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
            D = stainmat2decon(stain_rgb)
            deconvolved = deconvolve_img(self.to_array(), D)

            d_flat = deconvolved.reshape(-1, deconvolved.shape[2])
            dmax = np.percentile(d_flat, q, axis=0) + np.finfo("float").eps
            for i in range(deconvolved.shape[2]):
                deconvolved[..., i] = np.clip(deconvolved[..., i], 0, dmax[i])
                deconvolved[..., i] /= dmax[i]
            summary_img = deconvolved.mean(axis=2)
            summary_img = numpy_to_sitk_image(summary_img)
            summary_img.SetSpacing(self.spacing)
            return rescale_to_uint8(summary_img)  # type: ignore[no-any-return]
        return self.to_sitk()


class GrayPreprocessor(Preprocessor):
    """Convert to grayscale."""

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        if self.is_rgb:
            image = self.to_array()
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            image = numpy_to_sitk_image(image)  # type: ignore[assignment]
            image.SetSpacing(self.spacing)  # type: ignore[attr-defined]
            return image  # type: ignore[return-value]
        elif self.is_multi_channel:
            raise ValueError("Cannot convert multichannel image to grayscale.")
        return self.to_sitk()


class HandEDeconvolutionPreprocessor(Preprocessor):
    """Normalize staining appearance of hematoxylin and eosin (H&E) stained image and get the HE deconvolution image.

    Reference
    ---------
    A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009.
    """

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


class ContrastEnhancePreprocessor(Preprocessor):
    """Contrast enhancement."""

    def _apply(self, alpha: int = 7, beta: float = 1, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_array()
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = sitk.GetImageFromArray(image)  # type: ignore[assignment]
        image.SetSpacing(self.spacing)  # type: ignore[attr-defined]
        return image  # type: ignore[return-value]


class MaximumIntensityProcessor(Preprocessor):
    """Maximum intensity projection of a multichannel SimpleITK image."""

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        if self.is_multi_channel:
            image = sitk.MaximumProjection(image, 2)[:, :, 0]  # type: ignore[no-untyped-call]
        elif self.is_rgb:
            image = sitk.MaximumProjection(image, 0)[0, :, :]  # type: ignore[no-untyped-call]
        return image


class InvertIntensityPreprocessor(Preprocessor):
    """Invert intensity."""

    def _apply(self, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        return sitk.InvertIntensity(image)  # type: ignore[no-untyped-call, no-any-return]


class DownsamplePreprocessor(Preprocessor):
    """Downsample image."""

    def _apply(self, factor: int = 1, **kwargs: ty.Any) -> sitk.Image:
        image = self.to_sitk()
        if factor > 1:
            return sitk.Shrink(  # type: ignore[no-any-return]
                image,
                [int(factor)] * image.GetDimension(),  # type: ignore[no-untyped-call]
            )
        return image

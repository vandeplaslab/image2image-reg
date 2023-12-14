"""Transformation model."""
from __future__ import annotations

from warnings import warn

import numpy as np
import SimpleITK as sitk

from image2image_wsireg.utils.convert import convert_to_itk


class TransformMixin:
    """Mixin class for transforms."""

    resampler: sitk.ResampleImageFilter | None = None

    name: str = "unknown"
    is_linear: bool = True
    output_origin: tuple[float, float] | None = None
    output_size: tuple[int, int] | None = None
    output_spacing: tuple[float, float] | None = None
    output_direction: tuple[float, float] | None = None
    resample_interpolator: str = "FinalNearestNeighborInterpolator"
    final_transform: sitk.Transform | None = None
    inverse_transform: sitk.Transform
    transforms: list[Transform]

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform."""
        if self.resampler is None:
            self._build_resampler()
            if self.resampler is None:
                raise ValueError("Resampler not built, call `build_resampler` first")
        return self.resampler.Execute(image)  # type: ignore[no-any-return, no-untyped-call]

    def __repr__(self) -> str:
        """Return repr."""
        return f"{self.__class__.__name__}(name={self.name}; n={self.n_transforms}; is_linear={self.is_linear})"

    @property
    def n_transforms(self) -> int:
        """Number of transformations in sequence."""
        return len(self.transforms)

    @property
    def inverse_final_transform(self) -> sitk.Transform:
        """Return inverse final transform."""
        if self.final_transform is None:
            raise ValueError("Final transform does not exist yet.")
        if not self.is_linear:
            self.compute_inverse_nonlinear()
            return self.inverse_transform
        return self.final_transform.GetInverse()  # type: ignore[no-untyped-call]

    def _build_resampler(self, inverse: bool = False) -> None:
        """Build resampler."""
        from image2image_wsireg.enums import ELX_TO_ITK_INTERPOLATORS

        if any(v is None for v in [self.output_origin, self.output_size, self.output_spacing, self.output_direction]):
            raise ValueError("Output parameters not set, call `set_output_params` first")

        resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
        resampler.SetOutputOrigin(self.output_origin)  # type: ignore[no-untyped-call]
        resampler.SetOutputDirection(self.output_direction)  # type: ignore[no-untyped-call]
        resampler.SetSize(self.output_size)  # type: ignore[no-untyped-call]
        resampler.SetOutputSpacing(self.output_spacing)  # type: ignore[no-untyped-call]

        interpolator = ELX_TO_ITK_INTERPOLATORS[self.resample_interpolator]
        resampler.SetInterpolator(interpolator)  # type: ignore[no-untyped-call]
        # transform = self.inverse_final_transform if inverse else self.final_transform
        resampler.SetTransform(self.final_transform)  # type: ignore[no-untyped-call]
        self.resampler = resampler

    def compute_inverse_nonlinear(self) -> None:
        """Compute the inverse of a BSpline transform using ITK."""
        tform_to_disp_field = sitk.TransformToDisplacementFieldFilter()  # type: ignore[no-untyped-call]
        tform_to_disp_field.SetOutputSpacing(self.output_spacing)  # type: ignore[no-untyped-call]
        tform_to_disp_field.SetOutputOrigin(self.output_origin)  # type: ignore[no-untyped-call]
        tform_to_disp_field.SetOutputDirection(self.output_direction)  # type: ignore[no-untyped-call]
        tform_to_disp_field.SetSize(self.output_size)  # type: ignore[no-untyped-call]

        displacement_field = tform_to_disp_field.Execute(self.itk_transform)  # type: ignore[no-untyped-call]
        displacement_field = sitk.InvertDisplacementField(displacement_field)  # type: ignore[no-untyped-call]
        displacement_field = sitk.DisplacementFieldTransform(displacement_field)  # type: ignore[no-untyped-call]
        self.inverse_transform = displacement_field

    def transform_points(
        self, points: np.ndarray, is_px: bool = True, source_pixel_size: float = 1, px: bool = True
    ) -> np.ndarray:
        """
        Transform point sets using the transformation chain.

        Parameters
        ----------
        points: np.ndarray
            Point data in xy order
        is_px: bool
            Whether point data is in pixel or physical coordinate space
        source_pixel_size: float
            spacing of the pixels associated with pt_data if they are not in physical coordinate space
        px: bool
            return transformed points to pixel indices in the output_spacing's reference space.


        Returns
        -------
        transformed_points: np.ndarray
            Transformed points
        """
        if not self.output_spacing:
            raise ValueError("Output spacing not set, call `set_output_spacing` first")

        transformed_points = []
        for point in points:
            if is_px:
                point = point * source_pixel_size
            for _index, transform in enumerate(self.transforms):
                point = transform.inverse_transform.TransformPoint(point)
            transformed_point = np.array(point)

            if px:
                transformed_point *= 1 / self.output_spacing[0]
            transformed_points.append(transformed_point)
        return np.stack(transformed_points)

    def set_output_spacing(self, spacing: tuple[float, float] | tuple[int, int]) -> None:
        """Method that allows setting the output spacing of the resampler to resampled to any pixel spacing desired.

        This will also change the output_size to match.

        Parameters
        ----------
        spacing: tuple of float
            Spacing to set the new image. Will also change the output size to match.
        """
        output_size_scaling = np.asarray(self.output_spacing) / np.asarray(spacing)
        new_size = np.ceil(np.multiply(self.output_size, output_size_scaling))
        new_size: tuple[int, int] = tuple([int(i) for i in new_size])  # type: ignore[no-redef]

        self.output_spacing = spacing
        self.output_size = new_size  # type: ignore[assignment]
        self._build_resampler()

    def as_array(
        self,
        yx: bool = False,
        n_dim: int = 3,
        inverse: bool = False,
        px: bool = False,
    ) -> np.ndarray | None:
        """Creates an affine transform matrix as np.ndarray whether the center of rotation is 0,0.

        Optionally in physical or pixel coordinates.

        Parameters
        ----------
        yx: bool
            Use numpy ordering of yx (napari-compatible)
        n_dim: int
            Number of dimensions in the affine matrix, using 3 creates a 3x3 array
        inverse: bool
            return the inverse affine transformation
        px: bool
            return the transformation matrix specified in pixels or physical (microns).

        Returns
        -------
        full_matrix: np.ndarray
            Affine transformation matrix
        """
        if not self.final_transform:
            raise ValueError("Final transform does not exist yet.")

        if self.is_linear:
            if yx:
                order = slice(None, None, -1)
            else:
                order = slice(None, None, 1)

            transform = self.final_transform.GetInverse() if inverse else self.final_transform

            # pull transform values
            transform_matrix = np.array(transform.GetMatrix()[order]).reshape(2, 2)
            center = np.array(transform.GetCenter()[order])
            translation = np.array(transform.GetTranslation()[order])

            if px:
                phys_to_index = 1 / np.asarray(self.output_spacing).astype(np.float64)
                center *= phys_to_index
                translation *= phys_to_index

            # construct matrix
            full_matrix = np.eye(n_dim)
            full_matrix[0:2, 0:2] = transform_matrix
            full_matrix[0:2, n_dim - 1] = -np.dot(transform_matrix, center) + translation + center

            return full_matrix
        warn("Non-linear transformations can not be represented converted" "to homogenous matrix", stacklevel=2)
        return None


class Transform(TransformMixin):
    """Container for elastix transform that manages inversion and other metadata.

    Converts elastix transformation dict to it's SimpleITK representation.

    Attributes
    ----------
    elastix_transform: dict
        elastix transform stored in a python dict
    itk_transform: sitk.Transform
        elastix transform in SimpleITK container
    output_spacing: list of float
        Spacing of the targeted image during registration
    output_size: list of int
        Size of the targeted image during registration
    output_direction: list of float
        Direction of the targeted image during registration (not relevant for 2D applications)
    output_origin: list of float
        Origin of the targeted image during registration
    resampler_interpolator: str
        elastix interpolator setting for resampling the image
    is_linear: bool
        Whether the given transform is linear or non-linear (non-rigid)
    inverse_transform: sitk.Transform or None
        Inverse of the itk transform used for transforming from moving to fixed space
        Only calculated for non-rigid transforms when called by `compute_inverse_nonlinear`
        as the process is quite memory and computationally intensive

    """

    def __init__(self, elastix_transform: dict):
        self.elastix_transform: dict[str, list[str]] = elastix_transform
        self.itk_transform: sitk.Transform = convert_to_itk(self.elastix_transform)
        self.transforms = [self]

        self.output_spacing = [float(p) for p in self.elastix_transform["Spacing"]]
        self.output_size = [int(p) for p in self.elastix_transform["Size"]]
        self.output_origin = [float(p) for p in self.elastix_transform["Origin"]]
        self.output_direction = [float(p) for p in self.elastix_transform["Direction"]]
        self.resample_interpolator = self.elastix_transform["ResampleInterpolator"][0]
        self.is_linear = self.itk_transform.IsLinear()
        self.name = self.itk_transform.GetName()

        if self.is_linear:
            self.inverse_transform = self.itk_transform.GetInverse()
            transform_name = self.itk_transform.GetName()
            if transform_name == "Euler2DTransform":
                self.inverse_transform = sitk.Euler2DTransform(self.inverse_transform)
            elif transform_name == "AffineTransform":
                self.inverse_transform = sitk.AffineTransform(self.inverse_transform)
            elif transform_name == "Similarity2DTransform":
                self.inverse_transform = sitk.Similarity2DTransform(self.inverse_transform)
        else:
            self.inverse_transform = None

    @property
    def final_transform(self) -> sitk.Transform:  # type: ignore[override]
        """Return final transform."""
        return self.itk_transform

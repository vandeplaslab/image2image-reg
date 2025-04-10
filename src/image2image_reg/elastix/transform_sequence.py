"""Transformation sequence."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from warnings import warn

import numpy as np
import SimpleITK as sitk
from koyo.json import read_json, write_json
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from tqdm import tqdm

from image2image_reg.elastix.convert import convert_to_itk


def _hash_parameters(elastix_transform: dict) -> str:
    """Hash elastix parameters."""
    if elastix_transform["Transform"][0] in [
        "EulerTransform",
        "SimilarityTransform",
        "AffineTransform",
        "TranslationTransform",
    ]:
        return hash_parameters(n_in_hash=4, **elastix_transform)
    return hash_parameters(n_in_hash=4, exclude_keys=("TransformParameters",), **elastix_transform)


def transform_points(
    transform: sitk.Transform | sitk.CompositeTransform,
    points: np.ndarray,
    target_pixel_size: float = 1.0,
    is_px: bool = True,
    as_px: bool = True,
    source_pixel_size: float = 1,
    silent: bool = False,
    copy: bool = False,
):
    """Transform points."""
    inv_target_pixel_size = 1 / target_pixel_size

    # convert from px to um by multiplying by the pixel size
    transformed_points = np.asarray(points, dtype=np.float64)
    if copy:
        transformed_points = transformed_points.copy()
    if is_px:
        transformed_points = transformed_points * source_pixel_size

    for i, point in enumerate(
        tqdm(
            transformed_points,
            desc=f"Transforming points (is={is_px}; as={as_px}; s={source_pixel_size:.3f};"
            f" t={target_pixel_size:.3f}; t-inv={inv_target_pixel_size:.3f})",
            leave=False,
            disable=silent,
            mininterval=1,
        )
    ):
        transformed_points[i] = transform.TransformPoint(point)  # type: ignore[no-untyped-call]
    if as_px:
        transformed_points = transformed_points * inv_target_pixel_size
    return transformed_points


class TransformMixin:
    """Mixin class for transforms."""

    resampler: sitk.ResampleImageFilter | None = None

    name: str = "Unknown"
    is_linear: bool = True
    output_origin: tuple[float, float] | None = None
    output_size: tuple[int, int] | None = None
    output_spacing: tuple[float, float] | None = None
    output_direction: tuple[float, float] | None = None
    resample_interpolator: str = "FinalNearestNeighborInterpolator"
    transforms: list[Transform]
    final_transform: sitk.Transform | None = None
    _inverse_transform: sitk.Transform | None = None

    itk_transform: sitk.Transform
    _itk_transform = None
    _is_linear = None
    inverse: bool = False

    def __repr__(self) -> str:
        """Return repr."""
        spx, spy = self.output_spacing
        return f"{self.__class__.__name__}<name={self.name}; spacing=({spx:.4f}, {spy:.4f}); size={self.output_size}>"

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform."""
        if self.resampler is None:
            self._build_resampler()
            if self.resampler is None:
                raise ValueError("Resampler not built, call `build_resampler` first")
        return self.resampler.Execute(image)  # type: ignore[no-any-return, no-untyped-call]

    def set_inverse(self, inverse: bool) -> None:
        """Apply inverse transformation."""
        self.inverse = inverse
        self._build_resampler(inverse=inverse)

    def transform_points(
        self,
        points: np.ndarray,
        is_px: bool = True,
        as_px: bool = True,
        source_pixel_size: float = 1,
        silent: bool = False,
        inverse: bool = False,
    ) -> np.ndarray:
        """
        Transform point sets using the transformation chain.

        Parameters
        ----------
        points: np.ndarray
            Point data in xy order in micrometer space
        is_px: bool
            Whether point data is in pixel or physical coordinate space
        source_pixel_size: float
            spacing of the pixels associated with pt_data if they are not in physical coordinate space
        as_px: bool
            return transformed points to pixel indices in the output_spacing's reference space.
        silent: bool
            Whether to show progress bar
        inverse : bool
            Whether to apply the forward transformation - if points are being transformed from moving -> fixed, then we
            always used the inverse transformation (in forward order). IF you wish to go from moving -> fixed, then you
            must use the inverse transformation (in reverse order).

        Returns
        -------
        transformed_points: np.ndarray
            Transformed points
        """
        if not self.output_spacing:
            raise ValueError("Output spacing not set, call `set_output_spacing` first")

        target_pixel_size = self.output_spacing[0]
        transform = self.reverse_final_transform if inverse else self.inverse_final_transform

        return transform_points(
            transform,
            points,
            target_pixel_size,
            is_px=is_px,
            as_px=as_px,
            source_pixel_size=source_pixel_size,
            silent=silent,
        )

    @property
    def n_transforms(self) -> int:
        """Number of transformations in sequence."""
        return len(self.transforms)

    @property
    def resolution(self) -> float:
        """Return resolution."""
        return self.output_spacing[0]  # type: ignore[index]

    @property
    def reverse_final_transform(self) -> sitk.Transform:
        """Return inverse final transform."""
        return self.final_transform

    @property
    def inverse_final_transform(self) -> sitk.Transform:
        """Return inverse final transform."""
        return self.inverse_transform

    @property
    def reverse_inverse_final_transform(self) -> sitk.Transform:
        """Return inverse final transform."""
        return self.inverse_transform

    @property
    def inverse_transform(self) -> sitk.Transform:
        """Return inverse transform."""
        raise NotImplementedError("Must implement method")

    def _build_resampler(self, inverse: bool = False) -> None:
        """Build resampler."""
        from image2image_reg.enums import ELX_TO_ITK_INTERPOLATORS

        if any(v is None for v in [self.output_origin, self.output_size, self.output_spacing, self.output_direction]):
            raise ValueError("Output parameters not set, call `set_output_params` first")

        interpolator = ELX_TO_ITK_INTERPOLATORS[self.resample_interpolator]

        resampler = sitk.ResampleImageFilter()  # type: ignore[no-untyped-call]
        resampler.SetOutputOrigin(self.output_origin)  # type: ignore[no-untyped-call]
        resampler.SetOutputDirection(self.output_direction)  # type: ignore[no-untyped-call]
        resampler.SetSize(self.output_size)  # type: ignore[no-untyped-call]
        resampler.SetOutputSpacing(self.output_spacing)  # type: ignore[no-untyped-call]
        resampler.SetInterpolator(interpolator)  # type: ignore[no-untyped-call]
        resampler.SetTransform(  # type: ignore[no-untyped-call]
            self.final_transform if not inverse else self.inverse_final_transform,
        )
        self.resampler = resampler

    def set_output_size(self, output_size: tuple[int, int]) -> None:
        """Set output size."""
        self.output_size = output_size
        if hasattr(self, "elastix_transform"):
            self.elastix_transform["Size"] = [str(i) for i in output_size]
        else:
            self.transforms[-1].elastix_transform["Size"] = [str(i) for i in output_size]
        self._build_resampler()

    def set_output_spacing(
        self,
        spacing: tuple[float, float] | tuple[int, int],
        output_size: tuple[int, int] | None = None,
    ) -> None:
        """Method that allows setting the output spacing of the resampler to resampled to any pixel spacing desired.

        This will also change the output_size to match.

        Parameters
        ----------
        spacing: tuple of float
            Spacing to set the new image. Will also change the output size to match.
        output_size: tuple of int
            Size of the output image. If None, will be calculated from the output_spacing
        """
        if output_size is None:
            output_size_scaling = np.asarray(self.output_spacing) / np.asarray(spacing)
            new_size = np.ceil(np.multiply(self.output_size, output_size_scaling))
            output_size: tuple[int, int] = tuple([int(i) for i in new_size])  # type: ignore[no-redef]

        self.output_spacing = spacing
        self.output_size = output_size
        # update itk data
        if hasattr(self, "elastix_transform"):
            self.elastix_transform["Spacing"] = [str(i) for i in spacing]
            self.elastix_transform["Size"] = [str(i) for i in output_size]
        else:
            self.transforms[-1].elastix_transform["Spacing"] = [str(i) for i in spacing]
            self.transforms[-1].elastix_transform["Size"] = [str(i) for i in output_size]
        logger.trace(f"Updated output spacing and size of {self}")
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
            order = slice(None, None, -1 if yx else 1)

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
        warn("Non-linear transformations can not be represented converted to homogenous matrix", stacklevel=2)
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
    is_linear: bool
        Whether the given transform is linear or non-linear (non-rigid)
    inverse_transform: sitk.Transform or None
        Inverse of the itk transform used for transforming from moving to fixed space
        Only calculated for non-rigid transforms when called by `compute_inverse_nonlinear`
        as the process is quite memory and computationally intensive

    """

    def __init__(self, elastix_transform: dict):
        self.elastix_transform: dict[str, list[str]] = elastix_transform
        self._parameter_hash = _hash_parameters(elastix_transform)
        self.transforms = [self]

        self.output_spacing = [float(p) for p in self.elastix_transform["Spacing"]]
        self.output_size = [int(p) for p in self.elastix_transform["Size"]]
        self.output_origin = [float(p) for p in self.elastix_transform["Origin"]]
        self.output_direction = [float(p) for p in self.elastix_transform["Direction"]]
        self.resample_interpolator = self.elastix_transform["ResampleInterpolator"][0]
        self.name = f"{self.elastix_transform['Transform'][0]} ({self._parameter_hash})"

    @property
    def itk_transform(self) -> sitk.Transform:
        """SimpleITK Transform/."""
        if self._itk_transform is None:
            self._itk_transform: sitk.Transform = convert_to_itk(self.elastix_transform)  # type: ignore[no-redef]
            self._is_linear = self.itk_transform.IsLinear()
            self.name = f"{self.itk_transform.GetName()} ({self._parameter_hash})"
        return self._itk_transform

    @property
    def is_linear(self) -> bool:
        """Check if transform is linear."""
        if self._is_linear is None:
            self._is_linear = self.itk_transform.IsLinear()
        return self._is_linear

    @is_linear.setter
    def is_linear(self, value: bool) -> None:
        self._is_linear = value

    @property
    def final_transform(self) -> sitk.Transform:  # type: ignore[override]
        """Return final transform."""
        return self.itk_transform

    @property
    def inverse_transform(self) -> sitk.Transform:
        """Get inverse transform."""
        if self._inverse_transform is None:
            self._inverse_transform = (
                self.compute_inverse_linear() if self.is_linear else self.compute_inverse_nonlinear()
            )
        return self._inverse_transform

    def compute_inverse_linear(self) -> sitk.Transform:
        """Compute the inverse of a linear transform using ITK."""
        inverse_transform = self.itk_transform.GetInverse()
        transform_name = self.itk_transform.GetName()
        if transform_name == "Euler2DTransform":
            self._inverse_transform = sitk.Euler2DTransform(inverse_transform)  # type: ignore[no-untyped-call]
        elif transform_name == "AffineTransform":
            self._inverse_transform = sitk.AffineTransform(inverse_transform)  # type: ignore[no-untyped-call]
        elif transform_name == "Similarity2DTransform":
            self._inverse_transform = sitk.Similarity2DTransform(inverse_transform)  # type: ignore[no-untyped-call]
        else:
            raise ValueError(f"Transform: {transform_name} not recognized")
        return inverse_transform  # type: ignore[no-any-return]

    def compute_inverse_nonlinear(self) -> sitk.DisplacementFieldTransform:
        """Compute the inverse of a BSpline transform using ITK."""
        with MeasureTimer() as timer:
            tform_to_disp_field = sitk.TransformToDisplacementFieldFilter()  # type: ignore[no-untyped-call]
            tform_to_disp_field.SetOutputSpacing(self.output_spacing)  # type: ignore[no-untyped-call]
            tform_to_disp_field.SetOutputOrigin(self.output_origin)  # type: ignore[no-untyped-call]
            tform_to_disp_field.SetOutputDirection(self.output_direction)  # type: ignore[no-untyped-call]
            tform_to_disp_field.SetSize(self.output_size)  # type: ignore[no-untyped-call]

            displacement_field = tform_to_disp_field.Execute(self.final_transform)  # type: ignore[no-untyped-call]
            displacement_field = sitk.InvertDisplacementField(displacement_field)  # type: ignore[no-untyped-call]
            displacement_field = sitk.DisplacementFieldTransform(displacement_field)  # type: ignore[no-untyped-call]
        logger.trace(f"Computed inverse transform of {self} in {timer()}")
        return displacement_field  # type: ignore[no-any-return]

    def to_dict(self) -> dict:
        """Convert transformation to dictionary."""
        return self.elastix_transform


class TransformSequence(TransformMixin):
    """Class to concatenate and compose sequences of transformations.

    Parameters
    ----------
    transforms: List or single Transform or None
        Transforms to be composed
    transform_sequence_index: list of int
        Order in sequence of the transform. If a pre-reg transform, it will not be reversed like a sequence
        of elastix transforms would to make the composite ITK transform
    """

    def __init__(
        self,
        transforms: str | Path | list[dict[str, list[str]]] | Transform | list[Transform] | None = None,
        transform_sequence_index: list[int] | None = None,
    ) -> None:
        self._transform_sequence_index: list[int] = []
        self.transforms: list[Transform] = []
        self.resampler: sitk.ResampleImageFilter | None = None
        self._composite_transform: sitk.CompositeTransform | None = None
        self._reverse_composite_transform: sitk.CompositeTransform | None = None
        self._inverse_composite_transform: sitk.CompositeTransform | None = None
        self._reverse_inverse_composite_transform: sitk.CompositeTransform | None = None
        self._n_transforms = 0

        self.add_transforms(transforms, transform_sequence_index=transform_sequence_index)

    def __repr__(self) -> str:
        """Return repr."""
        seq = " > ".join([t.name for t in self.transforms])
        spacing = f"{self.output_spacing[0]:.4f}" if self.output_spacing else "None"
        rep = (
            f"{self.__class__.__name__}<name={self.name}; n={self.n_transforms}; spacing={spacing}; "
            f"size={self.output_size}; seq={seq}>"
        )
        return rep

    @property
    def final_transform(self) -> sitk.Transform:  # type: ignore[override]
        """Final ITK transform."""
        return self.composite_transform  # type: ignore[return-value]

    @property
    def reverse_final_transform(self) -> sitk.Transform:  # type: ignore[override]
        """Final ITK transform."""
        if self._reverse_composite_transform is None:
            self._reverse_composite_transform = self._build_reverse_composite_transform()  # type: ignore[return-value]
        return self._reverse_composite_transform

    @property
    def composite_transform(self) -> sitk.CompositeTransform | None:
        """Composite ITK transform from transformation sequence."""
        if self._composite_transform is None:
            self._build_composite_transform()
        return self._composite_transform

    @property
    def inverse_transform(self) -> sitk.Transform:
        """Compute inverse transform."""
        if not self._inverse_transform:
            if self.final_transform is None:
                raise ValueError("Final transform does not exist yet.")
            self._inverse_transform = self._build_inverse_composite_transform()
        return self._inverse_transform

    @property
    def reverse_inverse_final_transform(self) -> sitk.Transform:
        """Return reverse inverse final transform."""
        if self._reverse_inverse_composite_transform is None:
            self._reverse_inverse_composite_transform = self._build_reverse_inverse_composite_transform()
        return self._reverse_inverse_composite_transform

    def append(self, other: TransformSequence) -> None:
        """Concatenate transformation sequences."""
        self.add_transforms(other.transforms, other.transform_sequence_index)

    def insert(self, other: TransformSequence) -> None:
        """Insert transformation sequence before all the other transforms."""
        n_in_other = len(other.transforms)
        existing_indices = self.transform_sequence_index
        existing_transforms = self.transforms
        self.transforms = other.transforms + existing_transforms
        self._transform_sequence_index = list(range(n_in_other)) + [x + n_in_other for x in existing_indices]
        self._update_transform_properties()

    def add_transforms(
        self,
        transforms: str | Path | list[dict[str, list[str]]] | Transform | list[Transform] | None,
        transform_sequence_index: list[int] | None = None,
    ) -> None:
        """
        Add transforms to sequence.

        Parameters
        ----------
        transforms: path to wsireg transforms .json, elastix transform dict,Transform ot List of Transform
        transform_sequence_index: list of int
            Order in sequence of the transform. If a pre-reg transform, it will not be reversed like a sequence
            of elastix transforms would to make the composite ITK transform

        """
        if not transforms:
            return

        if isinstance(transforms, (str, Path)):
            tform_list, tform_idx = _read_elastix_transform(transforms)
            self.transform_sequence_index = tform_idx
            reg_transforms = [Transform(t) for t in tform_list]
            self.transforms = self.transforms + reg_transforms
        elif isinstance(transforms, list):
            if isinstance(transforms[0], dict):
                reg_transforms = [Transform(t) for t in transforms]
            elif isinstance(transforms[0], Transform):
                reg_transforms = transforms
            else:
                raise ValueError("Transforms must be a list of Transform objects or a list of dicts")
            if self.transforms:
                self.transforms = [*self.transforms, *reg_transforms]
            else:
                self.transforms = reg_transforms
            self.transform_sequence_index = transform_sequence_index
        elif isinstance(transforms, Transform):
            self.transforms = [*self.transforms, transforms]
            self.transform_sequence_index = transform_sequence_index
        self._update_transform_properties()

    @property
    def transform_sequence_index(self) -> list[int]:
        """Transformation sequence for all combined transformations."""
        return self._transform_sequence_index

    @transform_sequence_index.setter
    def transform_sequence_index(self, transform_seq: list[int]) -> None:
        reindex_val = 0
        if len(self._transform_sequence_index) > 0:
            reindex_val = np.max(self._transform_sequence_index) + 1
        transform_seq = [x + reindex_val for x in transform_seq]
        self._transform_sequence_index = self._transform_sequence_index + transform_seq

    def transform_iterator(
        self,
        transforms: list[Transform] | None = None,
        transform_sequence_index: list[int] | None = None,
        reverse: bool = False,
    ) -> ty.Generator[Transform, None, None]:
        """Transform iterator for all transforms in sequence."""
        if transforms is None:
            transforms = self.transforms
        if transform_sequence_index is None:
            transform_sequence_index = self.transform_sequence_index
        composite_index: list[int] = []
        for unique_idx in np.unique(transform_sequence_index):
            in_seq_tform_idx = np.where(transform_sequence_index == unique_idx)[0]
            if len(in_seq_tform_idx) > 1:
                composite_index = composite_index + list(in_seq_tform_idx[::-1])
            else:
                composite_index = composite_index + list(in_seq_tform_idx)

        for transform_index in sorted(composite_index, reverse=reverse):
            yield transforms[transform_index]

    def _update_transform_properties(self) -> None:
        self.output_size = self.transforms[-1].output_size
        self.output_spacing = self.transforms[-1].output_spacing
        self.output_direction = self.transforms[-1].output_direction
        self.output_origin = self.transforms[-1].output_origin
        self.resample_interpolator = self.transforms[-1].resample_interpolator
        self._build_composite_transform()
        self._build_resampler()

    def _build_composite_transform(self) -> sitk.CompositeTransform:
        """Build composite transform from a list of transforms."""
        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
        for transform in self.transform_iterator():
            composite_transform.AddTransform(transform.itk_transform)  # type: ignore[has-type,no-untyped-call]

        self._composite_transform = composite_transform
        self.is_linear = composite_transform.IsLinear()  # type: ignore[no-untyped-call]
        return self._composite_transform

    def _build_reverse_composite_transform(self) -> sitk.CompositeTransform:
        """Build composite transform from a list of transforms."""
        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
        for transform in self.transform_iterator(reverse=True):
            composite_transform.AddTransform(transform.itk_transform)  # type: ignore[no-untyped-call]
        return composite_transform

    def _build_inverse_composite_transform(self) -> sitk.CompositeTransform:
        """Build inverse composite transform.

        Maps from moving to fixed space.
        """
        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]

        # this might be counter-intuitive, but we need to reverse the order of the transforms if we are going
        # from moving to fixed and we are applying the inverse transformation
        for transform in self.transform_iterator(reverse=True):
            composite_transform.AddTransform(transform.inverse_transform)  # type: ignore[no-untyped-call]
        return composite_transform

    def _build_reverse_inverse_composite_transform(self) -> sitk.CompositeTransform:
        """Build inverse composite transform.

        Maps from fixed to moving space.
        """
        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
        for transform in self.transform_iterator():
            composite_transform.AddTransform(transform.inverse_transform)  # type: ignore[no-untyped-call]
        return composite_transform

    @classmethod
    def from_path(cls, path: PathLike, first: bool = False, skip_initial: bool = False) -> TransformSequence:
        """Load a transform sequence from a path.

        Parameters
        ----------
        path : PathLike
            Path to transform sequence file.
        first : bool, optional
            Load only the first transform from the file. This is necessary when e.g. reloading transform data from disk
            after a registration has been performed but the transform sequence will be altered before being applied.
            This is ESSENTIAL when doing anything within the ElastixReg object as it will apply other transforms to the
            image. The transformation json file usually stores ALL necessary transformations.
        skip_initial: bool, optional
            Skip the initial transform. This is necessary when e.g. reloading transform data from disk
        """
        # TODO: check what happens if there is initial transformation - e.g. user supplied affine matrix
        transforms, transform_sequence_index = _read_elastix_transform(path, first, skip_initial)
        return cls(transforms, transform_sequence_index)

    @classmethod
    def read_partial_and_full(cls, path: PathLike, delay: bool = False) -> tuple[TransformSequence, TransformSequence]:
        """Load a transform sequence from a path.

        Parameters
        ----------
        path : PathLike
            Path to transform sequence file.
        delay : bool
            Delay initialization of the transform sequence.
        """
        data = read_json(path)
        transforms, transform_sequence_index = _read_elastix_transform(data, first=False, skip_initial=False)
        transforms_full_seq = cls(transforms, transform_sequence_index)
        transforms, transform_sequence_index = _read_elastix_transform(path, first=True, skip_initial=True)
        transforms_partial_seq = cls(transforms, transform_sequence_index)
        return transforms_partial_seq, transforms_full_seq

    @classmethod
    def from_final(cls, path: PathLike) -> TransformSequence:
        """Read final elastix transform."""
        transforms, transform_sequence_index = _read_final_elastix_transform(path)
        return cls(transforms, transform_sequence_index)

    @classmethod
    def from_i2r(cls, path: PathLike, image_path: PathLike) -> TransformSequence:
        """Load transform sequence from a i2r path.

        Parameters
        ----------
        path: PathLike
            Path to image2image registration transformation file.
        image_path: PathLike
            Path to the image so that metadata can be read.
        """
        transforms, transform_sequence_index = _read_i2r_transform(path, image_path)
        return cls(transforms, transform_sequence_index)

    def to_dict(self) -> list[dict]:
        """Export transformation sequence to dictionary."""
        out = []
        for transform in self.transforms:
            out.append(transform.to_dict())
        return out

    def to_json(self, path: PathLike) -> Path:
        """Export transformation sequence to json file."""
        path = Path(path)
        if len(path.suffixes) == 1:
            path = path.with_suffix(".elastix.json")

        out = self.to_dict()
        write_json(path, out)
        return path


def _read_i2r_transform(path: PathLike, image_path: PathLike) -> tuple[list[dict[str, list[str]]], list[int]]:
    """Read data from i2r transform dict."""
    from image2image_io.readers import get_simple_reader

    from image2image_reg.elastix.transform_utils import affine_to_itk_affine

    transforms_data: dict[str, list[str]] = read_json(path)
    if "matrix_yx_um_inv" not in transforms_data:
        raise ValueError("Cannot retrieve affine transformation.")
    affine = np.asarray(transforms_data["matrix_yx_um_inv"])
    reader = get_simple_reader(image_path, init_pyramid=False, auto_pyramid=False, quick=True)
    transform_list = [affine_to_itk_affine(affine, reader.image_shape[::-1], spacing=reader.resolution, inverse=False)]
    return transform_list, [0]


def _read_elastix_transform(
    parameters_or_path: str | (Path | dict[str, list[str]]),
    first: bool = False,
    skip_initial: bool = False,
) -> tuple[list[dict[str, list[str]]], list[int]]:
    """Convert wsireg transform dict or from file to List of Transforms."""
    transform_data = parameters_or_path
    if isinstance(parameters_or_path, (str, Path)):
        transform_data: dict[str, list[str]] = read_json(parameters_or_path)

    allowed_n = 1 if first else -1
    index = 0
    transforms = []
    transform_sequence_index = []
    for key, value in transform_data.items():
        if "initial" in key and skip_initial:
            continue
        if "initial" not in key and first:
            allowed_n -= 1
        if key == "initial":
            if isinstance(value, dict):
                transforms.append(value)
                transform_sequence_index.append(index)
                index += 1
            elif isinstance(value, list):
                for init_tform in value:
                    transforms.append(init_tform)
                    transform_sequence_index.append(index)
                    index += 1
        else:
            if isinstance(value, dict):
                transforms.append(value)
                transform_sequence_index.append(index)
                index += 1
            elif isinstance(value, list):
                for tform in value:
                    transforms.append(tform)
                    transform_sequence_index.append(index)
                index += 1
        if allowed_n == 0 and first:
            break
    return transforms, transform_sequence_index


def _read_final_elastix_transform(path: PathLike) -> tuple[list[dict[str, list[str]]], list[int]]:
    """Read final elastix transform."""
    path = Path(path)
    if path.suffixes == [".json"]:
        return _read_elastix_transform(path)
    elif path.suffixes == [".elastix", ".json"]:
        transforms, transform_sequence_index = [], []
        transform_data = read_json(path)
        for index, tform in enumerate(transform_data):
            transforms.append(tform)
            transform_sequence_index.append(index)
        return transforms, transform_sequence_index
    raise ValueError("Cannot read final elastix transform.")


# CLEANUP
# def _invert_bspline_transform(tx, output_size, output_origin, output_spacing, output_direction):
#     displacement_field_image = sitk.TransformToDisplacementField(
#         tx, sitk.sitkVectorFloat64, output_size, output_origin, output_spacing, output_direction
#     )
#     return _invert_displacement_field_image(displacement_field_image)
#
# def _invert_displacement_field_transform(tx):
#     return _invert_displacement_field_image(sitk.DisplacementFieldTransform(tx).GetDisplacementField())
#
# def _invert_displacement_field_image(displacement_field_image):
#     # SimpleITK supports three different filters for inverting a displacement field
#     # arbitrary selection used with default values
#     return sitk.DisplacementFieldTransform(sitk.InvertDisplacementField(displacement_field_image))

# return self._inverse_composite_transformtra
# inverted_transform_list = []
# # for transform in self.transforms:
# for transform in reversed(self.transforms):
#     print(transform)
#     tx = transform.itk_transform
#     ttype = tx.GetTransformEnum()
#     if ttype is sitk.sitkDisplacementField:
#         inverted_transform_list.append(_invert_displacement_field_transform(tx))
#     elif ttype is sitk.sitkBSplineTransform:
#         physical_size = tx.GetTransformDomainPhysicalDimensions()
#         grid_spacing = transform.output_spacing
#         grid_size = [int(phys_sz / spc + 1) for phys_sz, spc in zip(physical_size, grid_spacing)]
#         displacement_field_image = sitk.TransformToDisplacementField(
#             tx,
#             outputPixelType=sitk.sitkVectorFloat64,
#             size=grid_size,
#             outputOrigin=tx.GetTransformDomainOrigin(),
#             outputSpacing=grid_spacing,
#             outputDirection=tx.GetTransformDomainDirection(),
#         )
#         displacement_field_inverter = sitk.InvertDisplacementFieldImageFilter()
#         displacement_field_inverter.SetMaximumNumberOfIterations(100)
#         displacement_field_inverter.SetEnforceBoundaryCondition(True)
#         inverted_transform_list.append(
#             sitk.DisplacementFieldTransform(displacement_field_inverter.Execute(displacement_field_image))
#         )
#         # inverted_transform_list.append(
#         #     _invert_bspline_transform(
#         #         tx,
#         #         transform.output_size,
#         #         transform.output_origin,
#         #         transform.output_spacing,
#         #         transform.output_direction,
#         #     )
#         # )
#     else:
#         inverted_transform_list.append(tx.GetInverse())
# self._inverse_composite_transform = sitk.CompositeTransform(inverted_transform_list)

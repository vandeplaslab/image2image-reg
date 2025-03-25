"""Transformation sequence."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from koyo.json import read_json, write_json
from koyo.typing import PathLike

from image2image_reg.models.transform import Transform, TransformMixin


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
        self.transform_itk_order: list[Transform] = []
        self._composite_transform = None
        self._n_transforms = 0

        self.add_transforms(transforms, transform_sequence_index=transform_sequence_index)

    def __repr__(self) -> str:
        """Return repr."""
        seq = " > ".join([t.name for t in self.transforms])
        rep = f"{self.__class__.__name__}(name={self.name}; n={self.n_transforms}; seq={seq})"
        return rep

    @property
    def final_transform(self) -> sitk.Transform:  # type: ignore[override]
        """Final ITK transform."""
        return self.composite_transform

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
    def composite_transform(self) -> sitk.CompositeTransform | None:
        """Composite ITK transform from transformation sequence."""
        return self._composite_transform

    @composite_transform.setter
    def composite_transform(self, composite_transform: sitk.CompositeTransform | None) -> None:
        self._composite_transform = composite_transform
        if composite_transform:
            self.is_linear = composite_transform.IsLinear()

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
        self, transforms: list[Transform] | None = None, transform_sequence_index: list[int] | None = None
    ) -> ty.Generator[tuple[list[int], Transform], None, None]:
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

        for transform_index in composite_index:
            yield composite_index, transforms[transform_index]

    def _update_transform_properties(self) -> None:
        self.output_size = self.transforms[-1].output_size
        self.output_spacing = self.transforms[-1].output_spacing
        self.output_direction = self.transforms[-1].output_direction
        self.output_origin = self.transforms[-1].output_origin
        self.resample_interpolator = self.transforms[-1].resample_interpolator
        self._build_transform_data()

    def _build_transform_data(self) -> None:
        self._build_composite_transform(self.transforms, self.transform_sequence_index)
        self._build_resampler()

    def _build_composite_transform(self, transforms: list[Transform], transform_sequence_index: list[int]) -> None:
        """Build composite transform from a list of transforms."""
        assert len(transforms) == len(transform_sequence_index), "Transforms and sequence index must be the same length"
        assert len(transforms) != 0, "No transforms to build composite transform"

        composite_index = []
        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
        for composite_index, transform in self.transform_iterator(transforms, transform_sequence_index):
            composite_transform.AddTransform(transform.itk_transform)  # type: ignore[no-untyped-call]
        self.composite_transform = composite_transform
        self.transform_itk_order = [self.transforms[i] for i in composite_index]

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
        for _, transform in self.transform_iterator():
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

    from image2image_reg.utils.transformation import affine_to_itk_affine

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

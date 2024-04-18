"""Transformation sequence."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from koyo.json import read_json_data
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
        self.composed_linear_matrix_map: dict[str, np.ndarray] | None = None
        self.transform_itk_order: list[Transform] = []

        if transforms:
            self.add_transforms(transforms, transform_sequence_index=transform_sequence_index)
        else:
            self._composite_transform = None
            self._n_transforms = 0

    def __repr__(self) -> str:
        """Return repr."""
        seq = " > ".join([t.name for t in self.transforms])
        return (
            f"{self.__class__.__name__}(name={self.name}; n={self.n_transforms}; is_linear={self.is_linear}; seq={seq})"
        )

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
        if isinstance(transforms, (str, Path)):
            tform_list, tform_idx = _read_wsireg_transform(transforms)
            self.transform_sequence_index = tform_idx
            reg_transforms = [Transform(t) for t in tform_list]
            self.transforms = self.transforms + reg_transforms
        elif isinstance(transforms, list):
            if isinstance(transforms[0], dict):
                reg_transforms = [Transform(t) for t in transforms]
                self.transforms = self.transforms + reg_transforms
            elif isinstance(transforms[0], Transform):
                self.transforms = self.transforms + transforms
            else:
                raise ValueError("Transforms must be a list of Transform objects or a list of dicts")
            self.transform_sequence_index = transform_sequence_index
        elif isinstance(transforms, (list, Transform)):
            if isinstance(transforms, Transform):
                transforms = [transforms]
            self.transforms = self.transforms + transforms
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
        if len(self._transform_sequence_index) > 0:
            reindex_val = np.max(self._transform_sequence_index) + 1
        else:
            reindex_val = 0
        transform_seq = [x + reindex_val for x in transform_seq]
        self._transform_sequence_index = self._transform_sequence_index + transform_seq

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

    def _build_composite_transform(self, reg_transforms: list[Transform], reg_transform_seq_idx: list[int]) -> None:
        """Build composite transform from a list of transforms."""
        composite_index: list[int] = []
        for unique_idx in np.unique(reg_transform_seq_idx):
            in_seq_tform_idx = np.where(reg_transform_seq_idx == unique_idx)[0]
            if len(in_seq_tform_idx) > 1:
                composite_index = composite_index + list(in_seq_tform_idx[::-1])
            else:
                composite_index = composite_index + list(in_seq_tform_idx)

        composite_transform = sitk.CompositeTransform(2)  # type: ignore[no-untyped-call]
        for transform_index in composite_index:
            composite_transform.AddTransform(  # type: ignore[no-untyped-call]
                reg_transforms[transform_index].itk_transform,
            )
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
            This is ESSENTIAL when doing anything within the IWsiReg object as it will apply other transforms to the
            image. The transformation json file usually stores ALL necessary transformations.
        skip_initial: bool, optional
            Skip the initial transform. This is necessary when e.g. reloading transform data from disk
        """
        # TODO: check what happens if there is initial transformation - e.g. user supplied affine matrix
        transforms, transform_sequence_index = _read_wsireg_transform(path, first, skip_initial)
        # if first:
        #     transforms = [transforms[0]]
        #     transform_sequence_index = [0]
        return cls(transforms, transform_sequence_index)


def _read_wsireg_transform(
    parameter_data: str | (Path | dict[str, list[str]]),
    first: bool = False,
    skip_initial: bool = False,
) -> tuple[list[dict[str, list[str]]], list[int]]:
    """Convert wsireg transform dict or from file to List of Transforms."""
    transforms = parameter_data
    if isinstance(parameter_data, (str, Path)):
        transforms: dict[str, list[str]] = read_json_data(parameter_data)

    allowed_n = 1 if first else -1
    index = 0
    transform_list = []
    transform_sequence_index = []
    for key, value in transforms.items():
        if "initial" in key and skip_initial:
            continue
        if "initial" not in key and first:
            allowed_n -= 1
        if key == "initial":
            if isinstance(value, dict):
                transform_list.append(value)
                transform_sequence_index.append(index)
                index += 1
            elif isinstance(value, list):
                for init_tform in value:
                    transform_list.append(init_tform)
                    transform_sequence_index.append(index)
                    index += 1
        else:
            if isinstance(value, dict):
                transform_list.append(value)
                transform_sequence_index.append(index)
                index += 1
            elif isinstance(value, list):
                for tform in value:
                    transform_list.append(tform)
                    transform_sequence_index.append(index)
                index += 1
        if allowed_n == 0 and first:
            break
    return transform_list, transform_sequence_index

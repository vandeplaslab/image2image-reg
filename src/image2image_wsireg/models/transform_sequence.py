"""Transformation sequence."""
from __future__ import annotations

import json
import typing as ty
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from koyo.typing import PathLike

from image2image_wsireg.enums import ELX_TO_ITK_INTERPOLATORS
from image2image_wsireg.models.transform import Transform


class TransformSequence:
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
        transforms: str | (Path | list[dict[str, list[str]]] | Transform | list[Transform]) | None = None,
        transform_sequence_index: list[int] | None = None,
    ) -> None:
        self._transform_sequence_index: list[int] = []
        self._output_size: tuple[int, int] | None = None
        self._output_spacing: tuple[float, float] | tuple[int, int] = (1, 1)
        self.transforms: list[Transform] = []
        self.resampler: sitk.ResampleImageFilter | None = None
        self.composed_linear_matrix_map: dict[str, np.ndarray] | None = None
        self.transform_itk_order: list[Transform] = []

        if transforms:
            self.add_transforms(transforms, transform_sequence_index=transform_sequence_index)
        else:
            self._composite_transform = None
            self._n_transforms = 0

    def __call__(self, image: sitk.Image) -> sitk.Image:
        """Transform."""
        return self.resampler.Execute(image)

    def add_transforms(
        self,
        transforms: str | (Path | (dict | (list[Transform] | Transform))),
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
        if isinstance(transforms, (str, Path, dict)):
            tform_list, tform_idx = _read_wsireg_transform(transforms)
            self.transform_sequence_index = tform_idx
            reg_transforms = [Transform(t) for t in tform_list]
            self.transforms = self.transforms + reg_transforms

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
    def composite_transform(self, transforms):
        self._composite_transform = transforms

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

    @property
    def n_transforms(self) -> int:
        """Number of transformations in sequence."""
        return self._n_transforms

    @n_transforms.setter
    def n_transforms(self, _: ty.Any = None) -> None:
        self._n_transforms = len(self.transforms)

    @property
    def output_size(self) -> tuple[int, int] | None:
        """Output size of image resampled by transform.

        Initially determined from the last transformation in the chain.
        """
        return self._output_size

    @output_size.setter
    def output_size(self, new_size: tuple[int, int]) -> None:
        """Set output size of image resampled by transform."""
        self._output_size = new_size

    @property
    def output_spacing(self) -> tuple[float, float] | tuple[int, int]:
        """Output spacing of image resampled by transform, initially determined from the last
        transformation in the chain.
        """
        return self._output_spacing

    @output_spacing.setter
    def output_spacing(self, new_spacing: tuple[float, float] | tuple[int, int]) -> None:
        """Set output spacing of image resampled by transform."""
        self._output_spacing = new_spacing

    def set_output_spacing(self, spacing: tuple[float, float] | tuple[int, int]) -> None:
        """
        Method that allows setting the output spacing of the resampler
        to resampled to any pixel spacing desired. This will also change the output_size
        to match.

        Parameters
        ----------
        spacing: tuple of float
            Spacing to set the new image. Will also change the output size to match.

        """
        output_size_scaling = np.asarray(self._output_spacing) / np.asarray(spacing)
        new_output_size = np.ceil(np.multiply(self._output_size, output_size_scaling))
        new_output_size: tuple[int, int] = tuple([int(i) for i in new_output_size])

        self._output_spacing = spacing
        self._output_size = new_output_size

        self._build_resampler()

    def _update_transform_properties(self) -> None:
        self._output_size = self.transforms[-1].output_size
        self._output_spacing = self.transforms[-1].output_spacing
        self._build_transform_data()

    def _build_transform_data(self) -> None:
        self._build_composite_transform(self.transforms, self.transform_sequence_index)
        self._build_resampler()

    def _build_composite_transform(self, reg_transforms: list[Transform], reg_transform_seq_idx: list[int]) -> None:
        """Build composite transform from list of transforms."""
        composite_index = []
        for unique_idx in np.unique(reg_transform_seq_idx):
            in_seq_tform_idx = np.where(reg_transform_seq_idx == unique_idx)[0]
            if len(in_seq_tform_idx) > 1:
                composite_index = composite_index + list(in_seq_tform_idx[::-1])
            else:
                composite_index = composite_index + list(in_seq_tform_idx)

        composite_transform = sitk.CompositeTransform(2)
        for transform_index in composite_index:
            composite_transform.AddTransform(reg_transforms[transform_index].itk_transform)

        self._composite_transform = composite_transform
        self.transform_itk_order = [self.transforms[i] for i in composite_index]

    def _build_resampler(self) -> None:
        """Build resampler."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputOrigin(self.transforms[-1].output_origin)
        resampler.SetOutputDirection(self.transforms[-1].output_direction)
        resampler.SetSize(self.output_size)
        resampler.SetOutputSpacing(self.output_spacing)

        interpolator = ELX_TO_ITK_INTERPOLATORS.get(self.transforms[-1].resample_interpolator)
        resampler.SetInterpolator(interpolator)
        resampler.SetTransform(self.composite_transform)
        self.resampler = resampler

    def transform_points(self, pt_data: np.ndarray, point_index=True, source_res=1, output_idx=True) -> np.ndarray:
        """
        Transform point sets using the transformation chain.

        Parameters
        ----------
        pt_data: np.ndarray
            Point data in xy order
        point_index: bool
            Whether point data is in pixel or physical coordinate space
        source_res: float
            spacing of the pixels associated with pt_data if they are not in physical coordinate space
        output_idx: bool
            return transformed points to pixel indices in the output_spacing's reference space.


        Returns
        -------
        transformed_points: np.ndarray
            Transformed points
        """
        transformed_points = []
        for point in pt_data:
            if point_index is True:
                point = point * source_res
            for _index, transform in enumerate(self.transforms):
                point = transform.inverse_transform.TransformPoint(point)
            transformed_point = np.array(point)

            if output_idx is True:
                transformed_point *= 1 / self._output_spacing[0]
            transformed_points.append(transformed_point)
        return np.stack(transformed_points)

    def append(self, other: TransformSequence) -> None:
        """
        Concatenate transformation sequences.

        Parameters
        ----------
        other: TransformSequence
            Append a TransformSeq to another

        """
        self.add_transforms(other.transforms, other.transform_sequence_index)

    @classmethod
    def from_path(cls, path: PathLike) -> TransformSequence:
        """Load transform sequence from path."""
        transforms, transform_sequence_index = _read_wsireg_transform(path)
        return cls(transforms, transform_sequence_index)


def _read_wsireg_transform(
    parameter_data: str | (Path | dict[ty.Any, ty.Any])
) -> tuple[list[dict[str, list[str]]], list[int]]:
    """Convert wsireg transform dict or from file to List of Transforms."""
    if isinstance(parameter_data, (str, Path)):
        parameter_data_in = json.load(open(parameter_data))
    else:
        parameter_data_in = parameter_data

    transform_list = []
    transform_sequence_index = []

    index = 0
    for key, value in parameter_data_in.items():
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
    return transform_list, transform_sequence_index

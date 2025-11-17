"""Test models/transform_sequence.py."""

from image2image_reg.elastix.transform_sequence import TransformSequence
from image2image_reg.utils._test import get_test_files


def test_transform_sequence_one_dataset_no_initial():
    files = get_test_files("no_initial_n=2_transformations.json")

    for file in files:
        transform_seq = TransformSequence.from_path(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."
        assert len(transform_seq.transforms) == 2, "No transforms found."
        assert transform_seq.output_size is not None, "Output size should not be None."
        assert transform_seq.output_spacing is not None, "Output spacing should not be None."

        transform_seq = TransformSequence(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."


def test_transform_sequence_from_file():
    files = get_test_files("complex_linear_reg_transform.json")

    for file in files:
        transform_seq = TransformSequence.from_path(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."
        assert len(transform_seq.transforms) == 3, "No transforms found."
        assert transform_seq.output_size is not None, "Output size should not be None."
        assert transform_seq.output_spacing is not None, "Output spacing should not be None."

        transform_seq = TransformSequence(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."

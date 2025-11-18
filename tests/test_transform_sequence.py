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


def test_transform_sequence_from_file(tmp_path):
    files = get_test_files("complex_linear_reg_transform.json")

    for file in files:
        transform_seq = TransformSequence.from_path(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."
        assert len(transform_seq.transforms) == 3, "No transforms found."
        assert transform_seq.n_transforms == 3, "n_transforms incorrect."
        assert transform_seq.output_size is not None, "Output size should not be None."
        assert transform_seq.output_spacing is not None, "Output spacing should not be None."

        transform_seq = TransformSequence(file)
        assert isinstance(transform_seq, TransformSequence), "Not a TransformSequence."

        # extract a single transform
        ts = transform_seq.extract_to_ts(0)
        assert isinstance(ts, TransformSequence), "Extracted transform is not a TransformSequence."

        # create transformation objects
        assert ts.composite_transform is not None, "Composite transform is None."
        assert ts.final_transform is not None, "Final transform is None."
        assert ts.inverse_transform is not None, "Inverse transform is None."
        assert ts.reverse_final_transform is not None, "Reverse final transform is None."

        # combine multiple transforms
        ts1 = transform_seq.extract_to_ts(0)
        assert ts1.n_transforms == 1, "Extracted transform should have 1 transform."
        ts2 = transform_seq.extract_to_ts(1)
        assert ts2.n_transforms == 1, "Extracted transform should have 1 transform."
        ts1.insert(ts2)
        assert ts1.n_transforms == 2, "Inserted transform should have 2 transforms."

        # export files
        out = transform_seq.to_dict()
        assert isinstance(out, list), "Extracted transform is not a dict."
        assert len(out) == 3, "Extracted transform should have 3 transforms."
        assert isinstance(out[0], dict), "Extracted transform is not a dict."

        path = transform_seq.to_json(tmp_path / "transform_seq.json")
        assert path.is_file(), "TransformSequence JSON file not created."
        assert path.exists(), "TransformSequence JSON file not created."

        path = transform_seq.to_gzip_json(tmp_path / "transform_seq.json.gz")
        assert path.is_file(), "TransformSequence GZIP JSON file not created."
        assert path.exists(), "TransformSequence GZIP JSON file not created."

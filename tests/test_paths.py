"""Test portable path serialization."""

from pathlib import Path

from image2image_reg.models import Preprocessing
from image2image_reg.models.paths import load_path_roots, resolve_path, serialize_path
from image2image_reg.workflows import ElastixReg


def test_serialize_project_relative_path(tmp_path: Path):
    project_dir = tmp_path / "project.wsireg"
    image_path = project_dir / "slides" / "image.tif"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")

    data = serialize_path(image_path, project_dir=project_dir)

    assert data == {"path_type": "relative", "base": "project", "path": "slides/image.tif"}
    assert resolve_path(data, project_dir=project_dir) == image_path


def test_serialize_windows_root_path(tmp_path: Path):
    data = serialize_path("Z:\\lab\\slides\\image.tif", path_roots={"dataset": "Z:/lab"})

    assert data == {"path_type": "root", "root": "dataset", "path": "slides/image.tif"}
    assert resolve_path(data, path_roots={"dataset": tmp_path / "lab"}) == tmp_path / "lab" / "slides" / "image.tif"


def test_load_path_roots_from_environment(monkeypatch):
    monkeypatch.setenv("I2REG_PATH_ROOTS", '{"dataset": "/mnt/dataset"}')

    assert load_path_roots() == {"dataset": "/mnt/dataset"}


def test_preprocessing_serializes_portable_mask_path(tmp_path: Path):
    project_dir = tmp_path / "project.wsireg"
    mask_path = project_dir / "masks" / "mask.tif"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"")

    data = Preprocessing(mask=mask_path).to_dict(project_dir=project_dir)

    assert data["mask"] == {"path_type": "relative", "base": "project", "path": "masks/mask.tif"}


def test_elastix_serializes_attachment_root_paths(tmp_path: Path):
    root_dir = tmp_path / "dataset"
    shape_path = root_dir / "shapes" / "roi.geojson"
    path_roots = {"dataset": root_dir}

    obj = ElastixReg(name="test.wsireg", output_dir=tmp_path, path_roots=path_roots)
    obj.attachment_shapes["roi"] = {"files": [shape_path], "pixel_size": 1.0, "attach_to": "source"}

    config = obj._get_config()

    assert config["attachment_shapes"]["roi"]["files"][0] == {
        "path_type": "root",
        "root": "dataset",
        "path": "shapes/roi.geojson",
    }

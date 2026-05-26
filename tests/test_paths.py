"""Test portable path serialization."""

import json
from pathlib import Path

from image2image_reg.models import Preprocessing
from image2image_reg.models.paths import load_path_roots, resolve_path, serialize_path
from image2image_reg.workflows import ElastixReg, ValisReg


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


def test_elastix_loads_legacy_absolute_paths_and_save_upgrades(tmp_path: Path):
    project_dir = tmp_path / "test.wsireg"
    image_path = project_dir / "slides" / "source.tiff"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"")
    _write_json(project_dir / ElastixReg.CONFIG_NAME, _elastix_config(str(image_path)))

    obj = ElastixReg.from_path(project_dir, raise_on_error=False)

    assert obj.modalities["source"].path == image_path

    obj.save()
    config = _read_json(project_dir / ElastixReg.CONFIG_NAME)
    assert config["modalities"]["source"]["path"] == {
        "path_type": "relative",
        "base": "project",
        "path": "slides/source.tiff",
    }


def test_elastix_migrates_legacy_paths_without_loading_files(tmp_path: Path):
    project_dir = tmp_path / "test.wsireg"
    source_path = "/old/data/slides/source.tiff"
    mask_path = "/old/data/masks/source-mask.tiff"
    modality_mask_path = "/old/data/masks/modality-mask.tiff"
    reg_mask_path = "/old/data/masks/reg-mask.tiff"
    shape_path = "/old/data/shapes/roi.geojson"
    points_path = "/old/data/points/roi.csv"
    _write_json(
        project_dir / ElastixReg.CONFIG_NAME,
        _elastix_config(
            source_path,
            modality_mask=modality_mask_path,
            preprocessing={"mask": mask_path},
            registration_paths={
                "reg_path_0": {
                    "source": "source",
                    "target": "source",
                    "through": None,
                    "reg_params": ["rigid"],
                    "source_preprocessing": {"mask": reg_mask_path},
                    "target_preprocessing": None,
                },
            },
            registration_graph_edges=[
                {
                    "modalities": {"source": "source", "target": "source"},
                    "params": ["rigid"],
                    "registered": False,
                    "transform_tag": None,
                    "source_preprocessing": {"mask": reg_mask_path},
                    "target_preprocessing": None,
                },
            ],
            attachment_shapes={
                "roi": {"shape_files": [shape_path], "pixel_size": 1.0, "attach_to": "source"},
            },
            attachment_points={
                "points": {"point_files": [points_path], "pixel_size": 1.0, "attach_to": "source"},
            },
        ),
    )

    updated_paths = ElastixReg.migrate_paths(project_dir, path_roots={"dataset": "/old/data"}, backup=False)

    assert updated_paths == [project_dir / ElastixReg.CONFIG_NAME]
    config = _read_json(project_dir / ElastixReg.CONFIG_NAME)
    assert config["modalities"]["source"]["path"] == _root_path("slides/source.tiff")
    assert config["modalities"]["source"]["mask"] == _root_path("masks/modality-mask.tiff")
    assert config["modalities"]["source"]["preprocessing"]["mask"] == _root_path("masks/source-mask.tiff")
    assert config["registration_paths"]["reg_path_0"]["source_preprocessing"]["mask"] == _root_path(
        "masks/reg-mask.tiff",
    )
    assert config["registration_graph_edges"][0]["source_preprocessing"]["mask"] == _root_path("masks/reg-mask.tiff")
    assert config["attachment_shapes"]["roi"]["files"] == [_root_path("shapes/roi.geojson")]
    assert "shape_files" not in config["attachment_shapes"]["roi"]
    assert config["attachment_points"]["points"]["files"] == [_root_path("points/roi.csv")]
    assert "point_files" not in config["attachment_points"]["points"]


def test_path_migration_is_idempotent(tmp_path: Path):
    project_dir = tmp_path / "test.wsireg"
    config = _elastix_config({"path_type": "root", "root": "dataset", "path": "slides/source.tiff"})
    _write_json(project_dir / ElastixReg.CONFIG_NAME, config)

    updated_paths = ElastixReg.migrate_paths(project_dir, path_roots={"dataset": "/old/data"}, backup=False)

    assert updated_paths == []
    assert _read_json(project_dir / ElastixReg.CONFIG_NAME) == config


def test_valis_migrates_legacy_project_paths(tmp_path: Path):
    project_dir = tmp_path / "test.valis"
    _write_json(project_dir / ValisReg.CONFIG_NAME, _valis_config("/old/data/slides/source.tiff"))

    updated_paths = ValisReg.migrate_paths(project_dir, path_roots={"dataset": "/old/data"}, backup=False)

    assert updated_paths == [project_dir / ValisReg.CONFIG_NAME]
    config = _read_json(project_dir / ValisReg.CONFIG_NAME)
    assert config["modalities"]["source"]["path"] == _root_path("slides/source.tiff")


def _elastix_config(
    source_path,
    modality_mask: str | None = None,
    preprocessing: dict | None = None,
    registration_paths: dict | None = None,
    registration_graph_edges: list[dict] | None = None,
    attachment_shapes: dict | None = None,
    attachment_points: dict | None = None,
) -> dict:
    return {
        "schema_version": "1.2",
        "name": "test",
        "cache_images": True,
        "pairwise": False,
        "modalities": {
            "source": {
                "name": "source",
                "path": source_path,
                "mask": modality_mask,
                "preprocessing": preprocessing,
                "pixel_size": 1.0,
            },
        },
        "registration_paths": registration_paths or {},
        "registration_graph_edges": registration_graph_edges,
        "original_size_transforms": None,
        "attachment_shapes": attachment_shapes,
        "attachment_points": attachment_points,
        "attachment_images": None,
        "merge": False,
        "merge_images": None,
    }


def _valis_config(source_path) -> dict:
    return {
        "schema_version": "1.1",
        "name": "test",
        "cache_images": True,
        "reference": "source",
        "check_for_reflections": False,
        "non_rigid_registration": False,
        "micro_registration": True,
        "micro_registration_fraction": 0.125,
        "feature_detector": "sensitive_vgg",
        "feature_matcher": "RANSAC",
        "modalities": {
            "source": {
                "name": "source",
                "path": source_path,
                "pixel_size": 1.0,
            },
        },
        "attachment_shapes": None,
        "attachment_points": None,
        "attachment_images": None,
        "merge": False,
        "merge_images": None,
    }


def _root_path(path: str) -> dict[str, str]:
    return {"path_type": "root", "root": "dataset", "path": path}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))

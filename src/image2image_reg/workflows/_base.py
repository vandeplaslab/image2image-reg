"""Base class for all workflows."""

from __future__ import annotations

import time
import typing as ty
from copy import deepcopy
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import clean_path
from loguru import logger

from image2image_reg._typing import AttachedShapeOrPointDict
from image2image_reg.models import Export, Modality, Preprocessing
from image2image_reg.models.bbox import BoundingBox, Polygon, _transform_to_bbox, _transform_to_polygon

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader

    from image2image_reg.wrapper import ImageWrapper


class Workflow:
    """Base class for all workflows."""

    CONFIG_NAME: str
    EXTENSION: str

    log_file: Path | None = None
    _name: str | None = None

    def __init__(
        self,
        name: str | None = None,
        output_dir: PathLike | None = None,
        project_dir: PathLike | None = None,
        cache: bool = True,
        merge: bool = False,
        log: bool = False,
        init: bool = True,
        **_kwargs: ty.Any,
    ):
        # setup project directory
        if project_dir:
            project_dir = Path(project_dir)
            if not project_dir.exists():
                project_dir = project_dir.with_suffix(self.EXTENSION)
            self.project_dir = project_dir.resolve()
            name = self.project_dir.stem
        else:
            if name is None:
                raise ValueError("Name must be provided.")
            if output_dir is None:
                raise ValueError("Output directory must be provided.")
            self.project_dir = (Path(output_dir) / self.format_project_name(name)).with_suffix(self.EXTENSION).resolve()
        if not self.project_dir.suffix == self.EXTENSION:
            logger.warning(f"Project directory '{self.project_dir}' does not have the correct extension but that's ok.")
        self.name = self.format_project_name(name)
        self.cache_images = cache
        self.merge_images = merge

        # setup modalities
        self.modalities: dict[str, Modality] = {}
        self.attachment_images: dict[str, str] = {}
        self.attachment_shapes: dict[str, AttachedShapeOrPointDict] = {}
        self.attachment_points: dict[str, AttachedShapeOrPointDict] = {}
        self.merge_modalities: dict[str, list[str]] = {}  # TODO

        # setup cache directory
        if init:
            self.project_dir.mkdir(exist_ok=True, parents=True)
        if log and init:
            self.set_logger()

    @property
    def cache_dir(self) -> Path:
        """Return cache directory."""
        cache_dir = self.project_dir / "Cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        return cache_dir

    @property
    def is_registered(self) -> bool:
        """Check if the project has been registered."""
        raise NotImplementedError("Must implement method")

    @classmethod
    def from_path(cls, path: PathLike, raise_on_error: bool = True) -> Workflow:
        """Initialize based on the project path."""
        raise NotImplementedError("Must implement method")

    def print_summary(self, func: ty.Callable = logger.info) -> None:
        """Print summary about the project."""
        raise NotImplementedError("Must implement method")

    def validate(self, allow_not_registered: bool = True, require_paths: bool = False) -> tuple[bool, list[str]]:
        """Perform several checks on the project."""
        raise NotImplementedError("Must implement method")

    def preview(self, **kwargs: ty.Any) -> None:
        """Preview registration."""

    def register(self, **kwargs: ty.Any) -> None:
        """Co-register images."""
        raise NotImplementedError("Must implement method")

    def write(self, **kwargs: ty.Any) -> list[Path] | None:
        """Export images after applying transformation."""
        raise NotImplementedError("Must implement method")

    def _get_config(self, **kwargs: ty.Any) -> dict:
        raise NotImplementedError("Must implement method")

    def save(self, **kwargs: ty.Any) -> Path:
        """Save configuration to file."""
        raise NotImplementedError("Must implement method")

    def clear(self, **kwargs: ty.Any) -> None:
        """Clear all data from the project."""
        raise NotImplementedError("Must implement method")

    @classmethod
    def format_project_name(cls, name: str) -> str:
        """Format project name so that it's properly formatted."""
        if cls.EXTENSION in name:
            name = name.replace(cls.EXTENSION, "")
        return name

    @property
    def name(self) -> str:
        """Project name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if self._name == value:
            return
        # if self.project_dir.exists():
        #     raise ValueError("Project directory already exists - cannot edit project name.")
        self._name = value

    @property
    def project_name(self) -> str:
        """Project name."""
        return self.project_dir.name

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        return self.project_dir.parent

    @output_dir.setter
    def output_dir(self, value: PathLike) -> None:
        value = (Path(value) / self.name).with_suffix(self.EXTENSION).resolve()
        if self.project_dir == value:
            return
        # if self.project_dir.exists():
        #     raise ValueError("Project directory already exists - cannot edit project name.")
        self.project_dir = value

    @property
    def log_dir(self) -> Path:
        """Log directory."""
        directory = self.project_dir / "Logs"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def progress_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "Progress"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def image_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "Images"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    def set_logger(self) -> None:
        """Setup logger."""
        import sys

        from koyo.logging import get_loguru_env, set_loguru_log

        from image2image_reg.utils.utilities import print_versions

        if self.log_file is None:
            n = len(list(self.log_dir.glob("*_log.txt")))
            ts = time.strftime("%Y%m%d-%H%M%S")
            self.log_file = self.log_dir / f"{str(n).zfill(3)}_{ts}_log.txt"
            _, level, enqueue, _ = get_loguru_env()
            set_loguru_log(
                self.log_file,
                level=level,
                enqueue=enqueue,
                colorize=False,
                no_color=True,
                catch=True,
                diagnose=True,
                logger=logger,
                remove=False,
            )
            [logger.enable(module) for module in ("image2image_io", "image2image_reg", "koyo")]  # type: ignore
            logger.info(f"Setup logging to file - '{self.log_file!s}'")
            logger.trace(f"Executed command: {sys.argv}")
            print_versions()

    def get_modality(
        self, name: str | None = None, path: PathLike | None = None, name_or_path: PathLike | None = None
    ) -> Modality | None:
        """Get modality."""
        for modality in self.modalities.values():
            if name and modality.name == str(name):
                return modality
            if path and modality.path == Path(path):
                return modality
            if name_or_path and (modality.name == str(name_or_path) or modality.path == Path(name_or_path)):
                return modality
        return None

    def get_wrapper(
        self, name: str | None = None, path: PathLike | None = None, name_or_path: PathLike | None = None
    ) -> ImageWrapper | None:
        """Get modality."""
        from image2image_reg.wrapper import ImageWrapper

        modality = self.get_modality(name, path, name_or_path)
        if modality:
            return ImageWrapper(modality)

    def has_modality(
        self, name: str | None = None, path: PathLike | None = None, name_or_path: PathLike | None = None
    ) -> bool:
        """Check whether modality has been previously added."""
        modality = self.get_modality(name, path, name_or_path)
        return modality is not None

    def rename_modality(self, old_name: str, new_name: str) -> None:
        """Rename modality."""
        if old_name not in self.modalities:
            raise ValueError(f"Modality '{old_name}' does not exist.")
        if new_name in self.modalities:
            raise ValueError(f"Modality '{new_name}' already exists.")
        # rename modalities
        self.modalities[old_name].name = new_name
        self.modalities[new_name] = self.modalities.pop(old_name)
        # rename attachment images
        attachment_images = deepcopy(self.attachment_images)
        for name, attach_to in self.attachment_images.items():
            if attach_to == old_name:
                attachment_images[name] = new_name
            if name == old_name:
                attachment_images[new_name] = attachment_images.pop(old_name)
        self.attachment_images = attachment_images
        # rename attachment shapes
        for _, shape_dict in self.attachment_shapes.items():
            if shape_dict["attach_to"] == old_name:
                shape_dict["attach_to"] = new_name
        # rename attachment points
        for _, points_dict in self.attachment_points.items():
            if points_dict["attach_to"] == old_name:
                points_dict["attach_to"] = new_name
        logger.trace(f"Renamed modality '{old_name}' to '{new_name}'.")

    def is_attachment(self, modality: str) -> bool:
        """Check if modality is an attachment."""
        return modality in self.attachment_images

    def has_attachments(self, name: str | None = None) -> bool:
        """Return True if there are any attachments."""
        if name:
            return (
                self.get_attachment_list(name, "image")
                or self.get_attachment_list(name, "geojson")
                or self.get_attachment_list(name, "points")
            )
        return bool(self.attachment_images or self.attachment_shapes or self.attachment_points)

    def get_image_modalities(self, with_attachment: bool = True) -> list[str]:
        """Return list of image modalities."""
        images = []
        for modality in self.modalities.values():
            if not with_attachment and self.is_attachment(modality.name):
                continue
            images.append(modality.name)
        return images

    def get_attachment_list(self, attach_to: str, kind: ty.Literal["image", "geojson", "points"]) -> list[str]:
        """Return list of attachment(s)."""
        if kind == "image":
            return [name for name, attach_to_ in self.attachment_images.items() if attach_to_ == attach_to]
        if kind == "geojson":
            return [name for name, attach_to_ in self.attachment_shapes.items() if attach_to_["attach_to"] == attach_to]
        if kind == "points":
            return [name for name, attach_to_ in self.attachment_points.items() if attach_to_["attach_to"] == attach_to]
        raise ValueError(f"Invalid kind '{kind}' - expected 'image', 'geojson', 'points'.")

    def get_attachment_count(self, attach_to: str, kind: ty.Literal["image", "geojson", "points"] | str) -> int:
        """Get number of attachments of a certain kind fora specified attachment image."""
        return len(self.get_attachment_list(attach_to, kind))

    def auto_add_modality(
        self,
        name: str,
        path: PathLike,
        preprocessing: Preprocessing | None = None,
        mask: PathLike | None = None,
        mask_bbox: tuple[int, int, int, int] | None = None,
        export: Export | dict[str, ty.Any] | None = None,
        method: str | None = None,
        overwrite: bool = False,
    ) -> Modality:
        """Add modality."""
        from image2image_io.readers import get_simple_reader, is_supported

        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if name in self.modalities and not overwrite:
            raise ValueError(f"Modality name '{name}' already exists.")
        reader: BaseReader = get_simple_reader(path, init_pyramid=False)
        if preprocessing:
            preprocessing.channel_names = reader.channel_names
            preprocessing.channel_indices = reader.channel_ids
            preprocessing.mask = mask
            preprocessing.mask_bbox = mask_bbox
            preprocessing.method = method

        return self.add_modality(
            name,
            path,
            pixel_size=reader.resolution or 1.0,
            channel_names=reader.channel_names,
            preprocessing=preprocessing,
            export=export,
            reader_kws=reader.reader_kws,
            overwrite=overwrite,
        )

    def add_modality(
        self,
        name: str,
        path: PathLike,
        pixel_size: float = 1.0,
        channel_names: list[str] | None = None,
        channel_colors: list[str] | None = None,
        preprocessing: Preprocessing | dict[str, ty.Any] | None = None,
        transform_mask: bool = True,
        mask: PathLike | np.ndarray | None = None,
        mask_bbox: tuple[int, int, int, int] | BoundingBox | None = None,
        mask_polygon: np.ndarray | Polygon | None = None,
        output_pixel_size: tuple[float, float] | None = None,
        export: Export | dict[str, ty.Any] | None = None,
        reader_kws: dict[str, ty.Any] | None = None,
        method: str | None = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
    ) -> Modality:
        """Add modality."""
        from image2image_io.readers import is_supported

        if "initial" in name:
            raise ValueError("Sorry, the word 'initial' cannot be used in the modality name as it's reserved.")

        path = Path(path)
        if not path.exists() and raise_on_error:
            raise ValueError("Path does not exist.")
        if not is_supported(path, raise_on_error):
            raise ValueError("Unsupported file format.")
        if pixel_size <= 0:
            raise ValueError("Pixel size must be greater than 0.")
        if name in self.modalities and not overwrite:
            raise ValueError(f"Modality '{name}' name already exists.")
        if mask is not None and mask_bbox is not None and mask_polygon is not None:
            raise ValueError("Mask can only be specified using one of the three options: mask, mask_bbox, mask_polygon")
        if isinstance(mask, (str, Path)):
            mask = Path(mask)
            if not mask.exists():
                raise ValueError("Mask path does not exist.")
        if mask_bbox is not None:
            mask_bbox = _transform_to_bbox(mask_bbox)
        if mask_polygon is not None:
            mask_polygon = _transform_to_polygon(mask_polygon)
        if isinstance(preprocessing, dict):
            preprocessing = Preprocessing(**preprocessing)
        if preprocessing:
            if preprocessing.mask is None and mask is not None:
                preprocessing.mask = mask
            if preprocessing.mask_bbox is None and mask_bbox is not None:
                preprocessing.mask_bbox = mask_bbox
            if preprocessing.mask_polygon is None and mask_polygon is not None:
                preprocessing.mask_polygon = mask_polygon
            if not preprocessing.transform_mask and transform_mask:
                preprocessing.transform_mask = transform_mask
            if preprocessing.method is None and method:
                preprocessing.method = method
        if isinstance(export, dict):
            export = Export(**export)
        if isinstance(output_pixel_size, (int, float)):
            output_pixel_size = (float(output_pixel_size), float(output_pixel_size))
        self.modalities[name] = Modality(
            name=name,
            path=path.resolve() if raise_on_error else path,
            pixel_size=pixel_size,
            channel_names=channel_names,
            channel_colors=channel_colors,
            preprocessing=preprocessing,
            output_pixel_size=output_pixel_size,
            export=export,
            reader_kws=reader_kws,
        )
        logger.trace(f"Added modality '{name}'.")
        return self.modalities[name]

    def remove_modality(self, name: str | None = None, path: PathLike | None = None) -> Modality | None:
        """Remove modality from the project."""
        modality = None
        if name is not None and name in self.modalities:
            modality = self.modalities.pop(name, None)
            logger.trace(f"Removed modality '{name}'.")
        elif path is not None:
            path = Path(path)
            for modality_ in self.modalities.values():
                if Path(modality_.path) == path:
                    modality = self.modalities.pop(modality_.name)
                    logger.trace(f"Removed modality '{modality.name}'.")
                    break
        # remove registration paths
        if modality:
            # remove attached images
            to_remove_attachment = []
            for attached, attach_to in self.attachment_images.items():
                if attach_to == modality.name:
                    self.remove_modality(name=attached, path=path)
                    to_remove_attachment.append(attached)
            [self.attachment_images.pop(name) for name in to_remove_attachment]
            # remove attached shapes
            to_remove_shapes = []
            for name, shapes_dict in self.attachment_shapes.items():
                if shapes_dict["attach_to"] == modality.name:
                    to_remove_shapes.append(name)
            [self.attachment_shapes.pop(name) for name in to_remove_shapes]
            # remove attached points
            to_remove_points = []
            for name, points_dict in self.attachment_points.items():
                if points_dict["attach_to"] == modality.name:
                    to_remove_points.append(name)
            [self.attachment_points.pop(name) for name in to_remove_points]

            # sub-models handle this
            self._remove_modality(modality)
        if not modality:
            logger.warning("Could not find modality to remove.")
        return modality

    def _remove_modality(self, modality: Modality) -> None:
        """Remove modality, handled by sub-class."""

    def auto_add_attachment_images(self, attach_to_modality: str, name: str, path: PathLike) -> None:
        """Add modality."""
        from image2image_io.readers import get_simple_reader, is_supported

        if not path:
            if name not in self.modalities:
                raise ValueError(f"Modality '{name}' does not exist. Please add it first.")
            path = self.modalities[name].path

        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if attach_to_modality not in self.modalities:
            raise ValueError("Modality does not exist. Please add it before trying to add an attachment.")
        reader: BaseReader = get_simple_reader(path, init_pyramid=False)
        self.add_attachment_images(attach_to_modality, name, path, reader.resolution, reader.channel_names)

    def remove_attachment_image(self, name: str) -> None:
        """Remove attachment data."""
        if name not in self.attachment_images:
            raise ValueError(f"Attachment image with name '{name}' does not exist.")
        self.attachment_images.pop(name)
        self.modalities.pop(name)
        logger.trace(f"Removed attachment image '{name}'.")

    def add_attachment_images(
        self,
        attach_to_modality: str,
        name: str,
        path: PathLike,
        pixel_size: float = 1.0,
        channel_names: list[str] | None = None,
        channel_colors: list[str] | None = None,
    ) -> None:
        """Images which are unregistered between modalities.

        These are transformed following the path of one of the graph's modalities.

        Parameters
        ----------
        attach_to_modality : str
            image modality to which the new image is attached
        name : str
            name of the added attachment image
        path : str
            path to the attachment modality, it will be imported and transformed without preprocessing
        pixel_size : float
            spatial resolution of attachment image data in units per px (i.e. 0.9 um / px)
        channel_names: List[str]
            names for the channels to go into the OME-TIFF
        channel_colors: List[str]
            channels colors for OME-TIFF (not implemented)
        """
        if attach_to_modality not in self.modalities:
            raise ValueError(
                f"The specified modality '{attach_to_modality}' does not exist. Please add it before adding attachment"
                f" images."
            )
        path = Path(path)
        if self.has_modality(path=path):
            logger.warning(f"This image '{name}' is already present in the project.")
            return
        if not path.exists():
            raise ValueError(f"Path '{path}' does not exist.")
        self.add_modality(
            name,
            path.resolve(),
            pixel_size,
            channel_names=channel_names,
            channel_colors=channel_colors,
            overwrite=True,
        )
        self.attachment_images[name] = attach_to_modality
        logger.trace(f"Added attachment image '{name}' to '{attach_to_modality}'.")

    def auto_add_attachment_geojson(self, attach_to_modality: str, name: str, path: PathLike) -> None:
        """Add modality."""
        from image2image_io.readers import is_supported

        if not path:
            if name not in self.modalities:
                raise ValueError(f"Modality '{name}' does not exist. Please add it first.")
            path = self.modalities[name].path

        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if Path(path).suffix not in [".json", ".geojson"]:
            raise ValueError("Attachment must be in GeoJSON format.")
        if attach_to_modality not in self.modalities:
            raise ValueError("Modality does not exist. Please add it before trying to add an attachment.")
        self.add_attachment_geojson(attach_to_modality, name, path)

    def remove_attachment_geojson(self, name: str) -> None:
        """Remove attachment data."""
        if name not in self.attachment_shapes:
            raise ValueError(f"Shape set with name '{name}' does not exist.")
        self.attachment_shapes.pop(name)
        logger.trace(f"Removed shape set '{name}'.")

    def add_attachment_geojson(
        self, attach_to: str, name: str, paths: PathLike | list[PathLike], pixel_size: float | None = None
    ) -> None:
        """
        Add attached shapes.

        Parameters
        ----------
        attach_to : str
            image modality to which the shapes are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        pixel_size : float, optional
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        """
        if attach_to not in self.modalities:
            raise ValueError(
                f"The specified modality '{attach_to}' does not exist. Please add it before adding attachment images."
            )
        if isinstance(paths, (str, Path)):
            paths = [paths]

        paths_: list[Path] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path '{path}' does not exist.")
            if path.suffix not in [".json", ".geojson"]:
                raise ValueError("Attachment must be in GeoJSON format.")
            paths_.append(path.resolve())
        if pixel_size is None:
            pixel_size = self.modalities[attach_to].pixel_size
        self._add_geojson_set(attach_to, name, paths_, pixel_size)

    def _add_geojson_set(self, attach_to: str, name: str, paths: list[Path], pixel_size: float) -> None:
        """
        Add a shape set to the graph.

        Parameters
        ----------
        attach_to : str
            image modality to which the shapes are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        pixel_size : float
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        """
        if name in self.attachment_shapes:
            raise ValueError(f"Shape set with name '{name}' already exists. Please use a different name.")
        self.attachment_shapes[name] = {
            "files": [str(p) for p in paths],
            "pixel_size": pixel_size,
            "attach_to": attach_to,
        }
        logger.trace(
            f"Added shape set '{name}' with {len(paths)} files, pixel size of {pixel_size:.3f}"
            f" and attached to {attach_to}."
        )

    def auto_add_attachment_points(
        self, attach_to: str, name: str, path: PathLike, pixel_size: float | None = None
    ) -> None:
        """Add modality."""
        from image2image_io.readers import is_supported

        if not path:
            if name not in self.modalities:
                raise ValueError(f"Modality '{name}' does not exist. Please add it first.")
            path = self.modalities[name].path

        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if Path(path).suffix not in [".csv", ".txt", ".tsv", ".parquet"]:
            raise ValueError("Attachment must be in Points format.")
        if attach_to not in self.modalities:
            raise ValueError("Modality does not exist. Please add it before trying to add an attachment.")
        if pixel_size is None:
            pixel_size = self.modalities[attach_to].pixel_size
        self.add_attachment_points(attach_to, name, path, pixel_size)

    def remove_attachment_points(self, name: str) -> None:
        """Remove attachment data."""
        if name not in self.attachment_points:
            raise ValueError(f"Points set with name '{name}' does not exist.")
        self.attachment_points.pop(name)
        logger.trace(f"Removed points set '{name}'.")

    def add_attachment_points(
        self, attach_to: str, name: str, paths: PathLike | list[PathLike], pixel_size: float | None = None
    ) -> None:
        """
        Add attached shapes.

        Parameters
        ----------
        attach_to : str
            image modality to which the shapes are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        pixel_size : float
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        if attach_to not in self.modalities:
            raise ValueError(
                f"The specified modality '{attach_to}' does not exist. Please add it before adding attachment objects."
            )
        paths_: list[Path] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path '{path}' does not exist.")
            if Path(path).suffix not in [".csv", ".txt", ".tsv", ".parquet"]:
                raise ValueError("Attachment must be in Points format.")
            paths_.append(path.resolve())
        if pixel_size is None:
            pixel_size = self.modalities[attach_to].pixel_size
        self._add_points_set(attach_to, name, paths_, pixel_size)

    def _add_points_set(self, attach_to: str, name: str, paths: list[Path], pixel_size: float) -> None:
        """
        Add a shape set to the graph.

        Parameters
        ----------
        attach_to : str
            image modality to which the points are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        pixel_size : float
            spatial resolution of shape data's associated image in units per px (i.e. 0.9 um / px)
        """
        if name in self.attachment_points:
            raise ValueError(f"Shape set with name '{name}' already exists. Please use a different name.")
        self.attachment_points[name] = {
            "files": paths,
            "pixel_size": pixel_size,
            "attach_to": attach_to,
        }
        logger.trace(
            f"Added points set '{name}' with {len(paths)} files, pixel size of {pixel_size:.3f}"
            f" and attached to {attach_to}."
        )

    def auto_add_merge_modalities(self, name: str):
        """Add merge modalities."""
        modalities = list(self.modalities.keys())
        self.add_merge_modalities(name, modalities)

    def add_merge_modalities(self, name: str, modalities: list[str]) -> None:
        """Add merge modality."""
        for modality in modalities:
            if modality not in self.modalities:
                raise ValueError(f"Modality '{modality}' does not exist. Please add it first.")
        self.merge_modalities[name] = modalities
        logger.trace(f"Added merge modality '{name}'. Going to merge: {self.merge_modalities[name]}")

    def run(self) -> None:
        """Execute workflow."""
        self.set_logger()
        self.register()
        self.preview()
        self.write()

    def _load_modalities_from_config(self, config: dict, raise_on_error: bool = True) -> None:
        with MeasureTimer() as timer:
            for name, modality in config["modalities"].items():
                if not Path(modality["path"]).exists() and raise_on_error:
                    raise ValueError(f"Modality path '{modality['path']}' does not exist.")
                preprocessing = modality.get("preprocessing", {})
                self.add_modality(
                    name=name,
                    path=modality["path"],
                    preprocessing=Preprocessing(**preprocessing) if preprocessing else None,
                    channel_names=modality.get("channel_names", None),
                    channel_colors=modality.get("channel_colors", None),
                    mask=modality.get("mask", None),
                    mask_bbox=modality.get("mask_bbox", None),
                    mask_polygon=modality.get("mask_polygon", None),
                    output_pixel_size=modality.get("output_pixel_size", None),
                    pixel_size=modality.get("pixel_size", 1.0),
                    transform_mask=preprocessing.get("transform_mask", False),
                    export=Export(**modality["export"]) if modality.get("export") else None,
                    raise_on_error=raise_on_error,
                )
            logger.trace(f"Loaded modalities in {timer()}")

    def _load_attachment_from_config(self, config: dict) -> None:
        # load attachment images
        with MeasureTimer() as timer:
            if config.get("attachment_images"):
                for name, attach_to in config["attachment_images"].items():
                    self.attachment_images[name] = attach_to
                    logger.trace(f"Added attachment image '{name}' attached to '{attach_to}'")
                logger.trace(f"Loaded attachment images in {timer(since_last=True)}")

            if config.get("attachment_shapes"):
                for name, shape_dict in config["attachment_shapes"].items():
                    if "shape_files" in shape_dict:
                        shape_dict["files"] = shape_dict.pop("shape_files")
                    assert "files" in shape_dict, "Shape dict missing 'files' key."
                    assert "pixel_size" in shape_dict, "Shape dict missing 'pixel_size' key."
                    if "attach_to_modality" in shape_dict:
                        shape_dict["attach_to"] = shape_dict.pop("attach_to_modality")
                    assert "attach_to" in shape_dict, "Shape dict missing 'attach_to' key."
                    self.attachment_shapes[name] = shape_dict
                logger.trace(f"Loaded attachment shapes in {timer(since_last=True)}")

            if config.get("attachment_points"):
                for name, shape_dict in config["attachment_points"].items():
                    if "point_files" in shape_dict:
                        shape_dict["files"] = shape_dict.pop("point_files")
                    assert "files" in shape_dict, "Shape dict missing 'files' key."
                    assert "pixel_size" in shape_dict, "Shape dict missing 'pixel_size' key."
                    if "attach_to_modality" in shape_dict:
                        shape_dict["attach_to"] = shape_dict.pop("attach_to_modality")
                    assert "attach_to" in shape_dict, "Shape dict missing 'attach_to' key."
                    self.attachment_points[name] = shape_dict
                logger.trace(f"Loaded attachment points in {timer(since_last=True)}")

    def _load_merge_from_config(self, config: dict) -> None:
        # load merge modalities
        with MeasureTimer() as timer:
            if config["merge_images"]:
                for name, merge_modalities in config["merge_images"].items():
                    self.merge_modalities[name] = merge_modalities
                logger.trace(f"Loaded merge modalities in {timer(since_last=True)}")

    @classmethod
    def read_config(cls, path: PathLike) -> dict:
        """Read config without instantiating class."""
        from koyo.json import read_json_data

        path = Path(path)
        if path.is_dir() and path.suffix in [cls.EXTENSION]:
            path = path / cls.CONFIG_NAME
        elif path.is_file() and path.name == cls.CONFIG_NAME:
            path = path.parent / cls.CONFIG_NAME
        if not path.exists():
            raise ValueError(f"Could not find config file at '{path}'.")
        return read_json_data(path)

    @classmethod
    def write_config(cls, path: PathLike, config: dict) -> None:
        """Write config without instantiating class."""
        from koyo.json import write_json_data

        path = Path(path)
        if path.is_dir() and path.suffix in [cls.EXTENSION]:
            path = path / cls.CONFIG_NAME
        elif path.is_file() and path.name == cls.CONFIG_NAME:
            path = path.parent / cls.CONFIG_NAME
        write_json_data(path, config)

    @classmethod
    def update_paths(cls, path: PathLike, source_dirs: PathLike | list[PathLike], recursive: bool = False) -> None:
        """Update source paths."""
        if isinstance(source_dirs, (str, Path)):
            source_dirs = [source_dirs]

        source_dirs = [Path(source_dir) for source_dir in source_dirs]
        config = cls.read_config(path)
        config["modalities"] = cls._update_modality_paths(config["modalities"], source_dirs, recursive)
        config["attachment_shapes"] = cls._update_attachment_paths(config["attachment_shapes"], source_dirs, recursive)
        config["attachment_points"] = cls._update_attachment_paths(config["attachment_points"], source_dirs, recursive)
        cls.write_config(path, config)

    @staticmethod
    def _update_modality_paths(config: dict[str, dict], source_dirs: list[Path], recursive: bool = False) -> dict:
        for modality in config.values():
            name = modality["name"]
            path = clean_path(modality["path"])
            if not path.exists():
                logger.trace(f"Path '{path}' does not exist for modality={name}.")
                for source_dir in source_dirs:
                    updated, new_path = _get_new_path(path, source_dir, recursive=recursive)
                    if updated:
                        modality["path"] = str(new_path)
                        logger.trace(f"Updated path for modality={name} to '{new_path}'.")
            else:
                logger.success(f"Path '{path}' exists for modality={name}.")
        return config

    @staticmethod
    def _update_attachment_paths(
        config: dict[str, AttachedShapeOrPointDict] | None, source_dirs: list[Path], recursive: bool = False
    ) -> AttachedShapeOrPointDict | None:
        if not config:
            return config
        for attachment in config.values():
            for i, path in enumerate(attachment["files"]):
                path = clean_path(path)
                if not path.exists():
                    logger.trace(f"Path '{path}' does not exist for attachment={attachment}.")
                    for source_dir in source_dirs:
                        updated, new_path = _get_new_path(path, source_dir, recursive=recursive)
                        if new_path.exists():
                            attachment["files"][i] = str(new_path)
                            logger.trace(f"Updated path for attachment={attachment} to '{new_path}'.")
                else:
                    logger.success(f"Path '{path}' exists for attachment={attachment}.")
        return config


def _get_new_path(path: Path, source_dir: Path, recursive: bool = False) -> tuple[bool, Path]:
    # check if the file exists in the source directory
    new_path = source_dir / path.name
    if new_path.exists():
        return True, new_path
    else:
        # check whether part of the source directory is in the rest of the path
        if source_dir.name in path.parts:
            for i, part in enumerate(path.parts):
                if part == source_dir.name:
                    new_path = Path(source_dir).joinpath(*path.parts[i + 1 :])
                    if new_path.exists():
                        return True, new_path
    if recursive:
        for sub_dir in source_dir.glob("*"):
            if sub_dir.is_dir():
                updated, new_path = _get_new_path(path, sub_dir, recursive=True)
                if updated:
                    return updated, new_path
    return False, path

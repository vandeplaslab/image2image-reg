"""Image registration in 2D."""

from __future__ import annotations

import time
import typing as ty
from contextlib import suppress
from copy import copy, deepcopy
from pathlib import Path
from warnings import warn

import numpy as np
from image2image_io.config import CONFIG
from koyo.json import read_json_data, write_json_data
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from mpire import WorkerPool
from tqdm import tqdm

from image2image_wsireg.enums import ArrayLike, WriterMode
from image2image_wsireg.models import Export, Modality, Preprocessing, Registration, Transform, TransformSequence
from image2image_wsireg.models.bbox import BoundingBox, Polygon, _transform_to_bbox, _transform_to_polygon

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader

    from image2image_wsireg.wrapper import ImageWrapper

# override certain parameters
CONFIG.init_pyramid = False
CONFIG.only_last_pyramid = False


class TransformPair(ty.TypedDict):
    """Transformation pair."""

    registration: TransformSequence | None
    initial: TransformSequence | None


class SourceTargetPair(ty.TypedDict):
    """Registration pair."""

    source: str
    target: str | None


class RegistrationNode(ty.TypedDict):
    """Registration node."""

    modalities: SourceTargetPair
    params: list[str]
    registered: bool
    transforms: TransformPair | None
    transform_tag: str | None
    source_preprocessing: Preprocessing | None
    target_preprocessing: Preprocessing | None


class SerializedRegistrationNode(ty.TypedDict):
    """Serialized registration node."""

    source: str
    target: str
    through: str | None
    reg_params: list[str]
    source_preprocessing: dict[str, ty.Any] | None
    target_preprocessing: dict[str, ty.Any] | None


class SerializedRegisteredRegistrationNode(ty.TypedDict):
    """Serialized registered registration node."""

    modalities: SourceTargetPair
    params: list[str]
    registered: bool
    transform_tag: str
    source_preprocessing: Preprocessing | None
    target_preprocessing: Preprocessing | None


class Config(ty.TypedDict):
    """Configuration."""

    schema_version: str
    name: str
    # output_dir: str
    cache_images: bool
    # cache_dir: str
    pairwise: bool
    registration_paths: dict[str, SerializedRegistrationNode]
    registration_graph_edges: list[SerializedRegisteredRegistrationNode]
    modalities: dict[str, dict]
    attachment_images: dict[str, dict]
    attachment_geojsons: dict[str, dict]
    merge: bool
    merge_images: dict[str, list[str]]


class IWsiReg:
    """Whole slide registration utilizing WsiReg approach of graph based registration."""

    CONFIG_NAME = "project.config.json"
    REGISTERED_CONFIG_NAME = "registered-project.config.json"

    log_file: Path | None = None

    def __init__(
        self,
        name: str | None = None,
        output_dir: PathLike | None = None,
        project_dir: PathLike | None = None,
        cache: bool = True,
        merge: bool = False,
        pairwise: bool = False,
        log: bool = False,
        **_kwargs: ty.Any,
    ):
        # setup project directory
        if project_dir:
            self.project_dir = Path(project_dir).with_suffix(".wsireg").resolve()
            name = self.project_dir.stem
        else:
            if name is None:
                raise ValueError("Name must be provided.")
            if output_dir is None:
                raise ValueError("Output directory must be provided.")
            self.project_dir = (Path(output_dir) / name).with_suffix(".wsireg").resolve()
        if ".wsireg" in name:
            name = name.replace(".wsireg", "")
        self.name = name
        self.project_dir.mkdir(exist_ok=True, parents=True)
        self.cache_images = cache
        self.pairwise = pairwise
        self.merge_images = merge

        # setup modalities
        self.modalities: dict[str, Modality] = {}
        self.attachment_images: dict[str, str] = {}
        self.attachment_shapes: dict[str, Modality] = {}
        self.attachment_geojsons: dict[str, dict] = {}
        self.attachment_points: dict[str, dict] = {}
        self.merge_modalities: dict[str, list[str]] = {}  # TODO

        # setup registration paths
        self.registration_paths: dict[str, list[str]] = {}
        self.registration_nodes: list[RegistrationNode] = []
        self.transform_path_map: dict[str, list[SourceTargetPair]] = {}
        self.original_size_transforms: dict[str, list[dict]] = {}
        self.transformations: dict[str, dict] = {}

        # cache where we will store temporary data
        self.preprocessed_cache: dict[str, dict] = {
            "image_spacing": {},
            "image_sizes": {},
            "iterations": {},
            "transformations": {},
        }

        # setup cache directory
        self.cache_dir = self.project_dir / "Cache"
        if self.cache_images:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.trace(f"Caching images to '{self.cache_dir}' directory.")
        if log:
            self.set_logger()

    def set_logger(self) -> None:
        """Setup logger."""
        import sys

        from koyo.logging import get_loguru_env, set_loguru_log

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
            logger.info(f"Setup logging to file - '{self.log_file!s}'")
            logger.trace(f"Executed command: {sys.argv}")

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

    @property
    def transformations_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "Transformations"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def log_dir(self) -> Path:
        """Log directory."""
        directory = self.project_dir / "Logs"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def is_registered(self) -> bool:
        """Check if the project has been registered."""
        return all(reg_edge["registered"] for reg_edge in self.registration_nodes)

    @classmethod
    def from_path(cls, path: PathLike, raise_on_error: bool = True) -> IWsiReg:
        """Initialize based on the project path."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist ({path}).")
        if not path.is_dir():
            raise ValueError("Path is not a directory.")
        if not path.suffix == ".wsireg":
            raise ValueError("Path is not a valid WsiReg project.")

        with MeasureTimer() as timer:
            config_path = path / cls.CONFIG_NAME
            config: dict | Config | None = None
            if config_path.exists():
                config = read_json_data(path / cls.CONFIG_NAME)
            if config and "name" not in config:
                config["name"] = path.stem
            if config and "pairwise" not in config:
                config["pairwise"] = False
            if config is None:
                config = {}

            obj = cls(project_dir=path, **config)
            if config_path.exists():
                obj.load_from_i2i_wsireg(raise_on_error=raise_on_error)
            elif list(obj.project_dir.glob("*.yaml")):
                obj.load_from_wsireg()
        logger.trace(f"Restored from config in {timer()}")
        return obj

    def print_summary(self, func: ty.Callable = logger.info) -> None:
        """Print summary about the project."""
        elbow, pipe, tee, blank = "└──", "│  ", "├──", "   "

        func(f"Project name: {self.name}")
        func(f"Project directory: {self.project_dir}")
        func(f"Merging images: {self.merge_images}")
        func(f"Pairwise registration: {self.pairwise}")
        # func information about the specified modalities
        func(f"Number of modalities: {len(self.modalities)}")
        n = len(self.modalities) - 1
        for i, modality in enumerate(self.modalities.values()):
            func(f" {elbow if i == n else tee}{modality.name} ({modality.path})")
            func(f" {pipe if i != n else blank}{tee}Preprocessing: {modality.preprocessing is not None}")
            func(f" {pipe if i != n else blank}{elbow}Export: {modality.export}")

        # func information about registration paths
        func(f"Number of registration paths: {len(self.registration_paths)}")
        n = len(self.registration_paths) - 1
        for i, (source, targets) in enumerate(self.registration_paths.items()):
            if len(targets) == 1:
                func(f" {elbow if i == n else tee}{source} to {targets[0]}")
            else:
                func(f" {elbow if i == n else tee}{source} to {targets[1]} via {targets[0]}")

        # func information about registration nodes
        func(f"Number of registrations: {self.n_registrations}")
        n = len(self.registration_nodes) - 1
        for i, edge in enumerate(self.registration_nodes):
            insert = pipe if i != n else blank
            modalities = edge["modalities"]
            func(f" {tee if i != n else elbow}{modalities['source']} to {modalities['target']}")
            func(f" {insert}{tee}Transformations: {edge['params']}")
            func(f" {insert}{tee}Registered: {edge['registered']}")
            func(f" {insert}{tee}Source preprocessing: {edge['source_preprocessing']}")
            func(f" {insert}{elbow}Target preprocessing: {edge['target_preprocessing']}")

        # func information about attachment images/shapes
        func(f"Number of attachment images: {len(self.attachment_images)}")
        n = len(self.attachment_images) - 1
        for i, (name, attach_to) in enumerate(self.attachment_images.items()):
            func(f" {elbow if i == n else tee}{name} ({attach_to})")
        func(f"Number of attachment shapes: {len(self.attachment_shapes)}")
        n = len(self.attachment_shapes) - 1
        for i, (name, modality) in enumerate(self.attachment_shapes.items()):
            func(f" {elbow if i == n else tee}{name} ({modality.path})")

        # func information about merge modalities
        func(f"Number of merge modalities: {len(self.merge_modalities)}")
        n = len(self.merge_modalities) - 1
        for i, (name, merge_modalities) in enumerate(self.merge_modalities.items()):
            func(f" {elbow if i == n else tee}{name} ({merge_modalities})")

    def validate(self, allow_not_registered: bool = True) -> bool:
        """Perform several checks on the project."""
        # check whether the paths are still where they were set up
        is_valid = True
        for modality in self.modalities.values():
            if not isinstance(modality.path, ArrayLike) and not Path(modality.path).exists():
                logger.error(f"❌ Modality '{modality.name}' path '{modality.path}' does not exist.")
                is_valid = False
            else:
                logger.success(f"✅ Modality '{modality.name}' exist.")

        # check whether all modalities exist
        for edge in self.registration_nodes:
            modalities = edge["modalities"]
            if modalities["source"] not in self.modalities:
                logger.error(f"❌ Source modality '{modalities['source']}' does not exist.")
                is_valid = False
            elif modalities["target"] not in self.modalities:
                logger.error(f"❌ Target modality '{modalities['target']}' does not exist.")
                is_valid = False
            elif modalities["source"] == modalities["target"]:
                logger.error("❌ Source and target modalities cannot be the same.")
                is_valid = False
            else:
                logger.success(f"✅ Modality pair {modalities['source']} - {modalities['target']} exist.")

        # check whether all registration paths exist
        for source, targets in self.registration_paths.items():
            if source not in self.modalities:
                logger.error(f"❌ Source modality '{source}' does not exist.")
                is_valid = False
            for target in targets:
                if target not in self.modalities:
                    logger.error(f"❌ Target modality '{target}' does not exist.")
                    is_valid = False
                if source == target:
                    logger.error("❌ Source and target modalities cannot be the same.")
                    is_valid = False

        # check whether all modalities have been registered
        if not allow_not_registered:
            for edge in self.registration_nodes:
                if not edge["registered"]:
                    logger.error(
                        f"❌ Modality pair {edge['modalities']['source']} - {edge['modalities']['target']} "
                        f"has not been registered."
                    )
                    is_valid = False
                else:
                    logger.success(
                        f"✅ Modality pair {edge['modalities']['source']} - {edge['modalities']['target']} "
                        f"has been registered."
                    )
        if not is_valid:
            logger.error("❌ Project configuration is invalid.")
        else:
            logger.success("✅ Project configuration is valid.")
        return is_valid

    def load_from_i2i_wsireg(self, raise_on_error: bool = True) -> None:
        """Load data from image2image-wsireg project file."""
        config: Config = read_json_data(self.project_dir / self.CONFIG_NAME)
        self.name = config["name"]
        self.cache_images = config["cache_images"]
        self.pairwise = config["pairwise"]
        self.merge_images = config["merge"]
        # add modality information
        with MeasureTimer() as timer:
            for name, modality in config["modalities"].items():
                if not Path(modality["path"]).exists() and raise_on_error:
                    raise ValueError(f"Modality path '{modality['path']}' does not exist.")
                self.add_modality(
                    name=name,
                    path=modality["path"],
                    preprocessing=Preprocessing(**modality["preprocessing"]) if modality.get("preprocessing") else None,
                    channel_names=modality.get("channel_names", None),
                    channel_colors=modality.get("channel_colors", None),
                    mask=modality.get("mask", None),
                    mask_bbox=modality.get("mask_bbox", None),
                    mask_polygon=modality.get("mask_polygon", None),
                    output_pixel_size=modality.get("output_pixel_size", None),
                    pixel_size=modality.get("pixel_size", None),
                    transform_mask=modality.get("transform_mask", True),
                    export=Export(**modality["export"]) if modality.get("export") else None,
                    raise_on_error=raise_on_error,
                )
            logger.trace(f"Loaded modalities in {timer()}")
            # add registration paths
            for _key, edge in config["registration_paths"].items():
                self.add_registration_path(
                    source=edge["source"],
                    target=edge["target"],
                    through=edge["through"],
                    transform=edge["reg_params"],
                    preprocessing={
                        "source": edge.get("source_preprocessing"),
                        "target": edge.get("target_preprocessing"),
                    },
                )
            logger.trace(f"Loaded registration paths in {timer(since_last=True)}")

            # check whether the registered version of the config exists
            if (self.project_dir / self.REGISTERED_CONFIG_NAME).exists():
                registered_config: Config = read_json_data(self.project_dir / self.REGISTERED_CONFIG_NAME)
                # load transformation data from file
                transformations = {}
                for registered_edge in registered_config["registration_graph_edges"]:
                    source = registered_edge["modalities"]["source"]
                    target = self.registration_paths[source][-1]
                    transforms_seq = self._load_registered_transform(
                        registered_edge, target, raise_on_error=raise_on_error
                    )
                    if transforms_seq:
                        transformations[source] = transforms_seq
                if transformations:
                    self.transformations = transformations
                else:
                    self.transformations = self._collate_transformations()
                logger.trace(f"Loaded registered transformations in {timer(since_last=True)}")

            # load merge modalities
            if config["merge_images"]:
                for name, merge_modalities in config["merge_images"].items():
                    self.merge_modalities[name] = merge_modalities
                logger.trace(f"Loaded merge modalities in {timer(since_last=True)}")

    def _load_registered_transform(
        self,
        edge: SerializedRegisteredRegistrationNode,
        target: str,
        raise_on_error: bool = True,
    ) -> dict[str, TransformSequence | None] | None:
        """Load registered transform and make sure all attributes are correctly set-up."""
        from image2image_wsireg.wrapper import ImageWrapper

        edge_ = self._find_edge_by_edge(edge)
        if not edge_:
            logger.warning("Could not find appropriate registration node.")
            return None
        source = edge["modalities"]["source"]
        transform_tag = f"{self.name}-{source}_to_{target}_transformations.json"
        if transform_tag and not (self.transformations_dir / transform_tag).exists():
            logger.warning(f"Could not find cached registration data. ('{transform_tag}' file does not exist)")
            return None

        target_modality = self.modalities[target]
        target_wrapper = ImageWrapper(
            target_modality, edge["target_preprocessing"], quick=True, raise_on_error=raise_on_error
        )

        source_modality = self.modalities[source]
        source_wrapper = ImageWrapper(
            source_modality, edge["source_preprocessing"], quick=True, raise_on_error=raise_on_error
        )
        source_wrapper.initial_transforms = source_wrapper.load_initial_transform(source_modality, self.cache_dir)

        initial_transforms_seq = None
        if source_wrapper.initial_transforms:
            initial_transforms_ = [Transform(t) for t in source_wrapper.initial_transforms]
            initial_transforms_index = [idx for idx, _ in enumerate(initial_transforms_)]
            initial_transforms_seq = TransformSequence(initial_transforms_, initial_transforms_index)

        transforms_partial_seq = TransformSequence.from_path(
            self.transformations_dir / transform_tag, first=True, skip_initial=True
        )
        transforms_full_seq = TransformSequence.from_path(self.transformations_dir / transform_tag, first=False)
        # if initial_transforms_seq:
        #     transforms_full_seq.insert(initial_transforms_seq)
        self.original_size_transforms[target_wrapper.name] = target_wrapper.original_size_transform

        # setup parameters
        edge_["transforms"] = {"registration": transforms_partial_seq, "initial": initial_transforms_seq}
        edge_["registered"] = True
        edge_["transform_tag"] = f"{self.name}-{source}_to_{target}_transformations.json"
        logger.trace(f"Restored previous transformation data for {source} - {target}")
        return {
            f"initial-{source}": initial_transforms_seq,
            f"000-to-{target}": transforms_partial_seq,
            "full-transform-seq": transforms_full_seq,
        }

    def _find_edge_by_edge(self, edge: SerializedRegisteredRegistrationNode) -> RegistrationNode | None:
        """Find edge by another edge, potentially from cache."""
        for edge_ in self.registration_nodes:
            if edge_["modalities"]["source"] == edge["modalities"]["source"]:
                if edge_["modalities"]["target"] == edge["modalities"]["target"]:
                    return edge_
        return None

    def load_from_wsireg(self) -> None:
        """Load data from WsiReg YAML project file."""

    def auto_add_modality(
        self,
        name: str,
        path: PathLike,
        preprocessing: Preprocessing | None = None,
        mask: PathLike | None = None,
        mask_bbox: tuple[int, int, int, int] | None = None,
        export: Export | dict[str, ty.Any] | None = None,
        override: bool = False,
    ) -> Modality:
        """Add modality."""
        from image2image_io.readers import get_simple_reader, is_supported

        path = Path(path)
        if not path.exists():
            raise ValueError("Path does not exist.")
        if not is_supported(path):
            raise ValueError("Unsupported file format.")
        if name in self.modalities and not override:
            raise ValueError(f"Modality name '{name}' already exists.")
        reader: BaseReader = get_simple_reader(path, init_pyramid=False)
        if preprocessing:
            preprocessing.channel_names = reader.channel_names
            preprocessing.channel_indices = reader.channel_ids

        return self.add_modality(
            name,
            path,
            pixel_size=reader.resolution or 1.0,
            channel_names=reader.channel_names,
            mask=mask,
            mask_bbox=mask_bbox,
            preprocessing=preprocessing,
            export=export,
            override=override,
        )

    def add_modality(
        self,
        name: str,
        path: PathLike,
        pixel_size: float = 1.0,
        channel_names: list[str] | None = None,
        channel_colors: list[str] | None = None,
        preprocessing: Preprocessing | dict[str, ty.Any] | None = None,
        mask: PathLike | np.ndarray | None = None,
        mask_bbox: tuple[int, int, int, int] | BoundingBox | None = None,
        mask_polygon: np.ndarray | Polygon | None = None,
        output_pixel_size: tuple[float, float] | None = None,
        transform_mask: bool = True,
        export: Export | dict[str, ty.Any] | None = None,
        override: bool = False,
        raise_on_error: bool = True,
    ) -> Modality:
        """Add modality."""
        from image2image_io.readers import is_supported

        path = Path(path)
        if not path.exists() and raise_on_error:
            raise ValueError("Path does not exist.")
        if not is_supported(path, raise_on_error):
            raise ValueError("Unsupported file format.")
        if name in self.modalities and not override:
            raise ValueError(f"Modality '{name}' name already exists.")
        if isinstance(preprocessing, dict):
            preprocessing = Preprocessing(**preprocessing)
        if isinstance(export, dict):
            export = Export(**export)
        if isinstance(mask, (str, Path)):
            mask = Path(mask)
            if not mask.exists():
                raise ValueError("Mask path does not exist.")
        if pixel_size <= 0:
            raise ValueError("Pixel size must be greater than 0.")
        if isinstance(output_pixel_size, (int, float)):
            output_pixel_size = (float(output_pixel_size), float(output_pixel_size))
        if mask is not None and mask_bbox is not None and mask_polygon is not None:
            raise ValueError("Mask can only be specified using one of the three options: mask, mask_bbox, mask_polygon")
        if mask_bbox is not None:
            mask_bbox = _transform_to_bbox(mask_bbox)
        if mask_polygon is not None:
            mask_polygon = _transform_to_polygon(mask_polygon)
        if "initial" in name:
            raise ValueError("Sorry, the word 'initial' cannot be used in the modality name as it's reserved.")

        self.modalities[name] = Modality(
            name=name,
            path=path.resolve() if raise_on_error else path,
            pixel_size=pixel_size,
            channel_names=channel_names,
            channel_colors=channel_colors,
            preprocessing=preprocessing,
            mask=mask,
            output_pixel_size=output_pixel_size,
            mask_bbox=mask_bbox,
            mask_polygon=mask_polygon,
            export=export,
            transform_mask=transform_mask,
        )
        logger.trace(f"Added modality '{name}'.")
        return self.modalities[name]

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
        if not path.exists():
            raise ValueError(f"Path '{path}' does not exist.")
        self.add_modality(
            name,
            path.resolve(),
            pixel_size,
            channel_names=channel_names,
            channel_colors=channel_colors,
            override=True,
        )
        self.attachment_images[name] = attach_to_modality
        logger.trace(f"Added attachment image '{name}'.")

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

    def add_attachment_geojson(
        self,
        attach_to_modality: str,
        name: str,
        paths: list[PathLike],
    ) -> None:
        """
        Add attached shapes.

        Parameters
        ----------
        attach_to_modality : str
            image modality to which the shapes are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        """
        if attach_to_modality not in self.modalities:
            raise ValueError(
                f"The specified modality '{attach_to_modality}' does not exist. Please add it before adding attachment"
                f" images."
            )
        paths_: list[Path] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path '{path}' does not exist.")
            paths_.append(path.resolve())

        pixel_size = self.modalities[attach_to_modality].pixel_size
        self._add_geojson_set(attach_to_modality, name, paths_, pixel_size)

    def _add_geojson_set(
        self,
        attach_to_modality: str,
        name: str,
        paths: list[Path],
        pixel_size: float,
    ) -> None:
        """
        Add a shape set to the graph.

        Parameters
        ----------
        attach_to_modality : str
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
        if name in self.attachment_geojsons:
            raise ValueError(f"Shape set with name '{name}' already exists. Please use a different name.")
        self.attachment_geojsons[name] = {
            "shape_files": paths,
            "pixel_size": pixel_size,
            "attach_to_modality": attach_to_modality,
        }
        logger.trace(f"Added shape set '{name}'.")

    def auto_add_attachment_points(self, attach_to_modality: str, name: str, path: PathLike) -> None:
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
        if Path(path).suffix not in [".csv", ".txt", ".parquet"]:
            raise ValueError("Attachment must be in Points format.")
        if attach_to_modality not in self.modalities:
            raise ValueError("Modality does not exist. Please add it before trying to add an attachment.")
        self.add_attachment_points(attach_to_modality, name, path)

    def add_attachment_points(
        self,
        attach_to_modality: str,
        name: str,
        paths: list[PathLike],
    ) -> None:
        """
        Add attached shapes.

        Parameters
        ----------
        attach_to_modality : str
            image modality to which the shapes are attached
        name : str
            Unique name identifier for the shape set
        paths : list of file paths
            list of shape data in geoJSON format or list of dicts containing the following keys:
            "array" = np.ndarray of xy coordinates, "shape_type" = geojson shape type (Polygon, MultiPolygon, etc.),
            "shape_name" = class of the shape("tumor","normal",etc.)
        """
        if attach_to_modality not in self.modalities:
            raise ValueError(
                f"The specified modality '{attach_to_modality}' does not exist. Please add it before adding attachment"
                f" objects."
            )
        paths_: list[Path] = []
        for path in paths:
            path = Path(path)
            if not path.exists():
                raise ValueError(f"Path '{path}' does not exist.")
            paths_.append(path.resolve())

        pixel_size = self.modalities[attach_to_modality].pixel_size
        self._add_points_set(attach_to_modality, name, paths_, pixel_size)

    def _add_points_set(
        self,
        attach_to_modality: str,
        name: str,
        paths: list[Path],
        pixel_size: float,
    ) -> None:
        """
        Add a shape set to the graph.

        Parameters
        ----------
        attach_to_modality : str
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
            "point_files": paths,
            "pixel_size": pixel_size,
            "attach_to_modality": attach_to_modality,
        }
        logger.trace(f"Added points set '{name}'.")

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

    @property
    def n_registrations(self) -> int:
        """Number of registrations to be performed."""
        return len(self.registration_nodes)

    def add_registration_path(
        self,
        source: str,
        target: str,
        transform: str | ty.Iterable[str],
        through: str | None = None,
        preprocessing: dict | None = None,
    ) -> None:
        """Add a registration path between modalities.

        You can define registration from source to target or from source through 'through' to target.
        """
        if source not in self.modalities:
            raise ValueError(f"Source modality '{source}' does not exist.")
        if target not in self.modalities:
            raise ValueError(f"Target modality '{target}' does not exist.")
        if through is not None:
            if through not in self.modalities:
                raise ValueError(f"Through modality '{through}' does not exist.")
            if through == target:
                raise ValueError("Through modality cannot be the same as target.")
            if through == source:
                raise ValueError("Through modality cannot be the same as source.")
        if preprocessing is None:
            preprocessing = {"target": None, "source": None}
        elif isinstance(preprocessing, dict):
            assert "target" in preprocessing, "Preprocessing must contain target key."
            assert "source" in preprocessing, "Preprocessing must contain source key."
        self._add_registration_node(source, target, through, transform, preprocessing)

    def _add_registration_node(
        self,
        source: str,
        target: str,
        through: str | None,
        transform: str | ty.Iterable[str],
        preprocessing: dict | None = None,
    ) -> None:
        """Add registration node."""
        # create the registration path
        self.registration_paths[source] = [target] if not through else [through, target]

        # setup override pre-processing
        source_preprocessing, target_preprocessing = None, None
        if preprocessing:
            if preprocessing.get("source"):
                source_preprocessing = Preprocessing(**preprocessing.get("source"))  # type: ignore[arg-type]
            if preprocessing.get("target"):
                target_preprocessing = Preprocessing(**preprocessing.get("target"))  # type: ignore[arg-type]

        # validate transform
        if isinstance(transform, str):
            transform = [transform]
        # if isinstance(transform, str):
        #     transform = [Registration.from_name(transform)]
        # elif isinstance(transform, Registration):
        #     transform = [transform]
        # elif isinstance(transform, list) and all(isinstance(t, str) for t in transform):
        #     transform = [Registration.from_name(tr) for tr in transform]

        if not isinstance(transform, list) and all(isinstance(t, Registration) for t in transform):
            raise ValueError("Transform must be a Transform object or a list of Transform objects.")

        # create graph edges
        self.registration_nodes.append(
            {
                "modalities": {"source": source, "target": through if through else target},
                "params": transform,
                "registered": False,
                "transforms": None,
                "transform_tag": None,
                "source_preprocessing": source_preprocessing,
                "target_preprocessing": target_preprocessing,
            }
        )
        self._create_transformation_paths(self.registration_paths)
        logger.trace(f"Added registration path from '{source}' to '{target}' through '{through}'.")

    def _create_transformation_paths(self, registration_paths: dict[str, list[str]]) -> None:
        """Create the path for each registration."""
        transform_path_map: dict[str, list[SourceTargetPair]] = {}
        for key, value in registration_paths.items():
            transform_path_modalities = self.find_registration_path(key, value[-1])
            if not transform_path_modalities:
                raise ValueError(f"Could not find registration path from {key} to {value[-1]}.")
            if self.pairwise:
                transform_path_modalities = transform_path_modalities[:1]
            transform_edges: list[SourceTargetPair] = []
            for modality in transform_path_modalities:
                for edge in self.registration_nodes:
                    edge_modality = edge["modalities"]["source"]
                    if modality == edge_modality:
                        transform_edges.append(edge["modalities"])
                    transform_path_map[key] = transform_edges
        self.transform_path_map = transform_path_map

    def find_registration_path(self, start: str, end: str, path: list[str] | None = None) -> list[str] | None:
        """Find the path from the 'start' modality to 'end' modality in the graph."""
        if path is None:
            path = []
        path = [*path, start]
        if start == end:
            return path
        if start not in self.registration_paths:
            return None
        for node in self.registration_paths[start]:
            if node not in path:
                extended_path = self.find_registration_path(node, end, path)
                if extended_path:
                    return extended_path
        return None

    def _preprocess_image(
        self, modality: Modality, preprocessing: Preprocessing | None = None, override: bool = False
    ) -> ImageWrapper:
        """Pre-process images."""
        from image2image_wsireg.wrapper import ImageWrapper

        wrapper = ImageWrapper(modality, preprocessing)
        cached = wrapper.check_cache(self.cache_dir, self.cache_images) if not override else False
        if not cached:
            wrapper.preprocess()
            wrapper.save_cache(self.cache_dir, self.cache_images)
        else:
            wrapper.load_cache(self.cache_dir, self.cache_images)
        if wrapper.image is None:
            raise ValueError(f"The '{modality.name}' image has not been pre-processed.")

        # update caches
        self.preprocessed_cache["image_spacing"][modality.name] = wrapper.image.GetSpacing()  # type:ignore[no-untyped-call]
        self.preprocessed_cache["image_sizes"][modality.name] = wrapper.image.GetSize()  # type:ignore[no-untyped-call]
        return wrapper

    @staticmethod
    def __preprocess_image(
        cache_dir: Path, cache_images: bool, modality: Modality, preprocessing: Preprocessing | None = None
    ) -> ImageWrapper:
        """Pre-process image."""
        from image2image_wsireg.wrapper import ImageWrapper

        wrapper = ImageWrapper(modality, preprocessing)
        cached = wrapper.check_cache(cache_dir, cache_images)
        if not cached:
            wrapper.preprocess()
            wrapper.save_cache(cache_dir, cache_images)
        else:
            wrapper.load_cache(cache_dir, cache_images)
        if wrapper.image is None:
            raise ValueError(f"The '{modality.name}' image has not been pre-processed.")
        return wrapper

    def _coregister_images(
        self,
        source_wrapper: ImageWrapper,
        target_wrapper: ImageWrapper,
        parameters: list[str],
        output_dir: Path,
        histogram_match: bool = False,
    ) -> tuple[TransformSequence, TransformSequence | None]:
        """Co-register images."""
        from image2image_wsireg.utils.registration import _prepare_reg_models, register_2d_images, sitk_pmap_to_dict

        with MeasureTimer() as timer:
            logger.trace(f"Co-registering {source_wrapper.name} to {target_wrapper.name} using {parameters}.")
            # co-register images
            transform = register_2d_images(
                source_wrapper,
                target_wrapper,
                _prepare_reg_models(parameters),
                output_dir,
                histogram_match=histogram_match,
            )
            # convert transformation to something readable
            transforms = [sitk_pmap_to_dict(tf) for tf in transform]
            # check whether the source modality had any initial transforms
            initial_transforms = source_wrapper.initial_transforms
            initial_transforms_seq = None
            if initial_transforms:
                initial_transforms_ = [Transform(t) for t in initial_transforms]
                initial_transforms_index = [idx for idx, _ in enumerate(initial_transforms_)]
                initial_transforms_seq = TransformSequence(initial_transforms_, initial_transforms_index)
            # convert transformation data to transform sequence
            transforms_seq = TransformSequence([Transform(t) for t in transforms], list(range(len(transforms))))
            self.original_size_transforms[target_wrapper.name] = target_wrapper.original_size_transform
        logger.info(f"Co-registration of {source_wrapper.name} to {target_wrapper.name} took {timer()}.")
        return transforms_seq, initial_transforms_seq

    def _generate_figures(self, source: str, target: str, output_dir: Path) -> None:
        """Generate figures."""
        from image2image_wsireg.utils.figures import (
            read_elastix_iteration_dir,
            read_elastix_transform_dir,
            write_iteration_plots,
        )

        with MeasureTimer() as timer:
            key = f"{source}_to_{target}"
            transform_data = read_elastix_transform_dir(output_dir)
            iteration_data = read_elastix_iteration_dir(output_dir)

            write_iteration_plots(iteration_data, key, output_dir)
            self.preprocessed_cache["iterations"][key] = iteration_data
            self.preprocessed_cache["transformations"][key] = transform_data
        logger.info(f"Generated figures for {key} in {timer}.")

    def _collate_transformations(self) -> dict[str, dict]:
        transforms = {}
        edge_modality_pairs = [v["modalities"] for v in self.registration_nodes]
        for modality, edges in self.transform_path_map.items():
            full_tform_seq = TransformSequence()
            for index, edge in enumerate(edges):
                registered_edge_transform: TransformPair = self.registration_nodes[edge_modality_pairs.index(edge)][
                    "transforms"
                ]
                if not registered_edge_transform:
                    logger.warning(f"No registration data for {edge} (not in 'registered_edge_transform'}}.")
                    continue
                if index == 0:
                    transforms[modality] = {
                        f"initial-{edges[index]['source']}": registered_edge_transform["initial"],
                        f"{str(index).zfill(3)}-to-{edges[index]['target']}": registered_edge_transform["registration"],
                    }
                    if registered_edge_transform["initial"]:
                        full_tform_seq.append(registered_edge_transform["initial"])
                    full_tform_seq.append(registered_edge_transform["registration"])
                else:
                    transforms[modality][
                        f"{str(index).zfill(3)}-to-{edges[index]['target']}"
                    ] = registered_edge_transform["registration"]
                    full_tform_seq.append(registered_edge_transform["registration"])
                transforms[modality]["full-transform-seq"] = full_tform_seq
        return transforms

    def preprocess(self, n_parallel: int = 1, override: bool = False) -> None:
        """Pre-process all images."""
        # TODO: add multi-core support
        self.set_logger()
        # if not self.registration_nodes:
        #     raise ValueError("No registration paths have been defined.")

        # compute transformation information
        with MeasureTimer() as timer:
            # to_preprocess = []
            # for modality in self.modalities.values():
            #     to_preprocess.append(modality.name)
            #
            # if n_parallel and len(to_preprocess) > 1:
            #     with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
            #         pool.imap_unordered(self._preprocess_image, to_preprocess)
            for modality in tqdm(self.modalities.values(), desc="Pre-processing images"):
                logger.trace(f"Pre-processing {modality.name}.")
                self._preprocess_image(modality, None, override=override)
                logger.info(f"Pre-processing of all images took {timer(since_last=True)}.")

    def register(self, n_parallel: int = 1, preprocess_first: bool = True, histogram_match: bool = False) -> None:
        """Co-register images."""
        # TODO: add multi-core support
        self.set_logger()
        self.save(registered=False)
        if not self.registration_nodes:
            raise ValueError("No registration paths have been defined.")
        if preprocess_first:
            self.preprocess(n_parallel=n_parallel)

        # compute transformation information
        for edge in tqdm(self.registration_nodes, desc="Registering nodes..."):
            if edge["registered"]:
                logger.trace(
                    f"Skipping registration for {edge['modalities']['source']} to {edge['modalities']['target']}."
                )
                continue

            # retrieve modalities
            source = edge["modalities"]["source"]
            source_modality = deepcopy(self.modalities[source])
            source_preprocessing = edge["source_preprocessing"]
            target = edge["modalities"]["target"]
            target_modality = deepcopy(self.modalities[target])
            target_preprocessing = edge["target_preprocessing"]

            source_wrapper = self._preprocess_image(source_modality, source_preprocessing)
            target_wrapper = self._preprocess_image(target_modality, target_preprocessing)

            # create registration directory
            registration_dir = self.progress_dir / f"{source}-{target}_reg_output"
            registration_dir.mkdir(exist_ok=True, parents=True)

            # register images
            registration, initial = self._coregister_images(
                source_wrapper,
                target_wrapper,
                edge["params"],
                registration_dir,
                histogram_match=histogram_match,
            )
            edge["transforms"] = {"registration": registration, "initial": initial}
            edge["registered"] = True
            edge["transform_tag"] = f"{self.name}-{source}_to_{target}_transformations.json"

            # load plot data
            self._generate_figures(source, target, registration_dir)

        # collect transformations
        self.transformations = self._collate_transformations()
        # save transformations
        self.save_transformations()
        self.save(registered=True)

    def clear(
        self, cache: bool = True, image: bool = True, transformations: bool = True, progress: bool = True
    ) -> None:
        """Clear existing data."""
        from shutil import rmtree

        def _safe_delete(file_: Path) -> None:
            if not file_.exists():
                return
            if file_.is_dir():
                try:
                    rmtree(file_)
                    logger.trace(f"Deleted directory {file_}.")
                except Exception as e:
                    logger.error(f"Could not delete {file_}. {e}")
            else:
                try:
                    file_.unlink()
                    logger.trace(f"Deleted file {file_}.")
                except Exception as e:
                    logger.error(f"Could not delete {file_}. {e}")

        # clear transformations, cache, images
        if cache:
            for file in self.cache_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.cache_dir)

        if progress:
            for file in self.progress_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.progress_dir)

        if image:
            for file in self.image_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.image_dir)

        if transformations:
            for file in self.transformations_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.transformations_dir)

        # remove config files
        file = self.project_dir / self.REGISTERED_CONFIG_NAME
        _safe_delete(file)

    def run(self) -> None:
        """Execute workflow."""
        self.set_logger()
        self.register()
        # write images
        self.write_images()

    @staticmethod
    def _transforms_to_txt(transformations: dict[str, TransformSequence]) -> dict[str, list[str]]:
        tform_txt = {}
        for key, transform in transformations.items():
            if key == "full-transform-seq":
                continue
            if "initial" in key:
                if transform:
                    for idx, registered_transform in enumerate(transform.transforms):
                        tform_txt.update({f"{key}-{str(idx).zfill(2)}": registered_transform.elastix_transform})
            else:
                # tform_txt.update({key: [rt.elastix_transform for rt in transform.transforms]})
                tform_txt[key] = [rt.elastix_transform for rt in transform.transforms]
        return tform_txt

    def _find_merge_modalities(self) -> list[str]:
        merge_modalities = []
        for _k, v in self.merge_modalities.items():
            merge_modalities.extend(v)
        return merge_modalities

    def _find_not_registered_modalities(self) -> list[str]:
        registered_modalities = [edge["modalities"]["source"] for edge in self.registration_nodes]
        non_reg_modalities = list(set(self.modalities.keys()).difference(registered_modalities))

        # remove attachment modalities
        for attachment_modality in self.attachment_images.keys():
            non_reg_modalities.pop(non_reg_modalities.index(attachment_modality))
        return non_reg_modalities

    def _check_if_all_registered(self) -> bool:
        if not all(reg_edge.get("registered") for reg_edge in self.registration_nodes):
            warn(
                "Registration has not been executed for the graph no transformations to save. Please run register()"
                " first.",
                stacklevel=2,
            )
            return False
        return True

    def save_transformations(self) -> list[Path] | None:
        """Save all transformations for a given modality as JSON."""
        if not self._check_if_all_registered():
            return None

        out = []
        for source_modality in self.registration_paths:
            target_modalities = self.registration_paths[source_modality]
            target_modality = target_modalities[-1]
            output_path = (
                self.transformations_dir / f"{self.name}-{source_modality}_to_{target_modality}_transformations.json"
            )
            tform_txt = self._transforms_to_txt(self.transformations[source_modality])
            write_json_data(output_path, tform_txt)
            out.append(output_path)
            logger.trace(
                f"Saved transformations to '{output_path}'. source={source_modality}; target={target_modalities}."
            )

        for source_modality, attachment_modality in self.attachment_images.items():
            if attachment_modality not in self._find_not_registered_modalities():
                target_modalities = self.registration_paths[attachment_modality]
                target_modality = target_modalities[-1]
                output_path = (
                    self.transformations_dir
                    / f"{self.name}-{source_modality}_to_{target_modality}_transformations.json"
                )
                tform_txt = self._transforms_to_txt(self.transformations[source_modality])
                write_json_data(output_path, tform_txt)
                logger.trace(
                    f"Saved transformations to '{output_path}'. source={source_modality}; target={target_modalities}."
                )
        return out

    def write_images(
        self,
        n_parallel: int = 1,
        fmt: WriterMode = "ome-tiff",
        write_registered: bool = True,
        write_not_registered: bool = True,
        remove_merged: bool = True,
        to_original_size: bool = True,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        write_merged: bool = True,
        override: bool = False,
    ) -> list | None:
        """Export images after applying transformation."""
        # TODO add multi-core support
        self.set_logger()
        if not self._check_if_all_registered():
            return None

        def _get_with_suffix(p: Path) -> Path:
            if fmt in ["ome-tiff", "ome-tiff-by-plane", "ome-tiff-by-tile"]:
                return p.parent / (p.name + ".ome.tiff")
            raise ValueError(f"Writer {fmt} is not supported.")

        paths = []

        # prepare merge modalities metadata
        merge_modalities = self._find_merge_modalities()
        if merge_modalities:
            logger.trace(f"Merge modalities: {merge_modalities}")
        modalities = list(self.registration_paths.keys())
        if modalities:
            logger.trace(f"Registered modalities: {modalities}")
        not_registered_modalities = self._find_not_registered_modalities()
        if not_registered_modalities:
            logger.trace(f"Not registered modalities: {not_registered_modalities}")

        if remove_merged:
            for merge_modality in merge_modalities:
                with suppress(ValueError):
                    index = modalities.index(merge_modality)
                    modalities.pop(index)
                    logger.trace(f"Removed {merge_modality} from registered modalities as it will be merged.")
                with suppress(ValueError):
                    index = not_registered_modalities.index(merge_modality)
                    not_registered_modalities.pop(index)
                    logger.trace(f"Removed {merge_modality} from not registered modalities as it will be merged.")

        # export non-registered nodes
        if write_not_registered:
            # preprocess and save unregistered nodes
            to_write = []
            for modality in tqdm(not_registered_modalities, desc="Exporting not-registered images..."):
                if modality in merge_modalities and remove_merged:
                    continue
                try:
                    image_modality, transformations, output_path = self._prepare_not_registered_image_transform(
                        modality,
                        to_original_size=to_original_size,
                    )
                    if _get_with_suffix(output_path).exists() and not override:
                        logger.trace(f"Skipping {modality} as it already exists ({output_path}).")
                        continue
                    logger.trace(f"Exporting {modality} to {output_path}...")
                    to_write.append((image_modality, transformations, output_path, fmt, tile_size, as_uint8))
                except KeyError:
                    logger.warning(f"Could not find transformation data for {modality}.")
            if to_write:
                if n_parallel > 1:
                    with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
                        res = pool.imap(self._transform_write_image, to_write)
                    paths.extend(list(res))
                else:
                    for args in tqdm(to_write, desc="Exporting attachment modalities..."):
                        path = self._transform_write_image(*args)
                        paths.append(path)

        # export modalities
        if write_registered:
            to_write = []
            for modality in tqdm(modalities, desc="Exporting registered modalities...", total=len(modalities)):
                image_modality, _, output_path = self._prepare_registered_image_transform(
                    modality, attachment=False, to_original_size=to_original_size
                )

                if _get_with_suffix(output_path).exists() and not override:
                    logger.trace(f"Skipping {modality} as it already exists. ({output_path})")
                    continue
                logger.trace(f"Exporting {modality} to {output_path}...")
                to_write.append(
                    (
                        image_modality.name,
                        None,
                        output_path,
                        fmt,
                        tile_size,
                        as_uint8,
                        lambda: (False, to_original_size),
                    )
                )
            if to_write:
                if n_parallel > 1:
                    with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
                        res = pool.imap(self._transform_write_image, to_write)
                    paths.extend(list(res))
                else:
                    for args in tqdm(to_write, desc="Exporting attachment modalities..."):
                        path = self._transform_write_image(*args)
                        paths.append(path)

            # export attachment modalities
            to_write = []
            for modality, attach_to_modality in tqdm(
                self.attachment_images.items(), desc="Exporting attachment modalities..."
            ):
                if modality in merge_modalities and remove_merged:
                    continue
                attach_modality = self.modalities[attach_to_modality]
                if attach_to_modality in self._find_not_registered_modalities():
                    image_modality, _, output_path = self._prepare_not_registered_image_transform(
                        modality,
                        attachment=True,
                        attachment_modality=attach_modality,
                        to_original_size=to_original_size,
                    )
                else:
                    image_modality, _, output_path = self._prepare_registered_image_transform(
                        modality,
                        attachment=True,
                        attachment_modality=attach_modality,
                        to_original_size=to_original_size,
                    )
                if _get_with_suffix(output_path).exists() and not override:
                    logger.trace(f"Skipping {attach_to_modality} as it already exists ({output_path}).")
                    continue
                logger.trace(f"Exporting {attach_modality} to {output_path}...")
                to_write.append(
                    (
                        image_modality.name,
                        None,
                        output_path,
                        fmt,
                        tile_size,
                        as_uint8,
                        lambda: (True, to_original_size),
                    )
                )
            if to_write:
                # if n_parallel > 1:
                #     with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
                #         res = pool.imap(self._transform_write_image, to_write)
                #     paths.extend(list(res))
                # else:
                for args in tqdm(to_write, desc="Exporting attachment modalities..."):
                    path = self._transform_write_image(*args)
                    paths.append(path)

        # export merge modalities
        if write_merged and self.merge_modalities:
            path = self._transform_write_merge_images(
                to_original_size=to_original_size, as_uint8=as_uint8, override=override
            )
            paths.append(path)

        return paths

    def _prepare_registered_image_transform(
        self,
        edge_key: str,
        attachment: bool = False,
        attachment_modality: Modality | None = None,
        to_original_size: bool = True,
    ) -> tuple[Modality, TransformSequence, Path]:
        if attachment and attachment_modality:
            final_modality = self.registration_paths[attachment_modality.name][-1]
            transformations = copy(self.transformations[attachment_modality.name]["full-transform-seq"])
        else:
            final_modality = self.registration_paths[edge_key][-1]
            transformations = copy(self.transformations[edge_key]["full-transform-seq"])

        output_path = self.image_dir / f"{self.name}-{edge_key}_to_{final_modality}_registered"
        modality_key = None
        if attachment and attachment_modality:
            modality_key = copy(edge_key)
            edge_key = attachment_modality.name

        modality = self.modalities[edge_key]
        if self.original_size_transforms.get(final_modality) and to_original_size:
            logger.trace("Adding transform to original size...")
            original_size_transform = self.original_size_transforms[final_modality]
            if isinstance(original_size_transform, list):
                original_size_transform = original_size_transform[0]
            orig_size_rt = TransformSequence(Transform(original_size_transform), transform_sequence_index=[0])
            transformations.append(orig_size_rt)

        if modality.preprocessing and modality.preprocessing.downsample > 1:
            if not modality.output_pixel_size:
                output_spacing_target = self.modalities[final_modality].output_pixel_size
                transformations.set_output_spacing((output_spacing_target, output_spacing_target))
            else:
                transformations.set_output_spacing(modality.output_pixel_size)
        elif modality.output_pixel_size:
            transformations.set_output_spacing(modality.output_pixel_size)
        if attachment and modality_key:
            modality = self.modalities[modality_key]
        return modality, transformations, output_path

    def _prepare_not_registered_image_transform(
        self,
        modality_key: str,
        attachment: bool = False,
        attachment_modality: Modality | None = None,
        to_original_size: bool = True,
    ):
        from image2image_wsireg.utils.transformation import identity_elx_transform
        from image2image_wsireg.wrapper import ImageWrapper

        logger.trace(f"Preparing transforms for non-registered modality : {modality_key} ")
        output_path = self.image_dir / f"{self.name}-{modality_key}_registered"

        im_data_key = None
        if attachment and attachment_modality:
            im_data_key = copy(modality_key)
            modality_key = attachment_modality.name

        transformations = None
        modality = self.modalities[modality_key]

        if modality.preprocessing and (
            modality.preprocessing.rotate_counter_clockwise != 0
            or modality.preprocessing.flip
            or modality.preprocessing.translate_x != 0
            or modality.preprocessing.translate_y != 0
            or modality.preprocessing.crop_to_bbox
            or modality.preprocessing.crop_bbox
            or modality.preprocessing.affine is not None
        ):
            initial_transform = ImageWrapper.load_initial_transform(modality, self.cache_dir)
            original_size_transform = ImageWrapper.load_original_size_transform(modality, self.cache_dir)

            if initial_transform:
                transformations = TransformSequence(
                    [Transform(t) for t in initial_transform],
                    transform_sequence_index=list(range(len(initial_transform))),
                )
            if original_size_transform:
                # TODO: this might be broken
                transformations = TransformSequence(
                    [Transform(t[0]) for t in original_size_transform],
                    transform_sequence_index=list(range(len(original_size_transform))),
                )

        if to_original_size and self.original_size_transforms[modality_key]:
            o_size_tform = self.original_size_transforms[modality_key]
            if isinstance(o_size_tform, list):
                o_size_tform = o_size_tform[0]

            orig_size_rt = TransformSequence(
                Transform(o_size_tform),
                transform_sequence_index=[0],
            )
            if transformations:
                transformations.append(orig_size_rt)
            else:
                transformations = orig_size_rt

        if modality.preprocessing and modality.preprocessing.downsample > 1 and transformations:
            if not modality.output_pixel_size:
                output_spacing_target = modality.pixel_size
                transformations.set_output_spacing((output_spacing_target, output_spacing_target))
            else:
                transformations.set_output_spacing(modality.output_pixel_size)

        elif modality.preprocessing and modality.preprocessing.downsample > 1 and not transformations:
            transformations = TransformSequence(
                [
                    Transform(
                        identity_elx_transform(
                            self.preprocessed_cache["image_sizes"][modality_key],
                            self.preprocessed_cache["image_spacing"][modality_key],
                        )
                    )
                ],
                transform_sequence_index=[0],
            )

            if not modality.output_pixel_size:
                output_spacing_target = modality.pixel_size
                transformations.set_output_spacing((output_spacing_target, output_spacing_target))
            else:
                transformations.set_output_spacing(modality.output_pixel_size)

        if attachment and im_data_key:
            modality = self.modalities[im_data_key]
        return modality, transformations, output_path

    def _transform_write_image(
        self,
        modality: Modality | str,
        transformations: TransformSequence | None,
        filename: Path,
        fmt: WriterMode = "ome-tiff",
        tile_size: int = 512,
        as_uint8: bool | None = None,
        prep_func: ty.Callable | None = None,
        preview: bool = False,
    ) -> Path:
        """Transform and write image."""
        from image2image_io.writers import OmeTiffWriter

        from image2image_wsireg.wrapper import ImageWrapper

        if not modality and not prep_func:
            raise ValueError("Either modality or prep_func must be specified.")
        if prep_func:
            attachment, to_original_size = prep_func()
            modality_name = modality
            modality, transformations, filename = self._prepare_registered_image_transform(
                modality_name, attachment=attachment, to_original_size=to_original_size
            )

        as_uint8_ = as_uint8
        wrapper = ImageWrapper(modality, preview=preview)
        logger.trace(f"Writing '{filename}' with {transformations} transform...")
        if fmt in ["ome-tiff", "ome-tiff-by-plane"]:
            writer = OmeTiffWriter(wrapper.reader, transformer=transformations)
        else:
            raise ValueError("Other writers are nto yet supported")
        name = filename.name if not preview else filename.name + "_preview"

        channel_ids, channel_names = None, None
        if modality.export:
            as_uint8 = modality.export.as_uint8
            channel_ids = modality.export.channel_ids
            channel_names = modality.export.channel_names
        if channel_ids:
            if len(channel_ids) == 0:
                raise ValueError("No channel ids have been specified.")
            if len(channel_ids) > wrapper.reader.n_channels:
                raise ValueError("More channel ids have been specified than there are channels.")
        if channel_names:
            if len(channel_names) == 0:
                raise ValueError("No channel names have been specified.")
            if len(channel_names) > wrapper.reader.n_channels:
                raise ValueError("More channel names have been specified than there are channels.")

        # override as_uint8 if explicitly specified
        if isinstance(as_uint8_, bool):
            as_uint8 = as_uint8_
        logger.trace(f"Writing {name} to {filename}; channel_ids={channel_ids}; as_uint8={as_uint8}")
        path = writer.write(
            name,
            output_dir=self.image_dir,
            channel_ids=channel_ids,
            as_uint8=as_uint8,
            tile_size=tile_size,
            channel_names=channel_names,
            override=True,
        )
        return path

    def _transform_write_merge_images(
        self,
        to_original_size: bool = True,
        preview: bool = False,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        override: bool = False,
    ) -> list[Path]:
        from image2image_io.models.merge import MergeImages
        from image2image_io.writers.merge_tiff_writer import MergeOmeTiffWriter

        def _determine_attachment(_image: str):
            if _image in self.attachment_images.keys():
                return self.attachment_images[_image], True
            return None, False

        as_uint8_ = as_uint8
        merged_paths = []
        for merge_name, sub_images in self.merge_modalities.items():
            paths = []
            pixel_sizes = []
            channel_names = []
            channel_ids = []
            as_uint_all = []
            transformations: list[TransformSequence] = []
            non_reg_modalities = self._find_not_registered_modalities()
            for name in sub_images:
                attachment, attachment_modality = _determine_attachment(name)
                modality = self.modalities[name]
                paths.append(modality.path)
                pixel_sizes.append(modality.pixel_size)
                channel_names_ = modality.channel_names
                if modality.export:
                    channel_ids.append(modality.export.channel_ids)
                    as_uint_all.append(modality.export.as_uint8)
                    # if modality.export.channel_names:
                    #     channel_names_ = modality.export.channel_names
                channel_names.append(channel_names_)
                as_uint8 = all(as_uint_all)
                if name not in non_reg_modalities and attachment_modality not in non_reg_modalities:
                    _, sub_im_transforms, _ = self._prepare_registered_image_transform(
                        name,
                        attachment=attachment,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )
                else:
                    _, sub_im_transforms, _ = self._prepare_not_registered_image_transform(
                        name,
                        attachment=attachment,
                        attachment_modality=attachment_modality,
                        to_original_size=to_original_size,
                    )
                transformations.append(sub_im_transforms)

            # override as_uint8 if explicitly specified
            if isinstance(as_uint8_, bool):
                as_uint8 = as_uint8_

            if self.name == merge_name:
                filename = f"{self.name}_merged-registered"
            else:
                filename = f"{self.name}-{merge_name}_merged-registered"
            output_path = self.image_dir / filename
            if output_path.with_suffix(".ome.tiff").exists() and not override:
                logger.trace(f"Skipping {modality} as it already exists ({output_path}).")
                continue

            logger.trace(f"Writing {merge_name} to {output_path.name}; channel_ids={channel_ids}; as_uint8={as_uint8}")
            merge = MergeImages(paths, pixel_sizes, channel_names=channel_names)
            writer = MergeOmeTiffWriter(merge, transformers=transformations)
            path = writer.write(
                output_path.name,
                sub_images,
                output_dir=self.image_dir,
                tile_size=tile_size,
                as_uint8=as_uint8,
                channel_ids=channel_ids,
            )
            merged_paths.append(path)
        return merged_paths

    def save(self, registered: bool = False, auto: bool = False) -> Path:
        """Save configuration to file."""
        status = "registered" if registered is True else "setup"
        registration_paths = {}
        for index, edge in enumerate(self.registration_nodes):
            source = edge["modalities"]["source"]
            target = self.registration_paths[source][-1]
            through = None if len(self.registration_paths[source]) == 1 else self.registration_paths[source][0]
            source_preprocessing = edge["source_preprocessing"].to_dict() if edge["source_preprocessing"] else None
            target_preprocessing = edge["target_preprocessing"].to_dict() if edge["target_preprocessing"] else None
            registration_paths[f"reg_path_{index}"] = {
                "source": source,
                "target": target,
                "through": through,
                "reg_params": edge.get("params"),
                "source_preprocessing": source_preprocessing,
                "target_preprocessing": target_preprocessing,
            }

        # clean-up edges
        reg_graph_edges = []
        if registered:
            for reg_edge in self.registration_nodes:
                reg_graph_edges.append(
                    {
                        "modalities": reg_edge["modalities"],
                        "params": reg_edge["params"],
                        "registered": reg_edge["registered"],
                        "transform_tag": reg_edge["transform_tag"],
                        "source_preprocessing": reg_edge["source_preprocessing"].to_dict()
                        if reg_edge["source_preprocessing"]
                        else None,
                        "target_preprocessing": reg_edge["target_preprocessing"].to_dict()
                        if reg_edge["target_preprocessing"]
                        else None,
                    }
                )

        modalities_out: dict[str, dict] = {}
        for modality in self.modalities.values():
            modalities_out[modality.name] = modality.to_dict()

        # write config
        config: Config = {
            "schema_version": "1.0",
            "name": self.name,
            # "output_dir": str(self.project_dir),
            "cache_images": self.cache_images,
            # "cache_dir": str(self.cache_dir),
            "pairwise": self.pairwise,
            "modalities": modalities_out,
            "registration_paths": registration_paths,
            "registration_graph_edges": reg_graph_edges if registered else None,
            "original_size_transforms": self.original_size_transforms if registered else None,
            "attachment_geojsons": self.attachment_geojsons if len(self.attachment_geojsons) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge": self.merge_images,
            "merge_images": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }

        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"{ts}-{self.name}-configuration-{status}.json"
            if auto
            else (self.REGISTERED_CONFIG_NAME if registered else self.CONFIG_NAME)
        )
        path = self.project_dir / filename
        write_json_data(path, config)
        logger.trace(f"Saved configuration to '{path}'.")
        return path

    def save_to_wsireg(self, filename: PathLike | None = None, registered: bool = False) -> Path:
        """Save workflow configuration."""
        import yaml  # type: ignore[import]

        ts = time.strftime("%Y%m%d-%H%M%S")
        status = "registered" if registered is True else "setup"

        registration_paths = {}
        for index, edge in enumerate(self.registration_nodes):
            source = edge["modalities"]["source"]
            if len(self.registration_paths[source]) > 1:
                through = self.registration_paths[source][0]
            else:
                through = None
            target = self.registration_paths[source][-1]
            registration_paths[f"reg_path_{index}"] = {
                "src_modality_name": source,
                "tgt_modality_name": target,
                "thru_modality": through,
                "reg_params": edge.get("params"),
            }

        # clean-up edges
        reg_graph_edges = deepcopy(self.registration_nodes)
        [rge.pop("transforms", None) for rge in reg_graph_edges]  # type: ignore[misc]

        modalities_out: dict[str, dict] = {}
        for modality in self.modalities.values():
            modalities_out[modality.name] = modality.to_dict(as_wsireg=True)

        # write config
        config = {
            "project_name": self.name,
            "output_dir": str(self.project_dir),
            "cache_images": self.cache_images,
            "modalities": modalities_out,
            "reg_paths": registration_paths,
            "reg_graph_edges": reg_graph_edges if status == "registered" else None,
            "original_size_transforms": self.original_size_transforms if status == "registered" else None,
            "attachment_shapes": self.attachment_geojsons if len(self.attachment_geojsons) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge_modalities": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }

        if not filename:
            filename = self.project_dir / f"{ts}-{self.name}-configuration-{status}.yaml"
        filename = Path(filename)
        with open(str(filename), "w") as f:
            yaml.dump(config, f, sort_keys=False)
        return filename

    # def _create_initial_overlap_image(self):
    #     """Create image showing how images overlap before registration"""
    #     from itertools import combinations
    #
    #     from image2image_wsireg.utils.visuals import color_multichannel, get_n_colors, jzazbz_cmap
    #
    #     min_r = np.inf
    #     max_r = 0
    #     min_c = np.inf
    #     max_c = 0
    #     composite_img_list = [None] * len(self.modalities)
    #     for src_modality, tgt_modality in combinations(self.modalities.values(), 2):
    #
    #         img = img_obj.image
    #         padded_img = transform.warp(img, img_obj.T, preserve_range=True, output_shape=img_obj.padded_shape_rc)
    #
    #         composite_img_list[i] = padded_img
    #
    #         img_corners_rc = warp_tools.get_corners_of_image(img.shape[0:2])
    #         warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], img_obj.T)
    #         min_r = min(warped_corners_xy[:, 1].min(), min_r)
    #         max_r = max(warped_corners_xy[:, 1].max(), max_r)
    #         min_c = min(warped_corners_xy[:, 0].min(), min_c)
    #         max_c = max(warped_corners_xy[:, 0].max(), max_c)
    #
    #     composite_img = np.dstack(composite_img_list)
    #     cmap = jzazbz_cmap()
    #     channel_colors = get_n_colors(cmap, composite_img.shape[2])
    #     overlap_img = color_multichannel(
    #         composite_img, channel_colors, rescale_channels=True, normalize_by="channel", cspace="CAM16UCS"
    #     )
    #
    #     min_r = int(min_r)
    #     max_r = int(np.ceil(max_r))
    #     min_c = int(min_c)
    #     max_c = int(np.ceil(max_c))
    #     overlap_img = overlap_img[min_r:max_r, min_c:max_c]
    #     overlap_img = (255 * overlap_img).astype(np.uint8)
    #
    #     return overlap_img

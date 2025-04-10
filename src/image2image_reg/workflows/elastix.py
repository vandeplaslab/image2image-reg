"""Image registration in 2D."""

from __future__ import annotations

import time
import typing as ty
from contextlib import suppress
from copy import copy, deepcopy
from pathlib import Path
from warnings import warn

from image2image_io.config import CONFIG
from koyo.json import read_json_data, write_json_data
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from mpire import WorkerPool
from tqdm import tqdm

from image2image_reg._typing import (
    ElastixRegConfig,
    RegistrationNode,
    SerializedRegisteredRegistrationNode,
    SourceTargetPair,
    TransformPair,
)
from image2image_reg.elastix.registration import Registration
from image2image_reg.elastix.transform_sequence import Transform, TransformSequence
from image2image_reg.enums import WriterMode
from image2image_reg.models import Modality, Preprocessing
from image2image_reg.utils.utilities import make_new_name
from image2image_reg.workflows._base import Workflow

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.valis import ValisReg
    from image2image_reg.wrapper import ImageWrapper

# override certain parameters
CONFIG.init_pyramid = False
CONFIG.only_last_pyramid = False


class ElastixReg(Workflow):
    """Whole slide registration utilizing WsiReg approach of graph based registration."""

    CONFIG_NAME = "project.config.json"
    REGISTERED_CONFIG_NAME = "registered-project.config.json"
    EXTENSION = ".wsireg"

    log_file: Path | None = None
    _name: str | None = None

    def __init__(
        self,
        name: str | None = None,
        output_dir: PathLike | None = None,
        project_dir: PathLike | None = None,
        cache: bool = True,
        merge: bool = False,
        pairwise: bool = False,
        log: bool = False,
        init: bool = True,
        **_kwargs: ty.Any,
    ):
        super().__init__(
            name=name,
            output_dir=output_dir,
            project_dir=project_dir,
            cache=cache,
            merge=merge,
            log=log,
            init=init,
            **_kwargs,
        )
        # setup project directory
        self.pairwise = pairwise

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

    @property
    def transformations_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "Transformations"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def final_transformations_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "FinalTransformations"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def overlap_dir(self) -> Path:
        """Results directory."""
        directory = self.project_dir / "Overlap"
        directory.mkdir(exist_ok=True, parents=True)
        return directory

    @property
    def is_registered(self) -> bool:
        """Check if the project has been registered."""
        return all(reg_edge["registered"] for reg_edge in self.registration_nodes)

    @classmethod
    def from_path(cls, path: PathLike, raise_on_error: bool = True, quick: bool = False) -> ElastixReg:
        """Initialize based on the project path."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist ({path}).")
        if path.is_file() and path.name in [cls.CONFIG_NAME, cls.REGISTERED_CONFIG_NAME]:
            path = path.parent
        if not path.is_dir():
            raise ValueError("Path is not a directory.")
        config_path = path / cls.CONFIG_NAME
        if not path.suffix == ".wsireg" and not config_path.exists():
            raise ValueError("Path is not a valid ElastiX project.")

        with MeasureTimer() as timer:
            config: dict | ElastixRegConfig | None = None
            if config_path.exists():
                config = read_json_data(config_path)
            if config and "name" not in config:
                config["name"] = path.stem
            if config and "pairwise" not in config:
                config["pairwise"] = False
            if config and "attachment_geojsons" in config:
                config["attachment_shapes"] = config.pop("attachment_geojsons")
            if config is None:
                config = {}

            obj = cls(project_dir=path, **config)
            if config_path.exists():
                obj.load_from_i2reg(raise_on_error=raise_on_error, quick=quick)
            elif list(obj.project_dir.glob("*.yaml")):
                obj.load_from_wsireg()
        logger.trace(f"Restored from config in {timer()}")
        return obj

    def load_preprocessed_cache(self) -> None:
        """Load cache of sizes and shapes."""
        from image2image_reg.wrapper import ImageWrapper

        for modality_name in self.get_image_modalities(with_attachment=False):
            modality = self.modalities[modality_name]
            if modality.name not in self.preprocessed_cache["image_spacing"]:
                wrapper = ImageWrapper(modality, quick=True, quiet=True)
                self.preprocessed_cache["image_spacing"][modality.name] = wrapper.reader.scale
                self.preprocessed_cache["image_sizes"][modality.name] = wrapper.reader.image_shape[::-1]

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

        # func information about attachment images
        func(f"Number of attachment images: {len(self.attachment_images)}")
        n = len(self.attachment_images) - 1
        for i, (name, attach_to) in enumerate(self.attachment_images.items()):
            func(f" {elbow if i == n else tee}{name} ({attach_to})")
        # func information about attachment shapes
        func(f"Number of attachment shapes: {len(self.attachment_shapes)}")
        n = len(self.attachment_shapes) - 1
        for i, (name, shape_dict) in enumerate(self.attachment_shapes.items()):
            func(f" {elbow if i == n else tee}{name}")
            func(f" {pipe if i != n else blank}{tee}Attached to: {shape_dict['attach_to']}")
            func(f" {pipe if i != n else blank}{tee}Pixel size: {shape_dict['pixel_size']}")
            nn = len(shape_dict["files"]) - 1
            for j, file in enumerate(shape_dict["files"]):
                func(f" {pipe if i != n else blank}{tee if j != nn else elbow}File: {file}")

        # func information about attachment shapes
        func(f"Number of attachment points: {len(self.attachment_points)}")
        n = len(self.attachment_points) - 1
        for i, (name, shape_dict) in enumerate(self.attachment_points.items()):
            func(f" {elbow if i == n else tee}{name}")
            func(f" {pipe if i != n else blank}{tee}Attached to: {shape_dict['attach_to']}")
            func(f" {pipe if i != n else blank}{tee}Pixel size: {shape_dict['pixel_size']}")
            nn = len(shape_dict["files"]) - 1
            for j, file in enumerate(shape_dict["files"]):
                func(f" {pipe if i != n else blank}{tee if j != nn else elbow}File: {file}")

        # func information about merge modalities
        func(f"Number of merge modalities: {len(self.merge_modalities)}")
        n = len(self.merge_modalities) - 1
        for i, (name, merge_modalities) in enumerate(self.merge_modalities.items()):
            func(f" {elbow if i == n else tee}{name} ({merge_modalities})")

    def _is_reference(self, modality: Modality) -> bool:
        """Check whether modality is reference, which doesn't have to be registered."""
        return (
            all(targets[-1] == modality.name for targets in self.registration_paths.values())
            if self.registration_paths
            else False
        )

    def _is_being_registered(self, modality: Modality, check_target: bool = False) -> bool:
        """Check whether the modality will be registered or is simply an attachment."""
        if modality.name in self.attachment_images:
            return False
        if check_target:
            for edge in self.registration_nodes:
                if modality.name == edge["modalities"]["target"]:
                    return True
        if modality.name not in self.registration_paths:
            return False
        return True

    def validate_paths(self, log: bool = True) -> tuple[bool, list[str]]:
        """Validate paths."""
        errors = []
        for name in self.get_image_modalities(with_attachment=False):
            modality = self.get_modality(name=name)
            if (
                modality is not None
                and not self._is_being_registered(modality, check_target=True)
                and not self._is_reference(modality)
            ):
                errors.append(f"❌ Modality '{name}' has been defined but isn't being registered.")

        # check whether all registration paths exist
        for source, targets in self.registration_paths.items():
            if source not in self.modalities:
                errors.append(f"❌ Source modality '{source}' does not exist.")
            for target in targets:
                if target not in self.modalities:
                    errors.append(f"❌ Target modality '{target}' does not exist.")
                if source == target:
                    errors.append("❌ Source and target modalities cannot be the same.")

        # check through modalities
        for targets in self.registration_paths.values():
            if len(targets) == 2:
                modality = self.get_modality(name=targets[0])
                if modality is not None and not self._is_being_registered(modality, check_target=False):
                    errors.append(f"❌ Through modality '{targets[0]}' has been defined but isn't being registered.")

        # log errors
        if log and errors:
            for error in errors:
                logger.error(error)
        elif log and not errors:
            logger.success("✅ Project paths are valid.")
        return len(errors) == 0, errors

    def validate(self, allow_not_registered: bool = True, require_paths: bool = False) -> tuple[bool, list[str]]:
        """Perform several checks on the project."""
        # check whether the paths are still where they were set up
        errors = []
        if not self.modalities:
            errors.append("❌ No modalities have been added.")
            logger.error(errors[-1])
        # check if the paths exist
        for modality in self.modalities.values():
            if isinstance(modality.path, (str, Path)) and not Path(modality.path).exists():
                errors.append(f"❌ Modality '{modality.name}' path '{modality.path}' does not exist.")
                logger.error(errors[-1])
            else:
                logger.success(f"✅ Modality '{modality.name}' exist.")

        # check whether all modalities exist
        for edge in self.registration_nodes:
            modalities = edge["modalities"]
            if modalities["source"] not in self.modalities:
                errors.append(f"❌ Source modality '{modalities['source']}' does not exist.")
                logger.error(errors[-1])
            elif modalities["target"] not in self.modalities:
                errors.append(f"❌ Target modality '{modalities['target']}' does not exist.")
                logger.error(errors[-1])
            elif modalities["source"] == modalities["target"]:
                errors.append("❌ Source and target modalities cannot be the same.")
                logger.error(errors[-1])
            else:
                logger.success(f"✅ Modality pair {modalities['source']} - {modalities['target']} exist.")
        if require_paths:
            if not self.registration_nodes:
                errors.append("❌ No registration paths have been added.")
                logger.error(errors[-1])

        # check whether all registration paths exist
        for source, targets in self.registration_paths.items():
            if source not in self.modalities:
                errors.append(f"❌ Source modality '{source}' does not exist.")
                logger.error(errors[-1])
            for target in targets:
                if target not in self.modalities:
                    errors.append(f"❌ Target modality '{target}' does not exist.")
                    logger.error(errors[-1])
                if source == target:
                    errors.append("❌ Source and target modalities cannot be the same.")
                    logger.error(errors[-1])

        # check whether all modalities have been registered
        if not allow_not_registered:
            for edge in self.registration_nodes:
                if not edge["registered"]:
                    errors.append(
                        f"❌ Modality pair {edge['modalities']['source']} - {edge['modalities']['target']} "
                        f"has not been registered."
                    )
                    logger.error(errors[-1])
                else:
                    logger.success(
                        f"✅ Modality pair {edge['modalities']['source']} - {edge['modalities']['target']} "
                        f"has been registered."
                    )

            # check through modalities
            for targets in self.registration_paths.values():
                if len(targets) == 2:
                    modality = self.get_modality(name=targets[0])
                    if modality is not None and not self._is_being_registered(modality, check_target=False):
                        errors.append(
                            f"❌ Through modality '{targets[0]}' has been defined but isn't being registered."
                        )

        is_valid = not errors
        if not is_valid:
            errors.append("❌ Project configuration is invalid.")
            logger.error(errors[-1])
        else:
            logger.success("✅ Project configuration is valid.")
        return is_valid, errors

    def load_from_i2reg(self, raise_on_error: bool = True, quick: bool = False) -> None:
        """Load data from image2image-reg project file."""
        config: ElastixRegConfig = read_json_data(self.project_dir / self.CONFIG_NAME)
        name = config.get("name")
        if name != self.project_dir.stem:
            name = self.project_dir.stem
            logger.trace(f"Name in config does not match directory name. Using '{name}' instead.")
        # restore parameters
        self.name = name
        self.cache_images = config["cache_images"]
        self.pairwise = config["pairwise"]
        self.merge_images = config["merge"]

        # add modality information
        with MeasureTimer() as timer:
            self._load_modalities_from_config(config, raise_on_error)
            # load attachment images
            self._load_attachment_from_config(config)
            # load merge
            self._load_merge_from_config(config)

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
            if not quick and (self.project_dir / self.REGISTERED_CONFIG_NAME).exists():
                registered_config: ElastixRegConfig = read_json_data(self.project_dir / self.REGISTERED_CONFIG_NAME)
                logger.trace(f"Loading registered configuration from {self.REGISTERED_CONFIG_NAME}.")
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

    def _load_registered_transform(
        self,
        edge: SerializedRegisteredRegistrationNode,
        target: str,
        raise_on_error: bool = True,
    ) -> dict[str, TransformSequence | None] | None:
        """Load registered transform and make sure all attributes are correctly set-up."""
        from image2image_reg.wrapper import ImageWrapper

        index, edge_ = self._find_edge_by_edge(edge)
        if not edge_:
            logger.warning("Could not find appropriate registration node.")
            return None

        with MeasureTimer() as timer:
            source = edge["modalities"]["source"]
            transform_tag = f"{source}_to_{target}_transformations.json"
            if transform_tag and not (self.transformations_dir / transform_tag).exists():
                logger.warning(f"Could not find cached registration data. ('{transform_tag}' file does not exist)")
                transform_tag = f"{self.name}-{source}_to_{target}_transformations.json"
                if transform_tag and not (self.transformations_dir / transform_tag).exists():
                    logger.warning(f"Could not find cached registration data. ('{transform_tag}' file does not exist)")
                    return None

            target_modality = self.modalities[target]
            target_wrapper = ImageWrapper(
                target_modality, edge["target_preprocessing"], quick=True, quiet=True, raise_on_error=raise_on_error
            )
            self.original_size_transforms[target_wrapper.name] = target_wrapper.original_size_transform

            source_modality = self.modalities[source]
            source_wrapper = ImageWrapper(
                source_modality, edge["source_preprocessing"], quick=True, quiet=True, raise_on_error=raise_on_error
            )
            source_wrapper.initial_transforms = source_wrapper.load_initial_transform(source_modality, self.cache_dir)

            initial_transforms_seq = None
            if source_wrapper.initial_transforms:
                initial_transforms_ = [Transform(t) for t in source_wrapper.initial_transforms]
                initial_transforms_index = [idx for idx, _ in enumerate(initial_transforms_)]
                initial_transforms_seq = TransformSequence(initial_transforms_, initial_transforms_index)

            transforms_partial_seq, transforms_full_seq = TransformSequence.read_partial_and_full(
                self.transformations_dir / transform_tag
            )

            # setup parameters
            edge_["transforms"] = {"registration": transforms_partial_seq, "initial": initial_transforms_seq}
            edge_["registered"] = True
            edge_["transform_tag"] = f"{source}_to_{target}_transformations.json"
            self.registration_nodes[index] = edge_
        logger.trace(f"Restored previous transformation data for {source} - {target} in {timer()}")
        return {
            f"initial-{source}": initial_transforms_seq,
            f"000-to-{target}": transforms_partial_seq,
            "full-transform-seq": transforms_full_seq,
        }

    def _find_edge_by_edge(
        self, edge: SerializedRegisteredRegistrationNode
    ) -> tuple[int | None, RegistrationNode | None]:
        """Find edge by another edge, potentially from cache."""
        for index, edge_ in enumerate(self.registration_nodes):
            if edge_["modalities"]["source"] == edge["modalities"]["source"]:
                if edge_["modalities"]["target"] == edge["modalities"]["target"]:
                    return index, edge_
        return None

    def load_from_wsireg(self) -> None:
        """Load data from WsiReg YAML project file."""

    def rename_modality(self, old_name: str, new_name: str) -> None:
        """Rename modality."""
        super().rename_modality(old_name, new_name)

        # rename registration paths
        for edge in self.registration_nodes:
            if edge["modalities"]["source"] == old_name:
                edge["modalities"]["source"] = new_name
            if edge["modalities"]["target"] == old_name:
                edge["modalities"]["target"] = new_name
        registration_paths = deepcopy(self.registration_paths)
        for source, targets in self.registration_paths.items():
            if source == old_name:
                registration_paths[new_name] = registration_paths.pop(old_name)
            if old_name in targets:
                targets[targets.index(old_name)] = new_name
                registration_paths[source] = targets
        self.registration_paths = registration_paths
        self._create_transformation_paths(self.registration_paths)

    @property
    def n_registrations(self) -> int:
        """Number of registrations to be performed."""
        return len(self.registration_nodes)

    def has_registration_path(self, source: str, target: str, through: str | None = None) -> bool:
        """Check whether registration path exists."""
        modalities = {"source": source, "target": through if through else target}
        for node in self.registration_nodes:
            if node["modalities"] == modalities:
                return True
        return False

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
        # check whether the (source, target) pair already exists
        if self.has_registration_path(source, target, through):
            logger.warning(f"Registration path from '{source}' to '{target}' through '{through}' already exists.")
            self.update_registration_path(source, target, transform, through, preprocessing)
        else:
            self._add_registration_node(source, target, through, transform, preprocessing)

    def update_registration_path(
        self,
        source: str,
        target: str,
        transform: str | ty.Iterable[str],
        through: str | None = None,
        preprocessing: dict | None = None,
    ):
        """Update registration path."""
        if not self.has_registration_path(source, target, through):
            return
        self.remove_registration_path(source, target, through)
        self.add_registration_path(source, target, transform, through, preprocessing)

    def find_index_of_registration_path(self, source: str, target: str, through: str | None = None) -> int | None:
        """Remove registration path."""
        index = None
        modalities = {"source": source, "target": through if through else target}
        for i, node in enumerate(self.registration_nodes):
            if node["modalities"] == modalities:
                index = i
                break
        if index is not None:
            return index
        return None

    def remove_registration_path(self, source: str, target: str, through: str | None = None) -> None:
        """Remove registration path."""
        index = self.find_index_of_registration_path(source, target, through)
        if index is not None:
            self.registration_nodes.pop(index)
            logger.trace(f"Removed registration path from '{source}' to '{target}' through '{through}'.")
        if source in self.registration_paths:
            self.registration_paths.pop(source)
        self._create_transformation_paths(self.registration_paths)

    def reset_registration_paths(self) -> None:
        """Reset registration paths."""
        self.registration_nodes.clear()
        self.registration_paths.clear()
        self._create_transformation_paths(self.registration_paths)

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
                source_preprocessing = preprocessing["source"]
                if isinstance(source_preprocessing, dict):
                    source_preprocessing = Preprocessing(**source_preprocessing)  # type: ignore[arg-type]
            if preprocessing.get("target"):
                target_preprocessing = preprocessing["target"]
                if isinstance(target_preprocessing, dict):
                    target_preprocessing = Preprocessing(**target_preprocessing)  # type: ignore[arg-type]

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
        self,
        modality: Modality,
        preprocessing: Preprocessing | None = None,
        overwrite: bool = False,
        quick: bool = False,
    ) -> ImageWrapper:
        """Pre-process images."""
        from image2image_reg.wrapper import ImageWrapper

        wrapper = ImageWrapper(modality, preprocessing, quick=True)
        cached = wrapper.check_cache(self.cache_dir, self.cache_images) if not overwrite else False
        if not cached:
            logger.trace(f"'{modality.name}' is not cached - pre-processing...")
            wrapper.preprocess()
            wrapper.save_cache(self.cache_dir, self.cache_images)
        else:
            if not quick:
                logger.trace(f"Loading cached '{modality.name}' image.")
                wrapper.load_cache(self.cache_dir, self.cache_images)
        if not quick:
            if wrapper.image is None:
                raise ValueError(f"'{modality.name}' image has not been pre-processed.")
            # update caches
            spacing = wrapper.image.GetSpacing()  # type:ignore[no-untyped-call]
            self.preprocessed_cache["image_spacing"][modality.name] = spacing
            size = wrapper.image.GetSize()  # type:ignore[no-untyped-call]
            self.preprocessed_cache["image_sizes"][modality.name] = size
        return wrapper

    @staticmethod
    def __preprocess_image(
        cache_dir: Path, cache_images: bool, modality: Modality, preprocessing: Preprocessing | None = None
    ) -> ImageWrapper:
        """Pre-process image."""
        from image2image_reg.wrapper import ImageWrapper

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
        from image2image_reg.elastix.registration_utils import (
            _prepare_reg_models,
            register_2d_images,
            sitk_pmap_to_dict,
        )

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
        from image2image_reg.elastix.figures import (
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
                    transforms[modality][f"{str(index).zfill(3)}-to-{edges[index]['target']}"] = (
                        registered_edge_transform["registration"]
                    )
                    full_tform_seq.append(registered_edge_transform["registration"])
                transforms[modality]["full-transform-seq"] = full_tform_seq
        return transforms

    def _remove_modality(self, modality: Modality) -> None:
        """Remove modality from project.

        This function should handle registration paths.
        """
        if self._is_being_registered(modality):
            logger.warning("Removed modality that is being used in registration paths.")
        name = modality.name

        # remove nodes
        to_remove_nodes = []
        for index, node in enumerate(self.registration_nodes):
            if name == node["modalities"]["source"]:
                to_remove_nodes.append(index)
            elif name == node["modalities"]["target"]:
                to_remove_nodes.append(index)
        if to_remove_nodes:
            for index in reversed(to_remove_nodes):
                node = self.registration_nodes.pop(index)
                logger.warning(f"Removed registration node ({node}).")
        # remove paths
        to_remove_paths = []
        for source, targets in self.registration_paths.items():
            if source == name:
                to_remove_paths.append(source)
            if name in targets:
                to_remove_paths.append(source)
        for source in to_remove_paths:
            self.registration_paths.pop(source)
            logger.warning(f"Removed registration path from {source}.")

    def preprocess(self, n_parallel: int = 1, overwrite: bool = False, quick: bool = False) -> None:
        """Pre-process all images."""
        # TODO: add multi-core support
        self.set_logger()

        with MeasureTimer() as timer:
            # to_preprocess = []
            # for modality in self.modalities.values():
            #     to_preprocess.append(modality.name)
            #
            # if n_parallel and len(to_preprocess) > 1:
            #     with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
            #         pool.imap_unordered(self._preprocess_image, to_preprocess)
            for modality in tqdm(self.modalities.values(), desc="Pre-processing images"):
                if not self._is_being_registered(modality):
                    logger.trace(f"Skipping pre-processing for {modality.name} as it's not being registered.")
                    continue
                logger.trace(f"Pre-processing {modality.name}.")
                # TODO: allow extra pre-processing specification
                self._preprocess_image(modality, None, overwrite=overwrite, quick=quick)
        logger.info(f"Pre-processing of all images took {timer(since_last=True)}.")

    def register(self, n_parallel: int = 1, preprocess_first: bool = True, histogram_match: bool = False) -> None:
        """Co-register images."""
        # TODO: add multi-core support
        self.set_logger()
        if self.is_registered:
            logger.info("Project has already been registered.")
            return

        self.save(registered=False)
        # check whether registration nodes have been specified
        if not self.registration_nodes:
            raise ValueError("No registration paths have been defined.")
        if preprocess_first:
            self.preprocess(n_parallel=n_parallel)

        # compute transformation information
        for edge in tqdm(
            reversed(self.registration_nodes), desc="Registering nodes...", total=len(self.registration_nodes)
        ):
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
            edge["transform_tag"] = f"{source}_to_{target}_transformations.json"

            # load plot data
            self._generate_figures(source, target, registration_dir)

        # collect transformations
        self.transformations = self._collate_transformations()
        # save transformations
        self.save_transformations()
        self.save(registered=True)

    def clear(
        self,
        cache: bool = False,
        image: bool = False,
        transformations: bool = False,
        final: bool = False,
        progress: bool = False,
        clear_all: bool = False,
    ) -> None:
        """Clear existing data."""
        from image2image_reg.utils.utilities import _safe_delete

        if clear_all:
            cache = image = transformations = progress = final = True

        # clear transformations, cache, images
        if cache:
            for file in self.cache_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.cache_dir)

        if progress:
            for file in self.progress_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.progress_dir)

            for file in self.overlap_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.overlap_dir)

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

        if final:
            for file in self.final_transformations_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.final_transformations_dir)

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
            logger.warning("Cannot save transformation as not all modalities have been registered.")
            return None

        out = []
        for source_modality in self.registration_paths:
            logger.trace(f"Saving transformations for {source_modality}...")
            target_modalities = self.registration_paths[source_modality]
            target_modality = target_modalities[-1]
            output_path = self.transformations_dir / f"{source_modality}_to_{target_modality}_transformations.json"
            tform_txt = self._transforms_to_txt(self.transformations[source_modality])
            write_json_data(output_path, tform_txt)
            out.append(output_path)
            logger.trace(
                f"Saved transformations to '{output_path}'. source={source_modality}; target={target_modalities}."
            )

        not_registered_modalities = self._find_not_registered_modalities()
        for attached_modality, attached_to_modality in self.attachment_images.items():
            if attached_to_modality not in not_registered_modalities:
                target_modalities = self.registration_paths[attached_to_modality]
                target_modality = target_modalities[-1]
                output_path = (
                    self.transformations_dir / f"{attached_modality}_to_{target_modality}_transformations.json"
                )
                tform_txt = self._transforms_to_txt(self.transformations[attached_to_modality])
                write_json_data(output_path, tform_txt)
                logger.trace(
                    f"Saved transformations to '{output_path}'. source={attached_to_modality}; "
                    f"target={target_modalities}."
                )
        return out

    def preview(self, pyramid: int = -1, overwrite: bool = False, **kwargs: ty.Any) -> None:
        """Preview registration."""
        self.load_preprocessed_cache()
        self._generate_overlap_image(pyramid=pyramid, overwrite=overwrite)

    def _generate_overlap_image(self, pyramid: int = -1, overwrite: bool = False) -> None:
        """Generate overlap of images."""
        from koyo.visuals import save_gray, save_rgb

        from image2image_reg.utils.visuals import create_overlap_img

        if not self.is_registered:
            logger.warning("Project has not been registered. Cannot generate overlap images.")
            return

        # let's iterate through all registration paths
        logger.trace(f"Generating overlap images at pyramid level of {pyramid}")
        images_from_all = []
        name_to_gray = {}
        for source, _target_pair in self.registration_paths.items():
            images, greys, names = self._generate_overlap_image_for_modality(source, pyramid, overwrite)
            images_from_all.extend(images)
            for name, grey in zip(names, greys):
                name_to_gray[name] = grey

        # export gray-scale images
        for name, grey in name_to_gray.items():
            path = self.overlap_dir / f"grey_{name}.png"
            if path.exists():
                continue
            save_gray(path, grey, multiplier=1)
            logger.trace(f"Saved greyscale image to '{path}'.")

        # also export all images but don't bother if it's a single image
        if len(images_from_all) <= 2:
            return
        try:
            overlap, _ = create_overlap_img(images_from_all)
            path = self.overlap_dir / "overlap_all.png"
            save_rgb(path, overlap)
            logger.trace(f"Saved overlap image to '{path}'.")
        except Exception as e:
            logger.error(f"Could not save overlap image of all images. {e}")

    def _generate_overlap_image_for_modality(
        self, source: str, pyramid: int = -1, overwrite: bool = False
    ) -> tuple[list, list, list]:
        from image2image_io.utils.utilities import get_shape_of_image
        from koyo.visuals import save_rgb

        from image2image_reg.elastix.transform import transform_images_for_pyramid
        from image2image_reg.utils.visuals import create_overlap_img

        images, greys, names = [], [], []  # type: ignore[var-annotated]
        target_pair = self.registration_paths[source]
        target, through = (target_pair[-1], target_pair[0]) if len(target_pair) == 2 else (target_pair[0], None)
        path = self.overlap_dir / f"overlap_{source}_to_{target}.png"
        if path.exists() and not overwrite:
            logger.warning(f"Overlap image already exists at '{path}'. Skipping.")
            return images, greys, names

        with MeasureTimer() as timer:
            target_modality = self.modalities[target]
            target_wrapper = self.get_wrapper(name=target_modality.name)
            assert target_wrapper, f"Could not find wrapper for {target_modality.name}"
            _, transform_seq, _ = self._prepare_transform(target_modality.name)
            if transform_seq:
                shape = target_wrapper.reader.pyramid[pyramid].shape
                _, _, shape = get_shape_of_image(shape)
                transform_seq.set_output_spacing(target_wrapper.reader.scale_for_pyramid(pyramid), shape[::-1])
            target_image = transform_images_for_pyramid(target_wrapper, transform_seq, pyramid)
            logger.trace(f"Transformed {target} in {timer(since_last=True)}")
            _, _, shape = get_shape_of_image(target_image)
            # TODO: resample to maximum shape e.g. 1000px
            images.append(target_image)
            names.append(target_modality.name)
            logger.trace(f"Previewing overview with {shape} ({pyramid})")

            if through:
                through_modality = self.modalities[through]
                through_wrapper = self.get_wrapper(name=through_modality.name)
                assert through_wrapper, f"Could not find wrapper for {through_modality.name}"
                _, transform_seq, _ = self._prepare_transform(through_modality.name)
                assert transform_seq is not None, f"Transformation is None for {through_modality.name}"
                transform_seq.set_output_spacing(target_wrapper.reader.scale_for_pyramid(pyramid), shape[::-1])
                images.append(transform_images_for_pyramid(through_wrapper, transform_seq, pyramid))
                names.append(through_modality.name)
                logger.trace(f"Transformed {through} in {timer(since_last=True)}")

            source_modality = self.modalities[source]
            source_wrapper = self.get_wrapper(name=source_modality.name)
            assert source_wrapper, f"Could not find wrapper for {source_modality.name}"
            _, transform_seq, _ = self._prepare_transform(source_modality.name)
            assert transform_seq is not None, f"Transformation is None for {source_modality.name}"
            transform_seq.set_output_spacing(target_wrapper.reader.scale_for_pyramid(pyramid), shape[::-1])
            images.append(transform_images_for_pyramid(source_wrapper, transform_seq, pyramid))
            names.append(source_modality.name)
            logger.trace(f"Transformed {source} in {timer(since_last=True)}")

            # create overlap images
            overlap, greys = create_overlap_img(images)

            # export gray-scale images
            save_rgb(path, overlap)
            logger.trace(f"Saved overlap image to '{path}'.")
        return images, greys, names

    def write(
        self,
        fmt: WriterMode = "ome-tiff",
        write_registered: bool = True,
        write_not_registered: bool = True,
        write_attached: bool = False,
        write_attached_images: bool | None = None,
        write_attached_points: bool | None = None,
        write_attached_shapes: bool | None = None,
        write_merged: bool = True,
        remove_merged: bool = True,
        to_original_size: bool = True,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        clip: str = "ignore",
        rename: bool = False,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> list[Path] | None:
        """Export images after applying transformation."""
        self.set_logger()
        if not self._check_if_all_registered():
            logger.warning("Cannot write images as not all modalities have been registered.")
            return None
        if write_attached:
            write_attached_images = write_attached_points = write_attached_shapes = True

        if n_parallel is None or n_parallel < 1:
            n_parallel = 1

        # update cache
        self.load_preprocessed_cache()

        paths = []
        reg_modality_list, not_reg_modality_list, merge_modalities = self._get_modalities_to_transform()
        if remove_merged:
            for merge_modality in merge_modalities:
                with suppress(ValueError):
                    index = reg_modality_list.index(merge_modality)
                    reg_modality_list.pop(index)
                    logger.trace(f"Removed {merge_modality} from registered modalities as it will be merged.")
                with suppress(ValueError):
                    index = not_reg_modality_list.index(merge_modality)
                    not_reg_modality_list.pop(index)
                    logger.trace(f"Removed {merge_modality} from not registered modalities as it will be merged.")

        # export non-registered nodes
        if write_not_registered and not_reg_modality_list:
            paths.extend(
                self._export_not_registered_images(
                    not_reg_modality_list,
                    fmt=fmt,
                    to_original_size=to_original_size,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                    rename=rename,
                    n_parallel=n_parallel,
                    overwrite=overwrite,
                )
            )

        # export modalities
        if write_registered and reg_modality_list:
            paths.extend(
                self._export_registered_images(
                    reg_modality_list,
                    fmt=fmt,
                    to_original_size=to_original_size,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                    rename=rename,
                    n_parallel=n_parallel,
                    overwrite=overwrite,
                )
            )

        if write_attached_shapes:
            paths.extend(
                self._export_attachment_shapes(n_parallel=n_parallel, clip=clip, rename=rename, overwrite=overwrite)
            )
        if write_attached_points:
            paths.extend(
                self._export_attachment_points(n_parallel=n_parallel, clip=clip, rename=rename, overwrite=overwrite)
            )
        if write_attached_images:
            paths.extend(
                self._export_attachment_images(
                    merge_modalities,
                    remove_merged,
                    to_original_size=to_original_size,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                    rename=rename,
                    n_parallel=n_parallel,
                    overwrite=overwrite,
                )
            )

        # export merge modalities
        if write_merged and self.merge_modalities:
            path = self._transform_write_merge_images(
                to_original_size=to_original_size, as_uint8=as_uint8, overwrite=overwrite
            )
            paths.append(path)
        return paths

    def export_transforms(
        self,
        write_registered: bool = True,
        write_not_registered: bool = True,
        write_attached: bool = True,
        to_original_size: bool = False,
        rename: bool = False,
    ) -> None:
        """Export transform data."""

        def _get_filename(filename: Path) -> Path:
            """Get filename for transformation file."""
            name = filename.stem.replace(".ome", "")
            return (self.final_transformations_dir / name).with_suffix(".elastix.json")

        if not self.is_registered:
            return

        self.load_preprocessed_cache()
        reg_modality_list, not_reg_modality_list, merge_modalities = self._get_modalities_to_transform()
        if write_registered and reg_modality_list:
            for modality in reg_modality_list:
                _, transform_seq, filename = self._prepare_transform(
                    modality, to_original_size=to_original_size, rename=rename
                )
                filename = _get_filename(filename)
                if transform_seq:
                    transform_seq.to_json(filename)
                    logger.info(f"Exported {transform_seq} for {modality} to {filename} (registered)")
                else:
                    logger.warning(f"No transformation file found for modality {modality} (registered)")

        if write_not_registered and reg_modality_list:
            for modality in not_reg_modality_list:
                _, transform_seq, filename = self._prepare_transform(
                    modality, to_original_size=to_original_size, rename=rename
                )
                filename = _get_filename(filename)
                if transform_seq:
                    transform_seq.to_json(filename)
                    logger.info(f"Exported {transform_seq} for {modality} to {filename} (not-registered)")
                else:
                    logger.warning(f"No transformation file found for modality {modality} (not-registered)")

        if write_attached and self.attachment_images:
            for modality, attach_to_modality_key in tqdm(
                self.attachment_images.items(), desc="Exporting attachment images..."
            ):
                attached_modality = self.modalities[modality]
                _, transform_seq, filename = self._prepare_transform(
                    attach_to_modality_key,
                    to_original_size=to_original_size,
                    rename=rename,
                    attachment=True,
                    attachment_modality=attached_modality,
                )
                filename = _get_filename(filename)
                if transform_seq:
                    transform_seq.to_json(filename)
                    logger.info(f"Exported {transform_seq} for {modality} to {filename} (attached)")
                else:
                    logger.warning(f"No transformation file found for modality {modality} (attached)")

    def _get_modalities_to_transform(self) -> tuple[list[str], list[str], list[str]]:
        # prepare merge modalities metadata
        merge_modalities = self._find_merge_modalities()
        if merge_modalities:
            logger.trace(f"Merge modalities: {merge_modalities}")
        reg_modality_list = list(self.registration_paths.keys())
        if reg_modality_list:
            logger.trace(f"Registered modalities: {reg_modality_list}")

        not_reg_modality_list = self._find_not_registered_modalities()
        if not_reg_modality_list:
            logger.trace(f"Not registered modalities: {not_reg_modality_list}")
        return reg_modality_list, not_reg_modality_list, merge_modalities

    def _export_registered_images(
        self,
        modalities: list[str],
        fmt: str | WriterMode = "ome-tiff",
        to_original_size: bool = True,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        rename: bool = False,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> list[Path]:
        paths = []
        to_write = []
        for name in tqdm(modalities, desc="Exporting registered modalities...", total=len(modalities)):
            modality, transform_seq, output_path = self._prepare_transform(
                name, to_original_size=to_original_size, rename=rename
            )

            if _get_with_suffix(output_path, fmt).exists() and not overwrite:
                logger.trace(f"Skipping {name} as it already exists. ({output_path})")
                continue
            logger.trace(f"Exporting {name} to {output_path}... (registered)")
            to_write.append(
                (
                    modality.name,
                    transform_seq,
                    output_path,
                    fmt,
                    tile_size,
                    as_uint8,
                    # lambda: (False, None, to_original_size),
                )
            )
        if to_write:
            if n_parallel > 1:
                with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
                    res = pool.imap(self._transform_write_image, to_write)
                paths.extend(list(res))
            else:
                for args in tqdm(to_write, desc="Exporting registered modalities (registered)..."):
                    path = self._transform_write_image(*args)
                    paths.append(path)
        return paths

    def _export_not_registered_images(
        self,
        modalities: list[str],
        fmt: str | WriterMode = "ome-tiff",
        to_original_size: bool = True,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        rename: bool = False,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> list[Path]:
        # preprocess and save unregistered nodes
        paths = []
        to_write = []
        for name in tqdm(modalities, desc="Exporting not-registered images..."):
            try:
                modality, transform_seq, output_path = self._prepare_transform(
                    name, to_original_size=to_original_size, rename=rename
                )
                if _get_with_suffix(output_path, fmt).exists() and not overwrite:
                    logger.trace(f"Skipping {name} as it already exists ({output_path}).")
                    continue
                logger.trace(f"Exporting {name} to {output_path}... (registered)")
                to_write.append(
                    (
                        modality,
                        transform_seq,
                        output_path,
                        fmt,
                        tile_size,
                        as_uint8,
                    )
                )
            except KeyError:
                logger.exception(f"Could not find transformation data for {name}.")
        if to_write:
            if n_parallel > 1:
                with WorkerPool(n_jobs=n_parallel, use_dill=True) as pool:
                    res = pool.imap(self._transform_write_image, to_write)
                paths.extend(list(res))
            else:
                for args in tqdm(to_write, desc="Exporting not-registered modalities..."):
                    path = self._transform_write_image(*args)
                    paths.append(path)
        return paths

    def _export_attachment_shapes(
        self, n_parallel: int = 1, clip: str = "ignore", rename: bool = False, overwrite: bool = False
    ) -> list[Path]:
        from image2image_reg.elastix.transform import transform_attached_shape

        errors = []
        paths = []
        attached_to_modality_transform = {}
        attached_to_modality_image_shape = {}

        # export attachment modalities
        with MeasureTimer() as timer:
            for _, attached_dict in tqdm(self.attachment_shapes.items(), desc="Exporting attachment shapes..."):
                attached_to = attached_dict["attach_to"]
                # get pixel size - if the pixel size is not 1.0, then data is in physical, otherwise index coordinates
                shape_pixel_size = attached_dict["pixel_size"]
                attach_to_modality = self.modalities[attached_to]
                if attached_to not in attached_to_modality_transform:
                    _, transform_seq, _ = self._prepare_transform(attach_to_modality.name)
                    attached_to_modality_transform[attached_to] = transform_seq
                transform_seq = attached_to_modality_transform[attached_to]
                if attached_to not in attached_to_modality_image_shape:
                    attached_to_modality_image_shape[attached_to] = self.get_wrapper(
                        name=attached_to
                    ).reader.image_shape
                image_shape = attached_to_modality_image_shape[attached_to]

                for file in attached_dict["files"]:
                    logger.trace(f"Exporting {file} to {attached_to} with {transform_seq}...")
                    name = Path(file).stem
                    suffix = Path(file).suffix
                    filename = make_new_name(name, attached_to, suffix=suffix) if rename else f"{name}{suffix}"
                    output_path = self.image_dir / filename
                    if output_path.exists() and not overwrite:
                        logger.trace(f"Skipping {attached_to} as it already exists ({output_path}).")
                        continue
                    if transform_seq is None:
                        output_path.write_bytes(file.read_bytes())
                    else:
                        try:
                            path = transform_attached_shape(
                                transform_seq,
                                file,
                                shape_pixel_size,
                                output_path,
                                silent=False,
                                mode="geojson",
                                image_shape=image_shape,
                                clip=clip,
                            )
                            logger.trace(f"Exported {file} to {attached_to} in {timer(since_last=True)}")
                            paths.append(path)
                        except (KeyError, ValueError):
                            errors.append(file)
                            logger.exception(f"Could not export {file} to {attached_to}.")
        if errors:
            logger.error(f"Failed to export {len(errors)} attachment shapes.")
        if paths:
            logger.info(f"Exporting attachment shapes took {timer()}.")
        return paths

    def _export_attachment_points(
        self, n_parallel: int = 1, clip: str = "ignore", rename: bool = False, overwrite: bool = False
    ) -> list[Path]:
        from image2image_reg.elastix.transform import transform_attached_point

        errors = []
        paths = []
        attached_to_modality_transform = {}
        attached_to_modality_image_shape = {}
        # export attachment modalities
        with MeasureTimer() as timer:
            for _, attached_dict in tqdm(self.attachment_points.items(), desc="Exporting attachment points..."):
                attached_to = attached_dict["attach_to"]
                # get pixel size - if the pixel size is not 1.0, then data is in physical, otherwise index coordinates
                shape_pixel_size = attached_dict["pixel_size"]
                attach_to_modality = self.modalities[attached_to]
                if attached_to not in attached_to_modality_transform:
                    _, transform_seq, _ = self._prepare_transform(attach_to_modality.name)
                    attached_to_modality_transform[attached_to] = transform_seq
                transform_seq = attached_to_modality_transform[attached_to]
                if attached_to not in attached_to_modality_image_shape:
                    attached_to_modality_image_shape[attached_to] = self.get_wrapper(
                        name=attached_to
                    ).reader.image_shape
                image_shape = attached_to_modality_image_shape[attached_to]
                for file in attached_dict["files"]:
                    logger.trace(f"Exporting {file} to {attached_to}...")
                    name = Path(file).stem
                    suffix = Path(file).suffix
                    filename = make_new_name(name, attached_to, suffix=suffix) if rename else f"{name}{suffix}"
                    output_path = self.image_dir / filename
                    if output_path.exists() and not overwrite:
                        logger.trace(f"Skipping {attached_to} as it already exists ({output_path}).")
                        continue
                    if transform_seq is None:
                        output_path.write_bytes(file.read_bytes())
                    else:
                        try:
                            path = transform_attached_point(
                                transform_seq,
                                file,
                                shape_pixel_size,
                                output_path,
                                silent=False,
                                as_image=False,  # not transform_sequence.is_linear,
                                image_shape=image_shape,
                                clip=clip,
                            )
                            logger.trace(f"Exported {file} to {attached_to} in {timer(since_last=True)}")
                            paths.append(path)
                        except KeyError:  # ValueError):
                            logger.exception(f"Could not export {file} to {attached_to}.")
                            errors.append(file)
        if errors:
            logger.error(f"Failed to export {len(errors)} attachment points.")
        if paths:
            logger.info(f"Exporting attachment points took {timer()}.")
        return paths

    def _export_attachment_images(
        self,
        merge_modalities: list[str],
        remove_merged: bool,
        fmt: str | WriterMode = "ome-tiff",
        to_original_size: bool = True,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        rename: bool = False,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> list[Path]:
        errors = []
        paths = []
        to_write = []

        # keys are: attached image - attached to image (the one that was registered)
        with MeasureTimer() as timer:
            for attached_modality_key, attach_to_modality_key in tqdm(
                self.attachment_images.items(), desc="Exporting attachment images..."
            ):
                if attached_modality_key in merge_modalities and remove_merged:
                    continue

                attached_modality = self.modalities[attached_modality_key]
                image_modality, transform_seq, output_path = self._prepare_transform(
                    attach_to_modality_key,
                    to_original_size=to_original_size,
                    rename=rename,
                    attachment=True,
                    attachment_modality=attached_modality,
                )
                if _get_with_suffix(output_path, fmt).exists() and not overwrite:
                    logger.trace(f"Skipping {attach_to_modality_key} as it already exists ({output_path}).")
                    continue
                logger.trace(f"Exporting {attached_modality} to {output_path}... (attached)")
                to_write.append(
                    (
                        image_modality.name,
                        transform_seq,
                        output_path,
                        fmt,
                        tile_size,
                        as_uint8,
                        None,
                        # lambda: (True, attach_to_modality_key, to_original_size),
                    )
                )
            if to_write:
                for args in tqdm(to_write, desc="Exporting attachment modalities..."):
                    try:
                        path = self._transform_write_image(*args)
                        paths.append(path)
                    except (ValueError, KeyError):
                        errors.append(args)
                        logger.exception(f"Could not export {args[0]} to {args[2]}.")
        if errors:
            logger.error(f"Failed to export {len(errors)} attachment images.")
        if paths:
            logger.info(f"Exporting attachment images took {timer()}.")
        return paths

    def _prepare_transform(
        self,
        edge_key: str,
        to_original_size: bool = True,
        rename: bool = False,
        attachment: bool = False,
        attachment_modality: Modality | None = None,
    ) -> tuple[Modality, TransformSequence | None, Path]:
        try:
            return self._prepare_registered_transform(
                edge_key,
                to_original_size=to_original_size,
                rename=rename,
                attachment=attachment,
                attachment_modality=attachment_modality,
            )
        except KeyError:
            return self._prepare_not_registered_transform(
                edge_key,
                to_original_size=to_original_size,
                rename=rename,
                attachment=attachment,
                attachment_modality=attachment_modality,
            )

    def _prepare_registered_transform(
        self,
        edge_key: str,
        attachment: bool = False,
        attachment_modality: Modality | None = None,
        to_original_size: bool = True,
        rename: bool = False,
    ) -> tuple[Modality, TransformSequence, Path]:
        final_modality_key = self.registration_paths[edge_key][-1]
        transform_seq = copy(self.transformations[edge_key]["full-transform-seq"])

        # modality_key = None
        preprocessing_modality = self.modalities[edge_key]
        if attachment and attachment_modality:
            edge_key = attachment_modality.name
        modality = self.modalities[edge_key]
        # get filename
        filename = make_new_name(edge_key, final_modality_key, suffix="") if rename else Path(modality.path).name

        # handle original size
        if self.original_size_transforms.get(final_modality_key) and to_original_size:
            logger.trace("Adding transform to original size...")
            original_size_transform = self.original_size_transforms[final_modality_key]
            if isinstance(original_size_transform, list):
                original_size_transform = original_size_transform[0]
            orig_size_rt = TransformSequence(Transform(original_size_transform), transform_sequence_index=[0])
            transform_seq.append(orig_size_rt)

        # handle downsampling
        if preprocessing_modality.preprocessing and preprocessing_modality.preprocessing.downsample > 1:
            if preprocessing_modality.output_pixel_size:
                transform_seq.set_output_spacing(preprocessing_modality.output_pixel_size)
            else:
                output_spacing_target = self.preprocessed_cache["image_spacing"][final_modality_key]
                transform_seq.set_output_spacing(output_spacing_target)
        elif preprocessing_modality.output_pixel_size:
            transform_seq.set_output_spacing(preprocessing_modality.output_pixel_size)

        # handle attachment
        if attachment:  # and modality_key:
            modality = attachment_modality
        return modality, transform_seq, self.image_dir / filename

    def _prepare_not_registered_transform(
        self,
        modality_key: str,
        attachment: bool = False,
        attachment_modality: Modality | None = None,
        to_original_size: bool = True,
        rename: bool = False,
    ) -> tuple[Modality, TransformSequence | None, Path]:
        from image2image_reg.elastix.transform_utils import identity_elx_transform
        from image2image_reg.wrapper import ImageWrapper

        logger.trace(f"Preparing transforms for non-registered modality : {modality_key} ")

        modality = self.modalities[modality_key]
        filename = f"{modality_key}_registered" if rename else Path(modality.path).name

        transform_seq = None
        # handle any spatial pre-processing
        if modality.preprocessing and (
            modality.preprocessing.rotate_counter_clockwise != 0
            or modality.preprocessing.flip
            or modality.preprocessing.translate_x != 0
            or modality.preprocessing.translate_y != 0
            or modality.preprocessing.affine is not None
            or modality.is_cropped()
        ):
            initial_transform = ImageWrapper.load_initial_transform(modality, self.cache_dir)
            original_size_transform = None
            if to_original_size:
                original_size_transform = ImageWrapper.load_original_size_transform(modality, self.cache_dir)
                self.original_size_transforms[modality_key] = original_size_transform

            if initial_transform:
                transform_seq = TransformSequence(
                    [Transform(t) for t in initial_transform],
                    transform_sequence_index=list(range(len(initial_transform))),
                )
            if original_size_transform:
                original_size_transform_seq = TransformSequence(
                    [Transform(t) for t in original_size_transform],
                    transform_sequence_index=list(range(len(original_size_transform))),
                )
                if transform_seq:
                    transform_seq.append(original_size_transform_seq)
                else:
                    transform_seq = original_size_transform_seq

        # handle original
        if to_original_size and self.original_size_transforms.get(modality_key, None):
            o_size_tform = self.original_size_transforms[modality_key]
            if isinstance(o_size_tform, list):
                o_size_tform = o_size_tform[0]
            original_size_transform_seq = TransformSequence(Transform(o_size_tform), transform_sequence_index=[0])
            if transform_seq:
                transform_seq.append(original_size_transform_seq)
            else:
                transform_seq = original_size_transform_seq

        # handle down-sampling
        if modality.preprocessing and modality.preprocessing.downsample > 1 and transform_seq:
            if not modality.output_pixel_size:
                output_spacing_target = self.preprocessed_cache["image_spacing"][modality.name]
                transform_seq.set_output_spacing(output_spacing_target)
            else:
                transform_seq.set_output_spacing(modality.output_pixel_size)
        elif modality.preprocessing and modality.preprocessing.downsample > 1 and not transform_seq:
            transform_seq = TransformSequence(
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
            output_pixel_size = (
                self.preprocessed_cache["image_spacing"][modality.name]
                if not modality.output_pixel_size
                else modality.output_pixel_size
            )
            transform_seq.set_output_spacing(output_pixel_size)
        # handle resampling
        if not transform_seq and modality.output_pixel_size:
            transform_seq = TransformSequence(
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
            transform_seq.set_output_spacing(modality.output_pixel_size)

        # handle attachment
        if attachment and attachment_modality:
            modality = attachment_modality
            filename = f"{attachment_modality.name}_registered" if rename else Path(modality.path).name
        return modality, transform_seq, self.image_dir / filename

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

        from image2image_reg.wrapper import ImageWrapper

        if not modality and not prep_func:
            raise ValueError("Either modality or prep_func must be specified.")
        if prep_func:
            attachment, attach_to_modality, to_original_size = prep_func()
            if attach_to_modality is None:
                attach_to_modality = modality
            modality_name = modality
            modality_for_name = self.modalities[modality_name] if isinstance(modality_name, str) else modality_name
            modality, transformations, filename = self._prepare_registered_transform(
                attach_to_modality,
                attachment=attachment,
                attachment_modality=modality_for_name,
                to_original_size=to_original_size,
            )
        elif isinstance(modality, str):
            modality = self.modalities[modality]

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
            overwrite=True,
        )
        return path

    def _transform_write_merge_images(
        self,
        to_original_size: bool = True,
        preview: bool = False,
        tile_size: int = 512,
        as_uint8: bool | None = None,
        overwrite: bool = False,
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
            self._find_not_registered_modalities()
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
                _, transform_seq, _ = self._prepare_transform(
                    name,
                    attachment=attachment,
                    attachment_modality=attachment_modality,
                    to_original_size=to_original_size,
                )
                transformations.append(transform_seq)

            # override as_uint8 if explicitly specified
            if isinstance(as_uint8_, bool):
                as_uint8 = as_uint8_

            if self.name == merge_name:
                filename = "merged-registered"
            else:
                filename = f"{merge_name}_merged-registered"
            output_path = self.image_dir / filename
            if output_path.with_suffix(".ome.tiff").exists() and not overwrite:
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

    def _get_config(self, registered: bool = False) -> dict:
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
        config: ElastixRegConfig = {
            "schema_version": "1.1",
            "name": self.name,
            # "output_dir": str(self.project_dir),
            "cache_images": self.cache_images,
            # "cache_dir": str(self.cache_dir),
            "pairwise": self.pairwise,
            "modalities": modalities_out,
            "registration_paths": registration_paths,
            "registration_graph_edges": reg_graph_edges if registered else None,
            "original_size_transforms": self.original_size_transforms if registered else None,
            "attachment_shapes": self.attachment_shapes if len(self.attachment_shapes) > 0 else None,
            "attachment_points": self.attachment_points if len(self.attachment_points) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge": self.merge_images,
            "merge_images": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }
        return config

    def save(self, registered: bool = False, auto: bool = False, **kwargs: ty.Any) -> Path:
        """Save configuration to file."""
        status = "registered" if registered is True else "setup"
        config = self._get_config(registered)
        ts = time.strftime("%Y%m%d-%H%M%S")
        filename = (
            f"{ts}-{self.name}-configuration-{status}.json"
            if auto
            else (self.REGISTERED_CONFIG_NAME if registered else self.CONFIG_NAME)
        )
        path = self.project_dir / filename
        self.project_dir.mkdir(exist_ok=True, parents=True)
        write_json_data(path, config)
        logger.trace(f"Saved configuration to '{path}'.")
        return path

    def save_to_wsireg(self, filename: PathLike | None = None, registered: bool = False) -> Path:
        """Save workflow configuration."""
        import yaml

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
            "attachment_shapes": self.attachment_shapes if len(self.attachment_shapes) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge_modalities": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }

        if not filename:
            filename = self.project_dir / f"{ts}-{self.name}-configuration-{status}.yaml"
        filename = Path(filename)
        with open(str(filename), "w") as f:
            yaml.dump(config, f, sort_keys=False)
        return filename

    def to_valis(self, output_dir: PathLike) -> ValisReg:
        """Convert the configuration to ValisReg."""
        from image2image_reg.workflows import ValisReg

        return ValisReg.from_wsireg(self, output_dir)

    @classmethod
    def from_valis(cls, obj: ValisReg, outout_dir: PathLike) -> ElastixReg:
        """Create ElastixReg from ValisReg."""
        iwsreg = cls(obj.name, output_dir=outout_dir, cache=obj.cache_images, merge=obj.merge_images)

        # add modalities
        for modality in obj.modalities.values():
            iwsreg.modalities[modality.name] = deepcopy(modality)
            if iwsreg.modalities[modality.name].preprocessing:
                iwsreg.modalities[modality.name].preprocessing.method = None

        # copy other attributes
        iwsreg.attachment_images = deepcopy(obj.attachment_images)
        iwsreg.attachment_shapes = deepcopy(obj.attachment_shapes)
        iwsreg.attachment_points = deepcopy(obj.attachment_points)
        iwsreg.merge_modalities = deepcopy(obj.merge_modalities)

        reference = obj.reference
        if reference:
            for name, modality in obj.modalities.items():
                if name == reference:
                    continue
                if iwsreg.is_attachment(name):
                    continue
                iwsreg.add_registration_path(name, reference, ["rigid", "affine", "nl"])
        iwsreg.save()
        return iwsreg


def _get_with_suffix(p: Path, fmt: str) -> Path:
    if fmt.startswith("."):
        fmt = fmt[1:]
    name = p.stem
    for ext in [".ome", ".tiff", ".czi", ".txt", ".geojson", ".tsv", ".csv"]:
        name = name.replace(ext, "")

    if fmt in ["ome-tiff", "ome-tiff-by-plane", "ome-tiff-by-tile"]:
        return p.parent / (name + ".ome.tiff")
    elif fmt in ["csv", "tsv", "txt", "geojson"]:
        return p.parent / (name + "." + fmt)
    raise ValueError(f"Writer {fmt} is not supported.")

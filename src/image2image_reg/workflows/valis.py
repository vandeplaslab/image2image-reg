"""Valis interface."""

from __future__ import annotations

import typing as ty
from copy import deepcopy
from pathlib import Path

from koyo.json import read_json_data, write_json_data
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import is_installed
from loguru import logger
from tqdm import tqdm

from image2image_reg._typing import ValisRegConfig
from image2image_reg.enums import WriterMode
from image2image_reg.workflows._base import Workflow

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.elastix import ElastixReg

logger = logger.bind(src="Valis")

HAS_VALIS = is_installed("valis") and is_installed("pyvips")


class ValisReg(Workflow):
    """Registration using Valis."""

    CONFIG_NAME = "valis.config.json"
    EXTENSION = ".valis"

    _registrar: ty.Any = None

    def __init__(
        self,
        name: str | None = None,
        output_dir: PathLike | None = None,
        project_dir: PathLike | None = None,
        cache: bool = True,
        merge: bool = False,
        log: bool = False,
        init: bool = True,
        check_for_reflections: bool = False,
        non_rigid_registration: bool = False,
        micro_registration: bool = True,
        micro_registration_fraction: float = 0.125,
        feature_detector: str = "sensitive_vgg",
        feature_matcher: str = "RANSAC",
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
        self._reference: str = None
        self.check_for_reflections: bool = check_for_reflections
        self.non_rigid_registration: bool = non_rigid_registration
        self.micro_registration: bool = micro_registration
        self.micro_registration_fraction: float = micro_registration_fraction
        self.feature_detector: str = feature_detector
        self.feature_matcher: str = feature_matcher
        if not HAS_VALIS:
            logger.warning("Valis is not installed. Please install it to use this functionality.")

    @property
    def is_registered(self) -> bool:
        """Check if the project has been registered."""
        from image2image_reg.valis.utilities import get_registrar_path

        registrar_path = get_registrar_path(self.project_dir, self.name)
        return registrar_path.exists()

    @classmethod
    def from_path(cls, path: PathLike, raise_on_error: bool = True) -> ValisReg:
        """Initialize based on the project path."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist ({path}).")
        if path.is_file() and path.name in [cls.CONFIG_NAME]:
            path = path.parent
        if not path.is_dir():
            raise ValueError("Path is not a directory.")
        config_path = path / cls.CONFIG_NAME
        if not path.suffix == ".valis" and not config_path.exists():
            raise ValueError("Path is not a valid Valis project.")

        with MeasureTimer() as timer:
            config: dict | ValisRegConfig | None = None
            if config_path.exists():
                if config_path.exists():
                    config = read_json_data(config_path)
            if config and "name" not in config:
                config["name"] = path.stem
            obj = cls(project_dir=path, **config)
            if config_path.exists():
                obj.load_from_i2valis(raise_on_error=raise_on_error)
        logger.trace(f"Restored from config in {timer()}")
        return obj

    def to_i2reg(self, output_dir: PathLike) -> ElastixReg:
        """Export to ElastixReg object."""
        from image2image_reg.workflows import ElastixReg

        return ElastixReg.from_valis(self, output_dir)

    @classmethod
    def from_wsireg(
        cls,
        obj: ElastixReg,
        output_dir: PathLike,
        check_for_reflections: bool = False,
        non_rigid_registration: bool = False,
        micro_registration: bool = True,
        micro_registration_fraction: float = 0.125,
        feature_detector: str = "sensitive_vgg",
        feature_matcher: str = "RANSAC",
    ) -> ValisReg:
        """Create Valis configuration from ElastixReg object."""
        valis = cls(
            obj.name,
            output_dir=output_dir,
            merge=obj.merge_images,
            check_for_reflections=check_for_reflections,
            non_rigid_registration=non_rigid_registration,
            micro_registration=micro_registration,
            micro_registration_fraction=micro_registration_fraction,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
        )
        # add modalities
        for modality in obj.modalities.values():
            valis.modalities[modality.name] = deepcopy(modality)
            if valis.modalities[modality.name].preprocessing:
                valis.modalities[modality.name].preprocessing.method = "I2RegPreprocessor"

        # try to get reference
        references = []
        for _source, targets in obj.registration_paths.items():
            references.append(targets[-1])  # last node is the final target
        references = list(set(references))
        if references and len(references) == 1:
            reference = references[0]
            valis.set_reference(name=reference)

        # copy other attributes
        valis.attachment_images = deepcopy(obj.attachment_images)
        valis.attachment_shapes = deepcopy(obj.attachment_shapes)
        valis.attachment_points = deepcopy(obj.attachment_points)
        valis.merge_modalities = deepcopy(obj.merge_modalities)
        valis.save()
        return valis

    def load_from_i2valis(self, raise_on_error: bool = True) -> None:
        """Load data from image2image-reg project file."""
        config: ValisRegConfig = read_json_data(self.project_dir / self.CONFIG_NAME)
        name = config.get("name")
        if name != self.project_dir.stem:
            name = self.project_dir.stem
            logger.trace(f"Name in config does not match directory name. Using '{name}' instead.")
        # restore parameters
        self.name = name
        self.cache_images = config["cache_images"]
        self.merge_images = config["merge"]
        self.check_for_reflections = config["check_for_reflections"]
        self.non_rigid_registration = config["non_rigid_registration"]
        self.micro_registration = config["micro_registration"]
        self.micro_registration_fraction = config["micro_registration_fraction"]

        # add modality information
        self._load_modalities_from_config(config, raise_on_error)
        # load attachment images
        self._load_attachment_from_config(config)
        # load merge
        self._load_merge_from_config(config)
        self.set_reference(config["reference"])

    def print_summary(self, func: ty.Callable = logger.info) -> None:
        """Print summary about the project."""
        elbow, pipe, tee, blank = "└──", "│  ", "├──", "   "

        func(f"Project name: {self.name}")
        func(f"Project directory: {self.project_dir}")
        func(f"Merging images: {self.merge_images}")
        func(f"Feature detector: {self.feature_detector}")
        func(f"Feature matcher: {self.feature_matcher}")
        func(f"Check for reflections: {self.check_for_reflections}")
        func(f"Non-rigid registration: {self.non_rigid_registration}")
        func(f"Micro registration: {self.micro_registration}")
        func(f"Micro registration fraction: {self.micro_registration_fraction}")

        # func information about the specified modalities
        func(f"Number of modalities: {len(self.modalities)}")
        n = len(self.modalities) - 1
        for i, modality in enumerate(self.modalities.values()):
            func(f" {elbow if i == n else tee}{modality.name} ({modality.path})")
            func(f" {pipe if i != n else blank}{tee}Preprocessing: {modality.preprocessing is not None}")
            func(f" {pipe if i != n else blank}{elbow}Export: {modality.export}")

        # func information about attachment images
        func(f"Number of attachment images: {len(self.attachment_images)}")
        n = len(self.attachment_images) - 1
        for i, (name, attach_to) in enumerate(self.attachment_images.items()):
            func(f" {elbow if i == n else tee}{name} ({attach_to})")
        # func information about attachment shapes
        func(f"Number of attachment shapes: {len(self.attachment_shapes)}")
        n = len(self.attachment_shapes) - 1
        for i, (name, shape_dict) in enumerate(self.attachment_shapes.items()):
            func(f" {elbow if i == n else tee}{name} ({shape_dict})")
        # func information about attachment shapes
        func(f"Number of attachment points: {len(self.attachment_points)}")
        n = len(self.attachment_points) - 1
        for i, (name, shape_dict) in enumerate(self.attachment_points.items()):
            func(f" {elbow if i == n else tee}{name} ({shape_dict})")
        # func information about merge modalities
        func(f"Number of merge modalities: {len(self.merge_modalities)}")
        n = len(self.merge_modalities) - 1
        for i, (name, merge_modalities) in enumerate(self.merge_modalities.items()):
            func(f" {elbow if i == n else tee}{name} ({merge_modalities})")

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

        # check if the reference exists
        if self.reference and not self.has_modality(name_or_path=self.reference):
            errors.append(f"❌ Reference modality '{self.reference}' not found.")
            logger.error(errors[-1])

        is_valid = not errors
        if not is_valid:
            errors.append("❌ Project configuration is invalid.")
            logger.error(errors[-1])
        else:
            logger.success("✅ Project configuration is valid.")
        return is_valid, errors

    def clear(
        self,
        cache: bool = True,
        valis: bool = False,
        image: bool = False,
        metadata: bool = False,
        clear_all: bool = False,
    ) -> None:
        """Clear existing data."""
        from image2image_reg.utils.utilities import _safe_delete

        if clear_all:
            cache = image = valis = metadata = True

        # clear transformations, cache, images
        if cache:
            for file in self.cache_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.cache_dir)

        if metadata:
            for sub_directory in [
                "masks",
                "matches-initial",
                "matches-final",
                "overlaps",
                "processed",
                "micro_registration",
                "registered",
                "rigid_registration",
                "non_rigid_registration",
                "deformation_fields",
            ]:
                directory = self.project_dir / sub_directory
                for file in directory.glob("*"):
                    _safe_delete(file)
                _safe_delete(directory)

        if image:
            for file in self.image_dir.glob("*"):
                _safe_delete(file)
            _safe_delete(self.image_dir)

        if valis:
            directory = self.project_dir / "data"
            for file in directory.glob("*"):
                _safe_delete(file)
            _safe_delete(directory)

    @property
    def filelist(self) -> list[PathLike]:
        """Return list of file paths."""
        from natsort import natsorted

        # get all modality paths
        modalities = self.get_image_modalities(with_attachment=False)
        filelist = []
        for modality_name in modalities:
            filelist.append(self.modalities[modality_name].path)

        # sort images
        filelist = natsorted(filelist)
        filelist = [Path(path) for path in filelist]
        for path in filelist:
            assert path.exists(), f"{path} does not exist."
        return filelist

    def set_reference(self, name: str | None = None, path: str | None = None) -> None:
        """Set reference image."""
        if not name and not path:
            self._reference = None
        else:
            modality = self.get_modality(name, path)
            if not modality:
                raise ValueError(f"Modality {name} not found.")
            assert modality.path.exists(), f"Path {modality.path} does not exist."
            self._reference = modality.name
        logger.trace(f"Set reference image to '{self._reference}'.")

    @property
    def reference(self) -> str | None:
        """Get reference image."""
        reference = self._reference
        if reference:
            if Path(reference).exists():
                reference = self.get_modality(name_or_path=reference).name
            modality = self.get_modality(reference)
            assert modality, f"Modality {reference} not found."
            reference = modality.name
        return reference

    @property
    def reference_path(self) -> PathLike | None:
        """Get reference image path."""
        reference = self.reference
        if reference:
            return self.modalities[reference].path
        return None

    @property
    def registrar(self) -> ty.Any:
        """Load registrar or retrieve it."""
        if self._registrar is None:
            from valis.valtils import get_name

            from image2image_reg.valis.utilities import get_valis_registrar_alt, update_registrar_paths

            self._registrar = get_valis_registrar_alt(self.project_dir, self.name, init_jvm=True)
            if self._registrar:
                for modality in self.modalities.values():
                    try:
                        slide_obj = self._registrar.get_slide(get_name(str(modality.path)))
                        if slide_obj and not Path(slide_obj.src_f).exists() and modality.path.exists():
                            slide_obj.src_f = str(modality.path)
                            logger.info(f"Set source file for '{modality.name}' to '{slide_obj.src_f}'")
                    except UnboundLocalError:
                        pass
                update_registrar_paths(self._registrar, self.project_dir)
            logger.trace(f"Loaded registrar from '{self.project_dir}'.")
        return self._registrar

    def register(self, **kwargs: ty.Any) -> None:
        """Co-register images."""
        from valis import registration

        from image2image_reg.valis.utilities import (
            get_feature_detector,
            get_feature_matcher,
            get_micro_registration_dimension,
        )

        # get filelist
        filelist = self.filelist
        filelist = [str(s) for s in filelist]
        logger.info(f"Filelist has {len(filelist)} images.")
        # get reference
        reference = self.reference_path
        logger.info(f"Reference image: {reference}")
        # get detector
        feature_detector_cls = get_feature_detector(self.feature_detector)
        logger.info(f"Feature detector: {feature_detector_cls}")
        # get matcher
        feature_matcher_cls = get_feature_matcher(self.feature_matcher)
        logger.info(f"Feature matcher: {feature_matcher_cls}")
        # Print configuration
        logger.info(f"Check for reflections: {self.check_for_reflections}")
        logger.info(f"Non-rigid registration: {self.non_rigid_registration}")
        logger.info(f"Micro-registration: {self.micro_registration}; fraction: {self.micro_registration_fraction}")

        kws = {}
        if not self.non_rigid_registration:
            kws["non_rigid_registrar_cls"] = None

        channel_kws = get_channel_kws(self)

        # initialize java
        registration.init_jvm()
        with MeasureTimer() as main_timer:
            try:
                # get registrar
                registrar_path = self.project_dir / "data" / f"{self.name}_registrar.pickle"
                if registrar_path.exists():
                    registrar = self.registrar
                    logger.info(f"Loaded registrar from '{registrar_path}'.")
                else:
                    registrar = registration.Valis(
                        str(self.project_dir),
                        str(self.project_dir),
                        name=self.name,
                        image_type="fluorescence",  # should it be auto
                        imgs_ordered=True,
                        img_list=filelist,
                        reference_img_f=str(reference) if reference else None,
                        align_to_reference=reference is not None,
                        check_for_reflections=self.check_for_reflections,
                        feature_detector_cls=feature_detector_cls,
                        matcher=feature_matcher_cls,
                        **kws,
                    )
                    registrar.dst_dir = str(self.project_dir)
                    registrar.set_dst_paths()

                    with MeasureTimer() as timer:
                        registrar.register(processor_dict=channel_kws)
                    logger.info(f"Registered low-res images in {timer()}")

                    # We can also plot the high resolution matches using `Valis.draw_matches`:
                    try:
                        with MeasureTimer() as timer:
                            matches_dst_dir = Path(registrar.dst_dir) / "matches-initial"
                            registrar.draw_matches(matches_dst_dir)
                        logger.info(f"Exported matches in {timer()}")
                    except Exception as exc:
                        logger.exception(f"Failed to export matches {exc}.")

                    # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an
                    # image that is proportion of the full resolution.
                    if self.micro_registration:
                        micro_reg_size = get_micro_registration_dimension(registrar, self.micro_registration_fraction)
                        logger.info(f"Micro-registering using {micro_reg_size} pixels.")

                        # Perform high resolution non-rigid registration
                        try:
                            with MeasureTimer() as timer:
                                registrar.register_micro(
                                    max_non_rigid_registration_dim_px=micro_reg_size,
                                    processor_dict=channel_kws,
                                    reference_img_f=str(reference) if reference else None,
                                    align_to_reference=reference is not None,
                                )
                                logger.info(f"Registered high-res images in {timer()}")
                        except Exception as exc:
                            logger.exception(f"Error during micro registration: {exc}")
                            self.micro_registration = False

                    # We can also plot the high resolution matches using `Valis.draw_matches`:
                    try:
                        with MeasureTimer() as timer:
                            matches_dst_dir = Path(registrar.dst_dir) / "matches-final"
                            registrar.draw_matches(matches_dst_dir)
                        logger.info(f"Exported matches in {timer()}")
                    except Exception as exc:
                        logger.exception(f"Failed to export matches: {exc}.")
                # set registrar
                self._registrar = registrar
            except Exception as exc:
                registration.kill_jvm()
                logger.exception(f"Error during registration: {exc}")
                raise exc
        logger.info(f"Completed registration in {main_timer()}")

    def write(
        self,
        n_parallel: int = 1,
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
        rename: bool = False,
        overwrite: bool = False,
        **kwargs: ty.Any,
    ) -> list | None:
        """Export images after applying transformation."""
        if not self.registrar:
            raise ValueError("Registrar not found. Please register first.")
        if write_attached:
            write_attached_images = write_attached_points = write_attached_shapes = True

        if n_parallel is None or n_parallel < 1:
            n_parallel = 1

        paths = []
        # export registered images
        if write_registered:
            paths.extend(
                self._export_registered_images(
                    fmt=fmt, tile_size=tile_size, as_uint8=as_uint8, rename=rename, overwrite=overwrite
                )
            )

        # export attachment modalities
        if write_attached_shapes:
            paths.extend(self._export_attachment_shapes(n_parallel=n_parallel, overwrite=overwrite))
        if write_attached_points:
            paths.extend(self._export_attachment_points(n_parallel=n_parallel, overwrite=overwrite))
        if write_attached_images:
            paths.extend(
                self._export_attachment_images(
                    fmt=fmt, tile_size=tile_size, as_uint8=as_uint8, rename=rename, overwrite=overwrite
                )
            )
        return paths

    def _export_registered_images(
        self,
        fmt: str | WriterMode = "ome-tiff",
        tile_size: int = 512,
        as_uint8: bool | None = None,
        rename: bool = False,
        overwrite: bool = False,
    ) -> list[Path]:
        from image2image_reg.valis.transform import transform_registered_image

        path_to_name_map = {Path(modality.path): modality.name for modality in self.modalities.values()}

        paths = []
        with MeasureTimer() as timer:
            paths_ = transform_registered_image(
                self.registrar,
                self.image_dir,
                non_rigid=self.non_rigid_registration,
                as_uint8=as_uint8,
                tile_size=tile_size,
                overwrite=overwrite,
                path_to_name_map=path_to_name_map,
                rename=rename,
            )
            paths.extend(paths_)
        if paths:
            logger.info(f"Exported registered images in {timer()}")
        return paths

    def _export_attachment_shapes(self, n_parallel: int = 1, overwrite: bool = False) -> list[Path]:
        from image2image_reg.valis.transform import transform_attached_shapes

        paths = []
        # export attachment modalities
        with MeasureTimer() as timer:
            for _name, attached_dict in tqdm(self.attachment_shapes.items(), desc="Exporting attachment shapes..."):
                attached_to = attached_dict["attach_to"]
                # get pixel size - if the pixel size is not 1.0, then data is in physical, otherwise index coordinates
                source_pixel_size = attached_dict["pixel_size"]  # source pixel size
                attach_to_modality = self.modalities[attached_to]

                paths_ = transform_attached_shapes(
                    self.registrar,
                    attach_to_modality.path,  # type: ignore[arg-type]
                    self.image_dir,
                    attached_dict["files"],
                    source_pixel_size,
                    overwrite=overwrite,
                    non_rigid=self.non_rigid_registration,
                    as_image=True,
                )
                paths.extend(paths_)
        if paths:
            logger.info(f"Exported attached shapes in {timer()}")
        return paths

    def _export_attachment_points(self, n_parallel: int = 1, overwrite: bool = False) -> list[Path]:
        from image2image_reg.valis.transform import transform_attached_points

        paths = []
        # export attachment modalities
        with MeasureTimer() as timer:
            for _name, attached_dict in tqdm(self.attachment_points.items(), desc="Exporting attachment shapes..."):
                attached_to = attached_dict["attach_to"]
                # get pixel size - if the pixel size is not 1.0, then data is in physical, otherwise index coordinates
                source_pixel_size = attached_dict["pixel_size"]
                attach_to_modality = self.modalities[attached_to]

                paths_ = transform_attached_points(
                    self.registrar,
                    attach_to_modality.path,  # type: ignore[arg-type]
                    self.image_dir,
                    attached_dict["files"],
                    source_pixel_size,
                    overwrite=overwrite,
                    non_rigid=self.non_rigid_registration,
                    as_image=True,
                )
                paths.extend(paths_)
        if paths:
            logger.info(f"Exported attached points in {timer()}")
        return paths

    def _export_attachment_images(
        self,
        fmt: str | WriterMode = "ome-tiff",
        tile_size: int = 512,
        as_uint8: bool | None = None,
        rename: bool = False,
        overwrite: bool = False,
    ) -> list[Path]:
        from image2image_reg.valis.transform import transform_attached_image

        path_to_name_map = {Path(modality.path): modality.name for modality in self.modalities.values()}

        # export attached images to OME-TIFFs
        paths = []
        with MeasureTimer() as timer:
            attached_images_ = self.attachment_images
            # create mapping of attached_to and images that should be transformed
            attached_images = {}
            for src_name, attached_to in attached_images_.items():
                if attached_to not in attached_images:
                    attached_images[attached_to] = []
                attached_images[attached_to].append(self.modalities[src_name].path)

            names = list(attached_images.keys())
            for name in names:
                attached_images[name] = list(set(attached_images[name]))
                logger.trace(f"Exporting {len(attached_images[name])} images attached to '{name}'")

            for attached_to, sources in attached_images.items():
                attached_to_modality = self.modalities[attached_to]
                paths_ = transform_attached_image(
                    self.registrar,
                    attached_to_modality.path,  # type: ignore[arg-type]
                    sources,
                    self.image_dir,
                    as_uint8=as_uint8,
                    tile_size=tile_size,
                    overwrite=overwrite,
                    path_to_name_map=path_to_name_map,
                    rename=rename,
                    non_rigid=self.non_rigid_registration,
                )
                paths.extend(paths_)
        if paths:
            logger.info(f"Exported attached images in {timer()}")
        return paths

    def _get_config(self, **kwargs: ty.Any) -> dict:
        """Get configuration."""
        modalities_out: dict[str, dict] = {}
        for modality in self.modalities.values():
            modalities_out[modality.name] = modality.to_dict()

        # write config
        config: ValisRegConfig = {
            "schema_version": "1.0",
            "name": self.name,
            "cache_images": self.cache_images,
            "reference": self.reference,
            "check_for_reflections": self.check_for_reflections,
            "non_rigid_registration": self.non_rigid_registration,
            "micro_registration": self.micro_registration,
            "micro_registration_fraction": self.micro_registration_fraction,
            "feature_detector": self.feature_detector,
            "feature_matcher": self.feature_matcher,
            "modalities": modalities_out,
            "attachment_shapes": self.attachment_shapes if len(self.attachment_shapes) > 0 else None,
            "attachment_points": self.attachment_points if len(self.attachment_points) > 0 else None,
            "attachment_images": self.attachment_images if len(self.attachment_images) > 0 else None,
            "merge": self.merge_images,
            "merge_images": self.merge_modalities if len(self.merge_modalities) > 0 else None,
        }
        return config

    def save(self) -> Path:
        """Save configuration to file."""
        config = self._get_config()
        filename = self.CONFIG_NAME
        path = self.project_dir / filename
        self.project_dir.mkdir(exist_ok=True, parents=True)
        write_json_data(path, config)
        logger.trace(f"Saved configuration to '{path}'.")
        return path


def get_config(config: dict[str, str] | PathLike) -> dict[str, str]:
    """Get configuration."""
    if isinstance(config, (str, Path)):
        config = Path(config)
        assert config.exists(), f"{config} does not exist."
        config = read_json_data(config)
        assert isinstance(config, dict), f"{config} is not a dictionary."
    return parse_config(config)


def parse_config(config: dict[str, ty.Any]) -> dict[str, ty.Any]:
    """Parse configuration."""
    from image2image_reg.valis.utilities import get_preprocessor

    if not config:
        raise ValueError("Configuration is empty.")
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")
    if "project_name" not in config:
        raise ValueError("Project name not found in configuration.")
    if "filelist" not in config:
        raise ValueError("Filelist not found in configuration.")
    if not isinstance(config["filelist"], list):
        raise ValueError("Filelist must be a list.")
    if not config["filelist"]:
        raise ValueError("Filelist is empty.")

    # validate and reformat channel_kws
    channel_names = {}
    for name, kws in config.get("channel_kws", {}).items():
        if not isinstance(kws, list):
            raise ValueError(f"Channel keyword arguments for {name} must be a list.")
        if len(kws) != 2:
            raise ValueError(f"Channel keyword arguments for {name} must have 2 elements.")
        preprocessor, kws = kws
        channel_names[name] = [get_preprocessor(preprocessor), kws]
    config["channel_kws"] = channel_names
    return config


def valis_init_configuration(
    project_name: str,
    output_dir: PathLike,
    filelist: list[PathLike],
    reference: PathLike | None = None,
    check_for_reflections: bool = True,
    micro_reg_fraction: float = 0.125,
    non_rigid_reg: bool = False,
    micro_reg: bool = False,
    feature_detector: str = "sensitive_vgg",
):
    """Create Valis configuration."""
    from natsort import natsorted, ns
    from valis import valtils

    from image2image_reg.valis.utilities import get_feature_detector_str, get_preprocessing_for_path

    logger.info(f"Creating configuration for '{project_name}' in '{output_dir}'")

    output_dir = Path(output_dir)
    # sort by name but ensure that its correctly rendered when not taking upper case into consideration
    filelist = natsorted(filelist, alg=ns.IGNORECASE)
    filelist = [Path(path).resolve() for path in filelist]
    for path in filelist:
        assert path.exists(), f"{path} does not exist."
    logger.info(f"Filelist has {len(filelist)} images.")

    if reference:
        reference = Path(reference)
        assert reference.exists(), f"{reference} does not exist."
        assert reference in filelist, f"{reference} not in filelist."
    logger.info(f"Reference image: {reference}")

    feature_detector = get_feature_detector_str(feature_detector)

    channel_kws = {}
    for path in filelist:
        name = valtils.get_name(str(path))
        channel_kws[name] = get_preprocessing_for_path(path)
        logger.trace(f"Preprocessing for {name}: {channel_kws[name]}")

    attachment_images = {}
    attachment_shapes = {}
    attachment_points = {}
    for path in filelist:
        name = valtils.get_name(str(path))
        attachment_images[name] = []
        attachment_shapes[name] = []
        attachment_points[name] = []

    config = {
        "project_name": project_name,
        "filelist": [str(path) for path in filelist],
        "reference": str(reference) if reference else None,
        "channel_kws": channel_kws,
        "check_for_reflections": check_for_reflections,
        "non_rigid_reg": non_rigid_reg,
        "micro_reg": micro_reg,
        "micro_reg_fraction": micro_reg_fraction,
        "feature_detector": feature_detector,
        "attachment_images": attachment_images,
        "attachment_shapes": attachment_shapes,
        "attachment_points": attachment_points,
    }

    filename = output_dir / f"{project_name}.valis.json"
    write_json_data(filename, config, check_existing=False)
    logger.info(f"Configuration saved to '{filename}'")


def valis_registration_from_config(config: PathLike, output_dir: PathLike) -> None:
    """Valis registration from config."""
    config = get_config(config)
    return valis_registration(output_dir=output_dir, **config)


def valis_registration(
    project_name: str,
    filelist: list[PathLike],
    reference: PathLike | None,
    output_dir: PathLike,
    channel_kws: dict[str, str] | PathLike = None,
    check_for_reflections: bool = True,
    micro_reg_fraction: float = 0.25,
    non_rigid: bool = False,
    micro_reg: bool = False,
    feature_detector: str = "sensitive_vgg",
    **kwargs,
) -> None:
    """Valis-based registration."""
    from koyo.timer import MeasureTimer
    from natsort import natsorted
    from valis import registration

    from image2image_reg.valis.transform import transform_attached_image, transform_registered_image
    from image2image_reg.valis.utilities import (
        get_feature_detector,
        get_micro_registration_dimension,
        get_slide_path,
    )

    output_dir = Path(output_dir)

    filelist = natsorted(filelist)
    filelist = [Path(path) for path in filelist]
    for path in filelist:
        assert path.exists(), f"{path} does not exist."
    logger.info(f"Filelist has {len(filelist)} images.")

    if reference:
        reference = Path(reference)
        assert reference.exists(), f"{reference} does not exist."
        assert reference in filelist, f"{reference} not in filelist."
    logger.info(f"Reference image: {reference}")

    feature_detector_cls = get_feature_detector(feature_detector)
    logger.info(f"Feature detector: {feature_detector_cls}")

    # initialize java
    registration.init_jvm()

    filelist = [str(s) for s in filelist]

    kws = {}
    if not non_rigid:
        kws["non_rigid_registrar_cls"] = None

    if kwargs:
        logger.warning(f"Unused keyword arguments: {kwargs}")

    with MeasureTimer() as main_timer:
        try:
            registrar_path = output_dir / project_name / "data" / f"{project_name}_registrar.pickle"
            if registrar_path.exists():
                import pickle

                with open(registrar_path, "rb") as f:
                    registrar = pickle.load(f)
            else:
                registrar = registration.Valis(
                    str(output_dir),
                    str(output_dir),
                    name=project_name,
                    image_type="fluorescence",
                    imgs_ordered=True,
                    img_list=filelist,
                    reference_img_f=str(reference) if reference else None,
                    align_to_reference=reference is not None,
                    check_for_reflections=check_for_reflections,
                    feature_detector_cls=feature_detector_cls,
                    **kws,
                )

                with MeasureTimer() as timer:
                    registrar.register(processor_dict=channel_kws)
                logger.info(f"Registered low-res images in {timer()}")

                # We can also plot the high resolution matches using `Valis.draw_matches`:
                try:
                    with MeasureTimer() as timer:
                        matches_dst_dir = Path(registrar.dst_dir) / "matches-initial"
                        registrar.draw_matches(matches_dst_dir)
                    logger.info(f"Exported matches in {timer()}")
                except Exception:  # type: ignore
                    logger.exception("Failed to export matches.")

                # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an
                # image that is proportion of the full resolution.
                if micro_reg:
                    micro_reg_size = get_micro_registration_dimension(registrar, micro_reg_fraction)
                    logger.info(f"Micro-registering using {micro_reg_size} pixels.")

                    # Perform high resolution non-rigid registration
                    with MeasureTimer() as timer:
                        try:
                            registrar.register_micro(
                                max_non_rigid_registration_dim_px=micro_reg_size,
                                processor_dict=channel_kws,
                                reference_img_f=str(reference) if reference else None,
                                align_to_reference=reference is not None,
                            )
                        except Exception as exc:  # type: ignore
                            logger.exception(f"Error during non-rigid registration: {exc}")
                    logger.info(f"Registered high-res images in {timer()}")

                # We can also plot the high resolution matches using `Valis.draw_matches`:
                try:
                    with MeasureTimer() as timer:
                        matches_dst_dir = Path(registrar.dst_dir) / "matches-final"
                        registrar.draw_matches(matches_dst_dir)
                    logger.info(f"Exported matches in {timer()}")
                except Exception as exc:
                    logger.exception(f"Failed to export matches: {exc}.")

            # export images to OME-TIFFs
            with MeasureTimer() as timer:
                registered_dir = output_dir / project_name / "registered"
                registered_dir.mkdir(exist_ok=True, parents=True)
                transform_registered_image(registrar, registered_dir, non_rigid=non_rigid)
            logger.info(f"Exported registered images in {timer()}")

            # export attached images to OME-TIFFs
            with MeasureTimer() as timer:
                attached_images = kwargs.get("attachment_images", {})
                for src_name, attachments in attached_images.items():
                    src_path = get_slide_path(registrar, src_name)
                    transform_attached_image(registrar, src_path, attachments, registered_dir)
            logger.info(f"Exported attached images in {timer()}")

        except Exception as exc:
            logger.exception(f"Failed to register images: {exc}")
            registration.kill_jvm()
            raise exc
        registration.kill_jvm()
    logger.info(f"Completed registration in {main_timer()}")


def get_channel_kws(obj: ValisReg) -> dict:
    """Get channel kws for each file."""
    from valis import valtils

    from image2image_reg.valis.utilities import get_preprocessor

    channel_kws = {}
    # iterate over modalities
    for modality_name in obj.get_image_modalities(with_attachment=False):
        modality = obj.modalities[modality_name]
        modality_name = valtils.get_name(str(modality.path))
        modality_kws = modality.preprocessing.to_valis()
        channel_kws[modality_name] = [get_preprocessor(modality_kws[0]), modality_kws[1]]
    return channel_kws

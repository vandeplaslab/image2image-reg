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

from image2image_reg._typing import ValisRegConfig
from image2image_reg.models import Export, Preprocessing
from image2image_reg.workflows._base import Workflow

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.iwsireg import IWsiReg

logger = logger.bind(src="Valis")

if not is_installed("valis") or not is_installed("pyvips"):
    logger.warning("Valis or pyvips is not installed. Valis registration will not work.")


class ValisReg(Workflow):
    """Registration using Valis."""

    CONFIG_NAME = "valis.config.json"
    EXTENSION = ".valis"

    registrar: ty.Any = None

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

    @property
    def is_registered(self) -> bool:
        """Check if the project has been registered."""
        registrar_path = self.project_dir / "data" / f"{self.name}_registrar.pickle"
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
        if not path.suffix == ".valis":
            raise ValueError("Path is not a valid Valis project.")

        with MeasureTimer() as timer:
            config_path = path / cls.CONFIG_NAME
            config: dict | ValisRegConfig | None = None
            if config_path.exists():
                config = read_json_data(path / cls.CONFIG_NAME)
            if config and "name" not in config:
                config["name"] = path.stem
            obj = cls(project_dir=path, **config)
            if config_path.exists():
                obj.load_from_i2valis(raise_on_error=raise_on_error)
        logger.trace(f"Restored from config in {timer()}")
        return obj

    @classmethod
    def from_wsireg(
        cls,
        obj: IWsiReg,
        output_dir: PathLike,
        check_for_reflections: bool = False,
        non_rigid_registration: bool = False,
        micro_registration: bool = True,
        micro_registration_fraction: float = 0.125,
        feature_detector: str = "sensitive_vgg",
        feature_matcher: str = "RANSAC",
    ) -> ValisReg:
        """Create Valis configuration from IWsiReg object."""
        obj = cls(
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
            obj.modalities[modality.name] = deepcopy(modality)
            obj.modalities[modality.name].preprocessing.method = "I2RegPreprocessor"
        # copy other attributes
        obj.attachment_images = deepcopy(obj.attachment_images)
        obj.attachment_shapes = deepcopy(obj.attachment_shapes)
        obj.attachment_points = deepcopy(obj.attachment_points)
        obj.merge_modalities = deepcopy(obj.merge_modalities)
        obj.save()
        return obj

    def load_from_i2valis(self, raise_on_error: bool = True) -> None:
        """Load data from image2image-reg project file."""
        config: ValisRegConfig = read_json_data(self.project_dir / self.CONFIG_NAME)
        # restore parameters
        self.name = config["name"]
        self.cache_images = config["cache_images"]
        self.merge_images = config["merge"]
        self._reference = config["reference"]
        self.check_for_reflections = config["check_for_reflections"]
        self.non_rigid_registration = config["non_rigid_registration"]
        self.micro_registration = config["micro_registration"]
        self.micro_registration_fraction = config["micro_registration_fraction"]

        # add modality information
        with MeasureTimer() as timer:
            for name, modality in config["modalities"].items():
                if not Path(modality["path"]).exists() and raise_on_error:
                    raise ValueError(f"Modality path '{modality['path']}' does not exist.")
                preprocessing = modality.get("preprocessing", dict())
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
                    pixel_size=modality.get("pixel_size", None),
                    transform_mask=preprocessing.get("transform_mask", False),
                    export=Export(**modality["export"]) if modality.get("export") else None,
                    raise_on_error=raise_on_error,
                )
            logger.trace(f"Loaded modalities in {timer()}")

            # load attachment images
            if config.get("attachment_images"):
                for name, attach_to in config["attachment_images"].items():
                    self.attachment_images[name] = attach_to
                    logger.trace(f"Added attachment image '{name}' attached to '{attach_to}'")
                logger.trace(f"Loaded attachment images in {timer(since_last=True)}")

            if config.get("attachment_shapes"):
                for name, shape_dict in config["attachment_shapes"].items():
                    assert "shape_files" in shape_dict, "Shape dict missing 'shape_files' key."
                    assert "pixel_size" in shape_dict, "Shape dict missing 'pixel_size' key."
                    assert "attach_to" in shape_dict, "Shape dict missing 'attach_to' key."
                    self.attachment_shapes[name] = shape_dict
                logger.trace(f"Loaded attachment images in {timer(since_last=True)}")

            if config.get("attachment_points"):
                for name, shape_dict in config["attachment_points"].items():
                    assert "shape_files" in shape_dict, "Shape dict missing 'shape_files' key."
                    assert "pixel_size" in shape_dict, "Shape dict missing 'pixel_size' key."
                    assert "attach_to" in shape_dict, "Shape dict missing 'attach_to' key."
                    self.attachment_points[name] = shape_dict
                logger.trace(f"Loaded attachment images in {timer(since_last=True)}")

            # load merge modalities
            if config["merge_images"]:
                for name, merge_modalities in config["merge_images"].items():
                    self.merge_modalities[name] = merge_modalities
                logger.trace(f"Loaded merge modalities in {timer(since_last=True)}")

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
        if self.reference and self.has_modality(name_or_path=self.reference):
            errors.append(f"❌ Reference modality '{self.reference}' not found.")
            logger.error(errors[-1])

        is_valid = not errors
        if not is_valid:
            errors.append("❌ Project configuration is invalid.")
            logger.error(errors[-1])
        else:
            logger.success("✅ Project configuration is valid.")
        return is_valid, errors

    def clear(self, cache: bool = True, valis: bool = False, image: bool = False, metadata: bool = False) -> None:
        """Clear existing data."""
        from image2image_reg.utils.utilities import _safe_delete

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

        filelist = [modality.path for modality in self.modalities.values()]
        filelist = natsorted(filelist)
        filelist = [Path(path) for path in filelist]
        for path in filelist:
            assert path.exists(), f"{path} does not exist."
        return filelist

    def set_reference(self, name: str | None = None, path: str | None = None) -> None:
        """Set reference image."""
        if not name and not path:
            self._reference = None
        elif name:
            if name not in self.modalities:
                raise ValueError(f"Modality {name} not found.")
            path = self.modalities[name].path
            if not Path(path).exists():
                raise ValueError(f"Path {path} does not exist.")
            self._reference = path
        logger.trace(f"Set reference image to '{self._reference}'.")

    @property
    def reference(self) -> PathLike | None:
        """Get reference image."""
        reference = self._reference
        if reference:
            filelist = self.filelist
            reference = Path(reference)
            assert reference.exists(), f"{reference} does not exist."
            assert reference in filelist, f"{reference} not in filelist."
        return reference

    def register(self, **kwargs: ty.Any) -> None:
        """Co-register images."""
        from valis import registration

        from image2image_reg.valis.slide_io import Image2ImageSlideReader
        from image2image_reg.valis.utilities import get_feature_detector, get_micro_registration_dimension

        # get filelist
        filelist = self.filelist
        filelist = [str(s) for s in filelist]
        logger.info(f"Filelist has {len(filelist)} images.")
        # get reference
        reference = self.reference
        logger.info(f"Reference image: {reference}")
        # get detector
        feature_detector_cls = get_feature_detector(self.feature_detector)
        logger.info(f"Feature detector: {feature_detector_cls}")
        # get matcher
        # feature_matcher_cls = get_feature_matcher(self.feature_matcher)
        # logger.info(f"Feature matcher: {feature_matcher_cls}")
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
                    import pickle

                    with open(registrar_path, "rb") as f:
                        registrar = pickle.load(f)
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
                        **kws,
                    )
                    registrar.dst_dir = str(self.project_dir)
                    registrar.set_dst_paths()

                    with MeasureTimer() as timer:
                        registrar.register(processor_dict=channel_kws, reader_cls=Image2ImageSlideReader)
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
                self.registrar = registrar
            except Exception as exc:
                registration.kill_jvm()
                logger.exception(f"Error during registration: {exc}")
                raise exc
        logger.info(f"Completed registration in {main_timer()}")

    def write(self, **kwargs: ty.Any) -> list | None:
        """Export images after applying transformation."""
        from image2image_reg.valis.utilities import get_slide_path, transform_attached_image, transform_registered_image

        if not self.registrar:
            raise ValueError("Registrar not found. Please register first.")

        paths = []
        # export images to OME-TIFFs
        with MeasureTimer() as timer:
            registered_dir = self.image_dir
            registered_dir.mkdir(exist_ok=True, parents=True)
            paths_ = transform_registered_image(
                self.registrar, registered_dir, non_rigid_reg=self.non_rigid_registration
            )
            paths.extend(paths_)
        logger.info(f"Exported registered images in {timer()}")

        # export attached images to OME-TIFFs
        with MeasureTimer() as timer:
            attached_images = kwargs.get("attachment_images", {})
            for src_name, attachments in attached_images.items():
                src_path = get_slide_path(self.registrar, src_name)
                paths_ = transform_attached_image(self.registrar, src_path, attachments, registered_dir)
                paths.extend(paths_)
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

    from image2image_reg.valis.utilities import get_feature_detector_str

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
    non_rigid_reg: bool = False,
    micro_reg: bool = False,
    feature_detector: str = "sensitive_vgg",
    **kwargs,
) -> None:
    """Valis-based registration."""
    from koyo.timer import MeasureTimer
    from natsort import natsorted
    from valis import registration

    from image2image_reg.valis.utilities import (
        get_feature_detector,
        get_micro_registration_dimension,
        get_slide_path,
        transform_attached_image,
        transform_registered_image,
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
    if not non_rigid_reg:
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
                transform_registered_image(registrar, registered_dir, non_rigid_reg=non_rigid_reg)
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
    for modality in obj.modalities.values():
        name = valtils.get_name(str(modality.path))
        modality_kws = modality.preprocessing.to_valis()
        channel_kws[name] = [get_preprocessor(modality_kws[0]), modality_kws[1]]
    return channel_kws

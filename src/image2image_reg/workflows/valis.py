"""Valis interface."""
from __future__ import annotations

import typing as ty
from pathlib import Path

from koyo.json import read_json_data, write_json_data
from koyo.typing import PathLike
from koyo.utilities import is_installed
from loguru import logger

logger = logger.bind(src="Valis")

if not is_installed("valis"):
    raise ImportError("Please install valis to use this module.")


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


def get_preprocessor(preprocessor: str | type) -> type:
    """Get pre-processor."""
    import valis.preprocessing as pre_valis

    import image2image_reg.valis.preprocessing as pre_wsireg

    if isinstance(preprocessor, str):
        if hasattr(pre_wsireg, preprocessor):
            preprocessor = getattr(pre_wsireg, preprocessor)
        elif hasattr(pre_valis, preprocessor):
            preprocessor = getattr(pre_valis, preprocessor)
        else:
            raise ValueError(f"Preprocessor {preprocessor} not found.")
    return preprocessor


def get_preprocessing_for_path(path: PathLike) -> list[str, dict]:
    """Get preprocessing kws for specified image."""
    from image2image_io.config import CONFIG
    from image2image_io.readers import get_simple_reader

    with CONFIG.temporary_overwrite(only_last_pyramid=True, init_pyramid=False):
        reader = get_simple_reader(path)
        if reader.is_rgb:
            kws = ["ColorfulStandardizer", {"c": 0.2, "h": 0}]
        else:
            kws = ["MaxIntensityProjection", {"channel_names": reader.channel_names}]
    return kws


def get_feature_detector_str(feature_detector: str) -> str:
    """Get feature detector."""
    available = {
        "vgg": "VggFD",
        "orb_vgg": "OrbVggFD",
        "boost": "BoostFD",
        "latch": "LatchFD",
        "daisy": "DaisyFD",
        "kaze": "KazeFD",
        "akaze": "AkazeFD",
        "brisk": "BriskFD",
        "orb": "OrbFD",
        # custom
        "sensitive_vgg": "SensitiveVggFD",
    }
    all_available = list(available.values()) + list(available.keys())
    if feature_detector not in all_available:
        raise ValueError(f"Feature detector {feature_detector} not found. Please one of use: {all_available}")
    return available[feature_detector] if feature_detector in available else feature_detector


def get_feature_detector(feature_detector: str) -> type:
    """Get feature detector object."""
    import valis.feature_detectors as fd_valis

    import image2image_reg.valis.detect as fd_wsireg

    feature_detector = get_feature_detector_str(feature_detector)
    if isinstance(feature_detector, str):
        if hasattr(fd_wsireg, feature_detector):
            feature_detector = getattr(fd_wsireg, feature_detector)
        elif hasattr(fd_valis, feature_detector):
            feature_detector = getattr(fd_valis, feature_detector)
        else:
            raise ValueError(f"Feature detector {feature_detector} not found.")
    return feature_detector


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
    import numpy as np
    from koyo.timer import MeasureTimer
    from natsort import natsorted
    from valis import registration, valtils

    from image2image_reg.valis.utilities import get_slide_path, transform_attached_image, transform_registered_image

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
                except Exception:
                    logger.error("Failed to export matches.")

                # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an
                # image that is proportion of the full resolution.
                if micro_reg:
                    try:
                        img_dims = np.array(
                            [slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()]
                        )
                        min_max_size = np.min([np.max(d) for d in img_dims])
                        micro_reg_size = np.floor(min_max_size * micro_reg_fraction).astype(int)
                    except Exception:
                        micro_reg_size = 3000

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
                        except Exception as exc:
                            logger.error(f"Error during non-rigid registration: {exc}")
                    logger.info(f"Registered high-res images in {timer()}")

                # We can also plot the high resolution matches using `Valis.draw_matches`:
                try:
                    with MeasureTimer() as timer:
                        matches_dst_dir = Path(registrar.dst_dir) / "matches-final"
                        registrar.draw_matches(matches_dst_dir)
                    logger.info(f"Exported matches in {timer()}")
                except Exception as exc:
                    logger.error(f"Failed to export matches: {exc}.")

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
            registration.kill_jvm()
            raise exc
        registration.kill_jvm()
    logger.info(f"Completed registration in {main_timer()}")


def get_valis_registrar(project_name: str, output_dir: PathLike, init_jvm: bool = False) -> None:
    """Get Valis registrar if it's available."""
    # initialize java
    if init_jvm:
        from valis import registration

        registration.init_jvm()

    registrar = None
    output_dir = Path(output_dir)
    registrar_path = output_dir / project_name / "data" / f"{project_name}_registrar.pickle"
    if registrar_path.exists():
        import pickle

        with open(registrar_path, "rb") as f:
            registrar = pickle.load(f)
    return registrar

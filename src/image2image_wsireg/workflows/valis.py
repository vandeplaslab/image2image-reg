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


def valis_init_configuration(
    project_name: str,
    output_dir: PathLike,
    filelist: list[PathLike],
    reference: PathLike | None = None,
    check_for_reflections: bool = True,
    micro_reg_fraction: float = 0.125,
    non_rigid_reg: bool = False,
    micro_reg: bool = False,
):
    """Create Valis configuration."""
    from natsort import natsorted
    from valis import valtils

    logger.info(f"Creating configuration for '{project_name}' in '{output_dir}'")

    output_dir = Path(output_dir)
    filelist = natsorted(filelist)
    filelist = [Path(path).resolve() for path in filelist]
    for path in filelist:
        assert path.exists(), f"{path} does not exist."
    logger.info(f"Filelist has {len(filelist)} images.")

    if reference:
        reference = Path(reference)
        assert reference.exists(), f"{reference} does not exist."
        assert reference in filelist, f"{reference} not in filelist."
    logger.info(f"Reference image: {reference}")

    channel_kws = {}
    for path in filelist:
        name = valtils.get_name(str(path))
        channel_kws[name] = get_preprocessing_for_path(path)
        logger.trace(f"Preprocessing for {name}: {channel_kws[name]}")

    config = {
        "project_name": project_name,
        "filelist": [str(path) for path in filelist],
        "reference": str(reference),
        "channel_kws": channel_kws,
        "check_for_reflections": check_for_reflections,
        "non_rigid_reg": non_rigid_reg,
        "micro_reg": micro_reg,
        "micro_reg_fraction": micro_reg_fraction,
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
    **kwargs,
) -> None:
    """Valis-based registration."""
    import numpy as np
    from koyo.timer import MeasureTimer
    from natsort import natsorted
    from valis import registration, valtils

    from image2image_reg.valis.detect import SensitiveVggFD
    from image2image_reg.valis.utilities import transform_registered_image

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

    # initialize java
    registration.init_jvm()

    filelist = [str(s) for s in filelist]

    registered_dir = output_dir / project_name / "registered"
    registered_dir.mkdir(exist_ok=True, parents=True)

    kws = {}
    if not non_rigid_reg:
        kws["non_rigid_registrar_cls"] = None

    if kwargs:
        logger.warning(f"Unused keyword arguments: {kwargs}")

    try:
        # registrar_path = base_dir / name / "data" / f"{name}_registrar.pickle"
        # if registrar_path.exists():
        #     registrar = pickle.load(open(registrar_path, "rb"))
        # else:
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
            feature_detector_cls=SensitiveVggFD,
            **kws,
        )

        with MeasureTimer() as timer:
            registrar.register(processor_dict=channel_kws)
        logger.info(f"Registered low-res images in {timer()}")

        # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that
        # is 25% full resolution.
        if micro_reg:
            try:
                img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
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
                        align_to_reference=True,
                    )
                except Exception as exc:
                    logger.error(f"Error during non-rigid registration: {exc}")
            logger.info(f"Registered high-res images in {timer()}")

        # We can also plot the high resolution matches using `Valis.draw_matches`:
        try:
            matches_dst_dir = Path(registrar.dst_dir) / "matches"
            registrar.draw_matches(matches_dst_dir)
        except Exception:
            logger.error("Failed to export matches.")

        # export images to OME-TIFFs
        transform_registered_image(registrar, registered_dir, non_rigid_reg=non_rigid_reg)

    except Exception as exc:
        registration.kill_jvm()
        raise exc
    registration.kill_jvm()

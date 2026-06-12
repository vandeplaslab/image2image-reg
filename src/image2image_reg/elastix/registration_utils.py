"""Registration utilities."""

from __future__ import annotations

import math
import typing as ty
from pathlib import Path

import itk
import SimpleITK as sitk
from koyo.json import read_json_data, write_json_data
from koyo.timer import MeasureTimer
from loguru import logger

from image2image_reg.elastix.registration import Registration
from image2image_reg.preprocessing.convert import itk_image_to_sitk_image, sitk_image_to_itk_image
from image2image_reg.wrapper import normalize_max_registration_pixels
from image2image_reg.constants import DEFAULT_MAX_REGISTRATION_PIXELS
if ty.TYPE_CHECKING:
    from image2image_reg.wrapper import ImageWrapper


def sitk_pmap_to_dict(pmap: sitk.ParameterMap) -> dict[str, ty.Any]:
    """
    Convert SimpleElastix ParameterMap to python dictionary.

    Parameters
    ----------
    pmap
        SimpleElastix ParameterMap

    Returns
    -------
    Python dict of SimpleElastix ParameterMap
    """
    pmap_dict = {}
    for k, v in pmap.items():
        if k in ["image", "invert"]:
            t_pmap = {}
            for k2, v2 in v.items():
                t_pmap[k2] = v2
            pmap_dict[k] = t_pmap
        else:
            pmap_dict[k] = v
    return pmap_dict


def pmap_dict_to_sitk(pmap_dict: dict[str, ty.Any]) -> sitk.ParameterMap:
    """
    Convert python dict to SimpleElastix ParameterMap.

    Parameters
    ----------
    pmap_dict
        SimpleElastix ParameterMap in python dictionary

    Returns
    -------
    SimpleElastix ParameterMap of Python dict
    """
    # pmap = sitk.ParameterMap()
    # pmap = {}
    # for k, v in pmap_dict.items():
    #     pmap[k] = v
    return pmap_dict


def pmap_dict_to_json(pmap_dict: dict, output_file: str) -> None:
    """
    Save python dict of ITKElastix to json.

    Parameters
    ----------
    pmap_dict : dict
        parameter map stored in python dict
    output_file : str
        filepath of where to save the json
    """
    write_json_data(output_file, pmap_dict)


def json_to_pmap_dict(json_file: str) -> dict[str, ty.Any]:
    """
    Load python dict of SimpleElastix stored in json.

    Parameters
    ----------
    json_file : dict
        filepath to json contained SimpleElastix parameter map
    """
    return read_json_data(json_file)


def _prepare_reg_models(reg_params: list[str | Registration | dict[str, list[str]]]) -> list[dict[str, list[str]]]:
    prepared_params = []
    for rp in reg_params:
        if isinstance(rp, Registration):
            prepared_params.append(rp.value)
        elif isinstance(rp, str):
            prepared_params.append(Registration[rp].value)
        elif isinstance(rp, dict):
            prepared_params.append(rp)
    return prepared_params


def parameter_to_itk_pobj(reg_param_map: dict[str, list[str]]) -> itk.ParameterObject:
    """
    Transfer parameter data stored in dict to ITKElastix ParameterObject.

    Parameters
    ----------
    reg_param_map: dict
        elastix registration parameters

    Returns
    -------
    itk_param_map:itk.ParameterObject
        ITKElastix object for registration parameters
    """
    parameter_object = itk.ParameterObject.New()
    itk_param_map = parameter_object.GetDefaultParameterMap("rigid")
    for k, v in reg_param_map.items():
        itk_param_map[k] = v
    return itk_param_map


def _cap_sitk_image(image: sitk.Image, max_registration_pixels: int | None) -> tuple[sitk.Image, int]:
    """Cap a SimpleITK image by integer shrinking."""
    max_registration_pixels = normalize_max_registration_pixels(max_registration_pixels)
    if max_registration_pixels is None:
        return image, 1

    size = image.GetSize()
    n_pixels = int(size[0]) * int(size[1])
    if n_pixels <= max_registration_pixels:
        return image, 1

    factor = math.ceil(math.sqrt(n_pixels / max_registration_pixels))
    shrink_factors = [factor, factor, *([1] * (image.GetDimension() - 2))]
    return sitk.Shrink(image, shrink_factors), factor  # type: ignore[no-untyped-call]


def _resample_mask_to_image(mask: sitk.Image | None, image: sitk.Image) -> sitk.Image | None:
    """Resample a mask onto an image grid."""
    if mask is None or mask.GetSize() == image.GetSize():
        return mask
    transform = sitk.Transform(image.GetDimension(), sitk.sitkIdentity)
    return sitk.Resample(mask, image, transform, sitk.sitkNearestNeighbor, 0, mask.GetPixelID())


def register_2d_images(
    source: ImageWrapper,
    target: ImageWrapper,
    reg_params: list[dict[str, list[str]]],
    output_dir: str | Path,
    histogram_match: bool = False,
    return_image: bool = False,
    max_registration_pixels: int | None = DEFAULT_MAX_REGISTRATION_PIXELS,
) -> list[dict[str, list[str]]] | tuple[list[dict[str, list[str]]], sitk.Image]:
    """Register 2D images with multiple models and return a list of elastix transformation maps.

    Parameters
    ----------
    source : SimpleITK.Image
        RegImage of image to be aligned
    target : SimpleITK.Image
        RegImage that is being aligned to (grammar is hard)
    reg_params : list of dict
        registration parameter maps stored in a dict, can be file paths to SimpleElastix parameterMaps stored
        as text or one of the default parameter maps (see parameter_load() function)
    output_dir : str
        where to store registration outputs (iteration data and transformation files)
    histogram_match : bool
        whether to attempt histogram matching to improve registration
    return_image : bool
        whether to return the registered image
    max_registration_pixels : int, optional
        Maximum pixels per registration input. Values less than one disable this cap.

    Returns
    -------
        tform_list: list
            list of ITKElastix transformation parameter maps
        image: itk.Image
            resulting registered moving image
    """
    if source.image is None:
        raise ValueError("Source image is None")
    if target.image is None:
        raise ValueError("Target image is None")

    source_image, source_factor = _cap_sitk_image(source.image, max_registration_pixels)
    target_image, target_factor = _cap_sitk_image(target.image, max_registration_pixels)
    source_mask = source.mask
    target_mask = target.mask
    if source_factor > 1 and source_mask is not None:
        source_mask = sitk.Shrink(source_mask, [source_factor, source_factor])  # type: ignore[no-untyped-call]
    if target_factor > 1 and target_mask is not None:
        target_mask = sitk.Shrink(target_mask, [target_factor, target_factor])  # type: ignore[no-untyped-call]
    source_mask = _resample_mask_to_image(source_mask, source_image)
    target_mask = _resample_mask_to_image(target_mask, target_image)

    if histogram_match:
        with MeasureTimer() as timer:
            matcher = sitk.HistogramMatchingImageFilter()  # type: ignore[no-untyped-call]
            matcher.SetNumberOfHistogramLevels(64)  # type: ignore[no-untyped-call]
            matcher.SetNumberOfMatchPoints(7)  # type: ignore[no-untyped-call]
            matcher.ThresholdAtMeanIntensityOn()  # type: ignore[no-untyped-call]
            source_image = matcher.Execute(source_image, target_image)  # type: ignore[no-untyped-call]
        logger.info(f"Histogram matching took {timer()}")

    pixel_id = source_image.GetPixelID()  # type: ignore[union-attr]
    with MeasureTimer() as timer:
        moving_image = sitk_image_to_itk_image(source_image)
        fixed_image = sitk_image_to_itk_image(target_image)
        moving_mask = sitk_image_to_itk_image(source_mask) if source_mask is not None else None
        fixed_mask = sitk_image_to_itk_image(target_mask) if target_mask is not None else None
        source.release_image_data()
        target.release_image_data()
        del source_image, target_image, source_mask, target_mask
    logger.info(f"Converting images to ITK took {timer()}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Created output directory {output_dir}")

    # Create a registration object
    with MeasureTimer() as timer:
        logger.trace("Creating Elastix registrar")
        registrar = itk.ElastixRegistrationMethod.New(moving_image, fixed_image)
        logger.trace("Create Elastix registrar")
        registrar.SetLogToConsole(True)
        registrar.LogToConsoleOn()
        registrar.SetLogToFile(True)
        registrar.SetOutputDirectory(str(output_dir))
        logger.trace("Setup registrar parameters")
        if moving_mask is not None:
            registrar.SetMovingMask(moving_mask)
        if fixed_mask is not None:
            registrar.SetFixedMask(fixed_mask)
        logger.trace("Setup registrar images")

        # Set registration parameters
        parameter_object_registration = itk.ParameterObject.New()
        for idx, pmap in enumerate(reg_params):
            pmap = dict(pmap)
            if idx == 0:
                pmap["WriteResultImage"] = ["true"] if return_image else ["false"]
                if fixed_mask is not None:
                    pmap["AutomaticTransformInitialization"] = ["false"]
                else:
                    pmap["AutomaticTransformInitialization"] = ["true"]

                parameter_object_registration.AddParameterMap(pmap)
            else:
                pmap["WriteResultImage"] = ["true"] if return_image else ["false"]
                pmap["AutomaticTransformInitialization"] = ["false"]
                parameter_object_registration.AddParameterMap(pmap)
            logger.trace(f"Added registration model {idx} to registrar")
        registrar.SetParameterObject(parameter_object_registration)
        logger.trace("Finished setting up registrar")
    logger.info(f"Setting up registration took {timer()}")

    # Update filter object (required)
    with MeasureTimer() as timer:
        registrar.UpdateLargestPossibleRegion()
    logger.info(f"Registration took {timer()}")

    # Results of Registration
    result_transform_parameters = registrar.GetTransformParameterObject()

    # execute registration:
    tform_list = []
    for idx in range(result_transform_parameters.GetNumberOfParameterMaps()):
        tform = {}
        for k, v in result_transform_parameters.GetParameterMap(idx).items():
            tform[k] = v
        tform_list.append(tform)

    if return_image:
        image = registrar.GetOutput()
        image = itk_image_to_sitk_image(image)
        image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]
        del registrar, moving_image, fixed_image, moving_mask, fixed_mask
        return tform_list, image
    del registrar, moving_image, fixed_image, moving_mask, fixed_mask
    return tform_list

"""Registration utilities."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import itk
import SimpleITK as sitk
from koyo.json import read_json_data, write_json_data
from koyo.timer import MeasureTimer
from loguru import logger

from image2image_reg.elastix.registration import Registration
from image2image_reg.preprocessing.convert import itk_image_to_sitk_image

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


def register_2d_images(
    source: ImageWrapper,
    target: ImageWrapper,
    reg_params: list[dict[str, list[str]]],
    output_dir: str | Path,
    histogram_match: bool = False,
    return_image: bool = False,
):
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

    if histogram_match:
        with MeasureTimer() as timer:
            matcher = sitk.HistogramMatchingImageFilter()  # type: ignore[no-untyped-call]
            matcher.SetNumberOfHistogramLevels(64)  # type: ignore[no-untyped-call]
            matcher.SetNumberOfMatchPoints(7)  # type: ignore[no-untyped-call]
            matcher.ThresholdAtMeanIntensityOn()  # type: ignore[no-untyped-call]
            source.image = matcher.Execute(source.image, target.image)  # type: ignore[no-untyped-call]
        logger.info(f"Histogram matching took {timer()}")

    pixel_id = source.image.GetPixelID()  # type: ignore[union-attr]
    with MeasureTimer() as timer:
        source.sitk_to_itk(True)
        target.sitk_to_itk(True)
    logger.info(f"Converting images to ITK took {timer()}")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create registration object
    with MeasureTimer() as timer:
        selx = itk.ElastixRegistrationMethod.New(source.image, target.image)
        selx.SetLogToConsole(True)
        selx.LogToConsoleOn()
        selx.SetLogToFile(True)
        selx.SetOutputDirectory(str(output_dir))
        if source.mask is not None:
            selx.SetMovingMask(source.mask)
        if target.mask is not None:
            selx.SetFixedMask(target.mask)
        selx.SetMovingImage(source.image)
        selx.SetFixedImage(target.image)

        # Set registration parameters
        parameter_object_registration = itk.ParameterObject.New()
        for idx, pmap in enumerate(reg_params):
            if idx == 0:
                pmap["WriteResultImage"] = ["true"] if return_image else ["false"]
                if target.mask is not None:
                    pmap["AutomaticTransformInitialization"] = ["false"]
                else:
                    pmap["AutomaticTransformInitialization"] = ["true"]

                parameter_object_registration.AddParameterMap(pmap)
            else:
                pmap["WriteResultImage"] = ["true"] if return_image else ["false"]
                pmap["AutomaticTransformInitialization"] = ["false"]
                parameter_object_registration.AddParameterMap(pmap)
        selx.SetParameterObject(parameter_object_registration)
    logger.info(f"Setting up registration took {timer()}")

    # Update filter object (required)
    with MeasureTimer() as timer:
        selx.UpdateLargestPossibleRegion()
    logger.info(f"Registration took {timer()}")

    # Results of Registration
    result_transform_parameters = selx.GetTransformParameterObject()

    # execute registration:
    tform_list = []
    for idx in range(result_transform_parameters.GetNumberOfParameterMaps()):
        tform = {}
        for k, v in result_transform_parameters.GetParameterMap(idx).items():
            tform[k] = v
        tform_list.append(tform)

    if return_image:
        image = selx.GetOutput()
        image = itk_image_to_sitk_image(image)
        image = sitk.Cast(image, pixel_id)  # type: ignore[no-untyped-call]
        return tform_list, image
    return tform_list

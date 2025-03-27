"""Conversion utilities."""

from __future__ import annotations

from copy import deepcopy

import SimpleITK as sitk


def euler_elx_to_itk2d(tform: dict, is_translation: bool = False) -> sitk.Euler2DTransform:
    """Convert Elastix Euler transform to ITK Euler transform."""
    euler2d = sitk.Euler2DTransform()
    if is_translation:
        elx_parameters = [0]
        elx_parameters_trans = [float(p) for p in tform["TransformParameters"]]
        elx_parameters.extend(elx_parameters_trans)
    else:
        center = [float(p) for p in tform["CenterOfRotationPoint"]]
        euler2d.SetFixedParameters(center)
        elx_parameters = [float(p) for p in tform["TransformParameters"]]
    euler2d.SetParameters(elx_parameters)
    return euler2d


def similarity_elx_to_itk2d(tform: dict) -> sitk.Similarity2DTransform:
    """Convert Elastix similarity transform to ITK similarity transform."""
    similarity2d = sitk.Similarity2DTransform()

    center = [float(p) for p in tform["CenterOfRotationPoint"]]
    similarity2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform["TransformParameters"]]
    similarity2d.SetParameters(elx_parameters)
    return similarity2d


def affine_elx_to_itk2d(tform: dict) -> sitk.AffineTransform:
    """Convert Elastix affine transform to ITK affine transform."""
    im_dimension = len(tform["Size"])
    affine2d = sitk.AffineTransform(im_dimension)

    center = [float(p) for p in tform["CenterOfRotationPoint"]]
    affine2d.SetFixedParameters(center)
    elx_parameters = [float(p) for p in tform["TransformParameters"]]
    affine2d.SetParameters(elx_parameters)
    return affine2d


def bspline_elx_to_itk2d(tform: dict) -> sitk.BSplineTransform:
    """Convert Elastix BSpline transform to ITK BSpline transform."""
    im_dimension = len(tform["Size"])

    bspline2d = sitk.BSplineTransform(im_dimension, 3)
    bspline2d.SetTransformDomainOrigin([float(p) for p in tform["Origin"]])  # from fixed image
    bspline2d.SetTransformDomainPhysicalDimensions([int(p) for p in tform["Size"]])  # from fixed image
    bspline2d.SetTransformDomainDirection([float(p) for p in tform["Direction"]])  # from fixed image

    fixed_params = [int(p) for p in tform["GridSize"]]
    fixed_params += [float(p) for p in tform["GridOrigin"]]
    fixed_params += [float(p) for p in tform["GridSpacing"]]
    fixed_params += [float(p) for p in tform["GridDirection"]]
    bspline2d.SetFixedParameters(fixed_params)
    bspline2d.SetParameters([float(p) for p in tform["TransformParameters"]])
    return bspline2d


def convert_to_itk(tform: dict) -> sitk.Transform:
    """Convert Elastix transform to ITK transform."""
    itk_tform: sitk.Euler2DTransform | sitk.Similarity2DTransform | sitk.AffineTransform | sitk.BSplineTransform
    if tform["Transform"][0] == "EulerTransform":
        itk_tform = euler_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "SimilarityTransform":
        itk_tform = similarity_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "AffineTransform":
        itk_tform = affine_elx_to_itk2d(tform)
    elif tform["Transform"][0] == "TranslationTransform":
        itk_tform = euler_elx_to_itk2d(tform, is_translation=True)
    elif tform["Transform"][0] == "BSplineTransform":
        itk_tform = bspline_elx_to_itk2d(tform)
    else:
        raise ValueError(f"Transform {tform['Transform'][0]} not supported")
    return itk_tform


def get_elastix_transforms(transformations):
    """Get elastix transforms from reg_transforms."""
    elastix_transforms = deepcopy(transformations)

    for k, v in elastix_transforms.items():
        elastix_transforms.update({k: [t.elastix_transform for t in v]})
    return elastix_transforms

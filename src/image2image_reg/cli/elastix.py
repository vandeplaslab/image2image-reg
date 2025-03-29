"""I2Reg command line interface."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path

import click
from click_groups import GroupedGroup
from koyo.click import Parameter, cli_parse_paths_sort, exit_with_error, info_msg, print_parameters, warning_msg
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import reraise_exception_if_debug
from loguru import logger

from image2image_reg.cli._common import (
    ALLOW_EXTRA_ARGS,
    arg_split_bbox,
    as_uint8_,
    attach_image_,
    attach_points_,
    attach_shapes_,
    attach_to_,
    clip_,
    files_,
    fmt_,
    get_preprocessing,
    image_,
    modality_multi_,
    modality_single_,
    n_parallel_,
    original_size_,
    output_dir_,
    output_dir_current_,
    overwrite_,
    parallel_mode_,
    pixel_size_opt_,
    project_path_multi_,
    project_path_single_,
    remove_merged_,
    rename_,
    write_attached_,
    write_attached_images_,
    write_attached_points_,
    write_attached_shapes_,
    write_merged_,
    write_not_registered_,
    write_registered_,
)
from image2image_reg.elastix.registration_map import AVAILABLE_REGISTRATIONS
from image2image_reg.enums import PreprocessingOptions, PreprocessingOptionsWithNone, WriterMode

final_ = click.option(
    "--final/--no_final",
    help="Export final transformations (include all pre-processing and post-processing transformations).",
    is_flag=True,
    default=False,
    show_default=True,
)


def is_valis(project_dir: Path) -> bool:
    """Check if project is a Valis project."""
    return Path(project_dir).suffix in [".valis", ".i2valis"]


@click.group("elastix", cls=GroupedGroup)
def elastix() -> None:
    """I2Reg registration."""


@click.option(
    "--merge/--no_merge", help="Merge modalities once co-registered.", is_flag=True, default=True, show_default=True
)
@click.option(
    "--cache/--no_cache",
    help="Cache pre-processed images to speed things up in case you must redo something.",
    is_flag=True,
    default=True,
    show_default=True,
)
@modality_single_
@click.option(
    "-n",
    "--name",
    help="Name to be given to the project.",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@output_dir_
@elastix.command("new", help_group="Project")
def new_cmd(output_dir: str, name: str, cache: bool, merge: bool) -> None:
    """Create a new project."""
    new_runner(output_dir, name, cache, merge)


def new_runner(
    output_dir: str,
    name: str,
    cache: bool,
    merge: bool,
    # valis
    valis: bool = False,
    check_for_reflections: bool = False,
    non_rigid_registration: bool = False,
    micro_registration: bool = True,
    micro_registration_fraction: float = 0.125,
    feature_detector: str = "sensitive_vgg",
    feature_matcher: str = "RANSAC",
) -> None:
    """Create a new project."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    print_parameters(
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Name", "-n/--name", name),
        Parameter("Cache", "--cache/--no_cache", cache),
        Parameter("Merge", "--merge/--no_merge", merge),
    )
    if not valis:
        obj = ElastixReg(name=name, output_dir=output_dir, cache=cache, merge=merge)
    else:
        obj = ValisReg(
            name=name,
            output_dir=output_dir,
            cache=cache,
            merge=merge,
            check_for_reflections=check_for_reflections,
            non_rigid_registration=non_rigid_registration,
            micro_registration=micro_registration,
            micro_registration_fraction=micro_registration_fraction,
            feature_detector=feature_detector,
            feature_matcher=feature_matcher,
        )
    obj.save()


@project_path_single_
@elastix.command("about", help_group="Project")
def about_cmd(project_dir: ty.Sequence[str]) -> None:
    """Print information about the registration project."""
    about_runner(project_dir)


def about_runner(project_dir: str, valis: bool = False) -> None:
    """Add images to the project."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
    obj.print_summary()


@project_path_multi_
@elastix.command("validate", help_group="Project", aliases=["check"])
def validate_cmd(project_dir: ty.Sequence[str]) -> None:
    """Validate project configuration."""
    validate_runner(project_dir)


def validate_runner(paths: ty.Sequence[str], valis: bool = False) -> None:
    """Add images to the project."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    for project_dir in paths:
        obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
        obj.validate()


@overwrite_
@click.option(
    "-A",
    "--affine",
    help="Path to affine transformation matrix. Matrix must be in JSON format and follow the yx convention."
    "It's assumed to be in physical units. The matrix will be inversed before it's applied to the image.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-P",
    "--preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality.",
    type=click.Choice(PreprocessingOptions, case_sensitive=False),
    default=["basic"],
    show_default=True,
    required=False,
    multiple=True,
)
@click.option(
    "-b",
    "--mask_bbox",
    help="Bound box to be used for the mask. It must be supplied in the format: x,y,width,height and be in PIXEL units."
    " It will throw an error if fewer or more than 4 values are supplied.",
    type=click.UNPROCESSED,
    show_default=True,
    required=False,
    callback=arg_split_bbox,
)
@click.option(
    "-m",
    "--mask",
    help="Path to the mask(s) that should be associated with the image.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=False,
    callback=cli_parse_paths_sort,
)
@image_
@modality_multi_
@project_path_single_
@elastix.command("add-image", help_group="Project", aliases=["ai"])
def add_modality_cmd(
    project_dir: str,
    name: ty.Sequence[str],
    image: ty.Sequence[str],
    mask: ty.Sequence[str] | None,
    mask_bbox: tuple[int, int, int, int] | None,
    preprocessing: ty.Sequence[str],
    affine: ty.Sequence[str] | None,
    overwrite: bool = False,
) -> None:
    """Add images to the project."""
    add_modality_runner(project_dir, name, image, mask, mask_bbox, preprocessing, affine, overwrite)


def add_modality_runner(
    project_dir: str,
    names: ty.Sequence[str],
    paths: ty.Sequence[str],
    masks: ty.Sequence[str] | None = None,
    mask_bbox: tuple[int, int, int, int] | None = None,
    preprocessings: ty.Sequence[str | None] | None = None,
    affines: ty.Sequence[str] | None = None,
    overwrite: bool = False,
    methods: ty.Sequence[str] | None = None,
    valis: bool = False,
    reference: bool = False,
) -> None:
    """Add images to the project."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    if not isinstance(names, (list, tuple)):
        names = [names]  # type: ignore[list-item]
    if not isinstance(paths, (list, tuple)):
        paths = [paths]  # type: ignore[list-item]
    if not masks:
        masks = [None] * len(paths)  # type: ignore[list-item]
    if not isinstance(masks, (list, tuple)):
        masks = [masks]  # type: ignore[list-item]
    if len(masks) != len(paths) and len(masks) > 0:
        masks = [None] * len(paths)  # type: ignore[list-item]
    if not affines:
        affines = [None] * len(paths)  # type: ignore[list-item]
    if not isinstance(affines, (list, tuple)):
        affines = [affines]  # type: ignore[list-item]
    if not isinstance(preprocessings, (list, tuple)):
        preprocessings = [preprocessings]  # type: ignore[list-item]
    if len(preprocessings) == 1 and len(paths) > 1:
        preprocessings = preprocessings * len(paths)
        info_msg(f"Using same pre-processing for all images: {preprocessings[0]}")
    if len(affines) == 1 and len(paths) > 1:
        affines = affines * len(paths)
        info_msg(f"Using same pre-processing for all images: {affines[0]}")
    if len(methods) == 1 and len(paths) > 1:
        methods = methods * len(paths)
        info_msg(f"Using same method for all images: {methods[0]}")

    if len(names) != len(paths) != len(preprocessings) != len(masks):
        raise ValueError("Number of names, paths and pre-processing must match.")
    if masks[0] and mask_bbox:
        raise ValueError("Mask bounding box cannot be specified if mask is provided.")

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Names", "-n/--name", names),
        Parameter("Paths", "-i/--image", paths),
        Parameter("Masks", "-m/--mask", masks),
        Parameter("Mask bounding box", "-b/--mask_bbox", mask_bbox),
        Parameter("Pre-processing", "-P/--preprocessing", preprocessings),
        Parameter("Method", "-M/--method", methods),
        Parameter("Affine", "-A/--affine", affines),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )
    obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
    obj.set_logger()
    for name, path, mask, preprocessing, affine, method in zip(names, paths, masks, preprocessings, affines, methods):
        obj.auto_add_modality(
            name,
            path,
            preprocessing=get_preprocessing(preprocessing, affine, method=method if valis else None),
            mask=mask,
            mask_bbox=mask_bbox,
            overwrite=overwrite,
            method=method,
        )
        if reference and hasattr(obj, "set_reference"):
            obj.set_reference(path=path)
    obj.save()


@click.option(
    "-P",
    "--target_preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality - this will override modality-specific"
    " pre-processing.",
    type=click.Choice(PreprocessingOptionsWithNone, case_sensitive=False),
    default="none",
    show_default=True,
    required=False,
)
@click.option(
    "-S",
    "--source_preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality - this will override modality-specific"
    " pre-processing.",
    type=click.Choice(PreprocessingOptionsWithNone, case_sensitive=False),
    default="none",
    show_default=True,
    required=False,
)
@click.option(
    "-R",
    "--registration",
    help="Registration steps to be taken. These will be stacked.",
    type=click.Choice(
        AVAILABLE_REGISTRATIONS,
        case_sensitive=False,
    ),
    default=None,
    show_default=True,
    required=True,
    multiple=True,
)
@click.option(
    "-T",
    "--through",
    help="Source modality.",
    type=click.STRING,
    show_default=True,
    required=False,
)
@click.option(
    "-t",
    "--target",
    help="Target modality.",
    type=click.STRING,
    show_default=True,
    required=True,
)
@click.option(
    "-s",
    "--source",
    help="Source modality.",
    type=click.STRING,
    show_default=True,
    required=True,
)
@project_path_single_
@elastix.command("add-path", help_group="Project", aliases=["ap"])
def add_path_cmd(
    project_dir: str,
    source: str,
    target: str,
    through: str | None,
    registration: ty.Sequence[str],
    source_preprocessing: str | None,
    target_preprocessing: str | None,
) -> None:
    """Specify the registration path between the source and target (and maybe through) modalities."""
    add_path_runner(project_dir, source, target, through, registration, source_preprocessing, target_preprocessing)


def add_path_runner(
    project_dir: str,
    source: str,
    target: str,
    through: str | None,
    registration: ty.Sequence[str],
    source_preprocessing: str | None = None,
    target_preprocessing: str | None = None,
) -> None:
    """Add images to the project."""
    from image2image_reg.workflows.elastix import ElastixReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Source", "-s/--source", source),
        Parameter("Target", "-t/--target", target),
        Parameter("Through", "-T/--through", through),
        Parameter("Registration", "-R/--registration", registration),
        Parameter("Source pre-processing", "-P/--source_preprocessing", source_preprocessing),
        Parameter("Target pre-processing", "-S/--target_preprocessing", target_preprocessing),
    )
    obj = ElastixReg.from_path(project_dir)
    obj.set_logger()
    obj.add_registration_path(
        source,
        target,
        list(registration),
        through,
        {"source": get_preprocessing(source_preprocessing), "target": get_preprocessing(target_preprocessing)},
    )
    obj.save()


@attach_image_
@modality_multi_
@attach_to_
@project_path_single_
@elastix.command("attach-image", help_group="Project", aliases=["ati"])
def add_attachment_cmd(project_dir: str, attach_to: str, name: list[str], image: list[str]) -> None:
    """Add attachment image to registered modality."""
    add_attachment_runner(project_dir, attach_to, name, image)


def add_attachment_runner(
    project_dir: str, attach_to: str, names: list[str], paths: list[str], valis: bool = False
) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    if not isinstance(paths, (list, tuple)):
        names = [names]
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if len(paths) > len(names):
        raise ValueError("Number of names and paths must match.")
    if len(names) > 0 and len(paths) == 0:
        paths = [None] * len(names)
    if len(names) != len(paths):
        raise ValueError("Number of names and paths must match.")

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Attach to", "-a/--attach_to", attach_to),
        Parameter("Name", "-n/--name", names),
        Parameter("Image", "-i/--image", paths),
    )
    obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
    obj.set_logger()
    for name, path in zip(names, paths):
        obj.auto_add_attachment_images(attach_to, name, path)
    obj.save()


@pixel_size_opt_
@attach_points_
@modality_single_
@attach_to_
@project_path_single_
@elastix.command("attach-points", help_group="Project", aliases=["atp"])
def add_points_cmd(
    project_dir: str, attach_to: str, name: str, file: list[str | Path], pixel_size: float | None
) -> None:
    """Add attachment points (csv/tsv/txt) to registered modality."""
    add_points_runner(project_dir, attach_to, name, file, pixel_size)


def add_points_runner(
    project_dir: str, attach_to: str, name: str, paths: list[str | Path], pixel_size: float | None, valis: bool = False
) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Attach to", "-a/--attach_to", attach_to),
        Parameter("Name", "-n/--name", name),
        Parameter("Point Files", "-f/--file", paths),
        Parameter("Pixel size", "-s/--pixel_size", pixel_size),
    )
    obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
    obj.set_logger()
    obj.add_attachment_points(attach_to, name, paths, pixel_size)
    obj.save()


@pixel_size_opt_
@attach_shapes_
@modality_single_
@attach_to_
@project_path_single_
@elastix.command("attach-shape", help_group="Project", aliases=["ats"])
def add_shape_cmd(
    project_dir: str, attach_to: str, name: str, file: list[str | Path], pixel_size: float | None
) -> None:
    """Add attachment shape (GeoJSON) to registered modality."""
    add_shape_runner(project_dir, attach_to, name, file, pixel_size)


def add_shape_runner(
    project_dir: str, attach_to: str, name: str, paths: list[str | Path], pixel_size: float | None, valis: bool = False
) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Attach to", "-a/--attach_to", attach_to),
        Parameter("Name", "-n/--name", name),
        Parameter("Shape files", "-f/--file", paths),
        Parameter("Pixel size", "-s/--pixel_size", pixel_size),
    )
    obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
    obj.set_logger()
    obj.add_attachment_geojson(attach_to, name, paths)
    obj.save()


@click.option(
    "--auto",
    help="Automatically add all modalities inside a project.",
    is_flag=True,
    default=None,
    show_default=True,
)
@click.option(
    "-m",
    "--modality",
    help="Name of the modality that should be merged.",
    type=click.STRING,
    show_default=True,
    multiple=True,
    required=False,
    default=None,
)
@modality_single_
@project_path_multi_
@elastix.command("add-merge", help_group="Project", aliases=["am"])
def add_merge_cmd(project_dir: ty.Sequence[str], name: str, modality: ty.Iterable[str] | None, auto: bool) -> None:
    """Specify how (if) images should be merged."""
    add_merge_runner(project_dir, name, modality, auto)


def add_merge_runner(
    paths: ty.Sequence[str], name: str, modalities: ty.Iterable[str] | None, auto: bool = False, valis: bool = False
) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Name", "-n/--name", name),
        Parameter("Modalities", "-m/--modality", modalities),
    )
    for project_dir in paths:
        obj = ElastixReg.from_path(project_dir) if not valis else ValisReg.from_path(project_dir)
        if auto:
            obj.auto_add_merge_modalities(name)
        else:
            obj.add_merge_modalities(name, list(modalities))
        obj.save()


@overwrite_
@parallel_mode_
@n_parallel_
@project_path_multi_
@elastix.command("preprocess", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS, aliases=["pre"])
def preprocess_cmd(project_dir: ty.Sequence[str], n_parallel: int, parallel_mode: str, overwrite: bool) -> None:
    """Preprocess images."""
    preprocess_runner(project_dir, n_parallel, parallel_mode, overwrite)


def preprocess_runner(
    paths: ty.Sequence[str], n_parallel: int = 1, parallel_mode: str = "outer", overwrite: bool = False
) -> None:
    """Register images."""
    from mpire import WorkerPool

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Parallel mode", "-P/--parallel_mode", parallel_mode),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    with MeasureTimer() as timer:
        if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
            logger.trace(f"Running {n_parallel} actions in parallel.")
            with WorkerPool(n_parallel) as pool:
                args = [(path, n_parallel, overwrite) for path in paths]
                for path in pool.imap(_preprocess, args):
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        else:
            for path in paths:
                _preprocess(path, n_parallel, overwrite)
                logger.info(f"Finished processing {path} in {timer(since_last=True)}")
    logger.info(f"Finished pre-processing all projects in {timer()}.")


def _preprocess(path: PathLike, n_parallel: int, overwrite: bool = False) -> PathLike:
    from image2image_reg.workflows.elastix import ElastixReg

    obj = ElastixReg.from_path(path)
    obj.set_logger()
    obj.preprocess(n_parallel, overwrite=overwrite, quick=True)
    return path


@overwrite_
@parallel_mode_
@n_parallel_
@clip_
@rename_
@as_uint8_
@original_size_
@remove_merged_
@write_merged_
@write_attached_
@write_not_registered_
@write_registered_
@fmt_
@click.option(
    "-w/-W",
    "--write/--no_write",
    help="Write images to disk.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--histogram_match/--no_histogram_match",
    help="Match image histograms before co-registering - this might improve co-registration.",
    is_flag=True,
    default=False,
    show_default=True,
)
@project_path_multi_
@elastix.command("register", help_group="Execute", aliases=["run"])
def register_cmd(
    project_dir: ty.Sequence[str],
    histogram_match: bool,
    write: bool,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_attached: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    rename: bool,
    clip: str,
    n_parallel: int,
    parallel_mode: str,
    overwrite: bool,
) -> None:
    """Register images."""
    register_runner(
        project_dir,
        histogram_match=histogram_match,
        write_images=write,
        fmt=fmt,
        write_registered=write_registered,
        write_not_registered=write_not_registered,
        write_attached=write_attached,
        write_merged=write_merged,
        remove_merged=remove_merged,
        original_size=original_size,
        as_uint8=as_uint8,
        rename=rename,
        clip=clip,
        n_parallel=n_parallel,
        parallel_mode=parallel_mode,
        overwrite=overwrite,
    )


def register_runner(
    paths: ty.Sequence[str],
    histogram_match: bool = False,
    write_images: bool = True,
    fmt: WriterMode = "ome-tiff",
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_attached: bool = True,
    write_merged: bool = True,
    remove_merged: bool = True,
    original_size: bool = False,
    as_uint8: bool | None = False,
    rename: bool = False,
    clip: str = "ignore",
    n_parallel: int = 1,
    parallel_mode: str = "outer",
    overwrite: bool = False,
) -> None:
    """Register images."""
    from mpire import WorkerPool

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Write images", "--write/--no_write", write_images),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Write registered images", "--write_registered/--no_write_registered", write_registered),
        Parameter(
            "Write not-registered images", "--write_not_registered/--no_write_not_registered", write_not_registered
        ),
        Parameter("Write merged images", "--write_merged/--no_write_merged", write_merged),
        Parameter("Remove merged images", "--remove_merged/--no_remove_merged", remove_merged),
        Parameter("Write images in original size", "--original_size/--no_original_size", original_size),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Rename", "-rename/--no_rename", rename),
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    errors = []
    with MeasureTimer() as timer:
        if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
            with WorkerPool(n_parallel) as pool:
                for path in pool.imap(
                    _register,
                    [
                        (
                            path,
                            histogram_match,
                            write_images,
                            fmt,
                            write_registered,
                            write_not_registered,
                            write_attached,
                            write_merged,
                            remove_merged,
                            original_size,
                            as_uint8,
                            rename,
                            clip,
                            overwrite,
                        )
                        for path in paths
                    ],
                ):
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        else:
            for path in paths:
                try:
                    _register(
                        path,
                        histogram_match,
                        write_images,
                        fmt,
                        write_registered,
                        write_not_registered,
                        write_attached,
                        write_merged,
                        remove_merged,
                        original_size,
                        as_uint8=as_uint8,
                        rename=rename,
                        clip=clip,
                        n_parallel=n_parallel,
                        overwrite=overwrite,
                    )
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
                except Exception as exc:
                    logger.exception(f"Failed to process {path}.")
                    errors.append(path)
                    reraise_exception_if_debug(exc)
            if errors:
                errors = "\n- ".join(errors)
                logger.error(f"Failed to register the following projects: {errors}")
    logger.info(f"Finished registering all projects in {timer()}.")
    if errors:
        return exit_with_error()


def _register(
    path: PathLike,
    histogram_match: bool,
    write_images: bool,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_attached: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    rename: bool = False,
    clip: str = "ignore",
    n_parallel: int = 1,
    overwrite: bool = False,
) -> PathLike:
    from image2image_reg.workflows.elastix import ElastixReg

    obj = ElastixReg.from_path(path)
    obj.set_logger()
    obj.register(histogram_match=histogram_match)
    obj.preview()
    if write_images:
        obj.write(
            fmt=fmt,
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_attached=write_attached,
            write_merged=write_merged,
            remove_merged=remove_merged,
            to_original_size=original_size,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            overwrite=overwrite,
            rename=rename,
            clip=clip,
        )
    return path


@overwrite_
@click.option(
    "-l",
    "--pyramid",
    help="Pyramid level. We will use the lowest level by default, but you can decrease it to preview higher resolution"
    " image.",
    type=click.IntRange(-3, -1, clamp=True),
    default=-1,
    show_default=True,
)
@project_path_multi_
@elastix.command("preview", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def preview_cmd(project_dir: ty.Sequence[str], pyramid: int, overwrite: bool = False) -> None:
    """Update project paths (e.g after folder move)."""
    preview_runner(project_dir, pyramid, overwrite)


def preview_runner(paths: ty.Sequence[str], pyramid: int, overwrite: bool = False, valis: bool = False) -> None:
    """Register images."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Source directories", "-l/--pyramid", pyramid),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    for path in paths:
        obj = (ElastixReg if not valis else ValisReg).from_path(path)
        obj.set_logger()
        obj.preview(pyramid, overwrite)


@overwrite_
@parallel_mode_
@n_parallel_
@final_
@clip_
@rename_
@as_uint8_
@original_size_
@remove_merged_
@write_merged_
@write_attached_points_
@write_attached_shapes_
@write_attached_images_
@write_attached_
@write_not_registered_
@write_registered_
@fmt_
@project_path_multi_
@elastix.command("export", help_group="Execute", aliases=["write"])
def export_cmd(
    project_dir: ty.Sequence[str],
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_merged: bool,
    write_attached: bool,
    write_attached_images: bool | None,
    write_attached_shapes: bool | None,
    write_attached_points: bool | None,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    rename: bool,
    clip: str,
    final: bool,
    n_parallel: int,
    parallel_mode: str,
    overwrite: bool,
) -> None:
    """Export images."""
    export_runner(
        project_dir,
        fmt=fmt,
        write_registered=write_registered,
        write_not_registered=write_not_registered,
        write_merged=write_merged,
        write_attached=write_attached,
        write_attached_points=write_attached_points,
        write_attached_shapes=write_attached_shapes,
        write_attached_images=write_attached_images,
        remove_merged=remove_merged,
        original_size=original_size,
        as_uint8=as_uint8,
        rename=rename,
        clip=clip,
        final=final,
        n_parallel=n_parallel,
        parallel_mode=parallel_mode,
        overwrite=overwrite,
    )


def export_runner(
    paths: ty.Sequence[str],
    fmt: WriterMode = "ome-tiff",
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_attached: bool = True,
    write_attached_images: bool | None = None,
    write_attached_shapes: bool | None = None,
    write_attached_points: bool | None = None,
    write_merged: bool = True,
    remove_merged: bool = True,
    original_size: bool = False,
    as_uint8: bool | None = None,
    rename: bool = False,
    clip: str = "ignore",
    final: bool = False,
    n_parallel: int = 1,
    parallel_mode: str = "outer",
    overwrite: bool = False,
) -> None:
    """Register images."""
    from mpire import WorkerPool

    if not write_merged:
        remove_merged = False

    if any(v is not None for v in [write_attached_images, write_attached_shapes, write_attached_points]):
        write_attached = False
    if write_attached:
        write_attached_shapes = write_attached_points = write_attached_images = True

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Write registered images", "--write_registered/--no_write_registered", write_registered),
        Parameter(
            "Write not-registered images", "--write_not_registered/--no_write_not_registered", write_not_registered
        ),
        Parameter("Write not-registered images", "--write_attached/--no_write_attached", write_attached),
        Parameter("Write merged images", "--write_merged/--no_write_merged", write_merged),
        Parameter("Remove merged images", "--remove_merged/--no_remove_merged", remove_merged),
        Parameter("Write images in original size", "--original_size/--no_original_size", original_size),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Rename", "-rename/--no_rename", rename),
        Parameter("Overwrite", "-W/--overwrite", overwrite),
    )

    errors = []
    with MeasureTimer() as timer:
        if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
            with WorkerPool(n_parallel) as pool:
                for path, success in pool.imap(
                    _export,
                    [
                        (
                            path,
                            fmt,
                            write_registered,
                            write_not_registered,
                            write_attached_images,
                            write_attached_shapes,
                            write_attached_points,
                            write_merged,
                            remove_merged,
                            original_size,
                            as_uint8,
                            rename,
                            clip,
                            final,
                            1,
                            overwrite,
                        )
                        for path in paths
                    ],
                ):
                    log_func = logger.info if success else logger.error
                    log_func(f"Finished processing {path} in {timer(since_last=True)}")
                    if not success:
                        errors.append(path)
        else:
            for path in paths:
                path, success = _export(
                    path,
                    fmt,
                    write_registered,
                    write_not_registered,
                    write_attached_images,
                    write_attached_shapes,
                    write_attached_points,
                    write_merged,
                    remove_merged,
                    original_size,
                    as_uint8,
                    rename=rename,
                    clip=clip,
                    final=final,
                    n_parallel=n_parallel,
                    overwrite=overwrite,
                )
                log_func = logger.info if success else logger.error
                log_func(f"Finished processing {path} in {timer(since_last=True)}")
                if not success:
                    errors.append(path)
    logger.info(f"Finished exporting all projects in {timer()}.")
    if errors:
        errors = "\n- ".join(errors)
        logger.error(f"Failed to export the following projects:\n{errors}")
        return exit_with_error()


def _export(
    path: PathLike,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_attached_images: bool | None,
    write_attached_shapes: bool | None,
    write_attached_points: bool | None,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    rename: bool = False,
    clip: str = "ignore",
    final: bool = False,
    n_parallel: int = 1,
    overwrite: bool = False,
) -> tuple[PathLike, bool]:
    from image2image_reg.workflows.elastix import ElastixReg

    try:
        obj = ElastixReg.from_path(path)
        obj.set_logger()
        if not obj.is_registered:
            warning_msg(f"Project {obj.name} is not registered.")
            return path, False
        obj.write(
            fmt=fmt,
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_attached_images=write_attached_images,
            write_attached_shapes=write_attached_shapes,
            write_attached_points=write_attached_points,
            write_merged=write_merged,
            remove_merged=remove_merged,
            to_original_size=original_size,
            as_uint8=as_uint8,
            rename=rename,
            clip=clip,
            n_parallel=n_parallel,
            overwrite=overwrite,
        )
        if final:
            obj.export_transforms(
                write_registered=write_registered,
                write_not_registered=write_not_registered,
                write_attached=write_attached_images,
            )
        return path, True
    except ValueError:
        logger.exception(f"Failed to export {path}.")
        return path, False


@write_attached_
@write_not_registered_
@write_registered_
@original_size_
@project_path_multi_
@elastix.command("final", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def final_cmd(
    project_dir: ty.Sequence[str],
    original_size: bool,
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_attached: bool = True,
) -> None:
    """Export final transformations."""
    final_runner(project_dir, original_size, write_registered, write_not_registered, write_attached)


def final_runner(
    paths: ty.Sequence[str],
    original_size: bool,
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_attached: bool = True,
) -> None:
    """Register images."""
    from image2image_reg.workflows import ElastixReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Original size", "-o/--original_size/-O/--no_original_size", original_size),
        Parameter("Write registered", "--write_registered/--no_write_registered", write_registered),
        Parameter("Write not-registered", "--write_not_registered/--no_write_not_registered", write_not_registered),
        Parameter("Write attached", "--write_attached/--no_write_attached", write_attached),
    )

    if not any([write_registered, write_not_registered, write_attached]):
        warning_msg("No output specified. Nothing to do.")
        return

    for path in paths:
        obj = ElastixReg.from_path(path)
        obj.set_logger()
        obj.export_transforms(
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_attached=write_attached,
        )


@click.option(
    "-S",
    "--suffix",
    help="Suffix appended to the filename.",
    type=click.STRING,
    show_default=True,
    default="_transformed",
    multiple=False,
    required=True,
)
@click.option(
    "-i/-I",
    "--inverse/--no_inverse",
    help="Apply inverse transformation to the modality.",
    is_flag=True,
    default=False,
    show_default=True,
)
@clip_
@pixel_size_opt_
@as_uint8_
@output_dir_current_
@click.option(
    "-t",
    "--transform_file",
    help="Path to the final transformation file (usually ends with <file>.elastix.json",
    type=click.Path(exists=True, resolve_path=True, file_okay=True, dir_okay=False),
    show_default=True,
    required=True,
)
@files_
@elastix.command("transform", aliases=["apply"], help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def transform(
    files: ty.Sequence[str],
    transform_file: str,
    output_dir: str,
    as_uint8: bool | None,
    pixel_size: float | None,
    clip: str,
    inverse: bool,
    suffix: str,
) -> None:
    """Transform image, mask, points or GeoJSON data using Elastix transformation."""
    transform_runner(files, transform_file, output_dir, as_uint8, pixel_size, clip, inverse, suffix)


def transform_runner(
    files: ty.Sequence[str],
    transform_file: str,
    output_dir: str,
    as_uint8: bool | None = None,
    pixel_size: float | None = None,
    clip: str = "ignore",
    inverse: bool = False,
    suffix: str = "_transformed",
) -> None:
    """Apply transformation."""
    from image2image_reg.workflows.transform import transform_elastix

    print_parameters(
        Parameter("Files to transform", "-f/--file", files),
        Parameter("Transformation file", "-t/--transform_file", transform_file),
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Pixel size", "-s/--pixel_size", pixel_size),
        Parameter("Clip", "--clip", clip),
        Parameter("Inverse", "-i/--inverse/-I/--no_inverse", inverse),
        Parameter("Suffix", "-S/--suffix", suffix),
    )

    transform_elastix(
        files,
        transform_file,
        output_dir,
        as_uint8=as_uint8,
        pixel_size=pixel_size,
        clip=clip,
        inverse=inverse,
        suffix=suffix,
    )


@click.option(
    "--all",
    "all_",
    help="Clear all results.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-r/-R",
    "--progress/--no_progress",
    help="Clear progress.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-t/-T",
    "--final/--no_final",
    help="Clear final transformations.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-t/-T",
    "--transformations/--no_transformations",
    help="Clear transformations.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option("-i/-I", "--image/--no_image", help="Clear images.", is_flag=True, default=False, show_default=True)
@click.option("-c/-C", "--cache/--no_cache", help="Clear cache.", is_flag=True, default=False, show_default=True)
@project_path_multi_
@elastix.command("clear", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def clear_cmd(
    project_dir: ty.Sequence[str],
    cache: bool,
    image: bool,
    transformations: bool,
    final: bool,
    progress: bool,
    all_: bool,
) -> None:
    """Clear project data (cache/images/transformations/etc...)."""
    clear_runner(project_dir, cache, image, transformations, final, progress, all_)


def clear_runner(
    paths: ty.Sequence[str],
    cache: bool = False,
    image: bool = False,
    transformations: bool = False,
    final: bool = False,
    progress: bool = False,
    all_: bool = False,
) -> None:
    """Register images."""
    from image2image_reg.workflows import ElastixReg

    if all_:
        cache = image = transformations = progress = True

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Don't clear cache", "--cache", cache),
        Parameter("Don't clear images", "--image", image),
        Parameter("Don't clear transformations", "--transformations", transformations),
        Parameter("Don't clear final transformations", "--final", final),
        Parameter("Don't clear progress", "--progress", progress),
    )

    with MeasureTimer() as timer:
        for path in paths:
            obj = ElastixReg.from_path(path)
            obj.set_logger()
            obj.clear(cache=cache, image=image, transformations=transformations, final=final, progress=progress)
            logger.info(f"Finished clearing {path} in {timer(since_last=True)}")
    logger.info(f"Finished clearing all projects in {timer()}.")


@click.option(
    "-R",
    "--recursive",
    help="Recursively search for paths.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-s",
    "--source_dir",
    help="Source directory where images/files should be searched for.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
    multiple=True,
)
@project_path_multi_
@elastix.command("update", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def update_cmd(project_dir: list[str], source_dir: list[str], recursive: bool) -> None:
    """Update project paths (e.g after folder move)."""
    update_runner(project_dir, source_dir, recursive, valis=False)


def update_runner(
    paths: list[PathLike], source_dirs: list[PathLike], recursive: bool = False, valis: bool = False
) -> None:
    """Register images."""
    from image2image_reg.workflows import ElastixReg, ValisReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Source directories", "--source_dirs", source_dirs),
        Parameter("Recursive", "-R/--recursive", recursive),
    )

    for path in paths:
        klass = ElastixReg if not valis else ValisReg
        klass.update_paths(path, source_dirs, recursive=recursive)
        with suppress(ValueError):
            obj = klass.from_path(path)
            obj.set_logger()
            obj.validate(allow_not_registered=True, require_paths=True)

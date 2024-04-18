"""CLI."""

from __future__ import annotations

import sys
import typing as ty
from pathlib import Path

import click
from click_groups import GroupedGroup
from image2image_reg import __version__
from image2image_reg.enums import AVAILABLE_REGISTRATIONS, WriterMode
from koyo.click import (
    Parameter,
    arg_parse_framelist_multi,
    cli_parse_paths_sort,
    info_msg,
    print_parameters,
    warning_msg,
)
from koyo.system import IS_MAC
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import is_installed, running_as_pyinstaller_app
from loguru import logger

if ty.TYPE_CHECKING:
    from image2image_reg.models import Preprocessing

valis_is_installed = is_installed("valis")

# declare common options
ALLOW_EXTRA_ARGS = {"help_option_names": ["-h", "--help"], "ignore_unknown_options": True, "allow_extra_args": True}
override_ = click.option(
    "-W",
    "--override",
    help="Override existing data.",
    is_flag=True,
    default=None,
    show_default=True,
)
as_uint8_ = click.option(
    "-u/-U",
    "--as_uint8/--no_as_uint8",
    help="Downcast the image data format to uint8 which will substantially reduce the size of the files (unless it's"
    " already in uint8...).",
    is_flag=True,
    default=None,
    show_default=True,
)
original_size_ = click.option(
    "-o/-O",
    "--original_size/--no_original_size",
    help="Write images in their original size after applying transformations.",
    is_flag=True,
    default=True,
    show_default=True,
)
remove_merged_ = click.option(
    "-g/-G",
    "--remove_merged/--no_remove_merged",
    help="Remove written images that have been merged.",
    is_flag=True,
    default=True,
    show_default=True,
)
write_merged_ = click.option(
    "-m/-M",
    "--write_merged/--no_write_merged",
    help="Write merge images. Nothing will happen if merge modalities have not been specified.",
    is_flag=True,
    default=True,
    show_default=True,
)
write_not_registered_ = click.option(
    "-n/-N",
    "--write_not_registered/--no_write_not_registered",
    help="Write not-registered images.",
    is_flag=True,
    default=False,
    show_default=True,
)
write_registered_ = click.option(
    "-r/-R",
    "--write_registered/--no_write_registered",
    help="Write registered images.",
    is_flag=True,
    default=True,
    show_default=True,
)
fmt_ = click.option(
    "-f",
    "--fmt",
    help="Output format.",
    type=click.Choice(["ome-tiff"], case_sensitive=False),
    default="ome-tiff",
    show_default=True,
    required=False,
)
write_ = click.option(
    "-w/-W",
    "--write/--no_write",
    help="Write images to disk.",
    is_flag=True,
    default=True,
    show_default=True,
)
n_parallel_ = click.option(
    "-j",
    "--n_parallel",
    help="How many actions should be taken simultaneously.",
    type=click.IntRange(1, 24, clamp=True),
    show_default=True,
    default=1,
)
parallel_mode_ = click.option(
    "-P",
    "--parallel_mode",
    help="Parallel mode.",
    type=click.Choice(["inner", "outer"], case_sensitive=False),
    default="outer",
    show_default=True,
    required=False,
)
project_path_multi_ = click.option(
    "-p",
    "--project_dir",
    help="Path to the project directory. It usually ends in .i2reg extension.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
project_path_single_ = click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .i2reg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)


# noinspection PyUnusedLocal
def arg_split_bbox(ctx, param, value):
    """Split arguments."""
    if value is None:
        return None
    args = [int(arg.strip()) for arg in value.split(",")]
    assert len(args) == 4, "Bounding box must have 4 values"
    return args


def set_logger(verbosity: float, no_color: bool, log: PathLike | None = None) -> None:
    """Setup logger."""
    from koyo.logging import get_loguru_config, set_loguru_env, set_loguru_log

    level = verbosity * 10
    level, fmt, colorize, enqueue = get_loguru_config(level, no_color=no_color)  # type: ignore[assignment]
    set_loguru_env(fmt, level, colorize, enqueue)  # type: ignore[arg-type]
    set_loguru_log(level=level.upper(), no_color=no_color, logger=logger)  # type: ignore[attr-defined]
    logger.enable("image2image_reg")
    logger.enable("image2image_io")
    logger.enable("koyo")
    # override koyo logger
    set_loguru_log(level=level.upper(), no_color=no_color)  # type: ignore[attr-defined]
    logger.debug(f"Activated logger with level '{level}'.")
    if log:
        set_loguru_log(
            log,
            level=level.upper(),  # type: ignore[attr-defined]
            enqueue=enqueue,
            colorize=False,
            no_color=True,
            catch=True,
            diagnose=True,
            logger=logger,
            remove=False,
        )
        logger.trace(f"Command: {' '.join(sys.argv)}")


def get_preprocessing(preprocessing: str | None, affine: str | None = None) -> Preprocessing | None:
    """Get a pre-processing object."""
    from image2image_reg.models import Preprocessing

    if preprocessing in ["dark", "fluorescence"]:
        pre = Preprocessing.fluorescence()
    elif preprocessing in ["light", "brightfield"]:
        pre = Preprocessing.brightfield()
    elif preprocessing in ["basic"]:
        pre = Preprocessing.basic()
    else:
        pre = None
    if pre and affine:
        if affine and not Path(affine).suffix == ".json":
            raise ValueError("Affine must be a JSON file.")
        # will be automatically converted to a numpy array
        pre.affine = affine  # type: ignore[assignment]
    return pre


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120, "ignore_unknown_options": True},
    chain=True,
    cls=GroupedGroup,
)
@click.version_option(__version__, prog_name="wsireg")
@click.option(
    "--dev",
    help="Flat to indicate that CLI should run in development mode and catch all errors.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--no_color",
    help="Flag to disable colored logs (essential when logging to file).",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--quiet", "-q", "verbosity", flag_value=0, help="Minimal output - only errors and exceptions will be shown."
)
@click.option("--debug", "verbosity", flag_value=0.5, help="Maximum output - all messages will be shown.")
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
@click.option(
    "--log",
    help="Write logs to file (specify log path).",
    type=click.Path(exists=False, resolve_path=True, file_okay=True, dir_okay=False),
    default=None,
    show_default=True,
)
def cli(
    verbosity: float = 1,
    no_color: bool = False,
    dev: bool = False,
    log: PathLike | None = None,
) -> None:
    r"""Launch registration app."""
    from koyo.hooks import install_debugger_hook, uninstall_debugger_hook

    if IS_MAC:
        import os

        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
        logger.trace("Disabled OBJC fork safety on macOS.")

    if dev:
        if running_as_pyinstaller_app():
            click.echo("Developer mode is disabled in bundled app.")
            dev = False
        else:
            verbosity = 0.5
    verbosity = min(0.2, verbosity) * 10

    if dev:
        install_debugger_hook()
        verbosity = 0
    elif dev:
        uninstall_debugger_hook()
    set_logger(verbosity, no_color, log)


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
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to the WsiReg project directory. It usually ends in .i2reg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("new", help_group="Project")
def new_cmd(output_dir: str, name: str, cache: bool, merge: bool) -> None:
    """Create a new project."""
    new_runner(output_dir, name, cache, merge)


def new_runner(output_dir: str, name: str, cache: bool, merge: bool) -> None:
    """Create a new project."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    print_parameters(
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Name", "-n/--name", name),
        Parameter("Cache", "--cache/--no_cache", cache),
        Parameter("Merge", "--merge/--no_merge", merge),
    )
    obj = IWsiReg(name=name, output_dir=output_dir, cache=cache, merge=merge)
    obj.save()


@project_path_single_
@cli.command("about", help_group="Project")
def about_cmd(project_dir: ty.Sequence[str]) -> None:
    """Print information about the registration project."""
    about_runner(project_dir)


def about_runner(project_dir: str) -> None:
    """Add images to the project."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    obj = IWsiReg.from_path(project_dir)
    obj.print_summary()


@project_path_multi_
@cli.command("validate", help_group="Project")
def validate_cmd(project_dir: ty.Sequence[str]) -> None:
    """Validate project configuration."""
    validate_runner(project_dir)


def validate_runner(paths: ty.Sequence[str]) -> None:
    """Add images to the project."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    for project_dir in paths:
        obj = IWsiReg.from_path(project_dir)
        obj.validate()


@override_
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
    type=click.Choice(["basic", "light", "dark"], case_sensitive=False),
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
@click.option(
    "-i",
    "--image",
    help="Path to the image(s) that should be co-registered.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=True,
    required=True,
)
@project_path_single_
@cli.command("add-image", help_group="Project")
def add_modality_cmd(
    project_dir: str,
    name: ty.Sequence[str],
    image: ty.Sequence[str],
    mask: ty.Sequence[str] | None,
    mask_bbox: tuple[int, int, int, int] | None,
    preprocessing: ty.Sequence[str],
    affine: ty.Sequence[str] | None,
    override: bool = False,
) -> None:
    """Add images to the project."""
    add_modality_runner(project_dir, name, image, mask, mask_bbox, preprocessing, affine, override)


def add_modality_runner(
    project_dir: str,
    names: ty.Sequence[str],
    paths: ty.Sequence[str],
    masks: ty.Sequence[str] | None = None,
    mask_bbox: tuple[int, int, int, int] | None = None,
    preprocessings: ty.Sequence[str | None] | None = None,
    affines: ty.Sequence[str] | None = None,
    override: bool = False,
) -> None:
    """Add images to the project."""
    from image2image_reg.workflows.iwsireg import IWsiReg

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
        Parameter("Affine", "-A/--affine", affines),
        Parameter("Override", "-W/--override", override),
    )
    obj = IWsiReg.from_path(project_dir)
    for name, path, mask, preprocessing, affine in zip(names, paths, masks, preprocessings, affines):
        obj.auto_add_modality(
            name,
            path,
            preprocessing=get_preprocessing(preprocessing, affine),
            mask=mask,
            mask_bbox=mask_bbox,
            override=override,
        )
    obj.save()


@click.option(
    "-P",
    "--target_preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality - this will override modality-specific"
    " pre-processing.",
    type=click.Choice(["none", "basic", "light", "dark"], case_sensitive=False),
    default="none",
    show_default=True,
    required=False,
)
@click.option(
    "-S",
    "--source_preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality - this will override modality-specific"
    " pre-processing.",
    type=click.Choice(["none", "basic", "light", "dark"], case_sensitive=False),
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
@cli.command("add-path", help_group="Project")
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
    from image2image_reg.workflows.iwsireg import IWsiReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Source", "-s/--source", source),
        Parameter("Target", "-t/--target", target),
        Parameter("Through", "-T/--through", through),
        Parameter("Registration", "-R/--registration", registration),
        Parameter("Source pre-processing", "-P/--source_preprocessing", source_preprocessing),
        Parameter("Target pre-processing", "-S/--target_preprocessing", target_preprocessing),
    )
    obj = IWsiReg.from_path(project_dir)
    obj.add_registration_path(
        source,
        target,
        list(registration),
        through,
        {"source": get_preprocessing(source_preprocessing), "target": get_preprocessing(target_preprocessing)},
    )
    obj.save()


@click.option(
    "-i",
    "--image",
    help="Path to image/GeoJSON/points file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=False,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=True,
    required=True,
)
@click.option(
    "-a",
    "--attach_to",
    help="Name of the modality to which the attachment should be added.",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@project_path_single_
@cli.command("add-attachment", help_group="Project")
def add_attachment_cmd(project_dir: str, attach_to: str, name: list[str], image: list[str]) -> None:
    """Add attachment image to registered modality."""
    add_attachment_runner(project_dir, attach_to, name, image)


def add_attachment_runner(project_dir: str, attach_to: str, names: list[str], paths: list[str]) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows.iwsireg import IWsiReg

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
    obj = IWsiReg.from_path(project_dir)
    for name, path in zip(names, paths):
        obj.auto_add_attachment_images(attach_to, name, path)
    obj.save()


@click.option(
    "-f",
    "--file",
    help="Path to GeoJSON file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@click.option(
    "-a",
    "--attach_to",
    help="Name of the modality to which the attachment should be added.",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@project_path_single_
@cli.command("add-points", help_group="Project")
def add_points_cmd(project_dir: str, attach_to: str, name: str, file: list[str | Path]) -> None:
    """Add attachment points (csv/tsv/txt) to registered modality."""
    add_points_runner(project_dir, attach_to, name, file)


def add_points_runner(project_dir: str, attach_to: str, name: str, paths: list[str | Path]) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Attach to", "-a/--attach_to", attach_to),
        Parameter("Name", "-n/--name", name),
        Parameter("Point Files", "-f/--file", paths),
    )
    obj = IWsiReg.from_path(project_dir)
    obj.add_attachment_points(attach_to, name, paths)
    obj.save()


@click.option(
    "-f",
    "--file",
    help="Path to GeoJSON file that should be attached to the <attach_to> modality.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@click.option(
    "-a",
    "--attach_to",
    help="Name of the modality to which the attachment should be added.",
    type=click.STRING,
    show_default=True,
    multiple=False,
    required=True,
)
@project_path_single_
@cli.command("add-shape", help_group="Project")
def add_shape_cmd(project_dir: str, attach_to: str, name: str, file: list[str | Path]) -> None:
    """Add attachment shape (GeoJSON) to registered modality."""
    add_shape_runner(project_dir, attach_to, name, file)


def add_shape_runner(project_dir: str, attach_to: str, name: str, paths: list[str | Path]) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    if not isinstance(paths, (list, tuple)):
        paths = [paths]

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", project_dir),
        Parameter("Attach to", "-a/--attach_to", attach_to),
        Parameter("Name", "-n/--name", name),
        Parameter("Shape files", "-f/--file", paths),
    )
    obj = IWsiReg.from_path(project_dir)
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
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    required=True,
)
@project_path_multi_
@cli.command("add-merge", help_group="Project")
def add_merge_cmd(project_dir: ty.Sequence[str], name: str, modality: ty.Iterable[str] | None, auto: bool) -> None:
    """Specify how (if) images should be merged."""
    add_merge_runner(project_dir, name, modality, auto)


def add_merge_runner(
    paths: ty.Sequence[str], name: str, modalities: ty.Iterable[str] | None, auto: bool = False
) -> None:
    """Add attachment modality."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Name", "-n/--name", name),
        Parameter("Modalities", "-m/--modality", modalities),
    )
    for path in paths:
        obj = IWsiReg.from_path(path)
        if auto:
            obj.auto_add_merge_modalities(name)
        else:
            obj.add_merge_modalities(name, list(modalities))
        obj.save()


@override_
@parallel_mode_
@n_parallel_
@project_path_multi_
@cli.command("preprocess", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def preprocess_cmd(project_dir: ty.Sequence[str], n_parallel: int, parallel_mode: str, override: bool) -> None:
    """Preprocess images."""
    preprocess_runner(project_dir, n_parallel, parallel_mode, override)


def preprocess_runner(
    paths: ty.Sequence[str], n_parallel: int = 1, parallel_mode: str = "outer", override: bool = False
) -> None:
    """Register images."""
    from mpire import WorkerPool

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Parallel mode", "-P/--parallel_mode", parallel_mode),
        Parameter("Override", "-W/--override", override),
    )

    with MeasureTimer() as timer:
        if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
            logger.trace(f"Running {n_parallel} actions in parallel.")
            with WorkerPool(n_parallel) as pool:
                args = [(path, n_parallel, override) for path in paths]
                for path in pool.imap(_preprocess, args):
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        else:
            for path in paths:
                _preprocess(path, n_parallel, override)
                logger.info(f"Finished processing {path} in {timer(since_last=True)}")
    logger.info(f"Finished processing all projects in {timer()}.")


def _preprocess(path: PathLike, n_parallel: int, override: bool = False) -> PathLike:
    from image2image_reg.workflows.iwsireg import IWsiReg

    obj = IWsiReg.from_path(path)
    obj.set_logger()
    obj.preprocess(n_parallel, override=override, quick=True)
    return path


@override_
@parallel_mode_
@n_parallel_
@as_uint8_
@original_size_
@remove_merged_
@write_merged_
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
@cli.command("register", help_group="Execute")
def register_cmd(
    project_dir: ty.Sequence[str],
    histogram_match: bool,
    write: bool,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    n_parallel: int,
    parallel_mode: str,
    override: bool,
) -> None:
    """Register images."""
    register_runner(
        project_dir,
        histogram_match=histogram_match,
        write_images=write,
        fmt=fmt,
        write_registered=write_registered,
        write_merged=write_merged,
        remove_merged=remove_merged,
        write_not_registered=write_not_registered,
        original_size=original_size,
        as_uint8=as_uint8,
        n_parallel=n_parallel,
        parallel_mode=parallel_mode,
        override=override,
    )


def register_runner(
    paths: ty.Sequence[str],
    histogram_match: bool = False,
    write_images: bool = True,
    fmt: WriterMode = "ome-tiff",
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_merged: bool = True,
    remove_merged: bool = True,
    original_size: bool = False,
    as_uint8: bool | None = False,
    n_parallel: int = 1,
    parallel_mode: str = "outer",
    override: bool = False,
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
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Override", "-W/--override", override),
    )

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
                            write_merged,
                            remove_merged,
                            original_size,
                            as_uint8,
                            override,
                        )
                        for path in paths
                    ],
                ):
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        else:
            errors = []
            for path in paths:
                try:
                    _register(
                        path,
                        histogram_match,
                        write_images,
                        fmt,
                        write_registered,
                        write_not_registered,
                        write_merged,
                        remove_merged,
                        original_size,
                        as_uint8,
                        n_parallel=n_parallel,
                        override=override,
                    )
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
                except Exception:
                    logger.exception(f"Failed to process {path}.")
                    errors.append(path)
            if errors:
                errors = "\n- ".join(errors)
                logger.error(f"Failed to register the following projects: {errors}")
    logger.info(f"Finished processing all projects in {timer()}.")


def _register(
    path: PathLike,
    histogram_match: bool,
    write_images: bool,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    n_parallel: int = 1,
    override: bool = False,
) -> PathLike:
    from image2image_reg.workflows.iwsireg import IWsiReg

    obj = IWsiReg.from_path(path)
    obj.set_logger()
    obj.register(histogram_match=histogram_match)
    if write_images:
        obj.write_images(
            fmt=fmt,
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_merged=write_merged,
            remove_merged=remove_merged,
            to_original_size=original_size,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            override=override,
        )
    return path


if valis_is_installed:

    @click.option(
        "-M",
        "--no_micro_reg",
        help="Perform non-rigid registration.",
        is_flag=True,
        default=None,
        show_default=True,
    )
    @click.option(
        "-N",
        "--no_non_rigid_reg",
        help="Perform non-rigid registration.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-R",
        "--check_for_reflection",
        help="Check for reflection.",
        is_flag=True,
        default=None,
        show_default=True,
    )
    @click.option(
        "-r",
        "--reference",
        help="Path to the reference image.",
        type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
        show_default=True,
        required=False,
        default=None,
    )
    @click.option(
        "-i",
        "--image",
        help="Path to the image(s) that should be co-registered.",
        type=click.UNPROCESSED,
        show_default=True,
        multiple=True,
        required=True,
        callback=cli_parse_paths_sort,
    )
    @click.option(
        "-o",
        "--output_dir",
        help="Path to the WsiReg project directory. It usually ends in .i2reg extension.",
        type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
        show_default=True,
        required=False,
        default=".",
    )
    @click.option(
        "-n",
        "--name",
        help="Name to be given to the specified image (modality).",
        type=click.STRING,
        show_default=True,
        multiple=False,
        required=True,
    )
    @cli.command("valis-init", help_group="Valis")
    def valis_init(
        name: str,
        output_dir: PathLike,
        image: list[PathLike],
        reference: PathLike,
        check_for_reflection: bool,
        no_non_rigid_reg: bool,
        no_micro_reg: bool,
    ) -> None:
        """Initialize Valis configuration file."""
        valis_init_runner(name, output_dir, image, reference)

    def valis_init_runner(
        project_name: str,
        output_dir: PathLike,
        path: list[PathLike],
        reference: PathLike,
        check_for_reflection: bool = True,
        no_non_rigid_reg: bool = False,
        no_micro_reg: bool = False,
    ):
        """Register list of images using Valis algorithm."""
        from image2image_reg.workflows.valis import valis_init_configuration

        print_parameters(
            Parameter("Project name", "-n/--name", project_name),
            Parameter("Output directory", "-o/--output_dir", output_dir),
            Parameter("Paths", "-i/--image", path),
            Parameter("Reference", "-r/--reference", reference),
            Parameter("Check for reflection", "-R/--check_for_reflection", check_for_reflection),
            Parameter("No non-rigid registration", "-N/--no_non_rigid_reg", no_non_rigid_reg),
            Parameter("No micro registration", "-M/--no_micro_reg", no_micro_reg),
        )
        valis_init_configuration(
            project_name,
            output_dir,
            path,
            reference,
            check_for_reflection,
            not no_non_rigid_reg,
            not no_micro_reg,
        )

    @click.option(
        "-o",
        "--output_dir",
        help="Path to the WsiReg project directory. It usually ends in .i2reg extension.",
        type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
        show_default=True,
        required=False,
        default=".",
    )
    @click.option(
        "-c",
        "--config",
        help="Path to the configuration file.",
        type=click.Path(file_okay=True, dir_okay=False, resolve_path=True),
        show_default=True,
        required=True,
    )
    @cli.command("valis-register", help_group="Valis")
    def valis_register(config: PathLike, output_dir: PathLike):
        """Register images using the Valis algorithm."""
        valis_register_runner(output_dir, config)

    def valis_register_runner(output_dir: PathLike, config: PathLike | None):
        """Register list of images using Valis algorithm."""
        from image2image_reg.workflows.valis import valis_registration_from_config

        print_parameters(
            Parameter("Output directory", "-o/--output_dir", output_dir),
            Parameter("Config", "-c/--config", config),
        )

        valis_registration_from_config(output_dir=output_dir, config=config)


@click.option(
    "-R",
    "--no_progress",
    help="Clear progress.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option(
    "-T",
    "--no_transformations",
    help="Clear transformations.",
    is_flag=True,
    default=False,
    show_default=True,
)
@click.option("-I", "--no_image", help="Clear images.", is_flag=True, default=False, show_default=True)
@click.option("-C", "--no_cache", help="Clear cache.", is_flag=True, default=False, show_default=True)
@project_path_multi_
@cli.command("clear", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
def clear_cmd(
    project_dir: ty.Sequence[str], no_cache: bool, no_image: bool, no_transformations: bool, no_progress: bool
) -> None:
    """Clear project data (cache/images/transformations/etc...)."""
    clear_runner(project_dir, no_cache, no_image, no_transformations, no_progress)


def clear_runner(
    paths: ty.Sequence[str], no_cache: bool, no_image: bool, no_transformations: bool, no_progress: bool
) -> None:
    """Register images."""
    from image2image_reg.workflows.iwsireg import IWsiReg

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Don't clear cache", "--no_cache", no_cache),
        Parameter("Don't clear images", "--no_image", no_image),
        Parameter("Don't clear transformations", "--no_transformations", no_transformations),
        Parameter("Don't clear progress", "--no_progress", no_progress),
    )

    with MeasureTimer() as timer:
        for path in paths:
            pro = IWsiReg.from_path(path)
            pro.clear(
                cache=not no_cache, image=not no_image, transformations=not no_transformations, progress=not no_progress
            )
            logger.info(f"Finished clearing {path} in {timer(since_last=True)}")
    logger.info(f"Finished clearing all projects in {timer()}.")


@override_
@parallel_mode_
@n_parallel_
@as_uint8_
@original_size_
@remove_merged_
@write_merged_
@write_not_registered_
@write_registered_
@fmt_
@project_path_multi_
@cli.command("export", help_group="Execute")
def export_cmd(
    project_dir: ty.Sequence[str],
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    n_parallel: int,
    parallel_mode: str,
    override: bool,
) -> None:
    """Export images."""
    export_runner(
        project_dir,
        fmt=fmt,
        write_registered=write_registered,
        write_not_registered=write_not_registered,
        write_merged=write_merged,
        remove_merged=remove_merged,
        original_size=original_size,
        as_uint8=as_uint8,
        n_parallel=n_parallel,
        parallel_mode=parallel_mode,
        override=override,
    )


def export_runner(
    paths: ty.Sequence[str],
    fmt: WriterMode = "ome-tiff",
    write_registered: bool = True,
    write_not_registered: bool = True,
    write_merged: bool = True,
    remove_merged: bool = True,
    original_size: bool = False,
    as_uint8: bool | None = False,
    n_parallel: int = 1,
    parallel_mode: str = "outer",
    override: bool = False,
) -> None:
    """Register images."""
    from mpire import WorkerPool

    if not write_merged:
        remove_merged = False

    print_parameters(
        Parameter("Project directory", "-p/--project_dir", paths),
        Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Write registered images", "--write_registered/--no_write_registered", write_registered),
        Parameter(
            "Write not-registered images", "--write_not_registered/--no_write_not_registered", write_not_registered
        ),
        Parameter("Write merged images", "--write_merged/--no_write_merged", write_merged),
        Parameter("Remove merged images", "--remove_merged/--no_remove_merged", remove_merged),
        Parameter("Write images in original size", "--original_size/--no_original_size", original_size),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Override", "-W/--override", override),
    )

    with MeasureTimer() as timer:
        if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
            with WorkerPool(n_parallel) as pool:
                for path in pool.imap(
                    _export,
                    [
                        (
                            path,
                            fmt,
                            write_registered,
                            write_not_registered,
                            write_merged,
                            remove_merged,
                            original_size,
                            as_uint8,
                            override,
                        )
                        for path in paths
                    ],
                ):
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        else:
            for path in paths:
                _export(
                    path,
                    fmt,
                    write_registered,
                    write_not_registered,
                    write_merged,
                    remove_merged,
                    original_size,
                    as_uint8,
                    n_parallel=n_parallel,
                    override=override,
                )
                logger.info(f"Finished processing {path} in {timer(since_last=True)}")
    logger.info(f"Finished processing all projects in {timer()}.")


def _export(
    path: PathLike,
    fmt: WriterMode,
    write_registered: bool,
    write_not_registered: bool,
    write_merged: bool,
    remove_merged: bool,
    original_size: bool,
    as_uint8: bool | None,
    n_parallel: int = 1,
    override: bool = False,
) -> PathLike:
    from image2image_reg.workflows.iwsireg import IWsiReg

    obj = IWsiReg.from_path(path)
    obj.set_logger()
    if not obj.is_registered:
        warning_msg(f"Project {obj.name} is not registered.")
        return path
    obj.write_images(
        fmt=fmt,
        write_registered=write_registered,
        write_not_registered=write_not_registered,
        write_merged=write_merged,
        remove_merged=remove_merged,
        to_original_size=original_size,
        as_uint8=as_uint8,
        n_parallel=n_parallel,
        override=override,
    )
    return path


@override_
@as_uint8_
@fmt_
@click.option(
    "-C",
    "--channel_ids",
    type=click.STRING,
    default=None,
    help="Specify channel ids in the format: 1,2,4-6. You can provide multiple. If you are providing any, make sure to"
    " provide one for each file you are trying to merge.",
    callback=arg_parse_framelist_multi,
    show_default=True,
    multiple=True,
    required=False,
)
@click.option(
    "-b",
    "--crop_bbox",
    help="Bound box to be used for cropping of the image(s). It must be supplied in the format: x,y,width,height and"
    " be in PIXEL units. It will throw an error if fewer or more than 4 values are supplied.",
    type=click.UNPROCESSED,
    show_default=True,
    required=False,
    callback=arg_split_bbox,
)
@click.option(
    "-o",
    "--output_dir",
    help="Path to images to be merged.",
    type=click.Path(file_okay=False, dir_okay=True, resolve_path=True),
    show_default=True,
    required=True,
)
@click.option(
    "-p",
    "--path",
    help="Path to images to be merged.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-n",
    "--name",
    help="Name of the merged image.",
    type=click.STRING,
    show_default=True,
    required=True,
)
@cli.command("merge", help_group="Utility")
def merge_cmd(
    name: str,
    path: ty.Sequence[str],
    output_dir: str,
    crop_bbox: tuple[int, int, int, int] | None,
    channel_ids: ty.Sequence[tuple] | None,
    fmt: WriterMode,
    as_uint8: bool | None,
    override: bool,
) -> None:
    """Export images."""
    merge_runner(name, path, output_dir, crop_bbox, channel_ids, fmt, as_uint8, override)


def merge_runner(
    name: str,
    paths: ty.Sequence[str],
    output_dir: str,
    crop_bbox: tuple[int, int, int, int] | None,
    channel_ids: ty.Sequence[tuple] | None,
    fmt: WriterMode = "ome-tiff",
    as_uint8: bool | None = False,
    override: bool = False,
) -> None:
    """Register images."""
    from image2image_reg.workflows.merge import merge as merge_images

    print_parameters(
        Parameter("Name", "-n/--name", name),
        Parameter("Image paths", "-p/--path", paths),
        Parameter("Output directory", "-o/--output_dir", output_dir),
        Parameter("Crop bounding box", "-b/--crop_bbox", crop_bbox),
        Parameter("Channel ids", "-C/--channel_ids", channel_ids),
        Parameter("Output format", "-f/--fmt", fmt),
        Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
        Parameter("Override", "-W/--override", override),
    )

    if channel_ids:
        if len(channel_ids) != len(paths):
            raise ValueError("Number of channel ids must match number of images.")

    with MeasureTimer() as timer:
        merge_images(
            name, list(paths), output_dir, crop_bbox, fmt, as_uint8, channel_ids=channel_ids, override=override
        )
    logger.info(f"Finished processing project in {timer()}.")


@cli.command("convert", help_group="Utility")
def convert_cmd() -> None:
    """Convert images to pyramidal OME-TIFF."""


def main():
    """Execute the "imimspy" command line program."""
    cli.main(windows_expand_args=False)


if __name__ == "__main__":
    cli.main(windows_expand_args=False)

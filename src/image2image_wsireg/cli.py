"""CLI."""
from __future__ import annotations

import typing as ty

import click
from click_groups import GroupedGroup
from koyo.click import cli_parse_paths, info_msg, warning_msg
from koyo.utilities import running_as_pyinstaller_app
from loguru import logger

from image2image_wsireg import __version__
from image2image_wsireg.enums import ImageType, WriterMode

if ty.TYPE_CHECKING:
    from image2image_wsireg.models import Preprocessing


def set_logger(verbosity: float, no_color: bool) -> None:
    """Setup logger."""
    from koyo.logging import get_loguru_config, set_loguru_env, set_loguru_log

    level = verbosity * 10
    level, fmt, colorize, enqueue = get_loguru_config(level, no_color=no_color)
    set_loguru_env(fmt, level, colorize, enqueue)
    set_loguru_log(level=level.upper(), no_color=no_color, logger=logger)
    logger.enable("image2image_wsireg")
    # override koyo logger
    set_loguru_log(level=level.upper(), no_color=no_color)
    logger.enable("koyo")
    logger.debug(f"Activated logger with level '{level}'.")


def get_preprocessing(preprocessing: str) -> Preprocessing | None:
    """Get pre-processing object."""
    from image2image_wsireg.models import Preprocessing

    if preprocessing is None:
        return None
    elif preprocessing in ["dark", "fluorescence"]:
        return Preprocessing(image_type=ImageType.DARK, as_uint8=True, max_int_proj=True, contrast_enhance=True)
    elif preprocessing in ["light", "brightfield"]:
        return Preprocessing(image_type=ImageType.LIGHT, as_uint8=True, max_int_proj=False, invert_intensity=True)
    return None


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120, "ignore_unknown_options": True},
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
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("--quiet", "-q", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=0.5, help="Maximum output")
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli(
    verbosity: float = 1,
    no_color: bool = False,
    dev: bool = False,
) -> None:
    r"""Launch registration app."""
    from koyo.hooks import install_debugger_hook, uninstall_debugger_hook

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
    set_logger(verbosity, no_color)


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
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("new", help_group="Project")
def new(output_dir: str, name: str, cache: bool, merge: bool) -> None:
    """Create a new project."""
    new_runner(output_dir, name, cache, merge)


def new_runner(output_dir: str, name: str, cache: bool, merge: bool) -> None:
    """Create a new project."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d(name=name, output_dir=output_dir, cache=cache, merge=merge)
    obj.save()


@click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("about", help_group="Project")
def about(project_dir: str) -> None:
    """Add images to the project."""
    about_runner(project_dir)


def about_runner(project_dir: str) -> None:
    """Add images to the project."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d.from_path(project_dir)
    obj.print_summary()


@click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("validate", help_group="Project")
def validate(project_dir: str) -> None:
    """Add images to the project."""
    validate_runner(project_dir)


def validate_runner(project_dir: str) -> None:
    """Add images to the project."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d.from_path(project_dir)
    obj.validate()


@click.option(
    "-P",
    "--preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality.",
    type=click.Choice(["none", "light", "dark"], case_sensitive=False),
    default=["none"],
    show_default=True,
    required=False,
    multiple=True,
)
@click.option(
    "-m",
    "--mask",
    help="Path to the mask(s) that should be associated with the image.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=False,
    callback=cli_parse_paths,
)
@click.option(
    "-i",
    "--image",
    help="Path to the image(s) that should be co-registered.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths,
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
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("add-image", help_group="Project")
def add_modality(
    project_dir: str,
    name: ty.Sequence[str],
    image: ty.Sequence[str],
    mask: ty.Sequence[str] | None,
    preprocessing: ty.Sequence[str],
) -> None:
    """Add images to the project."""
    add_modality_runner(project_dir, name, image, mask, preprocessing)


def add_modality_runner(
    project_dir: str,
    names: ty.Sequence[str],
    paths: ty.Sequence[str],
    masks: ty.Sequence[str],
    preprocessings: ty.Sequence[str | None] | None = None,
) -> None:
    """Add images to the project."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    if not isinstance(names, (list, tuple)):
        names = [names]
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    if not isinstance(masks, (list, tuple)):
        masks = [masks]
    if len(masks) != len(paths) and len(masks) > 0:
        masks = [None] * len(paths)
    if not isinstance(preprocessings, (list, tuple)):
        preprocessings = [preprocessings]

    if len(preprocessings) == 1 and len(paths) > 1:
        preprocessings = preprocessings * len(paths)
        info_msg(f"Using same pre-processing for all images: {preprocessings[0]}")

    if len(names) != len(paths) != len(preprocessings):
        raise ValueError("Number of names, paths and pre-processing must match.")

    obj = WsiReg2d.from_path(project_dir)
    for name, path, mask, preprocessing in zip(names, paths, masks, preprocessings):
        obj.auto_add_modality(name, path, mask, get_preprocessing(preprocessing))
    obj.save()


@click.option(
    "-P",
    "--preprocessing",
    help="Kind of pre-processing that will be applied to the specified modality - this will override modality-specific"
    " pre-processing.",
    type=click.Choice(["none", "light", "dark"], case_sensitive=False),
    default="none",
    show_default=True,
    required=False,
)
@click.option(
    "-R",
    "--registration",
    help="Registration steps to be taken. These will be stacked.",
    type=click.Choice(
        [
            "rigid",
            "affine",
            "similarity",
            "nl",
            "fi_correction",
            "nl_reduced",
            "nl_mid",
            "nl2",
            "rigid_expanded",
            "rigid_ams",
            "affine_ams",
            "nl_ams",
            "rigid_anc",
            "affine_anc",
            "similarity_anc",
            "nl_anc",
        ],
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
@click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("add-path", help_group="Project")
def add_path(
    project_dir: str,
    source: str,
    target: str,
    through: str | None,
    registration: ty.Sequence[str],
    preprocessing: str | None,
) -> None:
    """Specify registration path between source and target (and maybe through) modalities."""
    add_path_runner(project_dir, source, target, through, registration, preprocessing)


def add_path_runner(
    project_dir: str,
    source: str,
    target: str,
    through: str | None,
    registration: ty.Sequence[str],
    preprocessing: str | None = None,
) -> None:
    """Add images to the project."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d.from_path(project_dir)
    obj.add_registration_path(source, target, through, list(registration), get_preprocessing(preprocessing))
    obj.save()


@click.option(
    "-i",
    "--image",
    help="Path to the image(s) that should be co-registered.",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    required=True,
    callback=cli_parse_paths,
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
    multiple=True,
    required=True,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("add-attachment", help_group="Project")
def add_attachment(project_dir: str, attach_to: str, name: str, image: str) -> None:
    """Add attachment modality."""
    add_attachment_runner(project_dir, attach_to, name, image)


def add_attachment_runner(project_dir: str, attach_to: str, name: str, image: str) -> None:
    """Add attachment modality."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d.from_path(project_dir)
    obj.auto_add_attachment_images(attach_to, name, image)
    obj.save()


@click.option(
    "-m",
    "--modality",
    help="Name of the modality that should be merged.",
    type=click.STRING,
    show_default=True,
    multiple=True,
    required=True,
)
@click.option(
    "-n",
    "--name",
    help="Name to be given to the specified image (modality).",
    type=click.STRING,
    show_default=True,
    required=True,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the WsiReg project directory. It usually ends in .wsireg extension.",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
)
@cli.command("add-merge", help_group="Project")
def add_merge(project_dir: str, attach_to: str, modality: ty.Sequence[str]) -> None:
    """Specify how (if) images should be merged."""
    add_merge_runner(project_dir, attach_to, modality)


def add_merge_runner(project_dir: str, name: str, modalities: ty.Sequence[str]) -> None:
    """Add attachment modality."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    obj = WsiReg2d.from_path(project_dir)
    obj.add_merge_modalities(name, list(modalities))
    obj.save()


# def add_geojson():
#     """Add geojson modality."""


def from_template(project_dir: str, template: ty.Literal["mxif", "3d", "he-preaf-postaf"]) -> None:
    """Create a new project from a template."""
    # from_template_runner(project_dir, template)


@click.option(
    "-n",
    "--n_parallel",
    help="How many actions should be taken simultaneously.",
    type=click.IntRange(1, 24, clamp=True),
    show_default=True,
    default=1,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the project directory. It usually ends in .annotine extension.",
    type=click.UNPROCESSED,
    # type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths,
)
@cli.command("preprocess", help_group="Execute")
def preprocess(project_dir: ty.Sequence[str], n_parallel: int) -> None:
    """Preprocess images."""
    preprocess_runner(project_dir, n_parallel)


def preprocess_runner(paths: ty.Sequence[str], n_parallel: int = 1) -> None:
    """Register images."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    for path in paths:
        obj = WsiReg2d.from_path(path)
        obj.set_logger()
        obj.preprocess(n_parallel=n_parallel)


@click.option(
    "--original_size/--no_original_size",
    help="Write images in their original size after applying transformations.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--write_not_registered/--no_write_not_registered",
    help="Write not-registered images.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--clear_merged/--no_clear_merged",
    help="Remove written images that have been merged.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--write/--no_write",
    help="Write images to disk.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "-f",
    "--fmt",
    help="Output format.",
    type=click.Choice(["ome-tiff"], case_sensitive=False),
    default="ome-tiff",
    show_default=True,
    required=False,
)
@click.option(
    "-n",
    "--n_parallel",
    help="How many actions should be taken simultaneously.",
    type=click.IntRange(1, 24, clamp=True),
    show_default=True,
    default=1,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the project directory. It usually ends in .annotine extension.",
    type=click.UNPROCESSED,
    # type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths,
)
@cli.command("register", help_group="Execute")
def register(
    project_dir: ty.Sequence[str],
    n_parallel: int,
    fmt: WriterMode,
    write: bool,
    clear_merged: bool,
    write_not_registered: bool,
    original_size: bool,
) -> None:
    """Register images."""
    register_runner(project_dir, n_parallel, fmt, write, clear_merged, write_not_registered, original_size)


def register_runner(
    paths: ty.Sequence[str],
    n_parallel: int = 1,
    fmt: WriterMode = "ome-tiff",
    write_images: bool = True,
    remove_merged: bool = True,
    write_not_registered: bool = True,
    original_size: bool = False,
) -> None:
    """Register images."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    for path in paths:
        obj = WsiReg2d.from_path(path)
        obj.set_logger()
        obj.register(n_parallel=n_parallel)
        if write_images:
            obj.write_images(
                n_parallel=n_parallel,
                writer=fmt,
                write_not_registered=write_not_registered,
                remove_merged=remove_merged,
                to_original_size=original_size,
            )


# @click.option(
#     "--preview/--no_preview",
#     help="Preview transformation before generating final images.",
#     is_flag=True,
#     default=False,
#     show_default=True,
# )
@click.option(
    "--original_size/--no_original_size",
    help="Write images in their original size after applying transformations.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--write_not_registered/--no_write_not_registered",
    help="Write not-registered images.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "--remove_merged/--no_remove_merged",
    help="Remove written images that have been merged.",
    is_flag=True,
    default=True,
    show_default=True,
)
@click.option(
    "-f",
    "--fmt",
    help="Output format.",
    type=click.Choice(["ome-tiff"], case_sensitive=False),
    default="ome-tiff",
    show_default=True,
    required=False,
)
@click.option(
    "-n",
    "--n_parallel",
    help="How many actions should be taken simultaneously.",
    type=click.IntRange(1, 24, clamp=True),
    show_default=True,
    default=1,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the project directory. It usually ends in .annotine extension.",
    type=click.UNPROCESSED,
    show_default=True,
    required=True,
    multiple=True,
    callback=cli_parse_paths,
)
@cli.command("export", help_group="Execute")
def export(
    project_dir: ty.Sequence[str],
    n_parallel: int,
    fmt: WriterMode,
    remove_merged: bool,
    write_not_registered: bool,
    original_size: bool,
) -> None:
    """Export images."""
    export_runner(project_dir, n_parallel, fmt, remove_merged, write_not_registered, original_size)


def export_runner(
    paths: ty.Sequence[str],
    n_parallel: int = 1,
    fmt: WriterMode = "ome-tiff",
    remove_merged: bool = True,
    write_not_registered: bool = True,
    original_size: bool = False,
    preview: bool = False,
) -> None:
    """Register images."""
    from image2image_wsireg.workflows.wsireg2d import WsiReg2d

    for path in paths:
        obj = WsiReg2d.from_path(path)
        obj.set_logger()
        if not obj.is_registered:
            warning_msg(f"Project {obj.name} is not registered.")
            continue
        obj.write_images(
            n_parallel=n_parallel,
            writer=fmt,
            write_not_registered=write_not_registered,
            remove_merged=remove_merged,
            to_original_size=original_size,
            preview=preview,
        )


def main():
    """Execute the "imimspy" command line program."""
    cli.main(windows_expand_args=False)


if __name__ == "__main__":
    cli.main(windows_expand_args=False)

"""Valis registration"""

from __future__ import annotations

import typing as ty
from pathlib import Path

import click
from click_groups import GroupedGroup
from koyo.click import (
    Parameter,
    cli_parse_paths_sort,
    print_parameters,
    warning_msg,
)
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import is_installed, reraise_exception_if_debug
from loguru import logger

from image2image_reg.enums import ValisDetectorMethod, ValisMatcherMethod, ValisPreprocessingMethod, WriterMode

from ._common import (
    ALLOW_EXTRA_ARGS,
    as_uint8_,
    fmt_,
    n_parallel_,
    overwrite_,
    parallel_mode_,
    project_path_multi_,
    project_path_single_,
    remove_merged_,
    write_merged_,
    write_not_registered_,
    write_registered_,
)


def cli_parse_method(ctx, param, value: str) -> str:
    """Parse pre-processing."""
    return [
        {
            "cs": "ColorfulStandard",
            "lum": "Luminosity",
            "he": "HEPreprocessing",
            "mip": "MaxIntensityProjection",
            "i2r": "I2RegPreprocessor",
        }.get(v, v)
        for v in value
    ]


def cli_parse_detector(ctx, param, value: str) -> str:
    """Parse detector."""
    return {"svgg": "sensitive_vgg", "vsvgg": "very_sensitive_vgg"}.get(value, value)


def cli_parse_matcher(ctx, param, value: str) -> str:
    """Parse matcher."""
    return {"ransac": "RANSAC", "gms": "GMS"}.get(value, value)


valis = None
if is_installed("valis"):

    @click.group("valis", cls=GroupedGroup)
    def valis() -> None:
        """Valis registration."""

    @click.option(
        "-f",
        "--fraction",
        help="Micro-registration fraction.",
        type=click.FloatRange(0, 1, clamp=True),
        default=0.125,
        show_default=True,
    )
    @click.option(
        "--micro/--no_micro",
        help="Perform micro registration.",
        is_flag=True,
        default=True,
        show_default=True,
    )
    @click.option(
        "--reflect/--no_reflect",
        help="Check for reflections.",
        is_flag=True,
        default=True,
        show_default=True,
    )
    @click.option(
        "-M",
        "--match",
        help="Feature matching method.",
        type=click.Choice(ty.get_args(ValisMatcherMethod), case_sensitive=False),
        default="RANSAC",
        show_default=True,
        required=False,
        callback=cli_parse_matcher,
    )
    @click.option(
        "-D",
        "--detect",
        help="Feature detection method.",
        type=click.Choice(ty.get_args(ValisDetectorMethod), case_sensitive=False),
        default="sensitive_vgg",
        show_default=True,
        required=False,
        callback=cli_parse_detector,
    )
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
    @valis.command("new", help_group="Project")
    def new_cmd(
        output_dir: str,
        name: str,
        cache: bool,
        merge: bool,
        detect: bool,
        match: bool,
        reflect: bool,
        micro: bool,
        fraction: float,
    ) -> None:
        """Create a new project."""
        from image2image_reg.cli.i2reg import new_runner

        new_runner(
            output_dir,
            name,
            cache,
            merge,
            valis=True,
            feature_detector=detect,
            feature_matcher=match,
            check_for_reflections=reflect,
            micro_registration=micro,
            micro_registration_fraction=fraction,
        )

    @project_path_single_
    @valis.command("about", help_group="Project")
    def about_cmd(project_dir: ty.Sequence[str]) -> None:
        """Print information about the registration project."""
        from image2image_reg.cli.i2reg import about_runner

        about_runner(project_dir, valis=True)

    @project_path_multi_
    @valis.command("validate", help_group="Project")
    def validate_cmd(project_dir: ty.Sequence[str]) -> None:
        """Validate project configuration."""
        from image2image_reg.cli.i2reg import validate_runner

        validate_runner(project_dir, valis=True)

    @overwrite_
    @click.option(
        "-R",
        "--reference",
        help="Set modality as the reference image to which other images will be referenced to.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-M",
        "--method",
        help="Pre-processing method.",
        type=click.Choice(ty.get_args(ValisPreprocessingMethod), case_sensitive=False),
        default=["auto"],
        show_default=True,
        required=False,
        multiple=True,
        callback=cli_parse_method,
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
    @valis.command("add-image", help_group="Project")
    def add_modality_cmd(
        project_dir: str,
        name: ty.Sequence[str],
        image: ty.Sequence[str],
        preprocessing: ty.Sequence[str],
        method: str,
        reference: bool,
        overwrite: bool = False,
    ) -> None:
        """Add images to the project."""
        from image2image_reg.cli.i2reg import add_modality_runner

        add_modality_runner(
            project_dir,
            name,
            image,
            preprocessings=preprocessing,
            overwrite=overwrite,
            methods=method,
            valis=True,
            reference=reference,
        )

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
    @valis.command("add-attachment", help_group="Project")
    def add_attachment_cmd(project_dir: str, attach_to: str, name: list[str], image: list[str]) -> None:
        """Add attachment image to registered modality."""
        from image2image_reg.cli.i2reg import add_attachment_runner

        add_attachment_runner(project_dir, attach_to, name, image, valis=True)

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
    @valis.command("add-points", help_group="Project")
    def add_points_cmd(project_dir: str, attach_to: str, name: str, file: list[str | Path]) -> None:
        """Add attachment points (csv/tsv/txt) to registered modality."""
        from image2image_reg.cli.i2reg import add_points_runner

        add_points_runner(project_dir, attach_to, name, file, valis=True)

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
    @valis.command("add-shape", help_group="Project")
    def add_shape_cmd(project_dir: str, attach_to: str, name: str, file: list[str | Path]) -> None:
        """Add attachment shape (GeoJSON) to registered modality."""
        from image2image_reg.cli.i2reg import add_shape_runner

        add_shape_runner(project_dir, attach_to, name, file, valis=True)

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
    @valis.command("add-merge", help_group="Project")
    def add_merge_cmd(project_dir: ty.Sequence[str], name: str, modality: ty.Iterable[str] | None, auto: bool) -> None:
        """Specify how (if) images should be merged."""
        from image2image_reg.cli.i2reg import add_merge_runner

        add_merge_runner(project_dir, name, modality, auto, valis=True)

    @overwrite_
    @parallel_mode_
    @n_parallel_
    @as_uint8_
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
    @project_path_multi_
    @valis.command("register", help_group="Execute")
    def register_cmd(
        project_dir: ty.Sequence[str],
        write: bool,
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_merged: bool,
        remove_merged: bool,
        as_uint8: bool | None,
        n_parallel: int,
        parallel_mode: str,
        overwrite: bool,
    ) -> None:
        """Register images."""
        register_runner(
            project_dir,
            write_images=write,
            fmt=fmt,
            write_registered=write_registered,
            write_merged=write_merged,
            remove_merged=remove_merged,
            write_not_registered=write_not_registered,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            parallel_mode=parallel_mode,
            overwrite=overwrite,
        )

    def _valis_register(
        path: PathLike,
        write_images: bool,
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_merged: bool,
        remove_merged: bool,
        as_uint8: bool | None,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> PathLike:
        import os

        from image2image_reg.workflows import ValisReg

        # limit Vips concurrency to avoid memory issues
        os.environ["VIPS_CONCURRENCY"] = "6"

        obj = ValisReg.from_path(path)
        obj.set_logger()
        obj.register()
        if write_images:
            obj.write(
                fmt=fmt,
                write_registered=write_registered,
                write_not_registered=write_not_registered,
                write_merged=write_merged,
                remove_merged=remove_merged,
                as_uint8=as_uint8,
                n_parallel=n_parallel,
                overwrite=overwrite,
            )
        return path

    def register_runner(
        paths: ty.Sequence[str],
        write_images: bool = True,
        fmt: WriterMode = "ome-tiff",
        write_registered: bool = True,
        write_not_registered: bool = True,
        write_merged: bool = True,
        remove_merged: bool = True,
        as_uint8: bool | None = False,
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
            Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
            Parameter("Number of parallel actions", "-n/--n_parallel", n_parallel),
            Parameter("Overwrite", "-W/--overwrite", overwrite),
        )

        with MeasureTimer() as timer:
            if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
                with WorkerPool(n_parallel) as pool:
                    for path in pool.imap(
                        _valis_register,
                        [
                            (
                                path,
                                write_images,
                                fmt,
                                write_registered,
                                write_not_registered,
                                write_merged,
                                remove_merged,
                                as_uint8,
                                overwrite,
                            )
                            for path in paths
                        ],
                    ):
                        logger.info(f"Finished processing {path} in {timer(since_last=True)}")
            else:
                errors = []
                for path in paths:
                    # try:
                    _valis_register(
                        path,
                        write_images,
                        fmt,
                        write_registered,
                        write_not_registered,
                        write_merged,
                        remove_merged,
                        as_uint8,
                        n_parallel=n_parallel,
                        overwrite=overwrite,
                    )
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
                    # except Exception:
                    #     logger.exception(f"Failed to process {path}.")
                    #     errors.append(path)
                    # reraise_exception_if_debug(exc)
                if errors:
                    errors = "\n- ".join(errors)
                    logger.error(f"Failed to register the following projects: {errors}")
        logger.info(f"Finished processing all projects in {timer()}.")

    @overwrite_
    @parallel_mode_
    @n_parallel_
    @as_uint8_
    @remove_merged_
    @write_merged_
    @write_not_registered_
    @write_registered_
    @fmt_
    @project_path_multi_
    @valis.command("export", help_group="Execute")
    def export_cmd(
        project_dir: ty.Sequence[str],
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_merged: bool,
        remove_merged: bool,
        as_uint8: bool | None,
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
            remove_merged=remove_merged,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            parallel_mode=parallel_mode,
            overwrite=overwrite,
        )

    def export_runner(
        paths: ty.Sequence[str],
        fmt: WriterMode = "ome-tiff",
        write_registered: bool = True,
        write_not_registered: bool = True,
        write_merged: bool = True,
        remove_merged: bool = True,
        as_uint8: bool | None = False,
        n_parallel: int = 1,
        parallel_mode: str = "outer",
        overwrite: bool = False,
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
            Parameter("Write images as uint8", "--as_uint8/--no_as_uint8", as_uint8),
            Parameter("Overwrite", "-W/--overwrite", overwrite),
        )

        with MeasureTimer() as timer:
            if n_parallel > 1 and len(paths) > 1 and parallel_mode == "outer":
                with WorkerPool(n_parallel) as pool:
                    for path in pool.imap(
                        _valis_export,
                        [
                            (
                                path,
                                fmt,
                                write_registered,
                                write_not_registered,
                                write_merged,
                                remove_merged,
                                as_uint8,
                                overwrite,
                            )
                            for path in paths
                        ],
                    ):
                        logger.info(f"Finished processing {path} in {timer(since_last=True)}")
            else:
                for path in paths:
                    _valis_export(
                        path,
                        fmt,
                        write_registered,
                        write_not_registered,
                        write_merged,
                        remove_merged,
                        as_uint8,
                        n_parallel=n_parallel,
                        overwrite=overwrite,
                    )
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        logger.info(f"Finished processing all projects in {timer()}.")

    def _valis_export(
        path: PathLike,
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_merged: bool,
        remove_merged: bool,
        as_uint8: bool | None,
        n_parallel: int = 1,
        overwrite: bool = False,
    ) -> PathLike:
        from image2image_reg.workflows import ValisReg

        obj = ValisReg.from_path(path)
        obj.set_logger()
        if not obj.is_registered:
            warning_msg(f"Project {obj.name} is not registered.")
            return path
        obj.write(
            fmt=fmt,
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_merged=write_merged,
            remove_merged=remove_merged,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            overwrite=overwrite,
        )
        return path

    @click.option(
        "-V",
        "--no_valis",
        help="Clear valis.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-M",
        "--no_metadata",
        help="Clear metadata.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option("-I", "--no_image", help="Clear images.", is_flag=True, default=False, show_default=True)
    @click.option("-C", "--no_cache", help="Clear cache.", is_flag=True, default=False, show_default=True)
    @project_path_multi_
    @valis.command("clear", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
    def clear_cmd(
        project_dir: ty.Sequence[str], no_cache: bool, no_image: bool, no_metadata: bool, no_valis: bool
    ) -> None:
        """Clear project data (cache/images/transformations/etc...)."""
        clear_runner(project_dir, no_cache, no_image, no_metadata, no_valis)

    def clear_runner(
        paths: ty.Sequence[str],
        no_cache: bool = True,
        no_image: bool = True,
        no_metadata: bool = True,
        no_valis: bool = True,
    ) -> None:
        """Register images."""
        from image2image_reg.workflows import ValisReg

        print_parameters(
            Parameter("Project directory", "-p/--project_dir", paths),
            Parameter("Don't clear cache", "--no_cache", no_cache),
            Parameter("Don't clear images", "--no_image", no_image),
            Parameter("Don't clear metadata", "--no_metadata", no_metadata),
            Parameter("Don't clear valis", "--no_valis", no_valis),
        )

        with MeasureTimer() as timer:
            for path in paths:
                pro = ValisReg.from_path(path)
                pro.clear(cache=not no_cache, image=not no_image, metadata=not no_metadata, valis=not no_valis)
                logger.info(f"Finished clearing {path} in {timer(since_last=True)}")
        logger.info(f"Finished clearing all projects in {timer()}.")

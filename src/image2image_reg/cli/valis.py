"""Valis registration."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import click
from click_groups import GroupedGroup
from koyo.click import (
    Parameter,
    cli_parse_paths_sort,
    exit_with_error,
    print_parameters,
    warning_msg,
)
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import is_installed
from loguru import logger

from image2image_reg.cli._common import (
    ALLOW_EXTRA_ARGS,
    as_uint8_,
    attach_image_,
    attach_points_,
    attach_shapes_,
    attach_to_,
    fmt_,
    modality_multi_,
    modality_single_,
    n_parallel_,
    output_dir_,
    overwrite_,
    parallel_mode_,
    pixel_size_opt_,
    project_path_multi_,
    project_path_single_,
    remove_merged_,
    rename_,
    write_,
    write_attached_,
    write_attached_images_,
    write_attached_points_,
    write_attached_shapes_,
    write_merged_,
    write_not_registered_,
    write_registered_,
)
from image2image_reg.enums import (
    PreprocessingOptions,
    ValisDetectorMethod,
    ValisMatcherMethod,
    ValisPreprocessingMethod,
    WriterMode,
)


def cli_parse_method(ctx, param, value: list[str]) -> list[str]:
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


def cli_parse_detector(ctx: click.Context, param: str, value: str) -> str:
    """Parse detector."""
    return {"svgg": "sensitive_vgg", "vsvgg": "very_sensitive_vgg"}.get(value, value)


def cli_parse_matcher(ctx: click.Context, param: str, value: str) -> str:
    """Parse matcher."""
    return {"ransac": "RANSAC", "gms": "GMS"}.get(value, value)


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
    @modality_single_
    @output_dir_
    @valis.command("new", help_group="Project")
    def new_cmd(
        output_dir: str,
        name: str,
        cache: bool,
        merge: bool,
        detect: str,
        match: str,
        reflect: bool,
        micro: bool,
        fraction: float,
    ) -> None:
        """Create a new project."""
        from image2image_reg.cli.elastix import new_runner

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
    def about_cmd(project_dir: str) -> None:
        """Print information about the registration project."""
        from image2image_reg.cli.elastix import about_runner

        about_runner(project_dir, valis=True)

    @project_path_multi_
    @valis.command("validate", help_group="Project", aliases=["check"])
    def validate_cmd(project_dir: ty.Sequence[str]) -> None:
        """Validate project configuration."""
        from image2image_reg.cli.elastix import validate_runner

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
        type=click.Choice(PreprocessingOptions, case_sensitive=False),
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
    @modality_multi_
    @project_path_single_
    @valis.command("add-image", help_group="Project", aliases=["ai"])
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
        from image2image_reg.cli.elastix import add_modality_runner

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

    @attach_image_
    @modality_multi_
    @attach_to_
    @project_path_single_
    @valis.command("attach-image", help_group="Project", aliases=["ati"])
    def add_attachment_cmd(project_dir: str, attach_to: str, name: list[str], image: list[str]) -> None:
        """Add attachment image to registered modality."""
        from image2image_reg.cli.elastix import add_attachment_runner

        add_attachment_runner(project_dir, attach_to, name, image, valis=True)

    @pixel_size_opt_
    @attach_points_
    @modality_single_
    @attach_to_
    @project_path_single_
    @valis.command("attach-points", help_group="Project", aliases=["atp"])
    def add_points_cmd(
        project_dir: str, attach_to: str, name: str, file: list[str | Path], pixel_size: float | None
    ) -> None:
        """Add attachment points (csv/tsv/txt) to registered modality."""
        from image2image_reg.cli.elastix import add_points_runner

        add_points_runner(project_dir, attach_to, name, file, pixel_size, valis=True)

    @pixel_size_opt_
    @attach_shapes_
    @modality_single_
    @attach_to_
    @project_path_single_
    @valis.command("attach-shape", help_group="Project", aliases=["ats"])
    def add_shape_cmd(
        project_dir: str, attach_to: str, name: str, file: list[str | Path], pixel_size: float | None
    ) -> None:
        """Add attachment shape (GeoJSON) to registered modality."""
        from image2image_reg.cli.elastix import add_shape_runner

        add_shape_runner(project_dir, attach_to, name, file, pixel_size, valis=True)

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
    @valis.command("add-merge", help_group="Project", aliases=["am"])
    def add_merge_cmd(project_dir: ty.Sequence[str], name: str, modality: ty.Iterable[str] | None, auto: bool) -> None:
        """Specify how (if) images should be merged."""
        from image2image_reg.cli.elastix import add_merge_runner

        add_merge_runner(project_dir, name, modality, auto, valis=True)

    @overwrite_
    @parallel_mode_
    @n_parallel_
    @as_uint8_
    @rename_
    @remove_merged_
    @write_merged_
    @write_attached_
    @write_not_registered_
    @write_registered_
    @fmt_
    @write_
    @project_path_multi_
    @valis.command("register", help_group="Execute", aliases=["run"])
    def register_cmd(
        project_dir: ty.Sequence[str],
        write: bool,
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_attached: bool,
        write_merged: bool,
        remove_merged: bool,
        rename: bool,
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
            write_not_registered=write_not_registered,
            write_attached=write_attached,
            write_merged=write_merged,
            remove_merged=remove_merged,
            rename=rename,
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
        write_attached: bool,
        write_merged: bool,
        remove_merged: bool,
        rename: bool,
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
                write_attached=write_attached,
                write_merged=write_merged,
                remove_merged=remove_merged,
                rename=rename,
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
        write_attached: bool = True,
        write_merged: bool = True,
        remove_merged: bool = True,
        rename: bool = False,
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

        errors: list[str] = []
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
                                write_attached,
                                write_merged,
                                remove_merged,
                                rename,
                                as_uint8,
                                overwrite,
                            )
                            for path in paths
                        ],
                    ):
                        logger.info(f"Finished processing {path} in {timer(since_last=True)}")
            else:
                for path in paths:
                    # try:
                    _valis_register(
                        path,
                        write_images,
                        fmt,
                        write_registered,
                        write_not_registered,
                        write_attached,
                        write_merged,
                        remove_merged,
                        rename,
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
                    errors_str = "\n- ".join(errors)
                    logger.error(f"Failed to register the following projects: {errors_str}")
        logger.info(f"Finished registering all projects in {timer()}.")
        if errors:
            return exit_with_error()

    @overwrite_
    @parallel_mode_
    @n_parallel_
    @as_uint8_
    @rename_
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
    @valis.command("export", help_group="Execute", aliases=["write"])
    def export_cmd(
        project_dir: ty.Sequence[str],
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_attached: bool,
        write_attached_images: bool | None,
        write_attached_shapes: bool | None,
        write_attached_points: bool | None,
        write_merged: bool,
        remove_merged: bool,
        rename: bool,
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
            write_attached=write_attached,
            write_attached_points=write_attached_points,
            write_attached_shapes=write_attached_shapes,
            write_attached_images=write_attached_images,
            write_merged=write_merged,
            remove_merged=remove_merged,
            rename=rename,
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
        write_attached: bool = True,
        write_attached_images: bool | None = None,
        write_attached_shapes: bool | None = None,
        write_attached_points: bool | None = None,
        write_merged: bool = True,
        remove_merged: bool = True,
        rename: bool = True,
        as_uint8: bool | None = False,
        n_parallel: int = 1,
        parallel_mode: str = "outer",
        overwrite: bool = False,
    ) -> None:
        """Register images."""
        import os

        from mpire import WorkerPool

        # limit Vips concurrency to avoid memory issues
        os.environ["VIPS_CONCURRENCY"] = "6"

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
            Parameter("Write attached modalities", "--write_attached/--no_write_attached", write_attached),
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
                                write_attached_images,
                                write_attached_shapes,
                                write_attached_points,
                                write_merged,
                                remove_merged,
                                rename,
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
                        write_attached_images,
                        write_attached_shapes,
                        write_attached_points,
                        write_merged,
                        remove_merged,
                        rename,
                        as_uint8,
                        n_parallel=n_parallel,
                        overwrite=overwrite,
                    )
                    logger.info(f"Finished processing {path} in {timer(since_last=True)}")
        logger.info(f"Finished exporting all projects in {timer()}.")

    def _valis_export(
        path: PathLike,
        fmt: WriterMode,
        write_registered: bool,
        write_not_registered: bool,
        write_attached_images: bool | None,
        write_attached_shapes: bool | None,
        write_attached_points: bool | None,
        write_merged: bool,
        remove_merged: bool,
        rename: bool,
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
        if not obj.is_registered:
            warning_msg(f"Project {obj.name} is not registered.")
            return path
        obj.write(
            fmt=fmt,
            write_registered=write_registered,
            write_not_registered=write_not_registered,
            write_attached_images=write_attached_images,
            write_attached_shapes=write_attached_shapes,
            write_attached_points=write_attached_points,
            write_merged=write_merged,
            remove_merged=remove_merged,
            rename=rename,
            as_uint8=as_uint8,
            n_parallel=n_parallel,
            overwrite=overwrite,
        )
        return path

    @click.option(
        "--all",
        "all_",
        help="Clear all results.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-v/-V",
        "--valis/--no_valis",
        "valis_",
        help="Clear valis.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option(
        "-m/-M",
        "--metadata/--no_metadata",
        help="Clear metadata.",
        is_flag=True,
        default=False,
        show_default=True,
    )
    @click.option("-i/-I", "--image/--no_image", help="Clear images.", is_flag=True, default=False, show_default=True)
    @click.option("-c/-C", "--cache/--no_cache", help="Clear cache.", is_flag=True, default=False, show_default=True)
    @project_path_multi_
    @valis.command("clear", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
    def clear_cmd(
        project_dir: ty.Sequence[str], cache: bool, image: bool, metadata: bool, valis_: bool, all_: bool
    ) -> None:
        """Clear project data (cache/images/transformations/etc...)."""
        clear_runner(project_dir, cache, image, metadata, valis_, all_)

    def clear_runner(
        paths: ty.Sequence[str],
        cache: bool = True,
        image: bool = True,
        metadata: bool = True,
        valis_: bool = True,
        all_: bool = False,
    ) -> None:
        """Register images."""
        from image2image_reg.workflows import ValisReg

        if all_:
            cache = image = metadata = valis_ = True

        print_parameters(
            Parameter("Project directory", "-p/--project_dir", paths),
            Parameter("Don't clear cache", "--cache", cache),
            Parameter("Don't clear images", "--image", image),
            Parameter("Don't clear metadata", "--metadata", metadata),
            Parameter("Don't clear valis", "--no_valis", valis_),
        )

        with MeasureTimer() as timer:
            for path in paths:
                pro = ValisReg.from_path(path)
                pro.clear(cache=cache, image=image, metadata=metadata, valis=valis_)
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
        "-i",
        "--source_dir",
        help="Source directory where images/files should be searched for.",
        type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
        show_default=True,
        required=True,
        multiple=True,
    )
    @project_path_multi_
    @valis.command("update", help_group="Execute", context_settings=ALLOW_EXTRA_ARGS)
    def update_cmd(project_dir: list[str], source_dir: list[str], recursive: bool) -> None:
        """Update project paths (e.g after folder move)."""
        from image2image_reg.cli.elastix import update_runner

        update_runner(project_dir, source_dir, recursive, valis=True)
else:
    valis = None

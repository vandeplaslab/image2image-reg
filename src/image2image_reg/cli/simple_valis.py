"""Simple valis."""

from __future__ import annotations

import click
from click_groups import GroupedGroup
from koyo.click import (
    Parameter,
    cli_parse_paths_sort,
    print_parameters,
)
from koyo.typing import PathLike
from koyo.utilities import is_installed

simple_valis = None
if is_installed("valis"):

    @click.group("simple-valis", cls=GroupedGroup)
    def simple_valis() -> None:
        """Valis registration."""

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
    @simple_valis.command("init", help_group="Project")
    def valis_quick_init(
        name: str,
        output_dir: PathLike,
        image: list[PathLike],
        reference: PathLike | None,
        check_for_reflection: bool,
        no_non_rigid_reg: bool,
        no_micro_reg: bool,
    ) -> None:
        """Initialize Valis configuration file."""
        valis_quick_init_runner(
            name, output_dir, image, reference, check_for_reflection, no_non_rigid_reg, no_micro_reg
        )

    def valis_quick_init_runner(
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
            check_for_reflections=check_for_reflection,
            non_rigid_reg=not no_non_rigid_reg,
            micro_reg=not no_micro_reg,
        )

    @click.option(
        "-o",
        "--output_dir",
        help="Path to the WsiReg project directory. It usually ends in .valis extension.",
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
    @simple_valis.command("register", help_group="Project")
    def valis_quick_register(config: PathLike, output_dir: PathLike):
        """Register images using the Valis algorithm."""
        valis_quick_register_runner(output_dir, config)

    def valis_quick_register_runner(output_dir: PathLike, config: PathLike | None):
        """Register list of images using Valis algorithm."""
        from image2image_reg.workflows.valis import valis_registration_from_config

        print_parameters(
            Parameter("Output directory", "-o/--output_dir", output_dir),
            Parameter("Config", "-c/--config", config),
        )

        valis_registration_from_config(output_dir=output_dir, config=config)

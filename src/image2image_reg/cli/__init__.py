"""Init."""

from __future__ import annotations

import sys
from multiprocessing import freeze_support, set_start_method

import click
import koyo.compat
from click_groups import GroupedGroup
from image2image_io.cli.convert import convert
from koyo.system import IS_MAC
from koyo.typing import PathLike
from koyo.utilities import running_as_pyinstaller_app
from loguru import logger

from image2image_reg import __version__
from image2image_reg.cli._common import set_logger
from image2image_reg.cli.elastix import elastix
from image2image_reg.cli.merge import merge
from image2image_reg.cli.simple_valis import simple_valis
from image2image_reg.cli.valis import valis


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    cls=GroupedGroup,
)
@click.version_option(__version__, prog_name="i2reg")
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
    extra_args: tuple[str, ...] | None = None,
) -> None:
    """Launch registration app."""
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
    else:
        uninstall_debugger_hook()
    set_logger(verbosity, no_color, log)
    logger.trace(f"Executed command: {sys.argv}")
    if dev:
        logger.debug("Debugger hook installed.")


cli.add_command(elastix, help_group="Registration")
if valis:
    cli.add_command(valis, help_group="Registration")
if simple_valis:
    cli.add_command(simple_valis, help_group="Registration")
cli.add_command(convert, help_group="Utility")
cli.add_command(merge, help_group="Utility")


def main() -> None:
    """Execute the "i2reg" command line program."""
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli.main(windows_expand_args=False)


if __name__ == "__main__":
    main()

"""Utilities."""

import click


@click.command("convert")
def convert() -> None:
    """Convert images to pyramidal OME-TIFF."""

"""Whole slide image registration using elastix."""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

try:
    __version__ = version("image2image-reg")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

logger.disable("image2image_reg")

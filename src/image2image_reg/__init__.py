"""Whole slide image registration using elastix."""

from importlib.metadata import PackageNotFoundError, version

from loguru import logger

__version__ = "0.1.12"
__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

logger.disable("image2image_reg")

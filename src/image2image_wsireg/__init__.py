"""Whole slide image registration using elastix."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("image2image-wsireg")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Lukasz G. Migas"
__email__ = "lukas.migas@yahoo.com"

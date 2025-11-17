"""Utilities."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path
from tempfile import tempdir

from koyo.typing import PathLike

TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "tests" / "_test_data"


# noinspection PyMissingOrEmptyDocstring
def get_test_files(pattern: str = "*.json") -> list[Path]:
    """Get config files."""
    return list(TEST_DATA_DIR.glob(pattern))


def get_test_file(filename: str) -> Path:
    """Get config file."""
    path = TEST_DATA_DIR / filename
    assert path.exists(), f"File {path} does not exist."
    return path


def make_test_polygon(output_dir: Path) -> None:
    """Create test data."""
    import numpy as np
    import tifffile
    from skimage.draw import polygon2mask

    im = polygon2mask(
        (2048, 2048),
        polygon=np.array(
            (
                (300, 300),
                (980, 320),
                (380, 730),
                (220, 590),
                (300, 300),
            )
        ),
    )
    im = im.astype(np.uint8)
    im[im == 1] = 255
    tifffile.imwrite(output_dir / "moving.tiff", im, compression="deflate")

    im = polygon2mask(
        (2048, 2048),
        polygon=np.array(
            (
                (300, 300),
                (980, 320),
                (380, 730),
                (220, 590),
                (300, 300),
            )
        )
        + 900,
    )
    im = im.astype(np.uint8)
    im[im == 1] = 255
    tifffile.imwrite(output_dir / "fixed.tiff", im, compression="deflate")


class CleanupMixin:
    """Mixin class."""

    _temp_temp = None
    _base_temp = None

    @property
    def temp_dir(self) -> Path:
        """Temporary directory."""
        if self._temp_temp is None:
            self._temp_temp = Path(tempdir) / "temp"
            self._temp_temp.mkdir()
        return self._temp_temp

    def setup_method(self, method: ty.Any) -> None:
        """Setup class."""
        if self._base_temp is None:
            self._base_temp = Path(tempdir) / "autoims"
            self._base_temp.mkdir(exist_ok=True)

    def teardown_method(self, method: ty.Any) -> None:
        """Teardown class."""
        if self._base_temp:
            cleanup_directory(self._base_temp)
        if self._temp_temp:
            cleanup_directory(self._temp_temp)


def cleanup_directory(path: PathLike) -> None:
    """Clean directory."""
    import glob
    import os
    import shutil

    try:
        shutil.rmtree(path)
    except Exception:
        for f in glob.glob(os.path.join(path, "*"), recursive=True):
            if os.path.isdir(f):
                cleanup_directory(f)
            else:
                with suppress(PermissionError):
                    os.remove(f)

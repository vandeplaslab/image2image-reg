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

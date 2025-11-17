"""Directories where to store database and log files."""

from pathlib import Path

from koyo.utilities import is_installed

if is_installed("platformdirs"):
    from platformdirs import user_data_dir

    USER_DATA_DIR = Path(user_data_dir("image2image-reg"))
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    USER_CONFIG_DIR = USER_DATA_DIR / "Configs"
    USER_CONFIG_DIR.mkdir(exist_ok=True, parents=True)

    USER_LOG_DIR = USER_DATA_DIR / "Logs"
    USER_LOG_DIR.mkdir(exist_ok=True, parents=True)
else:
    USER_DATA_DIR = USER_CONFIG_DIR = USER_LOG_DIR = None

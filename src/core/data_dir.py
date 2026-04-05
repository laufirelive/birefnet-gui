# src/core/data_dir.py
"""Centralised data-directory resolution with a priority-based config chain."""

import json
import os
from typing import Optional

from src.core.paths import get_app_root

_DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".birefnet-gui")


def _get_app_root_config_path() -> str:
    """Return the path to config.json next to the application executable."""
    return os.path.join(get_app_root(), "config.json")


def _get_user_config_path() -> str:
    """Return the path to the per-user config.json."""
    return os.path.join(os.path.expanduser("~"), ".birefnet-gui", "config.json")


def _read_data_dir_from(config_path: str) -> Optional[str]:
    """Read the ``data_dir`` field from a JSON config file.

    Returns *None* on any error (missing file, bad JSON, missing key, etc.).
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        value = data.get("data_dir")
        if isinstance(value, str) and value:
            return value
    except Exception:
        pass
    return None


def resolve_data_dir() -> str:
    """Resolve the data directory using the priority chain.

    1. ``{app_root}/config.json``  (portable / frozen mode)
    2. ``~/.birefnet-gui/config.json``  (user override)
    3. Default: ``~/.birefnet-gui/``
    """
    for config_path in (_get_app_root_config_path(), _get_user_config_path()):
        result = _read_data_dir_from(config_path)
        if result is not None:
            return result
    return _DEFAULT_DATA_DIR


def save_config(data_dir: str, config_path: Optional[str] = None) -> None:
    """Persist a ``config.json`` with the given *data_dir*.

    If *config_path* is not provided it defaults to
    ``{data_dir}/config.json``.
    """
    if config_path is None:
        config_path = os.path.join(data_dir, "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump({"data_dir": data_dir}, fh, indent=2)


def get_cache_dir() -> str:
    """Return ``{data_dir}/cache/``."""
    return os.path.join(resolve_data_dir(), "cache")


def get_brm_path() -> str:
    """Return ``{data_dir}/queue.brm``."""
    return os.path.join(resolve_data_dir(), "queue.brm")


def get_settings_path() -> str:
    """Return ``{data_dir}/settings.json``."""
    return os.path.join(resolve_data_dir(), "settings.json")

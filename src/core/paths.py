# src/core/paths.py
"""Unified path resolution for both development and PyInstaller-frozen modes."""

import os
import sys


def is_frozen() -> bool:
    """Return True if running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_app_root() -> str:
    """Return the application root directory.

    Frozen (PyInstaller --onedir): directory containing the executable.
    Development: project root (two levels up from this file).
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_models_dir() -> str:
    """Return the path to the models directory (<app_root>/models/)."""
    return os.path.join(get_app_root(), "models")

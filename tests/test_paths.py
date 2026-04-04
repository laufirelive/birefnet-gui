# tests/test_paths.py
import os
import sys
from unittest.mock import patch

from src.core.paths import get_app_root, get_models_dir, is_frozen


class TestGetAppRoot:
    def test_dev_mode_returns_project_root(self):
        """In development mode (not frozen), returns the project root directory."""
        root = get_app_root()
        # paths.py is at src/core/paths.py, so root should contain main.py
        assert os.path.isfile(os.path.join(root, "main.py"))

    def test_frozen_mode_returns_executable_dir(self, tmp_path):
        """When frozen (PyInstaller), returns the directory containing the executable."""
        fake_exe = str(tmp_path / "BiRefNet-GUI.exe")
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", fake_exe):
                root = get_app_root()
                assert root == str(tmp_path)


class TestGetModelsDir:
    def test_returns_models_subdir(self):
        """models dir is <app_root>/models."""
        models = get_models_dir()
        assert models.endswith("models")
        assert os.path.dirname(models) == get_app_root()


class TestIsFrozen:
    def test_not_frozen_in_dev(self):
        assert is_frozen() is False

    def test_frozen_when_attr_set(self):
        with patch.object(sys, "frozen", True, create=True):
            assert is_frozen() is True

import os
from unittest.mock import patch, MagicMock

import pytest

from src.core.model_downloader import ModelDownloader


class TestModelDownloader:
    def test_get_installed_models(self, tmp_path):
        (tmp_path / "birefnet-general").mkdir()
        (tmp_path / "birefnet-lite").mkdir()
        downloader = ModelDownloader(str(tmp_path))
        installed = downloader.get_installed_models()
        assert "general" in installed
        assert "lite" in installed
        assert "hr" not in installed

    def test_is_installed(self, tmp_path):
        (tmp_path / "birefnet-general").mkdir()
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("general") is True
        assert downloader.is_installed("lite") is False

    def test_is_installed_unknown_key_returns_false(self, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("nonexistent") is False

    def test_delete_model_removes_directory(self, tmp_path):
        model_dir = tmp_path / "birefnet-lite"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("lite") is True
        downloader.delete_model("lite")
        assert downloader.is_installed("lite") is False
        assert not model_dir.exists()

    def test_delete_nonexistent_model_raises(self, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            downloader.delete_model("lite")

    @patch("src.core.model_downloader.snapshot_download")
    def test_download_calls_snapshot_download(self, mock_download, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        downloader.download_model("general")
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert "zhengpeng7/BiRefNet" in str(call_kwargs)

    @patch("src.core.model_downloader.snapshot_download")
    def test_download_tries_mirror_first(self, mock_download, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        downloader.download_model("lite")
        assert mock_download.called

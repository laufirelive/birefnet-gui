import os
from unittest.mock import patch, MagicMock

import pytest

from src.core.model_downloader import ModelDownloader


class TestModelDownloader:
    def test_get_installed_models(self, tmp_path):
        general_dir = tmp_path / "birefnet-general"
        general_dir.mkdir()
        (general_dir / "config.json").write_text("{}")
        lite_dir = tmp_path / "birefnet-lite"
        lite_dir.mkdir()
        (lite_dir / "config.json").write_text("{}")
        downloader = ModelDownloader(str(tmp_path))
        installed = downloader.get_installed_models()
        assert "general" in installed
        assert "lite" in installed
        assert "hr" not in installed

    def test_is_installed(self, tmp_path):
        general_dir = tmp_path / "birefnet-general"
        general_dir.mkdir()
        (general_dir / "config.json").write_text("{}")
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


class TestIsPartial:
    def test_nonexistent_dir_is_not_partial(self, tmp_path):
        dl = ModelDownloader(str(tmp_path / "models"))
        assert dl.is_partial("general") is False

    def test_empty_dir_is_partial(self, tmp_path):
        models_dir = tmp_path / "models"
        (models_dir / "birefnet-general").mkdir(parents=True)
        dl = ModelDownloader(str(models_dir))
        assert dl.is_partial("general") is True

    def test_complete_dir_is_not_partial(self, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "birefnet-general"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        dl = ModelDownloader(str(models_dir))
        assert dl.is_partial("general") is False

    def test_is_installed_requires_config_json(self, tmp_path):
        models_dir = tmp_path / "models"
        (models_dir / "birefnet-general").mkdir(parents=True)
        dl = ModelDownloader(str(models_dir))
        assert dl.is_installed("general") is False


class TestDownloadEndpoint:
    def test_download_passes_endpoint(self, tmp_path):
        from unittest.mock import patch
        dl = ModelDownloader(str(tmp_path / "models"))
        with patch.object(dl, "_do_download", return_value=str(tmp_path)) as mock_dl:
            dl.download_model("general", endpoint="https://my.mirror.com")
            mock_dl.assert_called_once()
            args = mock_dl.call_args
            # endpoint should be the 3rd positional arg
            assert args[0][2] == "https://my.mirror.com"

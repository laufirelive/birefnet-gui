import json
import os
import pytest

class TestResolveDataDir:
    def test_default_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(tmp_path / "nonexistent" / "config.json"))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path",
                            lambda: str(tmp_path / "also_nonexistent" / "config.json"))
        from src.core.data_dir import resolve_data_dir
        result = resolve_data_dir()
        expected = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        assert result == expected

    def test_app_root_config_takes_priority(self, tmp_path, monkeypatch):
        app_config = tmp_path / "app" / "config.json"
        app_config.parent.mkdir()
        app_config.write_text(json.dumps({"data_dir": "/app/data"}))
        user_config = tmp_path / "user" / "config.json"
        user_config.parent.mkdir()
        user_config.write_text(json.dumps({"data_dir": "/user/data"}))
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path", lambda: str(app_config))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path", lambda: str(user_config))
        from src.core.data_dir import resolve_data_dir
        assert resolve_data_dir() == "/app/data"

    def test_user_config_used_when_no_app_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(tmp_path / "nope" / "config.json"))
        user_config = tmp_path / "user" / "config.json"
        user_config.parent.mkdir()
        user_config.write_text(json.dumps({"data_dir": "/custom/path"}))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path", lambda: str(user_config))
        from src.core.data_dir import resolve_data_dir
        assert resolve_data_dir() == "/custom/path"

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        app_config = tmp_path / "app" / "config.json"
        app_config.parent.mkdir()
        app_config.write_text("NOT JSON")
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path", lambda: str(app_config))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path",
                            lambda: str(tmp_path / "nope" / "config.json"))
        from src.core.data_dir import resolve_data_dir
        expected = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        assert resolve_data_dir() == expected

class TestSaveConfig:
    def test_save_creates_config_json(self, tmp_path):
        from src.core.data_dir import save_config
        save_config(str(tmp_path / "mydata"), config_path=str(tmp_path / "config.json"))
        with open(tmp_path / "config.json") as f:
            data = json.load(f)
        assert data == {"data_dir": str(tmp_path / "mydata")}

    def test_save_creates_parent_dirs(self, tmp_path):
        from src.core.data_dir import save_config
        path = str(tmp_path / "deep" / "nested" / "config.json")
        save_config("/some/dir", config_path=path)
        assert os.path.exists(path)

class TestPathHelpers:
    def test_get_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.data_dir.resolve_data_dir", lambda: str(tmp_path))
        from src.core.data_dir import get_cache_dir
        assert get_cache_dir() == str(tmp_path / "cache")

    def test_get_brm_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.data_dir.resolve_data_dir", lambda: str(tmp_path))
        from src.core.data_dir import get_brm_path
        assert get_brm_path() == str(tmp_path / "queue.brm")

    def test_get_settings_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.core.data_dir.resolve_data_dir", lambda: str(tmp_path))
        from src.core.data_dir import get_settings_path
        assert get_settings_path() == str(tmp_path / "settings.json")

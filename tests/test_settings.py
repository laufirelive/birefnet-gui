import json
import os

from src.core.settings import AppSettings, load_settings, save_settings


class TestAppSettings:
    def test_default_values(self):
        s = AppSettings()
        assert s.download_source == "hf-mirror"
        assert s.custom_endpoint == ""

    def test_to_dict(self):
        s = AppSettings(download_source="custom", custom_endpoint="https://my.mirror/")
        d = s.to_dict()
        assert d["download_source"] == "custom"
        assert d["custom_endpoint"] == "https://my.mirror/"

    def test_from_dict(self):
        s = AppSettings.from_dict({"download_source": "huggingface", "custom_endpoint": ""})
        assert s.download_source == "huggingface"

    def test_from_dict_unknown_keys_ignored(self):
        s = AppSettings.from_dict({"download_source": "hf-mirror", "unknown_key": 42})
        assert s.download_source == "hf-mirror"

    def test_from_dict_missing_keys_use_defaults(self):
        s = AppSettings.from_dict({})
        assert s.download_source == "hf-mirror"
        assert s.custom_endpoint == ""


class TestSaveLoadSettings:
    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "settings.json")
        original = AppSettings(download_source="custom", custom_endpoint="https://x.com/")
        save_settings(original, path)
        loaded = load_settings(path)
        assert loaded.download_source == "custom"
        assert loaded.custom_endpoint == "https://x.com/"

    def test_load_nonexistent_returns_defaults(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        loaded = load_settings(path)
        assert loaded.download_source == "hf-mirror"

    def test_load_corrupt_returns_defaults(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("NOT JSON")
        loaded = load_settings(path)
        assert loaded.download_source == "hf-mirror"

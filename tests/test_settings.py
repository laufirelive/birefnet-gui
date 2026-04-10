import json
import os

from src.core.settings import AppSettings, load_settings, save_settings


class TestAppSettings:
    def test_default_values(self):
        s = AppSettings()
        assert s.download_source == "hf-mirror"
        assert s.custom_endpoint == ""
        assert s.panel_defaults == {}

    def test_to_dict(self):
        s = AppSettings(
            download_source="custom",
            custom_endpoint="https://my.mirror/",
            panel_defaults={"model_name": "BiRefNet-lite"},
        )
        d = s.to_dict()
        assert d["download_source"] == "custom"
        assert d["custom_endpoint"] == "https://my.mirror/"
        assert d["panel_defaults"]["model_name"] == "BiRefNet-lite"

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
        assert s.panel_defaults == {}

    def test_from_dict_non_dict_panel_defaults_falls_back(self):
        s = AppSettings.from_dict({"panel_defaults": "bad"})
        assert s.panel_defaults == {}


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

    def test_roundtrip_panel_defaults(self, tmp_path):
        path = str(tmp_path / "settings.json")
        original = AppSettings(
            panel_defaults={
                "model_name": "BiRefNet-lite",
                "output_format": "mp4_h264",
                "background_mode": "green",
                "bitrate_mode": "custom",
                "custom_bitrate_mbps": 10.0,
                "encoding_preset": "fast",
                "batch_size": 4,
                "inference_resolution": 512,
                "temporal_fix": False,
                "encoder_type": "software",
            }
        )
        save_settings(original, path)
        loaded = load_settings(path)
        assert loaded.panel_defaults["output_format"] == "mp4_h264"

    def test_load_invalid_panel_defaults_keeps_app_usable(self, tmp_path):
        path = tmp_path / "bad2.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"panel_defaults": {"batch_size": "oops"}}, f)
        loaded = load_settings(str(path))
        assert isinstance(loaded.panel_defaults, dict)

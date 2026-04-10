"""Application settings: load/save from settings.json."""

import json
import os
from dataclasses import dataclass


@dataclass
class AppSettings:
    download_source: str = "hf-mirror"  # "hf-mirror", "huggingface", "custom"
    custom_endpoint: str = ""
    panel_defaults: dict | None = None

    def __post_init__(self):
        if not isinstance(self.panel_defaults, dict):
            self.panel_defaults = {}

    def to_dict(self) -> dict:
        return {
            "download_source": self.download_source,
            "custom_endpoint": self.custom_endpoint,
            "panel_defaults": self.panel_defaults,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        panel_defaults = data.get("panel_defaults", {})
        if not isinstance(panel_defaults, dict):
            panel_defaults = {}
        return cls(
            download_source=data.get("download_source", "hf-mirror"),
            custom_endpoint=data.get("custom_endpoint", ""),
            panel_defaults=panel_defaults,
        )


def load_settings(path: str) -> AppSettings:
    """Load settings from JSON file. Returns defaults on any error."""
    try:
        with open(path) as f:
            return AppSettings.from_dict(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return AppSettings()


def save_settings(settings: AppSettings, path: str) -> None:
    """Save settings to JSON file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(settings.to_dict(), f, indent=2)

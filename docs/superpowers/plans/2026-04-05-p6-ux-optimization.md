# P6 UX Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable data directory, cache management GUI, system notifications, and model download improvements (custom endpoint, real progress, retry).

**Architecture:** New `data_dir` module centralizes all data path resolution with a priority chain (app_root config → user config → default). A new "设置" Tab hosts cache management and download settings. `QSystemTrayIcon`-based notifier fires on task/queue completion. Model download gets real tqdm-based progress and retry-on-failure.

**Tech Stack:** PyQt6 (QSystemTrayIcon, QDesktopServices), huggingface_hub (snapshot_download tqdm_class/endpoint params), existing MaskCacheManager.

---

### Task 1: Data Directory Resolution Module

**Files:**
- Create: `src/core/data_dir.py`
- Create: `tests/test_data_dir.py`

- [ ] **Step 1: Write failing tests for data_dir resolution**

```python
# tests/test_data_dir.py
import json
import os

import pytest


class TestResolveDataDir:
    def test_default_when_no_config(self, tmp_path, monkeypatch):
        """No config.json anywhere → falls back to ~/.birefnet-gui/."""
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(tmp_path / "nonexistent" / "config.json"))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path",
                            lambda: str(tmp_path / "also_nonexistent" / "config.json"))

        from src.core.data_dir import resolve_data_dir
        result = resolve_data_dir()
        expected = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        assert result == expected

    def test_app_root_config_takes_priority(self, tmp_path, monkeypatch):
        """App root config.json overrides user dir config.json."""
        app_config = tmp_path / "app" / "config.json"
        app_config.parent.mkdir()
        app_config.write_text(json.dumps({"data_dir": "/app/data"}))

        user_config = tmp_path / "user" / "config.json"
        user_config.parent.mkdir()
        user_config.write_text(json.dumps({"data_dir": "/user/data"}))

        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(app_config))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path",
                            lambda: str(user_config))

        from src.core.data_dir import resolve_data_dir
        assert resolve_data_dir() == "/app/data"

    def test_user_config_used_when_no_app_config(self, tmp_path, monkeypatch):
        """Only user dir config.json exists → use it."""
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(tmp_path / "nope" / "config.json"))

        user_config = tmp_path / "user" / "config.json"
        user_config.parent.mkdir()
        user_config.write_text(json.dumps({"data_dir": "/custom/path"}))
        monkeypatch.setattr("src.core.data_dir._get_user_config_path",
                            lambda: str(user_config))

        from src.core.data_dir import resolve_data_dir
        assert resolve_data_dir() == "/custom/path"

    def test_malformed_config_falls_back_to_default(self, tmp_path, monkeypatch):
        """Corrupt config.json → ignore it, use default."""
        app_config = tmp_path / "app" / "config.json"
        app_config.parent.mkdir()
        app_config.write_text("NOT JSON")
        monkeypatch.setattr("src.core.data_dir._get_app_root_config_path",
                            lambda: str(app_config))
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data_dir.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.data_dir'`

- [ ] **Step 3: Implement data_dir module**

```python
# src/core/data_dir.py
"""Centralized data directory resolution.

Priority chain:
  1. {app_root}/config.json  (portable mode)
  2. ~/.birefnet-gui/config.json  (user override)
  3. Default: ~/.birefnet-gui/
"""

import json
import os

from src.core.paths import get_app_root

_DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".birefnet-gui")


def _get_app_root_config_path() -> str:
    return os.path.join(get_app_root(), "config.json")


def _get_user_config_path() -> str:
    return os.path.join(_DEFAULT_DATA_DIR, "config.json")


def _read_data_dir_from(config_path: str) -> str | None:
    """Read data_dir from a config.json file. Returns None on any error."""
    try:
        with open(config_path) as f:
            data = json.load(f)
        if isinstance(data, dict) and "data_dir" in data:
            return data["data_dir"]
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return None


def resolve_data_dir() -> str:
    """Return the active data directory path."""
    # 1. App root config (portable mode)
    result = _read_data_dir_from(_get_app_root_config_path())
    if result is not None:
        return result
    # 2. User config
    result = _read_data_dir_from(_get_user_config_path())
    if result is not None:
        return result
    # 3. Default
    return _DEFAULT_DATA_DIR


def save_config(data_dir: str, config_path: str | None = None) -> None:
    """Write a config.json with the given data_dir."""
    if config_path is None:
        config_path = os.path.join(data_dir, "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({"data_dir": data_dir}, f, indent=2)


def get_cache_dir() -> str:
    return os.path.join(resolve_data_dir(), "cache")


def get_brm_path() -> str:
    return os.path.join(resolve_data_dir(), "queue.brm")


def get_settings_path() -> str:
    return os.path.join(resolve_data_dir(), "settings.json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data_dir.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/data_dir.py tests/test_data_dir.py
git commit -m "feat: add data_dir module for configurable data directory"
```

---

### Task 2: Wire data_dir Into Existing Code

**Files:**
- Modify: `src/worker/matting_worker.py` (line 11 — replace `CACHE_DIR` constant)
- Modify: `src/gui/main_window.py` (line 42 — replace `BRM_PATH` constant)
- Modify: `src/gui/queue_tab.py` (line 29 — import from `data_dir` instead of `matting_worker`)

- [ ] **Step 1: Update matting_worker.py to use data_dir**

Replace lines 1-11 of `src/worker/matting_worker.py`:

```python
# Old:
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "cache")

# New:
from src.core.data_dir import get_cache_dir

CACHE_DIR = get_cache_dir()
```

The full import section becomes:
```python
import os
import threading
import time
import uuid

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.cache import MaskCacheManager
from src.core.config import InputType, ProcessingConfig
from src.core.data_dir import get_cache_dir

CACHE_DIR = get_cache_dir()
```

- [ ] **Step 2: Update main_window.py to use data_dir**

Replace line 42 of `src/gui/main_window.py`:

```python
# Old:
BRM_PATH = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "queue.brm")

# New:
from src.core.data_dir import get_brm_path
BRM_PATH = get_brm_path()
```

Add the import near the top imports and remove the line.

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All 165 tests PASS (data_dir tests also pass since they were added in Task 1)

- [ ] **Step 4: Commit**

```bash
git add src/worker/matting_worker.py src/gui/main_window.py
git commit -m "refactor: use data_dir module instead of hardcoded paths"
```

---

### Task 3: Cache Size Calculation

**Files:**
- Modify: `src/core/cache.py` (add `get_total_size` method + `format_size` helper)
- Create: `tests/test_cache_size.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_cache_size.py
import os

from src.core.cache import MaskCacheManager, format_size


class TestGetTotalSize:
    def test_empty_cache_returns_zero(self, tmp_path):
        cache = MaskCacheManager(str(tmp_path / "cache"))
        assert cache.get_total_size() == 0

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        cache = MaskCacheManager(str(tmp_path / "nonexistent"))
        assert cache.get_total_size() == 0

    def test_counts_bytes_correctly(self, tmp_path):
        cache_dir = tmp_path / "cache"
        task_dir = cache_dir / "task1" / "masks"
        task_dir.mkdir(parents=True)
        # Write two 100-byte files
        (task_dir / "000000.png").write_bytes(b"x" * 100)
        (task_dir / "000001.png").write_bytes(b"x" * 200)
        # Also a metadata file
        (cache_dir / "task1" / "metadata.json").write_text('{"test": true}')

        cache = MaskCacheManager(str(cache_dir))
        total = cache.get_total_size()
        # 100 + 200 + len('{"test": true}') = 314
        assert total == 314

    def test_cleanup_all_resets_to_zero(self, tmp_path):
        cache_dir = tmp_path / "cache"
        task_dir = cache_dir / "task1" / "masks"
        task_dir.mkdir(parents=True)
        (task_dir / "000000.png").write_bytes(b"x" * 500)

        cache = MaskCacheManager(str(cache_dir))
        assert cache.get_total_size() > 0
        cache.cleanup_all()
        assert cache.get_total_size() == 0


class TestFormatSize:
    def test_zero(self):
        assert format_size(0) == "0 MB"

    def test_small_bytes(self):
        assert format_size(500_000) == "0 MB"

    def test_megabytes(self):
        assert format_size(150_000_000) == "143 MB"

    def test_gigabytes(self):
        assert format_size(2_500_000_000) == "2.3 GB"

    def test_exactly_one_gb(self):
        assert format_size(1_073_741_824) == "1.0 GB"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache_size.py -v`
Expected: FAIL — `ImportError: cannot import name 'format_size'`

- [ ] **Step 3: Implement get_total_size and format_size**

Add to `src/core/cache.py`:

```python
def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes >= 1_073_741_824:  # 1 GB
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    mb = size_bytes // 1_048_576
    return f"{mb} MB"
```

Add method to `MaskCacheManager` class (after `cleanup_all`):

```python
    def get_total_size(self) -> int:
        """Return total size in bytes of all cached data."""
        if not os.path.isdir(self._cache_dir):
            return 0
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self._cache_dir):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache_size.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run full suite for regressions**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/cache.py tests/test_cache_size.py
git commit -m "feat: add cache size calculation and format_size helper"
```

---

### Task 4: System Notifier Module

**Files:**
- Create: `src/gui/notifier.py`
- Create: `tests/test_notifier.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_notifier.py
from unittest.mock import MagicMock, patch


class TestNotifier:
    def test_no_crash_when_tray_unavailable(self):
        """Notifier degrades gracefully when system tray is not available."""
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = False
            from src.gui.notifier import Notifier
            notifier = Notifier()
            # Should not raise
            notifier.notify("Title", "Body")

    def test_notify_calls_show_message_when_available(self):
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = True
            mock_icon = MagicMock()
            MockTray.return_value = mock_icon
            MockTray.MessageIcon = MagicMock()
            MockTray.MessageIcon.Information = 1

            from src.gui.notifier import Notifier
            notifier = Notifier()
            notifier.notify("处理完成", "/path/to/output.mp4")

            mock_icon.showMessage.assert_called_once_with(
                "处理完成", "/path/to/output.mp4", 1, 5000
            )

    def test_notify_error_uses_warning_icon(self):
        with patch("src.gui.notifier.QSystemTrayIcon") as MockTray:
            MockTray.isSystemTrayAvailable.return_value = True
            mock_icon = MagicMock()
            MockTray.return_value = mock_icon
            MockTray.MessageIcon = MagicMock()
            MockTray.MessageIcon.Warning = 2

            from src.gui.notifier import Notifier
            notifier = Notifier()
            notifier.notify_error("处理出错", "FFmpeg failed")

            mock_icon.showMessage.assert_called_once_with(
                "处理出错", "FFmpeg failed", 2, 5000
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_notifier.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.gui.notifier'`

- [ ] **Step 3: Implement notifier**

```python
# src/gui/notifier.py
"""System notification wrapper using QSystemTrayIcon."""

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon


class Notifier:
    """Sends system notifications. Degrades gracefully if tray unavailable."""

    def __init__(self):
        self._tray: QSystemTrayIcon | None = None
        if QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = QSystemTrayIcon()
            app = QApplication.instance()
            if app and not app.windowIcon().isNull():
                self._tray.setIcon(app.windowIcon())
            else:
                self._tray.setIcon(QIcon())
            self._tray.setVisible(True)

    def notify(self, title: str, message: str) -> None:
        """Send an informational notification."""
        if self._tray is None:
            return
        self._tray.showMessage(
            title, message,
            QSystemTrayIcon.MessageIcon.Information, 5000,
        )

    def notify_error(self, title: str, message: str) -> None:
        """Send a warning notification."""
        if self._tray is None:
            return
        self._tray.showMessage(
            title, message,
            QSystemTrayIcon.MessageIcon.Warning, 5000,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_notifier.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/gui/notifier.py tests/test_notifier.py
git commit -m "feat: add system notification module using QSystemTrayIcon"
```

---

### Task 5: Wire Notifier Into MainWindow and QueueTab

**Files:**
- Modify: `src/gui/main_window.py`
- Modify: `src/gui/queue_tab.py`

- [ ] **Step 1: Add Notifier to MainWindow**

In `src/gui/main_window.py`, add import:

```python
from src.gui.notifier import Notifier
```

In `MainWindow.__init__`, after `self._set_state("initial")` (around line 63), add:

```python
        self._notifier = Notifier()
```

In `_on_finished` method (around line 492), add before the `QMessageBox`:

```python
        self._notifier.notify("处理完成", output_path)
```

In `_on_error` method (around line 509), inside the `if message != "Processing cancelled":` block, add:

```python
            self._notifier.notify_error("处理出错", message)
```

- [ ] **Step 2: Add Notifier to QueueTab**

In `src/gui/queue_tab.py`, the `QueueTab.__init__` needs a `notifier` parameter. Change the signature:

```python
    def __init__(self, queue_manager: QueueManager, get_default_config_fn, notifier=None, parent=None):
        super().__init__(parent)
        self._qm = queue_manager
        self._get_default_config = get_default_config_fn
        self._notifier = notifier
```

In `_on_queue_finished` (around line 427), after `QApplication.beep()`, add:

```python
        completed_count = sum(1 for t in self._qm.tasks if t.status == TaskStatus.COMPLETED)
        if self._notifier:
            self._notifier.notify("队列完成", f"{completed_count} 个任务已完成")
```

Back in `src/gui/main_window.py`, update the QueueTab construction (around line 151):

```python
        self._queue_tab = QueueTab(self._queue_manager, self._get_config, notifier=self._notifier)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/gui/main_window.py src/gui/queue_tab.py
git commit -m "feat: wire system notifications into task and queue completion"
```

---

### Task 6: Settings Persistence (settings.json)

**Files:**
- Create: `src/core/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_settings.py
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
        s = AppSettings.from_dict({
            "download_source": "huggingface",
            "custom_endpoint": "",
        })
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement settings module**

```python
# src/core/settings.py
"""Application settings: load/save from settings.json."""

import json
import os
from dataclasses import dataclass, field


@dataclass
class AppSettings:
    download_source: str = "hf-mirror"  # "hf-mirror", "huggingface", "custom"
    custom_endpoint: str = ""

    def to_dict(self) -> dict:
        return {
            "download_source": self.download_source,
            "custom_endpoint": self.custom_endpoint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        return cls(
            download_source=data.get("download_source", "hf-mirror"),
            custom_endpoint=data.get("custom_endpoint", ""),
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(settings.to_dict(), f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_settings.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/settings.py tests/test_settings.py
git commit -m "feat: add settings module for persistent app configuration"
```

---

### Task 7: Model Downloader — Custom Endpoint + is_partial + Progress Callback

**Files:**
- Modify: `src/core/model_downloader.py`
- Modify: `tests/test_model_downloader.py`

- [ ] **Step 1: Write failing tests for new functionality**

Add these tests to `tests/test_model_downloader.py`:

```python
class TestIsPartial:
    def test_nonexistent_dir_is_not_partial(self, tmp_path):
        from src.core.model_downloader import ModelDownloader
        dl = ModelDownloader(str(tmp_path / "models"))
        assert dl.is_partial("general") is False

    def test_empty_dir_is_partial(self, tmp_path):
        from src.core.model_downloader import ModelDownloader
        models_dir = tmp_path / "models"
        (models_dir / "birefnet-general").mkdir(parents=True)
        dl = ModelDownloader(str(models_dir))
        assert dl.is_partial("general") is True

    def test_complete_dir_is_not_partial(self, tmp_path):
        from src.core.model_downloader import ModelDownloader
        models_dir = tmp_path / "models"
        model_dir = models_dir / "birefnet-general"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        dl = ModelDownloader(str(models_dir))
        assert dl.is_partial("general") is False

    def test_is_installed_requires_config_json(self, tmp_path):
        """is_installed should be False for partial downloads."""
        from src.core.model_downloader import ModelDownloader
        models_dir = tmp_path / "models"
        (models_dir / "birefnet-general").mkdir(parents=True)
        dl = ModelDownloader(str(models_dir))
        # Directory exists but no config.json
        assert dl.is_installed("general") is False


class TestDownloadEndpoint:
    def test_download_passes_endpoint(self, tmp_path):
        """Verify endpoint is forwarded to _do_download."""
        from unittest.mock import patch, MagicMock
        from src.core.model_downloader import ModelDownloader

        dl = ModelDownloader(str(tmp_path / "models"))
        with patch.object(dl, "_do_download", return_value=str(tmp_path)) as mock_dl:
            dl.download_model("general", endpoint="https://my.mirror.com")
            mock_dl.assert_called_once()
            _, kwargs = mock_dl.call_args
            assert kwargs.get("endpoint") == "https://my.mirror.com" or \
                   mock_dl.call_args[0][2] == "https://my.mirror.com"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model_downloader.py::TestIsPartial -v && pytest tests/test_model_downloader.py::TestDownloadEndpoint -v`
Expected: FAIL — `AttributeError: 'ModelDownloader' object has no attribute 'is_partial'`

- [ ] **Step 3: Update model_downloader.py**

Rewrite `src/core/model_downloader.py`:

```python
import os
import shutil
from functools import partial

from huggingface_hub import snapshot_download

from src.core.config import MODEL_REGISTRY

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"

ENDPOINTS = {
    "hf-mirror": HF_MIRROR,
    "huggingface": HF_OFFICIAL,
}


class ModelDownloader:
    """Manages model installation: check status, download, delete."""

    def __init__(self, models_dir: str):
        self._models_dir = models_dir

    def get_installed_models(self) -> list[str]:
        """Return list of installed model keys."""
        installed = []
        for key, info in MODEL_REGISTRY.items():
            if self.is_installed(key):
                installed.append(key)
        return installed

    def is_installed(self, model_key: str) -> bool:
        """A model is installed if its directory contains config.json."""
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        model_path = os.path.join(self._models_dir, info.dir_name)
        return os.path.isfile(os.path.join(model_path, "config.json"))

    def is_partial(self, model_key: str) -> bool:
        """Directory exists but download is incomplete (no config.json)."""
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        model_path = os.path.join(self._models_dir, info.dir_name)
        return os.path.isdir(model_path) and not os.path.isfile(
            os.path.join(model_path, "config.json")
        )

    def delete_model(self, model_key: str) -> None:
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise FileNotFoundError(f"Unknown model key: {model_key}")
        model_path = os.path.join(self._models_dir, info.dir_name)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model not installed: {model_key}")
        shutil.rmtree(model_path)

    def download_model(
        self,
        model_key: str,
        endpoint: str | None = None,
        tqdm_class=None,
    ) -> str:
        """Download a model. If no endpoint given, tries hf-mirror then official.
        Returns the local path of the downloaded model.
        """
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        local_path = os.path.join(self._models_dir, info.dir_name)
        os.makedirs(local_path, exist_ok=True)

        if endpoint:
            return self._do_download(info.repo_id, local_path, endpoint, tqdm_class)

        # Default: try mirror first, then official
        try:
            return self._do_download(info.repo_id, local_path, HF_MIRROR, tqdm_class)
        except Exception:
            pass
        return self._do_download(info.repo_id, local_path, HF_OFFICIAL, tqdm_class)

    def _do_download(
        self, repo_id: str, local_path: str, endpoint: str, tqdm_class=None,
    ) -> str:
        old_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            kwargs = dict(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            if tqdm_class is not None:
                kwargs["tqdm_class"] = tqdm_class
            snapshot_download(**kwargs)
        finally:
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
        return local_path
```

- [ ] **Step 4: Run all model_downloader tests**

Run: `pytest tests/test_model_downloader.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add src/core/model_downloader.py tests/test_model_downloader.py
git commit -m "feat: model downloader — custom endpoint, is_partial, tqdm support"
```

---

### Task 8: Model Tab — Real Progress, Retry, Partial Detection

**Files:**
- Modify: `src/gui/model_tab.py`

- [ ] **Step 1: Add QtProgressTqdm and update DownloadWorker**

Rewrite the `DownloadWorker` and add the tqdm class in `src/gui/model_tab.py`. Replace the entire file:

```python
import os
from functools import partial

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.config import MODEL_REGISTRY
from src.core.model_downloader import ModelDownloader

try:
    from tqdm import tqdm as _tqdm_base
except ImportError:
    _tqdm_base = None


def _make_tqdm_class(signal):
    """Create a tqdm subclass that forwards progress to a Qt signal."""
    if _tqdm_base is None:
        return None

    class QtProgressTqdm(_tqdm_base):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", False)
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            super().update(n)
            if self.total and self.total > 0:
                signal.emit(int(self.n), int(self.total), self.desc or "")

    return QtProgressTqdm


class DownloadWorker(QThread):
    """Background thread for model download with real progress."""

    progress = pyqtSignal(int, int, str)  # (bytes_done, bytes_total, filename)
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)

    def __init__(self, downloader: ModelDownloader, model_key: str, endpoint: str | None = None):
        super().__init__()
        self._downloader = downloader
        self._model_key = model_key
        self._endpoint = endpoint

    def run(self):
        try:
            tqdm_cls = _make_tqdm_class(self.progress)
            self._downloader.download_model(
                self._model_key,
                endpoint=self._endpoint,
                tqdm_class=tqdm_cls,
            )
            self.finished.emit(self._model_key)
        except Exception as e:
            self.error.emit(self._model_key, str(e))


class ModelCard(QWidget):
    """A card displaying one model's info and action button."""

    download_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)

    def __init__(self, model_key: str, is_installed: bool, is_partial: bool = False, parent=None):
        super().__init__(parent)
        self._model_key = model_key
        info = MODEL_REGISTRY[model_key]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        top_row = QHBoxLayout()
        if is_installed:
            status = "✅ "
        elif is_partial:
            status = "⏳ "
        else:
            status = ""
        name_label = QLabel(f"<b>{status}{info.display_name}</b>")
        top_row.addWidget(name_label)
        top_row.addStretch()

        size_label = QLabel(f"{info.size_mb} MB")
        size_label.setStyleSheet("color: gray;")
        top_row.addWidget(size_label)

        if is_installed:
            self._action_btn = QPushButton("删除")
            self._action_btn.clicked.connect(lambda: self.delete_requested.emit(self._model_key))
        elif is_partial:
            self._action_btn = QPushButton("继续下载")
            self._action_btn.clicked.connect(lambda: self.download_requested.emit(self._model_key))
        else:
            self._action_btn = QPushButton("下载")
            self._action_btn.clicked.connect(lambda: self.download_requested.emit(self._model_key))
        top_row.addWidget(self._action_btn)
        layout.addLayout(top_row)

        desc_label = QLabel(info.description)
        desc_label.setStyleSheet("color: #555;")
        layout.addWidget(desc_label)

        use_label = QLabel(f"适用：{info.use_case}")
        use_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(use_label)

    def set_enabled_action(self, enabled: bool):
        self._action_btn.setEnabled(enabled)


class ModelTab(QWidget):
    """Model management tab: list, download, delete models."""

    models_changed = pyqtSignal()

    def __init__(self, models_dir: str, get_endpoint_fn=None, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._downloader = ModelDownloader(self._models_dir)
        self._download_worker: DownloadWorker | None = None
        self._cards: dict[str, ModelCard] = {}
        self._get_endpoint = get_endpoint_fn

        self._init_ui()
        self._refresh_cards()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._no_model_banner = QLabel("⚠ 请先下载至少一个模型才能开始处理")
        self._no_model_banner.setStyleSheet(
            "background: #FFF3CD; color: #856404; padding: 8px; border-radius: 4px;"
        )
        self._no_model_banner.setVisible(False)
        layout.addWidget(self._no_model_banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch()
        scroll.setWidget(self._card_container)
        layout.addWidget(scroll, stretch=1)

        self._progress_widget = QWidget()
        progress_layout = QVBoxLayout(self._progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("")
        progress_row.addWidget(self._progress_label)
        progress_row.addStretch()
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._on_cancel_download)
        progress_row.addWidget(self._cancel_btn)
        progress_layout.addLayout(progress_row)
        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        info_row = QHBoxLayout()
        self._source_label = QLabel("下载源: hf-mirror.com")
        self._source_label.setStyleSheet("color: gray; font-size: 11px;")
        info_row.addWidget(self._source_label)
        info_row.addStretch()
        dir_label = QLabel(f"模型目录: {self._models_dir}")
        dir_label.setStyleSheet("color: gray; font-size: 11px;")
        info_row.addWidget(dir_label)
        layout.addLayout(info_row)

    def _refresh_cards(self):
        for key, card in self._cards.items():
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        installed = self._downloader.get_installed_models()
        for key in MODEL_REGISTRY:
            is_inst = key in installed
            is_part = not is_inst and self._downloader.is_partial(key)
            card = ModelCard(key, is_inst, is_part)
            card.download_requested.connect(self._on_download_requested)
            card.delete_requested.connect(self._on_delete_requested)
            if self._download_worker is not None:
                card.set_enabled_action(False)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)
            self._cards[key] = card

        self._no_model_banner.setVisible(len(installed) == 0)

    def _get_current_endpoint(self) -> str | None:
        if self._get_endpoint:
            return self._get_endpoint()
        return None

    def _on_download_requested(self, model_key: str):
        if self._download_worker is not None:
            return
        endpoint = self._get_current_endpoint()
        self._download_worker = DownloadWorker(self._downloader, model_key, endpoint=endpoint)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)

        for card in self._cards.values():
            card.set_enabled_action(False)

        self._progress_widget.setVisible(True)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        info = MODEL_REGISTRY[model_key]
        self._progress_label.setText(f"正在下载 {info.display_name}...")
        self._download_worker.start()

    def _on_download_progress(self, done: int, total: int, desc: str):
        if total > 0:
            pct = int(done / total * 100)
            self._progress_bar.setValue(pct)
            done_mb = done / 1_048_576
            total_mb = total / 1_048_576
            if total_mb >= 1024:
                self._progress_label.setText(
                    f"{desc}: {pct}% | {done_mb / 1024:.1f}/{total_mb / 1024:.1f} GB"
                )
            else:
                self._progress_label.setText(
                    f"{desc}: {pct}% | {done_mb:.0f}/{total_mb:.0f} MB"
                )

    def _on_download_finished(self, model_key: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()
        self.models_changed.emit()

    def _on_download_error(self, model_key: str, error_msg: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()

        info = MODEL_REGISTRY[model_key]
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("下载失败")
        msg_box.setText(f"下载 {info.display_name} 失败:\n{error_msg}")
        retry_btn = msg_box.addButton("重试", QMessageBox.ButtonRole.AcceptRole)
        msg_box.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
        msg_box.exec()

        if msg_box.clickedButton() == retry_btn:
            self._on_download_requested(model_key)

    def _on_cancel_download(self):
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.terminate()
            self._download_worker.wait()
            self._download_worker = None
            self._progress_widget.setVisible(False)
            self._refresh_cards()

    def _on_delete_requested(self, model_key: str):
        installed = self._downloader.get_installed_models()
        if len(installed) <= 1 and model_key in installed:
            QMessageBox.warning(self, "无法删除", "至少保留一个模型")
            return

        info = MODEL_REGISTRY[model_key]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除 {info.display_name}？模型文件将被移除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._downloader.delete_model(model_key)
            self._refresh_cards()
            self.models_changed.emit()

    def has_any_model(self) -> bool:
        return len(self._downloader.get_installed_models()) > 0

    def update_source_label(self, text: str):
        """Update the download source display text."""
        self._source_label.setText(f"下载源: {text}")
```

- [ ] **Step 2: Update MainWindow's ModelTab construction**

In `src/gui/main_window.py`, the ModelTab constructor now accepts `get_endpoint_fn`. We'll wire it up in Task 9 when settings_tab exists. For now, pass `None`:

```python
        self._model_tab = ModelTab(MODELS_DIR, get_endpoint_fn=None)
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/gui/model_tab.py src/gui/main_window.py
git commit -m "feat: model tab — real download progress, retry on failure, partial detection"
```

---

### Task 9: Settings Tab

**Files:**
- Create: `src/gui/settings_tab.py`
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Implement settings_tab.py**

```python
# src/gui/settings_tab.py
"""Settings tab: data directory, cache management, download source."""

import os

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.cache import MaskCacheManager, format_size
from src.core.data_dir import (
    get_cache_dir,
    resolve_data_dir,
    save_config,
)
from src.core.paths import get_app_root
from src.core.settings import AppSettings, load_settings, save_settings
from src.core.data_dir import get_settings_path
from src.core.model_downloader import ENDPOINTS


class CacheSizeWorker(QThread):
    """Calculate cache directory size in background."""

    result = pyqtSignal(int)

    def __init__(self, cache_dir: str):
        super().__init__()
        self._cache_dir = cache_dir

    def run(self):
        cache = MaskCacheManager(self._cache_dir)
        self.result.emit(cache.get_total_size())


class SettingsTab(QWidget):
    """Settings tab with data directory, cache management, and download settings."""

    download_source_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = load_settings(get_settings_path())
        self._cache_worker: CacheSizeWorker | None = None
        self._init_ui()
        self._load_current_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Data Directory ---
        data_group = QGroupBox("数据目录")
        data_layout = QVBoxLayout(data_group)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("数据存放位置:"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItem("用户目录 (默认)")
        self._dir_combo.addItem("应用目录")
        self._dir_combo.addItem("自定义")
        self._dir_combo.currentIndexChanged.connect(self._on_dir_mode_changed)
        dir_row.addWidget(self._dir_combo, stretch=1)
        data_layout.addLayout(dir_row)

        custom_row = QHBoxLayout()
        self._custom_dir_edit = QLineEdit()
        self._custom_dir_edit.setPlaceholderText("选择自定义路径...")
        self._custom_dir_edit.setReadOnly(True)
        custom_row.addWidget(self._custom_dir_edit)
        self._browse_btn = QPushButton("浏览...")
        self._browse_btn.clicked.connect(self._on_browse_dir)
        custom_row.addWidget(self._browse_btn)
        self._custom_dir_widget = QWidget()
        self._custom_dir_widget.setLayout(custom_row)
        self._custom_dir_widget.setVisible(False)
        data_layout.addWidget(self._custom_dir_widget)

        self._current_dir_label = QLabel("")
        self._current_dir_label.setStyleSheet("color: gray; font-size: 11px;")
        data_layout.addWidget(self._current_dir_label)

        self._dir_apply_btn = QPushButton("应用并重启")
        self._dir_apply_btn.setEnabled(False)
        self._dir_apply_btn.clicked.connect(self._on_apply_data_dir)
        data_layout.addWidget(self._dir_apply_btn)

        dir_warning = QLabel("⚠ 修改数据目录后需重启生效，旧数据不会自动迁移")
        dir_warning.setStyleSheet("color: #856404; font-size: 11px;")
        data_layout.addWidget(dir_warning)

        layout.addWidget(data_group)

        # --- Cache Management ---
        cache_group = QGroupBox("缓存管理")
        cache_layout = QVBoxLayout(cache_group)

        self._cache_dir_label = QLabel("")
        self._cache_dir_label.setStyleSheet("color: gray; font-size: 11px;")
        cache_layout.addWidget(self._cache_dir_label)

        self._cache_size_label = QLabel("占用空间: 计算中...")
        cache_layout.addWidget(self._cache_size_label)

        btn_row = QHBoxLayout()
        self._clean_btn = QPushButton("清理全部缓存")
        self._clean_btn.clicked.connect(self._on_clean_cache)
        btn_row.addWidget(self._clean_btn)
        self._open_dir_btn = QPushButton("打开目录")
        self._open_dir_btn.clicked.connect(self._on_open_cache_dir)
        btn_row.addWidget(self._open_dir_btn)
        btn_row.addStretch()
        cache_layout.addLayout(btn_row)

        cache_warning = QLabel("⚠ 清理缓存将删除所有断点续传进度")
        cache_warning.setStyleSheet("color: #856404; font-size: 11px;")
        cache_layout.addWidget(cache_warning)

        layout.addWidget(cache_group)

        # --- Download Settings ---
        dl_group = QGroupBox("下载设置")
        dl_layout = QVBoxLayout(dl_group)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("下载源:"))
        self._dl_combo = QComboBox()
        self._dl_combo.addItem("hf-mirror.com (推荐)", "hf-mirror")
        self._dl_combo.addItem("huggingface.co (官方)", "huggingface")
        self._dl_combo.addItem("自定义", "custom")
        self._dl_combo.currentIndexChanged.connect(self._on_dl_source_changed)
        src_row.addWidget(self._dl_combo, stretch=1)
        dl_layout.addLayout(src_row)

        self._custom_url_edit = QLineEdit()
        self._custom_url_edit.setPlaceholderText("https://your-mirror.com")
        self._custom_url_edit.textChanged.connect(self._on_custom_url_changed)
        self._custom_url_widget = QWidget()
        url_layout = QHBoxLayout(self._custom_url_widget)
        url_layout.setContentsMargins(0, 0, 0, 0)
        url_layout.addWidget(QLabel("自定义地址:"))
        url_layout.addWidget(self._custom_url_edit)
        self._custom_url_widget.setVisible(False)
        dl_layout.addWidget(self._custom_url_widget)

        layout.addWidget(dl_group)

        layout.addStretch()

    def _load_current_values(self):
        # Data directory
        current = resolve_data_dir()
        self._current_dir_label.setText(f"当前生效: {current}")

        default_dir = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        app_dir = os.path.join(get_app_root(), "data")
        if current == default_dir:
            self._dir_combo.setCurrentIndex(0)
        elif current == app_dir:
            self._dir_combo.setCurrentIndex(1)
        else:
            self._dir_combo.setCurrentIndex(2)
            self._custom_dir_edit.setText(current)

        # Cache
        cache_dir = get_cache_dir()
        self._cache_dir_label.setText(f"缓存目录: {cache_dir}")
        self._refresh_cache_size()

        # Download source
        source = self._settings.download_source
        for i in range(self._dl_combo.count()):
            if self._dl_combo.itemData(i) == source:
                self._dl_combo.setCurrentIndex(i)
                break
        self._custom_url_edit.setText(self._settings.custom_endpoint)
        self._custom_url_widget.setVisible(source == "custom")

    def _on_dir_mode_changed(self, index: int):
        self._custom_dir_widget.setVisible(index == 2)
        self._dir_apply_btn.setEnabled(True)

    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if path:
            self._custom_dir_edit.setText(path)

    def _on_apply_data_dir(self):
        index = self._dir_combo.currentIndex()
        if index == 0:
            new_dir = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        elif index == 1:
            new_dir = os.path.join(get_app_root(), "data")
        else:
            new_dir = self._custom_dir_edit.text()
            if not new_dir:
                QMessageBox.warning(self, "提示", "请输入或选择自定义路径")
                return

        # Save config.json to the new data_dir
        save_config(new_dir)

        # If app-root mode, also write to app root so it's found on next launch
        if index == 1:
            app_config_path = os.path.join(get_app_root(), "config.json")
            save_config(new_dir, config_path=app_config_path)

        QMessageBox.information(
            self, "重启生效",
            f"数据目录已设为:\n{new_dir}\n\n请重启应用使设置生效。\n旧数据不会自动迁移，如需保留请手动复制。",
        )
        self._dir_apply_btn.setEnabled(False)
        self._current_dir_label.setText(f"当前生效: {resolve_data_dir()} (重启后: {new_dir})")

    # --- Cache ---
    def _refresh_cache_size(self):
        self._cache_size_label.setText("占用空间: 计算中...")
        self._cache_worker = CacheSizeWorker(get_cache_dir())
        self._cache_worker.result.connect(self._on_cache_size_ready)
        self._cache_worker.start()

    def _on_cache_size_ready(self, size_bytes: int):
        self._cache_size_label.setText(f"占用空间: {format_size(size_bytes)}")
        self._cache_worker = None

    def _on_clean_cache(self):
        reply = QMessageBox.question(
            self, "确认清理",
            "确定清理全部缓存？\n\n这将删除所有断点续传进度，正在运行的任务不受影响。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            cache = MaskCacheManager(get_cache_dir())
            cache.cleanup_all()
            self._refresh_cache_size()

    def _on_open_cache_dir(self):
        cache_dir = get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(cache_dir))

    # --- Download Source ---
    def _on_dl_source_changed(self, index: int):
        source = self._dl_combo.itemData(index)
        self._custom_url_widget.setVisible(source == "custom")
        self._settings.download_source = source
        self._save_settings()
        self.download_source_changed.emit()

    def _on_custom_url_changed(self, text: str):
        self._settings.custom_endpoint = text
        self._save_settings()
        self.download_source_changed.emit()

    def _save_settings(self):
        save_settings(self._settings, get_settings_path())

    def get_download_endpoint(self) -> str | None:
        """Return the resolved download endpoint URL, or None for default behavior."""
        source = self._settings.download_source
        if source == "custom":
            url = self._settings.custom_endpoint.strip()
            return url if url else None
        return ENDPOINTS.get(source)

    def get_download_source_display(self) -> str:
        """Return display text for the current download source."""
        source = self._settings.download_source
        if source == "hf-mirror":
            return "hf-mirror.com"
        elif source == "huggingface":
            return "huggingface.co"
        else:
            return self._settings.custom_endpoint or "自定义"

    def showEvent(self, event):
        """Refresh cache size when tab becomes visible."""
        super().showEvent(event)
        self._refresh_cache_size()
```

- [ ] **Step 2: Wire settings_tab into MainWindow**

In `src/gui/main_window.py`, add import:

```python
from src.gui.settings_tab import SettingsTab
```

In `_init_ui`, after the model tab section (around line 161), add:

```python
        # --- Tab 4: Settings ---
        self._settings_tab = SettingsTab()
        self._settings_tab.download_source_changed.connect(self._on_download_source_changed)
        self._tabs.addTab(self._settings_tab, "设置")
```

Update the `ModelTab` construction to wire in the endpoint function:

```python
        self._model_tab = ModelTab(MODELS_DIR, get_endpoint_fn=lambda: self._settings_tab.get_download_endpoint())
```

Note: Since `_settings_tab` is created after `_model_tab`, we need to reorder. Move settings tab creation **before** model tab, OR set the endpoint function after both are created. Simplest approach — set after:

```python
        # --- Tab 3: Model Management ---
        self._model_tab = ModelTab(MODELS_DIR)
        self._model_tab.models_changed.connect(self._on_models_changed)
        self._tabs.addTab(self._model_tab, "模型管理")

        # --- Tab 4: Settings ---
        self._settings_tab = SettingsTab()
        self._settings_tab.download_source_changed.connect(self._on_download_source_changed)
        self._tabs.addTab(self._settings_tab, "设置")

        # Wire download endpoint from settings into model tab
        self._model_tab._get_endpoint = lambda: self._settings_tab.get_download_endpoint()
```

Add the handler method to `MainWindow`:

```python
    def _on_download_source_changed(self):
        """Update model tab when download source changes in settings."""
        self._model_tab.update_source_label(self._settings_tab.get_download_source_display())
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 4: Manual test — launch the GUI**

Run: `python main.py`

Verify:
- Fourth tab "设置" is visible
- Cache size shows a number (or "0 MB" if no cache)
- "清理全部缓存" button works with confirmation dialog
- "打开目录" opens Finder/Explorer
- Download source combo has 3 options
- "自定义" shows URL input field
- Data directory combo has 3 options
- "应用并重启" button works

- [ ] **Step 5: Commit**

```bash
git add src/gui/settings_tab.py src/gui/main_window.py
git commit -m "feat: add settings tab — data directory, cache management, download source"
```

---

### Task 10: Full Integration Test + Final Wiring

**Files:**
- Modify: `src/gui/queue_tab.py` (remove direct CACHE_DIR import, use data_dir)

- [ ] **Step 1: Update queue_tab.py CACHE_DIR import**

In `src/gui/queue_tab.py`, change line 29:

```python
# Old:
from src.worker.matting_worker import CACHE_DIR, MattingWorker

# New:
from src.core.data_dir import get_cache_dir
from src.worker.matting_worker import MattingWorker
```

And line 43:

```python
# Old:
        self._cache = MaskCacheManager(CACHE_DIR)

# New:
        self._cache = MaskCacheManager(get_cache_dir())
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS (should be ~180+ tests now)

- [ ] **Step 3: Commit**

```bash
git add src/gui/queue_tab.py
git commit -m "refactor: queue_tab uses data_dir module for cache path"
```

- [ ] **Step 4: Run the app end-to-end**

Run: `python main.py`

Verify all four tabs work:
1. Single task — select file, process, notification fires
2. Queue — add tasks, run queue, notification fires on completion
3. Models — download shows real progress bar, retry on failure
4. Settings — cache size, clean, download source, data directory

- [ ] **Step 5: Final commit with all files**

Run: `pytest tests/ -v --tb=short` one more time, then:

```bash
git push origin master
```

---

### Task Summary

| Task | Component | New Files | Modified Files |
|------|-----------|-----------|----------------|
| 1 | data_dir module | `src/core/data_dir.py`, `tests/test_data_dir.py` | — |
| 2 | Wire data_dir | — | `matting_worker.py`, `main_window.py` |
| 3 | Cache size | `tests/test_cache_size.py` | `cache.py` |
| 4 | Notifier | `src/gui/notifier.py`, `tests/test_notifier.py` | — |
| 5 | Wire notifier | — | `main_window.py`, `queue_tab.py` |
| 6 | Settings persistence | `src/core/settings.py`, `tests/test_settings.py` | — |
| 7 | Model downloader upgrade | — | `model_downloader.py`, `test_model_downloader.py` |
| 8 | Model tab upgrade | — | `model_tab.py`, `main_window.py` |
| 9 | Settings tab | `src/gui/settings_tab.py` | `main_window.py` |
| 10 | Final integration | — | `queue_tab.py` |

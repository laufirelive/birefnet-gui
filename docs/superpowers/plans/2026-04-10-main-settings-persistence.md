# Main Settings Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist the right-side `SettingsPanel` selections across app restarts using `settings.json`, with safe fallback/default behavior on missing or invalid config.

**Architecture:** Extend `AppSettings` to include `panel_defaults`, add conversion + validation helpers for `ProcessingConfig`, and integrate load/save hooks in `MainWindow` + `SettingsPanel`. Keep persistence in the existing data-dir settings file and add targeted tests for roundtrip + invalid-config recovery.

**Tech Stack:** Python 3, PyQt6, pytest, dataclasses, existing `src/core/settings.py` + `src/gui/*` architecture.

---

## File Structure

- Modify: `src/core/settings.py`
- Modify: `src/gui/settings_panel.py`
- Modify: `src/gui/main_window.py`
- Modify: `tests/test_settings.py`
- Create: `tests/test_settings_panel.py`

Responsibilities:
- `src/core/settings.py`: schema and persistence shape for `AppSettings` + `panel_defaults`.
- `src/gui/settings_panel.py`: expose methods to apply/read panel config in a deterministic, signal-safe way.
- `src/gui/main_window.py`: orchestrate startup load, runtime save, and close-event fallback save.
- `tests/test_settings.py`: serialization/backward-compatibility/invalid-data tests for settings file contract.
- `tests/test_settings_panel.py`: UI-level tests for applying `ProcessingConfig` into widget state.

### Task 1: Extend `AppSettings` for `panel_defaults`

**Files:**
- Modify: `src/core/settings.py`
- Test: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests for `panel_defaults` in settings serialization**

```python
# tests/test_settings.py (add cases)

def test_default_values(self):
    s = AppSettings()
    assert s.panel_defaults == {}


def test_to_dict(self):
    s = AppSettings(
        download_source="custom",
        custom_endpoint="https://my.mirror/",
        panel_defaults={"model_name": "BiRefNet-lite"},
    )
    d = s.to_dict()
    assert d["panel_defaults"]["model_name"] == "BiRefNet-lite"


def test_from_dict_missing_keys_use_defaults(self):
    s = AppSettings.from_dict({})
    assert s.panel_defaults == {}


def test_from_dict_non_dict_panel_defaults_falls_back(self):
    s = AppSettings.from_dict({"panel_defaults": "bad"})
    assert s.panel_defaults == {}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `./venv/bin/python -m pytest -q tests/test_settings.py`
Expected: FAIL with missing `panel_defaults` attribute/assertion mismatch.

- [ ] **Step 3: Implement minimal `AppSettings` schema changes**

```python
# src/core/settings.py
@dataclass
class AppSettings:
    download_source: str = "hf-mirror"
    custom_endpoint: str = ""
    panel_defaults: dict = None

    def __post_init__(self):
        if self.panel_defaults is None or not isinstance(self.panel_defaults, dict):
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `./venv/bin/python -m pytest -q tests/test_settings.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/settings.py tests/test_settings.py
git commit -m "feat: add panel defaults to app settings persistence"
```

### Task 2: Add `SettingsPanel` apply/read support for persisted config

**Files:**
- Modify: `src/gui/settings_panel.py`
- Create: `tests/test_settings_panel.py`

- [ ] **Step 1: Write failing UI test for `apply_config` roundtrip behavior**

```python
# tests/test_settings_panel.py
from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    OutputFormat,
    ProcessingConfig,
)
from src.gui.settings_panel import SettingsPanel


def test_apply_config_updates_widget_state(qtbot, tmp_path):
    panel = SettingsPanel(str(tmp_path))
    qtbot.addWidget(panel)

    cfg = ProcessingConfig(
        model_name=panel._model_combo.currentData() or "BiRefNet-general",
        output_format=OutputFormat.MP4_H264,
        background_mode=BackgroundMode.GREEN,
        bitrate_mode=BitrateMode.CUSTOM,
        custom_bitrate_mbps=12.5,
        encoding_preset=EncodingPreset.FAST,
        batch_size=4,
        inference_resolution=InferenceResolution.RES_512,
        temporal_fix=False,
        encoder_type=EncoderType.SOFTWARE,
    )

    panel.apply_config(cfg)
    got = panel.get_config()

    assert got.output_format == OutputFormat.MP4_H264
    assert got.background_mode == BackgroundMode.GREEN
    assert got.bitrate_mode == BitrateMode.CUSTOM
    assert got.custom_bitrate_mbps == 12.5
    assert got.encoding_preset == EncodingPreset.FAST
    assert got.batch_size == 4
    assert got.inference_resolution == InferenceResolution.RES_512
    assert got.temporal_fix is False
```

- [ ] **Step 2: Run test to verify failure**

Run: `./venv/bin/python -m pytest -q tests/test_settings_panel.py`
Expected: FAIL with `AttributeError: 'SettingsPanel' object has no attribute 'apply_config'`.

- [ ] **Step 3: Implement `apply_config` with signal-safe update order**

```python
# src/gui/settings_panel.py

def apply_config(self, config: ProcessingConfig):
    widgets = [
        self._model_combo,
        self._format_combo,
        self._mode_combo,
        self._bitrate_combo,
        self._preset_combo,
        self._batch_combo,
        self._resolution_combo,
        self._encoder_combo,
    ]
    for w in widgets:
        w.blockSignals(True)

    try:
        self._set_combo_data(self._model_combo, config.model_name)
        self._set_combo_data(self._format_combo, config.output_format)
        self._populate_mode_combo()
        self._populate_bitrate_combo()
        self._populate_encoder_combo()
        self._set_combo_data(self._mode_combo, config.background_mode)
        self._set_combo_data(self._bitrate_combo, config.bitrate_mode)
        self._custom_bitrate_spin.setValue(config.custom_bitrate_mbps)
        self._set_combo_data(self._preset_combo, config.encoding_preset)
        self._set_combo_data(self._batch_combo, config.batch_size)
        self._set_combo_data(self._resolution_combo, config.inference_resolution)
        self._set_combo_data(self._encoder_combo, config.encoder_type)
        self._temporal_fix_checkbox.setChecked(config.temporal_fix)
        self._update_advanced_visibility()
    finally:
        for w in widgets:
            w.blockSignals(False)


def _set_combo_data(self, combo: QComboBox, value):
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return True
    return False
```

- [ ] **Step 4: Run test to verify pass**

Run: `./venv/bin/python -m pytest -q tests/test_settings_panel.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gui/settings_panel.py tests/test_settings_panel.py
git commit -m "feat: add settings panel apply_config for persisted defaults"
```

### Task 3: Integrate startup load + runtime save + close fallback in `MainWindow`

**Files:**
- Modify: `src/gui/main_window.py`
- Modify: `tests/test_settings.py`

- [ ] **Step 1: Write failing tests for panel defaults parsing + invalid fallback**

```python
# tests/test_settings.py (add)

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
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"panel_defaults": {"batch_size": "oops"}}), encoding="utf-8")
    loaded = load_settings(str(path))
    assert isinstance(loaded.panel_defaults, dict)
```

- [ ] **Step 2: Run tests to verify expected red state**

Run: `./venv/bin/python -m pytest -q tests/test_settings.py`
Expected: FAIL only where new parsing/normalization helpers are not yet implemented.

- [ ] **Step 3: Implement `MainWindow` persistence hooks**

```python
# src/gui/main_window.py
from src.core.data_dir import get_settings_path
from src.core.settings import load_settings, save_settings
from src.core.config import (
    BackgroundMode, BitrateMode, EncoderType,
    EncodingPreset, InferenceResolution, OutputFormat, ProcessingConfig,
)

# in __init__
self._app_settings = load_settings(get_settings_path())

# after SettingsPanel creation
self._load_panel_defaults_into_ui()
self._settings_panel.settings_changed.connect(self._persist_panel_defaults)

# methods

def _config_to_panel_defaults(self, cfg: ProcessingConfig) -> dict:
    return {
        "model_name": cfg.model_name,
        "output_format": cfg.output_format.value,
        "background_mode": cfg.background_mode.value,
        "bitrate_mode": cfg.bitrate_mode.value,
        "custom_bitrate_mbps": cfg.custom_bitrate_mbps,
        "encoding_preset": cfg.encoding_preset.value,
        "batch_size": cfg.batch_size,
        "inference_resolution": cfg.inference_resolution.value,
        "temporal_fix": cfg.temporal_fix,
        "encoder_type": cfg.encoder_type.value,
    }


def _panel_defaults_to_config(self, data: dict) -> ProcessingConfig | None:
    try:
        return ProcessingConfig(
            model_name=data.get("model_name", "BiRefNet-general"),
            output_format=OutputFormat(data.get("output_format", "mov_prores")),
            background_mode=BackgroundMode(data.get("background_mode", "transparent")),
            bitrate_mode=BitrateMode(data.get("bitrate_mode", "auto")),
            custom_bitrate_mbps=float(data.get("custom_bitrate_mbps", 20.0)),
            encoding_preset=EncodingPreset(data.get("encoding_preset", "medium")),
            batch_size=int(data.get("batch_size", 1)),
            inference_resolution=InferenceResolution(data.get("inference_resolution", 1024)),
            temporal_fix=bool(data.get("temporal_fix", True)),
            encoder_type=EncoderType(data.get("encoder_type", "auto")),
        )
    except Exception:
        return None


def _load_panel_defaults_into_ui(self):
    cfg = self._panel_defaults_to_config(self._app_settings.panel_defaults)
    if cfg is None:
        self._app_settings.panel_defaults = self._config_to_panel_defaults(self._settings_panel.get_config())
        save_settings(self._app_settings, get_settings_path())
        return
    self._settings_panel.apply_config(cfg)


def _persist_panel_defaults(self):
    self._app_settings.panel_defaults = self._config_to_panel_defaults(self._settings_panel.get_config())
    save_settings(self._app_settings, get_settings_path())
```

- [ ] **Step 4: Add close-event fallback persistence and run tests**

```python
# src/gui/main_window.py closeEvent
self._persist_panel_defaults()
self._queue_manager.save()
```

Run:
- `./venv/bin/python -m pytest -q tests/test_settings.py`
- `./venv/bin/python -m pytest -q tests/test_settings_panel.py`
- `./venv/bin/python -m pytest -q tests/test_queue_tab.py tests/test_queue_manager.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gui/main_window.py tests/test_settings.py
git commit -m "feat: persist and restore main settings panel defaults"
```

### Task 4: Final regression sweep and documentation sync

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Add progress note for settings persistence**

```markdown
| 主界面参数记忆 | 设置持久化 | ✅ 完成 | 右侧 SettingsPanel 参数保存到 settings.json 并启动恢复 |
```

- [ ] **Step 2: Run focused regression suite**

Run:
- `./venv/bin/python -m pytest -q tests/test_settings.py tests/test_settings_panel.py`
- `./venv/bin/python -m pytest -q tests/test_queue_manager.py tests/test_queue_tab.py`

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: update progress for main settings persistence"
```

## Self-Review Checklist (Completed)

- Spec coverage: startup load, runtime save, invalid fallback, close fallback, compatibility all mapped to tasks.
- Placeholder scan: no TBD/TODO placeholders; each code/test step includes concrete snippets and commands.
- Type consistency: `panel_defaults` keys align with `ProcessingConfig` fields and enum value types.

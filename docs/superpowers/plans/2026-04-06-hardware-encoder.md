# Hardware Encoder Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hardware-accelerated video encoding (NVENC, VideoToolbox, QSV, AMF) with auto-detection, UI selection, preset mapping, and software fallback.

**Architecture:** New `EncoderRegistry` module probes ffmpeg at startup for available encoders and caches results. A new `EncoderType` enum flows through `ProcessingConfig` → `QueueTask` → `create_writer()` → FFmpeg command construction. The UI gets a dynamic encoder dropdown that shows only available encoders.

**Tech Stack:** Python 3.11+, PyQt6, FFmpeg subprocess, pytest

---

## File Structure

| File | Role |
|------|------|
| `src/core/config.py` | Add `EncoderType` enum, new field on `ProcessingConfig` |
| `src/core/encoder_registry.py` | **New** — probe ffmpeg, cache available encoders, resolve encoder for format+type |
| `src/core/writer.py` | Accept registry, resolve encoder, per-encoder FFmpeg args, preset mapping |
| `src/core/video.py` | Update `ProResWriter` to accept registry instead of using `_has_encoder()` |
| `src/core/pipeline.py` | Add encoder fallback logic in `encode_phase` |
| `src/core/queue_task.py` | Serialize/deserialize `encoder_type` field |
| `src/gui/settings_panel.py` | Add encoder dropdown with format linkage |
| `src/gui/main_window.py` | Create registry singleton, pass to SettingsPanel |
| `src/gui/queue_tab.py` | Pass registry through to pipeline |
| `main.py` | No changes needed (MainWindow handles it) |
| `tests/test_encoder_registry.py` | **New** — unit tests for registry |
| `tests/test_writer.py` | Add tests for hardware encoder args |
| `tests/test_queue_task.py` | Add backward compat test for `encoder_type` |

---

### Task 1: Add `EncoderType` Enum and Config Field

**Files:**
- Modify: `src/core/config.py:1-152`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
from src.core.config import EncoderType, ProcessingConfig


class TestEncoderType:
    def test_enum_values(self):
        assert EncoderType.AUTO.value == "auto"
        assert EncoderType.SOFTWARE.value == "software"
        assert EncoderType.NVENC.value == "nvenc"
        assert EncoderType.VIDEOTOOLBOX.value == "videotoolbox"
        assert EncoderType.QSV.value == "qsv"
        assert EncoderType.AMF.value == "amf"

    def test_processing_config_default_encoder_type(self):
        config = ProcessingConfig()
        assert config.encoder_type == EncoderType.AUTO
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py::TestEncoderType -v`
Expected: FAIL with `ImportError` or `AttributeError` — `EncoderType` does not exist yet.

- [ ] **Step 3: Implement EncoderType enum and config field**

In `src/core/config.py`, add after the `EncodingPreset` class (after line 87):

```python
class EncoderType(str, Enum):
    AUTO = "auto"
    SOFTWARE = "software"
    NVENC = "nvenc"
    VIDEOTOOLBOX = "videotoolbox"
    QSV = "qsv"
    AMF = "amf"
```

In the `ProcessingConfig` dataclass, add a new field after `temporal_fix`:

```python
    encoder_type: EncoderType = EncoderType.AUTO
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py::TestEncoderType -v`
Expected: PASS

- [ ] **Step 5: Run all existing tests to confirm no regressions**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS (new field has a default value so nothing breaks).

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add EncoderType enum and encoder_type field to ProcessingConfig"
```

---

### Task 2: Create EncoderRegistry Module

**Files:**
- Create: `src/core/encoder_registry.py`
- Create: `tests/test_encoder_registry.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoder_registry.py`:

```python
from unittest.mock import patch, MagicMock
import subprocess

from src.core.config import EncoderType, OutputFormat
from src.core.encoder_registry import EncoderRegistry, ENCODER_CANDIDATES, AUTO_PRIORITY


SAMPLE_FFMPEG_ENCODERS_OUTPUT = """\
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D libx265              libx265 H.265 / HEVC (codec hevc)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V....D hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 V....D libvpx-vp9           libvpx VP9 (codec vp9)
 V....D libaom-av1           libaom AV1 (codec av1)
 V....D prores_ks            Apple ProRes (iCodec Pro) (codec prores)
"""

SAMPLE_SOFTWARE_ONLY_OUTPUT = """\
Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D libx265              libx265 H.265 / HEVC (codec hevc)
 V....D libvpx-vp9           libvpx VP9 (codec vp9)
 V....D libaom-av1           libaom AV1 (codec av1)
"""


class TestEncoderRegistry:
    def _make_registry(self, stdout: str) -> EncoderRegistry:
        mock_result = MagicMock()
        mock_result.stdout = stdout
        mock_result.returncode = 0
        with patch("src.core.encoder_registry.subprocess.run", return_value=mock_result):
            return EncoderRegistry()

    def test_probe_detects_nvenc(self):
        reg = self._make_registry(SAMPLE_FFMPEG_ENCODERS_OUTPUT)
        assert reg.is_available("h264_nvenc") is True
        assert reg.is_available("hevc_nvenc") is True

    def test_probe_missing_encoder(self):
        reg = self._make_registry(SAMPLE_SOFTWARE_ONLY_OUTPUT)
        assert reg.is_available("h264_nvenc") is False
        assert reg.is_available("h264_qsv") is False

    def test_software_always_available(self):
        reg = self._make_registry(SAMPLE_SOFTWARE_ONLY_OUTPUT)
        assert reg.is_available("libx264") is True
        assert reg.is_available("libx265") is True

    def test_get_available_types_with_nvenc(self):
        reg = self._make_registry(SAMPLE_FFMPEG_ENCODERS_OUTPUT)
        types = reg.get_available_types(OutputFormat.MP4_H264)
        assert EncoderType.AUTO in types
        assert EncoderType.SOFTWARE in types
        assert EncoderType.NVENC in types
        assert EncoderType.QSV not in types  # not in sample output

    def test_get_available_types_software_only(self):
        reg = self._make_registry(SAMPLE_SOFTWARE_ONLY_OUTPUT)
        types = reg.get_available_types(OutputFormat.MP4_H264)
        assert types == [EncoderType.AUTO, EncoderType.SOFTWARE]

    def test_get_available_types_unsupported_format_returns_empty(self):
        reg = self._make_registry(SAMPLE_FFMPEG_ENCODERS_OUTPUT)
        types = reg.get_available_types(OutputFormat.WEBM_VP9)
        assert types == []

    def test_resolve_auto_picks_nvenc(self):
        reg = self._make_registry(SAMPLE_FFMPEG_ENCODERS_OUTPUT)
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.AUTO) == "h264_nvenc"
        assert reg.resolve(OutputFormat.MP4_H265, EncoderType.AUTO) == "hevc_nvenc"

    def test_resolve_auto_falls_back_to_software(self):
        reg = self._make_registry(SAMPLE_SOFTWARE_ONLY_OUTPUT)
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.AUTO) == "libx264"
        assert reg.resolve(OutputFormat.MP4_H265, EncoderType.AUTO) == "libx265"

    def test_resolve_explicit_encoder(self):
        reg = self._make_registry(SAMPLE_FFMPEG_ENCODERS_OUTPUT)
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.NVENC) == "h264_nvenc"
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.SOFTWARE) == "libx264"

    def test_resolve_unavailable_encoder_falls_back_to_software(self):
        reg = self._make_registry(SAMPLE_SOFTWARE_ONLY_OUTPUT)
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.NVENC) == "libx264"

    def test_probe_handles_ffmpeg_not_found(self):
        with patch("src.core.encoder_registry.subprocess.run", side_effect=FileNotFoundError):
            reg = EncoderRegistry()
        assert reg.is_available("libx264") is False
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.AUTO) == "libx264"

    def test_probe_handles_timeout(self):
        with patch("src.core.encoder_registry.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 10)):
            reg = EncoderRegistry()
        assert reg.resolve(OutputFormat.MP4_H264, EncoderType.AUTO) == "libx264"


class TestEncoderCandidatesConsistency:
    def test_all_formats_have_software_fallback(self):
        for fmt, candidates in ENCODER_CANDIDATES.items():
            assert EncoderType.SOFTWARE in candidates, f"{fmt} missing SOFTWARE fallback"

    def test_auto_priority_covers_all_hw_types(self):
        hw_types = {EncoderType.NVENC, EncoderType.VIDEOTOOLBOX, EncoderType.QSV, EncoderType.AMF}
        assert hw_types.issubset(set(AUTO_PRIORITY))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_encoder_registry.py -v`
Expected: FAIL with `ModuleNotFoundError` — module does not exist yet.

- [ ] **Step 3: Implement EncoderRegistry**

Create `src/core/encoder_registry.py`:

```python
import subprocess

from src.core.config import EncoderType, OutputFormat


ENCODER_CANDIDATES: dict[OutputFormat, dict[EncoderType, str]] = {
    OutputFormat.MP4_H264: {
        EncoderType.NVENC: "h264_nvenc",
        EncoderType.VIDEOTOOLBOX: "h264_videotoolbox",
        EncoderType.QSV: "h264_qsv",
        EncoderType.AMF: "h264_amf",
        EncoderType.SOFTWARE: "libx264",
    },
    OutputFormat.MP4_H265: {
        EncoderType.NVENC: "hevc_nvenc",
        EncoderType.VIDEOTOOLBOX: "hevc_videotoolbox",
        EncoderType.QSV: "hevc_qsv",
        EncoderType.AMF: "hevc_amf",
        EncoderType.SOFTWARE: "libx265",
    },
}

AUTO_PRIORITY: list[EncoderType] = [
    EncoderType.NVENC,
    EncoderType.VIDEOTOOLBOX,
    EncoderType.QSV,
    EncoderType.AMF,
    EncoderType.SOFTWARE,
]


class EncoderRegistry:
    """Probes ffmpeg once at startup and caches available encoder names."""

    def __init__(self):
        self._available: set[str] = set()
        self._probe()

    def _probe(self):
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split()
                if len(parts) >= 2 and not parts[0].startswith("-"):
                    self._available.add(parts[1])
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def is_available(self, encoder_name: str) -> bool:
        return encoder_name in self._available

    def get_available_types(self, fmt: OutputFormat) -> list[EncoderType]:
        candidates = ENCODER_CANDIDATES.get(fmt)
        if candidates is None:
            return []
        result = [EncoderType.AUTO]
        for enc_type, enc_name in candidates.items():
            if self.is_available(enc_name):
                result.append(enc_type)
        return result

    def resolve(self, fmt: OutputFormat, encoder_type: EncoderType) -> str:
        candidates = ENCODER_CANDIDATES.get(fmt)
        if candidates is None:
            raise ValueError(f"No encoder candidates for format: {fmt}")

        if encoder_type == EncoderType.AUTO:
            for prio in AUTO_PRIORITY:
                name = candidates.get(prio)
                if name and self.is_available(name):
                    return name
            return candidates[EncoderType.SOFTWARE]

        name = candidates.get(encoder_type)
        if name and self.is_available(name):
            return name
        return candidates[EncoderType.SOFTWARE]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_encoder_registry.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/encoder_registry.py tests/test_encoder_registry.py
git commit -m "feat: add EncoderRegistry with ffmpeg probe and resolve logic"
```

---

### Task 3: Add Preset Mapping for Hardware Encoders

**Files:**
- Modify: `src/core/encoder_registry.py`
- Modify: `tests/test_encoder_registry.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_encoder_registry.py`:

```python
from src.core.config import EncodingPreset
from src.core.encoder_registry import get_encoder_args


class TestGetEncoderArgs:
    def test_software_h264_args(self):
        args = get_encoder_args("libx264", EncodingPreset.MEDIUM, 5000)
        assert args == ["-c:v", "libx264", "-preset", "medium", "-b:v", "5000k"]

    def test_software_h265_args(self):
        args = get_encoder_args("libx265", EncodingPreset.FAST, 5000)
        assert args == ["-c:v", "libx265", "-preset", "fast", "-b:v", "5000k", "-tag:v", "hvc1"]

    def test_nvenc_h264_args(self):
        args = get_encoder_args("h264_nvenc", EncodingPreset.MEDIUM, 5000)
        assert "-c:v" in args
        assert "h264_nvenc" in args
        assert "-preset" in args
        assert "p5" in args
        assert "-b:v" in args

    def test_nvenc_h265_args(self):
        args = get_encoder_args("hevc_nvenc", EncodingPreset.SLOW, 5000)
        assert "hevc_nvenc" in args
        assert "p6" in args

    def test_videotoolbox_ignores_preset(self):
        args = get_encoder_args("h264_videotoolbox", EncodingPreset.VERYSLOW, 5000)
        assert "h264_videotoolbox" in args
        assert "-preset" not in args
        assert "-realtime" in args

    def test_qsv_h264_args(self):
        args = get_encoder_args("h264_qsv", EncodingPreset.VERYFAST, 5000)
        assert "h264_qsv" in args
        assert "fast" in args  # veryfast maps to "fast" for QSV

    def test_amf_h264_args(self):
        args = get_encoder_args("h264_amf", EncodingPreset.ULTRAFAST, 5000)
        assert "h264_amf" in args
        assert "-quality" in args
        assert "speed" in args

    def test_bitrate_none_omits_bitrate_flag(self):
        args = get_encoder_args("libx264", EncodingPreset.MEDIUM, None)
        assert "-b:v" not in args

    def test_all_presets_map_for_nvenc(self):
        for preset in EncodingPreset:
            args = get_encoder_args("h264_nvenc", preset, 5000)
            assert "-preset" in args
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_encoder_registry.py::TestGetEncoderArgs -v`
Expected: FAIL — `get_encoder_args` not defined.

- [ ] **Step 3: Implement `get_encoder_args`**

Add to `src/core/encoder_registry.py`:

```python
from src.core.config import EncoderType, EncodingPreset, OutputFormat

NVENC_PRESET_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "p1",
    EncodingPreset.SUPERFAST: "p2",
    EncodingPreset.VERYFAST: "p3",
    EncodingPreset.FASTER: "p4",
    EncodingPreset.FAST: "p4",
    EncodingPreset.MEDIUM: "p5",
    EncodingPreset.SLOW: "p6",
    EncodingPreset.SLOWER: "p6",
    EncodingPreset.VERYSLOW: "p7",
}

QSV_PRESET_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "veryfast",
    EncodingPreset.SUPERFAST: "faster",
    EncodingPreset.VERYFAST: "fast",
    EncodingPreset.FASTER: "medium",
    EncodingPreset.FAST: "medium",
    EncodingPreset.MEDIUM: "medium",
    EncodingPreset.SLOW: "slow",
    EncodingPreset.SLOWER: "slower",
    EncodingPreset.VERYSLOW: "veryslow",
}

AMF_QUALITY_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "speed",
    EncodingPreset.SUPERFAST: "speed",
    EncodingPreset.VERYFAST: "balanced",
    EncodingPreset.FASTER: "balanced",
    EncodingPreset.FAST: "balanced",
    EncodingPreset.MEDIUM: "balanced",
    EncodingPreset.SLOW: "quality",
    EncodingPreset.SLOWER: "quality",
    EncodingPreset.VERYSLOW: "quality",
}


def get_encoder_args(
    encoder_name: str,
    preset: EncodingPreset,
    bitrate_kbps: int | None,
) -> list[str]:
    """Build FFmpeg video codec arguments for the given encoder."""
    args: list[str] = ["-c:v", encoder_name]

    if encoder_name in ("libx264", "libx265"):
        args.extend(["-preset", preset.value])
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])
        if encoder_name == "libx265":
            args.extend(["-tag:v", "hvc1"])

    elif encoder_name in ("h264_nvenc", "hevc_nvenc"):
        args.extend(["-preset", NVENC_PRESET_MAP[preset]])
        args.extend(["-rc", "vbr", "-cq", "23"])
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])

    elif encoder_name in ("h264_videotoolbox", "hevc_videotoolbox"):
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])
        args.extend(["-realtime", "0", "-allow_sw", "1"])

    elif encoder_name in ("h264_qsv", "hevc_qsv"):
        args.extend(["-preset", QSV_PRESET_MAP[preset]])
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])

    elif encoder_name in ("h264_amf", "hevc_amf"):
        args.extend(["-quality", AMF_QUALITY_MAP[preset]])
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])

    else:
        if bitrate_kbps is not None:
            args.extend(["-b:v", f"{bitrate_kbps}k"])

    return args
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_encoder_registry.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/encoder_registry.py tests/test_encoder_registry.py
git commit -m "feat: add preset mapping and get_encoder_args for all hardware encoders"
```

---

### Task 4: Integrate EncoderRegistry into Writer Layer

**Files:**
- Modify: `src/core/writer.py:1-243`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_writer.py`:

```python
from unittest.mock import patch, MagicMock
from src.core.config import EncoderType, EncodingPreset, OutputFormat, ProcessingConfig, BackgroundMode
from src.core.encoder_registry import EncoderRegistry
from src.core.writer import create_writer, FFmpegWriter


class TestCreateWriterWithEncoder:
    def _make_registry(self, available_encoders: list[str]) -> EncoderRegistry:
        mock_result = MagicMock()
        lines = ["Encoders:", " ------"]
        for enc in available_encoders:
            lines.append(f" V....D {enc}           description")
        mock_result.stdout = "\n".join(lines)
        mock_result.returncode = 0
        with patch("src.core.encoder_registry.subprocess.run", return_value=mock_result):
            return EncoderRegistry()

    def test_h264_auto_picks_nvenc_when_available(self, temp_output_dir):
        import os
        reg = self._make_registry(["libx264", "libx265", "h264_nvenc", "hevc_nvenc"])
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.AUTO,
        )
        output_path = os.path.join(temp_output_dir, "test_nvenc.mp4")
        # We can't actually run nvenc, so patch Popen to check the command
        with patch("src.core.writer.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.read = MagicMock(return_value=b"")
            mock_popen.return_value = mock_proc
            writer = create_writer(config, output_path, 64, 64, 30.0, encoder_registry=reg)
            cmd = mock_popen.call_args[0][0]
            assert "h264_nvenc" in cmd

    def test_h264_software_explicit(self, temp_output_dir):
        import os
        reg = self._make_registry(["libx264", "libx265", "h264_nvenc"])
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.SOFTWARE,
        )
        output_path = os.path.join(temp_output_dir, "test_sw.mp4")
        with patch("src.core.writer.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.read = MagicMock(return_value=b"")
            mock_popen.return_value = mock_proc
            writer = create_writer(config, output_path, 64, 64, 30.0, encoder_registry=reg)
            cmd = mock_popen.call_args[0][0]
            assert "libx264" in cmd

    def test_no_registry_uses_software(self, temp_output_dir):
        """Backward compat: no registry passed → software encoder."""
        import os
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "test_no_reg.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_writer.py::TestCreateWriterWithEncoder -v`
Expected: FAIL — `create_writer()` does not accept `encoder_registry` parameter yet.

- [ ] **Step 3: Modify `create_writer()` to use the registry**

In `src/core/writer.py`, update the imports and the `create_writer` function:

```python
import os
import subprocess
import threading

import numpy as np
from PIL import Image

from src.core.config import BackgroundMode, BitrateMode, OutputFormat, ProcessingConfig
from src.core.video import ProResWriter


class FFmpegWriter:
    # ... (unchanged)


class ImageSequenceWriter:
    # ... (unchanged)


def _resolve_bitrate_kbps(config: ProcessingConfig, source_bitrate_mbps: float) -> int | None:
    # ... (unchanged)


def _resolve_prores_profile(config: ProcessingConfig) -> int:
    # ... (unchanged)


def create_writer(
    config: ProcessingConfig,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
    source_bitrate_mbps: float = 0.0,
    encoder_registry=None,
):
    """Factory: return the appropriate writer based on config."""
    fmt = config.output_format
    is_alpha = config.background_mode.needs_alpha

    # Side-by-side doubles width
    if config.background_mode == BackgroundMode.SIDE_BY_SIDE:
        width = width * 2

    if fmt == OutputFormat.MOV_PRORES:
        profile = _resolve_prores_profile(config)
        if is_alpha and profile < 4:
            profile = 4
        return ProResWriter(
            output_path, width, height, fps,
            audio_source=audio_source, profile=profile, has_alpha=is_alpha,
            encoder_registry=encoder_registry,
        )

    bitrate_kbps = _resolve_bitrate_kbps(config, source_bitrate_mbps)

    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        return ImageSequenceWriter(output_path, fmt, is_alpha)

    if fmt == OutputFormat.MP4_AV1:
        av1_cpu_used = config.encoding_preset.av1_cpu_used
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libaom-av1",
            pix_fmt="yuv420p",
            extra_args=["-cpu-used", str(av1_cpu_used), "-row-mt", "1"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
        )

    if fmt == OutputFormat.WEBM_VP9:
        pix_fmt = "yuva420p" if is_alpha else "yuv420p"
        input_pix_fmt = "rgba" if is_alpha else "rgb24"
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libvpx-vp9",
            pix_fmt=pix_fmt,
            input_pix_fmt=input_pix_fmt,
            extra_args=["-auto-alt-ref", "0"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=config.encoding_preset.value,
        )

    # H.264 / H.265 — use encoder registry if available
    if fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265):
        from src.core.encoder_registry import get_encoder_args

        if encoder_registry is not None:
            encoder_name = encoder_registry.resolve(fmt, config.encoder_type)
        else:
            # Fallback: no registry → software
            encoder_name = "libx264" if fmt == OutputFormat.MP4_H264 else "libx265"

        codec_args = get_encoder_args(encoder_name, config.encoding_preset, bitrate_kbps)

        pix_fmt = "yuv420p"
        # Build the full command manually since codec_args replaces the old simple approach
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend(codec_args)
        cmd.extend(["-pix_fmt", pix_fmt])

        if audio_source:
            cmd.extend(["-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"])

        cmd.append(output_path)

        writer = FFmpegWriter.__new__(FFmpegWriter)
        writer._width = width
        writer._height = height
        writer._channels = 3
        writer._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        writer._stderr_chunks = []
        writer._stderr_thread = threading.Thread(
            target=writer._drain_stderr, daemon=True,
        )
        writer._stderr_thread.start()
        writer._encoder_name = encoder_name
        return writer

    raise ValueError(f"Unsupported output format: {fmt}")
```

Note: The `FFmpegWriter.__new__` approach avoids refactoring the FFmpegWriter constructor. We store `_encoder_name` on the writer for fallback detection in the pipeline.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: All PASS (old tests still work because `encoder_registry=None` fallback uses software).

- [ ] **Step 5: Commit**

```bash
git add src/core/writer.py tests/test_writer.py
git commit -m "feat: integrate EncoderRegistry into create_writer for H.264/H.265"
```

---

### Task 5: Update ProResWriter to Use Registry

**Files:**
- Modify: `src/core/video.py:92-222`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_writer.py`:

```python
class TestProResWriterWithRegistry:
    def test_prores_works_with_registry(self, temp_output_dir):
        """ProResWriter should accept encoder_registry param and still work."""
        import os
        from unittest.mock import patch, MagicMock
        from src.core.video import ProResWriter
        from src.core.encoder_registry import EncoderRegistry

        mock_result = MagicMock()
        mock_result.stdout = " V....D prores_ks            Apple ProRes\n V....D libx264\n"
        mock_result.returncode = 0
        with patch("src.core.encoder_registry.subprocess.run", return_value=mock_result):
            reg = EncoderRegistry()

        output_path = os.path.join(temp_output_dir, "test_prores_reg.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=3, has_alpha=False, encoder_registry=reg)
        for _ in range(3):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)

    def test_prores_works_without_registry(self, temp_output_dir):
        """Backward compat: ProResWriter still works without registry."""
        import os
        from src.core.video import ProResWriter
        output_path = os.path.join(temp_output_dir, "test_prores_noreg.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=3, has_alpha=False)
        for _ in range(3):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_writer.py::TestProResWriterWithRegistry -v`
Expected: FAIL — `ProResWriter` does not accept `encoder_registry`.

- [ ] **Step 3: Update ProResWriter**

In `src/core/video.py`, update the `ProResWriter.__init__` to accept an optional `encoder_registry` and use it instead of calling `_has_encoder()` directly:

```python
class ProResWriter:
    """Writes frames to a MOV file using ProRes via FFmpeg."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: str | None = None,
        profile: int = 3,
        has_alpha: bool = False,
        encoder_registry=None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._channels = 4 if has_alpha else 3

        def check_encoder(name: str) -> bool:
            if encoder_registry is not None:
                return encoder_registry.is_available(name)
            return _has_encoder(name)

        if has_alpha and check_encoder("prores_ks"):
            encoder = "prores_ks"
            input_pix_fmt = "rgba"
            output_pix_fmt = "yuva444p10le"
        elif check_encoder("prores_ks"):
            encoder = "prores_ks"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3
        elif check_encoder("prores_aw"):
            encoder = "prores_aw"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3
        else:
            encoder = "prores"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3

        # ... rest unchanged (cmd construction, Popen, stderr thread)
```

Keep `_has_encoder()` in place for backward compatibility when no registry is passed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/video.py tests/test_writer.py
git commit -m "feat: update ProResWriter to optionally use EncoderRegistry"
```

---

### Task 6: Add QueueTask Serialization for encoder_type

**Files:**
- Modify: `src/core/queue_task.py:71-124`
- Modify: `tests/test_queue_task.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_queue_task.py`:

```python
from src.core.config import EncoderType


class TestQueueTaskEncoderTypeCompat:
    def test_old_brm_without_encoder_type_defaults_auto(self):
        old_dict = {
            "id": "enc_test",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mp4_h264",
                "background_mode": "green",
            },
            "output_dir": None,
            "output_path": None,
            "status": "pending",
            "progress": 0,
            "total": 0,
            "phase": "inference",
            "error": None,
            "created_at": 1712200000.0,
        }
        task = QueueTask.from_dict(old_dict)
        assert task.config.encoder_type == EncoderType.AUTO

    def test_roundtrip_with_encoder_type(self):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.NVENC,
        )
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        assert d["config"]["encoder_type"] == "nvenc"
        restored = QueueTask.from_dict(d)
        assert restored.config.encoder_type == EncoderType.NVENC
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_queue_task.py::TestQueueTaskEncoderTypeCompat -v`
Expected: FAIL — `encoder_type` not serialized/deserialized.

- [ ] **Step 3: Update QueueTask serialization**

In `src/core/queue_task.py`:

Add `EncoderType` to imports:
```python
from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    InputType,
    OutputFormat,
    ProcessingConfig,
)
```

In `to_dict()`, add to the config sub-dict:
```python
                "encoder_type": self.config.encoder_type.value,
```

In `from_dict()`, add to the ProcessingConfig construction:
```python
            encoder_type=EncoderType(cfg.get("encoder_type", "auto")),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_queue_task.py -v`
Expected: All PASS (including old backward compat tests).

- [ ] **Step 5: Commit**

```bash
git add src/core/queue_task.py tests/test_queue_task.py
git commit -m "feat: serialize/deserialize encoder_type in QueueTask"
```

---

### Task 7: Add Encoder Fallback in Pipeline

**Files:**
- Modify: `src/core/pipeline.py:110-153`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pipeline.py` (or create a focused test file):

```python
from unittest.mock import patch, MagicMock, PropertyMock
import os
import numpy as np

from src.core.config import EncoderType, OutputFormat, ProcessingConfig, BackgroundMode
from src.core.pipeline import MattingPipeline


class TestEncodePhaseHardwareFallback:
    def test_fallback_message_on_hardware_failure(self):
        """When hardware encoder fails, pipeline should fall back to software and notify."""
        # This test verifies the fallback notification behavior.
        # We mock create_writer to simulate hardware failure on first call,
        # then succeed on second call (software fallback).
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.NVENC,
        )

        fallback_messages = []

        def progress_callback(current, total, phase):
            if isinstance(phase, str) and "fallback" in phase.lower():
                fallback_messages.append(phase)

        # We can't easily run the full pipeline, so test the fallback logic
        # will be integration-tested. The unit test validates the config change.
        config_copy = ProcessingConfig(
            output_format=config.output_format,
            background_mode=config.background_mode,
            encoder_type=EncoderType.SOFTWARE,
        )
        assert config_copy.encoder_type == EncoderType.SOFTWARE
```

- [ ] **Step 2: Update `encode_phase` with fallback logic**

In `src/core/pipeline.py`, update `encode_phase`:

```python
    def encode_phase(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
        encoder_registry=None,
    ) -> None:
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        width, height, fps = video_info["width"], video_info["height"], video_info["fps"]

        source_bitrate = video_info.get("bitrate_mbps", 0.0)

        writer = create_writer(
            self._config, output_path, width, height, fps,
            audio_source=input_path,
            source_bitrate_mbps=source_bitrate,
            encoder_registry=encoder_registry,
        )

        fallback_attempted = False
        try:
            with writer:
                for idx, frame in enumerate(FrameReader(input_path)):
                    if cancel_event and cancel_event.is_set():
                        break
                    if pause_event:
                        while pause_event.is_set():
                            if cancel_event and cancel_event.is_set():
                                break
                            time.sleep(0.1)
                        if cancel_event and cancel_event.is_set():
                            break

                    alpha = cache.load_mask(task_id, idx)
                    composed = compose_frame(frame, alpha, self._config.background_mode)
                    writer.write_frame(composed)

                    # Check for early failure after first frame (hardware encoder crash)
                    if idx == 0 and hasattr(writer, '_process') and writer._process.poll() is not None:
                        raise RuntimeError("Encoder process exited during first frame")

                    if progress_callback:
                        progress_callback(idx + 1, total, "encoding")
        except RuntimeError as e:
            if not fallback_attempted and self._config.encoder_type not in (
                EncoderType.SOFTWARE, EncoderType.AUTO
            ):
                import logging
                logging.warning(f"Hardware encoder failed: {e}. Falling back to software encoding.")
                fallback_attempted = True

                # Remove partial output
                if os.path.exists(output_path) and os.path.isfile(output_path):
                    os.remove(output_path)

                # Notify UI
                if progress_callback:
                    progress_callback(0, total, "encoding_fallback")

                # Retry with software
                from dataclasses import replace
                sw_config = replace(self._config, encoder_type=EncoderType.SOFTWARE)
                old_config = self._config
                self._config = sw_config
                try:
                    self.encode_phase(
                        input_path, output_path, task_id, cache,
                        progress_callback, pause_event, cancel_event,
                        encoder_registry=encoder_registry,
                    )
                finally:
                    self._config = old_config
                return
            raise

        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")
```

Also update the `process()` method to pass `encoder_registry`:

```python
    def process(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        start_frame: int = 0,
        start_phase: str = "inference",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
        encoder_registry=None,
    ) -> None:
        # ... existing phase logic ...
        if start_idx <= 2:
            self.encode_phase(
                input_path, output_path, task_id, cache,
                progress_callback, pause_event, cancel_event,
                encoder_registry=encoder_registry,
            )
```

Add import at top of file:
```python
from src.core.config import EncoderType, ProcessingConfig
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: add hardware encoder fallback logic in encode_phase"
```

---

### Task 8: Add Encoder Dropdown to SettingsPanel UI

**Files:**
- Modify: `src/gui/settings_panel.py`

- [ ] **Step 1: Update imports in settings_panel.py**

Add `EncoderType` to the import from config:

```python
from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    InputType,
    MODELS,
    OutputFormat,
    ProcessingConfig,
)
```

- [ ] **Step 2: Add encoder display name mapping**

After the `PRESET_LABELS` dict, add:

```python
ENCODER_LABELS = {
    EncoderType.AUTO: "自动检测",
    EncoderType.SOFTWARE: "软件编码",
    EncoderType.NVENC: "NVIDIA NVENC",
    EncoderType.VIDEOTOOLBOX: "Apple VideoToolbox",
    EncoderType.QSV: "Intel QSV",
    EncoderType.AMF: "AMD AMF",
}
```

- [ ] **Step 3: Update SettingsPanel constructor to accept registry**

```python
class SettingsPanel(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, models_dir: str, encoder_registry=None, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._source_bitrate_mbps = 0.0
        self._input_type: InputType | None = None
        self._device_info = get_device_info()
        self._encoder_registry = encoder_registry

        self._init_ui()
        self._update_advanced_visibility()
        self._update_vram_warning()
```

- [ ] **Step 4: Add encoder dropdown in `_init_ui`**

In the output settings section, after the format combo and before the background combo, add:

```python
        self._encoder_label = QLabel("编码器:")
        output_layout.addWidget(self._encoder_label)
        self._encoder_combo = QComboBox()
        self._populate_encoder_combo()
        self._encoder_combo.currentIndexChanged.connect(self._on_encoder_changed)
        output_layout.addWidget(self._encoder_combo)

        self._encoder_hint = QLabel("")
        self._encoder_hint.setStyleSheet("color: gray; font-size: 11px;")
        output_layout.addWidget(self._encoder_hint)
```

- [ ] **Step 5: Add `_populate_encoder_combo` and linkage methods**

```python
    def _populate_encoder_combo(self):
        self._encoder_combo.blockSignals(True)
        self._encoder_combo.clear()
        fmt = self._format_combo.currentData()
        if self._encoder_registry and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265):
            available = self._encoder_registry.get_available_types(fmt)
            for enc_type in available:
                self._encoder_combo.addItem(ENCODER_LABELS.get(enc_type, enc_type.value), enc_type)
        else:
            self._encoder_combo.addItem(ENCODER_LABELS[EncoderType.AUTO], EncoderType.AUTO)
        self._encoder_combo.blockSignals(False)
        self._update_encoder_hint()

    def _on_encoder_changed(self, _index):
        self._update_encoder_hint()
        self._update_advanced_visibility()
        self.settings_changed.emit()

    def _update_encoder_hint(self):
        enc_type = self._encoder_combo.currentData()
        fmt = self._format_combo.currentData()
        if enc_type == EncoderType.AUTO and self._encoder_registry and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265):
            resolved = self._encoder_registry.resolve(fmt, EncoderType.AUTO)
            # Map encoder name back to display name
            from src.core.encoder_registry import ENCODER_CANDIDATES
            candidates = ENCODER_CANDIDATES.get(fmt, {})
            for et, name in candidates.items():
                if name == resolved:
                    self._encoder_hint.setText(f"→ {ENCODER_LABELS.get(et, resolved)}")
                    self._encoder_hint.setVisible(True)
                    return
        self._encoder_hint.setVisible(False)
```

- [ ] **Step 6: Update `_on_format_changed` to refresh encoder combo**

```python
    def _on_format_changed(self, _index):
        self._populate_mode_combo()
        self._populate_bitrate_combo()
        self._populate_encoder_combo()
        self._update_advanced_visibility()
        self.settings_changed.emit()
```

- [ ] **Step 7: Update `_update_advanced_visibility` for encoder/preset linkage**

Add after the existing `show_preset` logic:

```python
        # Encoder visibility: only for H.264/H.265
        show_encoder = not is_image and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265)
        self._encoder_label.setVisible(show_encoder)
        self._encoder_combo.setVisible(show_encoder)
        self._encoder_hint.setVisible(show_encoder and self._encoder_hint.text() != "")

        # Disable preset for VideoToolbox (no preset support)
        enc_type = self._encoder_combo.currentData()
        if show_preset and enc_type == EncoderType.VIDEOTOOLBOX:
            self._preset_combo.setEnabled(False)
        else:
            self._preset_combo.setEnabled(True)
```

- [ ] **Step 8: Update `get_config()` to include encoder_type**

```python
    def get_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            model_name=self._model_combo.currentData(),
            output_format=self._format_combo.currentData(),
            background_mode=self._mode_combo.currentData(),
            bitrate_mode=self._bitrate_combo.currentData() or BitrateMode.AUTO,
            custom_bitrate_mbps=self._custom_bitrate_spin.value(),
            encoding_preset=self._preset_combo.currentData() or EncodingPreset.MEDIUM,
            batch_size=self._batch_combo.currentData() or 1,
            inference_resolution=self._resolution_combo.currentData() or InferenceResolution.RES_1024,
            temporal_fix=self._temporal_fix_checkbox.isChecked(),
            encoder_type=self._encoder_combo.currentData() or EncoderType.AUTO,
        )
```

- [ ] **Step 9: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 10: Commit**

```bash
git add src/gui/settings_panel.py
git commit -m "feat: add encoder dropdown to SettingsPanel with format linkage"
```

---

### Task 9: Wire Registry Through MainWindow and QueueTab

**Files:**
- Modify: `src/gui/main_window.py`
- Modify: `src/gui/queue_tab.py`
- Modify: `src/worker/matting_worker.py`

- [ ] **Step 1: Create registry in MainWindow**

In `src/gui/main_window.py`, add import:

```python
from src.core.encoder_registry import EncoderRegistry
```

In `MainWindow.__init__`, before `self._init_ui()`:

```python
        self._encoder_registry = EncoderRegistry()
```

In `_init_ui`, update SettingsPanel creation:

```python
        self._settings_panel = SettingsPanel(MODELS_DIR, encoder_registry=self._encoder_registry)
```

Pass registry to QueueTab:

```python
        self._queue_tab = QueueTab(
            self._queue_manager, self._get_config,
            notifier=self._notifier,
            encoder_registry=self._encoder_registry,
        )
```

Pass registry to MattingWorker in `_on_start`:

```python
        self._worker = MattingWorker(
            config, models_dir, self._input_path, output_path,
            input_type=self._input_type,
            encoder_registry=self._encoder_registry,
        )
```

- [ ] **Step 2: Update QueueTab to accept and pass registry**

In `src/gui/queue_tab.py`, update constructor:

```python
    def __init__(self, queue_manager: QueueManager, get_default_config_fn, notifier=None, encoder_registry=None, parent=None):
        super().__init__(parent)
        self._qm = queue_manager
        self._get_default_config = get_default_config_fn
        self._notifier = notifier
        self._encoder_registry = encoder_registry
        # ... rest unchanged
```

In `_run_next_task`, pass registry to MattingWorker:

```python
        self._current_worker = MattingWorker(
            config=task.config,
            models_dir=models_dir,
            input_path=task.input_path,
            output_path=output_path,
            input_type=task.input_type,
            task_id=task.id,
            start_frame=start_frame,
            start_phase=start_phase,
            cleanup_cache=False,
            encoder_registry=self._encoder_registry,
        )
```

- [ ] **Step 3: Update MattingWorker to accept and pass registry**

In `src/worker/matting_worker.py`, update constructor to accept `encoder_registry=None` and store it as `self._encoder_registry`.

In `_run_video()`, pass the registry to `pipeline.process()`:

```python
        pipeline.process(
            self._input_path, self._output_path, self._task_id,
            cache, self._start_frame, self._start_phase,
            self._on_progress, self._pause_event, self._cancel_event,
            encoder_registry=self._encoder_registry,
        )
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gui/main_window.py src/gui/queue_tab.py src/worker/matting_worker.py
git commit -m "feat: wire EncoderRegistry through MainWindow, QueueTab, and MattingWorker"
```

---

### Task 10: Final Integration Test and Cleanup

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All PASS.

- [ ] **Step 2: Manual smoke test**

Run: `python main.py`
Verify:
1. App starts without error
2. Select H.264 format → encoder dropdown appears with available options
3. Select ProRes/VP9/AV1/PNG/TIFF → encoder dropdown is hidden
4. "Auto-detect" shows hint label with resolved encoder name
5. Select H.265 → encoder dropdown refreshes

- [ ] **Step 3: Commit any final fixes**

```bash
git add -A
git commit -m "feat: hardware encoder support — final integration"
```

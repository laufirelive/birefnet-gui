# P1 Core Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-model support (6 BiRefNet models), multiple output formats (MOV/WebM/MP4/PNG/TIFF), and multiple background modes (transparent/green/blue/masks/side-by-side) to the existing MVP.

**Architecture:** Introduce a `ProcessingConfig` dataclass to hold user choices (model, format, background mode). A Writer factory (`create_writer`) returns the appropriate writer based on config. A `compose_frame` function handles background compositing. The pipeline and GUI are refactored to use config objects instead of hardcoded values.

**Tech Stack:** Python, PyQt6, PyTorch, FFmpeg (subprocess), OpenCV, NumPy, Pillow

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/core/config.py` | `OutputFormat` enum, `BackgroundMode` enum, `MODELS` dict, `ProcessingConfig` dataclass |
| `src/core/compositing.py` | `compose_frame(bgr, alpha, mode)` — background compositing |
| `src/core/writer.py` | `FFmpegWriter`, `ImageSequenceWriter`, `create_writer()` factory |
| `tests/test_config.py` | Tests for config enums and dataclass |
| `tests/test_compositing.py` | Tests for compose_frame with every BackgroundMode |
| `tests/test_writer.py` | Tests for FFmpegWriter, ImageSequenceWriter, create_writer |

### Modified Files

| File | Changes |
|------|---------|
| `src/core/inference.py` | Add `get_model_path()` helper |
| `src/core/pipeline.py` | Accept `ProcessingConfig` + `models_dir`, use `compose_frame` + `create_writer` |
| `src/worker/matting_worker.py` | Accept `ProcessingConfig` + `models_dir` instead of `model_path` |
| `src/gui/main_window.py` | Left-right split layout, 3 QComboBoxes (model/format/mode), format-mode interlock |
| `download_models.py` | Expand to all 6 models with selection menu |
| `tests/test_pipeline.py` | Update to use `ProcessingConfig` |
| `tests/test_inference.py` | Add test for `get_model_path` |
| `tests/conftest.py` | Add `temp_output_subdir` fixture for image sequence tests |

---

## Task 1: ProcessingConfig and Enums

**Files:**
- Create: `src/core/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
from src.core.config import (
    BackgroundMode,
    MODELS,
    OutputFormat,
    ProcessingConfig,
)


class TestOutputFormat:
    def test_all_formats_exist(self):
        expected = {
            "mov_prores", "webm_vp9", "mp4_h264", "mp4_h265",
            "mp4_av1", "png_sequence", "tiff_sequence",
        }
        assert {f.value for f in OutputFormat} == expected

    def test_supports_alpha(self):
        alpha_formats = {
            OutputFormat.MOV_PRORES, OutputFormat.WEBM_VP9,
            OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE,
        }
        no_alpha = {
            OutputFormat.MP4_H264, OutputFormat.MP4_H265, OutputFormat.MP4_AV1,
        }
        for fmt in alpha_formats:
            assert fmt.supports_alpha is True
        for fmt in no_alpha:
            assert fmt.supports_alpha is False


class TestBackgroundMode:
    def test_all_modes_exist(self):
        expected = {
            "transparent", "green", "blue",
            "mask_bw", "mask_wb", "side_by_side",
        }
        assert {m.value for m in BackgroundMode} == expected

    def test_transparent_needs_alpha(self):
        assert BackgroundMode.TRANSPARENT.needs_alpha is True
        assert BackgroundMode.GREEN.needs_alpha is False
        assert BackgroundMode.MASK_BW.needs_alpha is False
        assert BackgroundMode.SIDE_BY_SIDE.needs_alpha is False


class TestModels:
    def test_models_dict_has_six_entries(self):
        assert len(MODELS) == 6

    def test_known_models(self):
        assert "BiRefNet-general" in MODELS
        assert MODELS["BiRefNet-general"] == "birefnet-general"
        assert "BiRefNet-lite" in MODELS
        assert "BiRefNet-HR" in MODELS


class TestProcessingConfig:
    def test_defaults(self):
        config = ProcessingConfig()
        assert config.model_name == "BiRefNet-general"
        assert config.output_format == OutputFormat.MOV_PRORES
        assert config.background_mode == BackgroundMode.TRANSPARENT

    def test_custom_values(self):
        config = ProcessingConfig(
            model_name="BiRefNet-lite",
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        assert config.model_name == "BiRefNet-lite"
        assert config.output_format == OutputFormat.MP4_H264
        assert config.background_mode == BackgroundMode.GREEN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.config'`

- [ ] **Step 3: Write the implementation**

Create `src/core/config.py`:

```python
from dataclasses import dataclass
from enum import Enum


class OutputFormat(Enum):
    MOV_PRORES = "mov_prores"
    WEBM_VP9 = "webm_vp9"
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    MP4_AV1 = "mp4_av1"
    PNG_SEQUENCE = "png_sequence"
    TIFF_SEQUENCE = "tiff_sequence"

    @property
    def supports_alpha(self) -> bool:
        return self in {
            OutputFormat.MOV_PRORES,
            OutputFormat.WEBM_VP9,
            OutputFormat.PNG_SEQUENCE,
            OutputFormat.TIFF_SEQUENCE,
        }


class BackgroundMode(Enum):
    TRANSPARENT = "transparent"
    GREEN = "green"
    BLUE = "blue"
    MASK_BW = "mask_bw"
    MASK_WB = "mask_wb"
    SIDE_BY_SIDE = "side_by_side"

    @property
    def needs_alpha(self) -> bool:
        return self == BackgroundMode.TRANSPARENT


MODELS = {
    "BiRefNet-general": "birefnet-general",
    "BiRefNet-lite": "birefnet-lite",
    "BiRefNet-matting": "birefnet-matting",
    "BiRefNet-HR": "birefnet-hr",
    "BiRefNet-HR-matting": "birefnet-hr-matting",
    "BiRefNet-dynamic": "birefnet-dynamic",
}


@dataclass
class ProcessingConfig:
    model_name: str = "BiRefNet-general"
    output_format: OutputFormat = OutputFormat.MOV_PRORES
    background_mode: BackgroundMode = BackgroundMode.TRANSPARENT
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add ProcessingConfig dataclass with OutputFormat and BackgroundMode enums"
```

---

## Task 2: Background Compositing

**Files:**
- Create: `src/core/compositing.py`
- Create: `tests/test_compositing.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compositing.py`:

```python
import numpy as np
import pytest

from src.core.compositing import compose_frame
from src.core.config import BackgroundMode


@pytest.fixture
def sample_frame():
    """A 4x4 BGR frame with known pixel values."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # B
    frame[:, :, 1] = 150  # G
    frame[:, :, 2] = 200  # R
    return frame


@pytest.fixture
def sample_alpha():
    """A 4x4 alpha mask: top half opaque (255), bottom half transparent (0)."""
    alpha = np.zeros((4, 4), dtype=np.uint8)
    alpha[:2, :] = 255  # top half opaque
    return alpha


class TestComposeTransparent:
    def test_output_shape_is_rgba(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        assert result.shape == (4, 4, 4)
        assert result.dtype == np.uint8

    def test_rgb_channels_are_bgr_to_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        # R channel (index 0 in output) should be 200 (was index 2 in BGR)
        assert result[0, 0, 0] == 200
        # G channel (index 1) should be 150
        assert result[0, 0, 1] == 150
        # B channel (index 2) should be 100
        assert result[0, 0, 2] == 100

    def test_alpha_channel_preserved(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.TRANSPARENT)
        assert result[0, 0, 3] == 255  # top half opaque
        assert result[3, 0, 3] == 0    # bottom half transparent


class TestComposeGreen:
    def test_output_shape_is_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        assert result.shape == (4, 4, 3)
        assert result.dtype == np.uint8

    def test_opaque_pixels_show_foreground(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        # Top half is fully opaque: should be original RGB (200, 150, 100)
        assert result[0, 0, 0] == 200  # R
        assert result[0, 0, 1] == 150  # G
        assert result[0, 0, 2] == 100  # B

    def test_transparent_pixels_show_green(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.GREEN)
        # Bottom half is fully transparent: should be green (0, 255, 0)
        assert result[3, 0, 0] == 0    # R
        assert result[3, 0, 1] == 255  # G
        assert result[3, 0, 2] == 0    # B


class TestComposeBlue:
    def test_transparent_pixels_show_blue(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.BLUE)
        assert result.shape == (4, 4, 3)
        # Bottom half: blue (0, 0, 255)
        assert result[3, 0, 0] == 0
        assert result[3, 0, 1] == 0
        assert result[3, 0, 2] == 255


class TestComposeMaskBW:
    def test_output_is_grayscale_as_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.MASK_BW)
        assert result.shape == (4, 4, 3)
        # Top half (alpha=255) should be white
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 255
        assert result[0, 0, 2] == 255
        # Bottom half (alpha=0) should be black
        assert result[3, 0, 0] == 0
        assert result[3, 0, 1] == 0
        assert result[3, 0, 2] == 0


class TestComposeMaskWB:
    def test_inverted_mask(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.MASK_WB)
        assert result.shape == (4, 4, 3)
        # Top half (alpha=255) should be black (inverted)
        assert result[0, 0, 0] == 0
        # Bottom half (alpha=0) should be white (inverted)
        assert result[3, 0, 0] == 255


class TestComposeSideBySide:
    def test_output_width_doubled(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        assert result.shape == (4, 8, 3)  # width doubled
        assert result.dtype == np.uint8

    def test_left_half_is_original_rgb(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        # Left half: original BGR converted to RGB
        assert result[0, 0, 0] == 200  # R
        assert result[0, 0, 1] == 150  # G
        assert result[0, 0, 2] == 100  # B

    def test_right_half_is_mask(self, sample_frame, sample_alpha):
        result = compose_frame(sample_frame, sample_alpha, BackgroundMode.SIDE_BY_SIDE)
        # Right half: mask as grayscale RGB
        assert result[0, 4, 0] == 255  # top half opaque
        assert result[3, 4, 0] == 0    # bottom half transparent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.compositing'`

- [ ] **Step 3: Write the implementation**

Create `src/core/compositing.py`:

```python
import numpy as np

from src.core.config import BackgroundMode


def compose_frame(
    bgr_frame: np.ndarray,
    alpha_mask: np.ndarray,
    mode: BackgroundMode,
) -> np.ndarray:
    """Compose a BGR frame with an alpha mask according to the background mode.

    Args:
        bgr_frame: BGR uint8 array, shape (H, W, 3).
        alpha_mask: uint8 array, shape (H, W), values 0-255.
        mode: How to compose the output.

    Returns:
        Composed frame as uint8 array.
        - TRANSPARENT: RGBA shape (H, W, 4)
        - GREEN/BLUE: RGB shape (H, W, 3)
        - MASK_BW/MASK_WB: RGB shape (H, W, 3)
        - SIDE_BY_SIDE: RGB shape (H, W*2, 3)
    """
    rgb = bgr_frame[:, :, ::-1]  # BGR -> RGB

    if mode == BackgroundMode.TRANSPARENT:
        return np.dstack([rgb, alpha_mask])

    if mode in (BackgroundMode.GREEN, BackgroundMode.BLUE):
        bg_color = np.array([0, 255, 0], dtype=np.uint8) if mode == BackgroundMode.GREEN \
            else np.array([0, 0, 255], dtype=np.uint8)
        alpha_f = alpha_mask.astype(np.float32) / 255.0
        alpha_3 = alpha_f[:, :, np.newaxis]
        bg = np.full_like(rgb, bg_color)
        blended = (rgb.astype(np.float32) * alpha_3 + bg.astype(np.float32) * (1.0 - alpha_3))
        return blended.clip(0, 255).astype(np.uint8)

    if mode == BackgroundMode.MASK_BW:
        return np.dstack([alpha_mask, alpha_mask, alpha_mask])

    if mode == BackgroundMode.MASK_WB:
        inverted = 255 - alpha_mask
        return np.dstack([inverted, inverted, inverted])

    if mode == BackgroundMode.SIDE_BY_SIDE:
        mask_rgb = np.dstack([alpha_mask, alpha_mask, alpha_mask])
        return np.hstack([rgb, mask_rgb])

    raise ValueError(f"Unknown background mode: {mode}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_compositing.py -v`
Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/compositing.py tests/test_compositing.py
git commit -m "feat: add compose_frame with 6 background modes"
```

---

## Task 3: Writer Factory — FFmpegWriter

**Files:**
- Create: `src/core/writer.py`
- Create: `tests/test_writer.py`

This task adds `FFmpegWriter` and the `create_writer` factory. Image sequence writers are in the next task.

- [ ] **Step 1: Write the failing tests for FFmpegWriter**

Create `tests/test_writer.py`:

```python
import os

import numpy as np
import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import get_video_info
from src.core.writer import FFmpegWriter, create_writer


class TestFFmpegWriter:
    def test_writes_mp4_h264(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libx264",
            pix_fmt="yuv420p",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["frame_count"] == 5

    def test_writes_webm_vp9_with_alpha(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.webm")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libvpx-vp9",
            pix_fmt="yuva420p",
            input_pix_fmt="rgba",
            extra_args=["-auto-alt-ref", "0"],
        )
        for _ in range(5):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_context_manager(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        with FFmpegWriter(output_path, 64, 64, 30.0, "libx264", "yuv420p") as writer:
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        assert os.path.exists(output_path)

    def test_wrong_frame_shape_raises(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = FFmpegWriter(output_path, 64, 64, 30.0, "libx264", "yuv420p")
        with pytest.raises(ValueError):
            wrong_shape = np.full((32, 32, 3), 128, dtype=np.uint8)
            writer.write_frame(wrong_shape)
        writer.close()


class TestCreateWriter:
    def test_prores_returns_prores_writer(self, temp_output_dir):
        from src.core.video import ProResWriter
        config = ProcessingConfig(output_format=OutputFormat.MOV_PRORES)
        output_path = os.path.join(temp_output_dir, "test.mov")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, ProResWriter)
        writer.close()

    def test_h264_returns_ffmpeg_writer(self, temp_output_dir):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()

    def test_webm_returns_ffmpeg_writer(self, temp_output_dir):
        config = ProcessingConfig(output_format=OutputFormat.WEBM_VP9)
        output_path = os.path.join(temp_output_dir, "test.webm")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_writer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.core.writer'`

- [ ] **Step 3: Write the implementation**

Create `src/core/writer.py`:

```python
import subprocess

import numpy as np

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import ProResWriter


class FFmpegWriter:
    """Writes video frames via FFmpeg subprocess. Supports various codecs."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str,
        pix_fmt: str,
        input_pix_fmt: str = "rgb24",
        extra_args: list[str] | None = None,
    ):
        self._width = width
        self._height = height
        self._channels = 4 if input_pix_fmt == "rgba" else 3

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", codec,
            "-pix_fmt", pix_fmt,
        ]
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(output_path)

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write_frame(self, frame: np.ndarray):
        expected = (self._height, self._width, self._channels)
        if frame.shape != expected:
            raise ValueError(f"Expected frame shape {expected}, got {frame.shape}")
        self._process.stdin.write(frame.tobytes())

    def close(self):
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
            self._process.stdin.close()
        _, stderr_data = self._process.communicate()
        if self._process.returncode != 0:
            stderr = stderr_data.decode() if stderr_data else "unknown error"
            raise RuntimeError(f"FFmpeg failed (code {self._process.returncode}): {stderr}")


def create_writer(
    config: ProcessingConfig,
    output_path: str,
    width: int,
    height: int,
    fps: float,
):
    """Factory: return the appropriate writer based on config."""
    fmt = config.output_format
    is_alpha = config.background_mode.needs_alpha

    # Side-by-side doubles width
    if config.background_mode == BackgroundMode.SIDE_BY_SIDE:
        width = width * 2

    if fmt == OutputFormat.MOV_PRORES:
        return ProResWriter(output_path, width, height, fps)

    if fmt == OutputFormat.WEBM_VP9:
        if is_alpha:
            return FFmpegWriter(
                output_path, width, height, fps,
                codec="libvpx-vp9",
                pix_fmt="yuva420p",
                input_pix_fmt="rgba",
                extra_args=["-auto-alt-ref", "0"],
            )
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libvpx-vp9",
            pix_fmt="yuv420p",
            extra_args=["-auto-alt-ref", "0"],
        )

    if fmt == OutputFormat.MP4_H264:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx264",
            pix_fmt="yuv420p",
        )

    if fmt == OutputFormat.MP4_H265:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx265",
            pix_fmt="yuv420p",
            extra_args=["-tag:v", "hvc1"],
        )

    if fmt == OutputFormat.MP4_AV1:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libaom-av1",
            pix_fmt="yuv420p",
            extra_args=["-cpu-used", "8", "-row-mt", "1"],
        )

    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        from src.core.writer import ImageSequenceWriter
        return ImageSequenceWriter(output_path, fmt, is_alpha)

    raise ValueError(f"Unsupported output format: {fmt}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: All 7 tests PASS (ImageSequenceWriter import will be deferred — tests don't use it yet)

- [ ] **Step 5: Commit**

```bash
git add src/core/writer.py tests/test_writer.py
git commit -m "feat: add FFmpegWriter and create_writer factory for multi-format output"
```

---

## Task 4: Writer Factory — ImageSequenceWriter

**Files:**
- Modify: `src/core/writer.py` (add ImageSequenceWriter class)
- Modify: `tests/test_writer.py` (add ImageSequenceWriter tests)
- Modify: `tests/conftest.py` (add temp_output_subdir fixture)

- [ ] **Step 1: Add fixture to conftest.py**

Add to `tests/conftest.py` after the existing `temp_output_dir` fixture:

```python
@pytest.fixture
def temp_output_subdir(temp_output_dir):
    """Provide a subdirectory inside temp_output_dir for sequence output."""
    subdir = os.path.join(temp_output_dir, "sequence")
    yield subdir
    # Cleanup handled by temp_output_dir
```

Note: also update `temp_output_dir` teardown to handle subdirectories. Replace the existing `temp_output_dir` fixture with:

```python
@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
```

- [ ] **Step 2: Write the failing tests**

Add to `tests/test_writer.py`:

```python
from src.core.writer import ImageSequenceWriter


class TestImageSequenceWriter:
    def test_writes_png_sequence(self, temp_output_subdir):
        writer = ImageSequenceWriter(temp_output_subdir, OutputFormat.PNG_SEQUENCE, has_alpha=True)
        for _ in range(3):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(temp_output_subdir)
        files = sorted(os.listdir(temp_output_subdir))
        assert len(files) == 3
        assert files[0] == "frame_000001.png"
        assert files[2] == "frame_000003.png"

    def test_writes_tiff_sequence(self, temp_output_subdir):
        writer = ImageSequenceWriter(temp_output_subdir, OutputFormat.TIFF_SEQUENCE, has_alpha=False)
        for _ in range(2):
            frame = np.full((64, 64, 3), 200, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        files = sorted(os.listdir(temp_output_subdir))
        assert len(files) == 2
        assert files[0] == "frame_000001.tiff"

    def test_context_manager(self, temp_output_subdir):
        with ImageSequenceWriter(temp_output_subdir, OutputFormat.PNG_SEQUENCE, has_alpha=False) as writer:
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        assert os.path.exists(os.path.join(temp_output_subdir, "frame_000001.png"))


class TestCreateWriterSequence:
    def test_png_sequence_returns_image_writer(self, temp_output_subdir):
        config = ProcessingConfig(output_format=OutputFormat.PNG_SEQUENCE)
        writer = create_writer(config, temp_output_subdir, 64, 64, 30.0)
        assert isinstance(writer, ImageSequenceWriter)
        writer.close()

    def test_tiff_sequence_returns_image_writer(self, temp_output_subdir):
        config = ProcessingConfig(
            output_format=OutputFormat.TIFF_SEQUENCE,
            background_mode=BackgroundMode.MASK_BW,
        )
        writer = create_writer(config, temp_output_subdir, 64, 64, 30.0)
        assert isinstance(writer, ImageSequenceWriter)
        writer.close()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_writer.py::TestImageSequenceWriter -v`
Expected: FAIL with `ImportError: cannot import name 'ImageSequenceWriter'`

- [ ] **Step 4: Write the implementation**

Add to `src/core/writer.py` (before the `create_writer` function):

```python
import os

from PIL import Image


class ImageSequenceWriter:
    """Writes frames as individual image files (PNG or TIFF)."""

    def __init__(self, output_dir: str, fmt: OutputFormat, has_alpha: bool):
        self._output_dir = output_dir
        self._ext = "png" if fmt == OutputFormat.PNG_SEQUENCE else "tiff"
        self._has_alpha = has_alpha
        self._frame_num = 0
        os.makedirs(output_dir, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write_frame(self, frame: np.ndarray):
        self._frame_num += 1
        filename = f"frame_{self._frame_num:06d}.{self._ext}"
        filepath = os.path.join(self._output_dir, filename)

        if self._has_alpha and frame.shape[2] == 4:
            img = Image.fromarray(frame, "RGBA")
        else:
            img = Image.fromarray(frame, "RGB")
        img.save(filepath)

    def close(self):
        pass  # No resources to release
```

Also update the `create_writer` function to fix the circular import. Replace the lazy import line:

```python
    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        return ImageSequenceWriter(output_path, fmt, is_alpha)
```

And add `import os` and `from PIL import Image` to the top of `src/core/writer.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_writer.py -v`
Expected: All 12 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/core/writer.py tests/test_writer.py tests/conftest.py
git commit -m "feat: add ImageSequenceWriter for PNG/TIFF sequence output"
```

---

## Task 5: get_model_path helper in inference.py

**Files:**
- Modify: `src/core/inference.py:1-2` (add import and function)
- Modify: `tests/test_inference.py` (add test class)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_inference.py` after the existing imports (line 9):

```python
from src.core.inference import detect_device, get_model_path, load_model, predict
```

And add this test class at the end of the file:

```python
class TestGetModelPath:
    def test_returns_correct_path(self, tmp_path):
        models_dir = str(tmp_path)
        path = get_model_path("BiRefNet-general", models_dir)
        assert path == os.path.join(str(tmp_path), "birefnet-general")

    def test_unknown_model_raises(self, tmp_path):
        with pytest.raises(KeyError):
            get_model_path("NonExistentModel", str(tmp_path))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference.py::TestGetModelPath -v`
Expected: FAIL with `ImportError: cannot import name 'get_model_path'`

- [ ] **Step 3: Write the implementation**

Add to `src/core/inference.py` after the existing imports (after line 7), before `detect_device`:

```python
from src.core.config import MODELS


def get_model_path(model_name: str, models_dir: str) -> str:
    """Map a model display name to its local directory path."""
    dir_name = MODELS[model_name]
    return os.path.join(models_dir, dir_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference.py::TestGetModelPath -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Run all existing inference tests still pass**

Run: `python -m pytest tests/test_inference.py -v`
Expected: All tests PASS (existing tests unchanged)

- [ ] **Step 6: Commit**

```bash
git add src/core/inference.py tests/test_inference.py
git commit -m "feat: add get_model_path helper to map model names to directories"
```

---

## Task 6: Refactor Pipeline to use ProcessingConfig

**Files:**
- Modify: `src/core/pipeline.py` (entire file)
- Modify: `tests/test_pipeline.py` (update to use ProcessingConfig)

- [ ] **Step 1: Update the test file**

Replace the entire `tests/test_pipeline.py` with:

```python
import os
import threading

import numpy as np
import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import get_video_info

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.isdir(os.path.join(MODELS_DIR, "birefnet-general"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_prores_transparent(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig()  # defaults: general, prores, transparent
        output_path = os.path.join(temp_output_dir, "output.mov")
        progress_log = []

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(
            input_path=test_video_path,
            output_path=output_path,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_processes_video_h264_green(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "output.mp4")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64

    def test_processes_video_mask_bw(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.MASK_BW,
        )
        output_path = os.path.join(temp_output_dir, "output.mp4")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        assert os.path.exists(output_path)

    def test_processes_png_sequence(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.PNG_SEQUENCE,
            background_mode=BackgroundMode.TRANSPARENT,
        )
        output_path = os.path.join(temp_output_dir, "seq_output")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        files = sorted(os.listdir(output_path))
        assert len(files) == 10
        assert files[0] == "frame_000001.png"

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig()
        output_path = os.path.join(temp_output_dir, "output.mov")
        cancel_event = threading.Event()
        frame_count = []

        def on_progress(current, total):
            frame_count.append(current)
            if current >= 3:
                cancel_event.set()

        pipeline = MattingPipeline(config, MODELS_DIR)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        assert len(frame_count) < 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: FAIL because `MattingPipeline` still takes `model_path` not `config`

- [ ] **Step 3: Rewrite pipeline.py**

Replace the entire `src/core/pipeline.py` with:

```python
import os
import threading
import time
from typing import Callable, Optional

from src.core.compositing import compose_frame
from src.core.config import ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict
from src.core.video import FrameReader, get_video_info
from src.core.writer import create_writer


class MattingPipeline:
    """Orchestrates video read -> BiRefNet inference -> compositing -> write."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Process a video file with the configured model, format, and background mode.

        Args:
            input_path: Path to input video file.
            output_path: Path for output file or directory (for image sequences).
            progress_callback: Called with (current_frame, total_frames) after each frame.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.
        """
        video_info = get_video_info(input_path)
        total_frames = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]

        writer = create_writer(self._config, output_path, width, height, fps)

        with writer:
            for frame_idx, frame in enumerate(FrameReader(input_path), start=1):
                # Check cancel
                if cancel_event and cancel_event.is_set():
                    break

                # Check pause
                if pause_event:
                    while pause_event.is_set():
                        if cancel_event and cancel_event.is_set():
                            break
                        time.sleep(0.1)
                    if cancel_event and cancel_event.is_set():
                        break

                # Inference
                alpha = predict(self._model, frame, self._device)

                # Compose output frame
                composed = compose_frame(frame, alpha, self._config.background_mode)
                writer.write_frame(composed)

                # Report progress
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        # Handle cancel cleanup
        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All 5 tests PASS (or skip if model not downloaded)

- [ ] **Step 5: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "refactor: pipeline accepts ProcessingConfig, uses compose_frame and create_writer"
```

---

## Task 7: Update MattingWorker

**Files:**
- Modify: `src/worker/matting_worker.py` (accept config + models_dir)

- [ ] **Step 1: Rewrite matting_worker.py**

Replace the entire `src/worker/matting_worker.py` with:

```python
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.config import ProcessingConfig
from src.core.pipeline import MattingPipeline


class MattingWorker(QThread):
    """Runs the matting pipeline in a background thread.

    Signals:
        progress(int, int): (current_frame, total_frames)
        speed(float): frames per second
        finished(str): output file path on success
        error(str): error message on failure
    """

    progress = pyqtSignal(int, int)
    speed = pyqtSignal(float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config: ProcessingConfig, models_dir: str, input_path: str, output_path: str):
        super().__init__()
        self._config = config
        self._models_dir = models_dir
        self._input_path = input_path
        self._output_path = output_path

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._last_time = None

    def run(self):
        try:
            pipeline = MattingPipeline(self._config, self._models_dir)
            self._last_time = time.time()
            pipeline.process(
                input_path=self._input_path,
                output_path=self._output_path,
                progress_callback=self._on_progress,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
            )
            self.finished.emit(self._output_path)
        except InterruptedError:
            self.error.emit("Processing cancelled")
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total)
        now = time.time()
        elapsed = now - self._last_time
        if elapsed > 0:
            self.speed.emit(1.0 / elapsed)
        self._last_time = now

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def cancel(self):
        self._cancel_event.set()
```

- [ ] **Step 2: Run existing tests still pass**

Run: `python -m pytest tests/ -v --ignore=tests/test_pipeline.py --ignore=tests/test_inference.py -k "not Pipeline"`
Expected: PASS (worker is tested indirectly through pipeline tests)

- [ ] **Step 3: Commit**

```bash
git add src/worker/matting_worker.py
git commit -m "refactor: MattingWorker accepts ProcessingConfig instead of model_path"
```

---

## Task 8: GUI Left-Right Layout with ComboBoxes

**Files:**
- Modify: `src/gui/main_window.py` (entire file rewrite)

- [ ] **Step 1: Rewrite main_window.py**

Replace the entire `src/gui/main_window.py` with:

```python
import os
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import BackgroundMode, MODELS, OutputFormat, ProcessingConfig
from src.core.inference import detect_device
from src.core.video import get_video_info
from src.worker.matting_worker import MattingWorker

# Path to bundled models directory (relative to project root)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Display names for output formats
FORMAT_LABELS = {
    OutputFormat.MOV_PRORES: "MOV ProRes 4444",
    OutputFormat.WEBM_VP9: "WebM VP9",
    OutputFormat.MP4_H264: "MP4 H.264",
    OutputFormat.MP4_H265: "MP4 H.265/HEVC",
    OutputFormat.MP4_AV1: "MP4 AV1",
    OutputFormat.PNG_SEQUENCE: "PNG 序列",
    OutputFormat.TIFF_SEQUENCE: "TIFF 序列",
}

# Display names for background modes
MODE_LABELS = {
    BackgroundMode.TRANSPARENT: "透明背景",
    BackgroundMode.GREEN: "绿幕",
    BackgroundMode.BLUE: "蓝幕",
    BackgroundMode.MASK_BW: "黑底白蒙版",
    BackgroundMode.MASK_WB: "白底黑蒙版",
    BackgroundMode.SIDE_BY_SIDE: "原图+蒙版分轨",
}

# File extension mapping
FORMAT_EXTENSIONS = {
    OutputFormat.MOV_PRORES: ".mov",
    OutputFormat.WEBM_VP9: ".webm",
    OutputFormat.MP4_H264: ".mp4",
    OutputFormat.MP4_H265: ".mp4",
    OutputFormat.MP4_AV1: ".mp4",
    OutputFormat.PNG_SEQUENCE: "",  # directory, no extension
    OutputFormat.TIFF_SEQUENCE: "",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(750, 500)

        self._worker = None
        self._input_path = None
        self._output_dir = None
        self._start_time = None

        self._init_ui()
        self._set_state("initial")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Left panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # Input file
        left_panel.addWidget(QLabel("输入文件:"))
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setReadOnly(True)
        self._input_edit.setPlaceholderText("未选择文件")
        input_row.addWidget(self._input_edit)
        self._select_btn = QPushButton("选择文件")
        self._select_btn.clicked.connect(self._on_select_file)
        input_row.addWidget(self._select_btn)
        left_panel.addLayout(input_row)

        # Video info
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._info_label)

        # Separator
        sep1 = QLabel()
        sep1.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        left_panel.addWidget(sep1)

        # Output path
        left_panel.addWidget(QLabel("输出路径:"))
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("与输入文件同目录")
        output_row.addWidget(self._output_edit)
        self._output_btn = QPushButton("浏览...")
        self._output_btn.clicked.connect(self._on_select_output)
        output_row.addWidget(self._output_btn)
        left_panel.addLayout(output_row)

        # Separator
        sep2 = QLabel()
        sep2.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        left_panel.addWidget(sep2)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        left_panel.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._status_label)

        # Control buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        left_panel.addLayout(btn_row)

        left_panel.addStretch()

        # --- Right panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Model selection
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型:"))
        self._model_combo = QComboBox()
        self._populate_model_combo()
        model_layout.addWidget(self._model_combo)

        device = detect_device()
        device_text = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
        self._device_label = QLabel(f"设备: {device_text.get(device, device)}")
        self._device_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self._device_label)

        right_panel.addWidget(model_group)

        # Output settings
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)

        output_layout.addWidget(QLabel("格式:"))
        self._format_combo = QComboBox()
        for fmt, label in FORMAT_LABELS.items():
            self._format_combo.addItem(label, fmt)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        output_layout.addWidget(self._format_combo)

        output_layout.addWidget(QLabel("背景:"))
        self._mode_combo = QComboBox()
        self._populate_mode_combo()
        output_layout.addWidget(self._mode_combo)

        right_panel.addWidget(output_group)

        right_panel.addStretch()

        # --- Assemble ---
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addLayout(right_panel, stretch=1)

    def _populate_model_combo(self):
        """Populate model dropdown. Only show downloaded models as enabled."""
        models_dir = os.path.abspath(MODELS_DIR)
        for display_name, dir_name in MODELS.items():
            model_path = os.path.join(models_dir, dir_name)
            if os.path.isdir(model_path):
                self._model_combo.addItem(display_name, display_name)
            else:
                self._model_combo.addItem(f"{display_name} (未下载)", display_name)
                idx = self._model_combo.count() - 1
                # Disable undownloaded models
                self._model_combo.model().item(idx).setEnabled(False)

    def _populate_mode_combo(self):
        """Populate background mode dropdown based on current format."""
        self._mode_combo.clear()
        current_format = self._format_combo.currentData()
        for mode, label in MODE_LABELS.items():
            if mode.needs_alpha and current_format and not current_format.supports_alpha:
                continue  # Skip transparent for formats that don't support alpha
            self._mode_combo.addItem(label, mode)

    def _on_format_changed(self, _index):
        """When format changes, update available background modes."""
        self._populate_mode_combo()

    def _get_config(self) -> ProcessingConfig:
        """Build ProcessingConfig from current UI selections."""
        return ProcessingConfig(
            model_name=self._model_combo.currentData(),
            output_format=self._format_combo.currentData(),
            background_mode=self._mode_combo.currentData(),
        )

    def _set_state(self, state: str):
        self._state = state
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)

    def _on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if not path:
            return

        try:
            info = get_video_info(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取视频文件:\n{e}")
            return

        self._input_path = path
        self._input_edit.setText(path)

        w, h = info["width"], info["height"]
        fps = info["fps"]
        frames = info["frame_count"]
        dur = info["duration"]
        minutes = int(dur // 60)
        seconds = int(dur % 60)
        self._info_label.setText(
            f"视频信息: {w}x{h} | {fps:.1f}fps | {frames}帧 | {minutes:02d}:{seconds:02d}"
        )

        self._set_state("ready")

    def _on_select_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self._output_dir = directory
            self._output_edit.setText(directory)

    def _build_output_path(self) -> str:
        config = self._get_config()
        base_name = os.path.splitext(os.path.basename(self._input_path))[0]
        model_dir_name = MODELS[config.model_name]
        timestamp = int(time.time() * 1000)
        ext = FORMAT_EXTENSIONS[config.output_format]

        if ext:
            # Video file output
            filename = f"{base_name}_{model_dir_name}_{timestamp}{ext}"
        else:
            # Image sequence: output is a directory
            filename = f"{base_name}_{model_dir_name}_{timestamp}"

        if self._output_dir:
            return os.path.join(self._output_dir, filename)
        else:
            return os.path.join(os.path.dirname(self._input_path), filename)

    def _on_start(self):
        if not self._input_path:
            return

        config = self._get_config()
        models_dir = os.path.abspath(MODELS_DIR)

        # Check model exists
        model_dir_name = MODELS[config.model_name]
        model_path = os.path.join(models_dir, model_dir_name)
        if not os.path.isdir(model_path):
            QMessageBox.critical(
                self,
                "模型缺失",
                f"未找到 {config.model_name} 模型:\n{model_path}\n\n"
                "请运行 python download_models.py 下载模型。",
            )
            return

        output_path = self._build_output_path()
        self._start_time = time.time()
        self._set_state("processing")
        self._status_label.setText("正在加载模型...")

        self._worker = MattingWorker(config, models_dir, self._input_path, output_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.speed.connect(self._on_speed)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pause(self):
        if not self._worker:
            return
        if self._state == "processing":
            self._worker.pause()
            self._set_state("paused")
        elif self._state == "paused":
            self._worker.resume()
            self._set_state("processing")

    def _on_cancel(self):
        if not self._worker:
            return
        self._worker.cancel()
        self._worker.wait()
        self._set_state("ready")
        self._progress_bar.setValue(0)
        self._status_label.setText("已取消")

    def _on_progress(self, current: int, total: int):
        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            self._status_label.setText(
                f"帧: {current}/{total} | 速度: {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )

    def _on_speed(self, fps: float):
        pass

    def _on_finished(self, output_path: str):
        self._set_state("finished")
        self._progress_bar.setValue(100)
        elapsed = time.time() - self._start_time if self._start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._status_label.setText(f"处理完成! 耗时: {minutes:02d}:{seconds:02d}")

        QMessageBox.information(
            self,
            "完成",
            f"视频处理完成!\n\n输出文件:\n{output_path}",
        )

    def _on_error(self, message: str):
        self._set_state("ready")
        self._progress_bar.setValue(0)
        if message != "Processing cancelled":
            QMessageBox.critical(self, "错误", f"处理出错:\n{message}")
            self._status_label.setText(f"错误: {message}")
```

- [ ] **Step 2: Manually test the GUI**

Run: `python main.py`

Verify:
1. Window opens with left-right split layout
2. Right panel shows model dropdown (only downloaded models enabled)
3. Format dropdown has all 7 formats
4. Selecting MP4 H.264 removes "透明背景" from mode dropdown
5. Selecting MOV ProRes restores "透明背景"
6. Select a video file, click "开始处理", verify it processes correctly

- [ ] **Step 3: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: left-right split GUI with model/format/mode dropdowns and format-mode interlock"
```

---

## Task 9: Expand download_models.py

**Files:**
- Modify: `download_models.py`

- [ ] **Step 1: Rewrite download_models.py**

Replace the entire `download_models.py` with:

```python
#!/usr/bin/env python3
"""Download BiRefNet models for offline use.

Run: python download_models.py              # interactive selection
     python download_models.py --all        # download all models
     python download_models.py general lite # download specific models
"""

import os
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "general": ("birefnet-general", "zhengpeng7/BiRefNet"),
    "lite": ("birefnet-lite", "zhengpeng7/BiRefNet_lite"),
    "matting": ("birefnet-matting", "zhengpeng7/BiRefNet-matting"),
    "hr": ("birefnet-hr", "zhengpeng7/BiRefNet_HR"),
    "hr-matting": ("birefnet-hr-matting", "zhengpeng7/BiRefNet_HR-matting"),
    "dynamic": ("birefnet-dynamic", "zhengpeng7/BiRefNet_dynamic"),
}


def download_model(key: str, models_dir: str = "./models"):
    dir_name, repo_id = MODELS[key]
    local_path = os.path.join(models_dir, dir_name)
    os.makedirs(local_path, exist_ok=True)

    print(f"\nDownloading {key} from {repo_id}...")
    print(f"  -> {os.path.abspath(local_path)}")

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"  Done: {key}")


def main():
    args = sys.argv[1:]

    if not args:
        # Default: download general only
        print("No arguments. Downloading birefnet-general (default).")
        print("Use --all to download all models, or specify: general lite matting hr hr-matting dynamic")
        download_model("general")
        return

    if "--all" in args:
        keys = list(MODELS.keys())
    else:
        keys = []
        for arg in args:
            if arg not in MODELS:
                print(f"Unknown model: {arg}")
                print(f"Available: {', '.join(MODELS.keys())}")
                sys.exit(1)
            keys.append(arg)

    for key in keys:
        download_model(key)

    print(f"\nAll done. Models saved to {os.path.abspath('./models')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test the script**

Run: `python download_models.py --help` (should print usage since --help is not a model name)
Run: `python download_models.py` (should download general only, already exists so fast)

- [ ] **Step 3: Commit**

```bash
git add download_models.py
git commit -m "feat: expand download_models.py to support all 6 models with CLI selection"
```

---

## Task 10: Run Full Test Suite and Fix Any Issues

**Files:**
- All test files

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`

Expected: All tests PASS. The test counts should be approximately:
- `test_config.py`: 7 tests
- `test_compositing.py`: 12 tests
- `test_writer.py`: 12 tests
- `test_video.py`: 6 tests (unchanged)
- `test_inference.py`: 5 + 2 = 7 tests
- `test_pipeline.py`: 5 tests (model-dependent, may skip)

- [ ] **Step 2: Fix any failures**

If any test fails, fix the issue and re-run.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: verify all P1 core feature tests pass"
```

---

## Summary

| Task | Description | New/Modified Files | Tests |
|------|------------|-------------------|-------|
| 1 | ProcessingConfig + enums | `config.py` | 7 |
| 2 | Background compositing | `compositing.py` | 12 |
| 3 | FFmpegWriter + create_writer | `writer.py` | 7 |
| 4 | ImageSequenceWriter | `writer.py` | 5 |
| 5 | get_model_path helper | `inference.py` | 2 |
| 6 | Pipeline refactor | `pipeline.py` | 5 |
| 7 | MattingWorker update | `matting_worker.py` | — |
| 8 | GUI left-right layout | `main_window.py` | manual |
| 9 | download_models.py expansion | `download_models.py` | manual |
| 10 | Full test suite validation | all | all |

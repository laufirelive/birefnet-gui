# P1 剩余功能实现计划：音频保留、图片输入、拖拽支持

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add audio passthrough for video output, image/folder input support, and drag-and-drop to the BiRefNet GUI.

**Architecture:** Three independent features layered on the existing pipeline. Audio modifies FFmpeg writer commands. Image input adds a parallel `ImagePipeline` alongside the existing `MattingPipeline`. Drag-and-drop hooks into a shared `_handle_input()` method in `MainWindow`.

**Tech Stack:** PyQt6, FFmpeg (audio muxing), PIL/OpenCV (image I/O), pytest

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/core/config.py` | Add `InputType` enum | Modify |
| `src/core/video.py` | Add `audio_source` to `ProResWriter` | Modify |
| `src/core/writer.py` | Add `audio_source` to `FFmpegWriter` and `create_writer()` | Modify |
| `src/core/pipeline.py` | Pass `input_path` as audio source to writer | Modify |
| `src/core/image_pipeline.py` | New `ImagePipeline` for single image / folder processing | Create |
| `src/worker/matting_worker.py` | Branch between video/image pipeline based on `InputType` | Modify |
| `src/gui/main_window.py` | Image file selection, folder selection, drag-and-drop, UI mode switching | Modify |
| `tests/test_audio.py` | Tests for audio passthrough in writers | Create |
| `tests/test_image_pipeline.py` | Tests for `ImagePipeline` | Create |
| `tests/conftest.py` | Add image fixtures, video-with-audio fixture | Modify |

---

## Task 1: Add `InputType` enum to config

**Files:**
- Modify: `src/core/config.py`
- Test: `tests/test_config.py` (new)

- [ ] **Step 1: Write test for InputType enum**

```python
# tests/test_config.py
from src.core.config import InputType


class TestInputType:
    def test_video_type_exists(self):
        assert InputType.VIDEO.value == "video"

    def test_image_type_exists(self):
        assert InputType.IMAGE.value == "image"

    def test_image_folder_type_exists(self):
        assert InputType.IMAGE_FOLDER.value == "image_folder"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: ImportError — `InputType` does not exist yet.

- [ ] **Step 3: Implement InputType enum**

Add to `src/core/config.py` after the `BackgroundMode` class (line 35):

```python
class InputType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_FOLDER = "image_folder"
```

Also add a helper constant for file extension classification:

```python
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add InputType enum and file extension constants"
```

---

## Task 2: Audio passthrough in ProResWriter

**Files:**
- Modify: `src/core/video.py:50-105` (`ProResWriter` class)
- Test: `tests/test_audio.py` (new)
- Modify: `tests/conftest.py` (add audio fixture)

- [ ] **Step 1: Add test video with audio fixture to conftest.py**

Add to `tests/conftest.py`:

```python
@pytest.fixture
def test_video_with_audio_path():
    """Create a 10-frame 64x64 test video WITH a silent audio track."""
    tmpdir = tempfile.mkdtemp()
    raw_path = os.path.join(tmpdir, "raw.mp4")
    path = os.path.join(tmpdir, "test_with_audio.mp4")

    # First create raw video with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, 30.0, (64, 64))
    for i in range(10):
        frame = np.full((64, 64, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Add a silent audio track with FFmpeg
    import subprocess
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-i", raw_path,
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    os.remove(raw_path)

    yield path
    if os.path.exists(path):
        os.remove(path)
    os.rmdir(tmpdir)
```

- [ ] **Step 2: Write tests for audio passthrough**

Create `tests/test_audio.py`:

```python
import json
import os
import subprocess

import numpy as np
import pytest

from src.core.video import ProResWriter


def _has_audio_stream(filepath: str) -> bool:
    """Check if a file has an audio stream using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            filepath,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    info = json.loads(result.stdout)
    return any(s["codec_type"] == "audio" for s in info.get("streams", []))


class TestProResWriterAudio:
    def test_no_audio_source_produces_silent_output(self, temp_output_dir):
        """Without audio_source, output has no audio track."""
        output_path = os.path.join(temp_output_dir, "no_audio.mov")
        writer = ProResWriter(output_path, width=64, height=64, fps=30.0)
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)
        assert not _has_audio_stream(output_path)

    def test_audio_source_copies_audio_track(
        self, test_video_with_audio_path, temp_output_dir
    ):
        """With audio_source, output has an audio track."""
        output_path = os.path.join(temp_output_dir, "with_audio.mov")
        writer = ProResWriter(
            output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path,
        )
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)
        assert _has_audio_stream(output_path)

    def test_audio_source_without_audio_stream_still_works(
        self, test_video_path, temp_output_dir
    ):
        """If audio_source has no audio track, output is still valid (no crash)."""
        output_path = os.path.join(temp_output_dir, "no_audio_src.mov")
        writer = ProResWriter(
            output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_path,  # test_video_path has no audio
        )
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py -v`
Expected: FAIL — `ProResWriter.__init__()` got unexpected keyword argument `audio_source`.

- [ ] **Step 4: Implement audio_source in ProResWriter**

Replace `ProResWriter.__init__` in `src/core/video.py` (lines 53-75):

```python
class ProResWriter:
    """Writes RGBA frames to a MOV file using ProRes 4444 via FFmpeg."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: str | None = None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend([
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-vendor", "apl0",
        ])

        if audio_source:
            cmd.extend(["-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"])

        cmd.append(output_path)

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
```

The rest of `ProResWriter` (`__enter__`, `__exit__`, `write_frame`, `close`) stays unchanged.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: 3 tests PASS.

- [ ] **Step 6: Run existing tests to ensure no regression**

Run: `python -m pytest tests/test_video.py -v`
Expected: All 6 tests PASS (ProResWriter default behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/core/video.py tests/test_audio.py tests/conftest.py
git commit -m "feat: add audio passthrough to ProResWriter"
```

---

## Task 3: Audio passthrough in FFmpegWriter and create_writer

**Files:**
- Modify: `src/core/writer.py:12-73` (`FFmpegWriter` class) and `src/core/writer.py:108-168` (`create_writer`)
- Test: `tests/test_audio.py` (extend)

- [ ] **Step 1: Add tests for FFmpegWriter audio passthrough**

Append to `tests/test_audio.py`:

```python
from src.core.writer import FFmpegWriter, create_writer
from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig


class TestFFmpegWriterAudio:
    def test_h264_with_audio_source(self, test_video_with_audio_path, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "with_audio.mp4")
        writer = FFmpegWriter(
            output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p",
            audio_source=test_video_with_audio_path,
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert _has_audio_stream(output_path)

    def test_h264_without_audio_source(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "no_audio.mp4")
        writer = FFmpegWriter(
            output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert not _has_audio_stream(output_path)


class TestCreateWriterAudio:
    def test_create_writer_passes_audio_source(
        self, test_video_with_audio_path, temp_output_dir
    ):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "out.mp4")
        writer = create_writer(
            config, output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path,
        )
        with writer:
            for _ in range(5):
                frame = np.full((64, 64, 3), 128, dtype=np.uint8)
                writer.write_frame(frame)

        assert _has_audio_stream(output_path)

    def test_image_sequence_ignores_audio_source(
        self, test_video_with_audio_path, temp_output_dir
    ):
        config = ProcessingConfig(
            output_format=OutputFormat.PNG_SEQUENCE,
            background_mode=BackgroundMode.TRANSPARENT,
        )
        output_path = os.path.join(temp_output_dir, "seq")
        writer = create_writer(
            config, output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path,
        )
        with writer:
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)

        # Should produce image files without error
        assert os.path.isdir(output_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py::TestFFmpegWriterAudio -v`
Expected: FAIL — `FFmpegWriter.__init__()` got unexpected keyword argument `audio_source`.

- [ ] **Step 3: Implement audio_source in FFmpegWriter**

Replace `FFmpegWriter.__init__` in `src/core/writer.py` (lines 14-48):

```python
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
        audio_source: str | None = None,
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
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend(["-c:v", codec, "-pix_fmt", pix_fmt])

        if extra_args:
            cmd.extend(extra_args)

        if audio_source:
            cmd.extend(["-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"])

        cmd.append(output_path)

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
```

The rest of `FFmpegWriter` (`__enter__`, `__exit__`, `write_frame`, `close`) stays unchanged.

- [ ] **Step 4: Update create_writer to accept and pass audio_source**

Replace `create_writer` in `src/core/writer.py` (lines 108-168):

```python
def create_writer(
    config: ProcessingConfig,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
):
    """Factory: return the appropriate writer based on config."""
    fmt = config.output_format
    is_alpha = config.background_mode.needs_alpha

    # Side-by-side doubles width
    if config.background_mode == BackgroundMode.SIDE_BY_SIDE:
        width = width * 2

    if fmt == OutputFormat.MOV_PRORES:
        return ProResWriter(output_path, width, height, fps, audio_source=audio_source)

    if fmt == OutputFormat.WEBM_VP9:
        if is_alpha:
            return FFmpegWriter(
                output_path, width, height, fps,
                codec="libvpx-vp9",
                pix_fmt="yuva420p",
                input_pix_fmt="rgba",
                extra_args=["-auto-alt-ref", "0"],
                audio_source=audio_source,
            )
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libvpx-vp9",
            pix_fmt="yuv420p",
            extra_args=["-auto-alt-ref", "0"],
            audio_source=audio_source,
        )

    if fmt == OutputFormat.MP4_H264:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx264",
            pix_fmt="yuv420p",
            audio_source=audio_source,
        )

    if fmt == OutputFormat.MP4_H265:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx265",
            pix_fmt="yuv420p",
            extra_args=["-tag:v", "hvc1"],
            audio_source=audio_source,
        )

    if fmt == OutputFormat.MP4_AV1:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libaom-av1",
            pix_fmt="yuv420p",
            extra_args=["-cpu-used", "8", "-row-mt", "1"],
            audio_source=audio_source,
        )

    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        return ImageSequenceWriter(output_path, fmt, is_alpha)

    raise ValueError(f"Unsupported output format: {fmt}")
```

Note: `ImageSequenceWriter` does not receive `audio_source` — it has no use for it.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 6: Run full test suite for regression**

Run: `python -m pytest tests/ -v`
Expected: All existing tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/core/writer.py tests/test_audio.py
git commit -m "feat: add audio passthrough to FFmpegWriter and create_writer"
```

---

## Task 4: Wire audio through MattingPipeline

**Files:**
- Modify: `src/core/pipeline.py:45` (the `create_writer` call)

- [ ] **Step 1: Update pipeline to pass input_path as audio_source**

In `src/core/pipeline.py`, change line 45 from:

```python
        writer = create_writer(self._config, output_path, width, height, fps)
```

to:

```python
        writer = create_writer(
            self._config, output_path, width, height, fps,
            audio_source=input_path,
        )
```

- [ ] **Step 2: Run existing pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All tests PASS (audio_source is optional, no behavior change for test videos without audio).

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/core/pipeline.py
git commit -m "feat: wire audio passthrough through MattingPipeline"
```

---

## Task 5: Create ImagePipeline

**Files:**
- Create: `src/core/image_pipeline.py`
- Create: `tests/test_image_pipeline.py`
- Modify: `tests/conftest.py` (add image fixtures)

- [ ] **Step 1: Add image fixtures to conftest.py**

Append to `tests/conftest.py`:

```python
from PIL import Image


@pytest.fixture
def test_image_path():
    """Create a single 64x64 test image and return its path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_image.png")
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    img.save(path)
    yield path
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def test_image_folder(tmp_path):
    """Create a folder with 3 test images."""
    for i in range(3):
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(tmp_path / f"img_{i:03d}.png")
    return str(tmp_path)
```

- [ ] **Step 2: Write tests for ImagePipeline**

Create `tests/test_image_pipeline.py`:

```python
import os
import threading

import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.isdir(os.path.join(MODELS_DIR, "birefnet-general"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestImagePipeline:
    def test_single_image_transparent(self, test_image_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        result = pipeline.process(
            input_path=test_image_path,
            output_dir=temp_output_dir,
        )

        assert os.path.exists(result)
        assert result.endswith(".png")
        # Verify RGBA output
        from PIL import Image
        img = Image.open(result)
        assert img.mode == "RGBA"
        assert img.size == (64, 64)

    def test_single_image_green_screen(self, test_image_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.GREEN,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        result = pipeline.process(
            input_path=test_image_path,
            output_dir=temp_output_dir,
        )

        assert os.path.exists(result)
        from PIL import Image
        img = Image.open(result)
        assert img.mode == "RGB"

    def test_folder_processes_all_images(self, test_image_folder, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        progress_log = []
        result = pipeline.process(
            input_path=test_image_folder,
            output_dir=temp_output_dir,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )

        assert os.path.isdir(result)
        output_files = [f for f in os.listdir(result) if f.endswith(".png")]
        assert len(output_files) == 3
        assert len(progress_log) == 3
        assert progress_log[-1] == (3, 3)

    def test_cancel_stops_folder_processing(self, test_image_folder, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        cancel_event = threading.Event()
        progress_count = []

        def on_progress(current, total):
            progress_count.append(current)
            if current >= 1:
                cancel_event.set()

        pipeline = ImagePipeline(config, MODELS_DIR)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_image_folder,
                output_dir=temp_output_dir,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        assert len(progress_count) < 3

    def test_empty_folder_raises(self, tmp_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig()
        pipeline = ImagePipeline(config, MODELS_DIR)
        with pytest.raises(ValueError, match="No image files found"):
            pipeline.process(
                input_path=str(tmp_path),
                output_dir=temp_output_dir,
            )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_image_pipeline.py -v`
Expected: ImportError — `image_pipeline` module does not exist.

- [ ] **Step 4: Implement ImagePipeline**

Create `src/core/image_pipeline.py`:

```python
import os
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.core.compositing import compose_frame
from src.core.config import IMAGE_EXTENSIONS, MODELS, ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict


class ImagePipeline:
    """Processes single images or image folders through BiRefNet."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Process a single image or image folder.

        Args:
            input_path: Path to an image file or folder containing images.
            output_dir: Directory for output files.
            progress_callback: Called with (current, total) after each image.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.

        Returns:
            Output file path (single image) or output directory path (folder).
        """
        if os.path.isdir(input_path):
            return self._process_folder(
                input_path, output_dir,
                progress_callback, pause_event, cancel_event,
            )
        else:
            return self._process_single(
                input_path, output_dir,
                progress_callback, pause_event, cancel_event,
            )

    def _process_single(
        self,
        image_path: str,
        output_dir: str,
        progress_callback, pause_event, cancel_event,
    ) -> str:
        """Process a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        alpha = predict(self._model, frame, self._device)
        composed = compose_frame(frame, alpha, self._config.background_mode)

        # Build output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        model_dir_name = MODELS[self._config.model_name]
        timestamp = int(time.time() * 1000)
        output_filename = f"{base_name}_{model_dir_name}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        os.makedirs(output_dir, exist_ok=True)
        self._save_png(composed, output_path)

        if progress_callback:
            progress_callback(1, 1)

        return output_path

    def _process_folder(
        self,
        folder_path: str,
        output_dir: str,
        progress_callback, pause_event, cancel_event,
    ) -> str:
        """Process all images in a folder."""
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            raise ValueError(f"No image files found in: {folder_path}")

        # Create output subdirectory named after input folder
        folder_name = os.path.basename(folder_path)
        model_dir_name = MODELS[self._config.model_name]
        timestamp = int(time.time() * 1000)
        out_subdir = f"{folder_name}_{model_dir_name}_{timestamp}"
        out_path = os.path.join(output_dir, out_subdir)
        os.makedirs(out_path, exist_ok=True)

        total = len(image_files)
        for idx, filename in enumerate(image_files, start=1):
            # Check cancel
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")

            # Check pause
            if pause_event:
                while pause_event.is_set():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Processing cancelled by user")
                    time.sleep(0.1)

            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            if frame is None:
                continue  # Skip unreadable files

            alpha = predict(self._model, frame, self._device)
            composed = compose_frame(frame, alpha, self._config.background_mode)

            # Output with same base name + .png
            base_name = os.path.splitext(filename)[0]
            out_file = os.path.join(out_path, f"{base_name}.png")
            self._save_png(composed, out_file)

            if progress_callback:
                progress_callback(idx, total)

        return out_path

    def _save_png(self, composed: np.ndarray, path: str):
        """Save a composed frame as PNG. RGBA for transparent mode, RGB otherwise."""
        if composed.shape[2] == 4:
            img = Image.fromarray(composed, "RGBA")
        else:
            img = Image.fromarray(composed, "RGB")
        img.save(path)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_image_pipeline.py -v`
Expected: All 5 tests PASS (or skipped if model not downloaded).

- [ ] **Step 6: Commit**

```bash
git add src/core/image_pipeline.py tests/test_image_pipeline.py tests/conftest.py
git commit -m "feat: add ImagePipeline for single image and folder processing"
```

---

## Task 6: Update MattingWorker to support image input

**Files:**
- Modify: `src/worker/matting_worker.py`

- [ ] **Step 1: Update MattingWorker to branch by InputType**

Replace `src/worker/matting_worker.py` entirely:

```python
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.config import InputType, ProcessingConfig


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

    def __init__(
        self,
        config: ProcessingConfig,
        models_dir: str,
        input_path: str,
        output_path: str,
        input_type: InputType = InputType.VIDEO,
    ):
        super().__init__()
        self._config = config
        self._models_dir = models_dir
        self._input_path = input_path
        self._output_path = output_path
        self._input_type = input_type

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._last_time = None

    def run(self):
        try:
            self._last_time = time.time()

            if self._input_type == InputType.VIDEO:
                self._run_video()
            else:
                self._run_image()

            self.finished.emit(self._output_path)
        except InterruptedError:
            self.error.emit("Processing cancelled")
        except Exception as e:
            self.error.emit(str(e))

    def _run_video(self):
        from src.core.pipeline import MattingPipeline

        pipeline = MattingPipeline(self._config, self._models_dir)
        pipeline.process(
            input_path=self._input_path,
            output_path=self._output_path,
            progress_callback=self._on_progress,
            pause_event=self._pause_event,
            cancel_event=self._cancel_event,
        )

    def _run_image(self):
        from src.core.image_pipeline import ImagePipeline

        pipeline = ImagePipeline(self._config, self._models_dir)
        result = pipeline.process(
            input_path=self._input_path,
            output_dir=self._output_path,
            progress_callback=self._on_progress,
            pause_event=self._pause_event,
            cancel_event=self._cancel_event,
        )
        # Update output path to the actual result
        self._output_path = result

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

- [ ] **Step 2: Run existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS. The default `input_type=InputType.VIDEO` preserves existing behavior.

- [ ] **Step 3: Commit**

```bash
git add src/worker/matting_worker.py
git commit -m "feat: MattingWorker supports video and image input types"
```

---

## Task 7: Update MainWindow for image input and drag-and-drop

**Files:**
- Modify: `src/gui/main_window.py`

This is the largest task. It adds: file type detection, image/folder selection buttons, UI mode switching, and drag-and-drop.

- [ ] **Step 1: Add imports and constants**

At the top of `src/gui/main_window.py`, update imports:

```python
import os
import time

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    BackgroundMode,
    IMAGE_EXTENSIONS,
    InputType,
    MODELS,
    OutputFormat,
    ProcessingConfig,
    VIDEO_EXTENSIONS,
)
from src.core.inference import detect_device
from src.core.video import get_video_info
from src.worker.matting_worker import MattingWorker
```

- [ ] **Step 2: Add _input_type tracking and enable drag-and-drop**

In `MainWindow.__init__`, after `self._start_time = None`, add:

```python
        self._input_type = None  # InputType enum value
```

And after `self._init_ui()`, before `self._set_state("initial")`, add:

```python
        self.setAcceptDrops(True)
```

- [ ] **Step 3: Replace _on_select_file with input type menu and _handle_input**

Replace the `_on_select_file` method and add new methods. Find the input section in `_init_ui` and replace the select button setup:

Change the input section from:
```python
        self._select_btn = QPushButton("选择文件")
        self._select_btn.clicked.connect(self._on_select_file)
        input_row.addWidget(self._select_btn)
```
to:
```python
        self._select_btn = QPushButton("选择文件 ▼")
        select_menu = QMenu(self)
        select_menu.addAction("选择视频", self._on_select_video)
        select_menu.addAction("选择图片", self._on_select_image)
        select_menu.addAction("选择图片文件夹", self._on_select_folder)
        self._select_btn.setMenu(select_menu)
        input_row.addWidget(self._select_btn)
```

Then replace the `_on_select_file` method with these methods:

```python
    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self._handle_input(path)

    def _classify_input(self, path: str) -> InputType | None:
        """Determine input type from a file or directory path."""
        if os.path.isdir(path):
            # Check if folder contains any images
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            return InputType.IMAGE_FOLDER if has_images else None

        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return InputType.VIDEO
        if ext in IMAGE_EXTENSIONS:
            return InputType.IMAGE
        return None

    def _handle_input(self, path: str):
        """Unified entry point for all input methods (file dialog, drag-and-drop)."""
        input_type = self._classify_input(path)

        if input_type is None:
            QMessageBox.warning(
                self, "不支持的文件",
                f"无法识别的文件类型:\n{path}\n\n"
                "支持的格式: 视频(MP4/AVI/MOV/MKV) | 图片(PNG/JPG/TIFF/BMP/WebP)",
            )
            return

        if input_type == InputType.VIDEO:
            try:
                info = get_video_info(path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取视频文件:\n{e}")
                return

            self._input_path = path
            self._input_type = input_type
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

        elif input_type == InputType.IMAGE:
            from PIL import Image
            try:
                img = Image.open(path)
                w, h = img.size
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取图片:\n{e}")
                return

            self._input_path = path
            self._input_type = input_type
            self._input_edit.setText(path)
            self._info_label.setText(f"图片信息: {w}x{h} | {img.mode}")

        elif input_type == InputType.IMAGE_FOLDER:
            image_files = [
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                and os.path.isfile(os.path.join(path, f))
            ]
            count = len(image_files)

            self._input_path = path
            self._input_type = input_type
            self._input_edit.setText(path)
            self._info_label.setText(f"图片文件夹: {count} 张图片")

        # Update UI for input type
        self._update_ui_for_input_type()
        self._set_state("ready")

    def _update_ui_for_input_type(self):
        """Adjust output settings based on input type."""
        is_video = self._input_type == InputType.VIDEO
        self._format_combo.setEnabled(is_video)
        if not is_video:
            # For image input, format combo doesn't matter — output is always PNG
            pass
```

- [ ] **Step 4: Add drag-and-drop event handlers**

Add to `MainWindow` class:

```python
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._handle_input(path)
```

- [ ] **Step 5: Update _build_output_path for image input**

Replace `_build_output_path` method:

```python
    def _build_output_path(self) -> str:
        """Build output path based on input type."""
        config = self._get_config()
        model_dir_name = MODELS[config.model_name]
        timestamp = int(time.time() * 1000)

        if self._input_type == InputType.VIDEO:
            base_name = os.path.splitext(os.path.basename(self._input_path))[0]
            ext = FORMAT_EXTENSIONS[config.output_format]
            if ext:
                filename = f"{base_name}_{model_dir_name}_{timestamp}{ext}"
            else:
                filename = f"{base_name}_{model_dir_name}_{timestamp}"
            if self._output_dir:
                return os.path.join(self._output_dir, filename)
            else:
                return os.path.join(os.path.dirname(self._input_path), filename)
        else:
            # For image input, return the output directory
            if self._output_dir:
                return self._output_dir
            else:
                if os.path.isdir(self._input_path):
                    return os.path.dirname(self._input_path)
                else:
                    return os.path.dirname(self._input_path)
```

- [ ] **Step 6: Update _on_start to pass input_type to worker**

Replace the `_on_start` method:

```python
    def _on_start(self):
        if not self._input_path:
            return

        config = self._get_config()
        models_dir = os.path.abspath(MODELS_DIR)

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

        self._worker = MattingWorker(
            config, models_dir, self._input_path, output_path,
            input_type=self._input_type,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.speed.connect(self._on_speed)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
```

- [ ] **Step 7: Update _on_finished message for image input**

Replace `_on_finished`:

```python
    def _on_finished(self, output_path: str):
        self._set_state("finished")
        self._progress_bar.setValue(100)
        elapsed = time.time() - self._start_time if self._start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._status_label.setText(f"处理完成! 耗时: {minutes:02d}:{seconds:02d}")

        if self._input_type == InputType.VIDEO:
            msg = f"视频处理完成!\n\n输出文件:\n{output_path}"
        elif self._input_type == InputType.IMAGE:
            msg = f"图片处理完成!\n\n输出文件:\n{output_path}"
        else:
            msg = f"图片文件夹处理完成!\n\n输出目录:\n{output_path}"

        QMessageBox.information(self, "完成", msg)
```

- [ ] **Step 8: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 9: Manual smoke test**

Run: `python main.py`

Verify:
1. "选择文件 ▼" button shows a menu with 3 options
2. Selecting a video → shows video info, format combo enabled
3. Selecting an image → shows image info, format combo disabled
4. Selecting a folder → shows image count, format combo disabled
5. Dragging a video file → same as selecting video
6. Dragging an image file → same as selecting image
7. Dragging a folder → same as selecting folder
8. Dragging an unsupported file → shows warning dialog

- [ ] **Step 10: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: image input, folder input, and drag-and-drop support in GUI"
```

---

## Task 8: Final integration test and cleanup

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Manual end-to-end test with model**

If model is available, run `python main.py` and:
1. Process a video with audio → verify output contains audio track
2. Process a single image → verify PNG output
3. Process an image folder → verify all images processed
4. Drag and drop a video → verify it works
5. Cancel during folder processing → verify clean stop

- [ ] **Step 3: Update PROGRESS.md**

Update `PROGRESS.md` to reflect the completed P1 features:
- Mark audio preservation as ✅
- Mark image input as ✅
- Mark drag-and-drop as ✅
- Remove "设置持久化" from P1 (deferred to P2 with batch queue)
- Update test counts

- [ ] **Step 4: Commit progress update**

```bash
git add PROGRESS.md
git commit -m "docs: update PROGRESS.md — P1 features complete (audio, image, drag-drop)"
```

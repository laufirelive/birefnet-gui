# BiRefNet GUI MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working video matting tool that takes a video file, runs BiRefNet-general frame-by-frame, and outputs a MOV ProRes 4444 with transparency — controlled through a PyQt6 GUI.

**Architecture:** Single-process PyQt6 app. GUI on main thread, inference pipeline on a QThread worker. Worker communicates progress via Qt Signals. Video read with OpenCV, write with FFmpeg subprocess piping raw RGBA frames to `ffmpeg -c:v prores_ks -profile:v 4444`.

**Tech Stack:** Python 3.12+, PyQt6, PyTorch, transformers (BiRefNet), OpenCV, ffmpeg-python, FFmpeg CLI

---

## File Structure

```
birefnet-gui/
├── main.py                      # App entry point — creates QApplication, shows MainWindow
├── requirements.txt             # All pip dependencies
├── download_models.py           # Script to download BiRefNet-general for offline use
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── inference.py         # load_model(), predict() — BiRefNet model wrapper
│   │   ├── video.py             # get_video_info(), FrameReader, ProResWriter
│   │   └── pipeline.py          # MattingPipeline — orchestrates read→infer→write loop
│   ├── worker/
│   │   ├── __init__.py
│   │   └── matting_worker.py    # MattingWorker(QThread) — runs pipeline, emits signals
│   └── gui/
│       ├── __init__.py
│       └── main_window.py       # MainWindow(QMainWindow) — full UI
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures (test video, temp dirs)
│   ├── test_video.py            # Tests for video.py
│   ├── test_inference.py        # Tests for inference.py (with model fixture)
│   └── test_pipeline.py         # Integration test for pipeline.py
└── models/                      # Model files (git-ignored, downloaded separately)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `main.py`
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/worker/__init__.py`
- Create: `src/gui/__init__.py`
- Create: `tests/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual environment
venv/
.venv/

# Models (large files, download separately)
models/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Test artifacts
tests/fixtures/output/
.pytest_cache/
```

- [ ] **Step 2: Create `requirements.txt`**

```
PyQt6>=6.5
torch>=2.0
torchvision>=0.15
transformers>=4.30
opencv-python>=4.8
ffmpeg-python>=0.2
Pillow>=10.0
numpy>=1.24
pytest>=7.0
```

- [ ] **Step 3: Create package `__init__.py` files**

All `__init__.py` files are empty:
- `src/__init__.py`
- `src/core/__init__.py`
- `src/worker/__init__.py`
- `src/gui/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Create `main.py` entry point (placeholder)**

```python
import sys
from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BiRefNet Video Matting Tool")
    # MainWindow will be added in Task 6
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create virtual environment and install dependencies**

Run:
```bash
cd ~/birefnet-gui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Expected: All packages install successfully. PyQt6, torch, transformers, opencv-python, ffmpeg-python all importable.

- [ ] **Step 6: Verify imports work**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -c "import PyQt6; import torch; import transformers; import cv2; import ffmpeg; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 7: Commit**

```bash
cd ~/birefnet-gui
git add .gitignore requirements.txt main.py src/ tests/
git commit -m "feat: project scaffolding with dependencies and entry point"
```

---

## Task 2: Video Reader and Writer (`src/core/video.py`)

**Files:**
- Create: `src/core/video.py`
- Create: `tests/conftest.py`
- Create: `tests/test_video.py`

This module handles all video I/O: reading frames from input video via OpenCV, getting video metadata, and writing RGBA frames to MOV ProRes 4444 via FFmpeg subprocess.

- [ ] **Step 1: Create test fixture — a tiny test video**

Create `tests/conftest.py` with a fixture that generates a small 10-frame test video:

```python
import os
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def test_video_path():
    """Create a tiny 10-frame 64x64 test video and return its path."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_input.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30.0, (64, 64))
    for i in range(10):
        # Each frame is a solid color that changes per frame
        frame = np.full((64, 64, 3), fill_value=(i * 25) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)
    os.rmdir(tmpdir)


@pytest.fixture
def temp_output_dir():
    """Provide a temporary directory for output files."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup any files left behind
    for f in os.listdir(tmpdir):
        os.remove(os.path.join(tmpdir, f))
    os.rmdir(tmpdir)
```

- [ ] **Step 2: Write failing tests for `get_video_info`**

Create `tests/test_video.py`:

```python
import os

import numpy as np

from src.core.video import FrameReader, ProResWriter, get_video_info


class TestGetVideoInfo:
    def test_returns_metadata(self, test_video_path):
        info = get_video_info(test_video_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["fps"] == 30.0
        assert info["frame_count"] == 10
        assert info["duration"] > 0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")
```

Add the missing import at the top:

```python
import pytest
```

- [ ] **Step 3: Run tests to verify they fail**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestGetVideoInfo -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.video'`

- [ ] **Step 4: Implement `get_video_info`**

Create `src/core/video.py`:

```python
import os
import subprocess

import cv2
import numpy as np


def get_video_info(path: str) -> dict:
    """Get video metadata: width, height, fps, frame_count, duration."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
```

- [ ] **Step 5: Run `get_video_info` tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestGetVideoInfo -v
```

Expected: 2 passed

- [ ] **Step 6: Write failing tests for `FrameReader`**

Append to `tests/test_video.py`:

```python
class TestFrameReader:
    def test_reads_all_frames(self, test_video_path):
        reader = FrameReader(test_video_path)
        frames = list(reader)
        assert len(frames) == 10

    def test_frame_shape_is_bgr(self, test_video_path):
        reader = FrameReader(test_video_path)
        frame = next(iter(reader))
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            FrameReader("/nonexistent/video.mp4")
```

- [ ] **Step 7: Run to verify failure**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestFrameReader -v
```

Expected: FAIL

- [ ] **Step 8: Implement `FrameReader`**

Add to `src/core/video.py`:

```python
class FrameReader:
    """Iterator that yields BGR frames from a video file."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self._path = path

    def __iter__(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()
```

- [ ] **Step 9: Run FrameReader tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestFrameReader -v
```

Expected: 3 passed

- [ ] **Step 10: Write failing test for `ProResWriter`**

Append to `tests/test_video.py`:

```python
class TestProResWriter:
    def test_writes_mov_file(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "output.mov")
        writer = ProResWriter(output_path, width=64, height=64, fps=30.0)

        for i in range(5):
            # Create RGBA frame: solid color with full opacity
            rgba = np.full((64, 64, 4), fill_value=128, dtype=np.uint8)
            rgba[:, :, 3] = 255  # full alpha
            writer.write_frame(rgba)

        writer.close()

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify it's a valid video with correct frame count
        info = get_video_info(output_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["frame_count"] == 5
```

- [ ] **Step 11: Run to verify failure**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestProResWriter -v
```

Expected: FAIL

- [ ] **Step 12: Implement `ProResWriter`**

Add to `src/core/video.py`:

```python
class ProResWriter:
    """Writes RGBA frames to a MOV file using ProRes 4444 via FFmpeg."""

    def __init__(self, output_path: str, width: int, height: int, fps: float):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",  # overwrite output
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",  # read from stdin
                "-c:v", "prores_ks",
                "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le",
                "-vendor", "apl0",
                output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write_frame(self, rgba_frame: np.ndarray):
        """Write one RGBA uint8 frame. Shape must be (height, width, 4)."""
        assert rgba_frame.shape == (self._height, self._width, 4), (
            f"Expected ({self._height}, {self._width}, 4), got {rgba_frame.shape}"
        )
        self._process.stdin.write(rgba_frame.tobytes())

    def close(self):
        """Flush and close the FFmpeg process."""
        if self._process.stdin:
            self._process.stdin.close()
        self._process.wait()
        if self._process.returncode != 0:
            stderr = self._process.stderr.read().decode()
            raise RuntimeError(f"FFmpeg failed (code {self._process.returncode}): {stderr}")
```

- [ ] **Step 13: Run ProResWriter tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py::TestProResWriter -v
```

Expected: 1 passed

- [ ] **Step 14: Run all video tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_video.py -v
```

Expected: 6 passed

- [ ] **Step 15: Commit**

```bash
cd ~/birefnet-gui
git add src/core/video.py tests/conftest.py tests/test_video.py
git commit -m "feat: video I/O module — FrameReader, ProResWriter, get_video_info"
```

---

## Task 3: Inference Module (`src/core/inference.py`)

**Files:**
- Create: `src/core/inference.py`
- Create: `tests/test_inference.py`

This module wraps BiRefNet model loading and single-frame prediction. The model takes a 1024x1024 RGB image and outputs an alpha mask. This module handles the resize-in / resize-out logic.

**Note:** Tests in this task require the BiRefNet-general model to be downloaded to `models/birefnet-general/`. Run `python download_models.py` first if you haven't. Tests that need the model are marked with `@pytest.mark.skipif` so CI without a model still passes.

- [ ] **Step 1: Create `download_models.py`**

```python
#!/usr/bin/env python3
"""Download BiRefNet-general model for offline use.

Run: python download_models.py
Downloads to: ./models/birefnet-general/
"""

import os

from huggingface_hub import snapshot_download


def main():
    model_dir = "./models/birefnet-general"
    os.makedirs(model_dir, exist_ok=True)

    print("Downloading BiRefNet-general model...")
    print(f"Destination: {os.path.abspath(model_dir)}")

    snapshot_download(
        repo_id="zhengpeng7/BiRefNet",
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Done. Model saved to {os.path.abspath(model_dir)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download the model (one-time)**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
pip install huggingface_hub
python download_models.py
```

Expected: Model downloaded to `./models/birefnet-general/` (several hundred MB, may take a few minutes).

- [ ] **Step 3: Write failing tests for inference**

Create `tests/test_inference.py`:

```python
import os

import numpy as np
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "birefnet-general")
MODEL_EXISTS = os.path.isdir(MODEL_PATH)

from src.core.inference import detect_device, load_model, predict


class TestDetectDevice:
    def test_returns_string(self):
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestLoadModel:
    def test_loads_model(self):
        device = detect_device()
        model = load_model(MODEL_PATH, device)
        assert model is not None

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model", "cpu")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestPredict:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.device = detect_device()
        self.model = load_model(MODEL_PATH, self.device)

    def test_returns_alpha_mask(self):
        # Create a dummy 640x480 BGR image
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        alpha = predict(self.model, frame, self.device)
        assert alpha.shape == (480, 640)
        assert alpha.dtype == np.uint8
        assert alpha.min() >= 0
        assert alpha.max() <= 255

    def test_preserves_input_resolution(self):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        alpha = predict(self.model, frame, self.device)
        assert alpha.shape == (1080, 1920)
```

- [ ] **Step 4: Run tests to verify they fail**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_inference.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.inference'`

- [ ] **Step 5: Implement `src/core/inference.py`**

```python
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


def detect_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path: str, device: str) -> AutoModelForImageSegmentation:
    """Load BiRefNet model from a local directory."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    model = AutoModelForImageSegmentation.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model


_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(model, frame: np.ndarray, device: str) -> np.ndarray:
    """Run BiRefNet on a single BGR frame, return alpha mask at original resolution.

    Args:
        model: Loaded BiRefNet model.
        frame: BGR uint8 numpy array, shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.

    Returns:
        Alpha mask as uint8 numpy array, shape (H, W), values 0-255.
    """
    orig_h, orig_w = frame.shape[:2]

    # BGR -> RGB -> PIL
    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb)

    # Preprocess
    input_tensor = _transform(image).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        preds = model(input_tensor)[-1]
        pred = torch.sigmoid(preds[0, 0])

    # Resize back to original resolution
    pred_resized = torch.nn.functional.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    # Convert to uint8 numpy
    alpha = (pred_resized * 255).clamp(0, 255).byte().cpu().numpy()
    return alpha
```

- [ ] **Step 6: Run inference tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_inference.py -v
```

Expected: All tests pass (or skip if model not downloaded). On Mac with MPS, `detect_device` returns "mps". `predict` returns correct shape and dtype.

- [ ] **Step 7: Commit**

```bash
cd ~/birefnet-gui
git add src/core/inference.py tests/test_inference.py download_models.py
git commit -m "feat: inference module — BiRefNet model loading and prediction"
```

---

## Task 4: Processing Pipeline (`src/core/pipeline.py`)

**Files:**
- Create: `src/core/pipeline.py`
- Create: `tests/test_pipeline.py`

The pipeline orchestrates the full flow: read video frames → run inference → compose RGBA → write ProRes output. It supports pause/cancel via threading events and reports progress via callback.

- [ ] **Step 1: Write failing integration test**

Create `tests/test_pipeline.py`:

```python
import os
import threading

import numpy as np
import pytest

from src.core.video import get_video_info

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "birefnet-general")
MODEL_EXISTS = os.path.isdir(MODEL_PATH)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_end_to_end(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        output_path = os.path.join(temp_output_dir, "output.mov")
        progress_log = []

        def on_progress(current, total):
            progress_log.append((current, total))

        pipeline = MattingPipeline(MODEL_PATH)
        pipeline.process(
            input_path=test_video_path,
            output_path=output_path,
            progress_callback=on_progress,
        )

        # Output file exists and is valid
        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64

        # Progress was reported for each frame
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        output_path = os.path.join(temp_output_dir, "output.mov")
        cancel_event = threading.Event()
        frame_count = []

        def on_progress(current, total):
            frame_count.append(current)
            if current >= 3:
                cancel_event.set()

        pipeline = MattingPipeline(MODEL_PATH)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        # Should have stopped around frame 3-4 (not all 10)
        assert len(frame_count) < 10
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.pipeline'`

- [ ] **Step 3: Implement `src/core/pipeline.py`**

```python
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.core.inference import detect_device, load_model, predict
from src.core.video import FrameReader, ProResWriter, get_video_info


class MattingPipeline:
    """Orchestrates video read → BiRefNet inference → ProRes write."""

    def __init__(self, model_path: str):
        self._device = detect_device()
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Process a video file: extract frames, run inference, write output.

        Args:
            input_path: Path to input video file.
            output_path: Path for output MOV file.
            progress_callback: Called with (current_frame, total_frames) after each frame.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            InterruptedError: If cancel_event is set during processing.
        """
        video_info = get_video_info(input_path)
        total_frames = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]

        reader = FrameReader(input_path)
        writer = ProResWriter(output_path, width, height, fps)

        try:
            for frame_idx, frame in enumerate(reader, start=1):
                # Check cancel
                if cancel_event and cancel_event.is_set():
                    writer.close()
                    # Clean up partial output
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    raise InterruptedError("Processing cancelled by user")

                # Check pause
                if pause_event:
                    while pause_event.is_set():
                        if cancel_event and cancel_event.is_set():
                            writer.close()
                            if os.path.exists(output_path):
                                os.remove(output_path)
                            raise InterruptedError("Processing cancelled by user")
                        time.sleep(0.1)

                # Inference
                alpha = predict(self._model, frame, self._device)

                # Compose RGBA
                rgba = np.dstack([frame[:, :, ::-1], alpha])  # BGR→RGB + alpha
                # ProResWriter expects RGBA
                writer.write_frame(rgba)

                # Report progress
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

            writer.close()
        except InterruptedError:
            raise
        except Exception:
            writer.close()
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
```

- [ ] **Step 4: Run pipeline tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/test_pipeline.py -v
```

Expected: Tests pass (may take a few seconds for model inference on the small test video).

- [ ] **Step 5: Run all tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
cd ~/birefnet-gui
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: matting pipeline — orchestrates read, infer, write with pause/cancel"
```

---

## Task 5: QThread Worker (`src/worker/matting_worker.py`)

**Files:**
- Create: `src/worker/matting_worker.py`

The worker wraps the pipeline in a QThread so the GUI stays responsive. It emits Qt signals for progress updates, completion, and errors. No automated test for this — it's a thin adapter over the already-tested pipeline. Manual verification happens in Task 6.

- [ ] **Step 1: Implement `src/worker/matting_worker.py`**

```python
import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

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

    def __init__(self, model_path: str, input_path: str, output_path: str):
        super().__init__()
        self._model_path = model_path
        self._input_path = input_path
        self._output_path = output_path

        self._pause_event = threading.Event()  # set = paused
        self._cancel_event = threading.Event()  # set = cancelled
        self._last_time = None

    def run(self):
        try:
            pipeline = MattingPipeline(self._model_path)
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

- [ ] **Step 2: Commit**

```bash
cd ~/birefnet-gui
git add src/worker/matting_worker.py
git commit -m "feat: MattingWorker QThread — bridges pipeline to GUI via signals"
```

---

## Task 6: GUI Main Window (`src/gui/main_window.py`)

**Files:**
- Create: `src/gui/main_window.py`
- Modify: `main.py`

The main window provides: file selection, video info display, output path selection, start/pause/cancel controls, and a progress bar. All layout done in code (no .ui files).

- [ ] **Step 1: Implement `src/gui/main_window.py`**

```python
import os
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
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

from src.core.inference import detect_device
from src.core.video import get_video_info
from src.worker.matting_worker import MattingWorker

# Path to bundled model (relative to project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "birefnet-general")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(600, 450)

        self._worker = None
        self._input_path = None
        self._output_dir = None
        self._start_time = None

        self._init_ui()
        self._set_state("initial")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Input file section ---
        layout.addWidget(QLabel("输入文件:"))
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setReadOnly(True)
        self._input_edit.setPlaceholderText("未选择文件")
        input_row.addWidget(self._input_edit)
        self._select_btn = QPushButton("选择文件")
        self._select_btn.clicked.connect(self._on_select_file)
        input_row.addWidget(self._select_btn)
        layout.addLayout(input_row)

        # Video info
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray;")
        layout.addWidget(self._info_label)

        # --- Separator ---
        sep1 = QLabel()
        sep1.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(sep1)

        # --- Model / Device / Output format ---
        device = detect_device()
        device_text = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
        self._model_label = QLabel(f"模型: BiRefNet-general")
        layout.addWidget(self._model_label)
        self._device_label = QLabel(f"设备: {device_text.get(device, device)}")
        layout.addWidget(self._device_label)
        self._format_label = QLabel("输出: MOV ProRes 4444 (透明)")
        layout.addWidget(self._format_label)

        # --- Output path ---
        layout.addWidget(QLabel("输出路径:"))
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("与输入文件同目录")
        output_row.addWidget(self._output_edit)
        self._output_btn = QPushButton("浏览...")
        self._output_btn.clicked.connect(self._on_select_output)
        output_row.addWidget(self._output_btn)
        layout.addLayout(output_row)

        # --- Separator ---
        sep2 = QLabel()
        sep2.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(sep2)

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        layout.addWidget(self._status_label)

        # --- Control buttons ---
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
        layout.addLayout(btn_row)

        layout.addStretch()

    def _set_state(self, state: str):
        """Update button enabled/disabled state based on current state."""
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
        base_name = os.path.splitext(os.path.basename(self._input_path))[0]
        timestamp = int(time.time() * 1000)
        filename = f"{base_name}_birefnet-general_{timestamp}.mov"

        if self._output_dir:
            return os.path.join(self._output_dir, filename)
        else:
            return os.path.join(os.path.dirname(self._input_path), filename)

    def _on_start(self):
        if not self._input_path:
            return

        # Check model exists
        model_path = os.path.abspath(MODEL_PATH)
        if not os.path.isdir(model_path):
            QMessageBox.critical(
                self,
                "模型缺失",
                f"未找到 BiRefNet-general 模型:\n{model_path}\n\n"
                "请运行 python download_models.py 下载模型。",
            )
            return

        output_path = self._build_output_path()
        self._start_time = time.time()
        self._set_state("processing")
        self._status_label.setText("正在加载模型...")

        self._worker = MattingWorker(model_path, self._input_path, output_path)
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
        pass  # Speed is calculated in _on_progress from wall clock for accuracy

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

- [ ] **Step 2: Update `main.py` to show the window**

Replace the contents of `main.py`:

```python
import sys

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BiRefNet Video Matting Tool")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Manual smoke test**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python main.py
```

Verify:
1. Window opens with title "BiRefNet Video Matting Tool"
2. "选择文件" button opens a file dialog
3. After selecting a video, video info appears and "开始处理" becomes enabled
4. Clicking "开始处理" starts processing with progress bar updating
5. "暂停" pauses, button text changes to "继续"
6. "取消" stops processing
7. On completion, a dialog shows the output file path

- [ ] **Step 4: Commit**

```bash
cd ~/birefnet-gui
git add src/gui/main_window.py main.py
git commit -m "feat: PyQt6 main window with file selection, progress, and controls"
```

---

## Task 7: Final Integration and Cleanup

**Files:**
- Review: all files
- Potentially fix: any integration issues found

- [ ] **Step 1: Run all tests**

Run:
```bash
cd ~/birefnet-gui && source venv/bin/activate
python -m pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Full end-to-end manual test**

Run the app and process a real video (not just the tiny test fixture):

```bash
cd ~/birefnet-gui && source venv/bin/activate
python main.py
```

1. Select a short video clip (5-10 seconds)
2. Click "开始处理"
3. Watch progress bar advance
4. Test pause/resume
5. Let it complete
6. Open the output .mov and verify it has transparency

- [ ] **Step 3: Fix any issues found**

If integration issues arise, fix them and add tests if appropriate.

- [ ] **Step 4: Final commit**

```bash
cd ~/birefnet-gui
git add -A
git commit -m "chore: integration fixes and cleanup"
```

(Only commit this if there were actual changes to make.)

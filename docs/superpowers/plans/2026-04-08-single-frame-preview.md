# Single-Frame Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users preview matting results on a single frame before committing to full video processing.

**Architecture:** Four new modules (frame_extractor, preview_worker, image_viewer, checkerboard renderer) plus layout changes to main_window. FFmpeg extracts frames on seek, PreviewWorker runs model inference in a background thread and releases the model immediately after.

**Tech Stack:** PyQt6 (QGraphicsView, QSlider, QThread), FFmpeg subprocess, numpy, existing BiRefNet inference pipeline.

**Branch:** `feature/single-frame-preview` (from `master`)

---

### Task 0: Create feature branch

**Files:** None

- [ ] **Step 1: Create and switch to new branch**

```bash
git checkout master
git checkout -b feature/single-frame-preview
```

- [ ] **Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `feature/single-frame-preview`

---

### Task 1: Frame extractor module

**Files:**
- Create: `src/core/frame_extractor.py`
- Create: `tests/test_frame_extractor.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_frame_extractor.py`:

```python
import numpy as np
import pytest

from src.core.frame_extractor import extract_frame


class TestExtractFrame:
    def test_returns_rgb_array_with_correct_shape(self, tmp_path):
        """Generate a tiny test video with ffmpeg and extract frame 0."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        # Generate a 4-frame 2fps 64x48 red video
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=red:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=0, fps=2.0, width=64, height=48)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_frame_content_is_plausible(self, tmp_path):
        """Red video frame should have high R channel values."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=red:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=0, fps=2.0, width=64, height=48)
        # RGB: red channel should dominate
        assert frame[:, :, 0].mean() > 200  # R
        assert frame[:, :, 2].mean() < 50   # B

    def test_extract_later_frame(self, tmp_path):
        """Extracting frame 2 at 2fps should seek to t=1.0s."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=blue:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=2, fps=2.0, width=64, height=48)
        assert frame.shape == (48, 64, 3)

    def test_raises_on_invalid_path(self):
        with pytest.raises(RuntimeError):
            extract_frame("/nonexistent/video.mp4", frame_number=0, fps=30.0, width=64, height=48)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_frame_extractor.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.frame_extractor'`

- [ ] **Step 3: Write implementation**

Create `src/core/frame_extractor.py`:

```python
import subprocess

import numpy as np


def extract_frame(
    input_path: str,
    frame_number: int,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Extract a single frame from a video using FFmpeg.

    Args:
        input_path: Path to the video file.
        frame_number: Zero-based frame index to extract.
        fps: Frame rate of the video.
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        RGB uint8 numpy array of shape (height, width, 3).

    Raises:
        RuntimeError: If FFmpeg fails or returns unexpected data.
    """
    time_seconds = frame_number / fps

    cmd = [
        "ffmpeg",
        "-ss", f"{time_seconds:.6f}",
        "-i", input_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found in PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out extracting frame")

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {stderr}")

    expected_size = height * width * 3
    if len(result.stdout) < expected_size:
        raise RuntimeError(
            f"FFmpeg returned {len(result.stdout)} bytes, expected {expected_size}"
        )

    frame = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8)
    return frame.reshape((height, width, 3))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_frame_extractor.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/core/frame_extractor.py tests/test_frame_extractor.py
git commit -m "feat: add frame_extractor module for FFmpeg single-frame extraction"
```

---

### Task 2: Checkerboard preview renderer

**Files:**
- Modify: `src/core/compositing.py`
- Modify: `tests/test_compositing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compositing.py`:

```python
from src.core.compositing import render_checkerboard_preview


class TestRenderCheckerboardPreview:
    def test_output_shape_matches_input(self):
        rgb = np.full((32, 32, 3), 200, dtype=np.uint8)
        alpha = np.full((32, 32), 255, dtype=np.uint8)
        result = render_checkerboard_preview(rgb, alpha, cell_size=8)
        assert result.shape == (32, 32, 3)
        assert result.dtype == np.uint8

    def test_fully_opaque_returns_foreground(self):
        rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
        alpha = np.full((16, 16), 255, dtype=np.uint8)
        result = render_checkerboard_preview(rgb, alpha, cell_size=8)
        np.testing.assert_array_equal(result, rgb)

    def test_fully_transparent_returns_checkerboard(self):
        rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
        alpha = np.zeros((16, 16), dtype=np.uint8)
        result = render_checkerboard_preview(rgb, alpha, cell_size=8)
        # Top-left cell should be light (#FFFFFF)
        assert result[0, 0, 0] == 255
        # Cell at (0, 8) should be dark (#CCCCCC)
        assert result[0, 8, 0] == 204

    def test_half_alpha_blends(self):
        rgb = np.zeros((16, 16, 3), dtype=np.uint8)
        alpha = np.full((16, 16), 128, dtype=np.uint8)
        result = render_checkerboard_preview(rgb, alpha, cell_size=8)
        # Light cell (255) blended 50% with black (0) → ~128
        val = result[0, 0, 0]
        assert 120 <= val <= 135
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_compositing.py::TestRenderCheckerboardPreview -v
```

Expected: FAIL — `ImportError: cannot import name 'render_checkerboard_preview'`

- [ ] **Step 3: Write implementation**

Add to `src/core/compositing.py` at the end:

```python
def render_checkerboard_preview(
    rgb: np.ndarray,
    alpha: np.ndarray,
    cell_size: int = 16,
) -> np.ndarray:
    """Composite RGB + alpha onto a checkerboard background.

    Args:
        rgb: RGB uint8 array, shape (H, W, 3).
        alpha: uint8 array, shape (H, W), values 0-255.
        cell_size: Size of each checkerboard square in pixels.

    Returns:
        Composited RGB uint8 array, shape (H, W, 3).
    """
    h, w = alpha.shape
    # Build checkerboard: alternating 255 and 204 cells
    rows = np.arange(h) // cell_size
    cols = np.arange(w) // cell_size
    checkerboard = ((rows[:, None] + cols[None, :]) % 2 == 0).astype(np.uint8)
    bg = np.where(checkerboard[:, :, None], 255, 204).astype(np.float32)

    alpha_f = alpha.astype(np.float32) / 255.0
    blended = rgb.astype(np.float32) * alpha_f[:, :, None] + bg * (1.0 - alpha_f[:, :, None])
    return blended.clip(0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_compositing.py -v
```

Expected: All passed (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add src/core/compositing.py tests/test_compositing.py
git commit -m "feat: add render_checkerboard_preview to compositing module"
```

---

### Task 3: PreviewWorker background thread

**Files:**
- Create: `src/worker/preview_worker.py`
- Create: `tests/test_preview_worker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_preview_worker.py`:

```python
import numpy as np
import pytest

from unittest.mock import patch, MagicMock

from src.worker.preview_worker import PreviewWorker


class TestPreviewWorker:
    def test_emits_finished_with_alpha_mask(self, qtbot):
        """PreviewWorker should emit finished(np.ndarray) on success."""
        fake_alpha = np.full((48, 64), 128, dtype=np.uint8)
        fake_model = MagicMock()

        with patch("src.worker.preview_worker.detect_device", return_value="cpu"), \
             patch("src.worker.preview_worker.load_model", return_value=fake_model), \
             patch("src.worker.preview_worker.predict", return_value=fake_alpha):

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            worker = PreviewWorker(
                model_name="BiRefNet-general",
                models_dir="/fake/models",
                frame=frame,
                resolution=1024,
            )

            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start()

            result = blocker.args[0]
            assert isinstance(result, np.ndarray)
            assert result.shape == (48, 64)
            np.testing.assert_array_equal(result, fake_alpha)

    def test_emits_error_on_failure(self, qtbot):
        """PreviewWorker should emit error(str) if model loading fails."""
        with patch("src.worker.preview_worker.detect_device", return_value="cpu"), \
             patch("src.worker.preview_worker.load_model", side_effect=FileNotFoundError("Model not found")):

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            worker = PreviewWorker(
                model_name="BiRefNet-general",
                models_dir="/fake/models",
                frame=frame,
                resolution=1024,
            )

            with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
                worker.start()

            assert "Model not found" in blocker.args[0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_preview_worker.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.worker.preview_worker'`

- [ ] **Step 3: Write implementation**

Create `src/worker/preview_worker.py`:

```python
import gc

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.inference import detect_device, get_model_path, load_model, predict


class PreviewWorker(QThread):
    """Run single-frame matting inference in a background thread.

    Loads the model, runs prediction on one frame, then immediately
    releases the model and frees GPU memory.

    Signals:
        finished(np.ndarray): Emitted with the alpha mask (H, W) on success.
        error(str): Emitted with error message on failure.
    """

    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(
        self,
        model_name: str,
        models_dir: str,
        frame: np.ndarray,
        resolution: int,
    ):
        super().__init__()
        self._model_name = model_name
        self._models_dir = models_dir
        self._frame = frame
        self._resolution = resolution

    def run(self):
        model = None
        try:
            device = detect_device()
            model_path = get_model_path(self._model_name, self._models_dir)
            model = load_model(model_path, device)
            alpha = predict(model, self._frame, device, self._resolution)
            self.finished.emit(alpha)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_preview_worker.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/worker/preview_worker.py tests/test_preview_worker.py
git commit -m "feat: add PreviewWorker for single-frame matting inference"
```

---

### Task 4: ImageViewerDialog (zoom/pan popup)

**Files:**
- Create: `src/gui/image_viewer.py`
- Create: `tests/test_image_viewer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_image_viewer.py`:

```python
import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage

from src.gui.image_viewer import ImageViewerDialog


class TestImageViewerDialog:
    def test_creates_and_shows(self, qtbot):
        """Dialog should open with an image without error."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        dialog = ImageViewerDialog(img)
        qtbot.addWidget(dialog)
        dialog.show()
        assert dialog.isVisible()
        dialog.close()

    def test_closes_on_escape(self, qtbot):
        """Pressing Escape should close the dialog."""
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        dialog = ImageViewerDialog(img)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.keyClick(dialog, Qt.Key.Key_Escape)
        assert not dialog.isVisible()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
./venv/bin/python -m pytest tests/test_image_viewer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.gui.image_viewer'`

- [ ] **Step 3: Write implementation**

Create `src/gui/image_viewer.py`:

```python
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
)


class _ZoomableView(QGraphicsView):
    """QGraphicsView with scroll-to-zoom and drag-to-pan."""

    def __init__(self, scene: QGraphicsScene):
        super().__init__(scene)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


class ImageViewerDialog(QDialog):
    """Full-screen-ish dialog for viewing an image with zoom and pan.

    Args:
        image: RGB uint8 numpy array (H, W, 3).
        parent: Optional parent widget.
    """

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scene = QGraphicsScene(self)
        self._view = _ZoomableView(self._scene)
        self._view.setStyleSheet("background: #222222;")
        layout.addWidget(self._view)

        # Convert numpy RGB to QPixmap
        h, w, _ = image.shape
        qimage = QImage(image.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self._pixmap_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._pixmap_item)

        # Size to 80% of screen, centered
        screen = self.screen().availableGeometry()
        dialog_w = int(screen.width() * 0.8)
        dialog_h = int(screen.height() * 0.8)
        self.resize(dialog_w, dialog_h)
        self.move(
            screen.x() + (screen.width() - dialog_w) // 2,
            screen.y() + (screen.height() - dialog_h) // 2,
        )

        # Fit image in view
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event):
        super().showEvent(event)
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
./venv/bin/python -m pytest tests/test_image_viewer.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/gui/image_viewer.py tests/test_image_viewer.py
git commit -m "feat: add ImageViewerDialog with zoom and pan support"
```

---

### Task 5: Main window layout changes — preview pane, slider, frame label

This task restructures the left panel layout to insert the preview pane, frame slider, and frame info label between the output path area and the progress bar.

**Files:**
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Add new imports to main_window.py**

Add these imports at the top of the file:

```python
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
```

(Add `QSlider` to the existing import list, and add `QImage, QPixmap` to the `QtGui` import.)

- [ ] **Step 2: Add preview state fields to `__init__`**

After `self._current_phase = None` in `__init__`, add:

```python
self._video_info = None        # dict from get_video_info, set on video load
self._current_frame_rgb = None # np.ndarray (H, W, 3) of currently displayed frame
self._preview_worker = None
```

- [ ] **Step 3: Restructure left panel in `_init_ui`**

Replace the left panel section (from `# --- Left panel ---` through `left_panel.addStretch()`) with:

```python
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
        self._select_btn = QPushButton("选择文件 ▼")
        select_menu = QMenu(self)
        select_menu.addAction("选择视频", self._on_select_video)
        select_menu.addAction("选择图片", self._on_select_image)
        select_menu.addAction("选择图片文件夹", self._on_select_folder)
        self._select_btn.setMenu(select_menu)
        input_row.addWidget(self._select_btn)
        left_panel.addLayout(input_row)

        # Video/image info
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

        # --- Preview pane ---
        self._preview_label = QLabel("选择视频后预览")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_label.setStyleSheet(
            "background: #2a2a2a; color: #888; border: 1px solid #444; border-radius: 4px;"
        )
        self._preview_label.setMinimumHeight(120)
        self._preview_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._preview_label.mousePressEvent = self._on_preview_clicked
        left_panel.addWidget(self._preview_label, stretch=1)

        # Frame slider
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setEnabled(False)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.sliderReleased.connect(self._on_slider_released)
        left_panel.addWidget(self._frame_slider)

        # Frame info label
        self._frame_info_label = QLabel("")
        self._frame_info_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._frame_info_label)

        # Separator before progress
        sep3 = QLabel()
        sep3.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        left_panel.addWidget(sep3)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        left_panel.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._status_label)
```

- [ ] **Step 4: Add preview button to action bar**

In the action bar section, add the preview button before the start button:

```python
        self._preview_btn = QPushButton("预览")
        self._preview_btn.clicked.connect(self._on_preview)
        btn_row.addWidget(self._preview_btn)
```

Insert this right after `btn_row.addStretch()` (the first one) and before `self._start_btn = QPushButton("开始处理")`.

- [ ] **Step 5: Update `_set_state` to manage preview button and slider**

Add `self._preview_btn` state to each branch:

```python
    def _set_state(self, state: str):
        self._state = state
        has_input = self._input_path is not None
        is_video = self._input_type == InputType.VIDEO
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._frame_slider.setEnabled(False)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._preview_btn.setEnabled(has_input and self._input_type != InputType.IMAGE_FOLDER)
            self._select_btn.setEnabled(True)
            self._frame_slider.setEnabled(is_video)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._frame_slider.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._preview_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._frame_slider.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._preview_btn.setEnabled(has_input and self._input_type != InputType.IMAGE_FOLDER)
            self._select_btn.setEnabled(True)
            self._frame_slider.setEnabled(is_video)
            if self._queue_tab._queue_state == "idle":
                has_pending = self._queue_manager.next_pending_task() is not None
                self._queue_tab._start_btn.setEnabled(has_pending)
```

- [ ] **Step 6: Add stub handler methods**

Add these placeholder methods to `MainWindow` (they will be fleshed out in the next task):

```python
    def _on_slider_released(self):
        pass

    def _on_preview(self):
        pass

    def _on_preview_clicked(self, event):
        pass
```

- [ ] **Step 7: Verify the app launches without errors**

```bash
./venv/bin/python -c "from src.gui.main_window import MainWindow; print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: restructure left panel layout with preview pane, slider, and preview button"
```

---

### Task 6: Wire up frame extraction and preview display

This task connects the slider, frame extraction, preview inference, and checkerboard rendering to complete the feature.

**Files:**
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Add remaining imports**

At the top of `main_window.py`, add:

```python
import numpy as np

from src.core.frame_extractor import extract_frame
from src.core.compositing import render_checkerboard_preview
from src.gui.image_viewer import ImageViewerDialog
from src.worker.preview_worker import PreviewWorker
```

- [ ] **Step 2: Add `_display_frame` helper**

This converts a numpy RGB array to QPixmap and displays it in the preview label, scaled to fit:

```python
    def _display_frame(self, rgb: np.ndarray):
        """Display an RGB numpy array in the preview label, scaled to fit."""
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
```

- [ ] **Step 3: Add `_extract_and_show_frame` helper**

This is called when a video is loaded or the slider is released:

```python
    def _extract_and_show_frame(self, frame_number: int):
        """Extract a frame from the current video and display it."""
        if not self._video_info or not self._input_path:
            return
        try:
            rgb = extract_frame(
                self._input_path,
                frame_number,
                self._video_info["fps"],
                self._video_info["width"],
                self._video_info["height"],
            )
        except RuntimeError:
            return
        self._current_frame_rgb = rgb
        self._display_frame(rgb)
        # Update frame info label
        fps = self._video_info["fps"]
        total = self._video_info["frame_count"]
        t = frame_number / fps if fps > 0 else 0
        self._frame_info_label.setText(f"帧: {frame_number}/{total}  {t:.1f}s")
```

- [ ] **Step 4: Implement `_on_slider_released`**

Replace the stub:

```python
    def _on_slider_released(self):
        frame_number = self._frame_slider.value()
        self._extract_and_show_frame(frame_number)
```

- [ ] **Step 5: Wire video loading into preview**

In the `_handle_input` method, after the video info is displayed (after `self._settings_panel.set_source_bitrate(bitrate)`), add:

```python
            self._video_info = info
            self._frame_slider.setRange(0, max(0, info["frame_count"] - 1))
            self._frame_slider.setValue(0)
            self._extract_and_show_frame(0)
```

For image input (after `self._info_label.setText(f"图片信息: {w}x{h} | {img.mode}")`), add:

```python
            self._video_info = None
            self._frame_slider.setEnabled(False)
            self._frame_slider.setRange(0, 0)
            self._frame_info_label.setText("")
            # Display the image in preview pane
            rgb_array = np.array(img.convert("RGB"))
            self._current_frame_rgb = rgb_array
            self._display_frame(rgb_array)
```

For image folder input (after `self._info_label.setText(f"图片文件夹: {count} 张图片")`), add:

```python
            self._video_info = None
            self._frame_slider.setEnabled(False)
            self._frame_slider.setRange(0, 0)
            self._frame_info_label.setText("")
            self._preview_label.clear()
            self._preview_label.setText("文件夹模式无预览")
            self._current_frame_rgb = None
```

- [ ] **Step 6: Implement `_on_preview`**

Replace the stub with the full implementation:

```python
    def _on_preview(self):
        """Run single-frame matting preview on the current frame."""
        if self._current_frame_rgb is None:
            return
        if not self._model_tab.has_any_model():
            QMessageBox.warning(self, "提示", "请先在「模型管理」中下载至少一个模型")
            self._tabs.setCurrentWidget(self._model_tab)
            return

        config = self._get_config()
        models_dir = os.path.abspath(MODELS_DIR)

        # Check model exists
        from src.core.config import MODELS as MODEL_DIRS
        model_dir_name = MODEL_DIRS[config.model_name]
        model_path = os.path.join(models_dir, model_dir_name)
        if not os.path.isdir(model_path):
            QMessageBox.critical(self, "模型缺失", f"未找到 {config.model_name} 模型")
            return

        # Need BGR for predict (current_frame_rgb is RGB)
        frame_bgr = self._current_frame_rgb[:, :, ::-1].copy()

        self._preview_btn.setEnabled(False)
        self._status_label.setText("正在加载模型并预览...")

        self._preview_worker = PreviewWorker(
            model_name=config.model_name,
            models_dir=models_dir,
            frame=frame_bgr,
            resolution=config.inference_resolution.value,
        )
        self._preview_worker.finished.connect(self._on_preview_finished)
        self._preview_worker.error.connect(self._on_preview_error)
        self._preview_worker.start()

    def _on_preview_finished(self, alpha: np.ndarray):
        """Handle preview inference result."""
        self._preview_btn.setEnabled(True)
        self._status_label.setText("")
        # Render with checkerboard background
        result = render_checkerboard_preview(self._current_frame_rgb, alpha)
        self._display_frame(result)

    def _on_preview_error(self, message: str):
        """Handle preview inference error."""
        self._preview_btn.setEnabled(True)
        self._status_label.setText("")
        QMessageBox.critical(self, "预览失败", f"预览出错:\n{message}")
```

- [ ] **Step 7: Implement `_on_preview_clicked`**

Replace the stub to open the ImageViewerDialog:

```python
    def _on_preview_clicked(self, event):
        """Open the image viewer dialog when the preview pane is clicked."""
        if self._current_frame_rgb is None:
            return
        dialog = ImageViewerDialog(self._current_frame_rgb, parent=self)
        dialog.exec()
```

- [ ] **Step 8: Reset preview state in `_on_enqueue`**

In the `_on_enqueue` method, after `self._set_state("initial")`, add:

```python
        self._video_info = None
        self._current_frame_rgb = None
        self._preview_label.clear()
        self._preview_label.setText("选择视频后预览")
        self._frame_slider.setRange(0, 0)
        self._frame_info_label.setText("")
```

- [ ] **Step 9: Clean up preview worker on close**

In `closeEvent`, before `self._queue_manager.save()`, add:

```python
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.wait()
```

- [ ] **Step 10: Verify the app launches**

```bash
./venv/bin/python -c "from src.gui.main_window import MainWindow; print('OK')"
```

Expected: `OK`

- [ ] **Step 11: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: wire up frame extraction, preview inference, and image viewer"
```

---

### Task 7: Hide slider for image input and handle edge cases

**Files:**
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Hide slider and frame label for non-video input**

In `_handle_input`, for the image input branch, after setting `self._frame_slider.setEnabled(False)`, also hide the slider:

```python
            self._frame_slider.setVisible(False)
            self._frame_info_label.setVisible(False)
```

For the image folder branch, same:

```python
            self._frame_slider.setVisible(False)
            self._frame_info_label.setVisible(False)
```

For the video branch, ensure they are visible (add after `self._extract_and_show_frame(0)`):

```python
            self._frame_slider.setVisible(True)
            self._frame_info_label.setVisible(True)
```

- [ ] **Step 2: Also reset visibility in `_on_enqueue`**

After the existing preview state reset lines added in Task 6 Step 8, add:

```python
        self._frame_slider.setVisible(True)
        self._frame_info_label.setVisible(True)
```

- [ ] **Step 3: Handle the preview label displaying the checkerboard result for the viewer**

Currently `_on_preview_clicked` always passes `self._current_frame_rgb` (the original frame) to the viewer. After preview inference, the user sees the checkerboard result in the label, but clicking it would show the original. Fix by tracking the displayed image separately.

Add a new field in `__init__`:

```python
self._displayed_rgb = None  # what's currently shown in the preview label
```

In `_display_frame`, also store the image:

```python
    def _display_frame(self, rgb: np.ndarray):
        """Display an RGB numpy array in the preview label, scaled to fit."""
        self._displayed_rgb = rgb.copy()
        h, w, _ = rgb.shape
        qimage = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self._preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview_label.setPixmap(scaled)
```

In `_on_preview_clicked`, use `_displayed_rgb`:

```python
    def _on_preview_clicked(self, event):
        """Open the image viewer dialog when the preview pane is clicked."""
        if self._displayed_rgb is None:
            return
        dialog = ImageViewerDialog(self._displayed_rgb, parent=self)
        dialog.exec()
```

- [ ] **Step 4: Verify the app launches**

```bash
./venv/bin/python -c "from src.gui.main_window import MainWindow; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/gui/main_window.py
git commit -m "fix: hide slider for image input, show displayed image in viewer"
```

---

### Task 8: Run full test suite and final verification

**Files:** None (verification only)

- [ ] **Step 1: Run all tests**

```bash
./venv/bin/python -m pytest tests/ -x -q
```

Expected: All tests pass

- [ ] **Step 2: Verify app can be imported cleanly**

```bash
./venv/bin/python -c "
from src.gui.main_window import MainWindow
from src.core.frame_extractor import extract_frame
from src.core.compositing import render_checkerboard_preview
from src.worker.preview_worker import PreviewWorker
from src.gui.image_viewer import ImageViewerDialog
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 3: Final commit if any fixups needed**

Only if previous steps needed fixes:

```bash
git add -u
git commit -m "fix: address test/import issues in single-frame preview"
```

# P2 Batch Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a batch queue system with independent task settings, two-phase pipeline (inference mask cache + encoding), breakpoint resume via `.brm` project file, and a queue management UI tab.

**Architecture:** The existing single-task flow is refactored to a two-phase pipeline: phase 1 runs BiRefNet inference and caches alpha masks as PNGs, phase 2 reads cached masks + original frames to compose and encode the final video. A `QueueManager` orchestrates sequential task execution, persisting queue state to `~/.birefnet-gui/queue.brm` (JSON). The MainWindow gains a `QTabWidget` with the existing single-task view (Tab 1) and a new queue management view (Tab 2).

**Tech Stack:** PyQt6, Python dataclasses, JSON, cv2, numpy, PIL, threading

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/core/queue_task.py` | **NEW** — `QueueTask` dataclass, `TaskStatus` enum, `ProcessingPhase` enum, serialization helpers |
| `src/core/cache.py` | **NEW** — `MaskCacheManager`: save/load/validate/cleanup mask PNGs under `~/.birefnet-gui/cache/` |
| `src/core/pipeline.py` | **MODIFY** — Replace single `process()` with `infer_phase()` + `encode_phase()` |
| `src/core/image_pipeline.py` | **MODIFY** — Add resume support (skip already-processed images) |
| `src/worker/matting_worker.py` | **MODIFY** — Wire two-phase pipeline, new 3-arg progress signal `(current, total, phase)` |
| `src/core/queue_manager.py` | **NEW** — `QueueManager(QObject)`: add/remove/reorder tasks, .brm persistence, sequential execution via `MattingWorker` |
| `src/gui/main_window.py` | **MODIFY** — Wrap content in `QTabWidget`, add "加入队列" button, mutual exclusion |
| `src/gui/queue_tab.py` | **NEW** — Queue Tab: `QTableWidget` task list, progress bars, control buttons, drag-sort, context menu, drop zone |

---

### Task 1: QueueTask Data Model

**Files:**
- Create: `src/core/queue_task.py`
- Test: `tests/test_queue_task.py`

- [ ] **Step 1: Write the test file for QueueTask serialization**

```python
# tests/test_queue_task.py
import json
import time

from src.core.config import BackgroundMode, InputType, OutputFormat, ProcessingConfig
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus


class TestQueueTask:
    def test_create_default_queue_task(self):
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=ProcessingConfig(),
        )
        assert task.id  # non-empty UUID string
        assert task.input_path == "/tmp/video.mp4"
        assert task.input_type == InputType.VIDEO
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0
        assert task.total == 0
        assert task.phase == ProcessingPhase.INFERENCE
        assert task.error is None
        assert task.output_dir is None
        assert task.created_at > 0

    def test_to_dict_and_from_dict_roundtrip(self):
        config = ProcessingConfig(
            model_name="BiRefNet-lite",
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        task = QueueTask.create(
            input_path="/tmp/clip.mov",
            input_type=InputType.VIDEO,
            config=config,
            output_dir="/tmp/out",
        )
        task.status = TaskStatus.PROCESSING
        task.progress = 42
        task.total = 100
        task.phase = ProcessingPhase.ENCODING

        d = task.to_dict()
        restored = QueueTask.from_dict(d)

        assert restored.id == task.id
        assert restored.input_path == task.input_path
        assert restored.input_type == InputType.VIDEO
        assert restored.config.model_name == "BiRefNet-lite"
        assert restored.config.output_format == OutputFormat.MP4_H264
        assert restored.config.background_mode == BackgroundMode.GREEN
        assert restored.output_dir == "/tmp/out"
        assert restored.status == TaskStatus.PROCESSING
        assert restored.progress == 42
        assert restored.total == 100
        assert restored.phase == ProcessingPhase.ENCODING
        assert restored.created_at == task.created_at

    def test_to_dict_is_json_serializable(self):
        task = QueueTask.create(
            input_path="/tmp/img.png",
            input_type=InputType.IMAGE,
            config=ProcessingConfig(),
        )
        json_str = json.dumps(task.to_dict())
        assert isinstance(json_str, str)

    def test_from_dict_with_unknown_fields_ignores_them(self):
        config = ProcessingConfig()
        task = QueueTask.create(
            input_path="/tmp/a.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        d["unknown_future_field"] = "whatever"
        restored = QueueTask.from_dict(d)
        assert restored.id == task.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_queue_task.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.queue_task'`

- [ ] **Step 3: Implement QueueTask**

```python
# src/core/queue_task.py
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from src.core.config import (
    BackgroundMode,
    InputType,
    OutputFormat,
    ProcessingConfig,
)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPhase(Enum):
    INFERENCE = "inference"
    ENCODING = "encoding"
    DONE = "done"


@dataclass
class QueueTask:
    id: str
    input_path: str
    input_type: InputType
    config: ProcessingConfig
    output_dir: str | None
    status: TaskStatus
    progress: int
    total: int
    phase: ProcessingPhase
    error: str | None
    created_at: float

    @classmethod
    def create(
        cls,
        input_path: str,
        input_type: InputType,
        config: ProcessingConfig,
        output_dir: str | None = None,
    ) -> "QueueTask":
        return cls(
            id=uuid.uuid4().hex[:8],
            input_path=input_path,
            input_type=input_type,
            config=config,
            output_dir=output_dir,
            status=TaskStatus.PENDING,
            progress=0,
            total=0,
            phase=ProcessingPhase.INFERENCE,
            error=None,
            created_at=time.time(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_path": self.input_path,
            "input_type": self.input_type.value,
            "config": {
                "model_name": self.config.model_name,
                "output_format": self.config.output_format.value,
                "background_mode": self.config.background_mode.value,
            },
            "output_dir": self.output_dir,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "phase": self.phase.value,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueueTask":
        config = ProcessingConfig(
            model_name=d["config"]["model_name"],
            output_format=OutputFormat(d["config"]["output_format"]),
            background_mode=BackgroundMode(d["config"]["background_mode"]),
        )
        return cls(
            id=d["id"],
            input_path=d["input_path"],
            input_type=InputType(d["input_type"]),
            config=config,
            output_dir=d.get("output_dir"),
            status=TaskStatus(d["status"]),
            progress=d.get("progress", 0),
            total=d.get("total", 0),
            phase=ProcessingPhase(d.get("phase", "inference")),
            error=d.get("error"),
            created_at=d["created_at"],
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_queue_task.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/queue_task.py tests/test_queue_task.py
git commit -m "feat: add QueueTask data model with serialization"
```

---

### Task 2: MaskCacheManager

**Files:**
- Create: `src/core/cache.py`
- Test: `tests/test_cache.py`

- [ ] **Step 1: Write the test file for MaskCacheManager**

```python
# tests/test_cache.py
import os

import numpy as np
import pytest

from src.core.cache import MaskCacheManager


@pytest.fixture
def cache_manager(tmp_path):
    return MaskCacheManager(str(tmp_path))


class TestMaskCacheManager:
    def test_save_and_load_mask_roundtrip(self, cache_manager):
        mask = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cache_manager.save_mask("task1", 0, mask)
        loaded = cache_manager.load_mask("task1", 0)
        np.testing.assert_array_equal(mask, loaded)

    def test_save_creates_directory_structure(self, cache_manager, tmp_path):
        mask = np.zeros((32, 32), dtype=np.uint8)
        cache_manager.save_mask("abc", 5, mask)
        assert os.path.exists(os.path.join(str(tmp_path), "abc", "masks", "000005.png"))

    def test_get_cached_count_empty(self, cache_manager):
        assert cache_manager.get_cached_count("nonexistent") == 0

    def test_get_cached_count_after_saves(self, cache_manager):
        mask = np.zeros((16, 16), dtype=np.uint8)
        for i in range(5):
            cache_manager.save_mask("task2", i, mask)
        assert cache_manager.get_cached_count("task2") == 5

    def test_save_and_validate_metadata(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        cache_manager.save_metadata("t1", info)
        assert cache_manager.validate("t1", info) is True

    def test_validate_fails_on_mismatch(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        cache_manager.save_metadata("t1", info)
        different = {**info, "frame_count": 200}
        assert cache_manager.validate("t1", different) is False

    def test_validate_returns_false_when_no_metadata(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        assert cache_manager.validate("missing", info) is False

    def test_cleanup_removes_task_dir(self, cache_manager, tmp_path):
        mask = np.zeros((16, 16), dtype=np.uint8)
        cache_manager.save_mask("t1", 0, mask)
        assert os.path.isdir(os.path.join(str(tmp_path), "t1"))
        cache_manager.cleanup("t1")
        assert not os.path.isdir(os.path.join(str(tmp_path), "t1"))

    def test_cleanup_nonexistent_is_noop(self, cache_manager):
        cache_manager.cleanup("ghost")  # should not raise

    def test_cleanup_all(self, cache_manager, tmp_path):
        mask = np.zeros((16, 16), dtype=np.uint8)
        cache_manager.save_mask("a", 0, mask)
        cache_manager.save_mask("b", 0, mask)
        cache_manager.cleanup_all()
        assert len(os.listdir(str(tmp_path))) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.cache'`

- [ ] **Step 3: Implement MaskCacheManager**

```python
# src/core/cache.py
import json
import os
import shutil

import cv2
import numpy as np


class MaskCacheManager:
    """Manages alpha mask cache on disk for breakpoint resume."""

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

    def save_mask(self, task_id: str, frame_idx: int, mask: np.ndarray) -> None:
        masks_dir = os.path.join(self._cache_dir, task_id, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(path, mask)

    def load_mask(self, task_id: str, frame_idx: int) -> np.ndarray:
        path = os.path.join(self._cache_dir, task_id, "masks", f"{frame_idx:06d}.png")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cached mask not found: {path}")
        return mask

    def get_cached_count(self, task_id: str) -> int:
        masks_dir = os.path.join(self._cache_dir, task_id, "masks")
        if not os.path.isdir(masks_dir):
            return 0
        return len([f for f in os.listdir(masks_dir) if f.endswith(".png")])

    def save_metadata(self, task_id: str, video_info: dict) -> None:
        task_dir = os.path.join(self._cache_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        path = os.path.join(task_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(video_info, f)

    def validate(self, task_id: str, video_info: dict) -> bool:
        path = os.path.join(self._cache_dir, task_id, "metadata.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            cached = json.load(f)
        return cached == video_info

    def cleanup(self, task_id: str) -> None:
        task_dir = os.path.join(self._cache_dir, task_id)
        if os.path.isdir(task_dir):
            shutil.rmtree(task_dir)

    def cleanup_all(self) -> None:
        if not os.path.isdir(self._cache_dir):
            return
        for entry in os.listdir(self._cache_dir):
            path = os.path.join(self._cache_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/cache.py tests/test_cache.py
git commit -m "feat: add MaskCacheManager for alpha mask caching"
```

---

### Task 3: Two-Phase Pipeline Refactor

**Files:**
- Modify: `src/core/pipeline.py`
- Test: `tests/test_pipeline.py` (modify existing + add new tests)

This task replaces `MattingPipeline.process()` with `infer_phase()` + `encode_phase()`, keeping a convenience `process()` that calls both sequentially.

- [ ] **Step 1: Write new two-phase tests**

Add these tests to `tests/test_pipeline.py`, keeping the existing tests for now (they will be updated in step 5):

```python
# Append to tests/test_pipeline.py

import tempfile

from src.core.cache import MaskCacheManager


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestTwoPhasePipeline:
    def test_infer_phase_creates_cached_masks(self, test_video_path):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)

            progress_log = []
            pipeline.infer_phase(
                input_path=test_video_path,
                task_id="test1",
                cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )

            assert cache.get_cached_count("test1") == 10
            assert all(p == "inference" for _, _, p in progress_log)
            assert progress_log[-1] == (10, 10, "inference")

            # Verify masks are loadable and correct shape
            mask = cache.load_mask("test1", 0)
            assert mask.shape == (64, 64)
            assert mask.dtype == np.uint8

    def test_encode_phase_produces_video(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)

            # First run infer phase to populate cache
            pipeline.infer_phase(
                input_path=test_video_path,
                task_id="test2",
                cache=cache,
            )

            # Then run encode phase
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline.encode_phase(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test2",
                cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )

            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10
            assert all(p == "encoding" for _, _, p in progress_log)

    def test_infer_phase_resumes_from_start_frame(self, test_video_path):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)

            # Infer first 5 frames
            cancel = threading.Event()
            def stop_at_5(c, t, p):
                if c >= 5:
                    cancel.set()

            try:
                pipeline.infer_phase(
                    input_path=test_video_path,
                    task_id="resume_test",
                    cache=cache,
                    progress_callback=stop_at_5,
                    cancel_event=cancel,
                )
            except InterruptedError:
                pass

            first_count = cache.get_cached_count("resume_test")
            assert first_count >= 5

            # Resume from where we left off
            pipeline.infer_phase(
                input_path=test_video_path,
                task_id="resume_test",
                cache=cache,
                start_frame=first_count,
            )

            assert cache.get_cached_count("resume_test") == 10

    def test_process_convenience_runs_both_phases(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")

            progress_log = []
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="conv_test",
                cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )

            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10

            # Both phases should appear in progress log
            phases = set(p for _, _, p in progress_log)
            assert phases == {"inference", "encoding"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pipeline.py::TestTwoPhasePipeline -v`
Expected: FAIL — `infer_phase` / `encode_phase` methods don't exist yet

- [ ] **Step 3: Implement two-phase pipeline**

Replace `src/core/pipeline.py` with:

```python
# src/core/pipeline.py
import os
import threading
import time
from typing import Callable, Optional

from src.core.cache import MaskCacheManager
from src.core.compositing import compose_frame
from src.core.config import ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict
from src.core.video import FrameReader, get_video_info
from src.core.writer import create_writer


class MattingPipeline:
    """Two-phase video matting: inference (caches masks) then encoding."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def infer_phase(
        self,
        input_path: str,
        task_id: str,
        cache: MaskCacheManager,
        start_frame: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Run BiRefNet inference and cache alpha masks as PNGs.

        Args:
            input_path: Path to input video file.
            task_id: Unique task identifier for cache directory.
            cache: MaskCacheManager instance.
            start_frame: Frame index to resume from (0-based).
            progress_callback: Called with (current_frame, total_frames, "inference").
            pause_event: When set, processing pauses.
            cancel_event: When set, processing stops with InterruptedError.
        """
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        cache.save_metadata(task_id, video_info)

        for idx, frame in enumerate(FrameReader(input_path)):
            if idx < start_frame:
                continue

            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")
            if pause_event:
                while pause_event.is_set():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Processing cancelled by user")
                    time.sleep(0.1)

            alpha = predict(self._model, frame, self._device)
            cache.save_mask(task_id, idx, alpha)

            if progress_callback:
                progress_callback(idx + 1, total, "inference")

    def encode_phase(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Read original frames + cached masks, compose and encode output.

        Args:
            input_path: Path to input video file.
            output_path: Path for output file or directory.
            task_id: Unique task identifier matching infer_phase.
            cache: MaskCacheManager instance.
            progress_callback: Called with (current_frame, total_frames, "encoding").
            pause_event: When set, processing pauses.
            cancel_event: When set, processing stops with InterruptedError.
        """
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        width, height, fps = video_info["width"], video_info["height"], video_info["fps"]

        writer = create_writer(
            self._config, output_path, width, height, fps,
            audio_source=input_path,
        )

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

                if progress_callback:
                    progress_callback(idx + 1, total, "encoding")

        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")

    def process(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        start_frame: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Convenience method: run infer_phase then encode_phase."""
        self.infer_phase(
            input_path, task_id, cache, start_frame,
            progress_callback, pause_event, cancel_event,
        )
        self.encode_phase(
            input_path, output_path, task_id, cache,
            progress_callback, pause_event, cancel_event,
        )
```

- [ ] **Step 4: Update existing tests to use new API**

The existing `TestMattingPipeline` tests call `pipeline.process()` with the old signature `(input_path, output_path, progress_callback, pause_event, cancel_event)`. Update them to use the new signature with `task_id` and `cache`. Replace the entire `TestMattingPipeline` class in `tests/test_pipeline.py`:

```python
# Replace TestMattingPipeline in tests/test_pipeline.py
import tempfile

from src.core.cache import MaskCacheManager


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_prores_transparent(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []

            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test_prores",
                cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )

            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10
            assert info["width"] == 64
            assert info["height"] == 64
            assert len(progress_log) == 20  # 10 inference + 10 encoding

    def test_processes_video_h264_green(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.MP4_H264,
                background_mode=BackgroundMode.GREEN,
            )
            output_path = os.path.join(temp_output_dir, "output.mp4")

            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test_h264",
                cache=cache,
            )

            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10

    def test_processes_video_mask_bw(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.MP4_H264,
                background_mode=BackgroundMode.MASK_BW,
            )
            output_path = os.path.join(temp_output_dir, "output.mp4")

            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test_mask",
                cache=cache,
            )

            assert os.path.exists(output_path)

    def test_processes_png_sequence(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.PNG_SEQUENCE,
                background_mode=BackgroundMode.TRANSPARENT,
            )
            output_path = os.path.join(temp_output_dir, "seq_output")

            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test_seq",
                cache=cache,
            )

            files = sorted(os.listdir(output_path))
            assert len(files) == 10
            assert files[0] == "frame_000001.png"

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")
            cancel_event = threading.Event()
            frame_count = []

            def on_progress(current, total, phase):
                frame_count.append(current)
                if current >= 3 and phase == "inference":
                    cancel_event.set()

            pipeline = MattingPipeline(config, MODELS_DIR)
            with pytest.raises(InterruptedError):
                pipeline.process(
                    input_path=test_video_path,
                    output_path=output_path,
                    task_id="test_cancel",
                    cache=cache,
                    progress_callback=on_progress,
                    cancel_event=cancel_event,
                )

            assert len(frame_count) < 20  # Did not finish all inference + encoding
```

- [ ] **Step 5: Run all pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -v`
Expected: All tests PASS (both TestMattingPipeline and TestTwoPhasePipeline)

- [ ] **Step 6: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: refactor MattingPipeline to two-phase (infer + encode) with mask cache"
```

---

### Task 4: MattingWorker Two-Phase Adaptation

**Files:**
- Modify: `src/worker/matting_worker.py`
- Modify: `src/gui/main_window.py` (update progress signal handler)

- [ ] **Step 1: Update MattingWorker to use two-phase pipeline**

Replace `src/worker/matting_worker.py`:

```python
# src/worker/matting_worker.py
import os
import threading
import time
import uuid

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.cache import MaskCacheManager
from src.core.config import InputType, ProcessingConfig

# Default cache directory
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "cache")


class MattingWorker(QThread):
    """Runs the matting pipeline in a background thread.

    Signals:
        progress(int, int, str): (current_frame, total_frames, phase)
            phase is "inference" or "encoding" for video, "processing" for images
        speed(float): frames per second
        finished(str): output file path on success
        error(str): error message on failure
    """

    progress = pyqtSignal(int, int, str)
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
        task_id: str | None = None,
        start_frame: int = 0,
        cleanup_cache: bool = True,
    ):
        super().__init__()
        self._config = config
        self._models_dir = models_dir
        self._input_path = input_path
        self._output_path = output_path
        self._input_type = input_type
        self._task_id = task_id or uuid.uuid4().hex[:8]
        self._start_frame = start_frame
        self._cleanup_cache = cleanup_cache

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._last_time = None

    @property
    def task_id(self) -> str:
        return self._task_id

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

        cache = MaskCacheManager(CACHE_DIR)
        pipeline = MattingPipeline(self._config, self._models_dir)
        pipeline.process(
            input_path=self._input_path,
            output_path=self._output_path,
            task_id=self._task_id,
            cache=cache,
            start_frame=self._start_frame,
            progress_callback=self._on_progress,
            pause_event=self._pause_event,
            cancel_event=self._cancel_event,
        )
        if self._cleanup_cache:
            cache.cleanup(self._task_id)

    def _run_image(self):
        from src.core.image_pipeline import ImagePipeline

        pipeline = ImagePipeline(self._config, self._models_dir)
        result = pipeline.process(
            input_path=self._input_path,
            output_dir=self._output_path,
            progress_callback=lambda c, t: self._on_progress(c, t, "processing"),
            pause_event=self._pause_event,
            cancel_event=self._cancel_event,
        )
        self._output_path = result

    def _on_progress(self, current: int, total: int, phase: str):
        self.progress.emit(current, total, phase)
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

- [ ] **Step 2: Update MainWindow progress handler**

In `src/gui/main_window.py`, update `_on_progress` to accept the new 3-arg signal and display the phase. Also update the signal connection.

Find and replace the `_on_progress` method and the signal connection in `_on_start`:

In `_on_start`, change the connect line:
```python
# Old:
self._worker.progress.connect(self._on_progress)
# New:
self._worker.progress.connect(self._on_progress)
```
(The connect line itself doesn't change, but the slot signature does.)

Replace the `_on_progress` method:
```python
    def _on_progress(self, current: int, total: int, phase: str):
        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            phase_label = "推理中" if phase == "inference" else "编码中" if phase == "encoding" else "处理中"
            self._status_label.setText(
                f"{phase_label}: {current}/{total} | {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )
```

Also reset `_start_time` when phase changes. Add a `_current_phase` attribute. In `__init__`, add `self._current_phase = None`. Then update `_on_progress`:

```python
    def _on_progress(self, current: int, total: int, phase: str):
        # Reset timer when phase changes
        if phase != self._current_phase:
            self._current_phase = phase
            self._start_time = time.time()

        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            phase_label = {"inference": "推理中", "encoding": "编码中", "processing": "处理中"}.get(phase, phase)
            self._status_label.setText(
                f"{phase_label}: {current}/{total} | {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )
```

- [ ] **Step 3: Manually verify the GUI launches without errors**

Run: `python main.py`
Expected: App launches, no import errors. Select a video and process to verify both inference and encoding phases display in the status label.

- [ ] **Step 4: Commit**

```bash
git add src/worker/matting_worker.py src/gui/main_window.py
git commit -m "feat: wire MattingWorker to two-phase pipeline with phase-aware progress"
```

---

### Task 5: QueueManager

**Files:**
- Create: `src/core/queue_manager.py`
- Test: `tests/test_queue_manager.py`

- [ ] **Step 1: Write tests for QueueManager**

```python
# tests/test_queue_manager.py
import json
import os

import pytest

from src.core.config import InputType, OutputFormat, ProcessingConfig
from src.core.queue_manager import QueueManager
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus


@pytest.fixture
def qm(tmp_path):
    brm_path = os.path.join(str(tmp_path), "queue.brm")
    return QueueManager(brm_path=brm_path)


def _make_task(input_path="/tmp/video.mp4", **kwargs) -> QueueTask:
    return QueueTask.create(
        input_path=input_path,
        input_type=kwargs.get("input_type", InputType.VIDEO),
        config=kwargs.get("config", ProcessingConfig()),
        output_dir=kwargs.get("output_dir"),
    )


class TestQueueManagerTaskList:
    def test_add_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        assert len(qm.tasks) == 1
        assert qm.tasks[0].id == t.id

    def test_remove_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        qm.remove_task(t.id)
        assert len(qm.tasks) == 0

    def test_remove_nonexistent_is_noop(self, qm):
        qm.remove_task("ghost")  # should not raise

    def test_move_task(self, qm):
        t1 = _make_task("/tmp/a.mp4")
        t2 = _make_task("/tmp/b.mp4")
        t3 = _make_task("/tmp/c.mp4")
        qm.add_task(t1)
        qm.add_task(t2)
        qm.add_task(t3)

        qm.move_task(t3.id, 0)
        assert [t.id for t in qm.tasks] == [t3.id, t1.id, t2.id]

    def test_move_task_to_end(self, qm):
        t1 = _make_task("/tmp/a.mp4")
        t2 = _make_task("/tmp/b.mp4")
        qm.add_task(t1)
        qm.add_task(t2)

        qm.move_task(t1.id, 1)
        assert [t.id for t in qm.tasks] == [t2.id, t1.id]

    def test_clear_all(self, qm):
        qm.add_task(_make_task("/tmp/a.mp4"))
        qm.add_task(_make_task("/tmp/b.mp4"))
        qm.clear_all()
        assert len(qm.tasks) == 0

    def test_get_task(self, qm):
        t = _make_task()
        qm.add_task(t)
        found = qm.get_task(t.id)
        assert found is t

    def test_get_task_missing_returns_none(self, qm):
        assert qm.get_task("nope") is None


class TestQueueManagerPersistence:
    def test_save_and_load(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "queue.brm")
        qm1 = QueueManager(brm_path=brm_path)

        t1 = _make_task("/tmp/a.mp4")
        t1.status = TaskStatus.COMPLETED
        t2 = _make_task("/tmp/b.mp4")
        t2.status = TaskStatus.PROCESSING
        t2.progress = 50
        t2.total = 100
        qm1.add_task(t1)
        qm1.add_task(t2)
        qm1.save()

        assert os.path.exists(brm_path)

        qm2 = QueueManager(brm_path=brm_path)
        qm2.load()

        assert len(qm2.tasks) == 2
        assert qm2.tasks[0].id == t1.id
        assert qm2.tasks[0].status == TaskStatus.COMPLETED
        assert qm2.tasks[1].progress == 50

    def test_load_nonexistent_file_is_noop(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "nope.brm")
        qm = QueueManager(brm_path=brm_path)
        qm.load()  # should not raise
        assert len(qm.tasks) == 0

    def test_save_creates_parent_dirs(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "sub", "dir", "queue.brm")
        qm = QueueManager(brm_path=brm_path)
        qm.add_task(_make_task())
        qm.save()
        assert os.path.exists(brm_path)

    def test_processing_tasks_saved_as_paused(self, tmp_path):
        brm_path = os.path.join(str(tmp_path), "queue.brm")
        qm1 = QueueManager(brm_path=brm_path)
        t = _make_task()
        t.status = TaskStatus.PROCESSING
        qm1.add_task(t)
        qm1.save()

        qm2 = QueueManager(brm_path=brm_path)
        qm2.load()
        # PROCESSING should be saved as PAUSED so it can resume
        assert qm2.tasks[0].status == TaskStatus.PAUSED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_queue_manager.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.queue_manager'`

- [ ] **Step 3: Implement QueueManager**

```python
# src/core/queue_manager.py
import json
import os
from datetime import datetime

from src.core.queue_task import QueueTask, TaskStatus


class QueueManager:
    """Manages a list of QueueTasks with persistence to a .brm file."""

    def __init__(self, brm_path: str):
        self._brm_path = brm_path
        self._tasks: list[QueueTask] = []

    @property
    def tasks(self) -> list[QueueTask]:
        return list(self._tasks)

    def add_task(self, task: QueueTask) -> None:
        self._tasks.append(task)

    def remove_task(self, task_id: str) -> None:
        self._tasks = [t for t in self._tasks if t.id != task_id]

    def get_task(self, task_id: str) -> QueueTask | None:
        for t in self._tasks:
            if t.id == task_id:
                return t
        return None

    def move_task(self, task_id: str, new_index: int) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        self._tasks.remove(task)
        new_index = max(0, min(new_index, len(self._tasks)))
        self._tasks.insert(new_index, task)

    def clear_all(self) -> None:
        self._tasks.clear()

    def next_pending_task(self) -> QueueTask | None:
        for t in self._tasks:
            if t.status in (TaskStatus.PENDING, TaskStatus.PAUSED):
                return t
        return None

    def save(self) -> None:
        os.makedirs(os.path.dirname(self._brm_path), exist_ok=True)
        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "tasks": [],
        }
        for task in self._tasks:
            d = task.to_dict()
            # Save PROCESSING as PAUSED so it can resume on reload
            if d["status"] == TaskStatus.PROCESSING.value:
                d["status"] = TaskStatus.PAUSED.value
            data["tasks"].append(d)

        with open(self._brm_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        if not os.path.exists(self._brm_path):
            return
        with open(self._brm_path) as f:
            data = json.load(f)
        self._tasks = [QueueTask.from_dict(d) for d in data.get("tasks", [])]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_queue_manager.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/queue_manager.py tests/test_queue_manager.py
git commit -m "feat: add QueueManager with task list operations and .brm persistence"
```

---

### Task 6: MainWindow Tab Conversion + "加入队列" Button

**Files:**
- Modify: `src/gui/main_window.py`

This task wraps the existing MainWindow content into Tab 1 inside a QTabWidget, adds the "加入队列" button, and creates a placeholder Tab 2 (the full Queue Tab is Task 7).

- [ ] **Step 1: Add QTabWidget imports and QueueManager setup**

In `src/gui/main_window.py`, add to the imports at the top:

```python
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
    QTabWidget,        # ADD
    QVBoxLayout,
    QWidget,
)
```

And add import for QueueManager and QueueTask:

```python
from src.core.queue_manager import QueueManager
from src.core.queue_task import QueueTask
```

Add a constant for the default .brm path:

```python
import os

BRM_PATH = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "queue.brm")
```

- [ ] **Step 2: Refactor `__init__` and `_init_ui` to use QTabWidget**

In `__init__`, add queue_manager init and current_phase tracking:
```python
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(750, 500)

        self._worker = None
        self._input_path = None
        self._output_dir = None
        self._start_time = None
        self._input_type = None
        self._current_phase = None

        self._queue_manager = QueueManager(brm_path=BRM_PATH)
        self._queue_manager.load()

        self._init_ui()
        self.setAcceptDrops(True)
        self._set_state("initial")
```

In `_init_ui`, wrap existing content in a tab widget. Replace the current `_init_ui` method. The key structural change: the existing left+right panel layout becomes the content of Tab 1, and Tab 2 is a placeholder `QWidget`:

```python
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        outer_layout.addWidget(self._tabs)

        # --- Tab 1: Single Task ---
        tab1 = QWidget()
        self._tabs.addTab(tab1, "单任务")

        main_layout = QHBoxLayout(tab1)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ... (left_panel and right_panel code stays exactly the same as current _init_ui) ...
```

Everything after `main_layout = QHBoxLayout(tab1)` stays the same as the current code **except** the "加入队列" button is added to the button row.

- [ ] **Step 3: Add "加入队列" button to the control button row**

In the control button row section (after `self._start_btn`), add:

```python
        self._enqueue_btn = QPushButton("加入队列")
        self._enqueue_btn.clicked.connect(self._on_enqueue)
        btn_row.addWidget(self._enqueue_btn)
```

- [ ] **Step 4: Add Tab 2 placeholder**

After Tab 1 setup, at the end of `_init_ui`:

```python
        # --- Tab 2: Queue (placeholder, replaced in Task 7) ---
        self._queue_tab = QWidget()
        self._queue_tab_layout = QVBoxLayout(self._queue_tab)
        self._queue_tab_layout.addWidget(QLabel("队列功能加载中..."))
        self._update_queue_tab_title()
        self._tabs.addTab(self._queue_tab, self._get_queue_tab_title())
```

- [ ] **Step 5: Implement helper methods**

```python
    def _get_queue_tab_title(self) -> str:
        count = len(self._queue_manager.tasks)
        if count > 0:
            return f"批量队列 ({count})"
        return "批量队列"

    def _update_queue_tab_title(self):
        self._tabs.setTabText(1, self._get_queue_tab_title())

    def _on_enqueue(self):
        if not self._input_path or not self._input_type:
            QMessageBox.warning(self, "提示", "请先选择输入文件")
            return

        config = self._get_config()
        task = QueueTask.create(
            input_path=self._input_path,
            input_type=self._input_type,
            config=config,
            output_dir=self._output_dir,
        )
        self._queue_manager.add_task(task)
        self._queue_manager.save()
        self._update_queue_tab_title()

        # Clear input for next add
        self._input_path = None
        self._input_type = None
        self._input_edit.setText("")
        self._info_label.setText("")
        self._output_dir = None
        self._output_edit.setText("")
        self._set_state("initial")

        self.statusBar().showMessage("已加入队列", 3000)
```

- [ ] **Step 6: Update `_set_state` to also control the enqueue button**

In `_set_state`, add `self._enqueue_btn.setEnabled(...)` for each state:
- "initial": disabled (no file selected)
- "ready": enabled
- "processing": disabled
- "paused": disabled
- "finished": enabled

```python
    def _set_state(self, state: str):
        self._state = state
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
```

- [ ] **Step 7: Add closeEvent for auto-save**

```python
    def closeEvent(self, event):
        self._queue_manager.save()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        event.accept()
```

- [ ] **Step 8: Manually verify GUI launches**

Run: `python main.py`
Expected: App shows two tabs ("单任务" and "批量队列"). Select a file, click "加入队列", see tab title update to "批量队列 (1)", input clears.

- [ ] **Step 9: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: add QTabWidget with single-task tab and enqueue button"
```

---

### Task 7: Queue Tab UI

**Files:**
- Create: `src/gui/queue_tab.py`
- Modify: `src/gui/main_window.py` (replace placeholder Tab 2)

- [ ] **Step 1: Create QueueTab widget**

```python
# src/gui/queue_tab.py
import os
import time

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.cache import MaskCacheManager
from src.core.config import IMAGE_EXTENSIONS, MODELS, VIDEO_EXTENSIONS, InputType
from src.core.queue_manager import QueueManager
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus
from src.worker.matting_worker import CACHE_DIR, MattingWorker


class QueueTab(QWidget):
    """Queue management tab with task list, progress, and controls."""

    # Emitted when queue starts/stops (for MainWindow to disable Tab 1 controls)
    queue_running_changed = pyqtSignal(bool)
    # Emitted when task count changes (for MainWindow to update tab title)
    task_count_changed = pyqtSignal(int)

    def __init__(self, queue_manager: QueueManager, get_default_config_fn, parent=None):
        super().__init__(parent)
        self._qm = queue_manager
        self._get_default_config = get_default_config_fn
        self._current_worker: MattingWorker | None = None
        self._cache = MaskCacheManager(CACHE_DIR)
        self._start_time: float | None = None
        self._current_phase: str | None = None

        self.setAcceptDrops(True)
        self._init_ui()
        self._refresh_table()
        self._set_queue_state("idle")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Task table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["文件名", "模型", "格式", "状态"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        self._table.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self._table)

        # Current task progress
        progress_group = QGroupBox("当前任务")
        progress_layout = QVBoxLayout(progress_group)

        self._current_label = QLabel("")
        self._current_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._current_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        progress_layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._status_label)

        # Queue total progress
        self._total_label = QLabel("")
        self._total_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._total_label)

        layout.addWidget(progress_group)

        # Control buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._start_btn = QPushButton("开始队列")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消当前")
        self._cancel_btn.clicked.connect(self._on_cancel_current)
        btn_row.addWidget(self._cancel_btn)

        self._clear_btn = QPushButton("清空队列")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _set_queue_state(self, state: str):
        """state: 'idle', 'running', 'paused'"""
        self._queue_state = state
        is_idle = state == "idle"
        self._start_btn.setEnabled(is_idle and len(self._qm.tasks) > 0)
        self._pause_btn.setEnabled(state in ("running", "paused"))
        self._pause_btn.setText("继续" if state == "paused" else "暂停")
        self._cancel_btn.setEnabled(not is_idle)
        self._clear_btn.setEnabled(is_idle)
        self.queue_running_changed.emit(not is_idle)

    def _refresh_table(self):
        self._table.setRowCount(0)
        for task in self._qm.tasks:
            row = self._table.rowCount()
            self._table.insertRow(row)
            self._table.setItem(row, 0, QTableWidgetItem(os.path.basename(task.input_path)))
            model_dir = MODELS.get(task.config.model_name, task.config.model_name)
            self._table.setItem(row, 1, QTableWidgetItem(model_dir))
            self._table.setItem(row, 2, QTableWidgetItem(task.config.output_format.value))
            self._table.setItem(row, 3, QTableWidgetItem(self._status_text(task)))
        self.task_count_changed.emit(len(self._qm.tasks))

    def _status_text(self, task: QueueTask) -> str:
        if task.status == TaskStatus.COMPLETED:
            return "✅ 完成"
        if task.status == TaskStatus.FAILED:
            return f"❌ 失败"
        if task.status == TaskStatus.CANCELLED:
            return "⏹ 已取消"
        if task.status == TaskStatus.PROCESSING:
            phase = "推理" if task.phase == ProcessingPhase.INFERENCE else "编码"
            if task.total > 0:
                return f"▶ {phase} {task.progress}/{task.total}"
            return f"▶ {phase}中..."
        if task.status == TaskStatus.PAUSED:
            if task.total > 0:
                return f"⏸ 暂停 {task.progress}/{task.total}"
            return "⏸ 暂停"
        return "⏳ 等待"

    def _show_context_menu(self, pos):
        row = self._table.rowAt(pos.y())
        if row < 0 or row >= len(self._qm.tasks):
            return

        task = self._qm.tasks[row]
        if task.status == TaskStatus.PROCESSING:
            return  # Cannot modify running task

        menu = QMenu(self)
        menu.addAction("删除", lambda: self._remove_task(task.id))
        if row > 0:
            menu.addAction("移到顶部", lambda: self._move_task(task.id, 0))
        if row < len(self._qm.tasks) - 1:
            menu.addAction("移到底部", lambda: self._move_task(task.id, len(self._qm.tasks) - 1))
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _remove_task(self, task_id: str):
        self._qm.remove_task(task_id)
        self._cache.cleanup(task_id)
        self._qm.save()
        self._refresh_table()

    def _move_task(self, task_id: str, new_index: int):
        self._qm.move_task(task_id, new_index)
        self._qm.save()
        self._refresh_table()

    def _on_rows_moved(self):
        # Sync QueueManager order with table row order
        new_order = []
        for row in range(self._table.rowCount()):
            filename = self._table.item(row, 0).text()
            for task in self._qm.tasks:
                if os.path.basename(task.input_path) == filename and task not in new_order:
                    new_order.append(task)
                    break
        self._qm._tasks = new_order
        self._qm.save()

    # --- External drag-drop ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if not path:
                continue
            input_type = self._classify_input(path)
            if input_type is None:
                continue
            config = self._get_default_config()
            task = QueueTask.create(
                input_path=path,
                input_type=input_type,
                config=config,
            )
            self._qm.add_task(task)
        self._qm.save()
        self._refresh_table()

    def _classify_input(self, path: str) -> InputType | None:
        if os.path.isdir(path):
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            )
            return InputType.IMAGE_FOLDER if has_images else None
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return InputType.VIDEO
        if ext in IMAGE_EXTENSIONS:
            return InputType.IMAGE
        return None

    # --- Queue execution ---
    def _on_start(self):
        self._run_next_task()

    def _run_next_task(self):
        task = self._qm.next_pending_task()
        if task is None:
            self._on_queue_finished()
            return

        task.status = TaskStatus.PROCESSING
        self._qm.save()
        self._refresh_table()
        self._set_queue_state("running")

        # Determine start_frame from cache
        start_frame = 0
        if task.input_type == InputType.VIDEO and task.phase == ProcessingPhase.INFERENCE:
            start_frame = self._cache.get_cached_count(task.id)

        output_path = self._build_output_path(task)

        self._current_worker = MattingWorker(
            config=task.config,
            models_dir=os.path.join(os.path.dirname(__file__), "..", "..", "models"),
            input_path=task.input_path,
            output_path=output_path,
            input_type=task.input_type,
            task_id=task.id,
            start_frame=start_frame,
            cleanup_cache=False,  # Queue manages cache lifecycle
        )
        self._current_worker.progress.connect(
            lambda c, t, p: self._on_task_progress(task.id, c, t, p)
        )
        self._current_worker.finished.connect(
            lambda path: self._on_task_finished(task.id, path)
        )
        self._current_worker.error.connect(
            lambda msg: self._on_task_error(task.id, msg)
        )

        self._start_time = time.time()
        self._current_phase = None
        self._current_label.setText(f"当前: {os.path.basename(task.input_path)}")
        self._progress_bar.setValue(0)
        self._current_worker.start()

    def _build_output_path(self, task: QueueTask) -> str:
        from src.gui.main_window import FORMAT_EXTENSIONS
        model_dir = MODELS[task.config.model_name]
        timestamp = int(time.time() * 1000)

        if task.input_type == InputType.VIDEO:
            base_name = os.path.splitext(os.path.basename(task.input_path))[0]
            ext = FORMAT_EXTENSIONS.get(task.config.output_format, ".mov")
            filename = f"{base_name}_{model_dir}_{timestamp}{ext}"
            out_dir = task.output_dir or os.path.dirname(task.input_path)
            return os.path.join(out_dir, filename)
        else:
            return task.output_dir or os.path.dirname(task.input_path)

    def _on_task_progress(self, task_id: str, current: int, total: int, phase: str):
        task = self._qm.get_task(task_id)
        if task is None:
            return

        task.progress = current
        task.total = total
        task.phase = ProcessingPhase.INFERENCE if phase == "inference" else ProcessingPhase.ENCODING

        # Reset timer on phase change
        if phase != self._current_phase:
            self._current_phase = phase
            self._start_time = time.time()

        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            phase_label = {"inference": "推理中", "encoding": "编码中", "processing": "处理中"}.get(phase, phase)
            self._status_label.setText(
                f"{phase_label}: {current}/{total} | {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )

        # Update total queue progress
        self._update_total_progress()

        # Update table row status
        self._refresh_table()

        # Throttled save (save is cheap for JSON)
        self._qm.save()

    def _update_total_progress(self):
        completed_phases = 0
        total_phases = 0
        for t in self._qm.tasks:
            if t.input_type == InputType.VIDEO:
                weight = max(t.total, 1) * 2  # inference + encoding
                total_phases += weight
                if t.status == TaskStatus.COMPLETED:
                    completed_phases += weight
                elif t.status == TaskStatus.PROCESSING:
                    done = t.progress
                    if t.phase == ProcessingPhase.ENCODING:
                        done += t.total  # inference phase was complete
                    completed_phases += done
            else:
                weight = max(t.total, 1)
                total_phases += weight
                if t.status == TaskStatus.COMPLETED:
                    completed_phases += weight
                elif t.status == TaskStatus.PROCESSING:
                    completed_phases += t.progress

        task_count = len(self._qm.tasks)
        completed_count = sum(1 for t in self._qm.tasks if t.status == TaskStatus.COMPLETED)
        if total_phases > 0:
            pct = int(completed_phases / total_phases * 100)
            self._total_label.setText(f"队列: 任务 {completed_count + 1}/{task_count} — 总进度 {pct}%")

    def _on_task_finished(self, task_id: str, output_path: str):
        task = self._qm.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.phase = ProcessingPhase.DONE
            self._cache.cleanup(task_id)
        self._qm.save()
        self._refresh_table()
        self._current_worker = None
        self._run_next_task()

    def _on_task_error(self, task_id: str, message: str):
        task = self._qm.get_task(task_id)
        if task:
            if message == "Processing cancelled":
                task.status = TaskStatus.CANCELLED
            else:
                task.status = TaskStatus.FAILED
                task.error = message
        self._qm.save()
        self._refresh_table()
        self._current_worker = None
        self._run_next_task()

    def _on_queue_finished(self):
        self._set_queue_state("idle")
        self._current_label.setText("队列完成")
        self._status_label.setText("")
        self._progress_bar.setValue(100)
        QApplication.beep()

    def _on_pause(self):
        if self._queue_state == "running" and self._current_worker:
            self._current_worker.pause()
            self._set_queue_state("paused")
            # Update task status
            task = self._current_running_task()
            if task:
                task.status = TaskStatus.PAUSED
                self._qm.save()
                self._refresh_table()
        elif self._queue_state == "paused" and self._current_worker:
            self._current_worker.resume()
            self._set_queue_state("running")
            task = self._current_running_task()
            if task:
                task.status = TaskStatus.PROCESSING
                self._qm.save()
                self._refresh_table()

    def _on_cancel_current(self):
        if self._current_worker:
            self._current_worker.cancel()
            self._current_worker.wait()
            # _on_task_error will handle status update and advance to next

    def _on_clear(self):
        reply = QMessageBox.question(
            self, "确认", "清空队列将删除所有任务和缓存，确定吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._cache.cleanup_all()
            self._qm.clear_all()
            self._qm.save()
            self._refresh_table()
            self._current_label.setText("")
            self._status_label.setText("")
            self._total_label.setText("")
            self._progress_bar.setValue(0)

    def _current_running_task(self) -> QueueTask | None:
        for t in self._qm.tasks:
            if t.status in (TaskStatus.PROCESSING, TaskStatus.PAUSED):
                return t
        return None

    def refresh(self):
        """Called by MainWindow after adding tasks."""
        self._refresh_table()
        self._set_queue_state(self._queue_state)
```

- [ ] **Step 2: Replace placeholder Tab 2 in MainWindow**

In `src/gui/main_window.py`, update imports:

```python
from src.gui.queue_tab import QueueTab
```

Replace the Tab 2 placeholder code in `_init_ui` with:

```python
        # --- Tab 2: Queue ---
        self._queue_tab = QueueTab(self._queue_manager, self._get_config)
        self._queue_tab.queue_running_changed.connect(self._on_queue_running_changed)
        self._queue_tab.task_count_changed.connect(
            lambda count: self._tabs.setTabText(1, f"批量队列 ({count})" if count > 0 else "批量队列")
        )
        self._tabs.addTab(self._queue_tab, self._get_queue_tab_title())
```

Add the mutual exclusion handler:

```python
    def _on_queue_running_changed(self, running: bool):
        """Disable single-task controls when queue is running."""
        self._start_btn.setEnabled(not running and self._state == "ready")
        self._select_btn.setEnabled(not running)
        self._enqueue_btn.setEnabled(not running and self._state == "ready")
```

Update `_on_enqueue` to call refresh on the queue tab:

```python
    def _on_enqueue(self):
        if not self._input_path or not self._input_type:
            QMessageBox.warning(self, "提示", "请先选择输入文件")
            return

        config = self._get_config()
        task = QueueTask.create(
            input_path=self._input_path,
            input_type=self._input_type,
            config=config,
            output_dir=self._output_dir,
        )
        self._queue_manager.add_task(task)
        self._queue_manager.save()

        # Clear input for next add
        self._input_path = None
        self._input_type = None
        self._input_edit.setText("")
        self._info_label.setText("")
        self._output_dir = None
        self._output_edit.setText("")
        self._set_state("initial")

        self._queue_tab.refresh()
        self.statusBar().showMessage("已加入队列", 3000)
```

Remove the helper methods `_get_queue_tab_title` and `_update_queue_tab_title` from MainWindow — the QueueTab now manages this via signal.

- [ ] **Step 3: Manually verify the full queue flow**

Run: `python main.py`
Expected:
1. Tab 1: Select file, click "加入队列" → tab title updates, input clears
2. Tab 2: Shows task in table with "⏳ 等待" status
3. Drag file onto Tab 2 → auto-added
4. Right-click → context menu with "删除" / "移到顶部" / "移到底部"
5. Click "开始队列" → tasks process sequentially with phase progress
6. Beep on completion

- [ ] **Step 4: Commit**

```bash
git add src/gui/queue_tab.py src/gui/main_window.py
git commit -m "feat: add Queue Tab with task list, progress, drag-drop, and controls"
```

---

### Task 8: Integration Testing and Polish

**Files:**
- Modify: `src/gui/main_window.py` (minor fixes)
- Modify: `src/gui/queue_tab.py` (minor fixes)

- [ ] **Step 1: Test mutual exclusion between tabs**

Run: `python main.py`
Test manually:
1. Start a single-task process in Tab 1 → switch to Tab 2 → "开始队列" should be disabled
2. Start queue in Tab 2 → switch to Tab 1 → "开始处理" should be disabled

If Tab 1 doesn't disable "开始处理" during queue execution, add this check to `_on_start` in MainWindow:

```python
    def _on_start(self):
        if not self._input_path:
            return
        if self._queue_tab._queue_state != "idle":
            QMessageBox.warning(self, "提示", "队列正在执行中，请等待队列完成")
            return
        # ... rest of existing code
```

- [ ] **Step 2: Test breakpoint resume flow**

1. Add a video to queue, start queue
2. While processing (during inference), click "暂停"
3. Close the app
4. Reopen the app → Tab 2 should show the task with "⏸ 暂停 X/Y" status
5. Click "开始队列" → should resume from frame X, not restart

- [ ] **Step 3: Update PROGRESS.md**

Add the P2 batch queue to the completed features section. Update the version line, add new entries to the table, and update the project structure.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: P2 batch queue — complete with persistence, resume, and queue UI"
```

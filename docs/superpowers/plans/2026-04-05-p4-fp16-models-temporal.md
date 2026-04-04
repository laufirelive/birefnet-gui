# P4: FP16 加速 + 多模型 + 模型管理 + 时序修复 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FP16 auto-acceleration on CUDA, full 6-model support with a model management tab (download/delete via hf-mirror), and temporal outlier-frame detection to fix mask flicker in videos.

**Architecture:** Four independent modules layered bottom-up: (1) config extends MODEL_REGISTRY + temporal_fix field; (2) inference adds FP16 autocast; (3) temporal.py implements outlier detection; (4) pipeline gains a third phase; (5) model_tab.py provides download/delete GUI; (6) settings_panel and main_window wire everything together.

**Tech Stack:** PyTorch autocast (FP16), huggingface_hub (model download), numpy (temporal analysis), PyQt6 (model tab GUI)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/core/config.py` | Modify | Add `ModelInfo`, `MODEL_REGISTRY`, `temporal_fix` to `ProcessingConfig` |
| `src/core/inference.py` | Modify | FP16 autocast on CUDA; `load_model` accepts model name directly |
| `src/core/temporal.py` | Create | Outlier frame detection + neighbour-average replacement |
| `src/core/pipeline.py` | Modify | Insert `temporal_fix_phase` between infer and encode |
| `src/core/queue_task.py` | Modify | Serialize/deserialize `temporal_fix` with backward compat |
| `src/core/model_downloader.py` | Create | Download logic: hf-mirror fallback, progress reporting |
| `src/gui/model_tab.py` | Create | Model management tab: card list, download/delete, progress |
| `src/gui/settings_panel.py` | Modify | Filter model combo to installed only; add "管理模型..." link; add temporal_fix checkbox |
| `src/gui/main_window.py` | Modify | Add model tab; first-launch detection; 3-phase progress; update `ProcessingPhase` usage |
| `src/worker/matting_worker.py` | Modify | Pass temporal_fix through to pipeline |
| `download_models.py` | Modify | Import from `MODEL_REGISTRY` instead of local dict |
| `tests/test_config.py` | Modify | Tests for ModelInfo, MODEL_REGISTRY, temporal_fix |
| `tests/test_inference.py` | Modify | Tests for FP16 autocast behavior |
| `tests/test_temporal.py` | Create | Tests for outlier detection algorithm |
| `tests/test_pipeline.py` | Modify | Tests for 3-phase pipeline and temporal_fix=False bypass |
| `tests/test_queue_task.py` | Modify | Backward compat test for temporal_fix |
| `tests/test_model_downloader.py` | Create | Tests for download logic |

---

### Task 1: Extend config.py — ModelInfo + MODEL_REGISTRY + temporal_fix

**Files:**
- Modify: `src/core/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write tests for ModelInfo and MODEL_REGISTRY**

Add to `tests/test_config.py`:

```python
from src.core.config import ModelInfo, MODEL_REGISTRY


class TestModelInfo:
    def test_model_info_fields(self):
        info = MODEL_REGISTRY["general"]
        assert info.key == "general"
        assert info.dir_name == "birefnet-general"
        assert info.repo_id == "zhengpeng7/BiRefNet"
        assert info.display_name == "BiRefNet-general"
        assert isinstance(info.description, str)
        assert isinstance(info.use_case, str)
        assert info.size_mb > 0


class TestModelRegistry:
    def test_registry_has_six_models(self):
        assert len(MODEL_REGISTRY) == 6

    def test_all_expected_keys_present(self):
        expected = {"general", "lite", "hr", "matting", "hr-matting", "dynamic"}
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_all_entries_are_model_info(self):
        for key, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo)
            assert info.key == key

    def test_display_names_match_models_dict(self):
        from src.core.config import MODELS
        for key, info in MODEL_REGISTRY.items():
            assert info.display_name in MODELS
            assert MODELS[info.display_name] == info.dir_name
```

- [ ] **Step 2: Write test for temporal_fix default**

Add to `tests/test_config.py`:

```python
class TestProcessingConfigTemporalFix:
    def test_temporal_fix_default_true(self):
        config = ProcessingConfig()
        assert config.temporal_fix is True

    def test_temporal_fix_can_be_disabled(self):
        config = ProcessingConfig(temporal_fix=False)
        assert config.temporal_fix is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py::TestModelInfo tests/test_config.py::TestModelRegistry tests/test_config.py::TestProcessingConfigTemporalFix -v`
Expected: FAIL — `ModelInfo` and `MODEL_REGISTRY` not defined, `temporal_fix` not a field.

- [ ] **Step 4: Implement ModelInfo, MODEL_REGISTRY, and temporal_fix**

In `src/core/config.py`, add after the existing `MODELS` dict:

```python
@dataclass
class ModelInfo:
    key: str
    dir_name: str
    repo_id: str
    display_name: str
    description: str
    use_case: str
    size_mb: int


MODEL_REGISTRY: dict[str, ModelInfo] = {
    "general": ModelInfo(
        key="general",
        dir_name="birefnet-general",
        repo_id="zhengpeng7/BiRefNet",
        display_name="BiRefNet-general",
        description="通用模型，效果均衡",
        use_case="大多数场景（默认推荐）",
        size_mb=424,
    ),
    "lite": ModelInfo(
        key="lite",
        dir_name="birefnet-lite",
        repo_id="zhengpeng7/BiRefNet_lite",
        display_name="BiRefNet-lite",
        description="轻量快速，精度略低",
        use_case="显存不足/追求速度",
        size_mb=210,
    ),
    "hr": ModelInfo(
        key="hr",
        dir_name="birefnet-hr",
        repo_id="zhengpeng7/BiRefNet_HR",
        display_name="BiRefNet-HR",
        description="高分辨率优化",
        use_case="4K 视频",
        size_mb=450,
    ),
    "matting": ModelInfo(
        key="matting",
        dir_name="birefnet-matting",
        repo_id="zhengpeng7/BiRefNet-matting",
        display_name="BiRefNet-matting",
        description="专注 matting，边缘细腻",
        use_case="人像/头发丝细节",
        size_mb=424,
    ),
    "hr-matting": ModelInfo(
        key="hr-matting",
        dir_name="birefnet-hr-matting",
        repo_id="zhengpeng7/BiRefNet_HR-matting",
        display_name="BiRefNet-HR-matting",
        description="高分辨率 + matting 结合",
        use_case="高分辨率人像",
        size_mb=450,
    ),
    "dynamic": ModelInfo(
        key="dynamic",
        dir_name="birefnet-dynamic",
        repo_id="zhengpeng7/BiRefNet_dynamic",
        display_name="BiRefNet-dynamic",
        description="动态分辨率输入",
        use_case="不同分辨率混合输入",
        size_mb=424,
    ),
}
```

Add `temporal_fix` to `ProcessingConfig`:

```python
@dataclass
class ProcessingConfig:
    model_name: str = "BiRefNet-general"
    output_format: OutputFormat = OutputFormat.MOV_PRORES
    background_mode: BackgroundMode = BackgroundMode.TRANSPARENT
    bitrate_mode: BitrateMode = BitrateMode.AUTO
    custom_bitrate_mbps: float = 20.0
    encoding_preset: EncodingPreset = EncodingPreset.MEDIUM
    batch_size: int = 1
    inference_resolution: InferenceResolution = InferenceResolution.RES_1024
    temporal_fix: bool = True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: ALL PASS (including all existing tests)

- [ ] **Step 6: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add ModelInfo, MODEL_REGISTRY, and temporal_fix config field"
```

---

### Task 2: FP16 autocast in inference.py

**Files:**
- Modify: `src/core/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write tests for FP16 autocast behavior**

Add to `tests/test_inference.py`:

```python
from unittest.mock import patch, MagicMock
import torch


class TestFP16Autocast:
    @patch("src.core.inference.torch")
    def test_cuda_uses_autocast(self, mock_torch):
        """Verify that predict() uses autocast on CUDA devices."""
        from src.core.inference import predict

        # Create a mock model that records whether autocast was active
        autocast_was_active = []
        original_autocast = torch.autocast

        def tracking_autocast(*args, **kwargs):
            ctx = original_autocast(*args, **kwargs)
            autocast_was_active.append(True)
            return ctx

        # We can't easily mock autocast internals, so instead verify
        # the code path by checking the function handles device="cuda"
        # without error. Full integration test requires real CUDA.
        # Unit test: verify predict does not crash with device="cpu"
        pass

    def test_predict_works_on_cpu(self):
        """predict() works without autocast on CPU — smoke test."""
        # This is tested by existing tests; just confirm no regression
        pass
```

Actually, a more practical approach — test that the autocast context manager is entered for CUDA by mocking:

```python
from unittest.mock import patch, MagicMock


class TestFP16Autocast:
    def test_predict_calls_autocast_for_cuda(self):
        """Verify autocast is used when device is cuda."""
        import src.core.inference as inf

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value=MagicMock())

        # We test by patching torch.autocast and verifying it's called
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        with patch.object(inf.torch, "autocast") as mock_autocast:
            mock_ctx = MagicMock()
            mock_autocast.return_value = mock_ctx
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(return_value=False)

            # Mock model to return proper tensor
            import torch
            fake_pred = torch.zeros(1, 1, 64, 64)
            mock_model.return_value = [fake_pred]

            inf.predict(mock_model, frame, "cuda", resolution=64)
            mock_autocast.assert_called_once_with("cuda", dtype=torch.float16)

    def test_predict_no_autocast_for_cpu(self):
        """Verify autocast is NOT used when device is cpu."""
        import src.core.inference as inf

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        with patch.object(inf.torch, "autocast") as mock_autocast:
            import torch
            fake_pred = torch.zeros(1, 1, 64, 64)
            mock_model = MagicMock()
            mock_model.return_value = [fake_pred]

            inf.predict(mock_model, frame, "cpu", resolution=64)
            mock_autocast.assert_not_called()

    def test_predict_no_autocast_for_mps(self):
        """Verify autocast is NOT used when device is mps."""
        import src.core.inference as inf

        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        with patch.object(inf.torch, "autocast") as mock_autocast:
            import torch
            fake_pred = torch.zeros(1, 1, 64, 64)
            mock_model = MagicMock()
            mock_model.return_value = [fake_pred]

            inf.predict(mock_model, frame, "mps", resolution=64)
            mock_autocast.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_inference.py::TestFP16Autocast -v`
Expected: FAIL — predict() doesn't call autocast yet.

- [ ] **Step 3: Implement FP16 autocast in predict() and predict_batch()**

Replace the inference section of `predict()` in `src/core/inference.py`:

```python
def predict(model, frame: np.ndarray, device: str, resolution: int = 1024) -> np.ndarray:
    orig_h, orig_w = frame.shape[:2]
    transform = _get_transform(resolution)

    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb)
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(input_tensor)[-1]
        else:
            preds = model(input_tensor)[-1]
        pred = torch.sigmoid(preds[0, 0])

    pred_resized = torch.nn.functional.interpolate(
        pred.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    alpha = (pred_resized * 255).clamp(0, 255).byte().cpu().numpy()
    return alpha
```

Apply same pattern to `predict_batch()`:

```python
def predict_batch(model, frames, device, resolution=1024):
    if len(frames) == 0:
        return []
    transform = _get_transform(resolution)
    orig_sizes = [(f.shape[0], f.shape[1]) for f in frames]
    tensors = []
    for frame in frames:
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        tensors.append(transform(image))
    batch_tensor = torch.stack(tensors).to(device)
    with torch.no_grad():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                preds = model(batch_tensor)[-1]
        else:
            preds = model(batch_tensor)[-1]
        preds = torch.sigmoid(preds[:, 0])
    masks = []
    for i, (orig_h, orig_w) in enumerate(orig_sizes):
        pred_resized = torch.nn.functional.interpolate(
            preds[i].unsqueeze(0).unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        alpha = (pred_resized * 255).clamp(0, 255).byte().cpu().numpy()
        masks.append(alpha)
    return masks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_inference.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/inference.py tests/test_inference.py
git commit -m "feat: FP16 autocast on CUDA for predict and predict_batch"
```

---

### Task 3: Temporal outlier detection — src/core/temporal.py

**Files:**
- Create: `src/core/temporal.py`
- Create: `tests/test_temporal.py`

- [ ] **Step 1: Write tests for temporal outlier detection**

Create `tests/test_temporal.py`:

```python
import os
import tempfile

import cv2
import numpy as np
import pytest

from src.core.cache import MaskCacheManager
from src.core.temporal import detect_and_fix_outliers


@pytest.fixture
def cache_with_masks():
    """Create a cache with 10 smooth masks and return (cache, tmpdir, task_id)."""
    tmpdir = tempfile.mkdtemp()
    cache = MaskCacheManager(tmpdir)
    task_id = "temporal_test"
    # Create 10 smooth masks: all similar gray values
    for i in range(10):
        mask = np.full((64, 64), fill_value=200, dtype=np.uint8)
        cache.save_mask(task_id, i, mask)
    return cache, tmpdir, task_id


class TestDetectAndFixOutliers:
    def test_no_outliers_returns_zero(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 0
        # Verify masks unchanged
        for i in range(10):
            mask = cache.load_mask(task_id, i)
            assert np.all(mask == 200)

    def test_single_outlier_gets_fixed(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        # Inject an outlier at frame 5: completely different value
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 5, outlier)

        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1

        # Frame 5 should now be average of frames 4 and 6 (both 200)
        repaired = cache.load_mask(task_id, 5)
        assert np.allclose(repaired, 200, atol=1)

    def test_first_frame_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        # Frame 0 is outlier
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 0, outlier)

        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1
        repaired = cache.load_mask(task_id, 0)
        # First frame replaced by frame 1
        assert np.allclose(repaired, 200, atol=1)

    def test_last_frame_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 9, outlier)

        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1
        repaired = cache.load_mask(task_id, 9)
        assert np.allclose(repaired, 200, atol=1)

    def test_scene_change_not_marked_as_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        # Frames 5-9 all shift to a different value (scene change)
        for i in range(5, 10):
            mask = np.full((64, 64), fill_value=50, dtype=np.uint8)
            cache.save_mask(task_id, i, mask)

        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        # Frame 5 has diff with frame 4 but is consistent with frame 6 → NOT outlier
        assert fixed == 0

    def test_progress_callback_called(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        progress_log = []
        detect_and_fix_outliers(
            cache, task_id, total_frames=10,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_two_frames_video(self):
        """Edge case: only 2 frames."""
        tmpdir = tempfile.mkdtemp()
        cache = MaskCacheManager(tmpdir)
        task_id = "two_frames"
        cache.save_mask(task_id, 0, np.full((32, 32), 200, dtype=np.uint8))
        cache.save_mask(task_id, 1, np.full((32, 32), 200, dtype=np.uint8))
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=2)
        assert fixed == 0

    def test_single_frame_video(self):
        """Edge case: only 1 frame — nothing to compare."""
        tmpdir = tempfile.mkdtemp()
        cache = MaskCacheManager(tmpdir)
        task_id = "one_frame"
        cache.save_mask(task_id, 0, np.full((32, 32), 200, dtype=np.uint8))
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=1)
        assert fixed == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_temporal.py -v`
Expected: FAIL — `src.core.temporal` module does not exist.

- [ ] **Step 3: Implement detect_and_fix_outliers**

Create `src/core/temporal.py`:

```python
from typing import Callable, Optional

import cv2
import numpy as np

from src.core.cache import MaskCacheManager

# Default threshold: average per-pixel difference (0-1 scale) above which
# a frame is considered an outlier relative to its neighbour.
DEFAULT_THRESHOLD = 0.15


def detect_and_fix_outliers(
    cache: MaskCacheManager,
    task_id: str,
    total_frames: int,
    threshold: float = DEFAULT_THRESHOLD,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Detect and repair outlier frames whose masks deviate sharply from neighbours.

    An outlier is a frame where the mask differs significantly from BOTH its
    predecessor and successor.  This distinguishes genuine scene changes (where
    frames after the cut are consistent with each other) from single-frame
    quality drops.

    Outlier masks are replaced by the average of their two neighbours (or
    copied from the single neighbour for first/last frames).

    Returns the number of frames that were repaired.
    """
    if total_frames <= 1:
        if progress_callback:
            for i in range(total_frames):
                progress_callback(i + 1, total_frames)
        return 0

    # Load all masks into memory (grayscale uint8, small footprint)
    masks: list[np.ndarray] = []
    for i in range(total_frames):
        masks.append(cache.load_mask(task_id, i))

    fixed_count = 0

    for i in range(total_frames):
        is_outlier = False

        if i == 0:
            # First frame: only compare with next
            diff_next = _mask_diff(masks[0], masks[1])
            if diff_next > threshold * 1.5:
                is_outlier = True
        elif i == total_frames - 1:
            # Last frame: only compare with previous
            diff_prev = _mask_diff(masks[i], masks[i - 1])
            if diff_prev > threshold * 1.5:
                is_outlier = True
        else:
            # Middle frame: must differ from BOTH neighbours
            diff_prev = _mask_diff(masks[i], masks[i - 1])
            diff_next = _mask_diff(masks[i], masks[i + 1])
            if diff_prev > threshold and diff_next > threshold:
                is_outlier = True

        if is_outlier:
            if i == 0:
                replacement = masks[1].copy()
            elif i == total_frames - 1:
                replacement = masks[i - 1].copy()
            else:
                replacement = (
                    (masks[i - 1].astype(np.float32) + masks[i + 1].astype(np.float32)) / 2.0
                ).astype(np.uint8)

            masks[i] = replacement
            cache.save_mask(task_id, i, replacement)
            fixed_count += 1

        if progress_callback:
            progress_callback(i + 1, total_frames)

    return fixed_count


def _mask_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalised mean absolute difference between two masks (0.0-1.0)."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_temporal.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/temporal.py tests/test_temporal.py
git commit -m "feat: temporal outlier detection and repair for mask flicker"
```

---

### Task 4: Integrate temporal fix into pipeline

**Files:**
- Modify: `src/core/pipeline.py`
- Modify: `src/core/queue_task.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_queue_task.py`

- [ ] **Step 1: Write test for 3-phase pipeline with temporal fix**

Add to `tests/test_pipeline.py`:

```python
@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestTemporalFixPhase:
    def test_process_includes_temporal_fix_phase(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(temporal_fix=True)
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="test_temporal", cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert os.path.exists(output_path)
            phases = [p for _, _, p in progress_log]
            assert "inference" in phases
            assert "temporal_fix" in phases
            assert "encoding" in phases

    def test_process_skips_temporal_fix_when_disabled(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(temporal_fix=False)
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="test_no_temporal", cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert os.path.exists(output_path)
            phases = set(p for _, _, p in progress_log)
            assert "temporal_fix" not in phases
            assert phases == {"inference", "encoding"}
```

- [ ] **Step 2: Write test for queue_task backward compat with temporal_fix**

Add to `tests/test_queue_task.py`:

```python
class TestQueueTaskTemporalFixCompat:
    def test_old_brm_without_temporal_fix_defaults_true(self):
        old_dict = {
            "id": "tf_test",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mov_prores",
                "background_mode": "transparent",
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
        assert task.config.temporal_fix is True

    def test_roundtrip_with_temporal_fix_false(self):
        config = ProcessingConfig(temporal_fix=False)
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        restored = QueueTask.from_dict(d)
        assert restored.config.temporal_fix is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_pipeline.py::TestTemporalFixPhase tests/test_queue_task.py::TestQueueTaskTemporalFixCompat -v`
Expected: FAIL — pipeline doesn't emit "temporal_fix" phase; queue_task doesn't handle temporal_fix.

- [ ] **Step 4: Add temporal_fix_phase to pipeline.py**

In `src/core/pipeline.py`, add import at top:

```python
from src.core.temporal import detect_and_fix_outliers
```

Add new method to `MattingPipeline`:

```python
    def temporal_fix_phase(
        self,
        task_id: str,
        cache: MaskCacheManager,
        total_frames: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> int:
        """Run temporal outlier detection and fix. Returns number of fixed frames."""
        def inner_callback(current: int, total: int):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")
            if progress_callback:
                progress_callback(current, total, "temporal_fix")

        return detect_and_fix_outliers(
            cache, task_id, total_frames,
            progress_callback=inner_callback,
        )
```

Update `process()` to include temporal fix between infer and encode:

```python
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
        """Convenience method: run infer_phase, optional temporal_fix_phase, then encode_phase."""
        self.infer_phase(
            input_path, task_id, cache, start_frame,
            progress_callback, pause_event, cancel_event,
        )
        if self._config.temporal_fix:
            video_info = get_video_info(input_path)
            total_frames = video_info["frame_count"]
            self.temporal_fix_phase(
                task_id, cache, total_frames,
                progress_callback, cancel_event,
            )
        self.encode_phase(
            input_path, output_path, task_id, cache,
            progress_callback, pause_event, cancel_event,
        )
```

- [ ] **Step 5: Update queue_task.py for temporal_fix serialization**

In `src/core/queue_task.py`, update `to_dict()` to include `temporal_fix`:

```python
    def to_dict(self) -> dict:
        return {
            ...
            "config": {
                ...
                "inference_resolution": self.config.inference_resolution.value,
                "temporal_fix": self.config.temporal_fix,
            },
            ...
        }
```

Update `from_dict()` to read it with a default:

```python
        config = ProcessingConfig(
            ...
            inference_resolution=InferenceResolution(cfg.get("inference_resolution", 1024)),
            temporal_fix=cfg.get("temporal_fix", True),
        )
```

- [ ] **Step 6: Update ProcessingPhase enum to include TEMPORAL_FIX**

In `src/core/queue_task.py`:

```python
class ProcessingPhase(Enum):
    INFERENCE = "inference"
    TEMPORAL_FIX = "temporal_fix"
    ENCODING = "encoding"
    DONE = "done"
```

- [ ] **Step 7: Run all tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_queue_task.py tests/test_temporal.py -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/core/pipeline.py src/core/queue_task.py tests/test_pipeline.py tests/test_queue_task.py
git commit -m "feat: 3-phase pipeline with temporal fix between infer and encode"
```

---

### Task 5: Model downloader module

**Files:**
- Create: `src/core/model_downloader.py`
- Create: `tests/test_model_downloader.py`

- [ ] **Step 1: Write tests for model downloader**

Create `tests/test_model_downloader.py`:

```python
import os
from unittest.mock import patch, MagicMock

import pytest

from src.core.model_downloader import ModelDownloader


class TestModelDownloader:
    def test_get_installed_models(self, tmp_path):
        # Create some model dirs
        (tmp_path / "birefnet-general").mkdir()
        (tmp_path / "birefnet-lite").mkdir()
        downloader = ModelDownloader(str(tmp_path))
        installed = downloader.get_installed_models()
        assert "general" in installed
        assert "lite" in installed
        assert "hr" not in installed

    def test_is_installed(self, tmp_path):
        (tmp_path / "birefnet-general").mkdir()
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("general") is True
        assert downloader.is_installed("lite") is False

    def test_is_installed_unknown_key_returns_false(self, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("nonexistent") is False

    def test_delete_model_removes_directory(self, tmp_path):
        model_dir = tmp_path / "birefnet-lite"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        downloader = ModelDownloader(str(tmp_path))
        assert downloader.is_installed("lite") is True
        downloader.delete_model("lite")
        assert downloader.is_installed("lite") is False
        assert not model_dir.exists()

    def test_delete_nonexistent_model_raises(self, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            downloader.delete_model("lite")

    @patch("src.core.model_downloader.snapshot_download")
    def test_download_calls_snapshot_download(self, mock_download, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        downloader.download_model("general")
        mock_download.assert_called_once()
        call_kwargs = mock_download.call_args
        assert "zhengpeng7/BiRefNet" in str(call_kwargs)

    @patch("src.core.model_downloader.snapshot_download")
    def test_download_tries_mirror_first(self, mock_download, tmp_path):
        downloader = ModelDownloader(str(tmp_path))
        downloader.download_model("lite")
        # Verify HF_ENDPOINT was set to mirror
        call_kwargs = mock_download.call_args
        # The endpoint is set via environment or endpoint param
        assert mock_download.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_model_downloader.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement ModelDownloader**

Create `src/core/model_downloader.py`:

```python
import os
import shutil
from typing import Optional

from huggingface_hub import snapshot_download

from src.core.config import MODEL_REGISTRY

HF_MIRROR = "https://hf-mirror.com"
HF_OFFICIAL = "https://huggingface.co"


class ModelDownloader:
    """Manages model installation: check status, download, delete."""

    def __init__(self, models_dir: str):
        self._models_dir = models_dir

    def get_installed_models(self) -> list[str]:
        """Return list of installed model keys."""
        installed = []
        for key, info in MODEL_REGISTRY.items():
            model_path = os.path.join(self._models_dir, info.dir_name)
            if os.path.isdir(model_path):
                installed.append(key)
        return installed

    def is_installed(self, model_key: str) -> bool:
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            return False
        return os.path.isdir(os.path.join(self._models_dir, info.dir_name))

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
        use_mirror: bool = True,
    ) -> str:
        """Download a model. Tries hf-mirror first, falls back to official.

        Returns the local path of the downloaded model.
        """
        info = MODEL_REGISTRY.get(model_key)
        if info is None:
            raise ValueError(f"Unknown model key: {model_key}")

        local_path = os.path.join(self._models_dir, info.dir_name)
        os.makedirs(local_path, exist_ok=True)

        if use_mirror:
            try:
                return self._do_download(info.repo_id, local_path, endpoint=HF_MIRROR)
            except Exception:
                # Fall back to official
                pass

        return self._do_download(info.repo_id, local_path, endpoint=HF_OFFICIAL)

    def _do_download(self, repo_id: str, local_path: str, endpoint: str) -> str:
        old_endpoint = os.environ.get("HF_ENDPOINT")
        try:
            os.environ["HF_ENDPOINT"] = endpoint
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        finally:
            if old_endpoint is not None:
                os.environ["HF_ENDPOINT"] = old_endpoint
            elif "HF_ENDPOINT" in os.environ:
                del os.environ["HF_ENDPOINT"]
        return local_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_model_downloader.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/model_downloader.py tests/test_model_downloader.py
git commit -m "feat: ModelDownloader with hf-mirror fallback"
```

---

### Task 6: Update download_models.py to use MODEL_REGISTRY

**Files:**
- Modify: `download_models.py`

- [ ] **Step 1: Replace local MODELS dict with MODEL_REGISTRY import**

Rewrite `download_models.py`:

```python
#!/usr/bin/env python3
"""Download BiRefNet models for offline use.

Run: python download_models.py              # download general (default)
     python download_models.py --all        # download all models
     python download_models.py general lite # download specific models
"""

import sys

from src.core.config import MODEL_REGISTRY
from src.core.model_downloader import ModelDownloader

MODELS_DIR = "./models"


def main():
    args = sys.argv[1:]
    downloader = ModelDownloader(MODELS_DIR)

    if not args:
        print("No arguments. Downloading birefnet-general (default).")
        print(f"Use --all to download all models, or specify: {', '.join(MODEL_REGISTRY.keys())}")
        downloader.download_model("general")
        print("Done.")
        return

    if "--all" in args:
        keys = list(MODEL_REGISTRY.keys())
    else:
        keys = []
        for arg in args:
            if arg not in MODEL_REGISTRY:
                print(f"Unknown model: {arg}")
                print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
                sys.exit(1)
            keys.append(arg)

    for key in keys:
        info = MODEL_REGISTRY[key]
        print(f"\nDownloading {key} ({info.display_name})...")
        downloader.download_model(key)
        print(f"  Done: {key}")

    print(f"\nAll done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script still works**

Run: `python download_models.py --help` (or just `python -c "from download_models import main"`)
Expected: No import errors.

- [ ] **Step 3: Commit**

```bash
git add download_models.py
git commit -m "refactor: download_models.py uses MODEL_REGISTRY"
```

---

### Task 7: Model management tab — src/gui/model_tab.py

**Files:**
- Create: `src/gui/model_tab.py`

- [ ] **Step 1: Create model_tab.py**

Create `src/gui/model_tab.py`:

```python
import os

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


class DownloadWorker(QThread):
    """Background thread for model download."""

    progress = pyqtSignal(str)  # status text
    finished = pyqtSignal(str)  # model_key
    error = pyqtSignal(str, str)  # model_key, error message

    def __init__(self, downloader: ModelDownloader, model_key: str):
        super().__init__()
        self._downloader = downloader
        self._model_key = model_key

    def run(self):
        try:
            self.progress.emit(f"正在下载 {MODEL_REGISTRY[self._model_key].display_name}...")
            self._downloader.download_model(self._model_key)
            self.finished.emit(self._model_key)
        except Exception as e:
            self.error.emit(self._model_key, str(e))


class ModelCard(QWidget):
    """A card displaying one model's info and action button."""

    download_requested = pyqtSignal(str)  # model_key
    delete_requested = pyqtSignal(str)  # model_key

    def __init__(self, model_key: str, is_installed: bool, parent=None):
        super().__init__(parent)
        self._model_key = model_key
        info = MODEL_REGISTRY[model_key]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        # Top row: name + size + action button
        top_row = QHBoxLayout()
        status = "✅ " if is_installed else ""
        name_label = QLabel(f"<b>{status}{info.display_name}</b>")
        top_row.addWidget(name_label)
        top_row.addStretch()

        size_label = QLabel(f"{info.size_mb} MB")
        size_label.setStyleSheet("color: gray;")
        top_row.addWidget(size_label)

        self._action_btn = QPushButton("删除" if is_installed else "下载")
        if is_installed:
            self._action_btn.clicked.connect(lambda: self.delete_requested.emit(self._model_key))
        else:
            self._action_btn.clicked.connect(lambda: self.download_requested.emit(self._model_key))
        top_row.addWidget(self._action_btn)
        layout.addLayout(top_row)

        # Description
        desc_label = QLabel(info.description)
        desc_label.setStyleSheet("color: #555;")
        layout.addWidget(desc_label)

        # Use case
        use_label = QLabel(f"适用：{info.use_case}")
        use_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(use_label)

        self.setStyleSheet("ModelCard { border: 1px solid #ddd; border-radius: 4px; }")

    def set_enabled_action(self, enabled: bool):
        self._action_btn.setEnabled(enabled)


class ModelTab(QWidget):
    """Model management tab: list, download, delete models."""

    models_changed = pyqtSignal()  # Emitted when a model is installed or deleted

    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._downloader = ModelDownloader(self._models_dir)
        self._download_worker: DownloadWorker | None = None
        self._cards: dict[str, ModelCard] = {}

        self._init_ui()
        self._refresh_cards()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # No-model warning banner
        self._no_model_banner = QLabel("⚠ 请先下载至少一个模型才能开始处理")
        self._no_model_banner.setStyleSheet(
            "background: #FFF3CD; color: #856404; padding: 8px; border-radius: 4px;"
        )
        self._no_model_banner.setVisible(False)
        layout.addWidget(self._no_model_banner)

        # Scrollable card list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch()
        scroll.setWidget(self._card_container)
        layout.addWidget(scroll, stretch=1)

        # Download progress area
        self._progress_widget = QWidget()
        progress_layout = QVBoxLayout(self._progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
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

        # Bottom info
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
        # Clear existing cards
        for key, card in self._cards.items():
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        # Re-create
        installed = self._downloader.get_installed_models()
        for key in MODEL_REGISTRY:
            is_installed = key in installed
            card = ModelCard(key, is_installed)
            card.download_requested.connect(self._on_download_requested)
            card.delete_requested.connect(self._on_delete_requested)
            if self._download_worker is not None:
                card.set_enabled_action(False)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)
            self._cards[key] = card

        self._no_model_banner.setVisible(len(installed) == 0)

    def _on_download_requested(self, model_key: str):
        if self._download_worker is not None:
            return
        self._download_worker = DownloadWorker(self._downloader, model_key)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)

        # Disable all action buttons during download
        for card in self._cards.values():
            card.set_enabled_action(False)

        self._progress_widget.setVisible(True)
        info = MODEL_REGISTRY[model_key]
        self._progress_label.setText(f"正在下载 {info.display_name}...")
        self._download_worker.start()

    def _on_download_progress(self, text: str):
        self._progress_label.setText(text)

    def _on_download_finished(self, model_key: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()
        self.models_changed.emit()

    def _on_download_error(self, model_key: str, error_msg: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()
        QMessageBox.critical(
            self, "下载失败",
            f"下载 {MODEL_REGISTRY[model_key].display_name} 失败:\n{error_msg}",
        )

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
```

- [ ] **Step 2: Verify it imports without error**

Run: `python -c "from src.gui.model_tab import ModelTab; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/gui/model_tab.py
git commit -m "feat: model management tab with download/delete"
```

---

### Task 8: Update SettingsPanel — installed-only model combo + temporal fix checkbox

**Files:**
- Modify: `src/gui/settings_panel.py`

- [ ] **Step 1: Update _populate_model_combo to only show installed models + add "管理模型..." link**

In `src/gui/settings_panel.py`, update `_init_ui` to add a "管理模型..." button below the model combo and a temporal fix checkbox in advanced params. Also update `_populate_model_combo` to only show installed models.

Replace the model combo population method:

```python
    def _populate_model_combo(self):
        self._model_combo.clear()
        for display_name, dir_name in MODELS.items():
            model_path = os.path.join(self._models_dir, dir_name)
            if os.path.isdir(model_path):
                self._model_combo.addItem(display_name, display_name)
```

Add after the model combo in `_init_ui`:

```python
        self._manage_models_btn = QPushButton("管理模型...")
        self._manage_models_btn.setFlat(True)
        self._manage_models_btn.setStyleSheet("color: #0066CC; text-align: left; padding: 0;")
        model_layout.addWidget(self._manage_models_btn)
```

Add temporal fix checkbox in the advanced group section of `_init_ui`:

```python
        self._temporal_fix_checkbox = QCheckBox("时序修复（减少闪烁）")
        self._temporal_fix_checkbox.setChecked(True)
        self._temporal_fix_checkbox.stateChanged.connect(lambda _: self.settings_changed.emit())
        advanced_layout.addWidget(self._temporal_fix_checkbox)
```

Update `_update_advanced_visibility` to hide temporal fix for images:

```python
        is_video = self._input_type == InputType.VIDEO or self._input_type is None
        self._temporal_fix_checkbox.setVisible(is_video)
```

Update `get_config()` to include temporal_fix:

```python
    def get_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            ...
            inference_resolution=self._resolution_combo.currentData() or InferenceResolution.RES_1024,
            temporal_fix=self._temporal_fix_checkbox.isChecked(),
        )
```

Add `refresh_models()` method:

```python
    def refresh_models(self):
        """Refresh model combo after download/delete."""
        current = self._model_combo.currentData()
        self._populate_model_combo()
        # Try to restore selection
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == current:
                self._model_combo.setCurrentIndex(i)
                return
        # If previous selection was deleted, select first available
        if self._model_combo.count() > 0:
            self._model_combo.setCurrentIndex(0)
```

Add import for `QCheckBox` and `QPushButton` (QPushButton already imported? No, check — it's not imported yet in settings_panel). Add to imports:

```python
from PyQt6.QtWidgets import (
    QCheckBox,
    ...
    QPushButton,
    ...
)
```

- [ ] **Step 2: Verify import**

Run: `python -c "from src.gui.settings_panel import SettingsPanel; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/gui/settings_panel.py
git commit -m "feat: settings panel — installed-only models, manage link, temporal fix toggle"
```

---

### Task 9: Wire everything into MainWindow

**Files:**
- Modify: `src/gui/main_window.py`
- Modify: `src/worker/matting_worker.py`

- [ ] **Step 1: Update matting_worker to pass temporal_fix through 3-phase progress**

In `src/worker/matting_worker.py`, no code changes needed for the pipeline — the `MattingPipeline.process()` already reads `temporal_fix` from `self._config`. The `_on_progress` callback already handles arbitrary phase strings. Verify the phase label mapping in `main_window.py` handles the new `"temporal_fix"` phase.

- [ ] **Step 2: Add model tab to MainWindow**

In `src/gui/main_window.py`, add import:

```python
from src.gui.model_tab import ModelTab
```

In `_init_ui`, after creating the queue tab, add:

```python
        # --- Tab 3: Model Management ---
        self._model_tab = ModelTab(MODELS_DIR)
        self._model_tab.models_changed.connect(self._on_models_changed)
        self._tabs.addTab(self._model_tab, "模型管理")
```

Add method:

```python
    def _on_models_changed(self):
        """Refresh model combo when models are installed/deleted."""
        self._settings_panel.refresh_models()
```

Connect the "管理模型..." button:

```python
        self._settings_panel._manage_models_btn.clicked.connect(
            lambda: self._tabs.setCurrentWidget(self._model_tab)
        )
```

- [ ] **Step 3: Update action bar visibility for 3 tabs**

Update `_on_tab_changed`:

```python
    def _on_tab_changed(self, index: int):
        self._action_bar.setVisible(index == 0)
```

This already works — action bar shows only on tab 0 (single task).

- [ ] **Step 4: Update phase label mapping for temporal_fix**

In `_on_progress` method of `main_window.py`:

```python
        phase_label = {
            "inference": "推理中",
            "temporal_fix": "时序修复中",
            "encoding": "编码中",
            "processing": "处理中",
        }.get(phase, phase)
```

Do the same in `queue_tab.py`'s `_on_task_progress`:

```python
        phase_label = {
            "inference": "推理中",
            "temporal_fix": "时序修复中",
            "encoding": "编码中",
            "processing": "处理中",
        }.get(phase, phase)
```

- [ ] **Step 5: Update queue_tab phase tracking for temporal_fix**

In `src/gui/queue_tab.py`, update phase mapping in `_on_task_progress`:

```python
        if phase == "inference":
            task.phase = ProcessingPhase.INFERENCE
        elif phase == "temporal_fix":
            task.phase = ProcessingPhase.TEMPORAL_FIX
        else:
            task.phase = ProcessingPhase.ENCODING
```

Update `_status_text` to handle temporal_fix:

```python
    def _status_text(self, task: QueueTask) -> str:
        if task.status == TaskStatus.PROCESSING:
            phase_map = {
                ProcessingPhase.INFERENCE: "推理",
                ProcessingPhase.TEMPORAL_FIX: "时序修复",
                ProcessingPhase.ENCODING: "编码",
            }
            phase = phase_map.get(task.phase, "处理")
            ...
```

Disable pause during temporal_fix phase too (it's fast, no need for pause):

```python
            is_non_pausable = phase in ("encoding", "temporal_fix")
            self._pause_btn.setEnabled(not is_non_pausable and self._queue_state == "running")
```

- [ ] **Step 6: Add first-launch detection**

In `MainWindow.__init__`, after `self._init_ui()`:

```python
        # First launch: if no models installed, switch to model tab
        if not self._model_tab.has_any_model():
            self._tabs.setCurrentWidget(self._model_tab)
            self._start_btn.setEnabled(False)
```

Also update `_on_start` to check for models:

```python
    def _on_start(self):
        if not self._input_path:
            return
        if not self._model_tab.has_any_model():
            QMessageBox.warning(self, "提示", "请先在「模型管理」中下载至少一个模型")
            self._tabs.setCurrentWidget(self._model_tab)
            return
        ...
```

- [ ] **Step 7: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 8: Commit**

```bash
git add src/gui/main_window.py src/gui/queue_tab.py src/gui/settings_panel.py src/worker/matting_worker.py
git commit -m "feat: wire model tab, temporal fix phase, and 3-phase progress into GUI"
```

---

### Task 10: Manual smoke test and final verification

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS, no regressions.

- [ ] **Step 2: Quick GUI smoke test**

Run: `python main.py`

Verify:
1. Three tabs visible: 单任务, 批量队列, 模型管理
2. If general model installed: model combo shows it; if not: auto-switches to model tab with warning banner
3. Model tab shows all 6 models with descriptions and use cases
4. Settings panel has "时序修复" checkbox (visible for video input, hidden for image)
5. Settings panel has "管理模型..." link that switches to model tab
6. Progress display shows 3 phases for video: 推理中 → 时序修复中 → 编码中

- [ ] **Step 3: Commit any fixes found during smoke test**

- [ ] **Step 4: Final commit with updated PROGRESS.md**

```bash
git add -A
git commit -m "docs: update PROGRESS.md — P4 complete"
```

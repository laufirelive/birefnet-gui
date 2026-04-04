# P3: Advanced Output Parameters + VRAM Detection + Layout Refactoring

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bitrate control, encoding presets, batch inference, VRAM detection with OOM warnings, inference resolution selection, and refactor the single-task tab layout to a left-right + fixed bottom bar design.

**Architecture:** Extend `ProcessingConfig` with new fields (bitrate, preset, batch size, inference resolution). Add `device_info.py` for GPU detection. Modify `inference.py` to support batch prediction and variable resolution. Extend `writer.py` / `video.py` to accept bitrate/preset FFmpeg args. Extract a `SettingsPanel` widget for the right-side settings. Refactor `MainWindow` single-task tab layout. All new config fields must serialize into `.brm` queue files with backward compatibility.

**Tech Stack:** PyQt6, PyTorch, FFmpeg, pytest

**Branch:** Create `feature/p3-advanced-params` from `master` before starting.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `src/core/config.py` | `BitrateMode`, `EncodingPreset`, `InferenceResolution` enums; extended `ProcessingConfig` | Modify |
| `src/core/device_info.py` | `DeviceInfo` dataclass + `get_device_info()` for GPU/VRAM detection | Create |
| `src/core/video.py` | `get_video_info()` returns `bitrate_mbps` via ffprobe | Modify |
| `src/core/inference.py` | `predict()` accepts `resolution` param; new `predict_batch()` | Modify |
| `src/core/writer.py` | `FFmpegWriter` accepts `bitrate_kbps`/`preset`; `create_writer()` resolves these from config | Modify |
| `src/core/video.py` | `ProResWriter` accepts `profile` param | Modify |
| `src/core/pipeline.py` | Batch inference in `infer_phase()` using `predict_batch()` | Modify |
| `src/core/queue_task.py` | Serialize/deserialize new config fields with backward compat | Modify |
| `src/gui/settings_panel.py` | Right-side settings panel: model, output, advanced params, VRAM warning | Create |
| `src/gui/main_window.py` | Layout refactor: left-right split + fixed bottom action bar | Modify |
| `tests/test_config.py` | Tests for new enums and config defaults | Modify |
| `tests/test_device_info.py` | Tests for VRAM detection with mocked torch | Create |
| `tests/test_video.py` | Test `get_video_info` returns `bitrate_mbps` | Modify |
| `tests/test_inference.py` | Tests for `predict_batch` and variable resolution | Modify |
| `tests/test_writer.py` | Tests for bitrate/preset FFmpeg args | Modify |
| `tests/test_pipeline.py` | Tests for batch inference in pipeline | Modify |
| `tests/test_queue_task.py` | Tests for backward-compatible deserialization | Modify |

---

## Task 0: Create feature branch

- [ ] **Step 1: Create and switch to feature branch**

```bash
git checkout -b feature/p3-advanced-params
```

- [ ] **Step 2: Verify branch**

```bash
git branch --show-current
```

Expected: `feature/p3-advanced-params`

---

## Task 1: Extend `ProcessingConfig` with new enums and fields

**Files:**
- Modify: `src/core/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing tests for new enums and config fields**

Add to `tests/test_config.py`:

```python
from src.core.config import (
    BitrateMode,
    EncodingPreset,
    InferenceResolution,
)


class TestBitrateMode:
    def test_all_modes_exist(self):
        expected = {"auto", "low", "medium", "high", "very_high", "custom"}
        assert {m.value for m in BitrateMode} == expected

    def test_multiplier(self):
        assert BitrateMode.LOW.multiplier == 0.25
        assert BitrateMode.MEDIUM.multiplier == 0.5
        assert BitrateMode.HIGH.multiplier == 1.0
        assert BitrateMode.VERY_HIGH.multiplier == 2.0
        assert BitrateMode.AUTO.multiplier == 1.0
        assert BitrateMode.CUSTOM.multiplier is None


class TestEncodingPreset:
    def test_all_presets_exist(self):
        expected = {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        }
        assert {p.value for p in EncodingPreset} == expected

    def test_av1_cpu_used(self):
        assert EncodingPreset.ULTRAFAST.av1_cpu_used == 8
        assert EncodingPreset.MEDIUM.av1_cpu_used == 3
        assert EncodingPreset.VERYSLOW.av1_cpu_used == 0


class TestInferenceResolution:
    def test_all_resolutions_exist(self):
        assert InferenceResolution.RES_512.value == 512
        assert InferenceResolution.RES_1024.value == 1024
        assert InferenceResolution.RES_2048.value == 2048


class TestProcessingConfigExtended:
    def test_new_defaults(self):
        config = ProcessingConfig()
        assert config.bitrate_mode == BitrateMode.AUTO
        assert config.custom_bitrate_mbps == 20.0
        assert config.encoding_preset == EncodingPreset.MEDIUM
        assert config.batch_size == 1
        assert config.inference_resolution == InferenceResolution.RES_1024

    def test_custom_new_fields(self):
        config = ProcessingConfig(
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=50.0,
            encoding_preset=EncodingPreset.SLOW,
            batch_size=4,
            inference_resolution=InferenceResolution.RES_2048,
        )
        assert config.bitrate_mode == BitrateMode.CUSTOM
        assert config.custom_bitrate_mbps == 50.0
        assert config.encoding_preset == EncodingPreset.SLOW
        assert config.batch_size == 4
        assert config.inference_resolution == InferenceResolution.RES_2048
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_config.py::TestBitrateMode -v
python -m pytest tests/test_config.py::TestEncodingPreset -v
python -m pytest tests/test_config.py::TestInferenceResolution -v
python -m pytest tests/test_config.py::TestProcessingConfigExtended -v
```

Expected: FAIL — `BitrateMode`, `EncodingPreset`, `InferenceResolution` not found.

- [ ] **Step 3: Implement new enums and extend ProcessingConfig**

In `src/core/config.py`, add after the `InputType` class and before `VIDEO_EXTENSIONS`:

```python
class BitrateMode(Enum):
    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CUSTOM = "custom"

    @property
    def multiplier(self) -> float | None:
        """Return bitrate multiplier relative to source, or None for CUSTOM."""
        return {
            BitrateMode.AUTO: 1.0,
            BitrateMode.LOW: 0.25,
            BitrateMode.MEDIUM: 0.5,
            BitrateMode.HIGH: 1.0,
            BitrateMode.VERY_HIGH: 2.0,
            BitrateMode.CUSTOM: None,
        }[self]


class EncodingPreset(Enum):
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"

    @property
    def av1_cpu_used(self) -> int:
        """Map preset name to libaom-av1 -cpu-used value (0-8)."""
        return {
            EncodingPreset.ULTRAFAST: 8,
            EncodingPreset.SUPERFAST: 7,
            EncodingPreset.VERYFAST: 6,
            EncodingPreset.FASTER: 5,
            EncodingPreset.FAST: 4,
            EncodingPreset.MEDIUM: 3,
            EncodingPreset.SLOW: 2,
            EncodingPreset.SLOWER: 1,
            EncodingPreset.VERYSLOW: 0,
        }[self]


class InferenceResolution(Enum):
    RES_512 = 512
    RES_1024 = 1024
    RES_2048 = 2048
```

Update `ProcessingConfig`:

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
```

- [ ] **Step 4: Run all config tests**

```bash
python -m pytest tests/test_config.py -v
```

Expected: ALL PASS (including existing tests — `ProcessingConfig()` defaults are backward compatible).

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/test_config.py
git commit -m "feat: add BitrateMode, EncodingPreset, InferenceResolution enums to config"
```

---

## Task 2: Add `DeviceInfo` for VRAM detection

**Files:**
- Create: `src/core/device_info.py`
- Create: `tests/test_device_info.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_device_info.py`:

```python
from unittest.mock import MagicMock, patch

from src.core.device_info import DeviceInfo, get_device_info, estimate_vram_gb


class TestDeviceInfo:
    def test_dataclass_fields(self):
        info = DeviceInfo(
            device="cuda",
            device_name="NVIDIA RTX 3060",
            total_vram_gb=12.0,
            available_vram_gb=9.2,
        )
        assert info.device == "cuda"
        assert info.device_name == "NVIDIA RTX 3060"
        assert info.total_vram_gb == 12.0
        assert info.available_vram_gb == 9.2


class TestGetDeviceInfoCPU:
    @patch("src.core.device_info.torch")
    def test_cpu_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        info = get_device_info()
        assert info.device == "cpu"
        assert info.device_name == "CPU"
        assert info.total_vram_gb == 0.0
        assert info.available_vram_gb == 0.0


class TestGetDeviceInfoCUDA:
    @patch("src.core.device_info.torch")
    def test_cuda_detection(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False

        props = MagicMock()
        props.name = "NVIDIA RTX 3060"
        props.total_mem = 12 * 1024**3  # 12 GB
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.mem_get_info.return_value = (
            int(9.2 * 1024**3),  # free
            12 * 1024**3,        # total
        )

        info = get_device_info()
        assert info.device == "cuda"
        assert info.device_name == "NVIDIA RTX 3060"
        assert abs(info.total_vram_gb - 12.0) < 0.1
        assert abs(info.available_vram_gb - 9.2) < 0.1


class TestGetDeviceInfoMPS:
    @patch("src.core.device_info.platform")
    @patch("src.core.device_info.psutil")
    @patch("src.core.device_info.torch")
    def test_mps_detection(self, mock_torch, mock_psutil, mock_platform):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_platform.processor.return_value = "arm"

        mem = MagicMock()
        mem.total = 16 * 1024**3  # 16 GB
        mem.available = 8 * 1024**3  # 8 GB
        mock_psutil.virtual_memory.return_value = mem

        info = get_device_info()
        assert info.device == "mps"
        assert "Apple" in info.device_name
        assert abs(info.total_vram_gb - 12.0) < 0.1  # 16 * 0.75
        assert abs(info.available_vram_gb - 6.0) < 0.1  # 8 * 0.75


class TestEstimateVramGb:
    def test_1024_batch_1(self):
        est = estimate_vram_gb(resolution=1024, batch_size=1)
        assert abs(est - 2.5) < 0.01

    def test_512_batch_1(self):
        est = estimate_vram_gb(resolution=512, batch_size=1)
        assert abs(est - 1.0) < 0.01

    def test_2048_batch_1(self):
        est = estimate_vram_gb(resolution=2048, batch_size=1)
        assert abs(est - 8.0) < 0.01

    def test_batch_scaling(self):
        est_1 = estimate_vram_gb(resolution=1024, batch_size=1)
        est_4 = estimate_vram_gb(resolution=1024, batch_size=4)
        # batch scaling is sub-linear: batch_size * 0.7
        assert est_4 > est_1
        assert est_4 < est_1 * 4
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_device_info.py -v
```

Expected: FAIL — `src.core.device_info` not found.

- [ ] **Step 3: Implement device_info module**

Create `src/core/device_info.py`:

```python
import platform
from dataclasses import dataclass

import psutil
import torch


@dataclass
class DeviceInfo:
    device: str           # "cuda" / "mps" / "cpu"
    device_name: str      # "NVIDIA RTX 3060" / "Apple M1 Pro" / "CPU"
    total_vram_gb: float  # Total VRAM in GB (0 for CPU)
    available_vram_gb: float  # Available VRAM in GB (0 for CPU)


def get_device_info() -> DeviceInfo:
    """Detect GPU device and VRAM information."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return DeviceInfo(
            device="cuda",
            device_name=props.name,
            total_vram_gb=total_bytes / (1024 ** 3),
            available_vram_gb=free_bytes / (1024 ** 3),
        )

    if torch.backends.mps.is_available():
        mem = psutil.virtual_memory()
        # Apple Silicon uses unified memory; estimate 75% available for GPU
        total_gb = mem.total / (1024 ** 3) * 0.75
        available_gb = mem.available / (1024 ** 3) * 0.75
        chip = "Apple Silicon"
        if platform.processor() == "arm":
            chip = "Apple Silicon"
        return DeviceInfo(
            device="mps",
            device_name=chip,
            total_vram_gb=total_gb,
            available_vram_gb=available_gb,
        )

    return DeviceInfo(
        device="cpu",
        device_name="CPU",
        total_vram_gb=0.0,
        available_vram_gb=0.0,
    )


# Empirical VRAM estimates (GB) per resolution at batch_size=1
_BASE_VRAM = {512: 1.0, 1024: 2.5, 2048: 8.0}


def estimate_vram_gb(resolution: int, batch_size: int) -> float:
    """Estimate VRAM usage in GB for given resolution and batch size.

    Batch scaling is sub-linear (factor 0.7) since model weights are shared.
    """
    base = _BASE_VRAM.get(resolution, 2.5)
    if batch_size <= 1:
        return base
    return base * (1 + (batch_size - 1) * 0.7)
```

- [ ] **Step 4: Add psutil to requirements.txt**

Check if psutil is already in requirements.txt. If not, add it:

```
psutil>=5.9
```

- [ ] **Step 5: Run tests**

```bash
python -m pytest tests/test_device_info.py -v
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/device_info.py tests/test_device_info.py requirements.txt
git commit -m "feat: add DeviceInfo for GPU/VRAM detection"
```

---

## Task 3: Extend `get_video_info()` to return bitrate

**Files:**
- Modify: `src/core/video.py`
- Modify: `tests/test_video.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_video.py`:

```python
class TestGetVideoInfoBitrate:
    def test_bitrate_mbps_returned(self, test_video_path):
        info = get_video_info(test_video_path)
        assert "bitrate_mbps" in info
        assert isinstance(info["bitrate_mbps"], float)
        assert info["bitrate_mbps"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_video.py::TestGetVideoInfoBitrate -v
```

Expected: FAIL — `bitrate_mbps` not in info dict.

- [ ] **Step 3: Implement bitrate detection via ffprobe**

In `src/core/video.py`, modify `get_video_info()` — add bitrate detection after the existing OpenCV block:

```python
def get_video_info(path: str) -> dict:
    """Get video metadata: width, height, fps, frame_count, duration, bitrate_mbps."""
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

    # Get bitrate via ffprobe
    info["bitrate_mbps"] = _get_bitrate_mbps(path)

    return info


def _get_bitrate_mbps(path: str) -> float:
    """Get video bitrate in Mbps using ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            bps = result.stdout.strip()
            if bps.isdigit():
                return int(bps) / 1_000_000
        # Fallback: try format-level bitrate
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=bit_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            bps = result.stdout.strip()
            if bps.isdigit():
                return int(bps) / 1_000_000
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 0.0
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_video.py -v
```

Expected: ALL PASS (existing tests unaffected, new test passes).

- [ ] **Step 5: Commit**

```bash
git add src/core/video.py tests/test_video.py
git commit -m "feat: get_video_info returns bitrate_mbps via ffprobe"
```

---

## Task 4: Add `predict_batch()` and variable resolution to inference

**Files:**
- Modify: `src/core/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_inference.py`:

```python
class TestPredictBatch:
    def test_batch_returns_list_of_masks(self, loaded_model):
        model, device = loaded_model
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        masks = predict_batch(model, frames, device)
        assert len(masks) == 3
        for mask in masks:
            assert mask.shape == (64, 64)
            assert mask.dtype == np.uint8

    def test_batch_size_one_matches_single(self, loaded_model):
        model, device = loaded_model
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        single_mask = predict(model, frame, device)
        batch_masks = predict_batch(model, [frame], device)
        assert len(batch_masks) == 1
        np.testing.assert_array_equal(batch_masks[0], single_mask)


class TestPredictResolution:
    def test_predict_with_512_resolution(self, loaded_model):
        model, device = loaded_model
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = predict(model, frame, device, resolution=512)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_predict_batch_with_resolution(self, loaded_model):
        model, device = loaded_model
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
        masks = predict_batch(model, frames, device, resolution=512)
        assert len(masks) == 2
        for mask in masks:
            assert mask.shape == (64, 64)
```

Update the import at the top of the test file to include `predict_batch`:

```python
from src.core.inference import detect_device, load_model, predict, predict_batch
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_inference.py::TestPredictBatch -v
python -m pytest tests/test_inference.py::TestPredictResolution -v
```

Expected: FAIL — `predict_batch` not found.

- [ ] **Step 3: Implement `predict_batch()` and resolution parameter**

In `src/core/inference.py`, replace the `_transform` and `predict` function, and add `predict_batch`:

```python
def _make_transform(resolution: int):
    """Create preprocessing transform for a given resolution."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# Cache transforms to avoid recreating them
_transform_cache: dict[int, transforms.Compose] = {}


def _get_transform(resolution: int) -> transforms.Compose:
    if resolution not in _transform_cache:
        _transform_cache[resolution] = _make_transform(resolution)
    return _transform_cache[resolution]


def predict(model, frame: np.ndarray, device: str, resolution: int = 1024) -> np.ndarray:
    """Run BiRefNet on a single BGR frame, return alpha mask at original resolution.

    Args:
        model: Loaded BiRefNet model.
        frame: BGR uint8 numpy array, shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.
        resolution: Model input resolution (512, 1024, or 2048).

    Returns:
        Alpha mask as uint8 numpy array, shape (H, W), values 0-255.
    """
    orig_h, orig_w = frame.shape[:2]
    transform = _get_transform(resolution)

    # BGR -> RGB -> PIL
    rgb = frame[:, :, ::-1]
    image = Image.fromarray(rgb)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)

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


def predict_batch(
    model,
    frames: list[np.ndarray],
    device: str,
    resolution: int = 1024,
) -> list[np.ndarray]:
    """Run BiRefNet on a batch of BGR frames, return alpha masks at original resolutions.

    Args:
        model: Loaded BiRefNet model.
        frames: List of BGR uint8 numpy arrays, each shape (H, W, 3).
        device: 'cuda', 'mps', or 'cpu'.
        resolution: Model input resolution (512, 1024, or 2048).

    Returns:
        List of alpha masks, each uint8 numpy array matching its input frame size.
    """
    if len(frames) == 0:
        return []

    transform = _get_transform(resolution)
    orig_sizes = [(f.shape[0], f.shape[1]) for f in frames]

    # Preprocess all frames
    tensors = []
    for frame in frames:
        rgb = frame[:, :, ::-1]
        image = Image.fromarray(rgb)
        tensors.append(transform(image))

    batch_tensor = torch.stack(tensors).to(device)

    # Batch inference
    with torch.no_grad():
        preds = model(batch_tensor)[-1]
        preds = torch.sigmoid(preds[:, 0])  # (B, H, W)

    # Resize each prediction back to its original size
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

Also remove the old `_transform` module-level variable (replaced by `_get_transform`).

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_inference.py -v
```

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/inference.py tests/test_inference.py
git commit -m "feat: add predict_batch and variable inference resolution"
```

---

## Task 5: Extend writers with bitrate and preset parameters

**Files:**
- Modify: `src/core/writer.py`
- Modify: `src/core/video.py`
- Modify: `tests/test_writer.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_writer.py`:

```python
class TestFFmpegWriterBitrate:
    def test_writes_h264_with_bitrate(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test_br.mp4")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libx264",
            pix_fmt="yuv420p",
            bitrate_kbps=5000,
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)

    def test_writes_h264_with_preset(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test_preset.mp4")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libx264",
            pix_fmt="yuv420p",
            preset="fast",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)


class TestProResWriterProfile:
    def test_writes_prores_with_profile(self, temp_output_dir):
        from src.core.video import ProResWriter
        output_path = os.path.join(temp_output_dir, "test_profile.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=0)  # Proxy
        for _ in range(5):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)


class TestCreateWriterAdvanced:
    def test_h264_with_bitrate_and_preset(self, temp_output_dir):
        from src.core.config import BitrateMode, EncodingPreset
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=10.0,
            encoding_preset=EncodingPreset.FAST,
        )
        output_path = os.path.join(temp_output_dir, "test_adv.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0, source_bitrate_mbps=20.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()

    def test_prores_uses_profile_not_bitrate(self, temp_output_dir):
        from src.core.config import BitrateMode
        from src.core.video import ProResWriter
        config = ProcessingConfig(
            output_format=OutputFormat.MOV_PRORES,
            bitrate_mode=BitrateMode.LOW,
        )
        output_path = os.path.join(temp_output_dir, "test_pr.mov")
        writer = create_writer(config, output_path, 64, 64, 30.0, source_bitrate_mbps=20.0)
        assert isinstance(writer, ProResWriter)
        writer.close()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_writer.py::TestFFmpegWriterBitrate -v
python -m pytest tests/test_writer.py::TestProResWriterProfile -v
python -m pytest tests/test_writer.py::TestCreateWriterAdvanced -v
```

Expected: FAIL — `bitrate_kbps`, `preset`, `profile` not accepted.

- [ ] **Step 3: Add bitrate and preset to FFmpegWriter**

In `src/core/writer.py`, update `FFmpegWriter.__init__` to accept `bitrate_kbps` and `preset`:

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
        bitrate_kbps: int | None = None,
        preset: str | None = None,
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

        if bitrate_kbps is not None:
            cmd.extend(["-b:v", f"{bitrate_kbps}k"])

        if preset is not None:
            # For AV1, preset is already mapped to -cpu-used in extra_args
            if codec not in ("libaom-av1",):
                cmd.extend(["-preset", preset])

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

The `write_frame`, `close`, `__enter__`, `__exit__` methods remain unchanged.

- [ ] **Step 4: Add profile parameter to ProResWriter**

In `src/core/video.py`, update `ProResWriter.__init__`:

```python
class ProResWriter:
    """Writes RGBA frames to a MOV file using ProRes 4444 via FFmpeg."""

    # ProRes profile levels: 0=Proxy, 1=LT, 2=Standard, 3=HQ
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: str | None = None,
        profile: int = 3,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height

        cmd = [
            "ffmpeg",
            "-y",
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
            "-profile:v", str(profile),
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

Note: Changed `-profile:v` from hardcoded `"4444"` to the numeric `profile` parameter. The old value `"4444"` was the profile name, but the numeric mapping is: 0=Proxy, 1=LT, 2=Standard, 3=HQ (which is what "4444" resolved to). Default is `3` (HQ) to match previous behavior.

- [ ] **Step 5: Update `create_writer()` to resolve bitrate/preset/profile from config**

In `src/core/writer.py`, update the `create_writer` function signature and logic:

```python
def _resolve_bitrate_kbps(config: ProcessingConfig, source_bitrate_mbps: float) -> int | None:
    """Resolve the target bitrate in kbps from config and source bitrate."""
    mode = config.bitrate_mode
    if mode == BitrateMode.CUSTOM:
        return int(config.custom_bitrate_mbps * 1000)
    multiplier = mode.multiplier
    if multiplier is not None:
        return int(source_bitrate_mbps * multiplier * 1000)
    return None


def _resolve_prores_profile(config: ProcessingConfig) -> int:
    """Map BitrateMode to ProRes profile level."""
    return {
        BitrateMode.AUTO: 3,      # HQ (default)
        BitrateMode.LOW: 0,       # Proxy
        BitrateMode.MEDIUM: 1,    # LT
        BitrateMode.HIGH: 2,      # Standard
        BitrateMode.VERY_HIGH: 3, # HQ
        BitrateMode.CUSTOM: 3,    # HQ (custom not applicable to ProRes)
    }[config.bitrate_mode]


def create_writer(
    config: ProcessingConfig,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
    source_bitrate_mbps: float = 0.0,
):
    """Factory: return the appropriate writer based on config."""
    fmt = config.output_format
    is_alpha = config.background_mode.needs_alpha

    # Side-by-side doubles width
    if config.background_mode == BackgroundMode.SIDE_BY_SIDE:
        width = width * 2

    if fmt == OutputFormat.MOV_PRORES:
        profile = _resolve_prores_profile(config)
        return ProResWriter(output_path, width, height, fps,
                            audio_source=audio_source, profile=profile)

    # Resolve bitrate for video codecs
    bitrate_kbps = _resolve_bitrate_kbps(config, source_bitrate_mbps)
    preset = config.encoding_preset.value

    if fmt == OutputFormat.WEBM_VP9:
        extra_args = ["-auto-alt-ref", "0"]
        if bitrate_kbps is not None:
            # VP9: use -b:v via bitrate_kbps param, no preset
            pass
        if is_alpha:
            return FFmpegWriter(
                output_path, width, height, fps,
                codec="libvpx-vp9",
                pix_fmt="yuva420p",
                input_pix_fmt="rgba",
                extra_args=extra_args,
                audio_source=audio_source,
                bitrate_kbps=bitrate_kbps,
            )
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libvpx-vp9",
            pix_fmt="yuv420p",
            extra_args=extra_args,
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
        )

    if fmt == OutputFormat.MP4_H264:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx264",
            pix_fmt="yuv420p",
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=preset,
        )

    if fmt == OutputFormat.MP4_H265:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx265",
            pix_fmt="yuv420p",
            extra_args=["-tag:v", "hvc1"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=preset,
        )

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

    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        return ImageSequenceWriter(output_path, fmt, is_alpha)

    raise ValueError(f"Unsupported output format: {fmt}")
```

Add the import at the top of `src/core/writer.py`:

```python
from src.core.config import BackgroundMode, BitrateMode, OutputFormat, ProcessingConfig
```

- [ ] **Step 6: Run all writer tests**

```bash
python -m pytest tests/test_writer.py -v
```

Expected: ALL PASS (existing tests still work because new params are optional with defaults).

- [ ] **Step 7: Commit**

```bash
git add src/core/writer.py src/core/video.py tests/test_writer.py
git commit -m "feat: writers support bitrate, preset, and ProRes profile params"
```

---

## Task 6: Wire batch inference into pipeline

**Files:**
- Modify: `src/core/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_pipeline.py`:

```python
class TestPipelineBatchInference:
    def test_infer_phase_with_batch_size(self, test_video_path, temp_output_dir):
        """Verify infer_phase works with batch_size > 1."""
        from src.core.config import ProcessingConfig, InferenceResolution
        config = ProcessingConfig(batch_size=2, inference_resolution=InferenceResolution.RES_512)

        cache_dir = os.path.join(temp_output_dir, "cache")
        cache = MaskCacheManager(cache_dir)

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.infer_phase(test_video_path, "test_batch", cache)

        info = get_video_info(test_video_path)
        assert cache.get_cached_count("test_batch") == info["frame_count"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_pipeline.py::TestPipelineBatchInference -v
```

Expected: FAIL — pipeline doesn't use batch_size yet.

- [ ] **Step 3: Modify `infer_phase` to use batch inference**

In `src/core/pipeline.py`, update the import and `infer_phase`:

```python
from src.core.inference import detect_device, get_model_path, load_model, predict, predict_batch
```

Update `infer_phase`:

```python
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
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        batch_size = self._config.batch_size
        resolution = self._config.inference_resolution.value

        # Validate cache on resume; invalidate if source file changed
        if start_frame > 0 and not cache.validate(task_id, video_info):
            cache.cleanup(task_id)
            start_frame = 0

        cache.save_metadata(task_id, video_info)

        batch_frames = []
        batch_indices = []

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

            batch_frames.append(frame)
            batch_indices.append(idx)

            if len(batch_frames) >= batch_size:
                masks = predict_batch(self._model, batch_frames, self._device, resolution)
                for bi, mask in zip(batch_indices, masks):
                    cache.save_mask(task_id, bi, mask)
                if progress_callback:
                    progress_callback(batch_indices[-1] + 1, total, "inference")
                batch_frames.clear()
                batch_indices.clear()

        # Process remaining frames
        if batch_frames:
            masks = predict_batch(self._model, batch_frames, self._device, resolution)
            for bi, mask in zip(batch_indices, masks):
                cache.save_mask(task_id, bi, mask)
            if progress_callback:
                progress_callback(batch_indices[-1] + 1, total, "inference")
```

- [ ] **Step 4: Update `encode_phase` to pass `source_bitrate_mbps` to writer**

In `encode_phase`, update the `create_writer` call:

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
    ) -> None:
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        width, height, fps = video_info["width"], video_info["height"], video_info["fps"]
        source_bitrate = video_info.get("bitrate_mbps", 0.0)

        writer = create_writer(
            self._config, output_path, width, height, fps,
            audio_source=input_path,
            source_bitrate_mbps=source_bitrate,
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
```

- [ ] **Step 5: Run all pipeline tests**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline uses batch inference and passes bitrate to writer"
```

---

## Task 7: Backward-compatible .brm serialization for new config fields

**Files:**
- Modify: `src/core/queue_task.py`
- Modify: `tests/test_queue_task.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_queue_task.py`:

```python
from src.core.config import BitrateMode, EncodingPreset, InferenceResolution


class TestQueueTaskBackwardCompat:
    def test_old_brm_without_new_fields_uses_defaults(self):
        """Simulate loading an old .brm file that has no new config fields."""
        old_dict = {
            "id": "abc123",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mov_prores",
                "background_mode": "transparent",
                # No bitrate_mode, encoding_preset, batch_size, etc.
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
        assert task.config.bitrate_mode == BitrateMode.AUTO
        assert task.config.custom_bitrate_mbps == 20.0
        assert task.config.encoding_preset == EncodingPreset.MEDIUM
        assert task.config.batch_size == 1
        assert task.config.inference_resolution == InferenceResolution.RES_1024


class TestQueueTaskNewFieldsSerialization:
    def test_roundtrip_with_new_config_fields(self):
        config = ProcessingConfig(
            model_name="BiRefNet-HR",
            output_format=OutputFormat.MP4_H265,
            background_mode=BackgroundMode.GREEN,
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=35.0,
            encoding_preset=EncodingPreset.SLOW,
            batch_size=4,
            inference_resolution=InferenceResolution.RES_2048,
        )
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        restored = QueueTask.from_dict(d)
        assert restored.config.bitrate_mode == BitrateMode.CUSTOM
        assert restored.config.custom_bitrate_mbps == 35.0
        assert restored.config.encoding_preset == EncodingPreset.SLOW
        assert restored.config.batch_size == 4
        assert restored.config.inference_resolution == InferenceResolution.RES_2048
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_queue_task.py::TestQueueTaskBackwardCompat -v
python -m pytest tests/test_queue_task.py::TestQueueTaskNewFieldsSerialization -v
```

Expected: FAIL — new fields not serialized/deserialized.

- [ ] **Step 3: Update `to_dict` and `from_dict` in QueueTask**

In `src/core/queue_task.py`, update the imports:

```python
from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncodingPreset,
    InferenceResolution,
    InputType,
    OutputFormat,
    ProcessingConfig,
)
```

Update `to_dict`:

```python
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_path": self.input_path,
            "input_type": self.input_type.value,
            "config": {
                "model_name": self.config.model_name,
                "output_format": self.config.output_format.value,
                "background_mode": self.config.background_mode.value,
                "bitrate_mode": self.config.bitrate_mode.value,
                "custom_bitrate_mbps": self.config.custom_bitrate_mbps,
                "encoding_preset": self.config.encoding_preset.value,
                "batch_size": self.config.batch_size,
                "inference_resolution": self.config.inference_resolution.value,
            },
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "phase": self.phase.value,
            "error": self.error,
            "created_at": self.created_at,
        }
```

Update `from_dict`:

```python
    @classmethod
    def from_dict(cls, d: dict) -> "QueueTask":
        cfg = d["config"]
        config = ProcessingConfig(
            model_name=cfg["model_name"],
            output_format=OutputFormat(cfg["output_format"]),
            background_mode=BackgroundMode(cfg["background_mode"]),
            bitrate_mode=BitrateMode(cfg.get("bitrate_mode", "auto")),
            custom_bitrate_mbps=cfg.get("custom_bitrate_mbps", 20.0),
            encoding_preset=EncodingPreset(cfg.get("encoding_preset", "medium")),
            batch_size=cfg.get("batch_size", 1),
            inference_resolution=InferenceResolution(cfg.get("inference_resolution", 1024)),
        )
        return cls(
            id=d["id"],
            input_path=d["input_path"],
            input_type=InputType(d["input_type"]),
            config=config,
            output_dir=d.get("output_dir"),
            output_path=d.get("output_path"),
            status=TaskStatus(d["status"]),
            progress=d.get("progress", 0),
            total=d.get("total", 0),
            phase=ProcessingPhase(d.get("phase", "inference")),
            error=d.get("error"),
            created_at=d["created_at"],
        )
```

- [ ] **Step 4: Run all queue task tests**

```bash
python -m pytest tests/test_queue_task.py -v
```

Expected: ALL PASS (including existing roundtrip test — it will now serialize/deserialize new defaults).

- [ ] **Step 5: Commit**

```bash
git add src/core/queue_task.py tests/test_queue_task.py
git commit -m "feat: queue task serialization supports new config fields with backward compat"
```

---

## Task 8: Create `SettingsPanel` widget

**Files:**
- Create: `src/gui/settings_panel.py`

- [ ] **Step 1: Implement SettingsPanel**

Create `src/gui/settings_panel.py`:

```python
import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncodingPreset,
    InferenceResolution,
    InputType,
    MODELS,
    OutputFormat,
    ProcessingConfig,
)
from src.core.device_info import DeviceInfo, estimate_vram_gb, get_device_info


FORMAT_LABELS = {
    OutputFormat.MOV_PRORES: "MOV ProRes 4444",
    OutputFormat.WEBM_VP9: "WebM VP9",
    OutputFormat.MP4_H264: "MP4 H.264",
    OutputFormat.MP4_H265: "MP4 H.265/HEVC",
    OutputFormat.MP4_AV1: "MP4 AV1",
    OutputFormat.PNG_SEQUENCE: "PNG 序列",
    OutputFormat.TIFF_SEQUENCE: "TIFF 序列",
}

MODE_LABELS = {
    BackgroundMode.TRANSPARENT: "透明背景",
    BackgroundMode.GREEN: "绿幕",
    BackgroundMode.BLUE: "蓝幕",
    BackgroundMode.MASK_BW: "黑底白蒙版",
    BackgroundMode.MASK_WB: "白底黑蒙版",
    BackgroundMode.SIDE_BY_SIDE: "原图+蒙版分轨",
}

BITRATE_LABELS = {
    BitrateMode.AUTO: "自动",
    BitrateMode.LOW: "低",
    BitrateMode.MEDIUM: "中",
    BitrateMode.HIGH: "高",
    BitrateMode.VERY_HIGH: "极高",
    BitrateMode.CUSTOM: "自定义",
}

PRORES_PROFILE_LABELS = {
    BitrateMode.AUTO: "HQ (默认)",
    BitrateMode.LOW: "Proxy",
    BitrateMode.MEDIUM: "LT",
    BitrateMode.HIGH: "Standard",
    BitrateMode.VERY_HIGH: "HQ",
}

RESOLUTION_LABELS = {
    InferenceResolution.RES_512: "512×512 (快速)",
    InferenceResolution.RES_1024: "1024×1024 (默认)",
    InferenceResolution.RES_2048: "2048×2048 (高质量)",
}

PRESET_LABELS = {
    EncodingPreset.ULTRAFAST: "ultrafast (最快)",
    EncodingPreset.SUPERFAST: "superfast",
    EncodingPreset.VERYFAST: "veryfast",
    EncodingPreset.FASTER: "faster",
    EncodingPreset.FAST: "fast",
    EncodingPreset.MEDIUM: "medium (默认)",
    EncodingPreset.SLOW: "slow",
    EncodingPreset.SLOWER: "slower",
    EncodingPreset.VERYSLOW: "veryslow (最慢)",
}


class SettingsPanel(QWidget):
    """Right-side settings panel with model, output, and advanced parameters."""

    settings_changed = pyqtSignal()

    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._source_bitrate_mbps = 0.0
        self._input_type: InputType | None = None
        self._device_info = get_device_info()

        self._init_ui()
        self._update_advanced_visibility()
        self._update_vram_warning()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Model settings ---
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型:"))
        self._model_combo = QComboBox()
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        model_layout.addWidget(self._model_combo)

        # Device info
        self._device_label = QLabel()
        self._device_label.setStyleSheet("color: gray; font-size: 11px;")
        self._update_device_label()
        model_layout.addWidget(self._device_label)

        # Inference resolution
        model_layout.addWidget(QLabel("推理分辨率:"))
        self._resolution_combo = QComboBox()
        for res, label in RESOLUTION_LABELS.items():
            self._resolution_combo.addItem(label, res)
        self._resolution_combo.setCurrentIndex(1)  # 1024 default
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_or_batch_changed)
        model_layout.addWidget(self._resolution_combo)

        # Batch size
        model_layout.addWidget(QLabel("Batch Size:"))
        self._batch_combo = QComboBox()
        for bs in [1, 2, 4, 8, 16]:
            self._batch_combo.addItem(str(bs), bs)
        self._set_recommended_batch_size()
        self._batch_combo.currentIndexChanged.connect(self._on_resolution_or_batch_changed)
        model_layout.addWidget(self._batch_combo)

        layout.addWidget(model_group)

        # --- Output settings ---
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
        self._mode_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        output_layout.addWidget(self._mode_combo)

        layout.addWidget(output_group)

        # --- Advanced parameters ---
        self._advanced_group = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout(self._advanced_group)

        # Bitrate
        self._bitrate_label = QLabel("码率:")
        advanced_layout.addWidget(self._bitrate_label)
        bitrate_row = QHBoxLayout()
        self._bitrate_combo = QComboBox()
        self._populate_bitrate_combo()
        self._bitrate_combo.currentIndexChanged.connect(self._on_bitrate_changed)
        bitrate_row.addWidget(self._bitrate_combo, stretch=1)

        self._custom_bitrate_spin = QDoubleSpinBox()
        self._custom_bitrate_spin.setRange(0.1, 200.0)
        self._custom_bitrate_spin.setSingleStep(0.1)
        self._custom_bitrate_spin.setSuffix(" Mbps")
        self._custom_bitrate_spin.setValue(20.0)
        self._custom_bitrate_spin.setVisible(False)
        self._custom_bitrate_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        bitrate_row.addWidget(self._custom_bitrate_spin)
        advanced_layout.addLayout(bitrate_row)

        # Encoding preset
        self._preset_label = QLabel("编码预设:")
        advanced_layout.addWidget(self._preset_label)
        self._preset_combo = QComboBox()
        for preset, label in PRESET_LABELS.items():
            self._preset_combo.addItem(label, preset)
        # Default to medium (index 5)
        self._preset_combo.setCurrentIndex(5)
        self._preset_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        advanced_layout.addWidget(self._preset_combo)

        layout.addWidget(self._advanced_group)

        # --- VRAM warning ---
        self._vram_warning = QLabel()
        self._vram_warning.setStyleSheet("color: #CC7700; font-size: 11px;")
        self._vram_warning.setWordWrap(True)
        self._vram_warning.setVisible(False)
        layout.addWidget(self._vram_warning)

        layout.addStretch()

    def _populate_model_combo(self):
        for display_name, dir_name in MODELS.items():
            model_path = os.path.join(self._models_dir, dir_name)
            if os.path.isdir(model_path):
                self._model_combo.addItem(display_name, display_name)
            else:
                self._model_combo.addItem(f"{display_name} (未下载)", display_name)
                idx = self._model_combo.count() - 1
                self._model_combo.model().item(idx).setEnabled(False)

    def _populate_mode_combo(self):
        self._mode_combo.clear()
        current_format = self._format_combo.currentData()
        for mode, label in MODE_LABELS.items():
            if mode.needs_alpha and current_format and not current_format.supports_alpha:
                continue
            self._mode_combo.addItem(label, mode)

    def _populate_bitrate_combo(self):
        """Populate bitrate combo with dynamic labels based on source bitrate."""
        self._bitrate_combo.clear()
        br = self._source_bitrate_mbps
        is_prores = self._format_combo.currentData() == OutputFormat.MOV_PRORES

        if is_prores:
            for mode, label in PRORES_PROFILE_LABELS.items():
                self._bitrate_combo.addItem(label, mode)
        else:
            if br > 0:
                for mode, label in BITRATE_LABELS.items():
                    if mode == BitrateMode.CUSTOM:
                        self._bitrate_combo.addItem(label, mode)
                    elif mode.multiplier is not None:
                        actual = br * mode.multiplier
                        self._bitrate_combo.addItem(f"{label} ({actual:.1f} Mbps)", mode)
                    else:
                        self._bitrate_combo.addItem(label, mode)
            else:
                for mode, label in BITRATE_LABELS.items():
                    self._bitrate_combo.addItem(label, mode)

    def _update_device_label(self):
        info = self._device_info
        if info.device == "cuda":
            text = f"设备: CUDA — {info.device_name} ({info.total_vram_gb:.1f} GB, 可用 {info.available_vram_gb:.1f} GB)"
        elif info.device == "mps":
            text = f"设备: MPS — {info.device_name} (统一内存 {info.total_vram_gb:.1f} GB)"
        else:
            text = "设备: CPU（无 GPU 加速）"
        self._device_label.setText(text)

    def _set_recommended_batch_size(self):
        """Set default batch size based on available VRAM."""
        available = self._device_info.available_vram_gb
        if available <= 0:
            recommended = 1
        elif available < 3:
            recommended = 1
        elif available < 6:
            recommended = 1
        elif available < 10:
            recommended = 2
        elif available < 16:
            recommended = 4
        else:
            recommended = 8

        batch_values = [1, 2, 4, 8, 16]
        if recommended in batch_values:
            self._batch_combo.setCurrentIndex(batch_values.index(recommended))

    def _on_format_changed(self, _index):
        self._populate_mode_combo()
        self._populate_bitrate_combo()
        self._update_advanced_visibility()
        self.settings_changed.emit()

    def _on_bitrate_changed(self, _index):
        mode = self._bitrate_combo.currentData()
        self._custom_bitrate_spin.setVisible(mode == BitrateMode.CUSTOM)
        self.settings_changed.emit()

    def _on_resolution_or_batch_changed(self, _index=None):
        self._update_vram_warning()
        self.settings_changed.emit()

    def _update_advanced_visibility(self):
        """Show/hide advanced params based on format and input type."""
        fmt = self._format_combo.currentData()
        is_image = self._input_type in (InputType.IMAGE, InputType.IMAGE_FOLDER)
        is_sequence = fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE)

        # Bitrate: hide for images and sequences
        show_bitrate = not is_image and not is_sequence
        self._bitrate_label.setVisible(show_bitrate)
        self._bitrate_combo.setVisible(show_bitrate)
        self._custom_bitrate_spin.setVisible(
            show_bitrate and self._bitrate_combo.currentData() == BitrateMode.CUSTOM
        )

        # Preset: only for H.264, H.265, AV1 (and not images)
        show_preset = not is_image and fmt in (
            OutputFormat.MP4_H264, OutputFormat.MP4_H265, OutputFormat.MP4_AV1,
        )
        self._preset_label.setVisible(show_preset)
        self._preset_combo.setVisible(show_preset)

        # Hide entire advanced group if nothing visible
        self._advanced_group.setVisible(show_bitrate or show_preset)

        # Batch size: hide for single image
        show_batch = self._input_type != InputType.IMAGE
        # Find batch label and combo in model group and toggle
        self._batch_combo.setVisible(show_batch)
        # Find the "Batch Size:" label — it's the widget before _batch_combo in the layout
        model_group_layout = self._batch_combo.parent().layout()
        if model_group_layout:
            for i in range(model_group_layout.count()):
                widget = model_group_layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and widget.text() == "Batch Size:":
                    widget.setVisible(show_batch)
                    break

    def _update_vram_warning(self):
        """Show warning if estimated VRAM exceeds available."""
        if self._device_info.device == "cpu":
            self._vram_warning.setVisible(False)
            return

        resolution = self._resolution_combo.currentData()
        batch_size = self._batch_combo.currentData()
        if resolution is None or batch_size is None:
            return

        estimated = estimate_vram_gb(resolution.value, batch_size)
        available = self._device_info.available_vram_gb

        if available > 0 and estimated > available * 0.9:
            self._vram_warning.setText(
                f"⚠ 预计需要 ~{estimated:.1f} GB 显存，当前可用 {available:.1f} GB，可能导致内存不足"
            )
            self._vram_warning.setVisible(True)
        else:
            self._vram_warning.setVisible(False)

    # --- Public API ---

    def set_input_type(self, input_type: InputType | None):
        """Update visibility based on input type."""
        self._input_type = input_type
        self._format_combo.setEnabled(input_type == InputType.VIDEO)
        self._update_advanced_visibility()

    def set_source_bitrate(self, bitrate_mbps: float):
        """Update bitrate combo labels when source video changes."""
        self._source_bitrate_mbps = bitrate_mbps
        self._populate_bitrate_combo()

    def get_config(self) -> ProcessingConfig:
        """Build ProcessingConfig from current UI selections."""
        return ProcessingConfig(
            model_name=self._model_combo.currentData(),
            output_format=self._format_combo.currentData(),
            background_mode=self._mode_combo.currentData(),
            bitrate_mode=self._bitrate_combo.currentData() or BitrateMode.AUTO,
            custom_bitrate_mbps=self._custom_bitrate_spin.value(),
            encoding_preset=self._preset_combo.currentData() or EncodingPreset.MEDIUM,
            batch_size=self._batch_combo.currentData() or 1,
            inference_resolution=self._resolution_combo.currentData() or InferenceResolution.RES_1024,
        )
```

- [ ] **Step 2: Commit**

```bash
git add src/gui/settings_panel.py
git commit -m "feat: add SettingsPanel widget with model, output, and advanced params"
```

---

## Task 9: Refactor MainWindow single-task tab layout

**Files:**
- Modify: `src/gui/main_window.py`

- [ ] **Step 1: Refactor MainWindow**

Replace the contents of `src/gui/main_window.py` with the new layout. Key changes:

1. Remove right panel setup code (model combo, format combo, mode combo, device label) — replaced by `SettingsPanel`
2. Remove `FORMAT_LABELS` and `MODE_LABELS` constants — moved to `settings_panel.py`
3. Move control buttons (start/pause/cancel/enqueue) out of Tab 1 into a fixed bottom bar
4. Bottom bar visibility toggles with tab changes (hidden on queue tab)
5. `_get_config()` delegates to `SettingsPanel.get_config()`

```python
import os
import time

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    FORMAT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    InputType,
    MODELS,
    VIDEO_EXTENSIONS,
)
from src.core.queue_manager import QueueManager
from src.core.queue_task import QueueTask, TaskStatus
from src.core.video import get_video_info
from src.gui.queue_tab import QueueTab
from src.gui.settings_panel import SettingsPanel
from src.worker.matting_worker import MattingWorker

# Path to bundled models directory (relative to project root)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Path to queue persistence file
BRM_PATH = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "queue.brm")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(800, 550)

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

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # --- Tab widget ---
        self._tabs = QTabWidget()
        outer_layout.addWidget(self._tabs, stretch=1)

        # --- Tab 1: Single Task ---
        tab1 = QWidget()
        self._tabs.addTab(tab1, "单任务")

        main_layout = QHBoxLayout(tab1)
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
        self._select_btn = QPushButton("选择文件 ▼")
        select_menu = QMenu(self)
        select_menu.addAction("选择视频", self._on_select_video)
        select_menu.addAction("选择图片", self._on_select_image)
        select_menu.addAction("选择图片文件夹", self._on_select_folder)
        self._select_btn.setMenu(select_menu)
        input_row.addWidget(self._select_btn)
        left_panel.addLayout(input_row)

        # File info
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

        left_panel.addStretch()

        # --- Right panel: SettingsPanel ---
        self._settings_panel = SettingsPanel(MODELS_DIR)

        # --- Assemble left-right ---
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addWidget(self._settings_panel, stretch=1)

        # --- Tab 2: Queue ---
        self._queue_tab = QueueTab(self._queue_manager, self._get_config)
        self._queue_tab.queue_running_changed.connect(self._on_queue_running_changed)
        self._queue_tab.task_count_changed.connect(
            lambda count: self._tabs.setTabText(1, f"批量队列 ({count})" if count > 0 else "批量队列")
        )
        self._tabs.addTab(self._queue_tab, self._get_queue_tab_title())

        # --- Fixed bottom action bar ---
        self._action_bar = QWidget()
        action_layout = QHBoxLayout(self._action_bar)
        action_layout.setContentsMargins(20, 8, 20, 12)
        action_layout.addStretch()

        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._on_start)
        action_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        action_layout.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._on_cancel)
        action_layout.addWidget(self._cancel_btn)

        self._enqueue_btn = QPushButton("加入队列")
        self._enqueue_btn.clicked.connect(self._on_enqueue)
        action_layout.addWidget(self._enqueue_btn)

        action_layout.addStretch()
        outer_layout.addWidget(self._action_bar)

        # Toggle action bar visibility on tab switch
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index: int):
        """Show action bar only on single-task tab."""
        self._action_bar.setVisible(index == 0)

    def _get_config(self) -> ProcessingConfig:
        return self._settings_panel.get_config()

    def _set_state(self, state: str):
        self._state = state
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._select_btn.setEnabled(True)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._select_btn.setEnabled(True)
            if self._queue_tab._queue_state == "idle":
                has_pending = self._queue_manager.next_pending_task() is not None
                self._queue_tab._start_btn.setEnabled(has_pending)

    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self._handle_input(path)

    def _classify_input(self, path: str) -> InputType | None:
        if os.path.isdir(path):
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
            bitrate = info.get("bitrate_mbps", 0.0)
            minutes = int(dur // 60)
            seconds = int(dur % 60)
            bitrate_str = f" | {bitrate:.1f} Mbps" if bitrate > 0 else ""
            self._info_label.setText(
                f"视频信息: {w}x{h} | {fps:.1f}fps | {frames}帧 | {minutes:02d}:{seconds:02d}{bitrate_str}"
            )
            self._settings_panel.set_source_bitrate(bitrate)

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
            self._settings_panel.set_source_bitrate(0.0)

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
            self._settings_panel.set_source_bitrate(0.0)

        self._settings_panel.set_input_type(input_type)
        self._set_state("ready")

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

    def _on_select_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self._output_dir = directory
            self._output_edit.setText(directory)

    def _build_output_path(self) -> str:
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
        elif self._input_type == InputType.IMAGE_FOLDER:
            folder_name = os.path.basename(self._input_path.rstrip(os.sep))
            subdir = f"{folder_name}_{model_dir_name}_{timestamp}"
            base_dir = self._output_dir or os.path.dirname(self._input_path)
            return os.path.join(base_dir, subdir)
        else:
            if self._output_dir:
                return self._output_dir
            return os.path.dirname(self._input_path)

    def _on_start(self):
        if not self._input_path:
            return
        if self._queue_tab._queue_state != "idle":
            QMessageBox.warning(self, "提示", "队列正在执行中，请等待队列完成")
            return

        config = self._get_config()
        models_dir = os.path.abspath(MODELS_DIR)

        model_dir_name = MODELS[config.model_name]
        model_path = os.path.join(models_dir, model_dir_name)
        if not os.path.isdir(model_path):
            QMessageBox.critical(
                self, "模型缺失",
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

    def _on_progress(self, current: int, total: int, phase: str):
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

    def _on_error(self, message: str):
        self._set_state("ready")
        self._progress_bar.setValue(0)
        if message != "Processing cancelled":
            QMessageBox.critical(self, "错误", f"处理出错:\n{message}")
            self._status_label.setText(f"错误: {message}")

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
        self._queue_tab.refresh()

        self._input_path = None
        self._input_type = None
        self._input_edit.setText("")
        self._info_label.setText("")
        self._output_dir = None
        self._output_edit.setText("")
        self._settings_panel.set_input_type(None)
        self._set_state("initial")

        self.statusBar().showMessage("已加入队列", 3000)

    def _on_queue_running_changed(self, running: bool):
        if running:
            self._start_btn.setEnabled(False)
            has_input = self._input_path is not None
            self._select_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(has_input)
        else:
            self._set_state(self._state)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        if hasattr(self, '_queue_tab') and self._queue_tab._current_worker:
            worker = self._queue_tab._current_worker
            worker.disconnect()
            worker.cancel()
            worker.wait()
            task = self._queue_tab._current_running_task()
            if task and task.status in (TaskStatus.PROCESSING, TaskStatus.CANCELLED):
                task.status = TaskStatus.PAUSED
        self._queue_manager.save()
        event.accept()
```

- [ ] **Step 2: Run the application manually to verify layout**

```bash
cd ~/birefnet-gui && source venv/bin/activate && python main.py
```

Verify:
- Single-task tab shows left (input/output/progress) and right (settings panel with model, output, advanced params)
- Bottom action bar shows [开始处理] [暂停] [取消] [加入队列]
- Switching to queue tab hides the bottom action bar
- Format changes update bitrate/preset visibility
- Selecting an image hides video-only params
- VRAM warning appears when batch size or resolution is too high

- [ ] **Step 3: Commit**

```bash
git add src/gui/main_window.py
git commit -m "feat: refactor single-task tab to left-right layout with fixed bottom action bar"
```

---

## Task 10: Wire `ImagePipeline` with resolution parameter

**Files:**
- Modify: `src/core/image_pipeline.py`

- [ ] **Step 1: Update ImagePipeline to use inference_resolution from config**

In `src/core/image_pipeline.py`, update the `predict()` calls to pass resolution:

In `_process_single`:
```python
        resolution = self._config.inference_resolution.value
        alpha = predict(self._model, frame, self._device, resolution=resolution)
```

In `_process_folder`:
```python
            resolution = self._config.inference_resolution.value
            alpha = predict(self._model, frame, self._device, resolution=resolution)
```

- [ ] **Step 2: Run existing image pipeline tests**

```bash
python -m pytest tests/test_image_pipeline.py -v
```

Expected: ALL PASS (resolution defaults to 1024, same as before).

- [ ] **Step 3: Commit**

```bash
git add src/core/image_pipeline.py
git commit -m "feat: ImagePipeline uses inference_resolution from config"
```

---

## Task 11: Run full test suite and verify

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: ALL PASS.

- [ ] **Step 2: Run the application for manual smoke test**

```bash
cd ~/birefnet-gui && source venv/bin/activate && python main.py
```

Verify end-to-end:
1. Select a video → video info shows bitrate
2. Right panel shows all settings with correct defaults
3. Change format to H.264 → preset combo appears, bitrate shows dynamic values
4. Change format to PNG 序列 → bitrate and preset hidden
5. Select image folder → batch size visible, bitrate/preset hidden
6. VRAM warning triggers with high batch + resolution
7. Add to queue → queue tab shows task with all settings
8. Close and reopen → queue restores with all settings intact

- [ ] **Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: address smoke test issues"
```

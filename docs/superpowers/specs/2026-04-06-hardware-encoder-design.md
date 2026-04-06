# Hardware Encoder Support Design

## Overview

Add hardware-accelerated video encoding support to the export pipeline. Users can select a hardware encoder (NVENC, VideoToolbox, QSV, AMF) when exporting H.264/H.265 videos, or rely on auto-detection to pick the best available encoder.

## Goals

1. Support hardware encoding for H.264 and H.265 across NVIDIA (NVENC), Apple (VideoToolbox), Intel (QSV), and AMD (AMF)
2. Provide an "Encoder" dropdown in the export settings UI, dynamically filtered to show only available encoders
3. Auto-detect the best hardware encoder when no explicit choice is made (backward-compatible with old queue tasks)
4. Gracefully fall back to software encoding if hardware encoding fails at runtime

## Non-Goals

- Hardware encoding for AV1 (keep software-only libaom-av1)
- Hardware encoding for ProRes, VP9 WebM, PNG/TIFF sequences
- Hardware-accelerated decoding (input side)

---

## Data Model

### New Enum: `EncoderType` (in `config.py`)

```python
class EncoderType(str, Enum):
    AUTO = "auto"                     # Auto-detect best available
    SOFTWARE = "software"             # Force software (libx264/libx265)
    NVENC = "nvenc"                   # NVIDIA NVENC
    VIDEOTOOLBOX = "videotoolbox"     # Apple VideoToolbox
    QSV = "qsv"                      # Intel Quick Sync Video
    AMF = "amf"                       # AMD AMF
```

### `ProcessingConfig` New Field

```python
encoder_type: EncoderType = EncoderType.AUTO
```

### Backward Compatibility

Old `.brm` queue files that lack `encoder_type` will deserialize with the default value `AUTO`, which triggers auto-detection. No migration needed.

---

## EncoderRegistry Module

New file: `src/core/encoder_registry.py`

### Encoder Candidate Map

Each output format maps to a dict of `EncoderType → ffmpeg_encoder_name`, ordered by priority:

| Format | NVENC | VideoToolbox | QSV | AMF | Software |
|--------|-------|--------------|-----|-----|----------|
| MP4 H.264 | h264_nvenc | h264_videotoolbox | h264_qsv | h264_amf | libx264 |
| MP4 H.265 | hevc_nvenc | hevc_videotoolbox | hevc_qsv | hevc_amf | libx265 |

### AUTO Priority Order

NVENC > VideoToolbox > QSV > AMF > Software

### `EncoderRegistry` Class

- **`_probe()`**: Runs `ffmpeg -hide_banner -encoders` once at startup (~100ms), parses output to build a set of available encoder names.
- **`is_available(encoder_name: str) -> bool`**: Checks if a specific ffmpeg encoder is available.
- **`get_available_types(fmt: OutputFormat) -> list[EncoderType]`**: Returns `[AUTO, ...available types...]` for the given format. Used by UI to populate the dropdown.
- **`resolve(fmt: OutputFormat, encoder_type: EncoderType) -> str`**: Returns the actual ffmpeg encoder name. For `AUTO`, iterates the priority list and returns the first available. Falls back to software if nothing else is available.

### Singleton

Created once at application startup, passed to SettingsPanel and Writer via dependency injection.

---

## Writer Layer Changes

### FFmpegWriter Modifications

The `create_writer()` factory receives the `EncoderRegistry` instance. Before constructing the FFmpeg command, it calls `registry.resolve(fmt, config.encoder_type)` to determine the actual encoder.

### Per-Encoder FFmpeg Arguments

**Software (existing, unchanged):**
```
-c:v libx264 -preset {preset} -b:v {bitrate}k
```

**NVENC:**
```
-c:v h264_nvenc -preset {nvenc_preset} -rc vbr -cq 23 -b:v {bitrate}k
```

**VideoToolbox:**
```
-c:v h264_videotoolbox -b:v {bitrate}k -realtime 0 -allow_sw 1
```
Note: VideoToolbox does not support preset selection.

**QSV:**
```
-c:v h264_qsv -preset {qsv_preset} -b:v {bitrate}k
```

**AMF:**
```
-c:v h264_amf -quality {amf_quality} -b:v {bitrate}k
```

### Unified Preset Mapping

The existing 9-level preset (ultrafast through veryslow) maps to each hardware encoder's native presets:

| Abstract Preset | libx264/5 | NVENC | QSV | AMF | VideoToolbox |
|-----------------|-----------|-------|-----|-----|--------------|
| ultrafast | ultrafast | p1 | veryfast | speed | (ignored) |
| superfast | superfast | p2 | faster | speed | (ignored) |
| veryfast | veryfast | p3 | fast | balanced | (ignored) |
| faster | faster | p4 | medium | balanced | (ignored) |
| fast | fast | p4 | medium | balanced | (ignored) |
| medium | medium | p5 | medium | balanced | (ignored) |
| slow | slow | p6 | slow | quality | (ignored) |
| slower | slower | p6 | slower | quality | (ignored) |
| veryslow | veryslow | p7 | veryslow | quality | (ignored) |

### Fallback Mechanism

In `encode_phase` of `MattingPipeline`:

1. Start FFmpeg process with the resolved encoder
2. If the process exits with non-zero return code before the first frame is fully written (detected by checking `process.poll()` after the first `stdin.write()`):
   a. Log a warning with the stderr output
   b. Switch `encoder_type` to `SOFTWARE`
   c. Re-create the writer with the software encoder
   d. Retry encoding from the beginning
   e. Notify UI via progress callback: "Hardware encoder failed, falling back to software encoding"
3. This fallback happens at most once per export task

---

## UI Changes

### SettingsPanel (`settings_panel.py`)

**New "Encoder" Dropdown:**
- Placed directly below the "Format" dropdown
- Options are dynamically generated from `registry.get_available_types(current_format)`
- Default selection: `AUTO` ("Auto-detect")
- Display names:
  - `AUTO` → "自动检测"
  - `SOFTWARE` → "软件编码"
  - `NVENC` → "NVIDIA NVENC"
  - `VIDEOTOOLBOX` → "Apple VideoToolbox"
  - `QSV` → "Intel QSV"
  - `AMF` → "AMD AMF"

**Format-Encoder Linkage:**
- When format is H.264 or H.265: encoder dropdown is visible, options refreshed
- When format is ProRes / VP9 / AV1 / PNG / TIFF: encoder dropdown hidden

**Preset-Encoder Linkage:**
- When encoder is VideoToolbox: preset dropdown is grayed out (not supported)
- For all other encoders: preset dropdown works normally

**Auto-detect Hint:**
- When "Auto-detect" is selected, show a small label next to the dropdown indicating the resolved encoder, e.g. "→ NVIDIA NVENC"

**`get_config()` Update:**
- Includes `encoder_type` field in the returned `ProcessingConfig`

---

## Files to Modify

| File | Change |
|------|--------|
| `src/core/config.py` | Add `EncoderType` enum, add `encoder_type` field to `ProcessingConfig` |
| `src/core/encoder_registry.py` | **New file** — `EncoderRegistry` class with probe/resolve logic |
| `src/core/writer.py` | Accept registry, resolve encoder, build per-encoder FFmpeg args, preset mapping |
| `src/core/video.py` | Remove `_has_encoder()` (subsumed by registry), update `ProResWriter` to use registry |
| `src/core/pipeline.py` | Add fallback logic in `encode_phase` |
| `src/gui/settings_panel.py` | Add encoder dropdown, format/preset linkage, auto-detect hint |
| `src/gui/main_window.py` | Pass registry to SettingsPanel |
| `src/gui/queue_tab.py` | Pass registry to worker/pipeline |
| Entry point (main.py or app init) | Create `EncoderRegistry` singleton at startup |

---

## Testing Strategy

1. **Unit tests for EncoderRegistry**: Mock `ffmpeg -encoders` output, verify `get_available_types()` and `resolve()` logic
2. **Preset mapping tests**: Verify all 9 presets map correctly for each encoder type
3. **Backward compatibility test**: Deserialize old `.brm` file without `encoder_type`, verify it defaults to `AUTO`
4. **Fallback test**: Mock FFmpeg process failure, verify fallback to software encoding
5. **Manual integration test**: Export a short video with each available hardware encoder, verify output is valid

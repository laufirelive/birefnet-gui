# P1 核心功能设计：多模型 + 多输出格式 + 多输出模式

**日期**: 2026-04-04
**基于**: DESIGN.md 3.1.2, 3.1.3, 4.1
**前置**: MVP 已完成（单模型 BiRefNet-general, MOV ProRes 4444, 透明背景）

---

## 1. 目标

在 MVP 基础上实现三个核心功能：
1. **多模型支持** — 6 种 BiRefNet 模型自由切换
2. **多输出格式** — MOV/WebM/MP4/PNG序列/TIFF序列
3. **多输出模式** — 透明/绿幕/蓝幕/黑白蒙版/白黑蒙版/原图+蒙版分轨

## 2. 架构方案：配置对象 + Writer 工厂

引入 `ProcessingConfig` 承载用户选择，Pipeline 根据 config 选择 Writer 和合成方式。

### 2.1 新增文件

| 文件 | 职责 |
|------|------|
| `src/core/config.py` | ProcessingConfig 数据类 + OutputFormat/BackgroundMode 枚举 + MODELS 字典 |
| `src/core/writer.py` | FFmpegWriter + ImageSequenceWriter + create_writer() 工厂函数 |
| `src/core/compositing.py` | compose_frame() 背景合成函数 |

### 2.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/core/pipeline.py` | 接收 ProcessingConfig，使用 create_writer + compose_frame |
| `src/core/inference.py` | 新增 get_model_path() 辅助函数 |
| `src/worker/matting_worker.py` | 传递 ProcessingConfig 而非 model_path |
| `src/gui/main_window.py` | 左右分栏布局 + 模型/格式/模式下拉框 |
| `download_models.py` | 扩展为支持全部 6 个模型 |

## 3. ProcessingConfig

```python
from dataclasses import dataclass
from enum import Enum

class OutputFormat(Enum):
    MOV_PRORES = "mov_prores"       # MOV ProRes 4444 (透明)
    WEBM_VP9 = "webm_vp9"          # WebM VP9 (透明)
    MP4_H264 = "mp4_h264"          # MP4 H.264
    MP4_H265 = "mp4_h265"          # MP4 H.265/HEVC
    MP4_AV1 = "mp4_av1"            # MP4 AV1
    PNG_SEQUENCE = "png_sequence"   # PNG 序列 (RGBA)
    TIFF_SEQUENCE = "tiff_sequence" # TIFF 序列

class BackgroundMode(Enum):
    TRANSPARENT = "transparent"     # 保留 Alpha
    GREEN = "green"                 # 绿幕 #00FF00
    BLUE = "blue"                   # 蓝幕 #0000FF
    MASK_BW = "mask_bw"            # 黑底白蒙版
    MASK_WB = "mask_wb"            # 白底黑蒙版
    SIDE_BY_SIDE = "side_by_side"  # 原图+蒙版分轨

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

## 4. Writer 工厂

### 4.1 Writer 接口

所有 Writer 实现统一接口：
- `write_frame(frame: np.ndarray)` — 写一帧
- `close()` — 关闭资源
- 支持 context manager (`__enter__`/`__exit__`)

### 4.2 Writer 类型

| Writer | 用于 | 输入格式 |
|--------|------|---------|
| ProResWriter | MOV ProRes 4444 | RGBA uint8 |
| FFmpegWriter | WebM VP9, MP4 H.264/H.265/AV1 | RGBA 或 RGB uint8 |
| ImageSequenceWriter | PNG/TIFF 序列 | RGBA 或 RGB uint8 |

### 4.3 create_writer() 工厂

```python
def create_writer(config, output_path, width, height, fps):
    # 根据 config.output_format 返回对应 writer
    # SIDE_BY_SIDE 模式 width *= 2
    # 透明模式 pix_fmt = rgba, 其他 = rgb24
```

### 4.4 FFmpegWriter 参数

| 格式 | 编码器 | 容器 | 支持透明 |
|------|--------|------|---------|
| WebM VP9 | libvpx-vp9 | .webm | Y (yuva420p) |
| MP4 H.264 | libx264 | .mp4 | N (yuv420p) |
| MP4 H.265 | libx265 | .mp4 | N (yuv420p) |
| MP4 AV1 | libaom-av1 | .mp4 | N (yuv420p) |

## 5. 背景合成

`compose_frame(bgr_frame, alpha_mask, mode) -> np.ndarray`

| 模式 | 输出通道 | 逻辑 |
|------|---------|------|
| TRANSPARENT | 4 (RGBA) | RGB + alpha |
| GREEN | 3 (RGB) | alpha 混合前景与 #00FF00 |
| BLUE | 3 (RGB) | alpha 混合前景与 #0000FF |
| MASK_BW | 3 (RGB) | alpha → 灰度转三通道 |
| MASK_WB | 3 (RGB) | (255 - alpha) → 灰度转三通道 |
| SIDE_BY_SIDE | 3 (RGB) | 水平拼接：左=原图RGB，右=蒙版三通道，宽度x2 |

## 6. 格式-模式兼容矩阵

| 格式 | 透明 | 绿/蓝幕 | 蒙版 | 分轨 |
|------|------|---------|------|------|
| MOV ProRes 4444 | Y | Y | Y | Y |
| WebM VP9 | Y | Y | Y | Y |
| MP4 H.264 | **N** | Y | Y | Y |
| MP4 H.265 | **N** | Y | Y | Y |
| MP4 AV1 | **N** | Y | Y | Y |
| PNG 序列 | Y | Y | Y | Y |
| TIFF 序列 | Y | Y | Y | Y |

GUI 联动：选择不支持透明的格式时，背景模式下拉框自动排除"透明"选项。

## 7. 多模型管理

- `get_model_path(model_name, models_dir)` 将显示名映射到本地目录路径
- GUI 模型下拉框扫描 `models/` 目录，仅列出已下载模型
- 未下载模型灰显并带 "(未下载)" 后缀
- `download_models.py` 扩展为全部 6 个模型

## 8. Pipeline 改造

```python
class MattingPipeline:
    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def process(self, input_path, output_path, progress_callback, pause_event, cancel_event):
        video_info = get_video_info(input_path)
        writer = create_writer(self._config, output_path, ...)
        with writer:
            for frame_idx, frame in enumerate(FrameReader(input_path), 1):
                # pause/cancel checks
                alpha = predict(self._model, frame, self._device)
                composed = compose_frame(frame, alpha, self._config.background_mode)
                writer.write_frame(composed)
                # progress callback
```

## 9. GUI 左右分栏布局

```
┌────────────────────────────────────────────────────────────────┐
│  BiRefNet Video Matting Tool                                   │
├───────────────────────────┬────────────────────────────────────┤
│  左侧 (stretch=2)         │  右侧设置面板 (stretch=1)          │
│                           │                                    │
│  输入文件: [____] [选择]   │  模型: [BiRefNet-general     ▼]   │
│  视频信息: 1920x1080...   │  设备: MPS (Apple Silicon)         │
│                           │                                    │
│  输出路径: [____] [浏览]   │  ── 输出设置 ──                    │
│                           │  格式: [MOV ProRes 4444      ▼]   │
│  进度: [████████░░] 67%   │  背景: [透明背景            ▼]    │
│  帧: 1523/2340 | 8.5 FPS │                                    │
│                           │                                    │
│  [开始处理] [暂停] [取消]  │                                    │
├───────────────────────────┴────────────────────────────────────┤
```

- QHBoxLayout 分左右栏，左=2 右=1
- 右栏：QComboBox x3（模型、格式、模式）
- 格式-模式联动：MP4 格式自动隐藏"透明"选项

## 10. 输出路径逻辑

- 视频格式：`{filename}_{model}_{timestamp}.{ext}`
- 图片序列：`{filename}_{model}_{timestamp}/frame_000001.png`
- 扩展名映射：MOV→.mov, WebM→.webm, MP4→.mp4

## 11. 测试计划

| 模块 | 测试内容 |
|------|---------|
| config.py | 枚举值、默认值 |
| compositing.py | 每种 BackgroundMode 的输出通道数和像素值 |
| writer.py | create_writer 返回正确类型、FFmpegWriter 写入、ImageSequenceWriter 写入 |
| pipeline.py | 使用不同 config 的端到端处理 |
| main_window.py | 格式-模式联动逻辑 |

# P3 设计：高级输出参数 + 显存检测 + 布局重构

**日期**: 2026-04-04
**基于**: PROGRESS.md 未完成功能列表
**前置**: P2 批量队列已完成

---

## 1. 目标

在现有功能基础上实现三个关联模块：
1. **高级输出参数** — 码率控制、编码预设、batch size
2. **显存检测与自适应** — GPU 显存检测、推理分辨率选择、OOM 预警
3. **单任务 Tab 布局重构** — 上(左右分栏) + 下(固定操作栏)

---

## 2. 高级输出参数

### 2.1 码率控制

用户通过下拉框选择码率档位，预设值基于输入视频原始码率动态计算。

**码率档位：**

| 档位 | 显示文本 | 实际值 |
|------|---------|--------|
| 自动 | `自动 ({原码率} Mbps)` | 原视频码率（默认选项） |
| 低 | `低 ({原码率×0.25:.1f} Mbps)` | 原码率 × 0.25 |
| 中 | `中 ({原码率×0.5:.1f} Mbps)` | 原码率 × 0.5 |
| 高 | `高 ({原码率×1.0:.1f} Mbps)` | 原码率 × 1.0 |
| 极高 | `极高 ({原码率×2.0:.1f} Mbps)` | 原码率 × 2.0 |
| 自定义 | `自定义` | 用户输入，单位 Mbps |

**原视频码率获取**：扩展 `get_video_info()` 返回 `bitrate_mbps` 字段，通过 ffprobe 获取。

**自定义输入**：选择"自定义"时，下拉框右侧出现 QDoubleSpinBox，范围 0.1 ~ 200.0 Mbps，步长 0.1。

**格式适用性**：
- MOV ProRes 4444：ProRes 编码器不支持精确码率控制，使用 profile 级别（proxy/lt/standard/hq）代替。码率下拉框改为 ProRes profile 选择。
- WebM VP9 / MP4 H.264 / H.265 / AV1：码率生效，通过 `-b:v` 参数传递。
- PNG / TIFF 序列：无损格式，码率设置隐藏。

**ProRes profile 映射**：

| 档位 | ProRes profile | FFmpeg 参数 |
|------|---------------|-------------|
| 自动 | HQ (默认) | `-profile:v 3` |
| 低 | Proxy | `-profile:v 0` |
| 中 | LT | `-profile:v 1` |
| 高 | Standard | `-profile:v 2` |
| 极高 | HQ | `-profile:v 3` |

### 2.2 编码预设

仅对 H.264 / H.265 / AV1 生效。

**下拉框选项**：

| 预设 | 含义 |
|------|------|
| ultrafast | 最快编码，文件最大 |
| superfast | — |
| veryfast | — |
| faster | — |
| fast | — |
| medium | 平衡（**默认**） |
| slow | — |
| slower | — |
| veryslow | 最慢编码，文件最小 |

**格式适用性**：
- H.264 / H.265：通过 `-preset` 参数。
- AV1 (libaom-av1)：AV1 使用 `-cpu-used` 参数（0~8），映射预设名到对应值。
- MOV ProRes / WebM VP9 / PNG / TIFF：预设下拉框隐藏。

**AV1 预设映射**：

| 预设名 | `-cpu-used` 值 |
|--------|----------------|
| ultrafast | 8 |
| superfast | 7 |
| veryfast | 6 |
| faster | 5 |
| fast | 4 |
| medium | 3 |
| slow | 2 |
| slower | 1 |
| veryslow | 0 |

### 2.3 Batch Size

**下拉框选项**：1 / 2 / 4 / 8 / 16，默认根据显存自动推荐。

**推理改造**：`predict()` 函数新增 `predict_batch()` 变体，接受帧列表，返回 mask 列表。Pipeline 在推理阶段攒够 batch_size 帧后一次性推理。不足 batch_size 的最后一批正常处理。

### 2.4 参数适用性矩阵（按输入类型 & 输出格式）

| 参数 | 视频输入 | 图片输入 | 图片文件夹 |
|------|---------|---------|-----------|
| 码率/ProRes profile | ✅ (非序列格式) | ❌ 隐藏 | ❌ 隐藏 |
| 编码预设 | ✅ (H.264/H.265/AV1) | ❌ 隐藏 | ❌ 隐藏 |
| Batch size | ✅ | ❌ 隐藏 | ✅ |
| 推理分辨率 | ✅ | ✅ | ✅ |

---

## 3. 显存检测与自适应

### 3.1 显存检测

**CUDA**：`torch.cuda.get_device_properties(0).total_mem` + `torch.cuda.mem_get_info()` 获取总显存和可用显存。

**MPS (Apple Silicon)**：macOS 统一内存架构，无独立显存。通过 `psutil.virtual_memory()` 获取系统总内存，按 75% 估算可用于 GPU 的内存上限。

**CPU**：不涉及显存，batch size 和分辨率不受限（但速度极慢，给出提示即可）。

**新增 `src/core/device_info.py`**：

```python
@dataclass
class DeviceInfo:
    device: str           # "cuda" / "mps" / "cpu"
    device_name: str      # "NVIDIA RTX 3060" / "Apple M1" / "CPU"
    total_vram_gb: float  # 总显存 GB（CPU 时为系统内存）
    available_vram_gb: float  # 可用显存 GB

def get_device_info() -> DeviceInfo: ...
```

### 3.2 推理分辨率

| 分辨率 | 模型输入尺寸 | 效果 | 显存占用参考 (batch=1) |
|--------|------------|------|----------------------|
| 512×512 | 低质量，速度快 | ~1 GB |
| 1024×1024 | 默认，平衡 | ~2.5 GB |
| 2048×2048 | 最高质量 | ~8 GB |

**实现**：修改 `_transform` 为参数化，`predict()` 和 `predict_batch()` 接受 `inference_resolution` 参数。

### 3.3 自动推荐

启动时检测显存，设置默认值：

| 可用显存 | 推荐分辨率 | 推荐 batch size |
|---------|-----------|----------------|
| < 3 GB | 512 | 1 |
| 3 ~ 6 GB | 1024 | 1 |
| 6 ~ 10 GB | 1024 | 2 |
| 10 ~ 16 GB | 1024 | 4 |
| > 16 GB | 1024 | 8 |

注意：推荐值只设默认值，用户可以自由调整。

### 3.4 OOM 预警

**预估显存占用公式**（经验值，后续可调）：

```
base_vram = {512: 1.0, 1024: 2.5, 2048: 8.0}  # GB, batch=1
estimated_vram = base_vram[resolution] * batch_size * 0.7  # batch 不完全线性
```

当 `estimated_vram > available_vram_gb * 0.9` 时，在 GUI 中显示警告文本（黄色/橙色），内容示例：

> ⚠ 预计需要 ~7.0 GB 显存，当前可用 6.0 GB，可能导致内存不足

不阻止操作，仅警告。警告显示在右侧设置面板的 batch size / 分辨率下方。

### 3.5 设备信息展示

右侧面板的"设备"行改为显示更多信息：

```
设备: CUDA — NVIDIA RTX 3060 (12.0 GB, 可用 9.2 GB)
设备: MPS — Apple M1 Pro (统一内存 16 GB)
设备: CPU（无 GPU 加速）
```

---

## 4. 单任务 Tab 布局重构

### 4.1 新布局

```
┌──────────────────────────────────────────────────────────────────┐
│  BiRefNet Video Matting Tool                                      │
├──────────────────────┬───────────────────────┬───────────────────┤
│  [单任务]             │  [批量队列 (3)]        │                   │
├──────────────────────┴───────────────────────┴───────────────────┤
│                                                                   │
│  ┌─ 左侧 (stretch=2) ──────────┐  ┌─ 右侧 (stretch=1) ────────┐│
│  │                              │  │                            ││
│  │  输入文件:                    │  │  ── 模型设置 ──            ││
│  │  ┌────────────────┐ [选择▼]  │  │  模型: [BiRefNet-general▼] ││
│  │  │ wedding.mp4    │          │  │  设备: CUDA — RTX 3060     ││
│  │  └────────────────┘          │  │         (12GB, 可用 9.2GB) ││
│  │                              │  │  分辨率: [1024×1024    ▼]  ││
│  │  视频信息:                    │  │  Batch: [4           ▼]   ││
│  │  1920×1080 | 30fps | 2340帧  │  │                            ││
│  │  时长: 01:18 | 码率: 20 Mbps │  │  ── 输出设置 ──            ││
│  │                              │  │  格式: [MOV ProRes 4444▼]  ││
│  │  输出路径:                    │  │  背景: [透明背景       ▼]  ││
│  │  ┌────────────────┐ [浏览]   │  │                            ││
│  │  │ ~/Desktop/     │          │  │  ── 高级参数 ──            ││
│  │  └────────────────┘          │  │  码率: [自动(20Mbps)  ▼]  ││
│  │                              │  │  预设: [medium        ▼]  ││
│  │  [████████████░░░░░░] 67%    │  │                            ││
│  │  推理中 1523/2340 | 8.5 FPS  │  │  ⚠ 显存警告(如有)         ││
│  │  剩余: 01:36                 │  │                            ││
│  │                              │  │                            ││
│  └──────────────────────────────┘  └────────────────────────────┘│
│                                                                   │
├───────────────────────────────────────────────────────────────────┤
│  [开始处理]    [暂停]    [取消]    [加入队列]                       │
└───────────────────────────────────────────────────────────────────┘
```

### 4.2 布局结构

```
MainWindow
└── QVBoxLayout (central)
    ├── QTabWidget
    │   ├── Tab 1 "单任务"
    │   │   └── QHBoxLayout (stretch 2:1)
    │   │       ├── 左侧 QVBoxLayout
    │   │       │   ├── 输入文件区 (文件路径 + 选择按钮)
    │   │       │   ├── 文件信息标签
    │   │       │   ├── 输出路径区 (路径 + 浏览按钮)
    │   │       │   └── 进度区 (进度条 + 状态标签)
    │   │       └── 右侧 QVBoxLayout
    │   │           ├── GroupBox "模型设置"
    │   │           │   ├── 模型下拉框
    │   │           │   ├── 设备信息标签
    │   │           │   ├── 推理分辨率下拉框
    │   │           │   └── Batch size 下拉框
    │   │           ├── GroupBox "输出设置"
    │   │           │   ├── 格式下拉框
    │   │           │   └── 背景模式下拉框
    │   │           ├── GroupBox "高级参数"
    │   │           │   ├── 码率下拉框 (+ 自定义输入框)
    │   │           │   └── 编码预设下拉框
    │   │           └── 显存警告标签 (条件显示)
    │   └── Tab 2 "批量队列" (不变)
    └── QHBoxLayout "操作栏" (固定底部)
        ├── [开始处理]
        ├── [暂停]
        ├── [取消]
        └── [加入队列]
```

### 4.3 操作栏行为

操作栏在 Tab 外部，固定在窗口底部。

- **单任务 Tab 激活时**：显示 [开始处理] [暂停] [取消] [加入队列]
- **队列 Tab 激活时**：操作栏隐藏（队列 Tab 有自己的控制按钮）

通过 `QTabWidget.currentChanged` 信号切换操作栏可见性。

### 4.4 右侧面板联动规则

| 条件 | 联动行为 |
|------|---------|
| 输入为图片/图片文件夹 | 隐藏：码率、编码预设；Batch size 仅文件夹可见 |
| 格式为 PNG/TIFF 序列 | 隐藏：码率、编码预设 |
| 格式为 MOV ProRes | 码率下拉框 → 切换为 ProRes profile 选择 |
| 格式为 H.264/H.265 | 编码预设可见 |
| 格式为 AV1 | 编码预设可见（映射为 cpu-used） |
| 格式为 WebM VP9 | 隐藏编码预设 |
| 不支持透明的格式 | 背景模式排除"透明"选项 |
| 显存不足预估 | 显示黄色警告文本 |

---

## 5. ProcessingConfig 扩展

```python
class BitrateMode(Enum):
    AUTO = "auto"
    LOW = "low"           # ×0.25
    MEDIUM = "medium"     # ×0.5
    HIGH = "high"         # ×1.0
    VERY_HIGH = "very_high"  # ×2.0
    CUSTOM = "custom"

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

class InferenceResolution(Enum):
    RES_512 = 512
    RES_1024 = 1024
    RES_2048 = 2048

@dataclass
class ProcessingConfig:
    model_name: str = "BiRefNet-general"
    output_format: OutputFormat = OutputFormat.MOV_PRORES
    background_mode: BackgroundMode = BackgroundMode.TRANSPARENT
    # 新增字段
    bitrate_mode: BitrateMode = BitrateMode.AUTO
    custom_bitrate_mbps: float = 20.0
    encoding_preset: EncodingPreset = EncodingPreset.MEDIUM
    batch_size: int = 1
    inference_resolution: InferenceResolution = InferenceResolution.RES_1024
```

### 5.1 序列化兼容 (.brm)

QueueTask 序列化时包含完整 ProcessingConfig。新增字段使用默认值，确保旧 .brm 文件加载时向后兼容——缺失的字段用默认值填充。

---

## 6. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/core/config.py` | **修改** | 新增 BitrateMode / EncodingPreset / InferenceResolution 枚举；扩展 ProcessingConfig |
| `src/core/device_info.py` | **新增** | DeviceInfo 数据类 + get_device_info() 显存检测 |
| `src/core/inference.py` | **修改** | predict() 支持 inference_resolution 参数；新增 predict_batch() |
| `src/core/video.py` | **修改** | get_video_info() 返回 bitrate_mbps |
| `src/core/writer.py` | **修改** | FFmpegWriter 支持 bitrate / preset 参数；create_writer() 透传 |
| `src/core/pipeline.py` | **修改** | 推理阶段使用 batch 推理 + inference_resolution |
| `src/core/queue_task.py` | **修改** | 序列化/反序列化兼容新 config 字段 |
| `src/worker/matting_worker.py` | **修改** | 传递完整 config 到 pipeline |
| `src/gui/main_window.py` | **修改** | 布局重构：左右分栏 + 底部操作栏；右侧设置面板 |
| `src/gui/settings_panel.py` | **新增** | 右侧设置面板独立组件：模型/输出/高级参数/显存警告 |

---

## 7. 测试计划

| 模块 | 测试内容 |
|------|---------|
| config.py | 新枚举值、ProcessingConfig 默认值、序列化兼容 |
| device_info.py | CUDA/MPS/CPU 各路径检测（mock torch） |
| inference.py | predict_batch 多帧推理正确性、不同 resolution 输入 |
| video.py | get_video_info 返回 bitrate_mbps |
| writer.py | FFmpegWriter 带 bitrate/preset 参数的命令行构建 |
| pipeline.py | batch 推理端到端、不足一批的尾帧处理 |
| queue_task.py | 旧 .brm 加载兼容（缺失新字段用默认值） |

---

## 8. 实现顺序

1. **config.py 扩展** — 新增枚举和 ProcessingConfig 字段
2. **device_info.py** — 显存检测模块
3. **get_video_info 扩展** — 返回码率
4. **predict_batch + resolution 参数** — 推理模块改造
5. **writer 参数扩展** — FFmpeg 命令支持 bitrate/preset
6. **pipeline batch 推理** — 攒帧 + 批量推理
7. **queue_task 序列化兼容** — .brm 向后兼容
8. **settings_panel.py** — 右侧设置面板组件
9. **main_window 布局重构** — 左右分栏 + 底部操作栏 + 联动规则

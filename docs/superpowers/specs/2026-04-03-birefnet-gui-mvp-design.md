# BiRefNet GUI MVP 设计规格

## 1. 目标

创建 BiRefNet 视频抠图工具的最小可用版本（MVP），跑通"选择视频 → BiRefNet 推理 → 输出 MOV ProRes 4444 透明视频"这条完整主线。

**开发环境**：macOS 开发，Windows 编译部署
**目标平台**：Windows 10/11 + NVIDIA GPU（Mac 开发时用 CPU/MPS）

## 2. MVP 范围

### 2.1 本期实现

| 功能 | 说明 |
|------|------|
| PyQt6 主窗口 | 文件选择、参数显示、开始/暂停/取消按钮、进度条 |
| 单视频文件输入 | 支持 MP4、AVI、MOV、MKV，文件选择对话框 |
| BiRefNet-general 模型 | 本地离线加载，唯一模型 |
| 逐帧推理 | 读取视频帧 → BiRefNet 推理 → 获得 alpha mask |
| MOV ProRes 4444 输出 | 带透明通道，FFmpeg 编码 |
| 进度显示 | 进度条 + 当前帧/总帧数 + 预估剩余时间 |
| 暂停/继续/取消 | 通过 threading.Event 控制工作线程 |
| 设备自动检测 | CUDA > MPS > CPU 自动选择 |

### 2.2 TODO：后续阶段功能

以下功能本期不实现，留作后续开发：

- **TODO: 多模型支持** — 6 种模型切换（lite/HR/matting/HR-matting/dynamic）
- **TODO: 多输出格式** — WebM VP9、MP4 H.264/H.265/AV1、PNG 序列、TIFF 序列
- **TODO: 多输出模式** — 绿幕、蓝幕、黑白蒙版、反转蒙版、原图+蒙版分轨
- **TODO: 图片输入** — 单张图片、图片文件夹、PSD 支持
- **TODO: 拖拽支持** — 文件拖拽到界面
- **TODO: 摄像头实时输入** — 实时抠图预览
- **TODO: 视频预览** — 原图视频播放器（播放/暂停/seek）
- **TODO: 处理范围选择** — 全部/指定时间段/标记帧
- **TODO: ROI 区域选择** — 只处理感兴趣区域
- **TODO: 时序一致性** — 帧间平滑处理，减少闪烁
- **TODO: 批量队列** — 多任务排队处理，独立设置，拖拽添加
- **TODO: 工程文件 (.brm)** — 保存进度，断点续传
- **TODO: 音频保留** — FFmpeg 提取合并原音轨
- **TODO: 高级输出参数** — 码率、CRF、预设、色彩空间、帧率、分辨率调整
- **TODO: 批处理大小** — 多帧并行推理，根据显存自动推荐
- **TODO: 显存检测** — 自动检测显卡，提示模型兼容性
- **TODO: 分辨率自适应** — 根据显存自动选择处理分辨率
- **TODO: 设置持久化** — 用户配置保存（config.json）
- **TODO: 完成后操作** — 关机/休眠/播放提示音
- **TODO: 系统托盘** — 最小化到后台继续处理
- **TODO: ONNX/TensorRT 加速** — 可选加速方案
- **TODO: PyInstaller 打包** — Windows EXE 打包
- **TODO: 安装程序** — MSI/EXE 安装版 + 便携版
- **TODO: 缓存管理** — 临时文件清理策略

## 3. 架构设计

### 3.1 方案：单进程 + QThread 工作线程

GUI 主线程负责界面交互，MattingWorker（QThread）在工作线程中执行推理流水线，通过 Qt Signal/Slot 机制通信。

选择原因：简单直接，MVP 阶段足够，后续可重构为多进程。

### 3.2 项目结构

```
birefnet-gui/
├── main.py                  # 入口点
├── requirements.txt         # 依赖
├── src/
│   ├── __init__.py
│   ├── gui/
│   │   ├── __init__.py
│   │   ├── main_window.py   # 主窗口
│   │   └── widgets.py       # 自定义控件（进度面板等）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── inference.py     # BiRefNet 模型加载与推理
│   │   ├── video.py         # 视频读取/写入（OpenCV + FFmpeg）
│   │   └── pipeline.py      # 处理流水线
│   └── worker/
│       ├── __init__.py
│       └── matting_worker.py # QThread 工作线程
├── models/                   # 模型文件（不入 git）
├── download_models.py        # 模型下载脚本
├── DESIGN.md                 # 完整设计文档（已有）
└── docs/
    └── superpowers/
        └── specs/            # 设计规格
```

### 3.3 模块职责

**gui/main_window.py**
- PyQt6 QMainWindow 子类
- 布局：顶部文件选择区 → 中间模型/输出信息 → 底部进度和控制按钮
- 用户点击"开始处理"后创建 MattingWorker 并启动
- 接收 Worker 的 Signal 更新进度条和状态文字

**core/inference.py**
- `load_model(model_path, device)` — 加载 BiRefNet 模型到指定设备
- `predict(model, frame, device)` — 单帧推理，输入 numpy BGR 图像，输出 numpy alpha mask (0-255)
- 内部处理：resize 到 1024x1024 → 模型推理 → resize 回原尺寸

**core/video.py**
- `get_video_info(path)` — 获取视频元信息（宽高、帧数、帧率、时长）
- `FrameReader(path)` — 迭代器，逐帧读取视频（基于 cv2.VideoCapture）
- `ProResWriter(output_path, width, height, fps)` — FFmpeg subprocess，接收 RGBA 帧写入 MOV ProRes 4444

**core/pipeline.py**
- `MattingPipeline` 类
- `process(input_path, output_path, progress_callback, pause_event, cancel_event)`
- 串联 FrameReader → predict → ProResWriter
- 每帧调用 progress_callback(current_frame, total_frames)
- 每帧检查 pause_event 和 cancel_event

**worker/matting_worker.py**
- `MattingWorker(QThread)` 子类
- Signals: `progress(int, int)`, `finished(str)`, `error(str)`
- run() 方法中创建 pipeline 并执行

### 3.4 数据流

```
用户选择视频文件
  → main_window 验证文件，显示视频信息（分辨率、帧数、时长）
  → 用户点击"开始处理"
  → 创建 MattingWorker(input_path, output_path)
  → Worker.start()
    → 加载模型（首次较慢，后续可缓存）
    → 遍历视频帧:
      ├─ 检查 cancel_event → 如果取消，清理临时文件，emit error
      ├─ 检查 pause_event → 如果暂停，阻塞等待
      ├─ frame = reader.next()
      ├─ alpha = inference.predict(model, frame, device)
      ├─ rgba = 合并 frame + alpha
      ├─ writer.write(rgba)
      └─ emit progress(current, total)
    → writer.close()
    → emit finished(output_path)
  → main_window 显示"处理完成"，提供打开文件/文件夹按钮
```

### 3.5 技术选型

| 组件 | 选择 | 说明 |
|------|------|------|
| GUI | PyQt6 | 跨平台，原生体验 |
| 视频读取 | OpenCV (cv2.VideoCapture) | 简单逐帧读取 |
| 视频写出 | FFmpeg subprocess (ffmpeg-python) | ProRes 4444 编码 |
| 推理框架 | PyTorch + transformers | BiRefNet 原生方式 |
| 设备选择 | CUDA > MPS > CPU | torch 自动检测 |
| 线程通信 | Qt Signal/Slot | PyQt6 原生方式 |
| 暂停/取消 | threading.Event | 简单高效 |

### 3.6 依赖清单

```
PyQt6>=6.5
torch>=2.0
torchvision>=0.15
transformers>=4.30
opencv-python>=4.8
ffmpeg-python>=0.2
Pillow>=10.0
numpy>=1.24
```

开发环境额外需要：FFmpeg 可执行文件（系统安装或内置）

## 4. GUI 设计

### 4.1 主窗口布局（MVP 简化版）

```
┌─────────────────────────────────────────────────────────┐
│  BiRefNet Video Matting Tool                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入文件:                                              │
│  ┌─────────────────────────────────────┐  [选择文件]    │
│  │ (未选择)                            │               │
│  └─────────────────────────────────────┘               │
│                                                         │
│  视频信息: 1920x1080 | 30fps | 2340帧 | 01:18          │
│                                                         │
│  ────────────────────────────────────────────────       │
│                                                         │
│  模型: BiRefNet-general                                 │
│  设备: CUDA (RTX 3060) / MPS / CPU                     │
│  输出: MOV ProRes 4444 (透明)                           │
│                                                         │
│  输出路径:                                              │
│  ┌─────────────────────────────────────┐  [浏览...]    │
│  │ (与输入文件同目录)                   │               │
│  └─────────────────────────────────────┘               │
│                                                         │
│  ────────────────────────────────────────────────       │
│                                                         │
│  [████████████████░░░░░░░░░░] 67%                       │
│  帧: 1523/2340 | 速度: 8.5 FPS | 剩余: 01:36          │
│                                                         │
│       [开始处理]    [暂停]    [取消]                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 4.2 状态流转

```
初始状态 → 用户选择文件 → 就绪状态（开始按钮可用）
  → 点击开始 → 处理中（进度更新，暂停/取消可用）
    → 点击暂停 → 暂停中（继续/取消可用）
    → 点击取消 → 初始状态
    → 处理完成 → 完成状态（显示输出路径，可打开）
    → 处理出错 → 错误状态（显示错误信息）
```

## 5. 文件命名规则

输出文件名：`{原文件名}_birefnet-general_{毫秒时间戳}.mov`

示例：`wedding_video_birefnet-general_1743675123456.mov`

默认输出到与输入文件同目录，用户可通过"浏览"按钮更改。

## 6. 错误处理（MVP）

| 场景 | 处理方式 |
|------|---------|
| 文件不存在/无法读取 | 弹窗提示，回到初始状态 |
| 模型文件缺失 | 弹窗提示下载模型 |
| FFmpeg 未安装 | 弹窗提示安装 FFmpeg |
| 推理过程中 OOM | 捕获异常，弹窗提示显存不足 |
| 磁盘空间不足 | 捕获写入异常，提示清理空间 |
| 用户取消 | 清理已写入的临时输出文件 |

# BiRefNet GUI 开发进度

**更新日期**: 2026-04-05
**当前版本**: P5 完成 (部署发布: PyInstaller + GitHub Actions CI)
**分支**: `feature/p5-deployment`

---

## 一期完成内容 (MVP)

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| 项目脚手架 | 2.1 技术栈 | ✅ 完成 | PyQt6 + PyTorch + OpenCV + FFmpeg |
| 视频文件输入 | 3.1.1 输入模块 | ✅ 完成 | 支持 MP4/AVI/MOV/MKV，文件选择对话框 |
| 图片文件输入 | 3.1.1 输入模块 | ✅ 完成 | 单张图片、图片文件夹批量处理 |
| 拖拽支持 | 3.1.1 拖拽支持 | ✅ 完成 | 文件/文件夹拖拽到界面自动识别 |
| BiRefNet 推理 | 3.1.2 处理模块 / 4.1 模型选择 | ✅ 部分 | 仅 BiRefNet-general，逐帧推理 |
| MOV ProRes 4444 输出 | 3.1.3 输出模块 | ✅ 完成 | 支持音频保留 |
| 设备自动检测 | 3.1.2 显存检测 | ✅ 部分 | CUDA > MPS > CPU 自动选择 |
| PyQt6 主界面 | 3.2 界面设计 | ✅ 简化版 | 文件选择、进度条、开始/暂停/取消 |
| 进度显示 | 3.2 界面设计 | ✅ 完成 | 进度条 + 帧数 + FPS + 剩余时间 |
| 暂停/继续/取消 | 3.2 界面设计 | ✅ 完成 | threading.Event 控制 |
| 输出路径选择 | 3.2 界面设计 | ✅ 完成 | 默认同目录，可自选 |
| 文件命名规则 | 3.1.4 命名规则 | ✅ 完成 | `{filename}_{model}_{毫秒时间戳}.mov` |
| 模型离线加载 | 4.2 模型管理 / 4.3 加载方式 | ✅ 完成 | `local_files_only=True` |
| 模型下载脚本 | 4.4 模型下载脚本 | ✅ 简化版 | 仅下载 birefnet-general |
| 多模型支持 | 4.1 模型选择 | 6 种模型切换（lite/HR/matting/HR-matting/dynamic），界面下拉选择 |
| 多输出格式 | 3.1.3 输出格式 | WebM VP9, MP4 H.264/H.265/AV1, PNG 序列, TIFF 序列 |
| 多输出模式 | 3.1.3 输出模式 | 绿幕、蓝幕、黑白蒙版、反转蒙版、原图+蒙版分轨 |

### 项目结构

```
birefnet-gui/
├── main.py                      # 入口点
├── requirements.txt             # 依赖清单
├── download_models.py           # 模型下载脚本（开发用）
├── src/
│   ├── core/
│   │   ├── config.py            # 枚举 + ProcessingConfig 数据类
│   │   ├── inference.py         # 模型加载 + 单帧/批量推理 + 可变分辨率
│   │   ├── video.py             # FrameReader + ProResWriter + get_video_info(含码率)
│   │   ├── writer.py            # FFmpegWriter + ImageSequenceWriter + create_writer 工厂
│   │   ├── compositing.py       # 背景合成（透明/绿幕/蒙版等）
│   │   ├── pipeline.py          # 两阶段处理流水线（batch推理→编码）
│   │   ├── image_pipeline.py    # 图片/图片文件夹处理流水线
│   │   ├── cache.py             # MaskCacheManager: mask 缓存管理
│   │   ├── device_info.py       # GPU/显存检测 + VRAM 预估
│   │   ├── queue_task.py        # QueueTask 数据模型 + 枚举
│   │   └── queue_manager.py     # QueueManager: 队列管理 + .brm 持久化
│   ├── worker/
│   │   └── matting_worker.py    # QThread 工作线程（支持两阶段+续传）
│   └── gui/
│       ├── main_window.py       # PyQt6 主窗口（Tab 布局 + 底部操作栏）
│       ├── settings_panel.py    # 右侧设置面板（模型/输出/高级参数/显存警告）
│       └── queue_tab.py         # 批量队列 Tab
├── tests/                       # 125 个测试
├── models/birefnet-general/     # 模型文件（git-ignored）
├── DESIGN.md                    # 完整设计文档
└── docs/superpowers/
    ├── specs/                   # 设计规格
    └── plans/                   # 实现计划
```

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_video.py | 6 | get_video_info, FrameReader, ProResWriter |
| test_inference.py | 5 | detect_device, load_model, predict |
| test_pipeline.py | 2 | 端到端处理 + 取消中断 |
| **合计** | **13** | **全部通过** |

### 开发过程中的修复

| 问题 | 修复 | Commit |
|------|------|--------|
| ProResWriter 无 context manager，可能泄漏资源 | 添加 `__enter__`/`__exit__` | `6b036f2` |
| `write_frame` 用 assert 验证（-O 模式会跳过） | 改为 raise ValueError | `6b036f2` |
| BiRefNet 依赖缺失（einops, kornia, timm） | 补充到 requirements.txt | `2d4f901` |
| MPS 设备 float16/float32 类型不匹配 | 加载模型后调用 model.float() | `2d4f901` |

---

## 二期完成内容 (P2 批量队列)

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| Tab 切换 | 3.1.5 批量队列 | ✅ 完成 | QTabWidget: 单任务 Tab + 批量队列 Tab |
| 加入队列 | 3.1.5 批量队列 | ✅ 完成 | Tab 1 新增「加入队列」按钮 |
| 队列任务列表 | 3.1.5 批量队列 | ✅ 完成 | QTableWidget 显示文件名/模型/格式/状态 |
| 队列执行 | 3.1.5 批量队列 | ✅ 完成 | 开始/暂停/取消当前/清空队列 |
| 两阶段 Pipeline | 3.1.6 断点续传 | ✅ 完成 | 推理阶段缓存 mask + 编码阶段合成 |
| Mask 缓存 | 3.1.6 断点续传 | ✅ 完成 | grayscale PNG 缓存 + metadata 校验 |
| 断点续传 | 3.1.6 断点续传 | ✅ 完成 | 暂停/关闭后从上次进度继续 |
| 工程文件 (.brm) | 3.1.6 工程文件 | ✅ 完成 | JSON 格式持久化到 ~/.birefnet-gui/queue.brm |
| 自动恢复队列 | 3.1.6 工程文件 | ✅ 完成 | 启动时自动加载队列 |
| 拖拽排序/右键菜单 | 3.1.5 批量队列 | ✅ 完成 | 删除/移到顶部/移到底部 |
| 外部拖拽 | 3.1.5 批量队列 | ✅ 完成 | 拖入文件到队列 Tab 自动添加 |
| 互斥执行 | 3.1.5 批量队列 | ✅ 完成 | 单任务与队列不能同时运行 |
| 完成提示音 | 3.1.5 完成后操作 | ✅ 完成 | QApplication.beep() |
| 两阶段进度 | 3.2 界面设计 | ✅ 完成 | 推理/编码分别显示独立进度 |

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/queue_task.py` | QueueTask 数据类、TaskStatus/ProcessingPhase 枚举 |
| `src/core/cache.py` | MaskCacheManager: mask 保存/读取/校验/清理 |
| `src/core/queue_manager.py` | QueueManager: 队列任务管理 + .brm 持久化 |
| `src/gui/queue_tab.py` | 队列 Tab: 任务列表 + 进度 + 控制按钮 |
| `tests/test_queue_task.py` | QueueTask 序列化测试 (4个) |
| `tests/test_cache.py` | MaskCacheManager 测试 (10个) |
| `tests/test_queue_manager.py` | QueueManager 测试 (12个) |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/core/pipeline.py` | 拆分为 infer_phase + encode_phase 两阶段 |
| `src/worker/matting_worker.py` | 支持 task_id/start_frame/cleanup_cache, 3参数进度信号 |
| `src/gui/main_window.py` | QTabWidget + 加入队列 + 互斥控制 + closeEvent 保存 |
| `tests/test_pipeline.py` | 新增 4 个两阶段 pipeline 测试 |

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_queue_task.py | 4 | QueueTask 创建/序列化 |
| test_cache.py | 10 | MaskCacheManager 全功能 |
| test_queue_manager.py | 12 | 任务管理 + .brm 持久化 |
| test_pipeline.py | 9 | 端到端 + 两阶段 + 断点续传 |
| test_video.py | 6 | 视频 I/O |
| test_inference.py | 7 | 推理模块 |
| test_writer.py | 12 | FFmpegWriter + ImageSequenceWriter |
| test_config.py | 13 | 配置模块 |
| test_compositing.py | 12 | 合成模块 |
| test_audio.py | 7 | 音频处理 |
| test_image_pipeline.py | 5 | 图片处理流水线 |
| **合计** | **97** | **全部通过** |

---

## 三期完成内容 (P3 高级参数 + 显存检测 + 布局重构)

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| 码率控制 | 3.1.3 高级参数 | ✅ 完成 | 自动/低/中/高/极高/自定义(Mbps)，基于原视频码率动态计算 |
| ProRes Profile | 3.1.3 高级参数 | ✅ 完成 | Proxy/LT/Standard/HQ 级别选择 |
| 编码预设 | 3.1.3 高级参数 | ✅ 完成 | ultrafast~veryslow，H.264/H.265/AV1 适用 |
| Batch 推理 | 3.1.3 高级参数 | ✅ 完成 | 可配置 batch size (1/2/4/8/16)，根据显存自动推荐 |
| 推理分辨率 | 3.1.2 分辨率自适应 | ✅ 完成 | 512/1024/2048 可选，支持 BiRefNet 2048 高质量模式 |
| 显存检测 | 3.1.2 显存检测 | ✅ 完成 | CUDA/MPS/CPU 自动检测，显示设备名称和显存信息 |
| OOM 预警 | 3.1.2 显存检测 | ✅ 完成 | 预估显存占用，超过可用 90% 时显示黄色警告 |
| 布局重构 | 3.2 界面设计 | ✅ 完成 | 单任务 Tab: 左右分栏 + 底部固定操作栏 |
| SettingsPanel | 3.2 界面设计 | ✅ 完成 | 独立右侧设置面板：模型/输出/高级参数/显存警告 |
| 格式联动 | 3.2 界面设计 | ✅ 完成 | 图片模式隐藏码率/预设；PNG/TIFF 隐藏高级参数；格式切换更新可选项 |
| .brm 兼容 | 3.1.6 工程文件 | ✅ 完成 | 新字段向后兼容，旧 .brm 文件加载使用默认值 |

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/device_info.py` | DeviceInfo 数据类 + GPU/显存检测 + VRAM 预估 |
| `src/gui/settings_panel.py` | 右侧设置面板：模型/输出/高级参数/显存警告 |
| `tests/test_device_info.py` | 设备检测测试 (8个) |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/core/config.py` | 新增 BitrateMode/EncodingPreset/InferenceResolution 枚举；扩展 ProcessingConfig |
| `src/core/inference.py` | predict() 支持 resolution 参数；新增 predict_batch() 批量推理 |
| `src/core/video.py` | get_video_info() 返回 bitrate_mbps；ProResWriter 支持 profile 参数 |
| `src/core/writer.py` | FFmpegWriter 支持 bitrate_kbps/preset；create_writer 解析高级参数 |
| `src/core/pipeline.py` | infer_phase 使用 batch 推理；encode_phase 传递 source_bitrate |
| `src/core/queue_task.py` | 序列化/反序列化新 config 字段，向后兼容 |
| `src/core/image_pipeline.py` | 使用 inference_resolution 参数 |
| `src/gui/main_window.py` | 布局重构：左右分栏 + 底部操作栏 + SettingsPanel 集成 |

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_config.py | 20 | 配置模块（含新枚举） |
| test_device_info.py | 8 | GPU/VRAM 检测 |
| test_inference.py | 11 | 推理模块（含 batch + resolution） |
| test_pipeline.py | 10 | 端到端 + 两阶段 + batch |
| test_video.py | 7 | 视频 I/O（含 bitrate） |
| test_writer.py | 17 | Writer（含 bitrate/preset/profile） |
| test_queue_task.py | 6 | 队列任务序列化（含向后兼容） |
| test_cache.py | 10 | MaskCacheManager |
| test_queue_manager.py | 12 | 队列管理 |
| test_compositing.py | 12 | 合成模块 |
| test_audio.py | 7 | 音频处理 |
| test_image_pipeline.py | 5 | 图片处理 |
| **合计** | **125** | **全部通过** |

---

## 四期完成内容 (P4 FP16 + 多模型 + 模型管理 + 时序修复)

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| FP16 自动加速 | 2.1 加速方案 | ✅ 完成 | CUDA 自动开启半精度推理，MPS/CPU 保持 FP32 |
| 多模型支持 | 4.1 模型选择 | ✅ 完成 | 6 种模型全部支持加载和切换 |
| 模型注册表 | 4.2 模型管理 | ✅ 完成 | ModelInfo + MODEL_REGISTRY 统一管理元数据 |
| 模型管理 Tab | 4.2 模型管理 | ✅ 完成 | 模型卡片列表 + 下载/删除 + 进度显示 |
| 模型下载 | 4.4 模型下载 | ✅ 完成 | hf-mirror 镜像优先，失败回退 HuggingFace 官方 |
| 首次启动检测 | 4.2 模型管理 | ✅ 完成 | 无模型时自动切到模型管理 Tab |
| 时序修复 | 3.1.2 时序一致性 | ✅ 完成 | 异常帧检测 + 邻帧替换，消除 mask 闪现 |
| 三阶段 Pipeline | 3.1.2 时序一致性 | ✅ 完成 | 推理 → 时序修复 → 编码 |
| .brm 兼容 | 3.1.6 工程文件 | ✅ 完成 | temporal_fix 字段向后兼容 |

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/temporal.py` | 异常帧检测 + 邻帧替换算法 |
| `src/core/model_downloader.py` | 模型下载器（hf-mirror 优先 + 断点续传） |
| `src/gui/model_tab.py` | 模型管理 Tab：卡片列表 + 下载/删除 + 进度 |
| `tests/test_temporal.py` | 时序修复测试 (8个) |
| `tests/test_model_downloader.py` | 模型下载器测试 (7个) |

### 修改文件

| 文件 | 说明 |
|------|------|
| `src/core/config.py` | ModelInfo + MODEL_REGISTRY + temporal_fix 字段 |
| `src/core/inference.py` | FP16 autocast (CUDA) |
| `src/core/pipeline.py` | 三阶段流水线（推理 → 时序修复 → 编码） |
| `src/core/queue_task.py` | ProcessingPhase.TEMPORAL_FIX + temporal_fix 序列化 |
| `src/gui/settings_panel.py` | 仅显示已安装模型 + 管理模型链接 + 时序修复开关 |
| `src/gui/main_window.py` | 模型管理 Tab 集成 + 首次启动检测 + 三阶段进度 |
| `src/gui/queue_tab.py` | 三阶段进度显示 + 时序修复阶段暂停禁用 |
| `download_models.py` | 使用 MODEL_REGISTRY |

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_config.py | 27 | 配置模块（含 ModelInfo/MODEL_REGISTRY/temporal_fix） |
| test_device_info.py | 8 | GPU/VRAM 检测 |
| test_inference.py | 14 | 推理模块（含 FP16 autocast） |
| test_temporal.py | 8 | 时序修复算法 |
| test_pipeline.py | 12 | 端到端 + 三阶段 + temporal_fix 开关 |
| test_model_downloader.py | 7 | 模型下载器 |
| test_video.py | 7 | 视频 I/O |
| test_writer.py | 17 | Writer |
| test_queue_task.py | 8 | 队列任务序列化（含 temporal_fix 兼容） |
| test_cache.py | 10 | MaskCacheManager |
| test_queue_manager.py | 12 | 队列管理 |
| test_compositing.py | 12 | 合成模块 |
| test_audio.py | 7 | 音频处理 |
| test_image_pipeline.py | 5 | 图片处理 |
| **合计** | **154** | **全部通过** |

---

## 五期完成内容 (P5 部署发布)

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| 路径解析 | 5.1 打包策略 | ✅ 完成 | paths.py: frozen-safe 路径解析，支持 PyInstaller |
| FFmpeg 检测 | 5.2 环境依赖 | ✅ 完成 | 启动时检测，缺失则弹窗提示安装方法 |
| PyInstaller 配置 | 5.1 打包策略 | ✅ 完成 | --onedir 便携版，排除不需要的模块 |
| GitHub Actions CI | 5.3 安装包类型 | ✅ 完成 | Windows (CUDA) + macOS (ARM) 双平台自动构建 |
| 模型不内置 | 4.2 模型管理 | ✅ 设计决策 | 用户通过 GUI 下载，包体积 ~500-700MB |

### 新增文件

| 文件 | 说明 |
|------|------|
| `src/core/paths.py` | 统一路径解析（开发/打包模式） |
| `tests/test_paths.py` | 路径解析测试 (5个) |
| `tests/test_main.py` | FFmpeg 检测测试 (4个) |
| `birefnet-gui.spec` | PyInstaller 打包配置 |
| `.github/workflows/build.yml` | GitHub Actions 双平台 CI |

### 修改文件

| 文件 | 说明 |
|------|------|
| `main.py` | FFmpeg 启动检测 + freeze_support |
| `src/gui/main_window.py` | MODELS_DIR 改用 paths 模块 |
| `src/gui/queue_tab.py` | models_dir 改用 paths 模块 |

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_paths.py | 5 | 路径解析 |
| test_main.py | 4 | FFmpeg 检测 |
| (其余测试不变) | 156 | — |
| **合计** | **165** | **全部通过** |

---

## 未完成功能 (后续开发)

以下按 DESIGN.md 章节逐项列出所有待开发功能，分为建议优先级。

### 已砍掉的功能

| 功能 | 原因 |
|------|------|
| 设置持久化 | 价值不高，以后设置项多了再考虑 |
| 视频预览 | 增加复杂度，用系统播放器/剪辑软件查看 |
| 处理范围选择 | 应由专业剪辑软件处理 |
| 菜单栏 | 目前没有必须放菜单里的功能 |
| 完成后操作 | 关机/休眠，价值不高 |

### 建议下一期开发

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| 硬件编码加速 | 3.1.3 高级参数 | NVENC/VideoToolbox 硬件编码器，编码阶段提速（推理是主要瓶颈，优先级低） |
| 缓存管理 GUI | 3.1.6 清理策略 | 临时文件查看大小/手动清理按钮（代码层面已有 cleanup 方法） |

### 后续考虑

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| ROI 区域选择 | 3.1.2 ROI | 只处理感兴趣区域 |
| 摄像头实时输入 | 3.1.1 摄像头实时输入 | 实时抠图预览 |
| 系统托盘 | 3.2 PyQt6 特性 | 最小化到后台继续处理 |

### 部署发布

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| ~~PyInstaller 打包~~ | 5.1 打包策略 | ✅ P5 已完成 |
| ~~GitHub Actions CI~~ | 5.3 安装包类型 | ✅ P5 已完成 |
| ONNX/TensorRT 加速 | 2.1 加速方案 | 可选推理加速 |
| 安装版 (MSI/EXE) | 5.3 安装包类型 | 目前仅便携版 |
| 全模型打包 | 5.1 打包策略 | 目前不内置模型，用户通过 GUI 下载 |

---

## 快速启动

```bash
# 进入项目
cd ~/birefnet-gui
source venv/bin/activate

# 下载模型（首次需要，约 424MB）
python download_models.py

# 运行测试
python -m pytest tests/ -v

# 启动 GUI
python main.py
```

---

## Git 提交历史

```
2d4f901 fix: add missing BiRefNet deps and fix MPS float16 mismatch
6eaf196 feat: PyQt6 main window with file selection, progress, and controls
49a086e feat: MattingWorker QThread — bridges pipeline to GUI via signals
d89a64c feat: matting pipeline — orchestrates read, infer, write with pause/cancel
919f6cd feat: inference module — BiRefNet model loading and prediction
6b036f2 fix: ProResWriter context manager + explicit ValueError for shape validation
5f7bd6b feat: video I/O module — FrameReader, ProResWriter, get_video_info
dd8815f feat: project scaffolding with dependencies and entry point
4bd8030 docs: add MVP implementation plan
511014b docs: add MVP design spec for BiRefNet GUI
```

---

## 已确认的技术决策

| 决策 | 选择 | 来源 |
|------|------|------|
| GUI 框架 | PyQt6 | DESIGN.md 9 |
| 推理后端 | PyTorch（不用 ONNX） | DESIGN.md 9 |
| 视频编码 | FFmpeg-python | DESIGN.md 9 |
| 打包工具 | PyInstaller | DESIGN.md 9 |
| 架构模式 | 单进程 + QThread | MVP 设计 |
| 开发环境 | macOS 开发，Windows 部署 | 用户确认 |
| MPS float 处理 | model.float() 强制 float32 | 开发中发现并修复 |

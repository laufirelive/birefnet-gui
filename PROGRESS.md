# BiRefNet GUI 开发进度

**更新日期**: 2026-04-04
**当前版本**: P2 完成 (MVP + 音频/图片/拖拽 + 批量队列)
**分支**: `master`

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
│   │   ├── inference.py         # 模型加载 + 单帧推理
│   │   ├── video.py             # FrameReader + ProResWriter + get_video_info
│   │   ├── pipeline.py          # 两阶段处理流水线（推理→编码）
│   │   ├── cache.py             # MaskCacheManager: mask 缓存管理
│   │   ├── queue_task.py        # QueueTask 数据模型 + 枚举
│   │   └── queue_manager.py     # QueueManager: 队列管理 + .brm 持久化
│   ├── worker/
│   │   └── matting_worker.py    # QThread 工作线程（支持两阶段+续传）
│   └── gui/
│       ├── main_window.py       # PyQt6 主窗口（Tab 布局）
│       └── queue_tab.py         # 批量队列 Tab
├── tests/
│   ├── conftest.py              # 测试 fixtures
│   ├── test_video.py            # 视频 I/O 测试 (6个)
│   ├── test_inference.py        # 推理测试 (5个)
│   ├── test_pipeline.py         # 两阶段流水线测试 (9个)
│   ├── test_writer.py           # 输出格式测试 (12个)
│   ├── test_queue_task.py       # QueueTask 测试 (4个)
│   ├── test_cache.py            # MaskCacheManager 测试 (10个)
│   └── test_queue_manager.py    # QueueManager 测试 (12个)
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

## 未完成功能 (后续开发)

以下按 DESIGN.md 章节逐项列出所有待开发功能，分为建议优先级。

### P1: 建议下一期开发

这些功能是用户体验的关键补充，实现难度适中。

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| 音频保留 | 3.1.2 音频保留 | FFmpeg 提取原音轨合并到输出 |
| 图片输入 | 3.1.4 图像处理选项 | 单张图片、图片文件夹批量处理 |
| 拖拽支持 | 3.1.1 拖拽支持 | 文件拖拽到界面自动识别 |
| 设置持久化 | Phase 2 | 用户配置保存到 config.json |

### P2: 建议第三期开发

这些功能提升处理效率和专业度。

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| ~~批量队列~~ | ~~3.1.5 批量队列功能~~ | ~~已完成 (P2)~~ |
| 视频预览 | 3.1.2 / 3.2 界面设计 | 原图视频播放器（播放/暂停/seek） |
| 处理范围选择 | 3.1.2 处理范围选择 | 全部/指定时间段/标记帧 |
| 高级输出参数 | 3.1.3 高级参数 | 码率、CRF、预设、色彩空间、帧率、分辨率 |
| 显存检测与提示 | 3.1.2 显存检测 | 自动检测显卡，提示模型兼容性 |
| 分辨率自适应 | 3.1.2 分辨率自适应 | 根据显存自动选择处理分辨率 |
| 批处理大小 | 3.1.3 高级参数 | 多帧并行推理，根据显存自动推荐 |
| 菜单栏 | 3.2 界面设计 | [文件][视图][工具][设置][帮助] |

### P3: 建议第四期开发

这些功能面向生产环境的稳定性和完整性。

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| 工程文件 (.brm) | 3.1.6 工程文件与断点续传 | 保存/加载进度，断点续传 |
| 时序一致性 | 3.1.2 时序一致性 | 帧间平滑处理，减少闪烁 |
| ROI 区域选择 | 3.1.2 ROI | 只处理感兴趣区域 |
| 摄像头实时输入 | 3.1.1 摄像头实时输入 | 实时抠图预览 |
| 完成后操作 | 3.1.5 完成后操作 | 关机/休眠/播放提示音 |
| 系统托盘 | 3.2 PyQt6 特性 | 最小化到后台继续处理 |
| 缓存管理 | 3.1.6 清理策略 | 临时文件自动/手动清理 |

### P4: 部署发布

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| ONNX/TensorRT 加速 | 2.1 加速方案 | 可选推理加速 |
| PyInstaller 打包 | 5.1 打包策略 | Windows EXE 生成 |
| 安装程序 | 5.3 安装包类型 | MSI/EXE 安装版 + 便携版 |
| 全模型打包 | 5.1 打包策略 | 6 个模型全部内置 (~3GB) |

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

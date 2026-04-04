# BiRefNet GUI 开发进度

**更新日期**: 2026-04-04
**当前版本**: P1 核心功能开发中
**分支**: `master`

---

## MVP 完成内容

### 已实现功能

| 模块 | DESIGN.md 对应章节 | 完成情况 | 说明 |
|------|-------------------|---------|------|
| 项目脚手架 | 2.1 技术栈 | ✅ 完成 | PyQt6 + PyTorch + OpenCV + FFmpeg |
| 视频文件输入 | 3.1.1 输入模块 | ✅ 部分 | 支持 MP4/AVI/MOV/MKV，文件选择对话框 |
| BiRefNet 推理 | 3.1.2 处理模块 / 4.1 模型选择 | ✅ 部分 | 仅 BiRefNet-general，逐帧推理 |
| MOV ProRes 4444 输出 | 3.1.3 输出模块 | ✅ 部分 | 仅透明背景模式，单一编码格式 |
| 设备自动检测 | 3.1.2 显存检测 | ✅ 部分 | CUDA > MPS > CPU 自动选择 |
| PyQt6 主界面 | 3.2 界面设计 | ✅ 简化版 | 文件选择、进度条、开始/暂停/取消 |
| 进度显示 | 3.2 界面设计 | ✅ 完成 | 进度条 + 帧数 + FPS + 剩余时间 |
| 暂停/继续/取消 | 3.2 界面设计 | ✅ 完成 | threading.Event 控制 |
| 输出路径选择 | 3.2 界面设计 | ✅ 完成 | 默认同目录，可自选 |
| 文件命名规则 | 3.1.4 命名规则 | ✅ 完成 | `{filename}_{model}_{毫秒时间戳}.{ext}` |
| 模型离线加载 | 4.2 模型管理 / 4.3 加载方式 | ✅ 完成 | `local_files_only=True` |
| 模型下载脚本 | 4.4 模型下载脚本 | ✅ 简化版 | 仅下载 birefnet-general |

---

## P1 核心功能（进行中）

### 已完成

| 功能 | DESIGN.md 章节 | 完成情况 | 说明 |
|------|---------------|---------|------|
| 多模型支持 | 4.1 模型选择 | ✅ 完成 | 6 种模型切换，GUI 下拉选择，未下载模型灰显 |
| 多输出格式 | 3.1.3 输出格式 | ✅ 完成 | MOV ProRes / WebM VP9 / MP4 H.264/H.265/AV1 / PNG序列 / TIFF序列 |
| 多输出模式 | 3.1.3 输出模式 | ✅ 完成 | 透明/绿幕/蓝幕/黑白蒙版/反转蒙版/原图+蒙版分轨 |
| 左右分栏 GUI | 3.2 界面设计 | ✅ 完成 | 左侧文件+进度，右侧模型/格式/模式设置面板 |
| 格式-模式联动 | — | ✅ 完成 | MP4 格式自动禁用"透明"模式 |
| 模型下载脚本扩展 | 4.4 模型下载脚本 | ✅ 完成 | 支持全部 6 个模型，CLI 参数选择 |
| ProcessingConfig | — | ✅ 完成 | 配置对象 + Writer 工厂架构 |

### 未完成（P1 剩余）

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| 音频保留 | 3.1.2 音频保留 | FFmpeg 提取原音轨合并到输出 |
| 图片输入 | 3.1.4 图像处理选项 | 单张图片、图片文件夹批量处理 |
| 拖拽支持 | 3.1.1 拖拽支持 | 文件拖拽到界面自动识别 |
| 设置持久化 | Phase 2 | 用户配置保存到 config.json |

### 开发过程中的修复

| 问题 | 修复 | Commit |
|------|------|--------|
| ProResWriter 无 context manager | 添加 `__enter__`/`__exit__` | `6b036f2` |
| `write_frame` 用 assert 验证 | 改为 raise ValueError | `6b036f2` |
| BiRefNet 依赖缺失 | 补充 einops/kornia/timm | `2d4f901` |
| MPS float16/float32 不匹配 | model.float() 强制 float32 | `2d4f901` |
| WebM VP9 透明视频桌面播放器不可见 | alpha 编码正确但需浏览器播放，GUI 增加提示 | `71e520a` |

---

## P1 新增项目结构

```
birefnet-gui/
├── main.py                      # 入口点
├── requirements.txt             # 依赖清单
├── download_models.py           # 模型下载脚本（支持 6 个模型）
├── src/
│   ├── core/
│   │   ├── config.py            # [新] ProcessingConfig + OutputFormat/BackgroundMode 枚举
│   │   ├── compositing.py       # [新] compose_frame() 背景合成
│   │   ├── writer.py            # [新] FFmpegWriter + ImageSequenceWriter + create_writer 工厂
│   │   ├── inference.py         # 模型加载 + 推理 + get_model_path
│   │   ├── video.py             # FrameReader + ProResWriter + get_video_info
│   │   └── pipeline.py          # 处理流水线（接收 ProcessingConfig）
│   ├── worker/
│   │   └── matting_worker.py    # QThread 工作线程（接收 ProcessingConfig）
│   └── gui/
│       └── main_window.py       # PyQt6 左右分栏主窗口
├── tests/
│   ├── conftest.py              # 测试 fixtures
│   ├── test_config.py           # [新] 配置测试 (8个)
│   ├── test_compositing.py      # [新] 背景合成测试 (12个)
│   ├── test_writer.py           # [新] Writer 工厂测试 (12个)
│   ├── test_video.py            # 视频 I/O 测试 (6个)
│   ├── test_inference.py        # 推理测试 (7个)
│   └── test_pipeline.py         # 端到端流水线测试 (5个)
├── models/                      # 模型文件（git-ignored）
├── DESIGN.md                    # 完整设计文档
└── docs/superpowers/
    ├── specs/                   # 设计规格
    └── plans/                   # 实现计划
```

### 测试覆盖

| 测试文件 | 测试数 | 覆盖模块 |
|----------|--------|---------|
| test_config.py | 8 | OutputFormat, BackgroundMode, MODELS, ProcessingConfig |
| test_compositing.py | 12 | compose_frame 全部 6 种背景模式 |
| test_writer.py | 12 | FFmpegWriter, ImageSequenceWriter, create_writer |
| test_video.py | 6 | get_video_info, FrameReader, ProResWriter |
| test_inference.py | 7 | detect_device, load_model, predict, get_model_path |
| test_pipeline.py | 5 | 端到端: ProRes透明/H.264绿幕/蒙版/PNG序列/取消中断 |
| **合计** | **50** | **全部通过** |

---

## 后续开发计划

### P2: 建议第三期开发

这些功能提升处理效率和专业度。

| 功能 | DESIGN.md 章节 | 说明 |
|------|---------------|------|
| 批量队列 | 3.1.5 批量队列功能 | 多任务排队，独立设置，拖拽添加 |
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
cd ~/birefnet-gui
source venv/bin/activate

# 下载模型（首次需要）
python download_models.py              # 默认下载 general (~424MB)
python download_models.py --all        # 下载全部 6 个模型

# 运行测试
python -m pytest tests/ -v

# 启动 GUI
python main.py
```

---

## Git 提交历史

```
71e520a fix: restore WebM VP9 alpha support, add UI hint for browser-only playback
cf1f4b1 fix: remove WebM VP9 from supports_alpha — VP9 alpha encoding unreliable
8aaf7fd feat: P1 core features — multi-model, multi-format, multi-mode
dcd1c69 Merge MVP implementation
321b45f fix: ProResWriter flush stdin before close to finalize MOV moov atom
c7eb885 docs: add development progress document
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
| 视频编码 | FFmpeg subprocess | DESIGN.md 9 |
| 打包工具 | PyInstaller | DESIGN.md 9 |
| 架构模式 | 单进程 + QThread | MVP 设计 |
| 开发环境 | macOS 开发，Windows 部署 | 用户确认 |
| MPS float 处理 | model.float() 强制 float32 | 开发中发现并修复 |
| P1 架构 | ProcessingConfig + Writer 工厂 | P1 设计 |
| WebM VP9 透明 | 编码正确，仅浏览器可渲染 | 排查确认 |

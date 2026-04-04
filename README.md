# BiRefNet GUI — 视频抠图工具

基于 [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) 模型的桌面视频/图片抠图工具，开箱即用。

## 功能亮点

- **6 种模型** — 通用、轻量、高分辨率、精细抠图，按需选择
- **多种输出格式** — MOV ProRes 4444（透明）、WebM VP9、MP4 H.264/H.265/AV1、PNG/TIFF 序列
- **6 种背景模式** — 透明、绿幕、蓝幕、黑白蒙版、反转蒙版、原图+蒙版分轨
- **批量队列** — 多任务排队处理，支持断点续传
- **FP16 加速** — CUDA 设备自动开启半精度推理
- **时序修复** — 自动检测并修复 mask 闪烁
- **模型管理** — 内置下载器，支持 hf-mirror 镜像加速
- **显存检测** — 自动推荐 batch size 和分辨率，OOM 预警

<!-- 截图预留位 -->
<!-- ![主界面截图](docs/screenshots/main.png) -->

## 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10/11 或 macOS (Apple Silicon) |
| FFmpeg | 需自行安装（见下方说明） |
| 显卡 | 推荐 NVIDIA 4GB+ 显存；Apple Silicon 通过 MPS 加速；无显卡可用 CPU（较慢） |
| 内存 | 8GB 以上 |

## 快速开始

### 1. 下载

从 [Releases](https://github.com/laufirelive/birefnet-gui/releases) 下载对应平台的 zip，解压即可。

### 2. 安装 FFmpeg

**Windows:**
```bash
winget install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

### 3. 运行

双击 `BiRefNet-GUI`（macOS）或 `BiRefNet-GUI.exe`（Windows），首次启动会自动跳转到模型管理页面，下载至少一个模型即可开始使用。

## 从源码运行

```bash
git clone https://github.com/laufirelive/birefnet-gui.git
cd birefnet-gui
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 下载默认模型
python download_models.py

# 启动
python main.py
```

## 支持的格式

### 输入

| 类型 | 格式 |
|------|------|
| 视频 | MP4、AVI、MOV、MKV |
| 图片 | JPG、PNG、TIFF、BMP、WebP |
| 图片文件夹 | 自动识别序列批量处理 |

### 输出

| 格式 | 编码 | 透明通道 | 适用场景 |
|------|------|----------|----------|
| MOV | ProRes 4444 | ✅ | 专业后期 |
| WebM | VP9 | ✅ | 网页播放 |
| MP4 | H.264 | ❌ | 兼容性最好 |
| MP4 | H.265/HEVC | ❌ | 体积更小 |
| MP4 | AV1 | ❌ | 新一代编码 |
| PNG 序列 | PNG (RGBA) | ✅ | 逐帧输出 |
| TIFF 序列 | TIFF | ✅ | 无损后期 |

## 模型说明

| 模型 | 用途 | 显存需求 | 推理分辨率 |
|------|------|----------|-----------|
| BiRefNet-general | 通用分割（推荐） | ~4GB | 1024×1024 |
| BiRefNet_lite | 轻量快速 | ~2GB | 1024×1024 |
| BiRefNet-matting | 精细抠图 | ~4GB | 1024×1024 |
| BiRefNet_HR | 高分辨率 | ~8GB | 2048×2048 |
| BiRefNet_HR-matting | 高分辨率抠图 | ~8GB | 2048×2048 |
| BiRefNet_dynamic | 动态分辨率 | ~4-6GB | 256-2304 |

模型首次使用时通过内置下载器获取，支持 hf-mirror 镜像。

## 技术栈

| 组件 | 技术 |
|------|------|
| GUI | PyQt6 |
| 推理 | PyTorch + BiRefNet |
| 视频处理 | FFmpeg (subprocess) |
| 打包 | PyInstaller |
| CI/CD | GitHub Actions |

## 许可证

[MIT License](LICENSE)

## 致谢

- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) — Zheng Peng 等人的双边参考网络图像分割模型

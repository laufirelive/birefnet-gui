# P5 设计：部署发布 — PyInstaller 打包 + GitHub Actions CI

**日期**: 2026-04-05
**基于**: PROGRESS.md 部署发布章节
**前置**: P4 完成，功能 ready for shipping

---

## 1. 目标

将 BiRefNet GUI 打包为 Windows 和 macOS 双平台便携版应用，通过 GitHub Actions 自动构建，用户解压即用。

**核心原则**：
- 便携版优先（解压即用，无安装过程）
- 不内置模型（用户通过 GUI 下载，包体积 ~500-700MB）
- 不内置 FFmpeg（依赖系统安装，启动时检测）
- 双平台 CI 自动打包（Windows + macOS）

---

## 2. 需要适配的代码改动

### 2.1 路径解析器 — `src/core/paths.py`（新增）

**问题**：当前 `MODELS_DIR` 通过 `__file__` + `../..` 相对路径计算。PyInstaller 打包后 `__file__` 指向 `_MEIPASS` 临时目录，路径会失效。

**方案**：新增统一路径解析模块，所有路径基于 app root 计算。

```python
import sys
import os

def get_app_root() -> str:
    """返回应用根目录。
    
    打包后: exe 所在目录（--onedir 模式）
    开发时: 项目根目录
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后，sys.executable 是 exe 的路径
        return os.path.dirname(sys.executable)
    # 开发时：此文件在 src/core/paths.py，向上两级到项目根
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_dir() -> str:
    """模型目录：<app_root>/models/"""
    return os.path.join(get_app_root(), "models")

def is_frozen() -> bool:
    """是否在 PyInstaller 打包环境中运行"""
    return getattr(sys, 'frozen', False)
```

**需要替换的引用**：

| 文件 | 当前代码 | 改为 |
|------|---------|------|
| `src/gui/main_window.py:38` | `MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")` | `from src.core.paths import get_models_dir` |
| `src/gui/queue_tab.py:277` | `models_dir = os.path.abspath(os.path.join(...))` | `from src.core.paths import get_models_dir` |

**其他路径（无需改动）**：
- `CACHE_DIR`（`~/.birefnet-gui/cache/`）— 基于 `expanduser("~")`，安全
- `BRM_PATH`（`~/.birefnet-gui/queue.brm`）— 基于 `expanduser("~")`，安全

### 2.2 FFmpeg 启动检测

**问题**：当前代码直接 `subprocess.run(["ffmpeg", ...])` 调用 FFmpeg，如果用户未安装会报难以理解的错误。

**方案**：在 `main.py` 启动时检测 FFmpeg 是否可用，不可用则弹对话框提示后退出。

**检测逻辑**：

```python
import shutil

def check_ffmpeg() -> bool:
    """检测 ffmpeg 和 ffprobe 是否在 PATH 中"""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
```

**提示内容（按平台）**：

- **Windows**：
  > 未检测到 FFmpeg。请安装 FFmpeg 后重新启动。
  >
  > 推荐安装方式：
  > 1. 命令行：winget install ffmpeg
  > 2. 手动下载：https://www.gyan.dev/ffmpeg/builds/

- **macOS**：
  > 未检测到 FFmpeg。请安装 FFmpeg 后重新启动。
  >
  > 安装方式：brew install ffmpeg

**实现位置**：`main.py` 的 `main()` 函数中，创建 `QApplication` 后、创建 `MainWindow` 前执行检测。使用 `QMessageBox.critical()` 弹窗。

### 2.3 main.py 打包适配

```python
import multiprocessing

def main():
    multiprocessing.freeze_support()  # PyInstaller 安全措施
    
    app = QApplication(sys.argv)
    
    # FFmpeg 检测
    if not check_ffmpeg():
        show_ffmpeg_missing_dialog()
        sys.exit(1)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

---

## 3. PyInstaller 配置

### 3.1 打包方式

**--onedir 模式**（非 --onefile）：
- PyTorch 体积 ~2GB，--onefile 每次启动要解压到临时目录，等待 10-30s
- --onedir 启动快（~2s），文件夹结构对用户透明（zip 解压后双击 exe）

### 3.2 spec 文件 — `birefnet-gui.spec`

关键配置点：

```python
# birefnet-gui.spec
a = Analysis(
    ['main.py'],
    pathex=[],
    datas=[],           # 不打包模型和 FFmpeg
    hiddenimports=[
        'torch',
        'torchvision', 
        'transformers',
        'einops',
        'kornia',
        'timm',
        'PIL',
        'cv2',
        'numpy',
    ],
    excludes=[
        'matplotlib',
        'tkinter',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
)
```

### 3.3 体积优化

| 措施 | 预估节省 |
|------|---------|
| 排除 CUDA 不需要的库（如 cudnn 训练相关） | ~200MB |
| 排除 tkinter/matplotlib | ~50MB |
| 排除测试/文档文件 | ~10MB |
| UPX 压缩（可选，CI 中开启） | ~10-20% |

**预估最终体积**：
- Windows（含 CUDA）：~600-800MB zip
- macOS（CPU/MPS only）：~400-500MB zip

### 3.4 CUDA 处理

**Windows 版**：打包 CUDA 版 PyTorch（支持 NVIDIA GPU 加速）。
**macOS 版**：打包 CPU 版 PyTorch（MPS 通过 Apple 框架自动可用，不需要额外库）。

---

## 4. GitHub Actions CI

### 4.1 触发条件

```yaml
on:
  push:
    tags:
      - 'v*'      # 打 tag 时触发（如 v1.0.0）
  workflow_dispatch:  # 手动触发（调试用）
```

### 4.2 双平台 matrix

```yaml
strategy:
  matrix:
    include:
      - os: windows-latest
        torch-index: https://download.pytorch.org/whl/cu121
        artifact-name: BiRefNet-GUI-Windows
      - os: macos-latest          # ARM (Apple Silicon)
        torch-index: https://download.pytorch.org/whl/cpu
        artifact-name: BiRefNet-GUI-macOS-ARM
```

### 4.3 构建步骤（每个平台）

1. Checkout 代码
2. 安装 Python 3.11
3. 安装依赖（`pip install -r requirements.txt`）
   - Windows：安装 CUDA 版 PyTorch
   - macOS：安装 CPU 版 PyTorch
4. 安装 PyInstaller
5. 执行打包（`pyinstaller birefnet-gui.spec`）
6. 创建 `models/` 空目录（让用户知道模型放这里）
7. 打包为 zip
8. 上传到 GitHub Release

### 4.4 Release 产物

```
GitHub Release v1.0.0
├── BiRefNet-GUI-v1.0.0-Windows.zip        (~700MB)
└── BiRefNet-GUI-v1.0.0-macOS-ARM.zip     (~450MB, Apple Silicon)
```

用户下载对应平台的 zip，解压，双击运行。

---

## 5. 最终产物结构

### Windows

```
BiRefNet-GUI/
├── BiRefNet-GUI.exe          # 主程序（双击运行）
├── models/                    # 空目录，用户通过 GUI 下载模型
├── _internal/                 # PyInstaller 依赖（PyTorch、PyQt6 等）
│   ├── torch/
│   ├── PyQt6/
│   └── ...
└── README.txt                 # 简要使用说明
```

### macOS

```
BiRefNet-GUI/
├── BiRefNet-GUI               # 主程序（双击或终端运行）
├── models/                    # 空目录
├── _internal/
└── README.txt
```

---

## 6. README.txt 内容

简短的使用说明，随 zip 一起分发：

```
BiRefNet Video Matting Tool

使用前请确保已安装 FFmpeg：
  - Windows: winget install ffmpeg
  - macOS:   brew install ffmpeg

首次使用：
  1. 双击运行 BiRefNet-GUI
  2. 在"模型管理"标签页下载至少一个模型
  3. 选择视频/图片文件开始抠图

模型文件保存在 models/ 目录中。
```

---

## 7. 文件变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/core/paths.py` | **新增** | 统一路径解析（app root / models dir） |
| `main.py` | **修改** | FFmpeg 检测 + freeze_support + 提示对话框 |
| `src/gui/main_window.py` | **修改** | MODELS_DIR 改为引用 paths 模块 |
| `src/gui/queue_tab.py` | **修改** | models_dir 改为引用 paths 模块 |
| `birefnet-gui.spec` | **新增** | PyInstaller 打包配置 |
| `.github/workflows/build.yml` | **新增** | GitHub Actions 双平台 CI |
| `README.txt` | **新增** | 便携版使用说明（随 zip 分发） |

---

## 8. 测试计划

| 模块 | 测试内容 |
|------|---------|
| `paths.py` | `get_app_root()` 开发模式返回项目根；mock `sys.frozen` 测试打包模式 |
| `main.py` | FFmpeg 检测逻辑（mock `shutil.which`） |
| CI workflow | 手动触发一次验证双平台构建成功 |

---

## 9. 实现顺序

1. **paths.py** — 新增路径解析模块 + 测试
2. **替换 MODELS_DIR 引用** — main_window.py + queue_tab.py
3. **FFmpeg 启动检测** — main.py 改造 + 测试
4. **birefnet-gui.spec** — PyInstaller 配置
5. **GitHub Actions workflow** — CI 自动构建
6. **README.txt** — 使用说明
7. **推送到远程仓库** — 首次 push + tag 触发构建

# P5 Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package BiRefNet GUI as portable apps for Windows and macOS via PyInstaller, with GitHub Actions CI for automated builds.

**Architecture:** Add a `paths.py` module for frozen-safe path resolution, add FFmpeg detection at startup, create a PyInstaller spec file, and a GitHub Actions workflow for dual-platform builds. Models are not bundled — users download them via the existing Model Management tab.

**Tech Stack:** PyInstaller, GitHub Actions, PyQt6 QMessageBox (for FFmpeg dialog)

---

### Task 1: Create `paths.py` — unified path resolver

**Files:**
- Create: `src/core/paths.py`
- Create: `tests/test_paths.py`

- [ ] **Step 1: Write failing tests for `get_app_root()` and `get_models_dir()`**

```python
# tests/test_paths.py
import os
import sys
from unittest.mock import patch

from src.core.paths import get_app_root, get_models_dir, is_frozen


class TestGetAppRoot:
    def test_dev_mode_returns_project_root(self):
        """In development mode (not frozen), returns the project root directory."""
        root = get_app_root()
        # paths.py is at src/core/paths.py, so root should contain main.py
        assert os.path.isfile(os.path.join(root, "main.py"))

    def test_frozen_mode_returns_executable_dir(self, tmp_path):
        """When frozen (PyInstaller), returns the directory containing the executable."""
        fake_exe = str(tmp_path / "BiRefNet-GUI.exe")
        with patch.object(sys, "frozen", True, create=True):
            with patch.object(sys, "executable", fake_exe):
                root = get_app_root()
                assert root == str(tmp_path)


class TestGetModelsDir:
    def test_returns_models_subdir(self):
        """models dir is <app_root>/models."""
        models = get_models_dir()
        assert models.endswith("models")
        assert os.path.dirname(models) == get_app_root()


class TestIsFrozen:
    def test_not_frozen_in_dev(self):
        assert is_frozen() is False

    def test_frozen_when_attr_set(self):
        with patch.object(sys, "frozen", True, create=True):
            assert is_frozen() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.core.paths'`

- [ ] **Step 3: Implement `paths.py`**

```python
# src/core/paths.py
"""Unified path resolution for both development and PyInstaller-frozen modes."""

import os
import sys


def is_frozen() -> bool:
    """Return True if running inside a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def get_app_root() -> str:
    """Return the application root directory.

    Frozen (PyInstaller --onedir): directory containing the executable.
    Development: project root (two levels up from this file).
    """
    if is_frozen():
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_models_dir() -> str:
    """Return the path to the models directory (<app_root>/models/)."""
    return os.path.join(get_app_root(), "models")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_paths.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/paths.py tests/test_paths.py
git commit -m "feat: add paths module for frozen-safe path resolution"
```

---

### Task 2: Replace hardcoded `MODELS_DIR` with `paths.get_models_dir()`

**Files:**
- Modify: `src/gui/main_window.py:38`
- Modify: `src/gui/queue_tab.py:277`

- [ ] **Step 1: Update `main_window.py` — replace `MODELS_DIR` definition**

In `src/gui/main_window.py`, replace line 38:

```python
# Old:
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# New:
from src.core.paths import get_models_dir
MODELS_DIR = get_models_dir()
```

Place the import alongside the other `src.core` imports (around line 21-28). Remove the `MODELS_DIR = os.path.join(...)` line and replace it with:

```python
MODELS_DIR = get_models_dir()
```

- [ ] **Step 2: Update `queue_tab.py` — replace inline `models_dir` calculation**

In `src/gui/queue_tab.py`, at the top of the file add:

```python
from src.core.paths import get_models_dir
```

Then replace line 277:

```python
# Old:
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

# New:
models_dir = get_models_dir()
```

- [ ] **Step 3: Run all existing tests to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: all 154 tests PASS (paths change is transparent in dev mode)

- [ ] **Step 4: Commit**

```bash
git add src/gui/main_window.py src/gui/queue_tab.py
git commit -m "refactor: use paths module for MODELS_DIR resolution"
```

---

### Task 3: Add FFmpeg startup detection

**Files:**
- Modify: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests for FFmpeg detection**

```python
# tests/test_main.py
import sys
from unittest.mock import patch, MagicMock

from main import check_ffmpeg


class TestCheckFfmpeg:
    def test_returns_true_when_both_found(self):
        with patch("main.shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            assert check_ffmpeg() is True

    def test_returns_false_when_ffmpeg_missing(self):
        def which(name):
            return None if name == "ffmpeg" else f"/usr/bin/{name}"
        with patch("main.shutil.which", side_effect=which):
            assert check_ffmpeg() is False

    def test_returns_false_when_ffprobe_missing(self):
        def which(name):
            return None if name == "ffprobe" else f"/usr/bin/{name}"
        with patch("main.shutil.which", side_effect=which):
            assert check_ffmpeg() is False

    def test_returns_false_when_both_missing(self):
        with patch("main.shutil.which", return_value=None):
            assert check_ffmpeg() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_main.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_ffmpeg' from 'main'`

- [ ] **Step 3: Implement FFmpeg detection and startup dialog in `main.py`**

Replace the entire `main.py` with:

```python
# main.py
import multiprocessing
import platform
import shutil
import sys

from PyQt6.QtWidgets import QApplication, QMessageBox

from src.gui.main_window import MainWindow


def check_ffmpeg() -> bool:
    """Return True if ffmpeg and ffprobe are found in PATH."""
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _ffmpeg_install_message() -> str:
    """Return platform-specific FFmpeg installation instructions."""
    if platform.system() == "Darwin":
        return (
            "未检测到 FFmpeg，请安装后重新启动。\n\n"
            "安装方式:\n"
            "  brew install ffmpeg"
        )
    return (
        "未检测到 FFmpeg，请安装后重新启动。\n\n"
        "推荐安装方式:\n"
        "  1. 命令行: winget install ffmpeg\n"
        "  2. 手动下载: https://www.gyan.dev/ffmpeg/builds/"
    )


def main():
    multiprocessing.freeze_support()

    app = QApplication(sys.argv)
    app.setApplicationName("BiRefNet Video Matting Tool")

    if not check_ffmpeg():
        QMessageBox.critical(None, "缺少 FFmpeg", _ffmpeg_install_message())
        sys.exit(1)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_main.py -v`
Expected: all 4 PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: detect FFmpeg at startup, show install instructions if missing"
```

---

### Task 4: Create PyInstaller spec file

**Files:**
- Create: `birefnet-gui.spec`

- [ ] **Step 1: Create the spec file**

```python
# birefnet-gui.spec
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BiRefNet GUI — portable build (--onedir)."""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect transformers model config files needed at runtime
transformers_datas = collect_data_files("transformers", includes=["**/*.json"])

# Hidden imports that PyInstaller cannot detect via static analysis
hidden = [
    "einops",
    "kornia",
    "timm",
    "PIL",
    "cv2",
    "psutil",
    "huggingface_hub",
]
# transformers may lazy-import model classes
hidden += collect_submodules("transformers.models.bit")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=transformers_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "tkinter",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
        "IPython",
        "jedi",
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BiRefNet-GUI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,           # GUI app, no console window
    icon=None,               # TODO: add app icon later
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="BiRefNet-GUI",
)
```

- [ ] **Step 2: Verify spec syntax by doing a dry-run parse**

Run: `python -c "exec(open('birefnet-gui.spec').read())" 2>&1 || echo "Syntax OK if only ImportError about PyInstaller"`

Expected: `ImportError` about PyInstaller (not installed in dev venv) or clean exit — confirms no Python syntax errors.

- [ ] **Step 3: Add `.spec` temp artifacts to `.gitignore`**

Append to `.gitignore`:

```
# PyInstaller
*.spec.bak
```

(`dist/` and `build/` are already ignored.)

- [ ] **Step 4: Commit**

```bash
git add birefnet-gui.spec .gitignore
git commit -m "feat: add PyInstaller spec for portable --onedir build"
```

---

### Task 5: Create GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/build.yml`

- [ ] **Step 1: Create workflow directory**

```bash
mkdir -p .github/workflows
```

- [ ] **Step 2: Create the workflow file**

```yaml
# .github/workflows/build.yml
name: Build Portable Release

on:
  push:
    tags:
      - "v*"
  workflow_dispatch:       # Manual trigger for testing

permissions:
  contents: write          # Needed to create releases

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: windows-latest
            artifact: BiRefNet-GUI-Windows
            torch-extra: --index-url https://download.pytorch.org/whl/cu121
          - os: macos-latest            # ARM (Apple Silicon)
            artifact: BiRefNet-GUI-macOS-ARM
            torch-extra: ""

    runs-on: ${{ matrix.os }}

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install PyTorch
        run: |
          pip install torch torchvision ${{ matrix.torch-extra }}

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pyinstaller

      - name: Build with PyInstaller
        run: |
          pyinstaller birefnet-gui.spec

      - name: Create empty models directory
        shell: bash
        run: mkdir -p dist/BiRefNet-GUI/models

      - name: Create README.txt
        shell: bash
        run: |
          cat > dist/BiRefNet-GUI/README.txt << 'ENDOFREADME'
          BiRefNet Video Matting Tool

          使用前请确保已安装 FFmpeg:
            - Windows: winget install ffmpeg
            - macOS:   brew install ffmpeg

          首次使用:
            1. 运行 BiRefNet-GUI
            2. 在"模型管理"标签页下载至少一个模型
            3. 选择视频/图片文件开始抠图

          模型文件保存在 models/ 目录中。
          ENDOFREADME

      - name: Package as zip
        shell: bash
        run: |
          cd dist
          if [ "$RUNNER_OS" == "Windows" ]; then
            7z a -tzip "../${{ matrix.artifact }}.zip" BiRefNet-GUI/
          else
            zip -r "../${{ matrix.artifact }}.zip" BiRefNet-GUI/
          fi

      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.artifact }}
          path: ${{ matrix.artifact }}.zip

      - name: Upload to Release
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: ${{ matrix.artifact }}.zip
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/build.yml
git commit -m "ci: add GitHub Actions workflow for dual-platform portable builds"
```

---

### Task 6: Update PROGRESS.md and push to remote

**Files:**
- Modify: `PROGRESS.md`

- [ ] **Step 1: Add P5 section to PROGRESS.md**

After the P4 section and before the "未完成功能" section, add:

```markdown
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
| (其余测试不变) | 154 | — |
| **合计** | **163** | **全部通过** |
```

Also update the header of PROGRESS.md:

```markdown
**更新日期**: 2026-04-05
**当前版本**: P5 完成 (部署发布: PyInstaller + GitHub Actions CI)
**分支**: `feature/p5-deployment`
```

And in the "未完成功能" section, move the deployment items to "已完成" or remove them:
- Remove "PyInstaller 打包" from 部署发布
- Remove "安装程序" — 便携版已实现，安装版暂不做
- Keep "全模型打包" as future — currently not bundling models

- [ ] **Step 2: Run full test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: all tests PASS (163 total)

- [ ] **Step 3: Commit**

```bash
git add PROGRESS.md
git commit -m "docs: update PROGRESS.md — P5 deployment complete"
```

- [ ] **Step 4: Push to remote**

```bash
git push -u origin master
```

- [ ] **Step 5: Trigger a test build (optional)**

To test the CI workflow without creating a release, use the GitHub Actions manual trigger:

```bash
gh workflow run build.yml
```

Or push a test tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

This will trigger the workflow and create a GitHub Release with the zip artifacts.

# birefnet-gui.spec
# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BiRefNet GUI — portable build (--onedir)."""

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

# CUDA/PyTorch libs not needed for inference — need to fit GitHub 2GB limit
cuda_exclude = [
    # CUDA libs not used by neural network inference
    "cusolver*",
    "cusparse*",
    "npp*",
    "nvrtc*",
    "nvjpeg*",
    "curand*",
    "cufft*",
    "nvjitlink*",
    "cublasLt*",           # ~200MB, cublas alone suffices for inference
    # cuDNN training-only libs
    "cudnn_adv*",          # advanced ops (RNN training etc.)
    "cudnn_cnn_train*",
    "cudnn_ops_train*",
    # Triton compiler — not needed for eager-mode inference
    "triton*",
]

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

# Filter out unnecessary CUDA libraries from binaries
import fnmatch
a.binaries = [
    b for b in a.binaries
    if not any(fnmatch.fnmatch(b[0].lower(), pat) for pat in cuda_exclude)
]

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
    console=False,
    icon=None,
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

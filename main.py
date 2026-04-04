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

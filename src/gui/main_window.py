import os
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.inference import detect_device
from src.core.video import get_video_info
from src.worker.matting_worker import MattingWorker

# Path to bundled model (relative to project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "birefnet-general")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(600, 450)

        self._worker = None
        self._input_path = None
        self._output_dir = None
        self._start_time = None

        self._init_ui()
        self._set_state("initial")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Input file section ---
        layout.addWidget(QLabel("输入文件:"))
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setReadOnly(True)
        self._input_edit.setPlaceholderText("未选择文件")
        input_row.addWidget(self._input_edit)
        self._select_btn = QPushButton("选择文件")
        self._select_btn.clicked.connect(self._on_select_file)
        input_row.addWidget(self._select_btn)
        layout.addLayout(input_row)

        # Video info
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray;")
        layout.addWidget(self._info_label)

        # --- Separator ---
        sep1 = QLabel()
        sep1.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(sep1)

        # --- Model / Device / Output format ---
        device = detect_device()
        device_text = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
        self._model_label = QLabel("模型: BiRefNet-general")
        layout.addWidget(self._model_label)
        self._device_label = QLabel(f"设备: {device_text.get(device, device)}")
        layout.addWidget(self._device_label)
        self._format_label = QLabel("输出: MOV ProRes 4444 (透明)")
        layout.addWidget(self._format_label)

        # --- Output path ---
        layout.addWidget(QLabel("输出路径:"))
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("与输入文件同目录")
        output_row.addWidget(self._output_edit)
        self._output_btn = QPushButton("浏览...")
        self._output_btn.clicked.connect(self._on_select_output)
        output_row.addWidget(self._output_btn)
        layout.addLayout(output_row)

        # --- Separator ---
        sep2 = QLabel()
        sep2.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        layout.addWidget(sep2)

        # --- Progress ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        layout.addWidget(self._status_label)

        # --- Control buttons ---
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._start_btn = QPushButton("开始处理")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()

    def _set_state(self, state: str):
        """Update button enabled/disabled state based on current state."""
        self._state = state
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._select_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._select_btn.setEnabled(True)

    def _on_select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if not path:
            return

        try:
            info = get_video_info(path)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取视频文件:\n{e}")
            return

        self._input_path = path
        self._input_edit.setText(path)

        w, h = info["width"], info["height"]
        fps = info["fps"]
        frames = info["frame_count"]
        dur = info["duration"]
        minutes = int(dur // 60)
        seconds = int(dur % 60)
        self._info_label.setText(
            f"视频信息: {w}x{h} | {fps:.1f}fps | {frames}帧 | {minutes:02d}:{seconds:02d}"
        )

        self._set_state("ready")

    def _on_select_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self._output_dir = directory
            self._output_edit.setText(directory)

    def _build_output_path(self) -> str:
        base_name = os.path.splitext(os.path.basename(self._input_path))[0]
        timestamp = int(time.time() * 1000)
        filename = f"{base_name}_birefnet-general_{timestamp}.mov"

        if self._output_dir:
            return os.path.join(self._output_dir, filename)
        else:
            return os.path.join(os.path.dirname(self._input_path), filename)

    def _on_start(self):
        if not self._input_path:
            return

        # Check model exists
        model_path = os.path.abspath(MODEL_PATH)
        if not os.path.isdir(model_path):
            QMessageBox.critical(
                self,
                "模型缺失",
                f"未找到 BiRefNet-general 模型:\n{model_path}\n\n"
                "请运行 python download_models.py 下载模型。",
            )
            return

        output_path = self._build_output_path()
        self._start_time = time.time()
        self._set_state("processing")
        self._status_label.setText("正在加载模型...")

        self._worker = MattingWorker(model_path, self._input_path, output_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.speed.connect(self._on_speed)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_pause(self):
        if not self._worker:
            return
        if self._state == "processing":
            self._worker.pause()
            self._set_state("paused")
        elif self._state == "paused":
            self._worker.resume()
            self._set_state("processing")

    def _on_cancel(self):
        if not self._worker:
            return
        self._worker.cancel()
        self._worker.wait()
        self._set_state("ready")
        self._progress_bar.setValue(0)
        self._status_label.setText("已取消")

    def _on_progress(self, current: int, total: int):
        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            self._status_label.setText(
                f"帧: {current}/{total} | 速度: {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )

    def _on_speed(self, fps: float):
        pass  # Speed is calculated in _on_progress from wall clock for accuracy

    def _on_finished(self, output_path: str):
        self._set_state("finished")
        self._progress_bar.setValue(100)
        elapsed = time.time() - self._start_time if self._start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._status_label.setText(f"处理完成! 耗时: {minutes:02d}:{seconds:02d}")

        QMessageBox.information(
            self,
            "完成",
            f"视频处理完成!\n\n输出文件:\n{output_path}",
        )

    def _on_error(self, message: str):
        self._set_state("ready")
        self._progress_bar.setValue(0)
        if message != "Processing cancelled":
            QMessageBox.critical(self, "错误", f"处理出错:\n{message}")
            self._status_label.setText(f"错误: {message}")

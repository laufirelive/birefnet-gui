import os
import time

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    BackgroundMode,
    IMAGE_EXTENSIONS,
    InputType,
    MODELS,
    OutputFormat,
    ProcessingConfig,
    VIDEO_EXTENSIONS,
)
from src.core.inference import detect_device
from src.core.queue_manager import QueueManager
from src.core.queue_task import QueueTask
from src.core.video import get_video_info
from src.gui.queue_tab import QueueTab
from src.worker.matting_worker import MattingWorker

# Path to bundled models directory (relative to project root)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

# Path to queue persistence file
BRM_PATH = os.path.join(os.path.expanduser("~"), ".birefnet-gui", "queue.brm")

# Display names for output formats
FORMAT_LABELS = {
    OutputFormat.MOV_PRORES: "MOV ProRes 4444",
    OutputFormat.WEBM_VP9: "WebM VP9",
    OutputFormat.MP4_H264: "MP4 H.264",
    OutputFormat.MP4_H265: "MP4 H.265/HEVC",
    OutputFormat.MP4_AV1: "MP4 AV1",
    OutputFormat.PNG_SEQUENCE: "PNG 序列",
    OutputFormat.TIFF_SEQUENCE: "TIFF 序列",
}

# Display names for background modes
MODE_LABELS = {
    BackgroundMode.TRANSPARENT: "透明背景",
    BackgroundMode.GREEN: "绿幕",
    BackgroundMode.BLUE: "蓝幕",
    BackgroundMode.MASK_BW: "黑底白蒙版",
    BackgroundMode.MASK_WB: "白底黑蒙版",
    BackgroundMode.SIDE_BY_SIDE: "原图+蒙版分轨",
}

# File extension mapping
FORMAT_EXTENSIONS = {
    OutputFormat.MOV_PRORES: ".mov",
    OutputFormat.WEBM_VP9: ".webm",
    OutputFormat.MP4_H264: ".mp4",
    OutputFormat.MP4_H265: ".mp4",
    OutputFormat.MP4_AV1: ".mp4",
    OutputFormat.PNG_SEQUENCE: "",
    OutputFormat.TIFF_SEQUENCE: "",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BiRefNet Video Matting Tool")
        self.setMinimumSize(750, 500)

        self._worker = None
        self._input_path = None
        self._output_dir = None
        self._start_time = None
        self._input_type = None
        self._current_phase = None

        self._queue_manager = QueueManager(brm_path=BRM_PATH)
        self._queue_manager.load()

        self._init_ui()
        self.setAcceptDrops(True)
        self._set_state("initial")

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer_layout = QVBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        outer_layout.addWidget(self._tabs)

        # --- Tab 1: Single Task ---
        tab1 = QWidget()
        self._tabs.addTab(tab1, "单任务")

        main_layout = QHBoxLayout(tab1)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Left panel ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        # Input file
        left_panel.addWidget(QLabel("输入文件:"))
        input_row = QHBoxLayout()
        self._input_edit = QLineEdit()
        self._input_edit.setReadOnly(True)
        self._input_edit.setPlaceholderText("未选择文件")
        input_row.addWidget(self._input_edit)
        self._select_btn = QPushButton("选择文件 ▼")
        select_menu = QMenu(self)
        select_menu.addAction("选择视频", self._on_select_video)
        select_menu.addAction("选择图片", self._on_select_image)
        select_menu.addAction("选择图片文件夹", self._on_select_folder)
        self._select_btn.setMenu(select_menu)
        input_row.addWidget(self._select_btn)
        left_panel.addLayout(input_row)

        # Video info
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._info_label)

        # Separator
        sep1 = QLabel()
        sep1.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        left_panel.addWidget(sep1)

        # Output path
        left_panel.addWidget(QLabel("输出路径:"))
        output_row = QHBoxLayout()
        self._output_edit = QLineEdit()
        self._output_edit.setReadOnly(True)
        self._output_edit.setPlaceholderText("与输入文件同目录")
        output_row.addWidget(self._output_edit)
        self._output_btn = QPushButton("浏览...")
        self._output_btn.clicked.connect(self._on_select_output)
        output_row.addWidget(self._output_btn)
        left_panel.addLayout(output_row)

        # Separator
        sep2 = QLabel()
        sep2.setFrameStyle(QLabel.Shape.HLine | QLabel.Shadow.Sunken)
        left_panel.addWidget(sep2)

        # Progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        left_panel.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        left_panel.addWidget(self._status_label)

        # Control buttons
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

        self._enqueue_btn = QPushButton("加入队列")
        self._enqueue_btn.clicked.connect(self._on_enqueue)
        btn_row.addWidget(self._enqueue_btn)
        btn_row.addStretch()
        left_panel.addLayout(btn_row)

        left_panel.addStretch()

        # --- Right panel ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Model selection
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型:"))
        self._model_combo = QComboBox()
        self._populate_model_combo()
        model_layout.addWidget(self._model_combo)

        device = detect_device()
        device_text = {"cuda": "CUDA (GPU)", "mps": "MPS (Apple Silicon)", "cpu": "CPU"}
        self._device_label = QLabel(f"设备: {device_text.get(device, device)}")
        self._device_label.setStyleSheet("color: gray;")
        model_layout.addWidget(self._device_label)

        right_panel.addWidget(model_group)

        # Output settings
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)

        output_layout.addWidget(QLabel("格式:"))
        self._format_combo = QComboBox()
        for fmt, label in FORMAT_LABELS.items():
            self._format_combo.addItem(label, fmt)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        output_layout.addWidget(self._format_combo)

        output_layout.addWidget(QLabel("背景:"))
        self._mode_combo = QComboBox()
        self._populate_mode_combo()
        output_layout.addWidget(self._mode_combo)

        right_panel.addWidget(output_group)

        right_panel.addStretch()

        # --- Assemble ---
        main_layout.addLayout(left_panel, stretch=2)
        main_layout.addLayout(right_panel, stretch=1)

        # --- Tab 2: Queue ---
        self._queue_tab = QueueTab(self._queue_manager, self._get_config)
        self._queue_tab.queue_running_changed.connect(self._on_queue_running_changed)
        self._queue_tab.task_count_changed.connect(
            lambda count: self._tabs.setTabText(1, f"批量队列 ({count})" if count > 0 else "批量队列")
        )
        self._tabs.addTab(self._queue_tab, self._get_queue_tab_title())

    def _populate_model_combo(self):
        """Populate model dropdown. Only show downloaded models as enabled."""
        models_dir = os.path.abspath(MODELS_DIR)
        for display_name, dir_name in MODELS.items():
            model_path = os.path.join(models_dir, dir_name)
            if os.path.isdir(model_path):
                self._model_combo.addItem(display_name, display_name)
            else:
                self._model_combo.addItem(f"{display_name} (未下载)", display_name)
                idx = self._model_combo.count() - 1
                self._model_combo.model().item(idx).setEnabled(False)

    def _populate_mode_combo(self):
        """Populate background mode dropdown based on current format."""
        self._mode_combo.clear()
        current_format = self._format_combo.currentData()
        for mode, label in MODE_LABELS.items():
            if mode.needs_alpha and current_format and not current_format.supports_alpha:
                continue
            self._mode_combo.addItem(label, mode)

    def _on_format_changed(self, _index):
        """When format changes, update available background modes."""
        self._populate_mode_combo()

    def _get_config(self) -> ProcessingConfig:
        """Build ProcessingConfig from current UI selections."""
        return ProcessingConfig(
            model_name=self._model_combo.currentData(),
            output_format=self._format_combo.currentData(),
            background_mode=self._mode_combo.currentData(),
        )

    def _set_state(self, state: str):
        self._state = state
        if state == "initial":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(True)
            self._progress_bar.setValue(0)
            self._status_label.setText("")
        elif state == "ready":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._select_btn.setEnabled(True)
        elif state == "processing":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("暂停")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "paused":
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
            self._pause_btn.setText("继续")
            self._cancel_btn.setEnabled(True)
            self._enqueue_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._queue_tab._start_btn.setEnabled(False)
        elif state == "finished":
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
            self._cancel_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(True)
            self._select_btn.setEnabled(True)
            self._queue_tab._set_queue_state(self._queue_tab._queue_state)

    def _on_select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择视频文件",
            "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp);;所有文件 (*)",
        )
        if path:
            self._handle_input(path)

    def _on_select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if path:
            self._handle_input(path)

    def _classify_input(self, path: str) -> InputType | None:
        """Determine input type from a file or directory path."""
        if os.path.isdir(path):
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            )
            return InputType.IMAGE_FOLDER if has_images else None

        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return InputType.VIDEO
        if ext in IMAGE_EXTENSIONS:
            return InputType.IMAGE
        return None

    def _handle_input(self, path: str):
        """Unified entry point for all input methods (file dialog, drag-and-drop)."""
        input_type = self._classify_input(path)

        if input_type is None:
            QMessageBox.warning(
                self, "不支持的文件",
                f"无法识别的文件类型:\n{path}\n\n"
                "支持的格式: 视频(MP4/AVI/MOV/MKV) | 图片(PNG/JPG/TIFF/BMP/WebP)",
            )
            return

        if input_type == InputType.VIDEO:
            try:
                info = get_video_info(path)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取视频文件:\n{e}")
                return

            self._input_path = path
            self._input_type = input_type
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

        elif input_type == InputType.IMAGE:
            from PIL import Image
            try:
                img = Image.open(path)
                w, h = img.size
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法读取图片:\n{e}")
                return

            self._input_path = path
            self._input_type = input_type
            self._input_edit.setText(path)
            self._info_label.setText(f"图片信息: {w}x{h} | {img.mode}")

        elif input_type == InputType.IMAGE_FOLDER:
            image_files = [
                f for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                and os.path.isfile(os.path.join(path, f))
            ]
            count = len(image_files)

            self._input_path = path
            self._input_type = input_type
            self._input_edit.setText(path)
            self._info_label.setText(f"图片文件夹: {count} 张图片")

        self._update_ui_for_input_type()
        self._set_state("ready")

    def _update_ui_for_input_type(self):
        """Adjust output settings based on input type."""
        is_video = self._input_type == InputType.VIDEO
        self._format_combo.setEnabled(is_video)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self._handle_input(path)

    def _on_select_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self._output_dir = directory
            self._output_edit.setText(directory)

    def _build_output_path(self) -> str:
        """Build output path based on input type."""
        config = self._get_config()
        model_dir_name = MODELS[config.model_name]
        timestamp = int(time.time() * 1000)

        if self._input_type == InputType.VIDEO:
            base_name = os.path.splitext(os.path.basename(self._input_path))[0]
            ext = FORMAT_EXTENSIONS[config.output_format]
            if ext:
                filename = f"{base_name}_{model_dir_name}_{timestamp}{ext}"
            else:
                filename = f"{base_name}_{model_dir_name}_{timestamp}"
            if self._output_dir:
                return os.path.join(self._output_dir, filename)
            else:
                return os.path.join(os.path.dirname(self._input_path), filename)
        else:
            # For image input, return the output directory
            if self._output_dir:
                return self._output_dir
            else:
                if os.path.isdir(self._input_path):
                    return os.path.dirname(self._input_path)
                else:
                    return os.path.dirname(self._input_path)

    def _on_start(self):
        if not self._input_path:
            return
        if self._queue_tab._queue_state != "idle":
            QMessageBox.warning(self, "提示", "队列正在执行中，请等待队列完成")
            return

        config = self._get_config()
        models_dir = os.path.abspath(MODELS_DIR)

        model_dir_name = MODELS[config.model_name]
        model_path = os.path.join(models_dir, model_dir_name)
        if not os.path.isdir(model_path):
            QMessageBox.critical(
                self,
                "模型缺失",
                f"未找到 {config.model_name} 模型:\n{model_path}\n\n"
                "请运行 python download_models.py 下载模型。",
            )
            return

        output_path = self._build_output_path()
        self._start_time = time.time()
        self._set_state("processing")
        self._status_label.setText("正在加载模型...")

        self._worker = MattingWorker(
            config, models_dir, self._input_path, output_path,
            input_type=self._input_type,
        )
        self._worker.progress.connect(self._on_progress)
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

    def _on_progress(self, current: int, total: int, phase: str):
        if phase != self._current_phase:
            self._current_phase = phase
            self._start_time = time.time()

        percent = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(percent)

        elapsed = time.time() - self._start_time if self._start_time else 0
        if current > 0 and elapsed > 0:
            fps = current / elapsed
            remaining = (total - current) / fps if fps > 0 else 0
            rem_min = int(remaining // 60)
            rem_sec = int(remaining % 60)
            phase_label = {"inference": "推理中", "encoding": "编码中", "processing": "处理中"}.get(phase, phase)
            self._status_label.setText(
                f"{phase_label}: {current}/{total} | {fps:.1f} FPS | 剩余: {rem_min:02d}:{rem_sec:02d}"
            )

    def _on_finished(self, output_path: str):
        self._set_state("finished")
        self._progress_bar.setValue(100)
        elapsed = time.time() - self._start_time if self._start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        self._status_label.setText(f"处理完成! 耗时: {minutes:02d}:{seconds:02d}")

        if self._input_type == InputType.VIDEO:
            msg = f"视频处理完成!\n\n输出文件:\n{output_path}"
        elif self._input_type == InputType.IMAGE:
            msg = f"图片处理完成!\n\n输出文件:\n{output_path}"
        else:
            msg = f"图片文件夹处理完成!\n\n输出目录:\n{output_path}"

        QMessageBox.information(self, "完成", msg)

    def _on_error(self, message: str):
        self._set_state("ready")
        self._progress_bar.setValue(0)
        if message != "Processing cancelled":
            QMessageBox.critical(self, "错误", f"处理出错:\n{message}")
            self._status_label.setText(f"错误: {message}")

    def _get_queue_tab_title(self) -> str:
        count = len(self._queue_manager.tasks)
        if count > 0:
            return f"批量队列 ({count})"
        return "批量队列"

    def _update_queue_tab_title(self):
        self._tabs.setTabText(1, self._get_queue_tab_title())

    def _on_enqueue(self):
        if not self._input_path or not self._input_type:
            QMessageBox.warning(self, "提示", "请先选择输入文件")
            return

        config = self._get_config()
        task = QueueTask.create(
            input_path=self._input_path,
            input_type=self._input_type,
            config=config,
            output_dir=self._output_dir,
        )
        self._queue_manager.add_task(task)
        self._queue_manager.save()
        self._queue_tab.refresh()

        # Clear input for next add
        self._input_path = None
        self._input_type = None
        self._input_edit.setText("")
        self._info_label.setText("")
        self._output_dir = None
        self._output_edit.setText("")
        self._set_state("initial")

        self.statusBar().showMessage("已加入队列", 3000)

    def _on_queue_running_changed(self, running: bool):
        """Disable single-task controls when queue is running."""
        if running:
            self._start_btn.setEnabled(False)
            self._select_btn.setEnabled(False)
            self._enqueue_btn.setEnabled(False)
        else:
            # Restore based on current state
            self._set_state(self._state)

    def closeEvent(self, event):
        self._queue_manager.save()
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait()
        if hasattr(self, '_queue_tab') and self._queue_tab._current_worker:
            self._queue_tab._current_worker.cancel()
            self._queue_tab._current_worker.wait()
        event.accept()

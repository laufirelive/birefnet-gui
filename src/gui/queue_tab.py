import os
import time

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.core.cache import MaskCacheManager
from src.core.config import FORMAT_EXTENSIONS, IMAGE_EXTENSIONS, MODELS, VIDEO_EXTENSIONS, InputType, ProcessingConfig
from src.core.queue_manager import QueueManager
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus
from src.worker.matting_worker import CACHE_DIR, MattingWorker


class QueueTab(QWidget):
    """Queue management tab with task list, progress, and controls."""

    queue_running_changed = pyqtSignal(bool)
    task_count_changed = pyqtSignal(int)

    def __init__(self, queue_manager: QueueManager, get_default_config_fn, parent=None):
        super().__init__(parent)
        self._qm = queue_manager
        self._get_default_config = get_default_config_fn
        self._current_worker: MattingWorker | None = None
        self._cache = MaskCacheManager(CACHE_DIR)
        self._start_time: float | None = None
        self._current_phase: str | None = None
        self._queue_state = "idle"
        self._last_save_time = 0.0

        self.setAcceptDrops(True)
        self._init_ui()
        self._refresh_table()
        self._set_queue_state("idle")

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Task table
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["文件名", "模型", "格式", "状态"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._table)

        # Current task progress
        progress_group = QGroupBox("当前任务")
        progress_layout = QVBoxLayout(progress_group)

        self._current_label = QLabel("")
        self._current_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._current_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        progress_layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._status_label)

        self._total_label = QLabel("")
        self._total_label.setStyleSheet("color: gray;")
        progress_layout.addWidget(self._total_label)

        layout.addWidget(progress_group)

        # Control buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        self._start_btn = QPushButton("开始队列")
        self._start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self._start_btn)

        self._pause_btn = QPushButton("暂停")
        self._pause_btn.clicked.connect(self._on_pause)
        btn_row.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("取消当前")
        self._cancel_btn.clicked.connect(self._on_cancel_current)
        btn_row.addWidget(self._cancel_btn)

        self._clear_btn = QPushButton("清空队列")
        self._clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self._clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _set_queue_state(self, state: str):
        self._queue_state = state
        is_idle = state == "idle"
        has_pending = self._qm.next_pending_task() is not None
        self._start_btn.setEnabled(is_idle and has_pending)
        self._pause_btn.setEnabled(state in ("running", "paused"))
        self._pause_btn.setText("继续" if state == "paused" else "暂停")
        self._cancel_btn.setEnabled(not is_idle)
        self._clear_btn.setEnabled(is_idle)
        self.queue_running_changed.emit(not is_idle)

    def _refresh_table(self):
        self._table.setRowCount(0)
        for task in self._qm.tasks:
            row = self._table.rowCount()
            self._table.insertRow(row)
            display_name = os.path.basename(task.input_path.rstrip(os.sep)) or task.input_path
            self._table.setItem(row, 0, QTableWidgetItem(display_name))
            model_dir = MODELS.get(task.config.model_name, task.config.model_name)
            self._table.setItem(row, 1, QTableWidgetItem(model_dir))
            self._table.setItem(row, 2, QTableWidgetItem(task.config.output_format.value))
            self._table.setItem(row, 3, QTableWidgetItem(self._status_text(task)))
        self.task_count_changed.emit(len(self._qm.tasks))

    def _status_text(self, task: QueueTask) -> str:
        if task.status == TaskStatus.COMPLETED:
            return "完成"
        if task.status == TaskStatus.FAILED:
            return "失败"
        if task.status == TaskStatus.CANCELLED:
            return "已取消"
        if task.status == TaskStatus.PROCESSING:
            phase = "推理" if task.phase == ProcessingPhase.INFERENCE else "编码"
            if task.total > 0:
                return f"{phase} {task.progress}/{task.total}"
            return f"{phase}中..."
        if task.status == TaskStatus.PAUSED:
            if task.total > 0:
                return f"暂停 {task.progress}/{task.total}"
            return "暂停"
        return "等待"

    def _show_context_menu(self, pos):
        row = self._table.rowAt(pos.y())
        if row < 0 or row >= len(self._qm.tasks):
            return
        task = self._qm.tasks[row]
        if task.status == TaskStatus.PROCESSING:
            return

        menu = QMenu(self)
        menu.addAction("删除", lambda: self._remove_task(task.id))
        if row > 0:
            menu.addAction("移到顶部", lambda: self._move_task(task.id, 0))
        if row < len(self._qm.tasks) - 1:
            menu.addAction("移到底部", lambda: self._move_task(task.id, len(self._qm.tasks) - 1))
        menu.exec(self._table.viewport().mapToGlobal(pos))

    def _remove_task(self, task_id: str):
        self._qm.remove_task(task_id)
        self._cache.cleanup(task_id)
        self._qm.save()
        self._refresh_table()
        self._set_queue_state(self._queue_state)

    def _move_task(self, task_id: str, new_index: int):
        self._qm.move_task(task_id, new_index)
        self._qm.save()
        self._refresh_table()

    # --- External drag-drop ---
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        for url in urls:
            path = url.toLocalFile()
            if not path:
                continue
            input_type = self._classify_input(path)
            if input_type is None:
                continue
            config = self._get_default_config()
            task = QueueTask.create(
                input_path=path,
                input_type=input_type,
                config=config,
            )
            self._qm.add_task(task)
        self._qm.save()
        self._refresh_table()
        self._set_queue_state(self._queue_state)

    def _classify_input(self, path: str) -> InputType | None:
        if os.path.isdir(path):
            has_images = any(
                os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
            )
            return InputType.IMAGE_FOLDER if has_images else None
        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            return InputType.VIDEO
        if ext in IMAGE_EXTENSIONS:
            return InputType.IMAGE
        return None

    # --- Queue execution ---
    def _on_start(self):
        self._run_next_task()

    def _run_next_task(self):
        task = self._qm.next_pending_task()
        if task is None:
            self._on_queue_finished()
            return

        task.status = TaskStatus.PROCESSING
        self._qm.save()
        self._refresh_table()
        self._set_queue_state("running")

        start_frame = 0
        if task.input_type == InputType.VIDEO and task.phase == ProcessingPhase.INFERENCE:
            start_frame = self._cache.get_cached_count(task.id)

        output_path = self._build_output_path(task)
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

        self._current_worker = MattingWorker(
            config=task.config,
            models_dir=models_dir,
            input_path=task.input_path,
            output_path=output_path,
            input_type=task.input_type,
            task_id=task.id,
            start_frame=start_frame,
            cleanup_cache=False,
        )
        self._current_worker.progress.connect(
            lambda c, t, p: self._on_task_progress(task.id, c, t, p)
        )
        self._current_worker.finished.connect(
            lambda path: self._on_task_finished(task.id, path)
        )
        self._current_worker.error.connect(
            lambda msg: self._on_task_error(task.id, msg)
        )

        self._start_time = time.time()
        self._current_phase = None
        self._current_label.setText(f"当前: {os.path.basename(task.input_path)}")
        self._progress_bar.setValue(0)
        self._current_worker.start()

    def _build_output_path(self, task: QueueTask) -> str:
        model_dir = MODELS[task.config.model_name]
        timestamp = int(time.time() * 1000)

        if task.input_type == InputType.VIDEO:
            base_name = os.path.splitext(os.path.basename(task.input_path))[0]
            ext = FORMAT_EXTENSIONS.get(task.config.output_format, ".mov")
            filename = f"{base_name}_{model_dir}_{timestamp}{ext}"
            out_dir = task.output_dir or os.path.dirname(task.input_path)
            return os.path.join(out_dir, filename)
        else:
            return task.output_dir or os.path.dirname(task.input_path)

    def _on_task_progress(self, task_id: str, current: int, total: int, phase: str):
        task = self._qm.get_task(task_id)
        if task is None:
            return

        task.progress = current
        task.total = total
        task.phase = ProcessingPhase.INFERENCE if phase == "inference" else ProcessingPhase.ENCODING

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

        self._update_total_progress()

        # Throttle expensive operations: save + table refresh every 2s or 100 frames
        now = time.time()
        if now - self._last_save_time >= 2.0 or current % 100 == 0:
            self._qm.save()
            self._refresh_table()
            self._last_save_time = now

    def _update_total_progress(self):
        completed_work = 0
        total_work = 0
        for t in self._qm.tasks:
            if t.input_type == InputType.VIDEO:
                weight = max(t.total, 1) * 2
                total_work += weight
                if t.status == TaskStatus.COMPLETED:
                    completed_work += weight
                elif t.status == TaskStatus.PROCESSING:
                    done = t.progress
                    if t.phase == ProcessingPhase.ENCODING:
                        done += t.total
                    completed_work += done
            else:
                weight = max(t.total, 1)
                total_work += weight
                if t.status == TaskStatus.COMPLETED:
                    completed_work += weight
                elif t.status == TaskStatus.PROCESSING:
                    completed_work += t.progress

        task_count = len(self._qm.tasks)
        completed_count = sum(1 for t in self._qm.tasks if t.status == TaskStatus.COMPLETED)
        current_idx = completed_count + 1
        if total_work > 0:
            pct = int(completed_work / total_work * 100)
            self._total_label.setText(f"队列: 任务 {current_idx}/{task_count} — 总进度 {pct}%")

    def _on_task_finished(self, task_id: str, output_path: str):
        task = self._qm.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.phase = ProcessingPhase.DONE
            self._cache.cleanup(task_id)
        self._qm.save()
        self._refresh_table()
        self._current_worker = None
        self._run_next_task()

    def _on_task_error(self, task_id: str, message: str):
        task = self._qm.get_task(task_id)
        if task:
            if message == "Processing cancelled":
                task.status = TaskStatus.CANCELLED
            else:
                task.status = TaskStatus.FAILED
                task.error = message
        self._qm.save()
        self._refresh_table()
        self._current_worker = None
        self._run_next_task()

    def _on_queue_finished(self):
        self._set_queue_state("idle")
        self._current_label.setText("队列完成")
        self._status_label.setText("")
        self._progress_bar.setValue(100)
        QApplication.beep()

    def _on_pause(self):
        if self._queue_state == "running" and self._current_worker:
            self._current_worker.pause()
            self._set_queue_state("paused")
            task = self._current_running_task()
            if task:
                task.status = TaskStatus.PAUSED
                self._qm.save()
                self._refresh_table()
        elif self._queue_state == "paused" and self._current_worker:
            self._current_worker.resume()
            self._set_queue_state("running")
            task = self._current_running_task()
            if task:
                task.status = TaskStatus.PROCESSING
                self._qm.save()
                self._refresh_table()

    def _on_cancel_current(self):
        if self._current_worker:
            self._current_worker.cancel()
            self._current_worker.wait()

    def _on_clear(self):
        reply = QMessageBox.question(
            self, "确认", "清空队列将删除所有任务和缓存，确定吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._cache.cleanup_all()
            self._qm.clear_all()
            self._qm.save()
            self._refresh_table()
            self._current_label.setText("")
            self._status_label.setText("")
            self._total_label.setText("")
            self._progress_bar.setValue(0)
            self._set_queue_state("idle")

    def _current_running_task(self) -> QueueTask | None:
        for t in self._qm.tasks:
            if t.status in (TaskStatus.PROCESSING, TaskStatus.PAUSED):
                return t
        return None

    def refresh(self):
        self._refresh_table()
        self._set_queue_state(self._queue_state)

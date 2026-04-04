import os

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from src.core.config import MODEL_REGISTRY
from src.core.model_downloader import ModelDownloader


class DownloadWorker(QThread):
    """Background thread for model download."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str, str)

    def __init__(self, downloader: ModelDownloader, model_key: str):
        super().__init__()
        self._downloader = downloader
        self._model_key = model_key

    def run(self):
        try:
            self.progress.emit(f"正在下载 {MODEL_REGISTRY[self._model_key].display_name}...")
            self._downloader.download_model(self._model_key)
            self.finished.emit(self._model_key)
        except Exception as e:
            self.error.emit(self._model_key, str(e))


class ModelCard(QWidget):
    """A card displaying one model's info and action button."""

    download_requested = pyqtSignal(str)
    delete_requested = pyqtSignal(str)

    def __init__(self, model_key: str, is_installed: bool, parent=None):
        super().__init__(parent)
        self._model_key = model_key
        info = MODEL_REGISTRY[model_key]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        top_row = QHBoxLayout()
        status = "✅ " if is_installed else ""
        name_label = QLabel(f"<b>{status}{info.display_name}</b>")
        top_row.addWidget(name_label)
        top_row.addStretch()

        size_label = QLabel(f"{info.size_mb} MB")
        size_label.setStyleSheet("color: gray;")
        top_row.addWidget(size_label)

        self._action_btn = QPushButton("删除" if is_installed else "下载")
        if is_installed:
            self._action_btn.clicked.connect(lambda: self.delete_requested.emit(self._model_key))
        else:
            self._action_btn.clicked.connect(lambda: self.download_requested.emit(self._model_key))
        top_row.addWidget(self._action_btn)
        layout.addLayout(top_row)

        desc_label = QLabel(info.description)
        desc_label.setStyleSheet("color: #555;")
        layout.addWidget(desc_label)

        use_label = QLabel(f"适用：{info.use_case}")
        use_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(use_label)

    def set_enabled_action(self, enabled: bool):
        self._action_btn.setEnabled(enabled)


class ModelTab(QWidget):
    """Model management tab: list, download, delete models."""

    models_changed = pyqtSignal()

    def __init__(self, models_dir: str, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._downloader = ModelDownloader(self._models_dir)
        self._download_worker: DownloadWorker | None = None
        self._cards: dict[str, ModelCard] = {}

        self._init_ui()
        self._refresh_cards()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        self._no_model_banner = QLabel("⚠ 请先下载至少一个模型才能开始处理")
        self._no_model_banner.setStyleSheet(
            "background: #FFF3CD; color: #856404; padding: 8px; border-radius: 4px;"
        )
        self._no_model_banner.setVisible(False)
        layout.addWidget(self._no_model_banner)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setSpacing(8)
        self._card_layout.addStretch()
        scroll.setWidget(self._card_container)
        layout.addWidget(scroll, stretch=1)

        self._progress_widget = QWidget()
        progress_layout = QVBoxLayout(self._progress_widget)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate
        progress_layout.addWidget(self._progress_bar)
        progress_row = QHBoxLayout()
        self._progress_label = QLabel("")
        progress_row.addWidget(self._progress_label)
        progress_row.addStretch()
        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.clicked.connect(self._on_cancel_download)
        progress_row.addWidget(self._cancel_btn)
        progress_layout.addLayout(progress_row)
        self._progress_widget.setVisible(False)
        layout.addWidget(self._progress_widget)

        info_row = QHBoxLayout()
        self._source_label = QLabel("下载源: hf-mirror.com")
        self._source_label.setStyleSheet("color: gray; font-size: 11px;")
        info_row.addWidget(self._source_label)
        info_row.addStretch()
        dir_label = QLabel(f"模型目录: {self._models_dir}")
        dir_label.setStyleSheet("color: gray; font-size: 11px;")
        info_row.addWidget(dir_label)
        layout.addLayout(info_row)

    def _refresh_cards(self):
        for key, card in self._cards.items():
            self._card_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        installed = self._downloader.get_installed_models()
        for key in MODEL_REGISTRY:
            is_installed = key in installed
            card = ModelCard(key, is_installed)
            card.download_requested.connect(self._on_download_requested)
            card.delete_requested.connect(self._on_delete_requested)
            if self._download_worker is not None:
                card.set_enabled_action(False)
            self._card_layout.insertWidget(self._card_layout.count() - 1, card)
            self._cards[key] = card

        self._no_model_banner.setVisible(len(installed) == 0)

    def _on_download_requested(self, model_key: str):
        if self._download_worker is not None:
            return
        self._download_worker = DownloadWorker(self._downloader, model_key)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)

        for card in self._cards.values():
            card.set_enabled_action(False)

        self._progress_widget.setVisible(True)
        info = MODEL_REGISTRY[model_key]
        self._progress_label.setText(f"正在下载 {info.display_name}...")
        self._download_worker.start()

    def _on_download_progress(self, text: str):
        self._progress_label.setText(text)

    def _on_download_finished(self, model_key: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()
        self.models_changed.emit()

    def _on_download_error(self, model_key: str, error_msg: str):
        self._download_worker = None
        self._progress_widget.setVisible(False)
        self._refresh_cards()
        QMessageBox.critical(
            self, "下载失败",
            f"下载 {MODEL_REGISTRY[model_key].display_name} 失败:\n{error_msg}",
        )

    def _on_cancel_download(self):
        if self._download_worker and self._download_worker.isRunning():
            self._download_worker.terminate()
            self._download_worker.wait()
            self._download_worker = None
            self._progress_widget.setVisible(False)
            self._refresh_cards()

    def _on_delete_requested(self, model_key: str):
        installed = self._downloader.get_installed_models()
        if len(installed) <= 1 and model_key in installed:
            QMessageBox.warning(self, "无法删除", "至少保留一个模型")
            return

        info = MODEL_REGISTRY[model_key]
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定删除 {info.display_name}？模型文件将被移除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._downloader.delete_model(model_key)
            self._refresh_cards()
            self.models_changed.emit()

    def has_any_model(self) -> bool:
        return len(self._downloader.get_installed_models()) > 0

# src/gui/settings_tab.py
"""Settings tab: data directory, cache management, download source."""

import os

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.cache import MaskCacheManager, format_size
from src.core.data_dir import (
    get_cache_dir,
    get_settings_path,
    resolve_data_dir,
    save_config,
)
from src.core.paths import get_app_root
from src.core.settings import AppSettings, load_settings, save_settings
from src.core.model_downloader import ENDPOINTS


class CacheSizeWorker(QThread):
    """Calculate cache directory size in background."""

    result = pyqtSignal(int)

    def __init__(self, cache_dir: str):
        super().__init__()
        self._cache_dir = cache_dir

    def run(self):
        cache = MaskCacheManager(self._cache_dir)
        self.result.emit(cache.get_total_size())


class SettingsTab(QWidget):
    """Settings tab with data directory, cache management, and download settings."""

    download_source_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._settings = load_settings(get_settings_path())
        self._cache_worker: CacheSizeWorker | None = None
        self._init_ui()
        self._load_current_values()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Data Directory ---
        data_group = QGroupBox("数据目录")
        data_layout = QVBoxLayout(data_group)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("数据存放位置:"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItem("用户目录 (默认)")
        self._dir_combo.addItem("应用目录")
        self._dir_combo.addItem("自定义")
        self._dir_combo.currentIndexChanged.connect(self._on_dir_mode_changed)
        dir_row.addWidget(self._dir_combo, stretch=1)
        data_layout.addLayout(dir_row)

        custom_row = QHBoxLayout()
        self._custom_dir_edit = QLineEdit()
        self._custom_dir_edit.setPlaceholderText("选择自定义路径...")
        self._custom_dir_edit.setReadOnly(True)
        custom_row.addWidget(self._custom_dir_edit)
        self._browse_btn = QPushButton("浏览...")
        self._browse_btn.clicked.connect(self._on_browse_dir)
        custom_row.addWidget(self._browse_btn)
        self._custom_dir_widget = QWidget()
        self._custom_dir_widget.setLayout(custom_row)
        self._custom_dir_widget.setVisible(False)
        data_layout.addWidget(self._custom_dir_widget)

        self._current_dir_label = QLabel("")
        self._current_dir_label.setStyleSheet("color: gray; font-size: 11px;")
        data_layout.addWidget(self._current_dir_label)

        self._dir_apply_btn = QPushButton("应用并重启")
        self._dir_apply_btn.setEnabled(False)
        self._dir_apply_btn.clicked.connect(self._on_apply_data_dir)
        data_layout.addWidget(self._dir_apply_btn)

        dir_warning = QLabel("⚠ 修改数据目录后需重启生效，旧数据不会自动迁移")
        dir_warning.setStyleSheet("color: #856404; font-size: 11px;")
        data_layout.addWidget(dir_warning)

        layout.addWidget(data_group)

        # --- Cache Management ---
        cache_group = QGroupBox("缓存管理")
        cache_layout = QVBoxLayout(cache_group)

        self._cache_dir_label = QLabel("")
        self._cache_dir_label.setStyleSheet("color: gray; font-size: 11px;")
        cache_layout.addWidget(self._cache_dir_label)

        self._cache_size_label = QLabel("占用空间: 计算中...")
        cache_layout.addWidget(self._cache_size_label)

        btn_row = QHBoxLayout()
        self._clean_btn = QPushButton("清理全部缓存")
        self._clean_btn.clicked.connect(self._on_clean_cache)
        btn_row.addWidget(self._clean_btn)
        self._open_dir_btn = QPushButton("打开目录")
        self._open_dir_btn.clicked.connect(self._on_open_cache_dir)
        btn_row.addWidget(self._open_dir_btn)
        btn_row.addStretch()
        cache_layout.addLayout(btn_row)

        cache_warning = QLabel("⚠ 清理缓存将删除所有断点续传进度")
        cache_warning.setStyleSheet("color: #856404; font-size: 11px;")
        cache_layout.addWidget(cache_warning)

        layout.addWidget(cache_group)

        # --- Download Settings ---
        dl_group = QGroupBox("下载设置")
        dl_layout = QVBoxLayout(dl_group)

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("下载源:"))
        self._dl_combo = QComboBox()
        self._dl_combo.addItem("hf-mirror.com (推荐)", "hf-mirror")
        self._dl_combo.addItem("huggingface.co (官方)", "huggingface")
        self._dl_combo.addItem("自定义", "custom")
        self._dl_combo.currentIndexChanged.connect(self._on_dl_source_changed)
        src_row.addWidget(self._dl_combo, stretch=1)
        dl_layout.addLayout(src_row)

        self._custom_url_edit = QLineEdit()
        self._custom_url_edit.setPlaceholderText("https://your-mirror.com")
        self._custom_url_edit.textChanged.connect(self._on_custom_url_changed)
        self._custom_url_widget = QWidget()
        url_layout = QHBoxLayout(self._custom_url_widget)
        url_layout.setContentsMargins(0, 0, 0, 0)
        url_layout.addWidget(QLabel("自定义地址:"))
        url_layout.addWidget(self._custom_url_edit)
        self._custom_url_widget.setVisible(False)
        dl_layout.addWidget(self._custom_url_widget)

        layout.addWidget(dl_group)

        layout.addStretch()

    def _load_current_values(self):
        # Data directory
        current = resolve_data_dir()
        self._current_dir_label.setText(f"当前生效: {current}")

        default_dir = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        app_dir = os.path.join(get_app_root(), "data")
        if current == default_dir:
            self._dir_combo.setCurrentIndex(0)
        elif current == app_dir:
            self._dir_combo.setCurrentIndex(1)
        else:
            self._dir_combo.setCurrentIndex(2)
            self._custom_dir_edit.setText(current)

        # Cache
        cache_dir = get_cache_dir()
        self._cache_dir_label.setText(f"缓存目录: {cache_dir}")
        self._refresh_cache_size()

        # Download source
        source = self._settings.download_source
        for i in range(self._dl_combo.count()):
            if self._dl_combo.itemData(i) == source:
                self._dl_combo.setCurrentIndex(i)
                break
        self._custom_url_edit.setText(self._settings.custom_endpoint)
        self._custom_url_widget.setVisible(source == "custom")

    def _on_dir_mode_changed(self, index: int):
        self._custom_dir_widget.setVisible(index == 2)
        self._dir_apply_btn.setEnabled(True)

    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if path:
            self._custom_dir_edit.setText(path)

    def _on_apply_data_dir(self):
        index = self._dir_combo.currentIndex()
        if index == 0:
            new_dir = os.path.join(os.path.expanduser("~"), ".birefnet-gui")
        elif index == 1:
            new_dir = os.path.join(get_app_root(), "data")
        else:
            new_dir = self._custom_dir_edit.text()
            if not new_dir:
                QMessageBox.warning(self, "提示", "请输入或选择自定义路径")
                return

        old_dir = resolve_data_dir()

        # Save config.json to the new data_dir
        save_config(new_dir)

        # If app-root mode, also write to app root so it's found on next launch
        if index == 1:
            app_config_path = os.path.join(get_app_root(), "config.json")
            save_config(new_dir, config_path=app_config_path)

        QMessageBox.information(
            self, "重启生效",
            f"数据目录已设为:\n{new_dir}\n\n请重启应用使设置生效。\n旧数据不会自动迁移，如需保留请手动复制。",
        )
        self._dir_apply_btn.setEnabled(False)
        self._current_dir_label.setText(f"当前生效: {old_dir} (重启后: {new_dir})")

    # --- Cache ---
    def _refresh_cache_size(self):
        self._cache_size_label.setText("占用空间: 计算中...")
        self._cache_worker = CacheSizeWorker(get_cache_dir())
        self._cache_worker.result.connect(self._on_cache_size_ready)
        self._cache_worker.start()

    def _on_cache_size_ready(self, size_bytes: int):
        self._cache_size_label.setText(f"占用空间: {format_size(size_bytes)}")
        self._cache_worker = None

    def _on_clean_cache(self):
        reply = QMessageBox.question(
            self, "确认清理",
            "确定清理全部缓存？\n\n这将删除所有断点续传进度，正在运行的任务不受影响。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            cache = MaskCacheManager(get_cache_dir())
            cache.cleanup_all()
            self._refresh_cache_size()

    def _on_open_cache_dir(self):
        cache_dir = get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(cache_dir))

    # --- Download Source ---
    def _on_dl_source_changed(self, index: int):
        source = self._dl_combo.itemData(index)
        self._custom_url_widget.setVisible(source == "custom")
        self._settings.download_source = source
        self._save_settings()
        self.download_source_changed.emit()

    def _on_custom_url_changed(self, text: str):
        self._settings.custom_endpoint = text
        self._save_settings()
        self.download_source_changed.emit()

    def _save_settings(self):
        save_settings(self._settings, get_settings_path())

    def get_download_endpoint(self) -> str | None:
        """Return the resolved download endpoint URL, or None for default behavior."""
        source = self._settings.download_source
        if source == "custom":
            url = self._settings.custom_endpoint.strip()
            return url if url else None
        return ENDPOINTS.get(source)

    def get_download_source_display(self) -> str:
        """Return display text for the current download source."""
        source = self._settings.download_source
        if source == "hf-mirror":
            return "hf-mirror.com"
        elif source == "huggingface":
            return "huggingface.co"
        else:
            return self._settings.custom_endpoint or "自定义"

    def showEvent(self, event):
        """Refresh cache size when tab becomes visible."""
        super().showEvent(event)
        self._refresh_cache_size()

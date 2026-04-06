import os

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    InputType,
    MODELS,
    OutputFormat,
    ProcessingConfig,
)
from src.core.device_info import DeviceInfo, estimate_vram_gb, get_device_info


FORMAT_LABELS = {
    OutputFormat.MOV_PRORES: "MOV ProRes 4444",
    OutputFormat.WEBM_VP9: "WebM VP9",
    OutputFormat.MP4_H264: "MP4 H.264",
    OutputFormat.MP4_H265: "MP4 H.265/HEVC",
    OutputFormat.MP4_AV1: "MP4 AV1",
    OutputFormat.PNG_SEQUENCE: "PNG 序列",
    OutputFormat.TIFF_SEQUENCE: "TIFF 序列",
}

MODE_LABELS = {
    BackgroundMode.TRANSPARENT: "透明背景",
    BackgroundMode.GREEN: "绿幕",
    BackgroundMode.BLUE: "蓝幕",
    BackgroundMode.MASK_BW: "黑底白蒙版",
    BackgroundMode.MASK_WB: "白底黑蒙版",
    BackgroundMode.SIDE_BY_SIDE: "原图+蒙版分轨",
}

BITRATE_LABELS = {
    BitrateMode.AUTO: "自动",
    BitrateMode.LOW: "低",
    BitrateMode.MEDIUM: "中",
    BitrateMode.HIGH: "高",
    BitrateMode.VERY_HIGH: "极高",
    BitrateMode.CUSTOM: "自定义",
}

PRORES_PROFILE_LABELS = {
    BitrateMode.AUTO: "HQ (默认)",
    BitrateMode.LOW: "Proxy",
    BitrateMode.MEDIUM: "LT",
    BitrateMode.HIGH: "Standard",
    BitrateMode.VERY_HIGH: "HQ",
}

RESOLUTION_LABELS = {
    InferenceResolution.RES_512: "512×512 (快速)",
    InferenceResolution.RES_1024: "1024×1024 (默认)",
    InferenceResolution.RES_2048: "2048×2048 (高质量)",
}

PRESET_LABELS = {
    EncodingPreset.ULTRAFAST: "ultrafast (最快)",
    EncodingPreset.SUPERFAST: "superfast",
    EncodingPreset.VERYFAST: "veryfast",
    EncodingPreset.FASTER: "faster",
    EncodingPreset.FAST: "fast",
    EncodingPreset.MEDIUM: "medium (默认)",
    EncodingPreset.SLOW: "slow",
    EncodingPreset.SLOWER: "slower",
    EncodingPreset.VERYSLOW: "veryslow (最慢)",
}

ENCODER_LABELS = {
    EncoderType.AUTO: "自动检测",
    EncoderType.SOFTWARE: "软件编码",
    EncoderType.NVENC: "NVIDIA NVENC",
    EncoderType.VIDEOTOOLBOX: "Apple VideoToolbox",
    EncoderType.QSV: "Intel QSV",
    EncoderType.AMF: "AMD AMF",
}


class SettingsPanel(QWidget):
    """Right-side settings panel with model, output, and advanced parameters."""

    settings_changed = pyqtSignal()

    def __init__(self, models_dir: str, encoder_registry=None, parent=None):
        super().__init__(parent)
        self._models_dir = os.path.abspath(models_dir)
        self._source_bitrate_mbps = 0.0
        self._input_type: InputType | None = None
        self._device_info = get_device_info()
        self._encoder_registry = encoder_registry

        self._init_ui()
        self._update_advanced_visibility()
        self._update_vram_warning()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Model settings ---
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型:"))
        self._model_combo = QComboBox()
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        model_layout.addWidget(self._model_combo)

        self._manage_models_btn = QPushButton("管理模型...")
        self._manage_models_btn.setFlat(True)
        self._manage_models_btn.setStyleSheet("color: #0066CC; text-align: left; padding: 0;")
        model_layout.addWidget(self._manage_models_btn)

        self._device_label = QLabel()
        self._device_label.setStyleSheet("color: gray; font-size: 11px;")
        self._update_device_label()
        model_layout.addWidget(self._device_label)

        model_layout.addWidget(QLabel("推理分辨率:"))
        self._resolution_combo = QComboBox()
        for res, label in RESOLUTION_LABELS.items():
            self._resolution_combo.addItem(label, res)
        self._resolution_combo.setCurrentIndex(1)  # 1024 default
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_or_batch_changed)
        model_layout.addWidget(self._resolution_combo)

        self._batch_label = QLabel("Batch Size:")
        model_layout.addWidget(self._batch_label)
        self._batch_combo = QComboBox()
        for bs in [1, 2, 4, 8, 16]:
            self._batch_combo.addItem(str(bs), bs)
        self._set_recommended_batch_size()
        self._batch_combo.currentIndexChanged.connect(self._on_resolution_or_batch_changed)
        model_layout.addWidget(self._batch_combo)

        layout.addWidget(model_group)

        # --- Output settings ---
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout(output_group)

        output_layout.addWidget(QLabel("格式:"))
        self._format_combo = QComboBox()
        for fmt, label in FORMAT_LABELS.items():
            self._format_combo.addItem(label, fmt)
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        output_layout.addWidget(self._format_combo)

        self._encoder_label = QLabel("编码器:")
        output_layout.addWidget(self._encoder_label)
        self._encoder_combo = QComboBox()
        self._encoder_hint = QLabel("")
        self._encoder_hint.setStyleSheet("color: gray; font-size: 11px;")
        self._populate_encoder_combo()
        self._encoder_combo.currentIndexChanged.connect(self._on_encoder_changed)
        output_layout.addWidget(self._encoder_combo)
        output_layout.addWidget(self._encoder_hint)

        output_layout.addWidget(QLabel("背景:"))
        self._mode_combo = QComboBox()
        self._populate_mode_combo()
        self._mode_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        output_layout.addWidget(self._mode_combo)

        layout.addWidget(output_group)

        # --- Advanced parameters ---
        self._advanced_group = QGroupBox("高级参数")
        advanced_layout = QVBoxLayout(self._advanced_group)

        self._bitrate_label = QLabel("码率:")
        advanced_layout.addWidget(self._bitrate_label)
        bitrate_row = QHBoxLayout()
        self._bitrate_combo = QComboBox()
        self._populate_bitrate_combo()
        self._bitrate_combo.currentIndexChanged.connect(self._on_bitrate_changed)
        bitrate_row.addWidget(self._bitrate_combo, stretch=1)

        self._custom_bitrate_spin = QDoubleSpinBox()
        self._custom_bitrate_spin.setRange(0.1, 200.0)
        self._custom_bitrate_spin.setSingleStep(0.1)
        self._custom_bitrate_spin.setSuffix(" Mbps")
        self._custom_bitrate_spin.setValue(20.0)
        self._custom_bitrate_spin.setVisible(False)
        self._custom_bitrate_spin.valueChanged.connect(lambda _: self.settings_changed.emit())
        bitrate_row.addWidget(self._custom_bitrate_spin)
        advanced_layout.addLayout(bitrate_row)

        self._preset_label = QLabel("编码预设:")
        advanced_layout.addWidget(self._preset_label)
        self._preset_combo = QComboBox()
        for preset, label in PRESET_LABELS.items():
            self._preset_combo.addItem(label, preset)
        self._preset_combo.setCurrentIndex(5)  # medium
        self._preset_combo.currentIndexChanged.connect(lambda _: self.settings_changed.emit())
        advanced_layout.addWidget(self._preset_combo)

        self._temporal_fix_checkbox = QCheckBox("时序修复（减少闪烁）")
        self._temporal_fix_checkbox.setChecked(True)
        self._temporal_fix_checkbox.stateChanged.connect(lambda _: self.settings_changed.emit())
        advanced_layout.addWidget(self._temporal_fix_checkbox)

        layout.addWidget(self._advanced_group)

        # --- VRAM warning ---
        self._vram_warning = QLabel()
        self._vram_warning.setStyleSheet("color: #CC7700; font-size: 11px;")
        self._vram_warning.setWordWrap(True)
        self._vram_warning.setVisible(False)
        layout.addWidget(self._vram_warning)

        layout.addStretch()

    def _populate_model_combo(self):
        self._model_combo.clear()
        for display_name, dir_name in MODELS.items():
            model_path = os.path.join(self._models_dir, dir_name)
            if os.path.isdir(model_path):
                self._model_combo.addItem(display_name, display_name)

    def _populate_mode_combo(self):
        self._mode_combo.clear()
        current_format = self._format_combo.currentData()
        for mode, label in MODE_LABELS.items():
            if mode.needs_alpha and current_format and not current_format.supports_alpha:
                continue
            self._mode_combo.addItem(label, mode)

    def _populate_bitrate_combo(self):
        self._bitrate_combo.clear()
        br = self._source_bitrate_mbps
        is_prores = self._format_combo.currentData() == OutputFormat.MOV_PRORES

        if is_prores:
            for mode, label in PRORES_PROFILE_LABELS.items():
                self._bitrate_combo.addItem(label, mode)
        else:
            if br > 0:
                for mode, label in BITRATE_LABELS.items():
                    if mode == BitrateMode.CUSTOM:
                        self._bitrate_combo.addItem(label, mode)
                    elif mode.multiplier is not None:
                        actual = br * mode.multiplier
                        self._bitrate_combo.addItem(f"{label} ({actual:.1f} Mbps)", mode)
                    else:
                        self._bitrate_combo.addItem(label, mode)
            else:
                for mode, label in BITRATE_LABELS.items():
                    self._bitrate_combo.addItem(label, mode)

    def _update_device_label(self):
        info = self._device_info
        if info.device == "cuda":
            text = f"设备: CUDA — {info.device_name} ({info.total_vram_gb:.1f} GB, 可用 {info.available_vram_gb:.1f} GB)"
        elif info.device == "mps":
            text = f"设备: MPS — {info.device_name} (统一内存 {info.total_vram_gb:.1f} GB)"
        else:
            text = "设备: CPU（无 GPU 加速）"
        self._device_label.setText(text)

    def _set_recommended_batch_size(self):
        available = self._device_info.available_vram_gb
        if available < 3:
            recommended = 1
        elif available < 6:
            recommended = 1
        elif available < 10:
            recommended = 2
        elif available < 16:
            recommended = 4
        else:
            recommended = 8
        batch_values = [1, 2, 4, 8, 16]
        if recommended in batch_values:
            self._batch_combo.setCurrentIndex(batch_values.index(recommended))

    def _populate_encoder_combo(self):
        self._encoder_combo.blockSignals(True)
        self._encoder_combo.clear()
        fmt = self._format_combo.currentData()
        if self._encoder_registry and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265):
            available = self._encoder_registry.get_available_types(fmt)
            for enc_type in available:
                self._encoder_combo.addItem(ENCODER_LABELS.get(enc_type, enc_type.value), enc_type)
        else:
            self._encoder_combo.addItem(ENCODER_LABELS[EncoderType.AUTO], EncoderType.AUTO)
        self._encoder_combo.blockSignals(False)
        self._update_encoder_hint()

    def _on_encoder_changed(self, _index):
        self._update_encoder_hint()
        self._update_advanced_visibility()
        self.settings_changed.emit()

    def _update_encoder_hint(self):
        enc_type = self._encoder_combo.currentData()
        fmt = self._format_combo.currentData()
        if (enc_type == EncoderType.AUTO and self._encoder_registry
                and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265)):
            resolved = self._encoder_registry.resolve(fmt, EncoderType.AUTO)
            from src.core.encoder_registry import ENCODER_CANDIDATES
            candidates = ENCODER_CANDIDATES.get(fmt, {})
            for et, name in candidates.items():
                if name == resolved:
                    self._encoder_hint.setText(f"→ {ENCODER_LABELS.get(et, resolved)}")
                    self._encoder_hint.setVisible(True)
                    return
        self._encoder_hint.setVisible(False)

    def _on_format_changed(self, _index):
        self._populate_mode_combo()
        self._populate_bitrate_combo()
        self._populate_encoder_combo()
        self._update_advanced_visibility()
        self.settings_changed.emit()

    def _on_bitrate_changed(self, _index):
        mode = self._bitrate_combo.currentData()
        self._custom_bitrate_spin.setVisible(mode == BitrateMode.CUSTOM)
        self.settings_changed.emit()

    def _on_resolution_or_batch_changed(self, _index=None):
        self._update_vram_warning()
        self.settings_changed.emit()

    def _update_advanced_visibility(self):
        fmt = self._format_combo.currentData()
        is_image = self._input_type in (InputType.IMAGE, InputType.IMAGE_FOLDER)
        is_sequence = fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE)

        show_bitrate = not is_image and not is_sequence
        self._bitrate_label.setVisible(show_bitrate)
        self._bitrate_combo.setVisible(show_bitrate)
        self._custom_bitrate_spin.setVisible(
            show_bitrate and self._bitrate_combo.currentData() == BitrateMode.CUSTOM
        )

        show_preset = not is_image and fmt in (
            OutputFormat.MP4_H264, OutputFormat.MP4_H265, OutputFormat.MP4_AV1,
        )
        self._preset_label.setVisible(show_preset)
        self._preset_combo.setVisible(show_preset)

        is_video = self._input_type == InputType.VIDEO or self._input_type is None
        self._temporal_fix_checkbox.setVisible(is_video)

        self._advanced_group.setVisible(show_bitrate or show_preset or is_video)

        # Encoder visibility: only for H.264/H.265
        show_encoder = not is_image and fmt in (OutputFormat.MP4_H264, OutputFormat.MP4_H265)
        self._encoder_label.setVisible(show_encoder)
        self._encoder_combo.setVisible(show_encoder)
        self._encoder_hint.setVisible(
            show_encoder and self._encoder_hint.text() != ""
        )

        # Disable preset for VideoToolbox (no preset support)
        enc_type = self._encoder_combo.currentData()
        if show_preset and enc_type == EncoderType.VIDEOTOOLBOX:
            self._preset_combo.setEnabled(False)
        else:
            self._preset_combo.setEnabled(True)

        show_batch = self._input_type != InputType.IMAGE
        self._batch_combo.setVisible(show_batch)
        self._batch_label.setVisible(show_batch)

    def _update_vram_warning(self):
        if self._device_info.device == "cpu":
            self._vram_warning.setVisible(False)
            return

        resolution = self._resolution_combo.currentData()
        batch_size = self._batch_combo.currentData()
        if resolution is None or batch_size is None:
            return

        estimated = estimate_vram_gb(resolution.value, batch_size)
        available = self._device_info.available_vram_gb

        if available > 0 and estimated > available * 0.9:
            self._vram_warning.setText(
                f"⚠ 预计需要 ~{estimated:.1f} GB 显存，当前可用 {available:.1f} GB，可能导致内存不足"
            )
            self._vram_warning.setVisible(True)
        else:
            self._vram_warning.setVisible(False)

    def set_input_type(self, input_type: InputType | None):
        self._input_type = input_type
        self._format_combo.setEnabled(input_type == InputType.VIDEO)
        self._update_advanced_visibility()

    def set_source_bitrate(self, bitrate_mbps: float):
        self._source_bitrate_mbps = bitrate_mbps
        self._populate_bitrate_combo()

    def get_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            model_name=self._model_combo.currentData(),
            output_format=self._format_combo.currentData(),
            background_mode=self._mode_combo.currentData(),
            bitrate_mode=self._bitrate_combo.currentData() or BitrateMode.AUTO,
            custom_bitrate_mbps=self._custom_bitrate_spin.value(),
            encoding_preset=self._preset_combo.currentData() or EncodingPreset.MEDIUM,
            batch_size=self._batch_combo.currentData() or 1,
            inference_resolution=self._resolution_combo.currentData() or InferenceResolution.RES_1024,
            temporal_fix=self._temporal_fix_checkbox.isChecked(),
            encoder_type=self._encoder_combo.currentData() or EncoderType.AUTO,
        )

    def refresh_models(self):
        """Refresh model combo after download/delete."""
        current = self._model_combo.currentData()
        self._populate_model_combo()
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == current:
                self._model_combo.setCurrentIndex(i)
                return
        if self._model_combo.count() > 0:
            self._model_combo.setCurrentIndex(0)

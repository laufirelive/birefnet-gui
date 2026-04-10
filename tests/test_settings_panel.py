from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    OutputFormat,
    ProcessingConfig,
)
from src.gui.settings_panel import SettingsPanel


def test_apply_config_updates_widget_state(qtbot, tmp_path):
    panel = SettingsPanel(str(tmp_path))
    qtbot.addWidget(panel)

    cfg = ProcessingConfig(
        model_name=panel._model_combo.currentData() or "BiRefNet-general",
        output_format=OutputFormat.MP4_H264,
        background_mode=BackgroundMode.GREEN,
        bitrate_mode=BitrateMode.CUSTOM,
        custom_bitrate_mbps=12.5,
        encoding_preset=EncodingPreset.FAST,
        batch_size=4,
        inference_resolution=InferenceResolution.RES_512,
        temporal_fix=False,
        encoder_type=EncoderType.AUTO,
    )

    panel.apply_config(cfg)
    got = panel.get_config()

    assert got.output_format == OutputFormat.MP4_H264
    assert got.background_mode == BackgroundMode.GREEN
    assert got.bitrate_mode == BitrateMode.CUSTOM
    assert got.custom_bitrate_mbps == 12.5
    assert got.encoding_preset == EncodingPreset.FAST
    assert got.batch_size == 4
    assert got.inference_resolution == InferenceResolution.RES_512
    assert got.temporal_fix is False


def test_get_config_falls_back_when_combo_data_missing(qtbot, tmp_path):
    panel = SettingsPanel(str(tmp_path))
    qtbot.addWidget(panel)

    panel._mode_combo.clear()
    panel._format_combo.clear()
    panel._model_combo.clear()
    panel._encoder_combo.clear()

    cfg = panel.get_config()

    assert cfg.model_name == "BiRefNet-general"
    assert cfg.output_format == OutputFormat.MOV_PRORES
    assert cfg.background_mode == BackgroundMode.TRANSPARENT
    assert cfg.encoder_type == EncoderType.AUTO

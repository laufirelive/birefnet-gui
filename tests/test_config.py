from src.core.config import (
    BackgroundMode,
    MODELS,
    OutputFormat,
    ProcessingConfig,
)


class TestOutputFormat:
    def test_all_formats_exist(self):
        expected = {
            "mov_prores", "webm_vp9", "mp4_h264", "mp4_h265",
            "mp4_av1", "png_sequence", "tiff_sequence",
        }
        assert {f.value for f in OutputFormat} == expected

    def test_supports_alpha(self):
        alpha_formats = {
            OutputFormat.MOV_PRORES,
            OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE,
        }
        no_alpha = {
            OutputFormat.MP4_H264, OutputFormat.MP4_H265, OutputFormat.MP4_AV1,
            OutputFormat.WEBM_VP9,
        }
        for fmt in alpha_formats:
            assert fmt.supports_alpha is True
        for fmt in no_alpha:
            assert fmt.supports_alpha is False


class TestBackgroundMode:
    def test_all_modes_exist(self):
        expected = {
            "transparent", "green", "blue",
            "mask_bw", "mask_wb", "side_by_side",
        }
        assert {m.value for m in BackgroundMode} == expected

    def test_transparent_needs_alpha(self):
        assert BackgroundMode.TRANSPARENT.needs_alpha is True
        assert BackgroundMode.GREEN.needs_alpha is False
        assert BackgroundMode.MASK_BW.needs_alpha is False
        assert BackgroundMode.SIDE_BY_SIDE.needs_alpha is False


class TestModels:
    def test_models_dict_has_six_entries(self):
        assert len(MODELS) == 6

    def test_known_models(self):
        assert "BiRefNet-general" in MODELS
        assert MODELS["BiRefNet-general"] == "birefnet-general"
        assert "BiRefNet-lite" in MODELS
        assert "BiRefNet-HR" in MODELS


class TestProcessingConfig:
    def test_defaults(self):
        config = ProcessingConfig()
        assert config.model_name == "BiRefNet-general"
        assert config.output_format == OutputFormat.MOV_PRORES
        assert config.background_mode == BackgroundMode.TRANSPARENT

    def test_custom_values(self):
        config = ProcessingConfig(
            model_name="BiRefNet-lite",
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        assert config.model_name == "BiRefNet-lite"
        assert config.output_format == OutputFormat.MP4_H264
        assert config.background_mode == BackgroundMode.GREEN

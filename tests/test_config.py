from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncodingPreset,
    IMAGE_EXTENSIONS,
    InferenceResolution,
    InputType,
    ModelInfo,
    MODEL_REGISTRY,
    MODELS,
    OutputFormat,
    ProcessingConfig,
    VIDEO_EXTENSIONS,
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
            OutputFormat.MOV_PRORES, OutputFormat.WEBM_VP9,
            OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE,
        }
        no_alpha = {
            OutputFormat.MP4_H264, OutputFormat.MP4_H265, OutputFormat.MP4_AV1,
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


class TestInputType:
    def test_video_type_exists(self):
        assert InputType.VIDEO.value == "video"

    def test_image_type_exists(self):
        assert InputType.IMAGE.value == "image"

    def test_image_folder_type_exists(self):
        assert InputType.IMAGE_FOLDER.value == "image_folder"


class TestFileExtensions:
    def test_video_extensions(self):
        assert VIDEO_EXTENSIONS == {".mp4", ".avi", ".mov", ".mkv"}

    def test_image_extensions(self):
        assert IMAGE_EXTENSIONS == {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class TestModels:
    def test_models_dict_has_six_entries(self):
        assert len(MODELS) == 6

    def test_known_models(self):
        assert "BiRefNet-general" in MODELS
        assert MODELS["BiRefNet-general"] == "birefnet-general"
        assert "BiRefNet-lite" in MODELS
        assert "BiRefNet-HR" in MODELS


class TestBitrateMode:
    def test_all_modes_exist(self):
        expected = {"auto", "low", "medium", "high", "very_high", "custom"}
        assert {m.value for m in BitrateMode} == expected

    def test_multiplier(self):
        assert BitrateMode.LOW.multiplier == 0.25
        assert BitrateMode.MEDIUM.multiplier == 0.5
        assert BitrateMode.HIGH.multiplier == 1.0
        assert BitrateMode.VERY_HIGH.multiplier == 2.0
        assert BitrateMode.AUTO.multiplier == 1.0
        assert BitrateMode.CUSTOM.multiplier is None


class TestEncodingPreset:
    def test_all_presets_exist(self):
        expected = {
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        }
        assert {p.value for p in EncodingPreset} == expected

    def test_av1_cpu_used(self):
        assert EncodingPreset.ULTRAFAST.av1_cpu_used == 8
        assert EncodingPreset.MEDIUM.av1_cpu_used == 3
        assert EncodingPreset.VERYSLOW.av1_cpu_used == 0


class TestInferenceResolution:
    def test_all_resolutions_exist(self):
        assert InferenceResolution.RES_512.value == 512
        assert InferenceResolution.RES_1024.value == 1024
        assert InferenceResolution.RES_2048.value == 2048


class TestProcessingConfigExtended:
    def test_new_defaults(self):
        config = ProcessingConfig()
        assert config.bitrate_mode == BitrateMode.AUTO
        assert config.custom_bitrate_mbps == 20.0
        assert config.encoding_preset == EncodingPreset.MEDIUM
        assert config.batch_size == 1
        assert config.inference_resolution == InferenceResolution.RES_1024

    def test_custom_new_fields(self):
        config = ProcessingConfig(
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=50.0,
            encoding_preset=EncodingPreset.SLOW,
            batch_size=4,
            inference_resolution=InferenceResolution.RES_2048,
        )
        assert config.bitrate_mode == BitrateMode.CUSTOM
        assert config.custom_bitrate_mbps == 50.0
        assert config.encoding_preset == EncodingPreset.SLOW
        assert config.batch_size == 4
        assert config.inference_resolution == InferenceResolution.RES_2048


class TestModelInfo:
    def test_model_info_fields(self):
        info = MODEL_REGISTRY["general"]
        assert info.key == "general"
        assert info.dir_name == "birefnet-general"
        assert info.repo_id == "zhengpeng7/BiRefNet"
        assert info.display_name == "BiRefNet-general"
        assert isinstance(info.description, str)
        assert isinstance(info.use_case, str)
        assert info.size_mb > 0


class TestModelRegistry:
    def test_registry_has_six_models(self):
        assert len(MODEL_REGISTRY) == 6

    def test_all_expected_keys_present(self):
        expected = {"general", "lite", "hr", "matting", "hr-matting", "dynamic"}
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_all_entries_are_model_info(self):
        for key, info in MODEL_REGISTRY.items():
            assert isinstance(info, ModelInfo)
            assert info.key == key

    def test_display_names_match_models_dict(self):
        from src.core.config import MODELS
        for key, info in MODEL_REGISTRY.items():
            assert info.display_name in MODELS
            assert MODELS[info.display_name] == info.dir_name


class TestProcessingConfigTemporalFix:
    def test_temporal_fix_default_true(self):
        config = ProcessingConfig()
        assert config.temporal_fix is True

    def test_temporal_fix_can_be_disabled(self):
        config = ProcessingConfig(temporal_fix=False)
        assert config.temporal_fix is False


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

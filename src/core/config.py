from dataclasses import dataclass
from enum import Enum


class OutputFormat(Enum):
    MOV_PRORES = "mov_prores"
    WEBM_VP9 = "webm_vp9"
    MP4_H264 = "mp4_h264"
    MP4_H265 = "mp4_h265"
    MP4_AV1 = "mp4_av1"
    PNG_SEQUENCE = "png_sequence"
    TIFF_SEQUENCE = "tiff_sequence"

    @property
    def supports_alpha(self) -> bool:
        return self in {
            OutputFormat.MOV_PRORES,
            OutputFormat.WEBM_VP9,
            OutputFormat.PNG_SEQUENCE,
            OutputFormat.TIFF_SEQUENCE,
        }


class BackgroundMode(Enum):
    TRANSPARENT = "transparent"
    GREEN = "green"
    BLUE = "blue"
    MASK_BW = "mask_bw"
    MASK_WB = "mask_wb"
    SIDE_BY_SIDE = "side_by_side"

    @property
    def needs_alpha(self) -> bool:
        return self == BackgroundMode.TRANSPARENT


class InputType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    IMAGE_FOLDER = "image_folder"


class BitrateMode(Enum):
    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CUSTOM = "custom"

    @property
    def multiplier(self) -> float | None:
        return {
            BitrateMode.AUTO: 1.0,
            BitrateMode.LOW: 0.25,
            BitrateMode.MEDIUM: 0.5,
            BitrateMode.HIGH: 1.0,
            BitrateMode.VERY_HIGH: 2.0,
            BitrateMode.CUSTOM: None,
        }[self]


class EncodingPreset(Enum):
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"

    @property
    def av1_cpu_used(self) -> int:
        return {
            EncodingPreset.ULTRAFAST: 8,
            EncodingPreset.SUPERFAST: 7,
            EncodingPreset.VERYFAST: 6,
            EncodingPreset.FASTER: 5,
            EncodingPreset.FAST: 4,
            EncodingPreset.MEDIUM: 3,
            EncodingPreset.SLOW: 2,
            EncodingPreset.SLOWER: 1,
            EncodingPreset.VERYSLOW: 0,
        }[self]


class InferenceResolution(Enum):
    RES_512 = 512
    RES_1024 = 1024
    RES_2048 = 2048


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


MODELS = {
    "BiRefNet-general": "birefnet-general",
    "BiRefNet-lite": "birefnet-lite",
    "BiRefNet-matting": "birefnet-matting",
    "BiRefNet-HR": "birefnet-hr",
    "BiRefNet-HR-matting": "birefnet-hr-matting",
    "BiRefNet-dynamic": "birefnet-dynamic",
}


@dataclass
class ModelInfo:
    key: str
    dir_name: str
    repo_id: str
    display_name: str
    description: str
    use_case: str
    size_mb: int


MODEL_REGISTRY: dict[str, "ModelInfo"] = {
    "general": ModelInfo(key="general", dir_name="birefnet-general", repo_id="zhengpeng7/BiRefNet", display_name="BiRefNet-general", description="通用模型，效果均衡", use_case="大多数场景（默认推荐）", size_mb=424),
    "lite": ModelInfo(key="lite", dir_name="birefnet-lite", repo_id="zhengpeng7/BiRefNet_lite", display_name="BiRefNet-lite", description="轻量快速，精度略低", use_case="显存不足/追求速度", size_mb=210),
    "hr": ModelInfo(key="hr", dir_name="birefnet-hr", repo_id="zhengpeng7/BiRefNet_HR", display_name="BiRefNet-HR", description="高分辨率优化", use_case="4K 视频", size_mb=450),
    "matting": ModelInfo(key="matting", dir_name="birefnet-matting", repo_id="zhengpeng7/BiRefNet-matting", display_name="BiRefNet-matting", description="专注 matting，边缘细腻", use_case="人像/头发丝细节", size_mb=424),
    "hr-matting": ModelInfo(key="hr-matting", dir_name="birefnet-hr-matting", repo_id="zhengpeng7/BiRefNet_HR-matting", display_name="BiRefNet-HR-matting", description="高分辨率 + matting 结合", use_case="高分辨率人像", size_mb=450),
    "dynamic": ModelInfo(key="dynamic", dir_name="birefnet-dynamic", repo_id="zhengpeng7/BiRefNet_dynamic", display_name="BiRefNet-dynamic", description="动态分辨率输入", use_case="不同分辨率混合输入", size_mb=424),
}


FORMAT_EXTENSIONS = {
    OutputFormat.MOV_PRORES: ".mov",
    OutputFormat.WEBM_VP9: ".webm",
    OutputFormat.MP4_H264: ".mp4",
    OutputFormat.MP4_H265: ".mp4",
    OutputFormat.MP4_AV1: ".mp4",
    OutputFormat.PNG_SEQUENCE: "",
    OutputFormat.TIFF_SEQUENCE: "",
}


@dataclass
class ProcessingConfig:
    model_name: str = "BiRefNet-general"
    output_format: OutputFormat = OutputFormat.MOV_PRORES
    background_mode: BackgroundMode = BackgroundMode.TRANSPARENT
    bitrate_mode: BitrateMode = BitrateMode.AUTO
    custom_bitrate_mbps: float = 20.0
    encoding_preset: EncodingPreset = EncodingPreset.MEDIUM
    batch_size: int = 1
    inference_resolution: InferenceResolution = InferenceResolution.RES_1024
    temporal_fix: bool = True

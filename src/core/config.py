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

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


MODELS = {
    "BiRefNet-general": "birefnet-general",
    "BiRefNet-lite": "birefnet-lite",
    "BiRefNet-matting": "birefnet-matting",
    "BiRefNet-HR": "birefnet-hr",
    "BiRefNet-HR-matting": "birefnet-hr-matting",
    "BiRefNet-dynamic": "birefnet-dynamic",
}


@dataclass
class ProcessingConfig:
    model_name: str = "BiRefNet-general"
    output_format: OutputFormat = OutputFormat.MOV_PRORES
    background_mode: BackgroundMode = BackgroundMode.TRANSPARENT

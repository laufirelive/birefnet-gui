import time
import uuid
from dataclasses import dataclass
from enum import Enum

from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncodingPreset,
    InferenceResolution,
    InputType,
    OutputFormat,
    ProcessingConfig,
)


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPhase(Enum):
    INFERENCE = "inference"
    TEMPORAL_FIX = "temporal_fix"
    ENCODING = "encoding"
    DONE = "done"


@dataclass
class QueueTask:
    id: str
    input_path: str
    input_type: InputType
    config: ProcessingConfig
    output_dir: str | None
    output_path: str | None  # Resolved output path (stable across resume)
    status: TaskStatus
    progress: int
    total: int
    phase: ProcessingPhase
    error: str | None
    created_at: float

    @classmethod
    def create(
        cls,
        input_path: str,
        input_type: InputType,
        config: ProcessingConfig,
        output_dir: str | None = None,
    ) -> "QueueTask":
        return cls(
            id=uuid.uuid4().hex[:8],
            input_path=input_path,
            input_type=input_type,
            config=config,
            output_dir=output_dir,
            output_path=None,
            status=TaskStatus.PENDING,
            progress=0,
            total=0,
            phase=ProcessingPhase.INFERENCE,
            error=None,
            created_at=time.time(),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "input_path": self.input_path,
            "input_type": self.input_type.value,
            "config": {
                "model_name": self.config.model_name,
                "output_format": self.config.output_format.value,
                "background_mode": self.config.background_mode.value,
                "bitrate_mode": self.config.bitrate_mode.value,
                "custom_bitrate_mbps": self.config.custom_bitrate_mbps,
                "encoding_preset": self.config.encoding_preset.value,
                "batch_size": self.config.batch_size,
                "inference_resolution": self.config.inference_resolution.value,
                "temporal_fix": self.config.temporal_fix,
            },
            "output_dir": self.output_dir,
            "output_path": self.output_path,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "phase": self.phase.value,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueueTask":
        cfg = d["config"]
        config = ProcessingConfig(
            model_name=cfg["model_name"],
            output_format=OutputFormat(cfg["output_format"]),
            background_mode=BackgroundMode(cfg["background_mode"]),
            bitrate_mode=BitrateMode(cfg.get("bitrate_mode", "auto")),
            custom_bitrate_mbps=cfg.get("custom_bitrate_mbps", 20.0),
            encoding_preset=EncodingPreset(cfg.get("encoding_preset", "medium")),
            batch_size=cfg.get("batch_size", 1),
            inference_resolution=InferenceResolution(cfg.get("inference_resolution", 1024)),
            temporal_fix=cfg.get("temporal_fix", True),
        )
        return cls(
            id=d["id"],
            input_path=d["input_path"],
            input_type=InputType(d["input_type"]),
            config=config,
            output_dir=d.get("output_dir"),
            output_path=d.get("output_path"),
            status=TaskStatus(d["status"]),
            progress=d.get("progress", 0),
            total=d.get("total", 0),
            phase=ProcessingPhase(d.get("phase", "inference")),
            error=d.get("error"),
            created_at=d["created_at"],
        )

import time
import uuid
from dataclasses import dataclass
from enum import Enum

from src.core.config import (
    BackgroundMode,
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
    ENCODING = "encoding"
    DONE = "done"


@dataclass
class QueueTask:
    id: str
    input_path: str
    input_type: InputType
    config: ProcessingConfig
    output_dir: str | None
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
            },
            "output_dir": self.output_dir,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "phase": self.phase.value,
            "error": self.error,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueueTask":
        config = ProcessingConfig(
            model_name=d["config"]["model_name"],
            output_format=OutputFormat(d["config"]["output_format"]),
            background_mode=BackgroundMode(d["config"]["background_mode"]),
        )
        return cls(
            id=d["id"],
            input_path=d["input_path"],
            input_type=InputType(d["input_type"]),
            config=config,
            output_dir=d.get("output_dir"),
            status=TaskStatus(d["status"]),
            progress=d.get("progress", 0),
            total=d.get("total", 0),
            phase=ProcessingPhase(d.get("phase", "inference")),
            error=d.get("error"),
            created_at=d["created_at"],
        )

import json
import time

from src.core.config import BackgroundMode, InputType, OutputFormat, ProcessingConfig
from src.core.queue_task import ProcessingPhase, QueueTask, TaskStatus


class TestQueueTask:
    def test_create_default_queue_task(self):
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=ProcessingConfig(),
        )
        assert task.id  # non-empty UUID string
        assert task.input_path == "/tmp/video.mp4"
        assert task.input_type == InputType.VIDEO
        assert task.status == TaskStatus.PENDING
        assert task.progress == 0
        assert task.total == 0
        assert task.phase == ProcessingPhase.INFERENCE
        assert task.error is None
        assert task.output_dir is None
        assert task.created_at > 0

    def test_to_dict_and_from_dict_roundtrip(self):
        config = ProcessingConfig(
            model_name="BiRefNet-lite",
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        task = QueueTask.create(
            input_path="/tmp/clip.mov",
            input_type=InputType.VIDEO,
            config=config,
            output_dir="/tmp/out",
        )
        task.status = TaskStatus.PROCESSING
        task.progress = 42
        task.total = 100
        task.phase = ProcessingPhase.ENCODING

        d = task.to_dict()
        restored = QueueTask.from_dict(d)

        assert restored.id == task.id
        assert restored.input_path == task.input_path
        assert restored.input_type == InputType.VIDEO
        assert restored.config.model_name == "BiRefNet-lite"
        assert restored.config.output_format == OutputFormat.MP4_H264
        assert restored.config.background_mode == BackgroundMode.GREEN
        assert restored.output_dir == "/tmp/out"
        assert restored.status == TaskStatus.PROCESSING
        assert restored.progress == 42
        assert restored.total == 100
        assert restored.phase == ProcessingPhase.ENCODING
        assert restored.created_at == task.created_at

    def test_to_dict_is_json_serializable(self):
        task = QueueTask.create(
            input_path="/tmp/img.png",
            input_type=InputType.IMAGE,
            config=ProcessingConfig(),
        )
        json_str = json.dumps(task.to_dict())
        assert isinstance(json_str, str)

    def test_from_dict_with_unknown_fields_ignores_them(self):
        config = ProcessingConfig()
        task = QueueTask.create(
            input_path="/tmp/a.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        d["unknown_future_field"] = "whatever"
        restored = QueueTask.from_dict(d)
        assert restored.id == task.id

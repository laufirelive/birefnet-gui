import json
import time

from src.core.config import (
    BackgroundMode,
    BitrateMode,
    EncoderType,
    EncodingPreset,
    InferenceResolution,
    InputType,
    OutputFormat,
    ProcessingConfig,
)
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


class TestQueueTaskBackwardCompat:
    def test_old_brm_without_new_fields_uses_defaults(self):
        old_dict = {
            "id": "abc123",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mov_prores",
                "background_mode": "transparent",
            },
            "output_dir": None,
            "output_path": None,
            "status": "pending",
            "progress": 0,
            "total": 0,
            "phase": "inference",
            "error": None,
            "created_at": 1712200000.0,
        }
        task = QueueTask.from_dict(old_dict)
        assert task.config.bitrate_mode == BitrateMode.AUTO
        assert task.config.custom_bitrate_mbps == 20.0
        assert task.config.encoding_preset == EncodingPreset.MEDIUM
        assert task.config.batch_size == 1
        assert task.config.inference_resolution == InferenceResolution.RES_1024


class TestQueueTaskTemporalFixCompat:
    def test_old_brm_without_temporal_fix_defaults_true(self):
        old_dict = {
            "id": "tf_test",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mov_prores",
                "background_mode": "transparent",
            },
            "output_dir": None,
            "output_path": None,
            "status": "pending",
            "progress": 0,
            "total": 0,
            "phase": "inference",
            "error": None,
            "created_at": 1712200000.0,
        }
        task = QueueTask.from_dict(old_dict)
        assert task.config.temporal_fix is True

    def test_roundtrip_with_temporal_fix_false(self):
        config = ProcessingConfig(temporal_fix=False)
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        restored = QueueTask.from_dict(d)
        assert restored.config.temporal_fix is False


class TestQueueTaskNewFieldsSerialization:
    def test_roundtrip_with_new_config_fields(self):
        config = ProcessingConfig(
            model_name="BiRefNet-HR",
            output_format=OutputFormat.MP4_H265,
            background_mode=BackgroundMode.GREEN,
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=35.0,
            encoding_preset=EncodingPreset.SLOW,
            batch_size=4,
            inference_resolution=InferenceResolution.RES_2048,
        )
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        restored = QueueTask.from_dict(d)
        assert restored.config.bitrate_mode == BitrateMode.CUSTOM
        assert restored.config.custom_bitrate_mbps == 35.0
        assert restored.config.encoding_preset == EncodingPreset.SLOW
        assert restored.config.batch_size == 4
        assert restored.config.inference_resolution == InferenceResolution.RES_2048


class TestQueueTaskEncoderTypeCompat:
    def test_old_brm_without_encoder_type_defaults_auto(self):
        old_dict = {
            "id": "enc_test",
            "input_path": "/tmp/video.mp4",
            "input_type": "video",
            "config": {
                "model_name": "BiRefNet-general",
                "output_format": "mp4_h264",
                "background_mode": "green",
            },
            "output_dir": None, "output_path": None,
            "status": "pending", "progress": 0, "total": 0,
            "phase": "inference", "error": None, "created_at": 1712200000.0,
        }
        task = QueueTask.from_dict(old_dict)
        assert task.config.encoder_type == EncoderType.AUTO

    def test_roundtrip_with_encoder_type(self):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.NVENC,
        )
        task = QueueTask.create(
            input_path="/tmp/video.mp4",
            input_type=InputType.VIDEO,
            config=config,
        )
        d = task.to_dict()
        assert d["config"]["encoder_type"] == "nvenc"
        restored = QueueTask.from_dict(d)
        assert restored.config.encoder_type == EncoderType.NVENC

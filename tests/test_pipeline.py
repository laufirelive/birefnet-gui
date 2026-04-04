import os
import tempfile
import threading

import numpy as np
import pytest

from src.core.cache import MaskCacheManager
from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import get_video_info

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.isdir(os.path.join(MODELS_DIR, "birefnet-general"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_prores_transparent(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                task_id="test_prores",
                cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10
            assert info["width"] == 64
            assert info["height"] == 64
            assert len(progress_log) == 20  # 10 inference + 10 encoding

    def test_processes_video_h264_green(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.MP4_H264,
                background_mode=BackgroundMode.GREEN,
            )
            output_path = os.path.join(temp_output_dir, "output.mp4")
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="test_h264", cache=cache,
            )
            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10

    def test_processes_video_mask_bw(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.MP4_H264,
                background_mode=BackgroundMode.MASK_BW,
            )
            output_path = os.path.join(temp_output_dir, "output.mp4")
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="test_mask", cache=cache,
            )
            assert os.path.exists(output_path)

    def test_processes_png_sequence(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig(
                output_format=OutputFormat.PNG_SEQUENCE,
                background_mode=BackgroundMode.TRANSPARENT,
            )
            output_path = os.path.join(temp_output_dir, "seq_output")
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="test_seq", cache=cache,
            )
            files = sorted(os.listdir(output_path))
            assert len(files) == 10
            assert files[0] == "frame_000001.png"

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")
            cancel_event = threading.Event()
            frame_count = []
            def on_progress(current, total, phase):
                frame_count.append(current)
                if current >= 3 and phase == "inference":
                    cancel_event.set()
            pipeline = MattingPipeline(config, MODELS_DIR)
            with pytest.raises(InterruptedError):
                pipeline.process(
                    input_path=test_video_path, output_path=output_path,
                    task_id="test_cancel", cache=cache,
                    progress_callback=on_progress, cancel_event=cancel_event,
                )
            assert len(frame_count) < 20


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestTwoPhasePipeline:
    def test_infer_phase_creates_cached_masks(self, test_video_path):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)
            progress_log = []
            pipeline.infer_phase(
                input_path=test_video_path, task_id="test1", cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert cache.get_cached_count("test1") == 10
            assert all(p == "inference" for _, _, p in progress_log)
            assert progress_log[-1] == (10, 10, "inference")
            mask = cache.load_mask("test1", 0)
            assert mask.shape == (64, 64)
            assert mask.dtype == np.uint8

    def test_encode_phase_produces_video(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.infer_phase(input_path=test_video_path, task_id="test2", cache=cache)
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline.encode_phase(
                input_path=test_video_path, output_path=output_path,
                task_id="test2", cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10
            assert all(p == "encoding" for _, _, p in progress_log)

    def test_infer_phase_resumes_from_start_frame(self, test_video_path):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            pipeline = MattingPipeline(config, MODELS_DIR)
            cancel = threading.Event()
            def stop_at_5(c, t, p):
                if c >= 5:
                    cancel.set()
            try:
                pipeline.infer_phase(
                    input_path=test_video_path, task_id="resume_test", cache=cache,
                    progress_callback=stop_at_5, cancel_event=cancel,
                )
            except InterruptedError:
                pass
            first_count = cache.get_cached_count("resume_test")
            assert first_count >= 5
            pipeline.infer_phase(
                input_path=test_video_path, task_id="resume_test", cache=cache,
                start_frame=first_count,
            )
            assert cache.get_cached_count("resume_test") == 10

    def test_process_convenience_runs_both_phases(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline
        with tempfile.TemporaryDirectory() as cache_dir:
            cache = MaskCacheManager(cache_dir)
            config = ProcessingConfig()
            output_path = os.path.join(temp_output_dir, "output.mov")
            progress_log = []
            pipeline = MattingPipeline(config, MODELS_DIR)
            pipeline.process(
                input_path=test_video_path, output_path=output_path,
                task_id="conv_test", cache=cache,
                progress_callback=lambda c, t, p: progress_log.append((c, t, p)),
            )
            assert os.path.exists(output_path)
            info = get_video_info(output_path)
            assert info["frame_count"] == 10
            phases = set(p for _, _, p in progress_log)
            assert phases == {"inference", "encoding"}


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestPipelineBatchInference:
    def test_infer_phase_with_batch_size(self, test_video_path, temp_output_dir):
        from src.core.config import InferenceResolution
        from src.core.pipeline import MattingPipeline
        config = ProcessingConfig(batch_size=2, inference_resolution=InferenceResolution.RES_512)
        cache_dir = os.path.join(temp_output_dir, "cache")
        cache = MaskCacheManager(cache_dir)
        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.infer_phase(test_video_path, "test_batch", cache)
        info = get_video_info(test_video_path)
        assert cache.get_cached_count("test_batch") == info["frame_count"]

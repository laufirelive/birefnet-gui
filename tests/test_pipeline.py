import os
import threading

import numpy as np
import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import get_video_info

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.isdir(os.path.join(MODELS_DIR, "birefnet-general"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_prores_transparent(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig()  # defaults: general, prores, transparent
        output_path = os.path.join(temp_output_dir, "output.mov")
        progress_log = []

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(
            input_path=test_video_path,
            output_path=output_path,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_processes_video_h264_green(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "output.mp4")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64

    def test_processes_video_mask_bw(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.MASK_BW,
        )
        output_path = os.path.join(temp_output_dir, "output.mp4")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        assert os.path.exists(output_path)

    def test_processes_png_sequence(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig(
            output_format=OutputFormat.PNG_SEQUENCE,
            background_mode=BackgroundMode.TRANSPARENT,
        )
        output_path = os.path.join(temp_output_dir, "seq_output")

        pipeline = MattingPipeline(config, MODELS_DIR)
        pipeline.process(input_path=test_video_path, output_path=output_path)

        files = sorted(os.listdir(output_path))
        assert len(files) == 10
        assert files[0] == "frame_000001.png"

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        config = ProcessingConfig()
        output_path = os.path.join(temp_output_dir, "output.mov")
        cancel_event = threading.Event()
        frame_count = []

        def on_progress(current, total):
            frame_count.append(current)
            if current >= 3:
                cancel_event.set()

        pipeline = MattingPipeline(config, MODELS_DIR)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        assert len(frame_count) < 10

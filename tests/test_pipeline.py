import os
import threading

import numpy as np
import pytest

from src.core.video import get_video_info

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "birefnet-general")
MODEL_EXISTS = os.path.isdir(MODEL_PATH)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestMattingPipeline:
    def test_processes_video_end_to_end(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        output_path = os.path.join(temp_output_dir, "output.mov")
        progress_log = []

        def on_progress(current, total):
            progress_log.append((current, total))

        pipeline = MattingPipeline(MODEL_PATH)
        pipeline.process(
            input_path=test_video_path,
            output_path=output_path,
            progress_callback=on_progress,
        )

        # Output file exists and is valid
        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["frame_count"] == 10
        assert info["width"] == 64
        assert info["height"] == 64

        # Progress was reported for each frame
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_cancel_stops_processing(self, test_video_path, temp_output_dir):
        from src.core.pipeline import MattingPipeline

        output_path = os.path.join(temp_output_dir, "output.mov")
        cancel_event = threading.Event()
        frame_count = []

        def on_progress(current, total):
            frame_count.append(current)
            if current >= 3:
                cancel_event.set()

        pipeline = MattingPipeline(MODEL_PATH)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_video_path,
                output_path=output_path,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        # Should have stopped around frame 3-4 (not all 10)
        assert len(frame_count) < 10

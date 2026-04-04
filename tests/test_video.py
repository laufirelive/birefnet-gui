import os

import numpy as np
import pytest

from src.core.video import FrameReader, ProResWriter, get_video_info


class TestGetVideoInfo:
    def test_returns_metadata(self, test_video_path):
        info = get_video_info(test_video_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["fps"] == 30.0
        assert info["frame_count"] == 10
        assert info["duration"] > 0

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")


class TestFrameReader:
    def test_reads_all_frames(self, test_video_path):
        reader = FrameReader(test_video_path)
        frames = list(reader)
        assert len(frames) == 10

    def test_frame_shape_is_bgr(self, test_video_path):
        reader = FrameReader(test_video_path)
        frame = next(iter(reader))
        assert frame.shape == (64, 64, 3)
        assert frame.dtype == np.uint8

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            FrameReader("/nonexistent/video.mp4")


class TestProResWriter:
    def test_writes_mov_file(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "output.mov")
        writer = ProResWriter(output_path, width=64, height=64, fps=30.0)

        for i in range(5):
            rgba = np.full((64, 64, 4), fill_value=128, dtype=np.uint8)
            rgba[:, :, 3] = 255
            writer.write_frame(rgba)

        writer.close()

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        info = get_video_info(output_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["frame_count"] == 5

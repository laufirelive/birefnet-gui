import json
import os
import subprocess

import numpy as np
import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
from src.core.video import ProResWriter
from src.core.writer import FFmpegWriter, create_writer


def _has_audio_stream(filepath: str) -> bool:
    """Check if a file has an audio stream using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            filepath,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False
    info = json.loads(result.stdout)
    return any(s["codec_type"] == "audio" for s in info.get("streams", []))


class TestProResWriterAudio:
    def test_no_audio_source_produces_silent_output(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "no_audio.mov")
        writer = ProResWriter(output_path, width=64, height=64, fps=30.0, has_alpha=True)
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)
        assert not _has_audio_stream(output_path)

    def test_audio_source_copies_audio_track(
        self, test_video_with_audio_path, temp_output_dir
    ):
        output_path = os.path.join(temp_output_dir, "with_audio.mov")
        writer = ProResWriter(
            output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path, has_alpha=True,
        )
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)
        assert _has_audio_stream(output_path)

    def test_audio_source_without_audio_stream_still_works(
        self, test_video_path, temp_output_dir
    ):
        output_path = os.path.join(temp_output_dir, "no_audio_src.mov")
        writer = ProResWriter(
            output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_path, has_alpha=True,
        )
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)


class TestFFmpegWriterAudio:
    def test_h264_with_audio_source(self, test_video_with_audio_path, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "with_audio.mp4")
        writer = FFmpegWriter(
            output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p",
            audio_source=test_video_with_audio_path,
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert _has_audio_stream(output_path)

    def test_h264_without_audio_source(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "no_audio.mp4")
        writer = FFmpegWriter(
            output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert not _has_audio_stream(output_path)


class TestCreateWriterAudio:
    def test_create_writer_passes_audio_source(
        self, test_video_with_audio_path, temp_output_dir
    ):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "out.mp4")
        writer = create_writer(
            config, output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path,
        )
        with writer:
            for _ in range(5):
                frame = np.full((64, 64, 3), 128, dtype=np.uint8)
                writer.write_frame(frame)

        assert _has_audio_stream(output_path)

    def test_image_sequence_ignores_audio_source(
        self, test_video_with_audio_path, temp_output_dir
    ):
        config = ProcessingConfig(
            output_format=OutputFormat.PNG_SEQUENCE,
            background_mode=BackgroundMode.TRANSPARENT,
        )
        output_path = os.path.join(temp_output_dir, "seq")
        writer = create_writer(
            config, output_path, width=64, height=64, fps=30.0,
            audio_source=test_video_with_audio_path,
        )
        with writer:
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)

        assert os.path.isdir(output_path)

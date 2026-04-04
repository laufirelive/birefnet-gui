import json
import os
import subprocess

import numpy as np
import pytest

from src.core.video import ProResWriter


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
        writer = ProResWriter(output_path, width=64, height=64, fps=30.0)
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
            audio_source=test_video_with_audio_path,
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
            audio_source=test_video_path,
        )
        for _ in range(5):
            rgba = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(rgba)
        writer.close()

        assert os.path.exists(output_path)

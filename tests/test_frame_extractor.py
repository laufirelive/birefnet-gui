import numpy as np
import pytest

from src.core.frame_extractor import extract_frame


class TestExtractFrame:
    def test_returns_rgb_array_with_correct_shape(self, tmp_path):
        """Generate a tiny test video with ffmpeg and extract frame 0."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        # Generate a 4-frame 2fps 64x48 red video
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=red:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=0, fps=2.0, width=64, height=48)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (48, 64, 3)
        assert frame.dtype == np.uint8

    def test_frame_content_is_plausible(self, tmp_path):
        """Red video frame should have high R channel values."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=red:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=0, fps=2.0, width=64, height=48)
        # RGB: red channel should dominate
        assert frame[:, :, 0].mean() > 200  # R
        assert frame[:, :, 2].mean() < 50   # B

    def test_extract_later_frame(self, tmp_path):
        """Extracting frame 2 at 2fps should seek to t=1.0s."""
        import subprocess

        video_path = str(tmp_path / "test.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i",
                "color=c=blue:size=64x48:rate=2:duration=2",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path,
            ],
            capture_output=True, check=True,
        )

        frame = extract_frame(video_path, frame_number=2, fps=2.0, width=64, height=48)
        assert frame.shape == (48, 64, 3)

    def test_raises_on_invalid_path(self):
        with pytest.raises(RuntimeError):
            extract_frame("/nonexistent/video.mp4", frame_number=0, fps=30.0, width=64, height=48)

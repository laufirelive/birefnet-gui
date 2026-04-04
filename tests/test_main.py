# tests/test_main.py
from unittest.mock import patch

from main import check_ffmpeg


class TestCheckFfmpeg:
    def test_returns_true_when_both_found(self):
        with patch("main.shutil.which", side_effect=lambda x: f"/usr/bin/{x}"):
            assert check_ffmpeg() is True

    def test_returns_false_when_ffmpeg_missing(self):
        def which(name):
            return None if name == "ffmpeg" else f"/usr/bin/{name}"
        with patch("main.shutil.which", side_effect=which):
            assert check_ffmpeg() is False

    def test_returns_false_when_ffprobe_missing(self):
        def which(name):
            return None if name == "ffprobe" else f"/usr/bin/{name}"
        with patch("main.shutil.which", side_effect=which):
            assert check_ffmpeg() is False

    def test_returns_false_when_both_missing(self):
        with patch("main.shutil.which", return_value=None):
            assert check_ffmpeg() is False

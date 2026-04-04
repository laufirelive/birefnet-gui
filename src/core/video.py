import os
import subprocess

import cv2
import numpy as np


def _get_bitrate_mbps(path: str) -> float:
    """Get video bitrate in Mbps using ffprobe. Returns 0.0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            bps = result.stdout.strip()
            if bps.isdigit():
                return int(bps) / 1_000_000
        # Fallback: try format-level bitrate
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=bit_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            bps = result.stdout.strip()
            if bps.isdigit():
                return int(bps) / 1_000_000
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return 0.0


def get_video_info(path: str) -> dict:
    """Get video metadata: width, height, fps, frame_count, duration."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["frame_count"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    info["bitrate_mbps"] = _get_bitrate_mbps(path)
    return info


class FrameReader:
    """Iterator that yields BGR frames from a video file."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self._path = path

    def __iter__(self):
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self._path}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()


class ProResWriter:
    """Writes RGBA frames to a MOV file using ProRes 4444 via FFmpeg."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: str | None = None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend([
            "-c:v", "prores_ks",
            "-profile:v", "4444",
            "-pix_fmt", "yuva444p10le",
            "-vendor", "apl0",
        ])

        if audio_source:
            cmd.extend(["-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest"])

        cmd.append(output_path)

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write_frame(self, rgba_frame: np.ndarray):
        """Write one RGBA uint8 frame. Shape must be (height, width, 4)."""
        if rgba_frame.shape != (self._height, self._width, 4):
            raise ValueError(
                f"Expected frame shape ({self._height}, {self._width}, 4), got {rgba_frame.shape}"
            )
        self._process.stdin.write(rgba_frame.tobytes())

    def close(self):
        """Flush and close the FFmpeg process."""
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
            self._process.stdin.close()
        # Use communicate() to properly drain stderr and wait for process
        _, stderr_data = self._process.communicate()
        if self._process.returncode != 0:
            stderr = stderr_data.decode() if stderr_data else "unknown error"
            raise RuntimeError(f"FFmpeg failed (code {self._process.returncode}): {stderr}")

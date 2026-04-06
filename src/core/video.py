import os
import subprocess
import threading

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


def _has_encoder(name: str) -> bool:
    """Check if FFmpeg has a specific encoder available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=5,
        )
        return name in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class ProResWriter:
    """Writes frames to a MOV file using ProRes via FFmpeg.

    Supports both RGBA (4-channel, transparent) and RGB (3-channel) input.
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        audio_source: str | None = None,
        profile: int = 3,
        has_alpha: bool = False,
        encoder_registry=None,
    ):
        self._output_path = output_path
        self._width = width
        self._height = height
        self._channels = 4 if has_alpha else 3

        def check_encoder(name: str) -> bool:
            if encoder_registry is not None:
                return encoder_registry.is_available(name)
            return _has_encoder(name)

        # Determine best available ProRes encoder and pixel formats
        if has_alpha and check_encoder("prores_ks"):
            encoder = "prores_ks"
            input_pix_fmt = "rgba"
            output_pix_fmt = "yuva444p10le"
        elif check_encoder("prores_ks"):
            encoder = "prores_ks"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3
        elif check_encoder("prores_aw"):
            encoder = "prores_aw"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3
        else:
            encoder = "prores"
            input_pix_fmt = "rgb24"
            output_pix_fmt = "yuv422p10le"
            if profile >= 4:
                profile = 3

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend([
            "-c:v", encoder,
            "-profile:v", str(profile),
            "-pix_fmt", output_pix_fmt,
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
        # Drain stderr in background to prevent Windows pipe deadlock
        self._stderr_chunks: list[bytes] = []
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True,
        )
        self._stderr_thread.start()

    def _drain_stderr(self):
        try:
            for chunk in iter(lambda: self._process.stderr.read(4096), b""):
                self._stderr_chunks.append(chunk)
        except (OSError, ValueError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write_frame(self, frame: np.ndarray):
        """Write one frame. Shape must be (height, width, channels)."""
        expected = (self._height, self._width, self._channels)
        if frame.shape != expected:
            raise ValueError(
                f"Expected frame shape {expected}, got {frame.shape}"
            )
        self._process.stdin.write(frame.tobytes())

    def close(self):
        """Flush and close the FFmpeg process."""
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
            self._process.stdin.close()
        self._process.wait()
        self._stderr_thread.join(timeout=5)
        if self._process.returncode != 0:
            stderr = b"".join(self._stderr_chunks).decode(errors="replace")
            raise RuntimeError(f"FFmpeg failed (code {self._process.returncode}): {stderr}")

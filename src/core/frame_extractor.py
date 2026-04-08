import subprocess

import numpy as np


def extract_frame(
    input_path: str,
    frame_number: int,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Extract a single frame from a video using FFmpeg.

    Args:
        input_path: Path to the video file.
        frame_number: Zero-based frame index to extract.
        fps: Frame rate of the video.
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        RGB uint8 numpy array of shape (height, width, 3).

    Raises:
        RuntimeError: If FFmpeg fails or returns unexpected data.
    """
    time_seconds = frame_number / fps

    cmd = [
        "ffmpeg",
        "-ss", f"{time_seconds:.6f}",
        "-i", input_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=10,
        )
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found in PATH")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg timed out extracting frame")

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed (code {result.returncode}): {stderr}")

    expected_size = height * width * 3
    if len(result.stdout) < expected_size:
        raise RuntimeError(
            f"FFmpeg returned {len(result.stdout)} bytes, expected {expected_size}"
        )

    frame = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8)
    return frame.reshape((height, width, 3))

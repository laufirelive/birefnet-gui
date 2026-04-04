import os
import subprocess

import numpy as np
from PIL import Image

from src.core.config import BackgroundMode, BitrateMode, OutputFormat, ProcessingConfig
from src.core.video import ProResWriter


class FFmpegWriter:
    """Writes video frames via FFmpeg subprocess. Supports various codecs."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str,
        pix_fmt: str,
        input_pix_fmt: str = "rgb24",
        extra_args: list[str] | None = None,
        audio_source: str | None = None,
        bitrate_kbps: int | None = None,
        preset: str | None = None,
    ):
        self._width = width
        self._height = height
        self._channels = 4 if input_pix_fmt == "rgba" else 3

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", input_pix_fmt,
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
        ]
        if audio_source:
            cmd.extend(["-i", audio_source])

        cmd.extend(["-c:v", codec, "-pix_fmt", pix_fmt])

        if bitrate_kbps is not None:
            cmd.extend(["-b:v", f"{bitrate_kbps}k"])
        if preset is not None:
            if codec not in ("libaom-av1",):
                cmd.extend(["-preset", preset])

        if extra_args:
            cmd.extend(extra_args)

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

    def write_frame(self, frame: np.ndarray):
        expected = (self._height, self._width, self._channels)
        if frame.shape != expected:
            raise ValueError(f"Expected frame shape {expected}, got {frame.shape}")
        self._process.stdin.write(frame.tobytes())

    def close(self):
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.flush()
            except BrokenPipeError:
                pass
            self._process.stdin.close()
        _, stderr_data = self._process.communicate()
        if self._process.returncode != 0:
            stderr = stderr_data.decode() if stderr_data else "unknown error"
            raise RuntimeError(f"FFmpeg failed (code {self._process.returncode}): {stderr}")


class ImageSequenceWriter:
    """Writes frames as individual image files (PNG or TIFF)."""

    def __init__(self, output_dir: str, fmt: OutputFormat, has_alpha: bool):
        self._output_dir = output_dir
        self._ext = "png" if fmt == OutputFormat.PNG_SEQUENCE else "tiff"
        self._has_alpha = has_alpha
        self._frame_num = 0
        os.makedirs(output_dir, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def write_frame(self, frame: np.ndarray):
        self._frame_num += 1
        filename = f"frame_{self._frame_num:06d}.{self._ext}"
        filepath = os.path.join(self._output_dir, filename)

        if self._has_alpha and frame.shape[2] == 4:
            img = Image.fromarray(frame, "RGBA")
        else:
            img = Image.fromarray(frame, "RGB")
        img.save(filepath)

    def close(self):
        pass  # No resources to release


def _resolve_bitrate_kbps(config: ProcessingConfig, source_bitrate_mbps: float) -> int | None:
    mode = config.bitrate_mode
    if mode == BitrateMode.CUSTOM:
        return int(config.custom_bitrate_mbps * 1000)
    multiplier = mode.multiplier
    if multiplier is not None:
        return int(source_bitrate_mbps * multiplier * 1000)
    return None


def _resolve_prores_profile(config: ProcessingConfig) -> int:
    return {
        BitrateMode.AUTO: 3, BitrateMode.LOW: 0, BitrateMode.MEDIUM: 1,
        BitrateMode.HIGH: 2, BitrateMode.VERY_HIGH: 3, BitrateMode.CUSTOM: 3,
    }[config.bitrate_mode]


def create_writer(
    config: ProcessingConfig,
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str | None = None,
    source_bitrate_mbps: float = 0.0,
):
    """Factory: return the appropriate writer based on config."""
    fmt = config.output_format
    is_alpha = config.background_mode.needs_alpha

    # Side-by-side doubles width
    if config.background_mode == BackgroundMode.SIDE_BY_SIDE:
        width = width * 2

    if fmt == OutputFormat.MOV_PRORES:
        profile = _resolve_prores_profile(config)
        return ProResWriter(output_path, width, height, fps, audio_source=audio_source, profile=profile)

    bitrate_kbps = _resolve_bitrate_kbps(config, source_bitrate_mbps)
    preset = config.encoding_preset.value

    if fmt == OutputFormat.WEBM_VP9:
        if is_alpha:
            return FFmpegWriter(
                output_path, width, height, fps,
                codec="libvpx-vp9",
                pix_fmt="yuva420p",
                input_pix_fmt="rgba",
                extra_args=["-auto-alt-ref", "0"],
                audio_source=audio_source,
                bitrate_kbps=bitrate_kbps,
                preset=preset,
            )
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libvpx-vp9",
            pix_fmt="yuv420p",
            extra_args=["-auto-alt-ref", "0"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=preset,
        )

    if fmt == OutputFormat.MP4_H264:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx264",
            pix_fmt="yuv420p",
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=preset,
        )

    if fmt == OutputFormat.MP4_H265:
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libx265",
            pix_fmt="yuv420p",
            extra_args=["-tag:v", "hvc1"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
            preset=preset,
        )

    if fmt == OutputFormat.MP4_AV1:
        av1_cpu_used = config.encoding_preset.av1_cpu_used
        return FFmpegWriter(
            output_path, width, height, fps,
            codec="libaom-av1",
            pix_fmt="yuv420p",
            extra_args=["-cpu-used", str(av1_cpu_used), "-row-mt", "1"],
            audio_source=audio_source,
            bitrate_kbps=bitrate_kbps,
        )

    if fmt in (OutputFormat.PNG_SEQUENCE, OutputFormat.TIFF_SEQUENCE):
        return ImageSequenceWriter(output_path, fmt, is_alpha)

    raise ValueError(f"Unsupported output format: {fmt}")

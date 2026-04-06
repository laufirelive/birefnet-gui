"""EncoderRegistry: probes ffmpeg for available encoders and resolves
the best encoder name for a given OutputFormat + EncoderType combination.

Also provides :func:`get_encoder_args` which builds the ffmpeg codec argument
list for a given encoder name, encoding preset, and target bitrate.
"""

import subprocess
from typing import Optional

from src.core.config import EncoderType, EncodingPreset, OutputFormat

# ---------------------------------------------------------------------------
# Static data
# ---------------------------------------------------------------------------

ENCODER_CANDIDATES: dict[OutputFormat, dict[EncoderType, str]] = {
    OutputFormat.MP4_H264: {
        EncoderType.NVENC: "h264_nvenc",
        EncoderType.VIDEOTOOLBOX: "h264_videotoolbox",
        EncoderType.QSV: "h264_qsv",
        EncoderType.AMF: "h264_amf",
        EncoderType.SOFTWARE: "libx264",
    },
    OutputFormat.MP4_H265: {
        EncoderType.NVENC: "hevc_nvenc",
        EncoderType.VIDEOTOOLBOX: "hevc_videotoolbox",
        EncoderType.QSV: "hevc_qsv",
        EncoderType.AMF: "hevc_amf",
        EncoderType.SOFTWARE: "libx265",
    },
}

# Priority order used when EncoderType.AUTO is requested.
AUTO_PRIORITY: list[EncoderType] = [
    EncoderType.NVENC,
    EncoderType.VIDEOTOOLBOX,
    EncoderType.QSV,
    EncoderType.AMF,
    EncoderType.SOFTWARE,
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EncoderRegistry:
    """Queries ffmpeg at construction time to discover available encoders."""

    def __init__(self) -> None:
        self._available: set[str] = set()
        self._probe()

    # ------------------------------------------------------------------
    # Internal

    def _probe(self) -> None:
        """Run ``ffmpeg -hide_banner -encoders`` and parse encoder names."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output: str = result.stdout
        except FileNotFoundError:
            return
        except subprocess.TimeoutExpired:
            return

        for line in output.splitlines():
            parts = line.split()
            # Skip header / separator lines
            if len(parts) < 2:
                continue
            if parts[0].startswith("-"):
                continue
            # parts[0] is the capability flags (e.g. "V....."), parts[1] is name
            self._available.add(parts[1])

    # ------------------------------------------------------------------
    # Public API

    def is_available(self, encoder_name: str) -> bool:
        """Return True if *encoder_name* was found in ffmpeg's encoder list."""
        return encoder_name in self._available

    def get_available_types(self, fmt: OutputFormat) -> list[EncoderType]:
        """Return ``[AUTO, <available types...>]`` for supported formats.

        Returns an empty list for formats not present in ENCODER_CANDIDATES.
        """
        if fmt not in ENCODER_CANDIDATES:
            return []

        mapping = ENCODER_CANDIDATES[fmt]
        available = [
            enc_type
            for enc_type in AUTO_PRIORITY
            if self.is_available(mapping[enc_type])
        ]
        return [EncoderType.AUTO] + available

    def resolve(self, fmt: OutputFormat, encoder_type: EncoderType) -> str:
        """Return the ffmpeg encoder name for *fmt* and *encoder_type*.

        - For ``AUTO``: iterate AUTO_PRIORITY and return the first available
          encoder; guaranteed to return the SOFTWARE fallback at minimum.
        - For an explicit type: return that encoder if available, otherwise
          fall back to SOFTWARE.
        - Raises ``ValueError`` for formats not in ENCODER_CANDIDATES.
        """
        if fmt not in ENCODER_CANDIDATES:
            raise ValueError(
                f"OutputFormat {fmt!r} is not supported by EncoderRegistry. "
                f"Supported formats: {list(ENCODER_CANDIDATES)}"
            )

        mapping = ENCODER_CANDIDATES[fmt]
        software_encoder = mapping[EncoderType.SOFTWARE]

        if encoder_type == EncoderType.AUTO:
            for candidate in AUTO_PRIORITY:
                name = mapping[candidate]
                if self.is_available(name):
                    return name
            # Should never reach here because SOFTWARE is always in AUTO_PRIORITY,
            # but be safe.
            return software_encoder

        # Explicit type requested
        if encoder_type not in mapping:
            # Requested type not in candidates for this format → SOFTWARE
            return software_encoder

        requested_name = mapping[encoder_type]
        if self.is_available(requested_name):
            return requested_name

        # Requested encoder not available; fall back to SOFTWARE
        return software_encoder


# ---------------------------------------------------------------------------
# Preset / encoder-args helpers
# ---------------------------------------------------------------------------

# NVENC uses p1 (fastest) … p7 (slowest).
NVENC_PRESET_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "p1",
    EncodingPreset.SUPERFAST: "p2",
    EncodingPreset.VERYFAST: "p3",
    EncodingPreset.FASTER: "p4",
    EncodingPreset.FAST: "p4",
    EncodingPreset.MEDIUM: "p5",
    EncodingPreset.SLOW: "p6",
    EncodingPreset.SLOWER: "p6",
    EncodingPreset.VERYSLOW: "p7",
}
# Private alias for internal use (kept for backward compat if any code used private name)
_NVENC_PRESET_MAP = NVENC_PRESET_MAP

# QSV preset names differ from x264 names.
QSV_PRESET_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "veryfast",
    EncodingPreset.SUPERFAST: "faster",
    EncodingPreset.VERYFAST: "fast",
    EncodingPreset.FASTER: "medium",
    EncodingPreset.FAST: "medium",
    EncodingPreset.MEDIUM: "medium",
    EncodingPreset.SLOW: "slow",
    EncodingPreset.SLOWER: "slower",
    EncodingPreset.VERYSLOW: "veryslow",
}
_QSV_PRESET_MAP = QSV_PRESET_MAP

# AMF uses a -quality flag instead of -preset.
AMF_QUALITY_MAP: dict[EncodingPreset, str] = {
    EncodingPreset.ULTRAFAST: "speed",
    EncodingPreset.SUPERFAST: "speed",
    EncodingPreset.VERYFAST: "balanced",
    EncodingPreset.FASTER: "balanced",
    EncodingPreset.FAST: "balanced",
    EncodingPreset.MEDIUM: "balanced",
    EncodingPreset.SLOW: "quality",
    EncodingPreset.SLOWER: "quality",
    EncodingPreset.VERYSLOW: "quality",
}
_AMF_QUALITY_MAP = AMF_QUALITY_MAP

# Encoders that follow the standard libx264/libx265 preset names.
_SOFTWARE_ENCODERS = {"libx264", "libx265"}
_NVENC_ENCODERS = {"h264_nvenc", "hevc_nvenc"}
_VIDEOTOOLBOX_ENCODERS = {"h264_videotoolbox", "hevc_videotoolbox"}
_QSV_ENCODERS = {"h264_qsv", "hevc_qsv"}
_AMF_ENCODERS = {"h264_amf", "hevc_amf"}


def get_encoder_args(
    encoder_name: str,
    preset: EncodingPreset,
    bitrate_kbps: Optional[int],
) -> list[str]:
    """Build the ffmpeg codec argument list for *encoder_name*.

    Parameters
    ----------
    encoder_name:
        The ffmpeg encoder string (e.g. ``"libx264"``, ``"h264_nvenc"``).
    preset:
        The :class:`~src.core.config.EncodingPreset` to apply.
    bitrate_kbps:
        Target bitrate in kbps, or ``None`` to omit the ``-b:v`` flag.

    Returns
    -------
    list[str]
        List of ffmpeg argument strings ready to be appended to a command.
    """
    args: list[str] = ["-c:v", encoder_name]

    bitrate_args: list[str] = []
    if bitrate_kbps is not None:
        bitrate_args = ["-b:v", f"{bitrate_kbps}k"]

    if encoder_name in _SOFTWARE_ENCODERS:
        args += ["-preset", preset.value]
        args += bitrate_args
        if encoder_name == "libx265":
            args += ["-tag:v", "hvc1"]

    elif encoder_name in _NVENC_ENCODERS:
        nvenc_preset = NVENC_PRESET_MAP[preset]
        args += ["-preset", nvenc_preset, "-rc", "vbr", "-cq", "23"]
        args += bitrate_args

    elif encoder_name in _VIDEOTOOLBOX_ENCODERS:
        # VideoToolbox does not support -preset; use platform flags instead.
        args += bitrate_args
        args += ["-realtime", "0", "-allow_sw", "1"]

    elif encoder_name in _QSV_ENCODERS:
        qsv_preset = QSV_PRESET_MAP[preset]
        args += ["-preset", qsv_preset]
        args += bitrate_args

    elif encoder_name in _AMF_ENCODERS:
        amf_quality = AMF_QUALITY_MAP[preset]
        args += ["-quality", amf_quality]
        args += bitrate_args

    else:
        # Unknown encoder: just set bitrate if provided.
        args += bitrate_args

    return args

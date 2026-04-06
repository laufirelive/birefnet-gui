"""Tests for EncoderRegistry module."""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import EncoderType, OutputFormat
from src.core.encoder_registry import AUTO_PRIORITY, ENCODER_CANDIDATES, EncoderRegistry

# ---------------------------------------------------------------------------
# Sample ffmpeg -encoders output
# ---------------------------------------------------------------------------

FFMPEG_ENCODERS_OUTPUT = """\
Encoders:
 V..... = Video
 A..... = Audio
 S..... = Subtitle
 .F.... = Frame-level multithreading
 ..S... = Slice-level multithreading
 ...X.. = Codec is experimental
 ....B. = Supports draw_horiz_band
 .....D = Supports direct rendering method 1
 ------
 V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V..... libx265              libx265 H.265 / HEVC (codec hevc)
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
"""

FFMPEG_ENCODERS_NO_HW_OUTPUT = """\
Encoders:
 ------
 V..... libx264              libx264 H.264 / AVC
 V..... libx265              libx265 H.265 / HEVC
"""


def _make_run_result(stdout: str) -> MagicMock:
    result = MagicMock()
    result.stdout = stdout
    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry_with_nvenc():
    with patch("subprocess.run", return_value=_make_run_result(FFMPEG_ENCODERS_OUTPUT)):
        return EncoderRegistry()


@pytest.fixture()
def registry_no_hw():
    with patch("subprocess.run", return_value=_make_run_result(FFMPEG_ENCODERS_NO_HW_OUTPUT)):
        return EncoderRegistry()


# ---------------------------------------------------------------------------
# ENCODER_CANDIDATES structure
# ---------------------------------------------------------------------------


def test_encoder_candidates_has_h264():
    assert OutputFormat.MP4_H264 in ENCODER_CANDIDATES


def test_encoder_candidates_has_h265():
    assert OutputFormat.MP4_H265 in ENCODER_CANDIDATES


def test_encoder_candidates_h264_software():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H264][EncoderType.SOFTWARE] == "libx264"


def test_encoder_candidates_h265_software():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H265][EncoderType.SOFTWARE] == "libx265"


def test_encoder_candidates_h264_nvenc():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H264][EncoderType.NVENC] == "h264_nvenc"


def test_encoder_candidates_h265_nvenc():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H265][EncoderType.NVENC] == "hevc_nvenc"


def test_encoder_candidates_h264_videotoolbox():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H264][EncoderType.VIDEOTOOLBOX] == "h264_videotoolbox"


def test_encoder_candidates_h265_videotoolbox():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H265][EncoderType.VIDEOTOOLBOX] == "hevc_videotoolbox"


def test_encoder_candidates_h264_qsv():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H264][EncoderType.QSV] == "h264_qsv"


def test_encoder_candidates_h265_qsv():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H265][EncoderType.QSV] == "hevc_qsv"


def test_encoder_candidates_h264_amf():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H264][EncoderType.AMF] == "h264_amf"


def test_encoder_candidates_h265_amf():
    assert ENCODER_CANDIDATES[OutputFormat.MP4_H265][EncoderType.AMF] == "hevc_amf"


def test_all_formats_have_software_fallback():
    """Every format in ENCODER_CANDIDATES must have a SOFTWARE entry."""
    for fmt, mapping in ENCODER_CANDIDATES.items():
        assert EncoderType.SOFTWARE in mapping, f"{fmt} is missing SOFTWARE fallback"


# ---------------------------------------------------------------------------
# AUTO_PRIORITY
# ---------------------------------------------------------------------------


def test_auto_priority_order():
    assert AUTO_PRIORITY == [
        EncoderType.NVENC,
        EncoderType.VIDEOTOOLBOX,
        EncoderType.QSV,
        EncoderType.AMF,
        EncoderType.SOFTWARE,
    ]


def test_auto_priority_covers_all_hw_types():
    hw_types = {EncoderType.NVENC, EncoderType.VIDEOTOOLBOX, EncoderType.QSV, EncoderType.AMF}
    assert hw_types.issubset(set(AUTO_PRIORITY))


# ---------------------------------------------------------------------------
# _probe / is_available
# ---------------------------------------------------------------------------


def test_probe_detects_nvenc(registry_with_nvenc):
    assert registry_with_nvenc.is_available("h264_nvenc")


def test_probe_detects_software(registry_with_nvenc):
    assert registry_with_nvenc.is_available("libx264")
    assert registry_with_nvenc.is_available("libx265")


def test_probe_nvenc_absent_without_hw(registry_no_hw):
    assert not registry_no_hw.is_available("h264_nvenc")
    assert not registry_no_hw.is_available("hevc_nvenc")


def test_is_available_false_for_unknown(registry_with_nvenc):
    assert not registry_with_nvenc.is_available("nonexistent_encoder_xyz")


def test_probe_handles_file_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        reg = EncoderRegistry()
    assert not reg.is_available("libx264")


def test_probe_handles_timeout_expired():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)):
        reg = EncoderRegistry()
    assert not reg.is_available("libx264")


# ---------------------------------------------------------------------------
# get_available_types
# ---------------------------------------------------------------------------


def test_get_available_types_includes_auto(registry_with_nvenc):
    types = registry_with_nvenc.get_available_types(OutputFormat.MP4_H264)
    assert EncoderType.AUTO in types
    assert types[0] == EncoderType.AUTO


def test_get_available_types_nvenc_present(registry_with_nvenc):
    types = registry_with_nvenc.get_available_types(OutputFormat.MP4_H264)
    assert EncoderType.NVENC in types
    assert EncoderType.SOFTWARE in types


def test_get_available_types_no_hw_only_software(registry_no_hw):
    types = registry_no_hw.get_available_types(OutputFormat.MP4_H264)
    assert EncoderType.AUTO in types
    assert EncoderType.SOFTWARE in types
    assert EncoderType.NVENC not in types
    assert EncoderType.VIDEOTOOLBOX not in types
    assert EncoderType.QSV not in types
    assert EncoderType.AMF not in types


def test_get_available_types_unsupported_format_returns_empty(registry_with_nvenc):
    assert registry_with_nvenc.get_available_types(OutputFormat.WEBM_VP9) == []
    assert registry_with_nvenc.get_available_types(OutputFormat.MOV_PRORES) == []
    assert registry_with_nvenc.get_available_types(OutputFormat.PNG_SEQUENCE) == []


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_resolve_auto_picks_nvenc_when_available(registry_with_nvenc):
    encoder = registry_with_nvenc.resolve(OutputFormat.MP4_H264, EncoderType.AUTO)
    assert encoder == "h264_nvenc"


def test_resolve_auto_falls_back_to_software(registry_no_hw):
    encoder = registry_no_hw.resolve(OutputFormat.MP4_H264, EncoderType.AUTO)
    assert encoder == "libx264"


def test_resolve_auto_h265_picks_nvenc_when_available(registry_with_nvenc):
    encoder = registry_with_nvenc.resolve(OutputFormat.MP4_H265, EncoderType.AUTO)
    assert encoder == "hevc_nvenc"


def test_resolve_auto_h265_falls_back_to_software(registry_no_hw):
    encoder = registry_no_hw.resolve(OutputFormat.MP4_H265, EncoderType.AUTO)
    assert encoder == "libx265"


def test_resolve_explicit_software_returns_correct_encoder(registry_with_nvenc):
    assert registry_with_nvenc.resolve(OutputFormat.MP4_H264, EncoderType.SOFTWARE) == "libx264"
    assert registry_with_nvenc.resolve(OutputFormat.MP4_H265, EncoderType.SOFTWARE) == "libx265"


def test_resolve_explicit_nvenc_when_available(registry_with_nvenc):
    assert registry_with_nvenc.resolve(OutputFormat.MP4_H264, EncoderType.NVENC) == "h264_nvenc"


def test_resolve_explicit_nvenc_falls_back_to_software_when_unavailable(registry_no_hw):
    # NVENC not in ffmpeg output → should fall back to SOFTWARE
    encoder = registry_no_hw.resolve(OutputFormat.MP4_H264, EncoderType.NVENC)
    assert encoder == "libx264"


def test_resolve_raises_for_unsupported_format(registry_with_nvenc):
    with pytest.raises(ValueError):
        registry_with_nvenc.resolve(OutputFormat.WEBM_VP9, EncoderType.AUTO)

    with pytest.raises(ValueError):
        registry_with_nvenc.resolve(OutputFormat.MOV_PRORES, EncoderType.SOFTWARE)


# ---------------------------------------------------------------------------
# get_encoder_args
# ---------------------------------------------------------------------------


from src.core.encoder_registry import get_encoder_args  # noqa: E402
from src.core.config import EncodingPreset  # noqa: E402


class TestGetEncoderArgs:
    def test_software_h264_args(self):
        args = get_encoder_args("libx264", EncodingPreset.MEDIUM, 5000)
        assert args == ["-c:v", "libx264", "-preset", "medium", "-b:v", "5000k"]

    def test_software_h265_args(self):
        args = get_encoder_args("libx265", EncodingPreset.MEDIUM, 5000)
        assert "-tag:v" in args
        assert "hvc1" in args
        assert args == ["-c:v", "libx265", "-preset", "medium", "-b:v", "5000k", "-tag:v", "hvc1"]

    def test_nvenc_h264_args(self):
        args = get_encoder_args("h264_nvenc", EncodingPreset.MEDIUM, 5000)
        assert args == ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23", "-b:v", "5000k"]

    def test_nvenc_h265_args(self):
        args = get_encoder_args("hevc_nvenc", EncodingPreset.SLOW, 5000)
        assert "-preset" in args
        preset_idx = args.index("-preset")
        assert args[preset_idx + 1] == "p6"

    def test_videotoolbox_ignores_preset(self):
        args = get_encoder_args("h264_videotoolbox", EncodingPreset.FAST, 5000)
        assert "-preset" not in args
        assert "-realtime" in args
        assert args == ["-c:v", "h264_videotoolbox", "-b:v", "5000k", "-realtime", "0", "-allow_sw", "1"]

    def test_qsv_h264_args(self):
        args = get_encoder_args("h264_qsv", EncodingPreset.VERYFAST, 5000)
        preset_idx = args.index("-preset")
        assert args[preset_idx + 1] == "fast"

    def test_amf_h264_args(self):
        args = get_encoder_args("h264_amf", EncodingPreset.ULTRAFAST, 5000)
        quality_idx = args.index("-quality")
        assert args[quality_idx + 1] == "speed"

    def test_bitrate_none_omits_flag(self):
        args = get_encoder_args("libx264", EncodingPreset.MEDIUM, None)
        assert "-b:v" not in args

    def test_all_presets_map_for_nvenc(self):
        for preset in EncodingPreset:
            # Should not raise KeyError
            args = get_encoder_args("h264_nvenc", preset, 5000)
            assert "-preset" in args

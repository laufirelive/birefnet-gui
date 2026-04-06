import os

import numpy as np
import pytest

from src.core.config import BackgroundMode, EncoderType, OutputFormat, ProcessingConfig
from src.core.video import get_video_info
from src.core.writer import FFmpegWriter, ImageSequenceWriter, create_writer


class TestFFmpegWriter:
    def test_writes_mp4_h264(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libx264",
            pix_fmt="yuv420p",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        info = get_video_info(output_path)
        assert info["width"] == 64
        assert info["height"] == 64
        assert info["frame_count"] == 5

    def test_writes_webm_vp9_with_alpha(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.webm")
        writer = FFmpegWriter(
            output_path=output_path,
            width=64,
            height=64,
            fps=30.0,
            codec="libvpx-vp9",
            pix_fmt="yuva420p",
            input_pix_fmt="rgba",
            extra_args=["-auto-alt-ref", "0"],
        )
        for _ in range(5):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

    def test_context_manager(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        with FFmpegWriter(output_path, 64, 64, 30.0, "libx264", "yuv420p") as writer:
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        assert os.path.exists(output_path)

    def test_wrong_frame_shape_raises(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = FFmpegWriter(output_path, 64, 64, 30.0, "libx264", "yuv420p")
        with pytest.raises(ValueError):
            wrong_shape = np.full((32, 32, 3), 128, dtype=np.uint8)
            writer.write_frame(wrong_shape)
        writer.close()


class TestImageSequenceWriter:
    def test_writes_png_sequence(self, temp_output_dir):
        seq_dir = os.path.join(temp_output_dir, "png_seq")
        writer = ImageSequenceWriter(seq_dir, OutputFormat.PNG_SEQUENCE, has_alpha=True)
        for _ in range(3):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        assert os.path.exists(seq_dir)
        files = sorted(os.listdir(seq_dir))
        assert len(files) == 3
        assert files[0] == "frame_000001.png"
        assert files[2] == "frame_000003.png"

    def test_writes_tiff_sequence(self, temp_output_dir):
        seq_dir = os.path.join(temp_output_dir, "tiff_seq")
        writer = ImageSequenceWriter(seq_dir, OutputFormat.TIFF_SEQUENCE, has_alpha=False)
        for _ in range(2):
            frame = np.full((64, 64, 3), 200, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()

        files = sorted(os.listdir(seq_dir))
        assert len(files) == 2
        assert files[0] == "frame_000001.tiff"

    def test_context_manager(self, temp_output_dir):
        seq_dir = os.path.join(temp_output_dir, "ctx_seq")
        with ImageSequenceWriter(seq_dir, OutputFormat.PNG_SEQUENCE, has_alpha=False) as writer:
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        assert os.path.exists(os.path.join(seq_dir, "frame_000001.png"))


class TestCreateWriter:
    def test_prores_returns_prores_writer(self, temp_output_dir):
        from src.core.video import ProResWriter
        config = ProcessingConfig(output_format=OutputFormat.MOV_PRORES)
        output_path = os.path.join(temp_output_dir, "test.mov")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, ProResWriter)
        writer.close()

    def test_h264_returns_ffmpeg_writer(self, temp_output_dir):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()

    def test_webm_returns_ffmpeg_writer(self, temp_output_dir):
        config = ProcessingConfig(output_format=OutputFormat.WEBM_VP9)
        output_path = os.path.join(temp_output_dir, "test.webm")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()

    def test_png_sequence_returns_image_writer(self, temp_output_dir):
        seq_dir = os.path.join(temp_output_dir, "seq")
        config = ProcessingConfig(output_format=OutputFormat.PNG_SEQUENCE)
        writer = create_writer(config, seq_dir, 64, 64, 30.0)
        assert isinstance(writer, ImageSequenceWriter)
        writer.close()

    def test_tiff_sequence_returns_image_writer(self, temp_output_dir):
        seq_dir = os.path.join(temp_output_dir, "seq2")
        config = ProcessingConfig(
            output_format=OutputFormat.TIFF_SEQUENCE,
            background_mode=BackgroundMode.MASK_BW,
        )
        writer = create_writer(config, seq_dir, 64, 64, 30.0)
        assert isinstance(writer, ImageSequenceWriter)
        writer.close()


class TestFFmpegWriterBitrate:
    def test_writes_h264_with_bitrate(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test_br.mp4")
        writer = FFmpegWriter(
            output_path=output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p", bitrate_kbps=5000,
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)

    def test_writes_h264_with_preset(self, temp_output_dir):
        output_path = os.path.join(temp_output_dir, "test_preset.mp4")
        writer = FFmpegWriter(
            output_path=output_path, width=64, height=64, fps=30.0,
            codec="libx264", pix_fmt="yuv420p", preset="fast",
        )
        for _ in range(5):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)


class TestProResWriterProfile:
    def test_writes_prores_with_profile(self, temp_output_dir):
        from src.core.video import ProResWriter
        output_path = os.path.join(temp_output_dir, "test_profile.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=4, has_alpha=True)
        for _ in range(5):
            frame = np.full((64, 64, 4), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)


class TestCreateWriterAdvanced:
    def test_h264_with_bitrate_and_preset(self, temp_output_dir):
        from src.core.config import BitrateMode, EncodingPreset
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            bitrate_mode=BitrateMode.CUSTOM,
            custom_bitrate_mbps=10.0,
            encoding_preset=EncodingPreset.FAST,
        )
        output_path = os.path.join(temp_output_dir, "test_adv.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0, source_bitrate_mbps=20.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()

    def test_prores_uses_profile_not_bitrate(self, temp_output_dir):
        from src.core.config import BitrateMode
        from src.core.video import ProResWriter
        config = ProcessingConfig(
            output_format=OutputFormat.MOV_PRORES,
            bitrate_mode=BitrateMode.LOW,
        )
        output_path = os.path.join(temp_output_dir, "test_pr.mov")
        writer = create_writer(config, output_path, 64, 64, 30.0, source_bitrate_mbps=20.0)
        assert isinstance(writer, ProResWriter)
        writer.close()

    def test_prores_alpha_forces_profile_4444(self, temp_output_dir):
        """ProRes profiles 0-3 are 422 and silently drop alpha. Ensure profile >= 4 when alpha needed."""
        from unittest.mock import patch
        from src.core.config import BitrateMode
        from src.core.video import ProResWriter

        for bitrate_mode in [BitrateMode.AUTO, BitrateMode.LOW, BitrateMode.HIGH]:
            config = ProcessingConfig(
                output_format=OutputFormat.MOV_PRORES,
                background_mode=BackgroundMode.TRANSPARENT,
                bitrate_mode=bitrate_mode,
            )
            output_path = os.path.join(temp_output_dir, f"test_alpha_{bitrate_mode.value}.mov")
            with patch("src.core.writer.ProResWriter") as mock_prores:
                create_writer(config, output_path, 64, 64, 30.0)
                _, kwargs = mock_prores.call_args
                assert kwargs["profile"] >= 4, (
                    f"Alpha mode with {bitrate_mode} got profile {kwargs['profile']}, "
                    f"needs >= 4 (ProRes 4444) to preserve alpha"
                )

    def test_prores_non_alpha_allows_422_profiles(self, temp_output_dir):
        """Non-alpha modes can use any ProRes profile (0-3 are 422)."""
        from unittest.mock import patch
        from src.core.config import BitrateMode
        from src.core.video import ProResWriter

        config = ProcessingConfig(
            output_format=OutputFormat.MOV_PRORES,
            background_mode=BackgroundMode.GREEN,
            bitrate_mode=BitrateMode.LOW,  # maps to profile 0 (Proxy)
        )
        output_path = os.path.join(temp_output_dir, "test_noalpha.mov")
        with patch("src.core.writer.ProResWriter") as mock_prores:
            create_writer(config, output_path, 64, 64, 30.0)
            _, kwargs = mock_prores.call_args
            assert kwargs["profile"] == 0


class TestCreateWriterWithEncoder:
    def _make_registry(self, available_encoders: list[str]):
        from unittest.mock import patch, MagicMock
        from src.core.encoder_registry import EncoderRegistry
        mock_result = MagicMock()
        lines = ["Encoders:", " ------"]
        for enc in available_encoders:
            lines.append(f" V....D {enc}           description")
        mock_result.stdout = "\n".join(lines)
        mock_result.returncode = 0
        with patch("src.core.encoder_registry.subprocess.run", return_value=mock_result):
            return EncoderRegistry()

    def test_h264_auto_picks_nvenc_when_available(self, temp_output_dir):
        """With NVENC available, AUTO should resolve to h264_nvenc."""
        from unittest.mock import patch, MagicMock
        reg = self._make_registry(["libx264", "libx265", "h264_nvenc", "hevc_nvenc"])
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.AUTO,
        )
        output_path = os.path.join(temp_output_dir, "test.mp4")
        with patch("src.core.writer.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.read = MagicMock(return_value=b"")
            mock_popen.return_value = mock_proc
            writer = create_writer(config, output_path, 64, 64, 30.0, encoder_registry=reg)
            cmd = mock_popen.call_args[0][0]
            assert "h264_nvenc" in cmd

    def test_h264_software_explicit(self, temp_output_dir):
        from unittest.mock import patch, MagicMock
        reg = self._make_registry(["libx264", "libx265", "h264_nvenc"])
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
            encoder_type=EncoderType.SOFTWARE,
        )
        output_path = os.path.join(temp_output_dir, "test.mp4")
        with patch("src.core.writer.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_proc.stderr.read = MagicMock(return_value=b"")
            mock_popen.return_value = mock_proc
            writer = create_writer(config, output_path, 64, 64, 30.0, encoder_registry=reg)
            cmd = mock_popen.call_args[0][0]
            assert "libx264" in cmd

    def test_no_registry_uses_software(self, temp_output_dir):
        config = ProcessingConfig(
            output_format=OutputFormat.MP4_H264,
            background_mode=BackgroundMode.GREEN,
        )
        output_path = os.path.join(temp_output_dir, "test.mp4")
        writer = create_writer(config, output_path, 64, 64, 30.0)
        assert isinstance(writer, FFmpegWriter)
        writer.close()


class TestProResWriterWithRegistry:
    def test_prores_works_with_registry(self, temp_output_dir):
        from unittest.mock import patch, MagicMock
        from src.core.video import ProResWriter
        from src.core.encoder_registry import EncoderRegistry
        mock_result = MagicMock()
        mock_result.stdout = " V....D prores_ks            Apple ProRes\n V....D libx264\n"
        mock_result.returncode = 0
        with patch("src.core.encoder_registry.subprocess.run", return_value=mock_result):
            reg = EncoderRegistry()
        output_path = os.path.join(temp_output_dir, "test_reg.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=3, has_alpha=False, encoder_registry=reg)
        for _ in range(3):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)

    def test_prores_works_without_registry(self, temp_output_dir):
        from src.core.video import ProResWriter
        output_path = os.path.join(temp_output_dir, "test_noreg.mov")
        writer = ProResWriter(output_path, 64, 64, 30.0, profile=3, has_alpha=False)
        for _ in range(3):
            frame = np.full((64, 64, 3), 128, dtype=np.uint8)
            writer.write_frame(frame)
        writer.close()
        assert os.path.exists(output_path)

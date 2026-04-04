import os

import numpy as np
import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig
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

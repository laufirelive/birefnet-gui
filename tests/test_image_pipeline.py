import os
import threading

import pytest

from src.core.config import BackgroundMode, OutputFormat, ProcessingConfig

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_EXISTS = os.path.isdir(os.path.join(MODELS_DIR, "birefnet-general"))


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestImagePipeline:
    def test_single_image_transparent(self, test_image_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        result = pipeline.process(
            input_path=test_image_path,
            output_dir=temp_output_dir,
        )

        assert os.path.exists(result)
        assert result.endswith(".png")
        from PIL import Image
        img = Image.open(result)
        assert img.mode == "RGBA"
        assert img.size == (64, 64)

    def test_single_image_green_screen(self, test_image_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.GREEN,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        result = pipeline.process(
            input_path=test_image_path,
            output_dir=temp_output_dir,
        )

        assert os.path.exists(result)
        from PIL import Image
        img = Image.open(result)
        assert img.mode == "RGB"

    def test_folder_processes_all_images(self, test_image_folder, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        pipeline = ImagePipeline(config, MODELS_DIR)
        progress_log = []
        result = pipeline.process(
            input_path=test_image_folder,
            output_dir=temp_output_dir,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )

        assert os.path.isdir(result)
        output_files = [f for f in os.listdir(result) if f.endswith(".png")]
        assert len(output_files) == 3
        assert len(progress_log) == 3
        assert progress_log[-1] == (3, 3)

    def test_cancel_stops_folder_processing(self, test_image_folder, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig(
            background_mode=BackgroundMode.TRANSPARENT,
        )
        cancel_event = threading.Event()
        progress_count = []

        def on_progress(current, total):
            progress_count.append(current)
            if current >= 1:
                cancel_event.set()

        pipeline = ImagePipeline(config, MODELS_DIR)
        with pytest.raises(InterruptedError):
            pipeline.process(
                input_path=test_image_folder,
                output_dir=temp_output_dir,
                progress_callback=on_progress,
                cancel_event=cancel_event,
            )

        assert len(progress_count) < 3

    def test_empty_folder_raises(self, tmp_path, temp_output_dir):
        from src.core.image_pipeline import ImagePipeline

        config = ProcessingConfig()
        pipeline = ImagePipeline(config, MODELS_DIR)
        with pytest.raises(ValueError, match="No image files found"):
            pipeline.process(
                input_path=str(tmp_path),
                output_dir=temp_output_dir,
            )

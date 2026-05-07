import os
import threading

import numpy as np
import pytest
from PIL import Image

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


def _pipeline_without_model(monkeypatch, background_mode=BackgroundMode.TRANSPARENT):
    from src.core import image_pipeline
    from src.core.image_pipeline import ImagePipeline

    pipeline = ImagePipeline.__new__(ImagePipeline)
    pipeline._config = ProcessingConfig(background_mode=background_mode)
    pipeline._device = "cpu"
    pipeline._model = object()

    def fake_predict(model, frame, device, resolution=1024):
        return np.full(frame.shape[:2], 255, dtype=np.uint8)

    monkeypatch.setattr(image_pipeline, "predict", fake_predict)
    return pipeline


def test_single_image_falls_back_when_cv2_cannot_read_jpeg(monkeypatch, tmp_path):
    from src.core import image_pipeline

    input_path = tmp_path / "input.jfif"
    Image.new("RGB", (8, 6), (255, 0, 0)).save(input_path, "JPEG")
    monkeypatch.setattr(image_pipeline.cv2, "imread", lambda path: None)

    pipeline = _pipeline_without_model(monkeypatch)
    result = pipeline._process_single(
        str(input_path), str(tmp_path / "out"), None, None, None,
    )

    assert os.path.exists(result)
    with Image.open(result) as output:
        assert output.size == (8, 6)


def test_folder_processes_jfif_and_jpe_images(monkeypatch, tmp_path):
    pipeline = _pipeline_without_model(monkeypatch)
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "out"
    input_dir.mkdir()
    Image.new("RGB", (8, 6), (255, 0, 0)).save(input_dir / "one.jfif", "JPEG")
    Image.new("RGB", (7, 5), (0, 255, 0)).save(input_dir / "two.jpe", "JPEG")

    progress_log = []
    result = pipeline._process_folder(
        str(input_dir), str(output_dir),
        lambda current, total: progress_log.append((current, total)),
        None, None,
    )

    assert result == str(output_dir)
    assert sorted(os.listdir(output_dir)) == ["one.png", "two.png"]
    assert progress_log == [(1, 2), (2, 2)]

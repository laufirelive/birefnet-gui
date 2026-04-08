import numpy as np
import pytest

from unittest.mock import patch, MagicMock

from src.worker.preview_worker import PreviewWorker


class TestPreviewWorker:
    def test_emits_finished_with_alpha_mask(self, qtbot):
        """PreviewWorker should emit finished(np.ndarray) on success."""
        fake_alpha = np.full((48, 64), 128, dtype=np.uint8)
        fake_model = MagicMock()

        with patch("src.worker.preview_worker.detect_device", return_value="cpu"), \
             patch("src.worker.preview_worker.load_model", return_value=fake_model), \
             patch("src.worker.preview_worker.predict", return_value=fake_alpha):

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            worker = PreviewWorker(
                model_name="BiRefNet-general",
                models_dir="/fake/models",
                frame=frame,
                resolution=1024,
            )

            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start()

            result = blocker.args[0]
            assert isinstance(result, np.ndarray)
            assert result.shape == (48, 64)
            np.testing.assert_array_equal(result, fake_alpha)

    def test_emits_error_on_failure(self, qtbot):
        """PreviewWorker should emit error(str) if model loading fails."""
        with patch("src.worker.preview_worker.detect_device", return_value="cpu"), \
             patch("src.worker.preview_worker.load_model", side_effect=FileNotFoundError("Model not found")):

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            worker = PreviewWorker(
                model_name="BiRefNet-general",
                models_dir="/fake/models",
                frame=frame,
                resolution=1024,
            )

            with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
                worker.start()

            assert "Model not found" in blocker.args[0]

import os

import numpy as np
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "birefnet-general")
MODEL_EXISTS = os.path.isdir(MODEL_PATH)

from src.core.inference import detect_device, get_model_path, load_model, predict, predict_batch


class TestDetectDevice:
    def test_returns_string(self):
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestLoadModel:
    def test_loads_model(self):
        device = detect_device()
        model = load_model(MODEL_PATH, device)
        assert model is not None

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("/nonexistent/model", "cpu")


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestPredict:
    @pytest.fixture(autouse=True)
    def setup_model(self):
        self.device = detect_device()
        self.model = load_model(MODEL_PATH, self.device)

    def test_returns_alpha_mask(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        alpha = predict(self.model, frame, self.device)
        assert alpha.shape == (480, 640)
        assert alpha.dtype == np.uint8
        assert alpha.min() >= 0
        assert alpha.max() <= 255

    def test_preserves_input_resolution(self):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        alpha = predict(self.model, frame, self.device)
        assert alpha.shape == (1080, 1920)


@pytest.fixture
def loaded_model():
    device = detect_device()
    model = load_model(MODEL_PATH, device)
    return model, device


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestPredictBatch:
    def test_batch_returns_list_of_masks(self, loaded_model):
        model, device = loaded_model
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(3)]
        masks = predict_batch(model, frames, device)
        assert len(masks) == 3
        for mask in masks:
            assert mask.shape == (64, 64)
            assert mask.dtype == np.uint8

    def test_batch_size_one_matches_single(self, loaded_model):
        model, device = loaded_model
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        single_mask = predict(model, frame, device)
        batch_masks = predict_batch(model, [frame], device)
        assert len(batch_masks) == 1
        np.testing.assert_array_equal(batch_masks[0], single_mask)


@pytest.mark.skipif(not MODEL_EXISTS, reason="Model not downloaded")
class TestPredictResolution:
    def test_predict_with_512_resolution(self, loaded_model):
        model, device = loaded_model
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = predict(model, frame, device, resolution=512)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.uint8

    def test_predict_batch_with_resolution(self, loaded_model):
        model, device = loaded_model
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(2)]
        masks = predict_batch(model, frames, device, resolution=512)
        assert len(masks) == 2
        for mask in masks:
            assert mask.shape == (64, 64)


class TestGetModelPath:
    def test_returns_correct_path(self, tmp_path):
        models_dir = str(tmp_path)
        path = get_model_path("BiRefNet-general", models_dir)
        assert path == os.path.join(str(tmp_path), "birefnet-general")

    def test_unknown_model_raises(self, tmp_path):
        with pytest.raises(KeyError):
            get_model_path("NonExistentModel", str(tmp_path))

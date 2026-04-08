import gc

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.inference import detect_device, get_model_path, load_model, predict


class PreviewWorker(QThread):
    """Run single-frame matting inference in a background thread.

    Loads the model, runs prediction on one frame, then immediately
    releases the model and frees GPU memory.

    Signals:
        finished(np.ndarray): Emitted with the alpha mask (H, W) on success.
        error(str): Emitted with error message on failure.
    """

    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(
        self,
        model_name: str,
        models_dir: str,
        frame: np.ndarray,
        resolution: int,
    ):
        super().__init__()
        self._model_name = model_name
        self._models_dir = models_dir
        self._frame = frame
        self._resolution = resolution

    def run(self):
        model = None
        try:
            device = detect_device()
            model_path = get_model_path(self._model_name, self._models_dir)
            model = load_model(model_path, device)
            alpha = predict(model, self._frame, device, self._resolution)
            self.finished.emit(alpha)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

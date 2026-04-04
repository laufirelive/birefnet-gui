import threading
import time

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.config import ProcessingConfig
from src.core.pipeline import MattingPipeline


class MattingWorker(QThread):
    """Runs the matting pipeline in a background thread.

    Signals:
        progress(int, int): (current_frame, total_frames)
        speed(float): frames per second
        finished(str): output file path on success
        error(str): error message on failure
    """

    progress = pyqtSignal(int, int)
    speed = pyqtSignal(float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, config: ProcessingConfig, models_dir: str, input_path: str, output_path: str):
        super().__init__()
        self._config = config
        self._models_dir = models_dir
        self._input_path = input_path
        self._output_path = output_path

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._last_time = None

    def run(self):
        try:
            pipeline = MattingPipeline(self._config, self._models_dir)
            self._last_time = time.time()
            pipeline.process(
                input_path=self._input_path,
                output_path=self._output_path,
                progress_callback=self._on_progress,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
            )
            self.finished.emit(self._output_path)
        except InterruptedError:
            self.error.emit("Processing cancelled")
        except Exception as e:
            self.error.emit(str(e))

    def _on_progress(self, current: int, total: int):
        self.progress.emit(current, total)
        now = time.time()
        elapsed = now - self._last_time
        if elapsed > 0:
            self.speed.emit(1.0 / elapsed)
        self._last_time = now

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def cancel(self):
        self._cancel_event.set()

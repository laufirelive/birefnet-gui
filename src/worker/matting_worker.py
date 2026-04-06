import os
import threading
import time
import uuid

from PyQt6.QtCore import QThread, pyqtSignal

from src.core.cache import MaskCacheManager
from src.core.config import InputType, ProcessingConfig
from src.core.data_dir import get_cache_dir

CACHE_DIR = get_cache_dir()


class MattingWorker(QThread):
    """Runs the matting pipeline in a background thread.

    Signals:
        progress(int, int, str): (current_frame, total_frames, phase)
            phase is "inference" or "encoding" for video, "processing" for images
        speed(float): frames per second
        finished(str): output file path on success
        error(str): error message on failure
    """

    progress = pyqtSignal(int, int, str)
    speed = pyqtSignal(float)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(
        self,
        config: ProcessingConfig,
        models_dir: str,
        input_path: str,
        output_path: str,
        input_type: InputType = InputType.VIDEO,
        task_id: str | None = None,
        start_frame: int = 0,
        start_phase: str = "inference",
        cleanup_cache: bool = True,
        encoder_registry=None,
    ):
        super().__init__()
        self._config = config
        self._models_dir = models_dir
        self._input_path = input_path
        self._output_path = output_path
        self._input_type = input_type
        self._task_id = task_id or uuid.uuid4().hex[:8]
        self._start_frame = start_frame
        self._start_phase = start_phase
        self._cleanup_cache = cleanup_cache
        self._encoder_registry = encoder_registry

        self._pause_event = threading.Event()
        self._cancel_event = threading.Event()
        self._last_time = None

    @property
    def task_id(self) -> str:
        return self._task_id

    def run(self):
        try:
            self._last_time = time.time()
            if self._input_type == InputType.VIDEO:
                self._run_video()
            else:
                self._run_image()
            self.finished.emit(self._output_path)
        except InterruptedError:
            self.error.emit("Processing cancelled")
        except Exception as e:
            self.error.emit(str(e))

    def _run_video(self):
        from src.core.pipeline import MattingPipeline
        cache = MaskCacheManager(CACHE_DIR)
        pipeline = MattingPipeline(self._config, self._models_dir)
        try:
            pipeline.process(
                input_path=self._input_path,
                output_path=self._output_path,
                task_id=self._task_id,
                cache=cache,
                start_frame=self._start_frame,
                start_phase=self._start_phase,
                progress_callback=self._on_progress,
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
                encoder_registry=self._encoder_registry,
            )
        finally:
            pipeline.release()
        if self._cleanup_cache:
            cache.cleanup(self._task_id)

    def _run_image(self):
        from src.core.image_pipeline import ImagePipeline
        pipeline = ImagePipeline(self._config, self._models_dir)
        try:
            result = pipeline.process(
                input_path=self._input_path,
                output_dir=self._output_path,
                progress_callback=lambda c, t: self._on_progress(c, t, "processing"),
                pause_event=self._pause_event,
                cancel_event=self._cancel_event,
            )
            self._output_path = result
        finally:
            pipeline.release()

    def _on_progress(self, current: int, total: int, phase: str):
        self.progress.emit(current, total, phase)
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

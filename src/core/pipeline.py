import os
import threading
import time
from typing import Callable, Optional

from src.core.compositing import compose_frame
from src.core.config import ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict
from src.core.video import FrameReader, get_video_info
from src.core.writer import create_writer


class MattingPipeline:
    """Orchestrates video read -> BiRefNet inference -> compositing -> write."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Process a video file with the configured model, format, and background mode.

        Args:
            input_path: Path to input video file.
            output_path: Path for output file or directory (for image sequences).
            progress_callback: Called with (current_frame, total_frames) after each frame.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.
        """
        video_info = get_video_info(input_path)
        total_frames = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]

        writer = create_writer(self._config, output_path, width, height, fps)

        with writer:
            for frame_idx, frame in enumerate(FrameReader(input_path), start=1):
                # Check cancel
                if cancel_event and cancel_event.is_set():
                    break

                # Check pause
                if pause_event:
                    while pause_event.is_set():
                        if cancel_event and cancel_event.is_set():
                            break
                        time.sleep(0.1)
                    if cancel_event and cancel_event.is_set():
                        break

                # Inference
                alpha = predict(self._model, frame, self._device)

                # Compose output frame
                composed = compose_frame(frame, alpha, self._config.background_mode)
                writer.write_frame(composed)

                # Report progress
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        # Handle cancel cleanup
        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")

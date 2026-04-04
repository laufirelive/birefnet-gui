import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.core.inference import detect_device, load_model, predict
from src.core.video import FrameReader, ProResWriter, get_video_info


class MattingPipeline:
    """Orchestrates video read -> BiRefNet inference -> ProRes write."""

    def __init__(self, model_path: str):
        self._device = detect_device()
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_path: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        """Process a video file: extract frames, run inference, write output.

        Args:
            input_path: Path to input video file.
            output_path: Path for output MOV file.
            progress_callback: Called with (current_frame, total_frames) after each frame.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            InterruptedError: If cancel_event is set during processing.
        """
        video_info = get_video_info(input_path)
        total_frames = video_info["frame_count"]
        width = video_info["width"]
        height = video_info["height"]
        fps = video_info["fps"]

        reader = FrameReader(input_path)

        with ProResWriter(output_path, width, height, fps) as writer:
            for frame_idx, frame in enumerate(reader, start=1):
                # Check cancel
                if cancel_event and cancel_event.is_set():
                    # Clean up partial output after context manager closes
                    break

                # Check pause
                if pause_event:
                    while pause_event.is_set():
                        if cancel_event and cancel_event.is_set():
                            break
                        time.sleep(0.1)
                    # Re-check cancel after waking from pause
                    if cancel_event and cancel_event.is_set():
                        break

                # Inference
                alpha = predict(self._model, frame, self._device)

                # Compose RGBA (BGR -> RGB + alpha)
                rgba = np.dstack([frame[:, :, ::-1], alpha])
                writer.write_frame(rgba)

                # Report progress
                if progress_callback:
                    progress_callback(frame_idx, total_frames)

        # After writer is closed, handle cancel cleanup
        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")

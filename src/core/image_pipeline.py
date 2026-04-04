import os
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from src.core.compositing import compose_frame
from src.core.config import IMAGE_EXTENSIONS, MODELS, ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict


class ImagePipeline:
    """Processes single images or image folders through BiRefNet."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def process(
        self,
        input_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Process a single image or image folder.

        Args:
            input_path: Path to an image file or folder containing images.
            output_dir: Directory for output files.
            progress_callback: Called with (current, total) after each image.
            pause_event: When set, processing pauses until cleared.
            cancel_event: When set, processing stops and raises InterruptedError.

        Returns:
            Output file path (single image) or output directory path (folder).
        """
        if os.path.isdir(input_path):
            return self._process_folder(
                input_path, output_dir,
                progress_callback, pause_event, cancel_event,
            )
        else:
            return self._process_single(
                input_path, output_dir,
                progress_callback, pause_event, cancel_event,
            )

    def _process_single(
        self,
        image_path: str,
        output_dir: str,
        progress_callback, pause_event, cancel_event,
    ) -> str:
        """Process a single image file."""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")

        resolution = self._config.inference_resolution.value
        alpha = predict(self._model, frame, self._device, resolution=resolution)
        composed = compose_frame(frame, alpha, self._config.background_mode)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        model_dir_name = MODELS[self._config.model_name]
        timestamp = int(time.time() * 1000)
        output_filename = f"{base_name}_{model_dir_name}_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)

        os.makedirs(output_dir, exist_ok=True)
        self._save_png(composed, output_path)

        if progress_callback:
            progress_callback(1, 1)

        return output_path

    def _process_folder(
        self,
        folder_path: str,
        output_dir: str,
        progress_callback, pause_event, cancel_event,
    ) -> str:
        """Process all images in a folder."""
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

        if not image_files:
            raise ValueError(f"No image files found in: {folder_path}")

        # output_dir is the final output directory (caller determines the name)
        out_path = output_dir
        os.makedirs(out_path, exist_ok=True)

        total = len(image_files)
        for idx, filename in enumerate(image_files, start=1):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")

            if pause_event:
                while pause_event.is_set():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Processing cancelled by user")
                    time.sleep(0.1)

            base_name = os.path.splitext(filename)[0]
            out_file = os.path.join(out_path, f"{base_name}.png")

            # Skip already-processed images (for resume support)
            if os.path.exists(out_file):
                if progress_callback:
                    progress_callback(idx, total)
                continue

            image_path = os.path.join(folder_path, filename)
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            resolution = self._config.inference_resolution.value
            alpha = predict(self._model, frame, self._device, resolution=resolution)
            composed = compose_frame(frame, alpha, self._config.background_mode)
            self._save_png(composed, out_file)

            if progress_callback:
                progress_callback(idx, total)

        return out_path

    def _save_png(self, composed: np.ndarray, path: str):
        """Save a composed frame as PNG. RGBA for transparent mode, RGB otherwise."""
        if composed.shape[2] == 4:
            img = Image.fromarray(composed, "RGBA")
        else:
            img = Image.fromarray(composed, "RGB")
        img.save(path)

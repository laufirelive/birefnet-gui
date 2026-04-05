import os
import threading
import time
from typing import Callable, Optional

from src.core.cache import MaskCacheManager
from src.core.compositing import compose_frame
from src.core.config import ProcessingConfig
from src.core.inference import detect_device, get_model_path, load_model, predict, predict_batch
from src.core.temporal import detect_and_fix_outliers
from src.core.video import FrameReader, get_video_info
from src.core.writer import create_writer


class MattingPipeline:
    """Two-phase video matting: inference (caches masks) then encoding."""

    def __init__(self, config: ProcessingConfig, models_dir: str):
        self._config = config
        self._device = detect_device()
        model_path = get_model_path(config.model_name, models_dir)
        self._model = load_model(model_path, self._device)

    def release(self):
        """Release model and free GPU memory."""
        import gc
        import torch
        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer_phase(
        self,
        input_path: str,
        task_id: str,
        cache: MaskCacheManager,
        start_frame: int = 0,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]

        # Validate cache on resume; invalidate if source file changed
        if start_frame > 0 and not cache.validate(task_id, video_info):
            cache.cleanup(task_id)
            start_frame = 0

        cache.save_metadata(task_id, video_info)

        batch_size = self._config.batch_size
        resolution = self._config.inference_resolution.value

        batch_frames = []
        batch_indices = []

        for idx, frame in enumerate(FrameReader(input_path)):
            if idx < start_frame:
                continue
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")
            if pause_event:
                while pause_event.is_set():
                    if cancel_event and cancel_event.is_set():
                        raise InterruptedError("Processing cancelled by user")
                    time.sleep(0.1)

            batch_frames.append(frame)
            batch_indices.append(idx)

            if len(batch_frames) >= batch_size:
                masks = predict_batch(self._model, batch_frames, self._device, resolution)
                for bi, mask in zip(batch_indices, masks):
                    cache.save_mask(task_id, bi, mask)
                if progress_callback:
                    progress_callback(batch_indices[-1] + 1, total, "inference")
                batch_frames.clear()
                batch_indices.clear()

        if batch_frames:
            masks = predict_batch(self._model, batch_frames, self._device, resolution)
            for bi, mask in zip(batch_indices, masks):
                cache.save_mask(task_id, bi, mask)
            if progress_callback:
                progress_callback(batch_indices[-1] + 1, total, "inference")

    def temporal_fix_phase(
        self,
        task_id: str,
        cache: MaskCacheManager,
        total_frames: int,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> int:
        """Run temporal outlier detection and fix. Returns number of fixed frames."""
        def inner_callback(current: int, total: int):
            if cancel_event and cancel_event.is_set():
                raise InterruptedError("Processing cancelled by user")
            if progress_callback:
                progress_callback(current, total, "temporal_fix")

        return detect_and_fix_outliers(
            cache, task_id, total_frames,
            progress_callback=inner_callback,
        )

    def encode_phase(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        video_info = get_video_info(input_path)
        total = video_info["frame_count"]
        width, height, fps = video_info["width"], video_info["height"], video_info["fps"]

        source_bitrate = video_info.get("bitrate_mbps", 0.0)
        writer = create_writer(
            self._config, output_path, width, height, fps,
            audio_source=input_path,
            source_bitrate_mbps=source_bitrate,
        )

        with writer:
            for idx, frame in enumerate(FrameReader(input_path)):
                if cancel_event and cancel_event.is_set():
                    break
                if pause_event:
                    while pause_event.is_set():
                        if cancel_event and cancel_event.is_set():
                            break
                        time.sleep(0.1)
                    if cancel_event and cancel_event.is_set():
                        break

                alpha = cache.load_mask(task_id, idx)
                composed = compose_frame(frame, alpha, self._config.background_mode)
                writer.write_frame(composed)

                if progress_callback:
                    progress_callback(idx + 1, total, "encoding")

        if cancel_event and cancel_event.is_set():
            if os.path.exists(output_path) and os.path.isfile(output_path):
                os.remove(output_path)
            raise InterruptedError("Processing cancelled by user")

    def process(
        self,
        input_path: str,
        output_path: str,
        task_id: str,
        cache: MaskCacheManager,
        start_frame: int = 0,
        start_phase: str = "inference",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        pause_event: Optional[threading.Event] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        """Convenience method: run infer_phase, optional temporal_fix_phase, then encode_phase.

        *start_phase* can be ``"inference"``, ``"temporal_fix"`` or
        ``"encoding"`` to skip already-completed phases on resume.
        """
        phase_order = ["inference", "temporal_fix", "encoding"]
        start_idx = phase_order.index(start_phase) if start_phase in phase_order else 0

        if start_idx <= 0:
            self.infer_phase(
                input_path, task_id, cache, start_frame,
                progress_callback, pause_event, cancel_event,
            )
        if self._config.temporal_fix and start_idx <= 1:
            video_info = get_video_info(input_path)
            total_frames = video_info["frame_count"]
            self.temporal_fix_phase(
                task_id, cache, total_frames,
                progress_callback, cancel_event,
            )
        if start_idx <= 2:
            self.encode_phase(
                input_path, output_path, task_id, cache,
                progress_callback, pause_event, cancel_event,
            )

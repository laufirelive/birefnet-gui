from typing import Callable, Optional

import numpy as np

from src.core.cache import MaskCacheManager

DEFAULT_THRESHOLD = 0.15


def detect_and_fix_outliers(
    cache: MaskCacheManager,
    task_id: str,
    total_frames: int,
    threshold: float = DEFAULT_THRESHOLD,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Detect and repair outlier frames whose masks deviate sharply from neighbours.

    An outlier is a frame where the mask differs significantly from BOTH its
    predecessor and successor. This distinguishes genuine scene changes from
    single-frame quality drops.

    Returns the number of frames that were repaired.
    """
    if total_frames <= 1:
        if progress_callback:
            for i in range(total_frames):
                progress_callback(i + 1, total_frames)
        return 0

    masks: list[np.ndarray] = []
    for i in range(total_frames):
        masks.append(cache.load_mask(task_id, i))

    fixed_count = 0

    for i in range(total_frames):
        is_outlier = False

        if i == 0:
            diff_next = _mask_diff(masks[0], masks[1])
            if diff_next > threshold * 1.5:
                is_outlier = True
        elif i == total_frames - 1:
            diff_prev = _mask_diff(masks[i], masks[i - 1])
            if diff_prev > threshold * 1.5:
                is_outlier = True
        else:
            diff_prev = _mask_diff(masks[i], masks[i - 1])
            diff_next = _mask_diff(masks[i], masks[i + 1])
            if diff_prev > threshold and diff_next > threshold:
                is_outlier = True

        if is_outlier:
            if i == 0:
                replacement = masks[1].copy()
            elif i == total_frames - 1:
                replacement = masks[i - 1].copy()
            else:
                replacement = (
                    (masks[i - 1].astype(np.float32) + masks[i + 1].astype(np.float32)) / 2.0
                ).astype(np.uint8)

            masks[i] = replacement
            cache.save_mask(task_id, i, replacement)
            fixed_count += 1

        if progress_callback:
            progress_callback(i + 1, total_frames)

    return fixed_count


def _mask_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalised mean absolute difference between two masks (0.0-1.0)."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)

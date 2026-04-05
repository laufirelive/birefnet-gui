from typing import Callable, Optional

import numpy as np

from src.core.cache import MaskCacheManager

DEFAULT_THRESHOLD = 0.15
DEFAULT_WINDOW_RADIUS = 20


def detect_and_fix_outliers(
    cache: MaskCacheManager,
    task_id: str,
    total_frames: int,
    threshold: float = DEFAULT_THRESHOLD,
    window_radius: int = DEFAULT_WINDOW_RADIUS,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Detect and repair outlier frames using a sliding-window reference.

    Computes a running mean over ``±window_radius`` frames.  A frame whose
    diff from the window mean exceeds *threshold* is a candidate outlier.
    Consecutive candidates are grouped into runs; short runs (< half the
    window radius) are treated as outlier clusters and repaired with the
    window mean, while long runs are assumed to be scene changes and left
    untouched.

    Memory usage is O(window_radius × H × W), independent of total_frames.

    Returns the number of frames that were repaired.
    """
    if total_frames <= 1:
        if progress_callback:
            for i in range(total_frames):
                progress_callback(i + 1, total_frames)
        return 0

    radius = min(window_radius, total_frames - 1)
    min_scene_length = max(3, radius // 2)

    # --- bootstrap sliding window [0 .. min(radius, total-1)] -------------
    window: dict[int, np.ndarray] = {}
    first = cache.load_mask(task_id, 0)
    window[0] = first
    running_sum = first.astype(np.float64).copy()

    win_left = 0
    win_right = min(radius, total_frames - 1)
    for j in range(1, win_right + 1):
        m = cache.load_mask(task_id, j)
        window[j] = m
        running_sum += m.astype(np.float64)
    win_count = win_right - win_left + 1

    fixed_count = 0
    # Pending outlier candidates: (frame_idx, reference_mask | None).
    # reference_mask is stored only while the run is short enough to be an
    # outlier cluster; once the run reaches min_scene_length we discard the
    # heavy arrays to save memory (it's a scene change, not a cluster).
    pending: list[tuple[int, Optional[np.ndarray]]] = []

    for i in range(total_frames):
        desired_left = max(0, i - radius)
        desired_right = min(total_frames - 1, i + radius)

        # shrink left edge
        while win_left < desired_left:
            running_sum -= window[win_left].astype(np.float64)
            del window[win_left]
            win_left += 1
            win_count -= 1

        # expand right edge
        while win_right < desired_right:
            win_right += 1
            m = cache.load_mask(task_id, win_right)
            window[win_right] = m
            running_sum += m.astype(np.float64)
            win_count += 1

        curr = window[i]

        if win_count <= 1:
            if progress_callback:
                progress_callback(i + 1, total_frames)
            continue

        # reference = mean of window excluding current frame
        ref = (running_sum - curr.astype(np.float64)) / (win_count - 1)
        reference = np.round(ref).astype(np.uint8)

        diff = _mask_diff(curr, reference)

        if diff > threshold:
            if len(pending) < min_scene_length:
                pending.append((i, reference.copy()))
            else:
                # Run reached scene-change length → free stored references
                if len(pending) == min_scene_length:
                    pending = [(idx, None) for idx, _ in pending]
                pending.append((i, None))
        else:
            # Normal frame → flush pending run
            fixed_count += _flush_pending(
                pending, min_scene_length, cache, task_id,
                window, running_sum,
            )
            pending = []

        if progress_callback:
            progress_callback(i + 1, total_frames)

    # flush any trailing run
    fixed_count += _flush_pending(
        pending, min_scene_length, cache, task_id,
        window, running_sum,
    )

    return fixed_count


def _flush_pending(
    pending: list[tuple[int, Optional[np.ndarray]]],
    min_scene_length: int,
    cache: MaskCacheManager,
    task_id: str,
    window: dict[int, np.ndarray],
    running_sum: np.ndarray,
) -> int:
    """Fix frames in *pending* if the run is short (outlier cluster).

    Returns the number of frames fixed.
    """
    if not pending or len(pending) >= min_scene_length:
        return 0

    fixed = 0
    for idx, ref_mask in pending:
        if ref_mask is not None and idx in window:
            old_f64 = window[idx].astype(np.float64)
            new_f64 = ref_mask.astype(np.float64)
            # update running sum so subsequent references benefit
            running_sum -= old_f64
            running_sum += new_f64
            window[idx] = ref_mask
            cache.save_mask(task_id, idx, ref_mask)
            fixed += 1
    return fixed


def _mask_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalised mean absolute difference between two masks (0.0-1.0)."""
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)

import json
import os
import shutil

import cv2
import numpy as np


def format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if size_bytes >= 1_073_741_824:  # 1 GB
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    mb = size_bytes // 1_048_576
    return f"{mb} MB"


class MaskCacheManager:
    """Manages alpha mask cache on disk for breakpoint resume."""

    def __init__(self, cache_dir: str):
        self._cache_dir = cache_dir

    def save_mask(self, task_id: str, frame_idx: int, mask: np.ndarray) -> None:
        masks_dir = os.path.join(self._cache_dir, task_id, "masks")
        os.makedirs(masks_dir, exist_ok=True)
        path = os.path.join(masks_dir, f"{frame_idx:06d}.png")
        cv2.imwrite(path, mask)

    def load_mask(self, task_id: str, frame_idx: int) -> np.ndarray:
        path = os.path.join(self._cache_dir, task_id, "masks", f"{frame_idx:06d}.png")
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cached mask not found: {path}")
        return mask

    def get_cached_count(self, task_id: str) -> int:
        """Return the number of contiguous cached masks starting from frame 0."""
        masks_dir = os.path.join(self._cache_dir, task_id, "masks")
        if not os.path.isdir(masks_dir):
            return 0
        idx = 0
        while os.path.exists(os.path.join(masks_dir, f"{idx:06d}.png")):
            idx += 1
        return idx

    def save_metadata(self, task_id: str, video_info: dict) -> None:
        task_dir = os.path.join(self._cache_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        path = os.path.join(task_dir, "metadata.json")
        with open(path, "w") as f:
            json.dump(video_info, f)

    def validate(self, task_id: str, video_info: dict) -> bool:
        path = os.path.join(self._cache_dir, task_id, "metadata.json")
        if not os.path.exists(path):
            return False
        with open(path) as f:
            cached = json.load(f)
        return cached == video_info

    def cleanup(self, task_id: str) -> None:
        task_dir = os.path.join(self._cache_dir, task_id)
        if os.path.isdir(task_dir):
            shutil.rmtree(task_dir)

    def get_total_size(self) -> int:
        """Return total size in bytes of all cached data."""
        if not os.path.isdir(self._cache_dir):
            return 0
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self._cache_dir):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total

    def cleanup_all(self) -> None:
        if not os.path.isdir(self._cache_dir):
            return
        for entry in os.listdir(self._cache_dir):
            path = os.path.join(self._cache_dir, entry)
            if os.path.isdir(path):
                shutil.rmtree(path)

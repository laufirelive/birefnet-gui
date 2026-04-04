import os
import tempfile

import cv2
import numpy as np
import pytest

from src.core.cache import MaskCacheManager
from src.core.temporal import detect_and_fix_outliers


@pytest.fixture
def cache_with_masks():
    """Create a cache with 10 smooth masks and return (cache, tmpdir, task_id)."""
    tmpdir = tempfile.mkdtemp()
    cache = MaskCacheManager(tmpdir)
    task_id = "temporal_test"
    for i in range(10):
        mask = np.full((64, 64), fill_value=200, dtype=np.uint8)
        cache.save_mask(task_id, i, mask)
    return cache, tmpdir, task_id


class TestDetectAndFixOutliers:
    def test_no_outliers_returns_zero(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 0
        for i in range(10):
            mask = cache.load_mask(task_id, i)
            assert np.all(mask == 200)

    def test_single_outlier_gets_fixed(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 5, outlier)
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1
        repaired = cache.load_mask(task_id, 5)
        assert np.allclose(repaired, 200, atol=1)

    def test_first_frame_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 0, outlier)
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1
        repaired = cache.load_mask(task_id, 0)
        assert np.allclose(repaired, 200, atol=1)

    def test_last_frame_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        outlier = np.full((64, 64), fill_value=0, dtype=np.uint8)
        cache.save_mask(task_id, 9, outlier)
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 1
        repaired = cache.load_mask(task_id, 9)
        assert np.allclose(repaired, 200, atol=1)

    def test_scene_change_not_marked_as_outlier(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        for i in range(5, 10):
            mask = np.full((64, 64), fill_value=50, dtype=np.uint8)
            cache.save_mask(task_id, i, mask)
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=10)
        assert fixed == 0

    def test_progress_callback_called(self, cache_with_masks):
        cache, tmpdir, task_id = cache_with_masks
        progress_log = []
        detect_and_fix_outliers(
            cache, task_id, total_frames=10,
            progress_callback=lambda c, t: progress_log.append((c, t)),
        )
        assert len(progress_log) == 10
        assert progress_log[-1] == (10, 10)

    def test_two_frames_video(self):
        tmpdir = tempfile.mkdtemp()
        cache = MaskCacheManager(tmpdir)
        task_id = "two_frames"
        cache.save_mask(task_id, 0, np.full((32, 32), 200, dtype=np.uint8))
        cache.save_mask(task_id, 1, np.full((32, 32), 200, dtype=np.uint8))
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=2)
        assert fixed == 0

    def test_single_frame_video(self):
        tmpdir = tempfile.mkdtemp()
        cache = MaskCacheManager(tmpdir)
        task_id = "one_frame"
        cache.save_mask(task_id, 0, np.full((32, 32), 200, dtype=np.uint8))
        fixed = detect_and_fix_outliers(cache, task_id, total_frames=1)
        assert fixed == 0

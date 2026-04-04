import os

import numpy as np
import pytest

from src.core.cache import MaskCacheManager


@pytest.fixture
def cache_manager(tmp_path):
    return MaskCacheManager(str(tmp_path))


class TestMaskCacheManager:
    def test_save_and_load_mask_roundtrip(self, cache_manager):
        mask = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        cache_manager.save_mask("task1", 0, mask)
        loaded = cache_manager.load_mask("task1", 0)
        np.testing.assert_array_equal(mask, loaded)

    def test_save_creates_directory_structure(self, cache_manager, tmp_path):
        mask = np.zeros((32, 32), dtype=np.uint8)
        cache_manager.save_mask("abc", 5, mask)
        assert os.path.exists(os.path.join(str(tmp_path), "abc", "masks", "000005.png"))

    def test_get_cached_count_empty(self, cache_manager):
        assert cache_manager.get_cached_count("nonexistent") == 0

    def test_get_cached_count_after_saves(self, cache_manager):
        mask = np.zeros((16, 16), dtype=np.uint8)
        for i in range(5):
            cache_manager.save_mask("task2", i, mask)
        assert cache_manager.get_cached_count("task2") == 5

    def test_save_and_validate_metadata(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        cache_manager.save_metadata("t1", info)
        assert cache_manager.validate("t1", info) is True

    def test_validate_fails_on_mismatch(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        cache_manager.save_metadata("t1", info)
        different = {**info, "frame_count": 200}
        assert cache_manager.validate("t1", different) is False

    def test_validate_returns_false_when_no_metadata(self, cache_manager):
        info = {"input_path": "/tmp/v.mp4", "width": 1920, "height": 1080, "fps": 30.0, "frame_count": 100}
        assert cache_manager.validate("missing", info) is False

    def test_cleanup_removes_task_dir(self, cache_manager, tmp_path):
        mask = np.zeros((16, 16), dtype=np.uint8)
        cache_manager.save_mask("t1", 0, mask)
        assert os.path.isdir(os.path.join(str(tmp_path), "t1"))
        cache_manager.cleanup("t1")
        assert not os.path.isdir(os.path.join(str(tmp_path), "t1"))

    def test_cleanup_nonexistent_is_noop(self, cache_manager):
        cache_manager.cleanup("ghost")  # should not raise

    def test_cleanup_all(self, cache_manager, tmp_path):
        mask = np.zeros((16, 16), dtype=np.uint8)
        cache_manager.save_mask("a", 0, mask)
        cache_manager.save_mask("b", 0, mask)
        cache_manager.cleanup_all()
        assert len(os.listdir(str(tmp_path))) == 0

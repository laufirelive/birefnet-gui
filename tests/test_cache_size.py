import os

from src.core.cache import MaskCacheManager, format_size


class TestGetTotalSize:
    def test_empty_cache_returns_zero(self, tmp_path):
        cache = MaskCacheManager(str(tmp_path / "cache"))
        assert cache.get_total_size() == 0

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        cache = MaskCacheManager(str(tmp_path / "nonexistent"))
        assert cache.get_total_size() == 0

    def test_counts_bytes_correctly(self, tmp_path):
        cache_dir = tmp_path / "cache"
        task_dir = cache_dir / "task1" / "masks"
        task_dir.mkdir(parents=True)
        (task_dir / "000000.png").write_bytes(b"x" * 100)
        (task_dir / "000001.png").write_bytes(b"x" * 200)
        (cache_dir / "task1" / "metadata.json").write_text('{"test": true}')
        cache = MaskCacheManager(str(cache_dir))
        total = cache.get_total_size()
        assert total == 314  # 100 + 200 + 14

    def test_cleanup_all_resets_to_zero(self, tmp_path):
        cache_dir = tmp_path / "cache"
        task_dir = cache_dir / "task1" / "masks"
        task_dir.mkdir(parents=True)
        (task_dir / "000000.png").write_bytes(b"x" * 500)
        cache = MaskCacheManager(str(cache_dir))
        assert cache.get_total_size() > 0
        cache.cleanup_all()
        assert cache.get_total_size() == 0


class TestFormatSize:
    def test_zero(self):
        assert format_size(0) == "0 MB"

    def test_small_bytes(self):
        assert format_size(500_000) == "0 MB"

    def test_megabytes(self):
        assert format_size(150_000_000) == "143 MB"

    def test_gigabytes(self):
        assert format_size(2_500_000_000) == "2.3 GB"

    def test_exactly_one_gb(self):
        assert format_size(1_073_741_824) == "1.0 GB"

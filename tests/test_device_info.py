from unittest.mock import MagicMock, patch

from src.core.device_info import DeviceInfo, get_device_info, estimate_vram_gb


class TestDeviceInfo:
    def test_dataclass_fields(self):
        info = DeviceInfo(
            device="cuda",
            device_name="NVIDIA RTX 3060",
            total_vram_gb=12.0,
            available_vram_gb=9.2,
        )
        assert info.device == "cuda"
        assert info.device_name == "NVIDIA RTX 3060"
        assert info.total_vram_gb == 12.0
        assert info.available_vram_gb == 9.2


class TestGetDeviceInfoCPU:
    @patch("src.core.device_info.torch")
    def test_cpu_fallback(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        info = get_device_info()
        assert info.device == "cpu"
        assert info.device_name == "CPU"
        assert info.total_vram_gb == 0.0
        assert info.available_vram_gb == 0.0


class TestGetDeviceInfoCUDA:
    @patch("src.core.device_info.torch")
    def test_cuda_detection(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        props = MagicMock()
        props.name = "NVIDIA RTX 3060"
        props.total_mem = 12 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = props
        mock_torch.cuda.mem_get_info.return_value = (
            int(9.2 * 1024**3),
            12 * 1024**3,
        )
        info = get_device_info()
        assert info.device == "cuda"
        assert info.device_name == "NVIDIA RTX 3060"
        assert abs(info.total_vram_gb - 12.0) < 0.1
        assert abs(info.available_vram_gb - 9.2) < 0.1


class TestGetDeviceInfoMPS:
    @patch("src.core.device_info.platform")
    @patch("src.core.device_info.psutil")
    @patch("src.core.device_info.torch")
    def test_mps_detection(self, mock_torch, mock_psutil, mock_platform):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_platform.processor.return_value = "arm"
        mem = MagicMock()
        mem.total = 16 * 1024**3
        mem.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mem
        info = get_device_info()
        assert info.device == "mps"
        assert "Apple" in info.device_name
        assert abs(info.total_vram_gb - 12.0) < 0.1  # 16 * 0.75
        assert abs(info.available_vram_gb - 6.0) < 0.1  # 8 * 0.75


class TestEstimateVramGb:
    def test_1024_batch_1(self):
        est = estimate_vram_gb(resolution=1024, batch_size=1)
        assert abs(est - 2.5) < 0.01

    def test_512_batch_1(self):
        est = estimate_vram_gb(resolution=512, batch_size=1)
        assert abs(est - 1.0) < 0.01

    def test_2048_batch_1(self):
        est = estimate_vram_gb(resolution=2048, batch_size=1)
        assert abs(est - 8.0) < 0.01

    def test_batch_scaling(self):
        est_1 = estimate_vram_gb(resolution=1024, batch_size=1)
        est_4 = estimate_vram_gb(resolution=1024, batch_size=4)
        assert est_4 > est_1
        assert est_4 < est_1 * 4

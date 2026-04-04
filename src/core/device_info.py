import platform
from dataclasses import dataclass

import psutil
import torch


@dataclass
class DeviceInfo:
    device: str
    device_name: str
    total_vram_gb: float
    available_vram_gb: float


def get_device_info() -> DeviceInfo:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return DeviceInfo(
            device="cuda",
            device_name=props.name,
            total_vram_gb=total_bytes / (1024 ** 3),
            available_vram_gb=free_bytes / (1024 ** 3),
        )

    if torch.backends.mps.is_available():
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3) * 0.75
        available_gb = mem.available / (1024 ** 3) * 0.75
        chip = "Apple Silicon"
        if platform.processor() == "arm":
            chip = "Apple Silicon"
        return DeviceInfo(
            device="mps",
            device_name=chip,
            total_vram_gb=total_gb,
            available_vram_gb=available_gb,
        )

    return DeviceInfo(device="cpu", device_name="CPU", total_vram_gb=0.0, available_vram_gb=0.0)


_BASE_VRAM = {512: 1.0, 1024: 2.5, 2048: 8.0}


def estimate_vram_gb(resolution: int, batch_size: int) -> float:
    base = _BASE_VRAM.get(resolution, 2.5)
    if batch_size <= 1:
        return base
    return base * (1 + (batch_size - 1) * 0.7)

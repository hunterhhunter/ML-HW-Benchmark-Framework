from typing import Dict, Optional

from .base import Collector


class MockNpuCollector(Collector):
    """NPU monitor plugin 계약을 검증하기 위한 SDK-free collector."""

    def __init__(self, device_id: str = "npu0"):
        self.device_id = device_id
        self._started = False

    def is_available(self) -> bool:
        return True

    def start(self) -> None:
        self._started = True

    def collect(self) -> Dict[str, Optional[float]]:
        if not self._started:
            return {}
        return {
            "hw_accel_util": 12.5,
            "hw_accel_mem_used_mb": 256.0,
            "hw_accel_mem_proc_mb": 128.0,
            "hw_accel_temp_c": 42.0,
            "hw_accel_power_w": 7.5,
        }

    def stop(self) -> None:
        self._started = False

    def get_static_info(self) -> Dict[str, str]:
        return {
            "hw_accel_vendor": "MockNPU",
            "hw_accel_name": "Mock NPU PCIe Adapter",
            "hw_accel_device_id": self.device_id,
        }

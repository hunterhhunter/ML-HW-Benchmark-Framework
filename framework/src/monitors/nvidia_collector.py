"""
NVIDIA GPU 메트릭 수집기.

nvidia-ml-py (pynvml)를 사용하여 GPU 사용률, VRAM, 온도, 전력, 클럭을 수집한다.
각 API 호출을 개별 try/except로 감싸서 구형 GPU에서 일부 메트릭이
지원되지 않아도 나머지 메트릭은 정상 수집한다.
"""

from typing import Dict, Optional

from .base import Collector

try:
    import pynvml
    _PYNVML_AVAILABLE = True
except ImportError:
    _PYNVML_AVAILABLE = False


class NvidiaCollector(Collector):
    """NVIDIA GPU 메트릭 수집기. nvidia-ml-py (NVML) 래핑."""

    def __init__(self, gpu_index: int = 0):
        self._gpu_index = gpu_index
        self._handle = None

    def is_available(self) -> bool:
        if not _PYNVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    def start(self) -> None:
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

    def collect(self) -> Dict[str, Optional[float]]:
        result = {}

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            result["hw_gpu_util"] = float(util.gpu)
        except pynvml.NVMLError:
            result["hw_gpu_util"] = None

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            result["hw_gpu_mem_used_mb"] = round(mem.used / (1024 ** 2), 2)
        except pynvml.NVMLError:
            result["hw_gpu_mem_used_mb"] = None

        try:
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            result["hw_gpu_temp_c"] = float(temp)
        except pynvml.NVMLError:
            result["hw_gpu_temp_c"] = None

        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            result["hw_gpu_power_w"] = round(power / 1000.0, 2)
        except pynvml.NVMLError:
            result["hw_gpu_power_w"] = None

        try:
            clock_sm = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_SM
            )
            result["hw_gpu_clock_sm_mhz"] = float(clock_sm)
        except pynvml.NVMLError:
            result["hw_gpu_clock_sm_mhz"] = None

        try:
            clock_mem = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_MEM
            )
            result["hw_gpu_clock_mem_mhz"] = float(clock_mem)
        except pynvml.NVMLError:
            result["hw_gpu_clock_mem_mhz"] = None

        return result

    def stop(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self._handle = None

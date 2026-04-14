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
        self._vram_baseline_mb: float = 0.0
        self._vram_after_load_mb: float = 0.0
        self._gpu_name: str = ""
        self._gpu_total_mb: float = 0.0

    def is_available(self) -> bool:
        if not _PYNVML_AVAILABLE:
            return False
        try:
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    def init_nvml(self) -> None:
        """NVML 초기화 + GPU 정보 + VRAM 베이스라인 캡처. start()와 독립적으로 호출 가능."""
        if self._handle is not None:
            return
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)

        try:
            self._gpu_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode("utf-8")
        except pynvml.NVMLError:
            self._gpu_name = ""

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._vram_baseline_mb = round(mem.used / (1024 ** 2), 2)
            self._gpu_total_mb = round(mem.total / (1024 ** 2), 2)
        except pynvml.NVMLError:
            self._vram_baseline_mb = 0.0
            self._gpu_total_mb = 0.0

    def start(self) -> None:
        """폴링 시작 전 초기화. init_nvml()이 이미 호출되었으면 스킵."""
        self.init_nvml()

    def snapshot_vram(self) -> float:
        """현재 VRAM 사용량을 MB 단위로 반환. runtime.load() 전후 측정에 사용."""
        self.init_nvml()
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return round(mem.used / (1024 ** 2), 2)
        except pynvml.NVMLError:
            return 0.0

    def set_after_load_vram(self, vram_mb: float) -> None:
        """모델 로드 후 VRAM 스냅샷을 기록한다."""
        self._vram_after_load_mb = vram_mb

    def collect(self) -> Dict[str, Optional[float]]:
        result = {}

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            result["hw_gpu_util"] = float(util.gpu)
        except pynvml.NVMLError:
            result["hw_gpu_util"] = None

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = round(mem.used / (1024 ** 2), 2)
            result["hw_gpu_mem_used_mb"] = used_mb
            # 벤치마크 전 시스템 점유분을 뺀 실제 벤치마크 VRAM 사용량
            result["hw_gpu_mem_delta_mb"] = round(max(0, used_mb - self._vram_baseline_mb), 2)
        except pynvml.NVMLError:
            result["hw_gpu_mem_used_mb"] = None
            result["hw_gpu_mem_delta_mb"] = None

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

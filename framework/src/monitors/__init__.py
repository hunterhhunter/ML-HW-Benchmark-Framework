"""
하드웨어 모니터링 모듈.

사용 가능한 Collector를 자동 감지하여 HWMonitor를 생성하는 팩토리 함수를 제공한다.
"""

from typing import Optional

from .base import Collector, HWMonitor


def create_hw_monitor(interval: float = 0.2, gpu_index: int = 0) -> Optional[HWMonitor]:
    """
    사용 가능한 하드웨어 수집기를 자동 감지하여 HWMonitor를 생성한다.

    GPU가 없거나 nvidia-ml-py가 설치되지 않은 환경에서는
    CPU/RAM 메트릭만 수집하는 모니터를 반환한다.
    """
    monitor = HWMonitor(interval=interval)

    # GPU collector (graceful degradation)
    try:
        from .nvidia_collector import NvidiaCollector
        nvidia = NvidiaCollector(gpu_index=gpu_index)
        if nvidia.is_available():
            monitor.add_collector(nvidia)
            print("[HWMonitor] NVIDIA GPU collector enabled")
        else:
            print("[HWMonitor] NVIDIA GPU not detected, skipping GPU monitoring")
    except ImportError:
        print("[HWMonitor] nvidia-ml-py not installed, skipping GPU monitoring")

    # System collector (always available)
    from .system_collector import SystemCollector
    monitor.add_collector(SystemCollector())
    print("[HWMonitor] System collector (CPU/RAM) enabled")

    return monitor


__all__ = ["Collector", "HWMonitor", "create_hw_monitor"]

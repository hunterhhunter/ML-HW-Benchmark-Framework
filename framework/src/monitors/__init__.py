"""
하드웨어 모니터링 모듈.

사용 가능한 Collector를 자동 감지하여 HWMonitor를 생성하는 팩토리 함수를 제공한다.
"""

from typing import Optional

from .base import Collector, HWMonitor


def create_hw_monitor(interval: float = 0.2, gpu_index: int = 0, device: str = "cpu") -> Optional[HWMonitor]:
    """
    사용 가능한 하드웨어 수집기를 자동 감지하여 HWMonitor를 생성한다.

    device가 'cuda'일 때만 GPU 수집기를 활성화한다.
    CPU 전용 벤치마크에서는 GPU 메트릭을 수집하지 않는다.
    """
    monitor = HWMonitor(interval=interval)

    # GPU collector: cuda 디바이스일 때만 활성화
    use_gpu = device.startswith("cuda")
    if use_gpu:
        try:
            from .nvidia_collector import NvidiaCollector
            nvidia = NvidiaCollector(gpu_index=gpu_index)
            if nvidia.is_available():
                nvidia.init_nvml()  # 즉시 초기화: handle + baseline 캡처
                monitor.add_collector(nvidia)
                print(f"[HWMonitor] NVIDIA GPU collector enabled (baseline VRAM: {nvidia._vram_baseline_mb:.1f} MB)")
            else:
                print("[HWMonitor] NVIDIA GPU not detected, skipping GPU monitoring")
        except ImportError:
            print("[HWMonitor] nvidia-ml-py not installed, skipping GPU monitoring")
    else:
        print(f"[HWMonitor] Device is '{device}', skipping GPU monitoring")

    # System collector (always available)
    from .system_collector import SystemCollector
    monitor.add_collector(SystemCollector())
    print("[HWMonitor] System collector (CPU/RAM) enabled")

    return monitor


__all__ = ["Collector", "HWMonitor", "create_hw_monitor"]

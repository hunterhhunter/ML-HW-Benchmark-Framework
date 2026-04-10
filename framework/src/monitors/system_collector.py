"""
시스템 메트릭 수집기.

psutil을 사용하여 CPU 사용률, RAM 사용량을 수집한다.
"""

from typing import Dict, Optional

import psutil

from .base import Collector


class SystemCollector(Collector):
    """CPU/RAM 메트릭 수집기. psutil 래핑."""

    def start(self) -> None:
        # 첫 호출은 의미없는 0.0을 반환하므로 초기화용으로 호출
        psutil.cpu_percent(interval=None)

    def collect(self) -> Dict[str, Optional[float]]:
        mem = psutil.virtual_memory()
        return {
            "hw_cpu_util": psutil.cpu_percent(interval=None),
            "hw_ram_used_mb": round(mem.used / (1024 ** 2), 2),
            "hw_ram_total_mb": round(mem.total / (1024 ** 2), 2),
        }

    def stop(self) -> None:
        pass

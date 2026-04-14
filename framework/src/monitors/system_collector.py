"""
시스템 메트릭 수집기.

psutil을 사용하여 CPU 사용률, RAM 사용량을 수집한다.

디바이스 레벨 지표 (hw_cpu_util, hw_ram_used_mb)는 호스트 전체의 사용량이며,
프로세스 레벨 지표 (hw_cpu_util_proc, hw_ram_proc_mb)는 현재 벤치마크 프로세스와
그 자식 프로세스(vLLM worker 등)만 합산한 값이다. 같은 머신에서 다른 프로그램이
돌고 있어도 벤치마크의 순수 CPU/RAM 사용량만 기록된다.
"""

import os
from typing import Dict, Optional

import psutil

from .base import Collector


class SystemCollector(Collector):
    """CPU/RAM 메트릭 수집기. psutil 래핑."""

    def __init__(self):
        self._cpu_count = psutil.cpu_count(logical=True) or 1
        self._proc_cache: Dict[int, psutil.Process] = {}

    def start(self) -> None:
        # 첫 호출은 의미없는 0.0을 반환하므로 초기화용으로 호출
        psutil.cpu_percent(interval=None)
        own_pid = os.getpid()
        try:
            root = psutil.Process(own_pid)
            self._proc_cache[own_pid] = root
            root.cpu_percent(interval=None)  # 스테이트풀 초기화
            for child in root.children(recursive=True):
                self._proc_cache[child.pid] = child
                try:
                    child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _get_own_procs(self):
        """현재 프로세스 + 자식 프로세스 리스트. cpu_percent 스테이트풀 계산을 위해 캐시."""
        own_pid = os.getpid()
        result = []
        try:
            root = self._proc_cache.get(own_pid)
            if root is None or not root.is_running():
                root = psutil.Process(own_pid)
                self._proc_cache[own_pid] = root
                root.cpu_percent(interval=None)
            result.append(root)
            for child in root.children(recursive=True):
                cached = self._proc_cache.get(child.pid)
                if cached is None:
                    self._proc_cache[child.pid] = child
                    try:
                        child.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                    result.append(child)
                else:
                    result.append(cached)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        # 죽은 프로세스는 캐시에서 정리
        dead = [pid for pid, p in self._proc_cache.items() if not p.is_running()]
        for pid in dead:
            self._proc_cache.pop(pid, None)
        return result

    def collect(self) -> Dict[str, Optional[float]]:
        mem = psutil.virtual_memory()
        result: Dict[str, Optional[float]] = {
            "hw_cpu_util": psutil.cpu_percent(interval=None),
            "hw_ram_used_mb": round(mem.used / (1024 ** 2), 2),
            "hw_ram_total_mb": round(mem.total / (1024 ** 2), 2),
        }

        # 프로세스 레벨 CPU/RAM 집계
        proc_cpu_raw = 0.0  # 코어당 100% 기준 합산 (다코어 시 100% 초과 가능)
        proc_rss_bytes = 0
        alive_count = 0
        for p in self._get_own_procs():
            try:
                proc_cpu_raw += p.cpu_percent(interval=None)
                proc_rss_bytes += p.memory_info().rss
                alive_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if alive_count > 0:
            # 호스트 전체 CPU% 기준으로 정규화 (코어 수로 나눔)
            result["hw_cpu_util_proc"] = round(proc_cpu_raw / self._cpu_count, 2)
            result["hw_ram_proc_mb"] = round(proc_rss_bytes / (1024 ** 2), 2)
        else:
            result["hw_cpu_util_proc"] = None
            result["hw_ram_proc_mb"] = None

        return result

    def stop(self) -> None:
        self._proc_cache.clear()

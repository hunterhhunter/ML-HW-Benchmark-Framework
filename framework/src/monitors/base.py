"""
하드웨어 모니터링 기본 모듈.

Collector ABC와 HWMonitor 오케스트레이터를 정의한다.
백그라운드 스레드에서 주기적으로 하드웨어 메트릭을 수집하고,
벤치마크 종료 후 요약 통계를 반환한다.
"""

import abc
import threading
import time
from typing import Any, Dict, List, Optional


class Collector(abc.ABC):
    """하드웨어 메트릭 수집기 추상 클래스."""

    @abc.abstractmethod
    def start(self) -> None:
        """수집 시작 전 초기화 (예: nvmlInit)."""
        pass

    @abc.abstractmethod
    def collect(self) -> Dict[str, Optional[float]]:
        """현재 하드웨어 상태를 수집하여 반환. 실패한 메트릭은 None."""
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        """수집 종료 후 정리 (예: nvmlShutdown)."""
        pass

    def is_available(self) -> bool:
        """이 수집기가 현재 환경에서 사용 가능한지 확인."""
        return True


class HWMonitor:
    """
    백그라운드 스레드에서 등록된 Collector들을 주기적으로 폴링하여
    하드웨어 메트릭 시계열을 수집하는 오케스트레이터.

    사용법:
        monitor = HWMonitor(interval=0.2)
        monitor.add_collector(NvidiaCollector())
        monitor.add_collector(SystemCollector())
        monitor.start()
        # ... 벤치마크 실행 ...
        monitor.stop()
        hw_metrics = monitor.summary()
    """

    def __init__(self, interval: float = 0.2):
        self._interval = interval
        self._collectors: List[Collector] = []
        self._samples: List[Dict[str, Optional[float]]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def add_collector(self, collector: Collector) -> None:
        self._collectors.append(collector)

    def snapshot_vram(self) -> float:
        """GPU collector가 있으면 현재 VRAM 스냅샷을 반환한다."""
        for collector in self._collectors:
            if hasattr(collector, 'snapshot_vram'):
                return collector.snapshot_vram()
        return 0.0

    def record_after_load_vram(self) -> None:
        """모델 로드 후 VRAM을 기록한다. main.py에서 runtime.load() 직후 호출."""
        for collector in self._collectors:
            if hasattr(collector, 'snapshot_vram') and hasattr(collector, 'set_after_load_vram'):
                vram = collector.snapshot_vram()
                collector.set_after_load_vram(vram)

    def start(self) -> None:
        """모든 collector를 초기화하고 폴링 스레드를 시작한다."""
        self._samples.clear()
        self._stop_event.clear()

        for collector in self._collectors:
            collector.start()

        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """폴링 스레드를 중지하고 모든 collector를 정리한다."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=5)
        self._thread = None

        for collector in self._collectors:
            collector.stop()

    def _poll_loop(self) -> None:
        """백그라운드 폴링 루프. stop_event가 설정될 때까지 반복."""
        while not self._stop_event.is_set():
            sample = {}
            for collector in self._collectors:
                try:
                    data = collector.collect()
                    sample.update(data)
                except Exception:
                    pass
            if sample:
                self._samples.append(sample)
            self._stop_event.wait(self._interval)

    def summary(self) -> Dict[str, Any]:
        """
        수집된 시계열 데이터에서 요약 통계를 계산하여 반환한다.

        반환 키 예시:
            hw_gpu_util_avg, hw_gpu_util_max, hw_gpu_mem_peak_mb,
            hw_gpu_temp_avg_c, hw_gpu_temp_max_c, hw_gpu_power_avg_w,
            hw_gpu_clock_avg_mhz, hw_cpu_util_avg, hw_ram_peak_mb
        """
        if not self._samples:
            return {}

        result = {}

        # GPU 정적 정보 (NvidiaCollector에서 가져옴)
        for collector in self._collectors:
            if hasattr(collector, '_gpu_name') and collector._gpu_name:
                result["hw_gpu_name"] = collector._gpu_name
                result["hw_gpu_total_mb"] = collector._gpu_total_mb
                result["hw_gpu_vram_baseline_mb"] = collector._vram_baseline_mb
                # 모델 VRAM = 로드 후 - 로드 전
                model_vram = round(collector._vram_after_load_mb - collector._vram_baseline_mb, 2)
                result["hw_gpu_vram_model_mb"] = max(0, model_vram)
                break

        # GPU 디바이스 레벨 (GPU 전체)
        self._aggregate(result, "hw_gpu_util", agg_types=["avg", "max"])
        self._aggregate(result, "hw_gpu_mem_used_mb", agg_types=["max"],
                        output_key="hw_gpu_mem_peak_mb")
        # 베이스라인 대비 벤치마크 실제 VRAM 사용량 (레거시)
        self._aggregate(result, "hw_gpu_mem_delta_mb", agg_types=["max"],
                        output_key="hw_gpu_mem_benchmark_mb")
        # GPU 프로세스 레벨 (벤치마크 프로세스 + 자식만)
        self._aggregate(result, "hw_gpu_util_proc", agg_types=["avg", "max"])
        self._aggregate(result, "hw_gpu_proc_count", agg_types=["max"])
        self._aggregate(result, "hw_gpu_mem_proc_mb", agg_types=["max"],
                        output_key="hw_gpu_mem_proc_peak_mb")
        self._aggregate(result, "hw_gpu_mem_proc_pct", agg_types=["max"],
                        output_key="hw_gpu_mem_proc_peak_pct")
        self._aggregate(result, "hw_gpu_mem_proc_of_used_pct", agg_types=["max"],
                        output_key="hw_gpu_mem_proc_of_used_peak_pct")
        # 물리 지표 (디바이스 레벨만 가능)
        self._aggregate(result, "hw_gpu_temp_c", agg_types=["avg", "max"])
        self._aggregate(result, "hw_gpu_power_w", agg_types=["avg"])
        self._aggregate(result, "hw_gpu_clock_sm_mhz", agg_types=["avg"],
                        output_key="hw_gpu_clock_avg_mhz")

        # System 디바이스 레벨 (호스트 전체)
        self._aggregate(result, "hw_cpu_util", agg_types=["avg"])
        self._aggregate(result, "hw_ram_used_mb", agg_types=["max"],
                        output_key="hw_ram_peak_mb")
        # System 프로세스 레벨 (벤치마크 프로세스 + 자식만)
        self._aggregate(result, "hw_cpu_util_proc", agg_types=["avg", "max"])
        self._aggregate(result, "hw_ram_proc_mb", agg_types=["max"],
                        output_key="hw_ram_proc_peak_mb")

        # Accelerator 공통 지표 (NPU 등 벤더 collector가 hw_accel_*로 제공)
        for collector in self._collectors:
            if hasattr(collector, "get_static_info"):
                try:
                    result.update(collector.get_static_info())
                except Exception:
                    pass
        self._aggregate(result, "hw_accel_util", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_mem_used_mb", agg_types=["max"],
                        output_key="hw_accel_mem_peak_mb")
        self._aggregate(result, "hw_accel_mem_proc_mb", agg_types=["max"],
                        output_key="hw_accel_mem_proc_peak_mb")
        self._aggregate(result, "hw_accel_temp_c", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_power_w", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_power_min_w", agg_types=["min"],
                        output_key="hw_accel_power_min_w")
        self._aggregate(result, "hw_accel_power_max_w", agg_types=["max"],
                        output_key="hw_accel_power_max_w")
        self._aggregate(result, "hw_accel_power_sample_period_ms", agg_types=["avg"],
                        output_key="hw_accel_power_sample_period_ms")

        return result

    def _aggregate(
        self,
        result: Dict[str, Any],
        key: str,
        agg_types: List[str],
        output_key: Optional[str] = None,
    ) -> None:
        """시계열에서 특정 키의 값을 추출하고 통계를 계산한다."""
        values = [
            s[key] for s in self._samples
            if key in s and s[key] is not None
        ]
        if not values:
            return

        base_key = output_key if output_key else key

        for agg in agg_types:
            if agg == "avg":
                if output_key and len(agg_types) == 1:
                    result[base_key] = round(sum(values) / len(values), 2)
                else:
                    result[f"{base_key}_avg"] = round(sum(values) / len(values), 2)
            elif agg == "max":
                if output_key and len(agg_types) == 1:
                    result[base_key] = round(max(values), 2)
                else:
                    result[f"{base_key}_max"] = round(max(values), 2)
            elif agg == "min":
                if output_key and len(agg_types) == 1:
                    result[base_key] = round(min(values), 2)
                else:
                    result[f"{base_key}_min"] = round(min(values), 2)

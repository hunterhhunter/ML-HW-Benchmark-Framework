"""HWMonitor 오케스트레이터 단위 테스트."""

import time
import threading
from unittest.mock import MagicMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitors.base import Collector, HWMonitor


class FakeCollector(Collector):
    """테스트용 가짜 수집기."""

    def __init__(self, data=None):
        self._data = data or {"hw_fake_metric": 42.0}
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def collect(self):
        return self._data.copy()

    def stop(self):
        self.stopped = True


class FailingCollector(Collector):
    """collect()에서 예외를 발생시키는 수집기."""

    def start(self):
        pass

    def collect(self):
        raise RuntimeError("collect failed")

    def stop(self):
        pass


class TestHWMonitorLifecycle:
    def test_start_stop_lifecycle(self):
        monitor = HWMonitor(interval=0.05)
        collector = FakeCollector()
        monitor.add_collector(collector)

        monitor.start()
        assert collector.started
        time.sleep(0.15)
        monitor.stop()
        assert collector.stopped
        assert len(monitor._samples) > 0

    def test_stop_without_start(self):
        monitor = HWMonitor()
        # stop without start should not raise
        monitor.stop()

    def test_no_collectors(self):
        monitor = HWMonitor(interval=0.05)
        monitor.start()
        time.sleep(0.1)
        monitor.stop()
        # 수집기 없으면 샘플도 없음
        assert len(monitor._samples) == 0

    def test_custom_interval(self):
        monitor = HWMonitor(interval=0.05)
        collector = FakeCollector()
        monitor.add_collector(collector)

        monitor.start()
        time.sleep(0.25)
        monitor.stop()

        # 0.25초 / 0.05초 간격 = ~5 샘플 (타이밍에 따라 3-7)
        assert len(monitor._samples) >= 2

    def test_failing_collector_does_not_crash(self):
        monitor = HWMonitor(interval=0.05)
        monitor.add_collector(FailingCollector())
        monitor.add_collector(FakeCollector({"hw_good": 1.0}))

        monitor.start()
        time.sleep(0.15)
        monitor.stop()

        # FailingCollector는 건너뛰고 FakeCollector 샘플만 수집
        assert len(monitor._samples) > 0
        assert all("hw_good" in s for s in monitor._samples)


class TestHWMonitorSummary:
    def test_summary_with_samples(self):
        monitor = HWMonitor()
        # 수동으로 샘플 주입
        monitor._samples = [
            {"hw_gpu_util": 50.0, "hw_gpu_mem_used_mb": 1000.0,
             "hw_gpu_proc_count": 1.0, "hw_gpu_mem_proc_mb": 250.0,
             "hw_gpu_mem_proc_pct": 5.0, "hw_gpu_mem_proc_of_used_pct": 25.0,
             "hw_gpu_temp_c": 60.0, "hw_gpu_power_w": 150.0,
             "hw_gpu_clock_sm_mhz": 1500.0,
             "hw_cpu_util": 30.0, "hw_ram_used_mb": 4000.0},
            {"hw_gpu_util": 80.0, "hw_gpu_mem_used_mb": 2000.0,
             "hw_gpu_proc_count": 2.0, "hw_gpu_mem_proc_mb": 1000.0,
             "hw_gpu_mem_proc_pct": 20.0, "hw_gpu_mem_proc_of_used_pct": 50.0,
             "hw_gpu_temp_c": 70.0, "hw_gpu_power_w": 200.0,
             "hw_gpu_clock_sm_mhz": 1600.0,
             "hw_cpu_util": 40.0, "hw_ram_used_mb": 5000.0},
        ]

        result = monitor.summary()

        assert result["hw_gpu_util_avg"] == 65.0
        assert result["hw_gpu_util_max"] == 80.0
        assert result["hw_gpu_mem_peak_mb"] == 2000.0
        assert result["hw_gpu_proc_count_max"] == 2.0
        assert result["hw_gpu_mem_proc_peak_mb"] == 1000.0
        assert result["hw_gpu_mem_proc_peak_pct"] == 20.0
        assert result["hw_gpu_mem_proc_of_used_peak_pct"] == 50.0
        assert result["hw_gpu_temp_c_avg"] == 65.0
        assert result["hw_gpu_temp_c_max"] == 70.0
        assert result["hw_gpu_power_w_avg"] == 175.0
        assert result["hw_gpu_clock_avg_mhz"] == 1550.0
        assert result["hw_cpu_util_avg"] == 35.0
        assert result["hw_ram_peak_mb"] == 5000.0

    def test_summary_empty_samples(self):
        monitor = HWMonitor()
        result = monitor.summary()
        assert result == {}

    def test_summary_with_none_values(self):
        monitor = HWMonitor()
        monitor._samples = [
            {"hw_gpu_util": 50.0, "hw_gpu_power_w": None},
            {"hw_gpu_util": 80.0, "hw_gpu_power_w": None},
        ]

        result = monitor.summary()

        assert result["hw_gpu_util_avg"] == 65.0
        assert result["hw_gpu_util_max"] == 80.0
        # power는 모두 None이므로 결과에 없어야 함
        assert "hw_gpu_power_w_avg" not in result

    def test_summary_partial_none(self):
        monitor = HWMonitor()
        monitor._samples = [
            {"hw_gpu_util": 50.0, "hw_gpu_power_w": 100.0},
            {"hw_gpu_util": 80.0, "hw_gpu_power_w": None},
        ]

        result = monitor.summary()

        assert result["hw_gpu_util_avg"] == 65.0
        # power는 100.0 하나만 유효
        assert result["hw_gpu_power_w_avg"] == 100.0

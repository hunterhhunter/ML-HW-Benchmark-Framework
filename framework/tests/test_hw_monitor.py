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


class EventCollector(Collector):
    def __init__(self, name, events, *, fail_start=False, fail_stop=False):
        self.name = name
        self.events = events
        self.fail_start = fail_start
        self.fail_stop = fail_stop

    def start(self):
        self.events.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"start failed: {self.name}")

    def collect(self):
        return {}

    def stop(self):
        self.events.append(f"stop:{self.name}")
        if self.fail_stop:
            raise RuntimeError(f"stop failed: {self.name}")


class SummaryCollector(FakeCollector):
    def __init__(self, summary_metrics, *, static_info=None, summary_error=None):
        super().__init__({})
        self.summary_metrics = summary_metrics
        self.static_info = static_info or {}
        self.summary_error = summary_error
        self.summary_calls = 0

    def get_static_info(self):
        return dict(self.static_info)

    def get_summary_metrics(self):
        self.summary_calls += 1
        if self.summary_error is not None:
            raise self.summary_error
        return self.summary_metrics


def test_start_failure_rolls_back_started_collectors_in_reverse_order():
    events = []
    monitor = HWMonitor()
    monitor.add_collector(EventCollector("first", events))
    monitor.add_collector(EventCollector("second", events))
    monitor.add_collector(EventCollector("failing", events, fail_start=True))

    with pytest.raises(RuntimeError, match="start failed: failing"):
        monitor.start()

    assert events == [
        "start:first",
        "start:second",
        "start:failing",
        "stop:second",
        "stop:first",
    ]
    assert monitor._thread is None
    assert monitor._started_collectors == []


def test_stop_is_idempotent_after_partial_start_rollback():
    events = []
    monitor = HWMonitor()
    monitor.add_collector(EventCollector("started", events))
    monitor.add_collector(EventCollector("failing", events, fail_start=True))

    with pytest.raises(RuntimeError, match="start failed: failing"):
        monitor.start()

    monitor.stop()
    monitor.stop()
    assert events == ["start:started", "start:failing", "stop:started"]


def test_stop_attempts_all_collectors_and_is_idempotent_after_stop_error():
    events = []
    monitor = HWMonitor()
    monitor.add_collector(EventCollector("first", events, fail_stop=True))
    monitor.add_collector(EventCollector("second", events))
    monitor.start()

    with pytest.raises(RuntimeError, match="stop failed: first"):
        monitor.stop()

    monitor.stop()
    assert events == [
        "start:first",
        "start:second",
        "stop:second",
        "stop:first",
    ]


def test_summary_hooks_run_without_samples_and_do_not_overwrite_existing_keys():
    monitor = HWMonitor()
    first = SummaryCollector(
        {
            "hw_accel_vendor": "summary-must-not-replace-static",
            "hw_accel_energy_j": 12.5,
        },
        static_info={"hw_accel_vendor": "Mobilint"},
    )
    second = SummaryCollector({"hw_accel_energy_j": 999.0})
    monitor.add_collector(first)
    monitor.add_collector(second)

    result = monitor.summary()

    assert result["hw_accel_vendor"] == "Mobilint"
    assert result["hw_accel_energy_j"] == 12.5
    assert first.summary_calls == 1
    assert second.summary_calls == 1


def test_summary_hook_cannot_overwrite_time_series_and_failures_are_isolated():
    monitor = HWMonitor()
    monitor._samples = [{"hw_accel_power_w": 10.0, "hw_accel_current_a": 2.0}]
    monitor.add_collector(SummaryCollector({"hw_accel_power_w_avg": 999.0}))
    monitor.add_collector(SummaryCollector([]))
    monitor.add_collector(
        SummaryCollector({}, summary_error=RuntimeError("summary failed"))
    )
    monitor.add_collector(SummaryCollector({"hw_accel_energy_j": 3.0}))

    result = monitor.summary()

    assert result["hw_accel_power_w_avg"] == 10.0
    assert result["hw_accel_current_a_avg"] == 2.0
    assert result["hw_accel_current_a_max"] == 2.0
    assert result["hw_accel_energy_j"] == 3.0


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
             "hw_cpu_util": 30.0, "hw_ram_used_mb": 4000.0,
             "hw_accel_voltage_mv": 750.0, "hw_accel_clock_mhz": 1000.0},
            {"hw_gpu_util": 80.0, "hw_gpu_mem_used_mb": 2000.0,
             "hw_gpu_proc_count": 2.0, "hw_gpu_mem_proc_mb": 1000.0,
             "hw_gpu_mem_proc_pct": 20.0, "hw_gpu_mem_proc_of_used_pct": 50.0,
             "hw_gpu_temp_c": 70.0, "hw_gpu_power_w": 200.0,
             "hw_gpu_clock_sm_mhz": 1600.0,
             "hw_cpu_util": 40.0, "hw_ram_used_mb": 5000.0,
             "hw_accel_voltage_mv": 800.0, "hw_accel_clock_mhz": 1200.0},
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
        assert result["hw_accel_voltage_mv_avg"] == 775.0
        assert result["hw_accel_voltage_mv_max"] == 800.0
        assert result["hw_accel_clock_mhz_avg"] == 1100.0
        assert result["hw_accel_clock_mhz_max"] == 1200.0

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

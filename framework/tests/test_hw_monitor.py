"""HWMonitor 오케스트레이터 단위 테스트."""

import time
import threading
from unittest.mock import MagicMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monitors.base import Collector, HWMonitor
import monitors.base as monitor_base


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


class StopSignalCollector(EventCollector):
    def __init__(self, name, events, monitor):
        super().__init__(name, events)
        self.monitor = monitor

    def stop(self):
        self.events.append(
            f"stop:{self.name}:signaled={self.monitor._stop_event.is_set()}"
        )


class ThreadRecordingCollector(StopSignalCollector):
    def collect(self):
        self.events.append(f"collect:{threading.current_thread().name}")
        return {}


class CollectorStopBaseError(BaseException):
    pass


class BaseExceptionStopCollector(EventCollector):
    def __init__(self, name, events):
        super().__init__(name, events)
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        self.events.append(f"stop:{self.name}")
        if self.stop_calls == 1:
            raise CollectorStopBaseError(f"stop interrupted: {self.name}")


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
        "stop:failing",
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
    assert events == [
        "start:started",
        "start:failing",
        "stop:failing",
        "stop:started",
    ]


def test_stop_attempts_all_collectors_and_is_idempotent_after_stop_error():
    events = []
    monitor = HWMonitor()
    first = EventCollector("first", events)
    second = EventCollector("second", events, fail_stop=True)
    monitor.add_collector(first)
    monitor.add_collector(second)
    monitor.start()

    with pytest.raises(RuntimeError, match="stop failed: second"):
        monitor.stop()

    assert monitor._started_collectors == [second]
    with pytest.raises(RuntimeError, match="already started"):
        monitor.start()

    second.fail_stop = False
    monitor.stop()
    monitor.stop()
    assert events == [
        "start:first",
        "start:second",
        "stop:second",
        "stop:first",
        "stop:second",
    ]
    assert monitor._started_collectors == []


def test_start_rollback_handles_baseexception_and_preserves_start_error():
    events = []
    monitor = HWMonitor()
    monitor.add_collector(EventCollector("first", events))
    second = BaseExceptionStopCollector("second", events)
    monitor.add_collector(second)
    monitor.add_collector(EventCollector("failing", events, fail_start=True))

    with pytest.raises(RuntimeError, match="start failed: failing"):
        monitor.start()

    assert events == [
        "start:first",
        "start:second",
        "start:failing",
        "stop:failing",
        "stop:second",
        "stop:first",
    ]
    assert monitor._thread is None
    assert monitor._started_collectors == [second]

    monitor.stop()
    assert events[-1] == "stop:second"
    assert monitor._started_collectors == []


def test_thread_constructor_failure_signals_before_collector_rollback(monkeypatch):
    events = []
    monitor = HWMonitor()
    monitor.add_collector(StopSignalCollector("collector", events, monitor))

    def fail_constructor(*, target, daemon):
        assert callable(target)
        assert daemon is True
        events.append("thread:construct")
        raise RuntimeError("thread construction failed")

    monkeypatch.setattr(monitor_base.threading, "Thread", fail_constructor)

    with pytest.raises(RuntimeError, match="thread construction failed"):
        monitor.start()

    assert events == [
        "start:collector",
        "thread:construct",
        "stop:collector:signaled=True",
    ]
    assert monitor._stop_event.is_set()
    assert monitor._thread is None
    assert monitor._started_collectors == []


def test_prelaunch_thread_start_failure_is_joined_before_rollback(monkeypatch):
    events = []
    monitor = HWMonitor()
    monitor.add_collector(StopSignalCollector("collector", events, monitor))

    class PrelaunchFailingThread:
        def __init__(self, *, target, daemon):
            assert callable(target)
            assert daemon is True
            events.append("thread:construct")

        def start(self):
            events.append("thread:start")
            raise RuntimeError("thread launch failed")

        def join(self, timeout):
            assert timeout == 5
            events.append("thread:join")
            raise RuntimeError("cannot join thread before it is started")

        def is_alive(self):
            return False

    monkeypatch.setattr(
        monitor_base.threading,
        "Thread",
        PrelaunchFailingThread,
    )

    with pytest.raises(RuntimeError, match="thread launch failed"):
        monitor.start()

    assert events == [
        "start:collector",
        "thread:construct",
        "thread:start",
        "thread:join",
        "stop:collector:signaled=True",
    ]
    assert monitor._thread is None
    assert monitor._started_collectors == []


def test_start_then_raise_thread_is_joined_and_cannot_revive_on_retry(monkeypatch):
    events = []
    monitor = HWMonitor(interval=60.0)
    monitor.add_collector(StopSignalCollector("collector", events, monitor))
    real_thread_type = threading.Thread
    poll_entered = threading.Event()

    class StartThenRaiseThread:
        instance = None

        def __init__(self, *, target, daemon):
            events.append("thread:construct")

            def entered_target():
                poll_entered.set()
                target()

            self.inner = real_thread_type(target=entered_target, daemon=daemon)
            type(self).instance = self

        def start(self):
            events.append("thread:start")
            self.inner.start()
            assert poll_entered.wait(timeout=1.0)
            raise RuntimeError("thread start raised after launch")

        def join(self, timeout):
            events.append("thread:join")
            self.inner.join(timeout=timeout)

        def is_alive(self):
            return self.inner.is_alive()

    monkeypatch.setattr(monitor_base.threading, "Thread", StartThenRaiseThread)

    try:
        with pytest.raises(RuntimeError, match="thread start raised after launch"):
            monitor.start()

        failed_thread = StartThenRaiseThread.instance
        assert failed_thread is not None
        assert events == [
            "start:collector",
            "thread:construct",
            "thread:start",
            "thread:join",
            "stop:collector:signaled=True",
        ]
        assert not failed_thread.inner.is_alive()
        assert monitor._thread is None
        assert monitor._started_collectors == []

        monkeypatch.setattr(monitor_base.threading, "Thread", real_thread_type)
        monitor.start()
        assert not failed_thread.inner.is_alive()
        monitor.stop()
        assert not failed_thread.inner.is_alive()
    finally:
        monitor._stop_event.set()
        failed_thread = StartThenRaiseThread.instance
        if failed_thread is not None:
            failed_thread.inner.join(timeout=1.0)


def test_stop_join_error_still_stops_every_collector_and_raises_first_error():
    events = []
    monitor = HWMonitor()
    first = EventCollector("first", events)
    second = EventCollector("second", events, fail_stop=True)
    monitor.add_collector(first)
    monitor.add_collector(second)

    class JoinFailingThread:
        def join(self, timeout):
            assert timeout == 5
            events.append("thread:join")
            raise RuntimeError("thread join failed")

        def is_alive(self):
            return False

    monitor._thread = JoinFailingThread()
    monitor._started_collectors = [first, second]

    with pytest.raises(RuntimeError, match="thread join failed"):
        monitor.stop()

    assert monitor._started_collectors == [second]
    second.fail_stop = False
    monitor.stop()
    assert events == [
        "thread:join",
        "stop:second",
        "stop:first",
        "stop:second",
    ]
    assert monitor._stop_event.is_set()
    assert monitor._thread is None
    assert monitor._started_collectors == []


def test_delayed_prelaunch_race_keeps_old_stop_event_after_retry(monkeypatch):
    events = []
    monitor = HWMonitor(interval=60.0)
    collector = ThreadRecordingCollector("collector", events, monitor)
    monitor.add_collector(collector)
    real_thread_type = threading.Thread

    class DelayedPrelaunchThread:
        instance = None

        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon
            self.inner = None
            type(self).instance = self

        def start(self):
            raise RuntimeError("thread launch status ambiguous")

        def join(self, timeout):
            assert timeout == 5
            raise RuntimeError("cannot join thread before it is started")

        def is_alive(self):
            return False

        def launch_late(self):
            self.inner = real_thread_type(
                target=self.target,
                daemon=self.daemon,
                name="delayed-old",
            )
            self.inner.start()
            self.inner.join(timeout=1.0)

    monkeypatch.setattr(
        monitor_base.threading,
        "Thread",
        DelayedPrelaunchThread,
    )

    with pytest.raises(RuntimeError, match="thread launch status ambiguous"):
        monitor.start()

    delayed = DelayedPrelaunchThread.instance
    assert delayed is not None
    old_stop_event = monitor._stop_event
    assert old_stop_event.is_set()
    assert monitor._thread is None
    assert monitor._started_collectors == []

    monkeypatch.setattr(monitor_base.threading, "Thread", real_thread_type)
    try:
        monitor.start()
        assert monitor._stop_event is not old_stop_event
        delayed.launch_late()
        monitor.stop()

        assert delayed.inner is not None
        assert not delayed.inner.is_alive()
        assert "collect:delayed-old" not in events
    finally:
        if monitor._thread is not None:
            monitor.stop()


def test_ambiguous_join_after_launch_retains_ownership_until_stop(monkeypatch):
    events = []
    monitor = HWMonitor(interval=60.0)
    collector = ThreadRecordingCollector("collector", events, monitor)
    monitor.add_collector(collector)
    real_thread_type = threading.Thread
    target_entered = threading.Event()
    release_target = threading.Event()

    class AmbiguousJoinThread:
        instance = None

        def __init__(self, *, target, daemon):
            def gated_target():
                target_entered.set()
                release_target.wait()
                target()

            self.inner = real_thread_type(target=gated_target, daemon=daemon)
            self.join_calls = 0
            type(self).instance = self

        def start(self):
            self.inner.start()
            assert target_entered.wait(timeout=1.0)
            raise RuntimeError("start raised after launch")

        def join(self, timeout):
            self.join_calls += 1
            if self.join_calls == 1:
                raise RuntimeError("thread reports not started")
            self.inner.join(timeout=timeout)

        def is_alive(self):
            return self.inner.is_alive()

    monkeypatch.setattr(monitor_base.threading, "Thread", AmbiguousJoinThread)

    try:
        with pytest.raises(RuntimeError, match="start raised after launch"):
            monitor.start()

        ambiguous = AmbiguousJoinThread.instance
        assert ambiguous is not None
        old_stop_event = monitor._stop_event
        assert old_stop_event.is_set()
        assert monitor._thread is ambiguous
        assert monitor._started_collectors == [collector]

        monkeypatch.setattr(monitor_base.threading, "Thread", real_thread_type)
        with pytest.raises(RuntimeError, match="already started"):
            monitor.start()
        assert monitor._stop_event is old_stop_event

        release_target.set()
        monitor.stop()
        assert ambiguous.join_calls == 2
        assert not ambiguous.inner.is_alive()
        assert monitor._thread is None
        assert monitor._started_collectors == []
        assert not any(event.startswith("collect:") for event in events)
    finally:
        release_target.set()
        ambiguous = AmbiguousJoinThread.instance
        if ambiguous is not None:
            ambiguous.inner.join(timeout=1.0)
        if monitor._thread is not None:
            monitor.stop()


def test_start_rollback_join_timeout_rejects_retry_until_later_cleanup(monkeypatch):
    events = []
    monitor = HWMonitor(interval=60.0)
    collector = StopSignalCollector("collector", events, monitor)
    monitor.add_collector(collector)
    real_thread_type = threading.Thread
    target_entered = threading.Event()
    release_target = threading.Event()

    class TimeoutThread:
        instance = None

        def __init__(self, *, target, daemon):
            def gated_target():
                target_entered.set()
                release_target.wait()
                target()

            self.inner = real_thread_type(target=gated_target, daemon=daemon)
            self.join_calls = 0
            type(self).instance = self

        def start(self):
            self.inner.start()
            assert target_entered.wait(timeout=1.0)
            raise RuntimeError("start failed after launch")

        def join(self, timeout):
            self.join_calls += 1
            if self.join_calls == 1:
                return
            self.inner.join(timeout=timeout)

        def is_alive(self):
            return self.inner.is_alive()

    monkeypatch.setattr(monitor_base.threading, "Thread", TimeoutThread)

    try:
        with pytest.raises(RuntimeError, match="start failed after launch"):
            monitor.start()

        timed_out = TimeoutThread.instance
        assert timed_out is not None
        assert timed_out.is_alive()
        assert monitor._thread is timed_out
        assert monitor._started_collectors == [collector]
        assert "stop:collector:signaled=True" not in events

        monkeypatch.setattr(monitor_base.threading, "Thread", real_thread_type)
        with pytest.raises(RuntimeError, match="already started"):
            monitor.start()

        release_target.set()
        monitor.stop()
        assert timed_out.join_calls == 2
        assert not timed_out.is_alive()
        assert monitor._thread is None
        assert monitor._started_collectors == []
        assert events.count("stop:collector:signaled=True") == 1
    finally:
        release_target.set()
        timed_out = TimeoutThread.instance
        if timed_out is not None:
            timed_out.inner.join(timeout=1.0)
        if monitor._thread is not None:
            monitor.stop()


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

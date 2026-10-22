"""Integration of a reserved raw trace with the monitor lifecycle."""

import csv
import threading
import time

from core.power_trace import PowerReading, PowerTraceSource
from core.result_store import reserve_run_artifacts
from monitors.base import Collector, HWMonitor


class FakeClock:
    def __init__(self):
        self.now_ns = 0

    def __call__(self):
        return self.now_ns

    def advance(self, seconds):
        self.now_ns += int(seconds * 1_000_000_000)


class PowerCollector(Collector):
    def __init__(self):
        self.power_reads = 0

    def is_available(self):
        return True

    def start(self):
        return None

    def stop(self):
        return None

    def collect(self):
        return {}

    def power_trace_source(self):
        return PowerTraceSource(
            collector="fake",
            monitor_source="fake-api",
            device_id="0",
            power_scope="device",
            sample_interval_sec=10.0,
        )

    def collect_power(self):
        self.power_reads += 1
        return PowerReading(status="ok", power_w=10.0)


def test_monitor_finishes_baseline_before_inference(tmp_path, monkeypatch):
    clock = FakeClock()
    sleeps = []

    def wait(seconds):
        sleeps.append(seconds)
        clock.advance(seconds)

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="baseline-test",
    )
    collector = PowerCollector()
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=60.0,
        clock_ns=clock,
        wait_fn=wait,
    )
    monitor.add_collector(collector)
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")

    monitor.start()
    assert sleeps == [60.0]
    assert monitor.startup_timeout_hint_sec() == 60.0
    monitor.stop()

    with reservation.power_trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["phase"] for row in rows] == [
        "baseline",
        *(["baseline"] * 6),
        "inference",
        "inference",
    ]
    assert [row["sample_status"] for row in rows[1:7]] == [
        "overrun"
    ] * 6
    assert collector.power_reads == 3
    assert monitor.power_trace_metadata()["power_trace_status"] == "partial"


def test_monitor_stop_records_unpolled_inference_deadlines(tmp_path, monkeypatch):
    clock = FakeClock()

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    class FastPowerCollector(PowerCollector):
        def power_trace_source(self):
            return PowerTraceSource(
                collector="fake",
                monitor_source="fake-api",
                device_id="0",
                power_scope="device",
                sample_interval_sec=0.1,
            )

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="stop-deadlines"
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        clock_ns=clock,
        wait_fn=lambda seconds: None,
    )
    monitor.add_collector(FastPowerCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.start()
    clock.advance(0.35)
    monitor.stop()

    with reservation.power_trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["scheduled_elapsed_ms"] for row in rows if row["sample_status"] == "overrun"] == [
        "100.0", "200.0", "300.0"
    ]
    assert monitor.power_trace_metadata()["power_trace_status"] == "partial"


def test_summary_window_excludes_raw_trace_baseline(tmp_path, monkeypatch):
    clock = FakeClock()

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    class SummaryCollector(PowerCollector):
        def __init__(self):
            super().__init__()
            self.inference = False
            self.summary_resets = 0

        def start(self):
            return {"hw_accel_power_w": 10.0}

        def collect(self):
            return {"hw_accel_power_w": 20.0 if self.inference else 10.0}

        def stop(self):
            return {"hw_accel_power_w": 20.0}

        def begin_inference_summary(self):
            self.summary_resets += 1
            self.inference = True

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="summary-window"
    )
    collector = SummaryCollector()

    def wait(seconds):
        monitor._collect_summary_once()
        clock.advance(seconds)

    monitor = HWMonitor(
        summary_enabled=True,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.05,
        clock_ns=clock,
        wait_fn=wait,
    )
    monitor.add_collector(collector)
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.start()
    monitor._collect_summary_once()
    monitor.stop()

    assert collector.summary_resets == 1
    assert monitor.summary()["hw_accel_power_w_avg"] == 20.0


def test_abandoned_async_monitor_cannot_publish_late_trace(tmp_path):
    entered_baseline = threading.Event()
    release_baseline = threading.Event()
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="late-monitor"
    )

    def wait(seconds):
        entered_baseline.set()
        assert release_baseline.wait(timeout=2)

    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.1,
        wait_fn=wait,
    )
    monitor.add_collector(PowerCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    worker = threading.Thread(target=monitor.start)
    worker.start()
    try:
        assert entered_baseline.wait(timeout=2)
        monitor.abandon_power_trace()
    finally:
        release_baseline.set()
        worker.join(timeout=2)
        monitor.stop()

    assert not reservation.power_trace_path.exists()
    assert not list((tmp_path / "power").glob("*.power.partial.csv"))
    assert monitor.power_trace_metadata()["power_trace_status"] == "failed"


def test_async_trace_publication_waits_for_main_thread_finalization(
    tmp_path, monkeypatch
):
    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="deferred-publish"
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        wait_fn=lambda seconds: None,
    )
    monitor.add_collector(PowerCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.defer_power_trace_publication()
    monitor.start()
    monitor.stop()
    assert not reservation.power_trace_path.exists()

    monitor.finalize_power_trace()
    assert reservation.power_trace_path.is_file()
    assert monitor.power_trace_metadata()["power_trace_status"] == "complete"


def test_async_trace_finalize_does_not_block_on_stalled_collector_lock(
    tmp_path, monkeypatch
):
    real_thread = threading.Thread

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="bounded-finalize"
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        wait_fn=lambda seconds: None,
    )
    monitor.add_collector(PowerCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.defer_power_trace_publication()
    monitor.start()
    monitor.stop()

    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with monitor._power_attempt_lock:
            acquired.set()
            assert release.wait(timeout=2)

    holder = real_thread(target=hold_lock)
    holder.start()
    try:
        assert acquired.wait(timeout=2)
        started = time.monotonic()
        monitor.finalize_power_trace()
        assert time.monotonic() - started < 0.5
        assert monitor.power_trace_metadata()["power_trace_status"] == "failed"
        assert not reservation.power_trace_path.exists()
    finally:
        release.set()
        holder.join(timeout=2)
        monitor.abandon_power_trace()

    assert not list((tmp_path / "power").glob("*.power.partial.csv"))


def test_stalled_power_poll_cleans_temporary_trace_after_abandon(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingCollector(PowerCollector):
        def power_trace_source(self):
            original = super().power_trace_source()
            return PowerTraceSource(
                collector=original.collector,
                monitor_source=original.monitor_source,
                device_id=original.device_id,
                power_scope=original.power_scope,
                sample_interval_sec=0.01,
            )

        def collect_power(self):
            self.power_reads += 1
            if self.power_reads >= 3:
                entered.set()
                assert release.wait(timeout=2)
            return PowerReading(status="ok", power_w=10.0)

    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="late-poll-cleanup"
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        wait_fn=lambda seconds: None,
    )
    monitor.add_collector(BlockingCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.defer_power_trace_publication()
    monitor.start()
    try:
        assert entered.wait(timeout=1)
        monitor._stop_event.set()
        monitor.finalize_power_trace()
        assert monitor.power_trace_metadata()["power_trace_status"] == "failed"
        assert not reservation.power_trace_path.exists()
    finally:
        release.set()
        monitor.stop()

    assert list((tmp_path / "power").iterdir()) == []


def test_stalled_stop_boundary_read_cleans_temporary_trace_after_abandon(
    tmp_path, monkeypatch
):
    real_thread = threading.Thread
    entered = threading.Event()
    release = threading.Event()

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    class BlockingCollector(PowerCollector):
        def collect_power(self):
            self.power_reads += 1
            if self.power_reads == 3:
                entered.set()
                assert release.wait(timeout=2)
            return PowerReading(status="ok", power_w=10.0)

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv", run_id="late-stop-cleanup"
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        wait_fn=lambda seconds: None,
    )
    monitor.add_collector(BlockingCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.defer_power_trace_publication()
    monitor.start()
    stop_thread = real_thread(target=monitor.stop)
    stop_thread.start()
    try:
        assert entered.wait(timeout=1)
        monitor.abandon_power_trace()
        assert monitor.power_trace_metadata()["power_trace_status"] == "failed"
    finally:
        release.set()
        stop_thread.join(timeout=2)

    assert not stop_thread.is_alive()
    assert list((tmp_path / "power").iterdir()) == []


def test_monitor_marks_missed_power_deadline_partial(tmp_path, monkeypatch):
    clock = FakeClock()

    class DormantThread:
        def __init__(self, *, target, daemon):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

        def is_alive(self):
            return False

    class SlowCollector(PowerCollector):
        def power_trace_source(self):
            source = super().power_trace_source()
            return PowerTraceSource(
                collector=source.collector,
                monitor_source=source.monitor_source,
                device_id=source.device_id,
                power_scope=source.power_scope,
                sample_interval_sec=0.1,
            )

        def collect_power(self):
            reading = super().collect_power()
            if self.power_reads > 2:
                clock.advance(0.22)
            return reading

    monkeypatch.setattr("monitors.base.threading.Thread", DormantThread)
    reservation = reserve_run_artifacts(
        results_path=tmp_path / "results.csv",
        run_id="overrun-test",
    )
    monitor = HWMonitor(
        summary_enabled=False,
        power_trace_enabled=True,
        power_trace_baseline_sec=0.0,
        clock_ns=clock,
        wait_fn=lambda seconds: clock.advance(seconds),
    )
    monitor.add_collector(SlowCollector())
    monitor.configure_power_trace(reservation=reservation, target_id="test-npu")
    monitor.start()
    clock.advance(0.1)
    monitor._poll_power_if_due()
    monitor.stop()

    with reservation.power_trace_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missed = [row for row in rows if row["sample_status"] == "overrun"]
    assert [row["scheduled_elapsed_ms"] for row in missed] == ["200.0", "300.0"]
    assert all(row["power_w"] == "" for row in missed)
    assert monitor.power_trace_metadata()["power_trace_status"] == "partial"


def test_power_trace_default_baseline_is_sixty_seconds():
    monitor = HWMonitor(power_trace_enabled=True)
    assert monitor.startup_timeout_hint_sec() == 60.0
    assert HWMonitor().startup_timeout_hint_sec() == 0.0

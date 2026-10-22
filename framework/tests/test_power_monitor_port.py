"""Integration of a reserved raw trace with the monitor lifecycle."""

import csv

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
        "inference",
        "inference",
    ]
    assert collector.power_reads == 3
    assert monitor.power_trace_metadata()["power_trace_status"] == "complete"


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

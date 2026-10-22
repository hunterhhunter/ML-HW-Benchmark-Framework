"""Preload power and inference summaries keep separate timing boundaries."""

import pytest

from core.edge_power_session import PreloadPowerSession


class Monitor:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def start_preload_baseline(self):
        self.events.append("power:baseline")

    def begin_inference(self):
        self.events.append("power:inference")

    def start(self):
        self.events.append(f"{self.name}:start")

    def stop(self):
        self.events.append(f"{self.name}:stop")

    def abort(self):
        self.events.append(f"{self.name}:abort")

    def summary(self):
        return {"hw_test": 3}

    def power_trace_metadata(self):
        return {"power_trace_status": "complete"}


def test_preload_session_keeps_summary_monitor_inside_inference_window():
    events = []
    power = Monitor("power", events)
    summary = Monitor("summary", events)
    session = PreloadPowerSession(power, reservation=object())

    session.capture_baseline()
    session.attach_summary(summary)
    session.start()
    session.stop()
    session.abort()

    assert events == [
        "power:baseline",
        "summary:start",
        "power:inference",
        "power:stop",
        "summary:stop",
    ]
    assert session.summary() == {"hw_test": 3}
    assert session.power_trace_metadata()["power_trace_status"] == "complete"
    assert session.startup_timeout_hint_sec() == 0


def test_preload_session_aborts_active_reader_when_setup_fails():
    events = []
    session = PreloadPowerSession(Monitor("power", events), reservation=object())
    session.capture_baseline()
    session.abort()
    session.abort()
    assert events == ["power:baseline", "power:abort"]


def test_preload_session_retries_summary_cleanup_after_stop_error():
    events = []

    class FlakySummary(Monitor):
        def __init__(self):
            super().__init__("summary", events)
            self.stop_calls = 0

        def stop(self):
            self.stop_calls += 1
            super().stop()
            if self.stop_calls == 1:
                raise RuntimeError("summary stop failed")

    session = PreloadPowerSession(Monitor("power", events), reservation=object())
    session.capture_baseline()
    session.attach_summary(FlakySummary())
    session.start()
    with pytest.raises(RuntimeError, match="summary stop failed"):
        session.stop()
    session.abort()
    assert events[-2:] == ["summary:stop", "summary:stop"]

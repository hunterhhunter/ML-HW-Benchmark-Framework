"""The unloaded baseline and inference must keep one trace reservation."""

from core.preload_power_session import PreloadPowerSession


class RecordingMonitor:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def defer_power_trace_publication(self):
        self.events.append("defer")

    def start_preload_baseline(self):
        self.events.append("baseline")

    def begin_inference(self):
        self.events.append("inference")

    def start(self):
        self.events.append("summary-start")

    def stop(self):
        self.events.append(f"{self.name}-stop")

    def abort(self):
        self.events.append("power-abort")

    def abandon_power_trace(self):
        self.events.append("power-abandon")

    def finalize_power_trace(self):
        self.events.append("power-finalize")

    def summary(self):
        return {"hw_test": 3}

    def power_trace_metadata(self):
        return {"power_trace_status": "complete"}

    def record_after_load_vram(self):
        self.events.append("vram-after-load")


def test_preload_session_resumes_only_after_summary_starts():
    events = []
    reservation = object()
    session = PreloadPowerSession(RecordingMonitor("power", events), reservation=reservation)

    session.defer_power_trace_publication()
    session.capture_baseline()
    session.attach_summary(RecordingMonitor("summary", events))
    session.record_after_load_vram()
    session.start()
    session.stop()
    session.finalize_power_trace()

    assert session.reservation is reservation
    assert events == [
        "defer", "baseline", "vram-after-load", "summary-start",
        "inference", "power-stop", "summary-stop", "power-finalize",
    ]
    assert session.summary() == {"hw_test": 3}
    assert session.power_trace_metadata() == {"power_trace_status": "complete"}
    assert session.startup_timeout_hint_sec() == 0


def test_preload_session_aborts_power_collector_on_setup_failure():
    events = []
    session = PreloadPowerSession(RecordingMonitor("power", events), reservation=object())

    session.capture_baseline()
    session.abort()
    session.abort()

    assert events == ["baseline", "power-abort"]


def test_preload_session_can_abandon_late_async_publication():
    events = []
    session = PreloadPowerSession(RecordingMonitor("power", events), reservation=object())

    session.capture_baseline()
    session.stop()
    session.abandon_power_trace()

    assert events == ["baseline", "power-stop", "power-abandon"]

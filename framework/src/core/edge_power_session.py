"""Coordinate a preload INA3221 trace with an inference-only summary monitor."""


class PreloadPowerSession:
    """Keep preload idle and inference power in one run-bound trace."""

    def __init__(self, power_monitor, *, reservation):
        self.power_monitor = power_monitor
        self.reservation = reservation
        self.summary_monitor = None
        self._active = False
        self._summary_started = False

    def capture_baseline(self) -> None:
        self._active = True
        try:
            self.power_monitor.start_preload_baseline()
        except BaseException:
            self.abort()
            raise

    def attach_summary(self, summary_monitor):
        self.summary_monitor = summary_monitor
        return self

    def start(self) -> None:
        if not self._active:
            raise RuntimeError("preload power session is not active")
        try:
            if self.summary_monitor is not None:
                self._summary_started = True
                self.summary_monitor.start()
            self.power_monitor.begin_inference()
        except BaseException:
            self.abort()
            raise

    def stop(self) -> None:
        if not self._active:
            return
        try:
            self.power_monitor.stop()
            self._active = False
        finally:
            self._stop_summary()

    def abort(self) -> None:
        if not self._active and not self._summary_started:
            return
        try:
            if self._active:
                self.power_monitor.abort()
                self._active = False
        finally:
            self._stop_summary()

    def _stop_summary(self) -> None:
        if self._summary_started:
            self.summary_monitor.stop()
            self._summary_started = False

    def summary(self) -> dict:
        if self.summary_monitor is None:
            return {}
        return self.summary_monitor.summary()

    def power_trace_metadata(self) -> dict:
        return self.power_monitor.power_trace_metadata()

    def startup_timeout_hint_sec(self) -> float:
        return 0.0

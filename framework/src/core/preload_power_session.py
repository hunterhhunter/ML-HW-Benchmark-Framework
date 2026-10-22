"""Keep model-unloaded idle and inference in one run-bound trace."""


class PreloadPowerSession:
    """Adapt a prestarted power collector to the benchmark monitor lifecycle."""

    def __init__(self, power_monitor, *, reservation):
        self.power_monitor = power_monitor
        self.reservation = reservation
        self.summary_monitor = None
        self._active = False
        self._summary_started = False

    def defer_power_trace_publication(self) -> None:
        self.power_monitor.defer_power_trace_publication()

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

    def record_after_load_vram(self) -> None:
        if self.summary_monitor is not None:
            self.summary_monitor.record_after_load_vram()

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

    def abandon_power_trace(self) -> None:
        self.power_monitor.abandon_power_trace()

    def finalize_power_trace(self) -> None:
        self.power_monitor.finalize_power_trace()

    def _stop_summary(self) -> None:
        if self._summary_started:
            self.summary_monitor.stop()
            self._summary_started = False

    def summary(self) -> dict:
        return self.summary_monitor.summary() if self.summary_monitor is not None else {}

    def power_trace_metadata(self) -> dict:
        return self.power_monitor.power_trace_metadata()

    def startup_timeout_hint_sec(self) -> float:
        return 0.0

"""
하드웨어 모니터링 기본 모듈.

Collector ABC와 HWMonitor 오케스트레이터를 정의한다.
백그라운드 스레드에서 주기적으로 하드웨어 메트릭을 수집하고,
벤치마크 종료 후 요약 통계를 반환한다.
"""

import abc
import math
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from core.artifact_reservation import RunArtifactReservation
from core.power_trace import (
    PowerReading,
    PowerTraceArtifact,
    PowerTraceConfig,
    PowerTraceSample,
    PowerTraceSource,
    PowerTraceWriter,
)


class Collector(abc.ABC):
    """하드웨어 메트릭 수집기 추상 클래스."""

    @abc.abstractmethod
    def start(self) -> Optional[Dict[str, Optional[float]]]:
        """Initialize collection and optionally return a boundary sample."""
        pass

    @abc.abstractmethod
    def collect(self) -> Dict[str, Optional[float]]:
        """현재 하드웨어 상태를 수집하여 반환. 실패한 메트릭은 None."""
        pass

    @abc.abstractmethod
    def stop(self) -> Optional[Dict[str, Optional[float]]]:
        """Stop collection and optionally return a boundary sample."""
        pass

    def is_available(self) -> bool:
        """이 수집기가 현재 환경에서 사용 가능한지 확인."""
        return True

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Return collector-owned final metrics after sampling has stopped."""
        return {}

    def power_trace_source(self) -> PowerTraceSource | None:
        """Return the optional raw power source exposed by this collector."""
        return None

    def collect_power(self) -> PowerReading:
        """Read one raw power value from the declared source."""
        raise RuntimeError("collector has no power trace source")


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

    def __init__(
        self,
        interval: float = 0.2,
        *,
        summary_enabled: bool = True,
        power_trace_enabled: bool = False,
        power_trace_baseline_sec: float = 3.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        wait_fn: Callable[[float], None] = time.sleep,
    ):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("monitor interval must be finite and positive")
        if (
            not math.isfinite(power_trace_baseline_sec)
            or power_trace_baseline_sec < 0
        ):
            raise ValueError(
                "power trace baseline duration must be finite and non-negative"
            )
        self._interval = interval
        self._summary_enabled = summary_enabled
        self._power_trace_enabled = power_trace_enabled
        self._power_trace_baseline_sec = power_trace_baseline_sec
        self._clock_ns = clock_ns
        self._wait_fn = wait_fn
        self._collectors: List[Collector] = []
        self._collector_locks: Dict[int, threading.RLock] = {}
        self._samples: List[Dict[str, Optional[float]]] = []
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_collectors: List[Collector] = []
        self._power_attempt_lock = threading.RLock()
        self._power_reservation: RunArtifactReservation | None = None
        self._power_target_id = ""
        self._power_collector: Collector | None = None
        self._power_source: PowerTraceSource | None = None
        self._power_writer: PowerTraceWriter | None = None
        self._power_phase = "baseline"
        self._power_origin_ns: int | None = None
        self._next_power_deadline_ns: int | None = None
        initial_status = "unavailable" if power_trace_enabled else "disabled"
        self._power_artifact = PowerTraceArtifact(status=initial_status)

    def add_collector(self, collector: Collector) -> None:
        self._collectors.append(collector)
        self._collector_locks[id(collector)] = threading.RLock()

    def configure_power_trace(
        self,
        *,
        reservation: RunArtifactReservation,
        target_id: str,
    ) -> None:
        """Bind the optional trace to a pre-reserved benchmark run."""
        if self._thread is not None or self._started_collectors:
            raise RuntimeError("cannot configure power trace after monitor start")
        if type(reservation) is not RunArtifactReservation:
            raise ValueError("a valid RunArtifactReservation is required")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("target_id must be a non-empty string")
        self._power_reservation = reservation
        self._power_target_id = target_id

    def power_trace_metadata(self) -> dict[str, object]:
        return self._power_artifact.as_result_metadata()

    def startup_timeout_hint_sec(self) -> float:
        if self._power_trace_enabled:
            return self._power_trace_baseline_sec
        return 0.0

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
        """Initialize collectors transactionally and start the polling thread."""
        if self._thread is not None or self._started_collectors:
            raise RuntimeError("HWMonitor is already started")

        self._samples.clear()
        stop_event = threading.Event()
        self._stop_event = stop_event

        try:
            for collector in self._collectors:
                # start() may acquire resources before it raises, so retain
                # cleanup ownership from the moment the attempt begins.
                self._started_collectors.append(collector)
                with self._collector_lock(collector):
                    boundary_sample = collector.start()
                if self._summary_enabled:
                    self._record_boundary_sample(boundary_sample)

            self._prepare_power_trace()

            def poll_loop() -> None:
                self._poll_loop(stop_event)

            thread = threading.Thread(target=poll_loop, daemon=True)
            self._thread = thread
            thread.start()

            if self._power_writer is not None:
                self._wait_fn(self._power_trace_baseline_sec)
                with self._power_attempt_lock:
                    self._power_phase = "inference"
                    self._force_power_sample()
                    self._flush_power_trace()
        except BaseException:
            stop_event.set()

            thread = self._thread
            thread_alive = False
            if thread is not None:
                try:
                    thread.join(timeout=5)
                except BaseException:
                    pass
                try:
                    thread_alive = thread.is_alive()
                except BaseException:
                    thread_alive = True

            if not thread_alive:
                self._thread = None
                self._abort_power_trace()
                self._stop_started_collectors()
            raise

    def stop(self) -> None:
        """Stop polling and release every possible collector cleanup owner."""
        thread = self._thread

        first_error: BaseException | None = None
        if thread is not None or self._started_collectors:
            self._stop_event.set()
        thread_alive = False
        if thread is not None:
            try:
                thread.join(timeout=5)
            except BaseException as exc:
                first_error = exc
            try:
                thread_alive = thread.is_alive()
            except BaseException as exc:
                thread_alive = True
                if first_error is None:
                    first_error = exc

        if thread_alive:
            if first_error is None:
                first_error = RuntimeError(
                    "HWMonitor polling thread did not stop within 5 seconds"
                )
            raise first_error

        self._thread = None
        self._finish_power_trace()
        stop_error = self._stop_started_collectors()
        if first_error is None:
            first_error = stop_error

        if first_error is not None:
            raise first_error

    def _stop_started_collectors(self) -> BaseException | None:
        """Stop owners in reverse order and retain only failed cleanups."""
        failed_in_stop_order: List[Collector] = []
        first_error: BaseException | None = None
        for collector in reversed(self._started_collectors):
            try:
                with self._collector_lock(collector):
                    boundary_sample = collector.stop()
            except BaseException as exc:
                failed_in_stop_order.append(collector)
                if first_error is None:
                    first_error = exc
            else:
                if self._summary_enabled:
                    self._record_boundary_sample(boundary_sample)

        # Restore acquisition order so a later retry is reverse-ordered too.
        self._started_collectors = list(reversed(failed_in_stop_order))
        return first_error

    def _record_boundary_sample(self, sample: object) -> None:
        """Record one successful lifecycle sample returned by a collector."""
        if type(sample) is dict and sample:
            self._samples.append(dict(sample))

    def _poll_loop(self, stop_event: threading.Event) -> None:
        """Poll summary metrics and raw power on absolute deadlines."""
        next_summary_ns = self._clock_ns()
        summary_period_ns = max(1, int(self._interval * 1_000_000_000))
        while not stop_event.is_set():
            now_ns = self._clock_ns()
            if (
                self._power_writer is not None
                and self._next_power_deadline_ns is not None
                and now_ns >= self._next_power_deadline_ns
            ):
                self._poll_power_if_due()
                now_ns = self._clock_ns()

            if self._summary_enabled and now_ns >= next_summary_ns:
                self._collect_summary_once()
                now_ns = self._clock_ns()
                while next_summary_ns <= now_ns:
                    next_summary_ns += summary_period_ns

            deadlines = []
            if self._summary_enabled:
                deadlines.append(next_summary_ns)
            if (
                self._power_writer is not None
                and self._next_power_deadline_ns is not None
            ):
                deadlines.append(self._next_power_deadline_ns)
            if deadlines:
                delay_sec = max(0.0, (min(deadlines) - self._clock_ns()) / 1e9)
            else:
                delay_sec = self._interval
            stop_event.wait(delay_sec)

    def _collect_summary_once(self) -> None:
        sample: Dict[str, Optional[float]] = {}
        for collector in self._collectors:
            try:
                with self._collector_lock(collector):
                    data = collector.collect()
                sample.update(data)
            except Exception:
                pass
        if sample:
            self._samples.append(sample)

    def _collector_lock(self, collector: Collector) -> threading.RLock:
        lock = self._collector_locks.get(id(collector))
        if lock is None:
            lock = threading.RLock()
            self._collector_locks[id(collector)] = lock
        return lock

    def _prepare_power_trace(self) -> None:
        if not self._power_trace_enabled:
            return
        if self._power_reservation is None or not self._power_target_id:
            raise RuntimeError("power trace must be configured before monitor start")

        selected: tuple[Collector, PowerTraceSource] | None = None
        for collector in self._collectors:
            try:
                source = collector.power_trace_source()
            except Exception:
                self._power_artifact = PowerTraceArtifact(status="failed")
                return
            if source is not None:
                selected = (collector, source)
                break
        if selected is None:
            self._power_artifact = PowerTraceArtifact(status="unavailable")
            return

        collector, source = selected
        self._power_collector = collector
        self._power_source = source
        writer = PowerTraceWriter(
            self._power_reservation,
            PowerTraceConfig(
                run_id=self._power_reservation.run_id,
                target_id=self._power_target_id,
                baseline_duration_sec=self._power_trace_baseline_sec,
            ),
            source,
        )
        try:
            writer.start()
        except Exception:
            try:
                self._power_artifact = writer.fail()
            except Exception:
                self._power_artifact = self._failed_power_artifact()
            return

        self._power_writer = writer
        self._power_phase = "baseline"
        self._power_origin_ns = self._clock_ns()
        period_ns = max(1, int(source.sample_interval_sec * 1_000_000_000))
        self._next_power_deadline_ns = self._power_origin_ns + period_ns
        self._force_power_sample()

    def _force_power_sample(self) -> None:
        with self._power_attempt_lock:
            if self._power_writer is None:
                return
            scheduled_ns = self._clock_ns()
            self._record_power_read(scheduled_ns)

    def _poll_power_if_due(self) -> bool:
        with self._power_attempt_lock:
            writer = self._power_writer
            source = self._power_source
            deadline_ns = self._next_power_deadline_ns
            if (
                writer is None
                or source is None
                or deadline_ns is None
                or self._clock_ns() < deadline_ns
            ):
                return False

            period_ns = max(1, int(source.sample_interval_sec * 1_000_000_000))
            self._next_power_deadline_ns = deadline_ns + period_ns
            self._record_power_read(deadline_ns)
            query_end_ns = self._clock_ns()
            while (
                self._power_writer is not None
                and self._next_power_deadline_ns is not None
                and self._next_power_deadline_ns <= query_end_ns
            ):
                missed_ns = self._next_power_deadline_ns
                self._write_power_sample(
                    PowerTraceSample(
                        phase=self._power_phase,
                        scheduled_elapsed_ms=self._elapsed_ms(missed_ns),
                        observed_elapsed_ms=self._elapsed_ms(query_end_ns),
                        query_latency_ms=0.0,
                        reading=PowerReading(
                            status="overrun",
                            error_code="deadline_missed",
                        ),
                    )
                )
                self._next_power_deadline_ns = missed_ns + period_ns
            return True

    def _record_power_read(self, scheduled_ns: int) -> None:
        collector = self._power_collector
        if collector is None or self._power_writer is None:
            return
        query_start_ns = self._clock_ns()
        try:
            with self._collector_lock(collector):
                reading = collector.collect_power()
            if type(reading) is not PowerReading:
                raise ValueError("collector returned an invalid PowerReading")
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            error_name = type(exc).__name__
            reading = PowerReading(
                status="read_error",
                error_code=f"collector:{error_name}"[:128],
            )
        query_end_ns = self._clock_ns()
        self._write_power_sample(
            PowerTraceSample(
                phase=self._power_phase,
                scheduled_elapsed_ms=self._elapsed_ms(scheduled_ns),
                observed_elapsed_ms=self._elapsed_ms(
                    (query_start_ns + query_end_ns) // 2
                ),
                query_latency_ms=max(0.0, (query_end_ns - query_start_ns) / 1e6),
                reading=reading,
            )
        )

    def _write_power_sample(self, sample: PowerTraceSample) -> None:
        writer = self._power_writer
        if writer is None:
            return
        try:
            writer.write_attempt(sample)
        except Exception:
            self._fail_power_trace()

    def _elapsed_ms(self, timestamp_ns: int) -> float:
        origin_ns = self._power_origin_ns
        if origin_ns is None:
            return 0.0
        return max(0.0, (timestamp_ns - origin_ns) / 1e6)

    def _flush_power_trace(self) -> None:
        with self._power_attempt_lock:
            writer = self._power_writer
            if writer is None:
                return
            try:
                writer.flush()
            except Exception:
                self._fail_power_trace()

    def _finish_power_trace(self) -> None:
        with self._power_attempt_lock:
            writer = self._power_writer
            if writer is None:
                return
            self._record_power_read(self._clock_ns())
            writer = self._power_writer
            if writer is None:
                return
            try:
                self._power_artifact = writer.finish()
            except Exception:
                self._fail_power_trace()
            else:
                self._power_writer = None

    def _abort_power_trace(self) -> None:
        with self._power_attempt_lock:
            if self._power_writer is not None:
                self._fail_power_trace()

    def _fail_power_trace(self) -> None:
        writer = self._power_writer
        self._power_writer = None
        if writer is None:
            return
        try:
            self._power_artifact = writer.fail()
        except Exception:
            self._power_artifact = self._failed_power_artifact()

    def _failed_power_artifact(self) -> PowerTraceArtifact:
        source = self._power_source
        return PowerTraceArtifact(
            status="failed",
            monitor_source=source.monitor_source if source is not None else "",
            power_scope=source.power_scope if source is not None else "",
        )

    def summary(self) -> Dict[str, Any]:
        """
        수집된 시계열 데이터에서 요약 통계를 계산하여 반환한다.

        반환 키 예시:
            hw_gpu_util_avg, hw_gpu_util_max, hw_gpu_mem_peak_mb,
            hw_gpu_temp_avg_c, hw_gpu_temp_max_c, hw_gpu_power_avg_w,
            hw_gpu_clock_avg_mhz, hw_cpu_util_avg, hw_ram_peak_mb
        """
        if not self._summary_enabled:
            return {}

        result: Dict[str, Any] = {}

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
        self._aggregate(result, "hw_accel_current_a", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_voltage_mv", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_clock_mhz", agg_types=["avg", "max"])
        self._aggregate(result, "hw_accel_power_min_w", agg_types=["min"],
                        output_key="hw_accel_power_min_w")
        self._aggregate(result, "hw_accel_power_max_w", agg_types=["max"],
                        output_key="hw_accel_power_max_w")
        self._aggregate(result, "hw_accel_power_sample_period_ms", agg_types=["avg"],
                        output_key="hw_accel_power_sample_period_ms")

        for collector in self._collectors:
            try:
                final_metrics = collector.get_summary_metrics()
            except Exception:
                continue
            if not isinstance(final_metrics, dict):
                continue
            for key, value in final_metrics.items():
                if key not in result:
                    result[key] = value

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

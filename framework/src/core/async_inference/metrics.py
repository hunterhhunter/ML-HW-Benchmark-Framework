from array import array
from collections import Counter
from threading import Lock, local
from typing import Any, Dict

import numpy as np

from .types import RequestTrace, TerminalStatus


PERCENTILES = (50.0, 90.0, 95.0, 97.0, 99.0, 99.9)


class TimingDistribution:
    def __init__(self):
        self.values = array("d")

    def add(self, value_ms: float) -> None:
        self.values.append(float(value_ms))

    def summary(self) -> Dict[str, float | int | None]:
        if not self.values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "sum": 0.0,
                "p50": None,
                "p90": None,
                "p95": None,
                "p97": None,
                "p99": None,
                "p99_9": None,
            }
        values = np.frombuffer(self.values, dtype=np.float64)
        percentiles = np.percentile(values, PERCENTILES)
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "sum": float(values.sum()),
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p95": float(percentiles[2]),
            "p97": float(percentiles[3]),
            "p99": float(percentiles[4]),
            "p99_9": float(percentiles[5]),
        }


class TimeWeightedGauge:
    def __init__(self, started_ns: int, initial: int = 0):
        self.last_ns = started_ns
        self.value = initial
        self.area = 0
        self.minimum = initial
        self.maximum = initial

    def update(self, value: int, now_ns: int) -> None:
        effective_ns = max(now_ns, self.last_ns)
        self.area += self.value * (effective_ns - self.last_ns)
        self.value = value
        self.last_ns = effective_ns
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self, end_ns: int, started_ns: int) -> Dict[str, float | int]:
        self.update(self.value, end_ns)
        duration = max(1, end_ns - started_ns)
        return {
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.area / duration,
        }


class _AcceptanceClaim:
    __slots__ = (
        "accepted_before",
        "committed",
        "closed",
        "queue_transition",
    )

    def __init__(self, accepted_before: int, queue_transition=None):
        self.accepted_before = accepted_before
        self.committed = False
        self.closed = False
        self.queue_transition = queue_transition


class AsyncMetricsCollector:
    def __init__(
        self,
        started_ns: int,
        worker_count: int,
        latency_slo_ms: float | None = None,
    ):
        self.started_ns = started_ns
        self.worker_count = worker_count
        self.latency_slo_ms = latency_slo_ms
        self.lock = Lock()
        self._acceptance_local = local()
        self._has_events = False
        self.counters = Counter()
        self.invalid_reasons = set()
        self.warnings = set()
        self.error_types = Counter()
        self.error_request_examples = {}
        self.queue_depth = TimeWeightedGauge(started_ns)
        self._next_queue_sequence = 1
        self._pending_queue_transitions = {}
        self.inflight = TimeWeightedGauge(started_ns)
        self.worker_busy_ns = Counter()
        self.worker_batches = Counter()
        self.worker_samples = Counter()
        self.batch_sizes = TimingDistribution()
        self.generation_timing_sources = Counter()
        self.timings = {
            "scheduler_delay": TimingDistribution(),
            "submit_wait": TimingDistribution(),
            "queue_wait": TimingDistribution(),
            "service_time": TimingDistribution(),
            "completion_overhead": TimingDistribution(),
            "e2e_latency": TimingDistribution(),
            "ttft_event": TimingDistribution(),
            "reported_ttft": TimingDistribution(),
            "reported_tpot": TimingDistribution(),
        }

    def begin_measurement(self, started_ns: int) -> None:
        with self.lock:
            if self._has_events:
                raise RuntimeError("measurement already contains events")
            self.started_ns = started_ns
            self.queue_depth = TimeWeightedGauge(started_ns)
            self._next_queue_sequence = 1
            self._pending_queue_transitions = {}
            self.inflight = TimeWeightedGauge(started_ns)

    def record_submitted(self) -> None:
        with self.lock:
            self._has_events = True
            self.counters["submitted"] += 1

    def claim_acceptance(self, queue_transition=None):
        if getattr(self._acceptance_local, "claim", None) is not None:
            raise RuntimeError("acceptance claim already active")
        with self.lock:
            accepted_before = self.counters["accepted"]
        claim = _AcceptanceClaim(accepted_before, queue_transition)
        self._acceptance_local.claim = claim
        return claim

    def finish_acceptance(self, claim) -> bool:
        active = getattr(self._acceptance_local, "claim", None)
        if active is not claim or claim.closed:
            raise RuntimeError("acceptance claim is not active")
        with self.lock:
            if self.counters["accepted"] > claim.accepted_before:
                claim.committed = True
        claim.closed = True
        del self._acceptance_local.claim
        return claim.committed

    def record_accepted(self, now_ns: int, queue_depth: int) -> None:
        with self.lock:
            self._has_events = True
            self.counters["accepted"] += 1
            outstanding = self.counters["accepted"] - self.counters["terminal"]
            self.inflight.update(outstanding, now_ns)
            claim = getattr(self._acceptance_local, "claim", None)
            if claim is not None and claim.queue_transition is not None:
                transition = claim.queue_transition
                self._record_queue_depth_locked(
                    transition.depth,
                    transition.now_ns,
                    transition.sequence,
                )
            else:
                self.queue_depth.update(queue_depth, now_ns)
            if claim is not None:
                claim.committed = True

    def record_rejected(self, reason: str) -> None:
        with self.lock:
            self._has_events = True
            self.counters["rejected"] += 1
            self.counters[f"rejected:{reason}"] += 1
            self.invalid_reasons.add("request_rejected")

    def record_queue_depth(
        self,
        depth: int,
        now_ns: int,
        sequence: int | None = None,
    ) -> None:
        with self.lock:
            self._has_events = True
            self._record_queue_depth_locked(depth, now_ns, sequence)

    def _record_queue_depth_locked(
        self,
        depth: int,
        now_ns: int,
        sequence: int | None,
    ) -> None:
        if sequence is None:
            self.queue_depth.update(depth, now_ns)
            return
        if sequence < self._next_queue_sequence:
            return
        self._pending_queue_transitions[sequence] = (depth, now_ns)
        while self._next_queue_sequence in self._pending_queue_transitions:
            queued_depth, queued_ns = self._pending_queue_transitions.pop(
                self._next_queue_sequence
            )
            self.queue_depth.update(queued_depth, queued_ns)
            self._next_queue_sequence += 1

    def record_queue_full(self) -> None:
        with self.lock:
            self._has_events = True
            self.counters["queue_full_events"] += 1

    def record_worker_busy(
        self,
        worker_id: int,
        started_ns: int,
        finished_ns: int,
        batch_size: int = 1,
        sample_count: int | None = None,
    ) -> None:
        with self.lock:
            self._has_events = True
            if finished_ns < started_ns:
                self.invalid_reasons.add("timing_invariant_failed")
                return
            self.worker_busy_ns[worker_id] += finished_ns - started_ns
            self.worker_batches[worker_id] += 1
            self.worker_samples[worker_id] += (
                batch_size if sample_count is None else sample_count
            )
            self.batch_sizes.add(batch_size)

    def add_invalid_reason(self, reason: str) -> None:
        with self.lock:
            self._has_events = True
            self.invalid_reasons.add(reason)

    def add_warning(self, warning: str) -> None:
        with self.lock:
            self._has_events = True
            self.warnings.add(warning)

    def record_first_token(self, request, event) -> None:
        with self.lock:
            self._has_events = True
            if event.first_token_ns < request.issued_ns:
                self.invalid_reasons.add("timing_invariant_failed")
                return
            self.counters["first_token_events"] += 1
            self.timings["ttft_event"].add(
                (event.first_token_ns - request.issued_ns) / 1_000_000.0
            )

    def record_generation(self, generated_tokens: int, timing_ms) -> None:
        if generated_tokens <= 0:
            return
        with self.lock:
            self._has_events = True
            self.counters["completed_tokens"] += generated_tokens
            if not isinstance(timing_ms, dict):
                return
            reported_ttft = timing_ms.get("ttft_ms")
            reported_tpot = timing_ms.get("tpot_ms")
            if reported_ttft is not None:
                self.timings["reported_ttft"].add(reported_ttft)
            if reported_tpot is not None:
                self.timings["reported_tpot"].add(reported_tpot)
            self.generation_timing_sources[
                timing_ms.get("timing_source", "unknown")
            ] += 1

    def record_terminal(self, trace: RequestTrace) -> None:
        with self.lock:
            self._has_events = True
            status = trace.status.value
            self.counters[status] += 1
            self.counters[f"{status}_samples"] += trace.sample_count
            self.counters["terminal"] += 1
            if trace.status is TerminalStatus.FAILED:
                self.invalid_reasons.add("request_failed")
            if trace.timed_out:
                self.counters["timed_out"] += 1
                self.invalid_reasons.add("request_timeout")
            if trace.error_type:
                self.error_types[trace.error_type] += 1
                examples = self.error_request_examples.setdefault(
                    trace.error_type,
                    [],
                )
                if len(examples) < 5:
                    examples.append(trace.request_id)
            self.inflight.update(
                self.counters["accepted"] - self.counters["terminal"],
                trace.completed_ns,
            )

            timestamps = (
                trace.scheduled_ns,
                trace.issued_ns,
                trace.enqueued_ns,
                trace.runtime_started_ns,
                trace.runtime_finished_ns,
                trace.completed_ns,
            )
            if any(
                earlier_ns > later_ns
                for earlier_ns, later_ns in zip(timestamps, timestamps[1:])
            ):
                self.invalid_reasons.add("timing_invariant_failed")
                return

            ns_to_ms = 1.0 / 1_000_000.0
            values = {
                "scheduler_delay": trace.issued_ns - trace.scheduled_ns,
                "submit_wait": trace.enqueued_ns - trace.issued_ns,
                "queue_wait": trace.runtime_started_ns - trace.enqueued_ns,
                "service_time": trace.runtime_finished_ns - trace.runtime_started_ns,
                "completion_overhead": trace.completed_ns - trace.runtime_finished_ns,
                "e2e_latency": trace.completed_ns - trace.issued_ns,
            }
            for name, value_ns in values.items():
                self.timings[name].add(value_ns * ns_to_ms)
            if (
                self.latency_slo_ms is not None
                and values["e2e_latency"] * ns_to_ms > self.latency_slo_ms
            ):
                self.counters["over_latency_slo"] += 1
            timing_sum = (
                values["submit_wait"]
                + values["queue_wait"]
                + values["service_time"]
                + values["completion_overhead"]
            )
            if abs(values["e2e_latency"] - timing_sum) > 50_000:
                self.invalid_reasons.add("timing_invariant_failed")

    def finalize(self, end_ns: int) -> Dict[str, Dict[str, Any]]:
        with self.lock:
            submitted = self.counters["submitted"]
            accepted = self.counters["accepted"]
            rejected = self.counters["rejected"]
            completed = self.counters["completed"]
            completed_samples = self.counters["completed_samples"]
            failed = self.counters["failed"]
            outstanding = accepted - completed - failed
            invariant_valid = (
                submitted == accepted + rejected
                and accepted == completed + failed + outstanding
                and outstanding >= 0
            )
            if not invariant_valid:
                self.invalid_reasons.add("counter_invariant_failed")
            if outstanding:
                self.invalid_reasons.add("flush_timeout")

            duration_ns = max(1, end_ns - self.started_ns)
            duration_sec = duration_ns / 1_000_000_000.0
            queue = self.queue_depth.summary(end_ns, self.started_ns)
            inflight = self.inflight.summary(end_ns, self.started_ns)
            timing = {
                name: distribution.summary()
                for name, distribution in self.timings.items()
            }
            total_busy = sum(self.worker_busy_ns.values())
            worker_slots = max(1, self.worker_count)
            worker_capacity_ns = worker_slots * duration_ns
            if total_busy > worker_capacity_ns or any(
                busy_ns > duration_ns for busy_ns in self.worker_busy_ns.values()
            ):
                self.invalid_reasons.add("timing_invariant_failed")
            utilization = min(
                1.0,
                total_busy / worker_capacity_ns,
            )
            summary = {
                "async_submitted_requests": submitted,
                "async_accepted_requests": accepted,
                "async_completed_requests": completed,
                "async_completed_samples": completed_samples,
                "async_failed_requests": failed,
                "async_rejected_requests": rejected,
                "async_timed_out_requests": self.counters["timed_out"],
                "async_over_latency_slo_requests": self.counters[
                    "over_latency_slo"
                ],
                "async_outstanding_requests": outstanding,
                "async_issued_requests_per_sec": submitted / duration_sec,
                "async_completed_samples_per_sec": completed_samples / duration_sec,
                "async_completed_tokens_per_sec": (
                    self.counters["completed_tokens"] / duration_sec
                ),
                "async_queue_depth_max": queue["max"],
                "async_worker_utilization": utilization,
                "async_e2e_latency_p50_ms": timing["e2e_latency"]["p50"],
                "async_e2e_latency_p95_ms": timing["e2e_latency"]["p95"],
                "async_e2e_latency_p99_ms": timing["e2e_latency"]["p99"],
                "async_queue_wait_p99_ms": timing["queue_wait"]["p99"],
                "async_service_time_p99_ms": timing["service_time"]["p99"],
            }
            details = {
                "measurement_duration_sec": duration_sec,
                "measurement": {
                    "started_monotonic_ns": self.started_ns,
                    "ended_monotonic_ns": end_ns,
                    "duration_sec": duration_sec,
                },
                "invalid_reasons": sorted(self.invalid_reasons),
                "warnings": sorted(self.warnings),
                "counter_invariants": {
                    "valid": invariant_valid,
                    "submitted_equals_accepted_plus_rejected": (
                        submitted == accepted + rejected
                    ),
                    "accepted_equals_terminal_plus_outstanding": (
                        accepted == completed + failed + outstanding
                    ),
                },
                "counts": dict(self.counters),
                "timing_ms": timing,
                "queue": {
                    "depth_min": queue["min"],
                    "depth_max": queue["max"],
                    "depth_mean": queue["mean"],
                    "full_events": self.counters["queue_full_events"],
                    "submit_block_total_ms": timing["submit_wait"]["sum"],
                    "inflight_min": inflight["min"],
                    "inflight_max": inflight["max"],
                    "inflight_mean": inflight["mean"],
                },
                "workers": {
                    "utilization": utilization,
                    "busy_ns": dict(self.worker_busy_ns),
                    "batches": dict(self.worker_batches),
                    "samples": dict(self.worker_samples),
                },
                "batch_size": self.batch_sizes.summary(),
                "failure_types": dict(self.error_types),
                "failure_request_examples": {
                    error_type: list(request_ids)
                    for error_type, request_ids in self.error_request_examples.items()
                },
                "generation": {
                    "completed_tokens": self.counters["completed_tokens"],
                    "timing_sources": dict(self.generation_timing_sources),
                    "event_ttft_ms": timing["ttft_event"],
                    "reported_ttft_ms": timing["reported_ttft"],
                    "reported_tpot_ms": timing["reported_tpot"],
                },
            }
            return {"summary": summary, "details": details}

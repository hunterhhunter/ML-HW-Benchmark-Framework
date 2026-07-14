import weakref
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


class _SealedAccountingState:
    __slots__ = (
        "lock",
        "started_ns",
        "has_events",
        "counters",
        "invalid_reasons",
        "inflight_last_ns",
        "inflight_value",
        "inflight_area",
        "inflight_minimum",
        "inflight_maximum",
        "queue_last_ns",
        "queue_value",
        "queue_area",
        "queue_minimum",
        "queue_maximum",
        "queue_transitions",
        "queue_failed_sequences",
        "queue_duplicate_same",
        "queue_duplicate_conflict",
        "legacy_queue_events",
        "queue_sequence_high_water",
        "queue_latched_missing_ranges",
    )

    def __init__(self, started_ns: int):
        self.lock = Lock()
        self.started_ns = started_ns
        self.has_events = False
        self.counters = Counter()
        self.invalid_reasons = set()
        self.inflight_last_ns = started_ns
        self.inflight_value = 0
        self.inflight_area = 0
        self.inflight_minimum = 0
        self.inflight_maximum = 0
        self.queue_last_ns = started_ns
        self.queue_value = 0
        self.queue_area = 0
        self.queue_minimum = 0
        self.queue_maximum = 0
        self.queue_transitions = {}
        self.queue_failed_sequences = set()
        self.queue_duplicate_same = 0
        self.queue_duplicate_conflict = 0
        self.legacy_queue_events = 0
        self.queue_sequence_high_water = 0
        self.queue_latched_missing_ranges = []


_SEALED_ACCOUNTING_REGISTRY = {}
_SEALED_ACCOUNTING_REGISTRY_LOCK = Lock()


def _discard_sealed_accounting(reference, identity: int) -> None:
    with _SEALED_ACCOUNTING_REGISTRY_LOCK:
        current = _SEALED_ACCOUNTING_REGISTRY.get(identity)
        if current is not None and current[0] is reference:
            _SEALED_ACCOUNTING_REGISTRY.pop(identity, None)


def _register_sealed_accounting(metrics, started_ns: int):
    identity = id(metrics)
    reference = weakref.ref(
        metrics,
        lambda item, key=identity: _discard_sealed_accounting(item, key),
    )
    state = _SealedAccountingState(started_ns)
    with _SEALED_ACCOUNTING_REGISTRY_LOCK:
        _SEALED_ACCOUNTING_REGISTRY[identity] = (reference, state)
    return state


def _sealed_accounting(metrics):
    identity = id(metrics)
    with _SEALED_ACCOUNTING_REGISTRY_LOCK:
        entry = _SEALED_ACCOUNTING_REGISTRY.get(identity)
        if entry is not None and entry[0]() is metrics:
            return entry[1]
    raise RuntimeError("sealed metrics accounting state is unavailable")


def _reset_inflight_locked(state, started_ns: int) -> None:
    state.inflight_last_ns = started_ns
    state.inflight_value = 0
    state.inflight_area = 0
    state.inflight_minimum = 0
    state.inflight_maximum = 0


def _update_inflight_locked(state, value: int, now_ns: int) -> None:
    effective_ns = max(now_ns, state.inflight_last_ns)
    state.inflight_area += state.inflight_value * (
        effective_ns - state.inflight_last_ns
    )
    state.inflight_value = value
    state.inflight_last_ns = effective_ns
    state.inflight_minimum = min(state.inflight_minimum, value)
    state.inflight_maximum = max(state.inflight_maximum, value)


def _inflight_summary_locked(state, end_ns: int):
    _update_inflight_locked(state, state.inflight_value, end_ns)
    duration = max(1, end_ns - state.started_ns)
    return {
        "min": state.inflight_minimum,
        "max": state.inflight_maximum,
        "mean": state.inflight_area / duration,
    }


def _reset_queue_depth_locked(state, started_ns: int) -> None:
    state.queue_last_ns = started_ns
    state.queue_value = 0
    state.queue_area = 0
    state.queue_minimum = 0
    state.queue_maximum = 0


def _update_queue_depth_locked(state, value: int, now_ns: int) -> None:
    effective_ns = max(now_ns, state.queue_last_ns)
    state.queue_area += state.queue_value * (effective_ns - state.queue_last_ns)
    state.queue_value = value
    state.queue_last_ns = effective_ns
    state.queue_minimum = min(state.queue_minimum, value)
    state.queue_maximum = max(state.queue_maximum, value)


def _queue_depth_summary_locked(state, end_ns: int):
    _update_queue_depth_locked(state, state.queue_value, end_ns)
    duration = max(1, end_ns - state.started_ns)
    return {
        "min": state.queue_minimum,
        "max": state.queue_maximum,
        "mean": state.queue_area / duration,
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


def _record_queue_depth_event_locked(state, depth, now_ns, sequence) -> None:
    if sequence is None:
        state.legacy_queue_events += 1
        _update_queue_depth_locked(state, depth, now_ns)
        return
    state.queue_sequence_high_water = max(
        state.queue_sequence_high_water,
        sequence,
    )
    existing = state.queue_transitions.get(sequence)
    transition = (depth, now_ns)
    if existing is None:
        state.queue_transitions[sequence] = transition
        return
    if existing == transition:
        state.queue_duplicate_same += 1
    else:
        state.queue_duplicate_conflict += 1
        state.invalid_reasons.add("metrics_unavailable")


def _record_queue_sequence_allocated(metrics, sequence: int) -> None:
    state = _sealed_accounting(metrics)
    with state.lock:
        state.has_events = True
        state.queue_sequence_high_water = max(
            state.queue_sequence_high_water,
            sequence,
        )


def _commit_acceptance_internal(
    metrics,
    now_ns: int,
    queue_depth: int,
    queue_transition=None,
) -> None:
    state = _sealed_accounting(metrics)
    with state.lock:
        state.has_events = True
        if queue_transition is not None:
            state.queue_sequence_high_water = max(
                state.queue_sequence_high_water,
                queue_transition.sequence,
            )
        state.counters["accepted"] += 1
        outstanding = state.counters["accepted"] - state.counters["terminal"]
        _update_inflight_locked(state, outstanding, now_ns)
        if queue_transition is not None:
            _record_queue_depth_event_locked(
                state,
                queue_transition.depth,
                queue_transition.now_ns,
                queue_transition.sequence,
            )
        else:
            state.legacy_queue_events += 1
            _update_queue_depth_locked(state, queue_depth, now_ns)


def _record_rejected_internal(metrics, reason: str) -> None:
    state = _sealed_accounting(metrics)
    with state.lock:
        state.has_events = True
        state.counters["rejected"] += 1
        state.counters[f"rejected:{reason}"] += 1
        state.invalid_reasons.add("request_rejected")


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
        _register_sealed_accounting(self, started_ns)
        self.warnings = set()
        self.error_types = Counter()
        self.error_request_examples = {}
        self.queue_depth = TimeWeightedGauge(started_ns)
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

    @property
    def counters(self):
        state = _sealed_accounting(self)
        with state.lock:
            return Counter(state.counters)

    @property
    def invalid_reasons(self):
        state = _sealed_accounting(self)
        with state.lock:
            return set(state.invalid_reasons)

    def begin_measurement(self, started_ns: int) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            if state.has_events:
                raise RuntimeError("measurement already contains events")
            self.started_ns = started_ns
            state.started_ns = started_ns
            state.counters.clear()
            state.invalid_reasons.clear()
            self.queue_depth = TimeWeightedGauge(started_ns)
            state.queue_transitions = {}
            state.queue_failed_sequences = set()
            state.queue_duplicate_same = 0
            state.queue_duplicate_conflict = 0
            state.legacy_queue_events = 0
            state.queue_sequence_high_water = 0
            state.queue_latched_missing_ranges = []
            self.inflight = TimeWeightedGauge(started_ns)
            _reset_inflight_locked(state, started_ns)
            _reset_queue_depth_locked(state, started_ns)

    def record_submitted(self) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.counters["submitted"] += 1

    def claim_acceptance(self, queue_transition=None):
        if getattr(self._acceptance_local, "claim", None) is not None:
            raise RuntimeError("acceptance claim already active")
        state = _sealed_accounting(self)
        with state.lock:
            accepted_before = state.counters["accepted"]
        claim = _AcceptanceClaim(accepted_before, queue_transition)
        self._acceptance_local.claim = claim
        return claim

    def finish_acceptance(self, claim) -> bool:
        active = getattr(self._acceptance_local, "claim", None)
        if active is not claim or claim.closed:
            raise RuntimeError("acceptance claim is not active")
        state = _sealed_accounting(self)
        with state.lock:
            if state.counters["accepted"] > claim.accepted_before:
                claim.committed = True
        claim.closed = True
        del self._acceptance_local.claim
        return claim.committed

    def preflight_acceptance(self, _request) -> None:
        return None

    def record_accepted(self, now_ns: int, queue_depth: int) -> None:
        self.commit_acceptance(now_ns, queue_depth)

    def commit_acceptance(self, now_ns: int, queue_depth: int) -> None:
        claim = getattr(self._acceptance_local, "claim", None)
        transition = None if claim is None else claim.queue_transition
        _commit_acceptance_internal(
            self,
            now_ns,
            queue_depth,
            queue_transition=transition,
        )
        if claim is not None:
            claim.committed = True

    def record_rejected(self, reason: str) -> None:
        _record_rejected_internal(self, reason)

    def record_queue_depth(
        self,
        depth: int,
        now_ns: int,
        sequence: int | None = None,
    ) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            _record_queue_depth_event_locked(
                state,
                depth,
                now_ns,
                sequence,
            )

    def record_queue_depth_failure(self, sequence: int) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.queue_failed_sequences.add(sequence)
            state.invalid_reasons.add("metrics_unavailable")

    def record_queue_full(self) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.counters["queue_full_events"] += 1

    def record_worker_busy(
        self,
        worker_id: int,
        started_ns: int,
        finished_ns: int,
        batch_size: int = 1,
        sample_count: int | None = None,
    ) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            if finished_ns < started_ns:
                state.invalid_reasons.add("timing_invariant_failed")
                return
            self.worker_busy_ns[worker_id] += finished_ns - started_ns
            self.worker_batches[worker_id] += 1
            self.worker_samples[worker_id] += (
                batch_size if sample_count is None else sample_count
            )
            self.batch_sizes.add(batch_size)

    def add_invalid_reason(self, reason: str) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.invalid_reasons.add(reason)

    def add_warning(self, warning: str) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            self.warnings.add(warning)

    def record_first_token(self, request, event) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            if event.first_token_ns < request.issued_ns:
                state.invalid_reasons.add("timing_invariant_failed")
                return
            state.counters["first_token_events"] += 1
            self.timings["ttft_event"].add(
                (event.first_token_ns - request.issued_ns) / 1_000_000.0
            )

    def record_generation(self, generated_tokens: int, timing_ms) -> None:
        if generated_tokens <= 0:
            return
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.counters["completed_tokens"] += generated_tokens
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
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            status = trace.status.value
            state.counters[status] += 1
            state.counters[f"{status}_samples"] += trace.sample_count
            state.counters["terminal"] += 1
            if trace.status is TerminalStatus.FAILED:
                state.invalid_reasons.add("request_failed")
            if trace.timed_out:
                state.counters["timed_out"] += 1
                state.invalid_reasons.add("request_timeout")
            if trace.error_type:
                self.error_types[trace.error_type] += 1
                examples = self.error_request_examples.setdefault(
                    trace.error_type,
                    [],
                )
                if len(examples) < 5:
                    examples.append(trace.request_id)
            _update_inflight_locked(
                state,
                state.counters["accepted"] - state.counters["terminal"],
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
                state.invalid_reasons.add("timing_invariant_failed")
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
                state.counters["over_latency_slo"] += 1
            timing_sum = (
                values["submit_wait"]
                + values["queue_wait"]
                + values["service_time"]
                + values["completion_overhead"]
            )
            if abs(values["e2e_latency"] - timing_sum) > 50_000:
                state.invalid_reasons.add("timing_invariant_failed")

    @staticmethod
    def _missing_sequence_ranges(sequences, maximum):
        missing = []
        expected = 1
        for sequence in sequences:
            if sequence > expected:
                missing.append([expected, sequence - 1])
            expected = max(expected, sequence + 1)
        if expected <= maximum:
            missing.append([expected, maximum])
        return missing

    @staticmethod
    def _merge_sequence_ranges(ranges):
        merged = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1] + 1:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        return merged

    def _queue_summary(self, state, end_ns: int):
        sequences = sorted(state.queue_transitions)
        evidence = set(sequences)
        evidence.update(state.queue_failed_sequences)
        maximum = max(
            state.queue_sequence_high_water,
            max(evidence, default=0),
        )
        current_missing = self._missing_sequence_ranges(sequences, maximum)
        if current_missing:
            state.queue_latched_missing_ranges = self._merge_sequence_ranges(
                state.queue_latched_missing_ranges + current_missing
            )
        missing = [list(item) for item in state.queue_latched_missing_ranges]
        mixed_observations = bool(sequences and state.legacy_queue_events)
        sequence_valid = not (
            missing
            or state.queue_failed_sequences
            or state.queue_duplicate_conflict
            or mixed_observations
        )
        if not sequence_valid:
            state.invalid_reasons.add("metrics_unavailable")

        if sequences and sequence_valid:
            gauge = TimeWeightedGauge(state.started_ns)
            for sequence in sequences:
                depth, now_ns = state.queue_transitions[sequence]
                gauge.update(depth, now_ns)
            summary = gauge.summary(end_ns, state.started_ns)
        elif (
            sequence_valid
            and not sequences
            and not state.queue_failed_sequences
        ):
            summary = _queue_depth_summary_locked(state, end_ns)
        else:
            summary = {"min": None, "max": None, "mean": None}

        return {
            **summary,
            "sequence_valid": sequence_valid,
            "sequence_high_water": maximum,
            "event_count": len(state.queue_transitions),
            "legacy_event_count": state.legacy_queue_events,
            "missing_sequence_ranges": missing,
            "failed_sequences": sorted(state.queue_failed_sequences),
            "duplicate_same": state.queue_duplicate_same,
            "duplicate_conflict": state.queue_duplicate_conflict,
        }

    def finalize(self, end_ns: int) -> Dict[str, Dict[str, Any]]:
        state = _sealed_accounting(self)
        with state.lock:
            counters = state.counters
            submitted = counters["submitted"]
            accepted = counters["accepted"]
            rejected = counters["rejected"]
            completed = counters["completed"]
            completed_samples = counters["completed_samples"]
            failed = counters["failed"]
            outstanding = accepted - completed - failed
            invariant_valid = (
                submitted == accepted + rejected
                and accepted == completed + failed + outstanding
                and outstanding >= 0
            )
            if not invariant_valid:
                state.invalid_reasons.add("counter_invariant_failed")
            if outstanding:
                state.invalid_reasons.add("flush_timeout")

            duration_ns = max(1, end_ns - state.started_ns)
            duration_sec = duration_ns / 1_000_000_000.0
            queue = self._queue_summary(state, end_ns)
            inflight = _inflight_summary_locked(state, end_ns)
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
                state.invalid_reasons.add("timing_invariant_failed")
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
                "async_timed_out_requests": counters["timed_out"],
                "async_over_latency_slo_requests": counters[
                    "over_latency_slo"
                ],
                "async_outstanding_requests": outstanding,
                "async_issued_requests_per_sec": submitted / duration_sec,
                "async_completed_samples_per_sec": completed_samples / duration_sec,
                "async_completed_tokens_per_sec": (
                    counters["completed_tokens"] / duration_sec
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
                    "started_monotonic_ns": state.started_ns,
                    "ended_monotonic_ns": end_ns,
                    "duration_sec": duration_sec,
                },
                "invalid_reasons": sorted(state.invalid_reasons),
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
                "counts": dict(counters),
                "timing_ms": timing,
                "queue": {
                    "depth_min": queue["min"],
                    "depth_max": queue["max"],
                    "depth_mean": queue["mean"],
                    "sequence_valid": queue["sequence_valid"],
                    "sequence_high_water": queue["sequence_high_water"],
                    "event_count": queue["event_count"],
                    "legacy_event_count": queue["legacy_event_count"],
                    "missing_sequence_ranges": queue[
                        "missing_sequence_ranges"
                    ],
                    "failed_sequences": queue["failed_sequences"],
                    "duplicate_same": queue["duplicate_same"],
                    "duplicate_conflict": queue["duplicate_conflict"],
                    "full_events": counters["queue_full_events"],
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
                    "completed_tokens": counters["completed_tokens"],
                    "timing_sources": dict(self.generation_timing_sources),
                    "event_ttft_ms": timing["ttft_event"],
                    "reported_ttft_ms": timing["reported_ttft"],
                    "reported_tpot_ms": timing["reported_tpot"],
                },
            }
            return {"summary": summary, "details": details}

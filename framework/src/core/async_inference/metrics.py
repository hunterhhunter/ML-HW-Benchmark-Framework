import weakref
from array import array
from collections import Counter
from dataclasses import dataclass
from threading import Lock, RLock, local
from typing import Any, Dict

import numpy as np

from ..runtime_executor import (
    GenerationObservation,
    GenerationOutputEvent,
)
from .types import InferenceRequest, RequestTrace, TerminalStatus


PERCENTILES = (50.0, 85.0, 90.0, 95.0, 97.0, 99.0, 99.9)
PERCENTILE_METHOD = "linear"


@dataclass(frozen=True)
class GenerationTimingSample:
    source: str
    request_ttft_ms: float
    backend_ttft_ms: float
    request_mean_tpot_ms: float | None
    generated_tokens: int
    stream_event_itl_ms: tuple[float, ...]
    exact_stream: bool


def derive_generation_timing(
    generated_tokens: int,
    observation,
    requests,
) -> tuple[GenerationTimingSample | None, str | None]:
    """Derive request-level generation timing without guessing token arrivals."""
    if observation is None:
        return None, None
    if type(requests) is not tuple:
        try:
            requests = tuple(requests)
        except (TypeError, ValueError, OverflowError):
            return None, "timing_invariant_failed"
    if len(requests) != 1:
        return None, "generation_timing_batch_ambiguous"
    if type(observation) is not GenerationObservation:
        return None, "timing_invariant_failed"
    if type(observation.events) is not tuple or not observation.events:
        return None, "timing_invariant_failed"

    try:
        generated_tokens = _exact_int(generated_tokens)
        issued_ns = _exact_int(requests[0].issued_ns)
        backend_submitted_ns = _exact_int(observation.backend_submitted_ns)
        source = _exact_str(observation.source)
        events = tuple(
            (
                _exact_int(event.observed_ns),
                _exact_int(event.cumulative_tokens),
            )
            for event in observation.events
            if type(event) is GenerationOutputEvent
        )
    except (TypeError, ValueError, OverflowError):
        return None, "timing_invariant_failed"
    if (
        len(events) != len(observation.events)
        or not source
        or len(source) > 128
        or generated_tokens <= 0
        or backend_submitted_ns < 0
    ):
        return None, "timing_invariant_failed"

    previous_ns = backend_submitted_ns
    previous_count = 0
    exact_stream = True
    event_itl_ms = []
    for index, (observed_ns, cumulative_tokens) in enumerate(events):
        if (
            observed_ns < previous_ns
            or cumulative_tokens <= previous_count
            or cumulative_tokens > generated_tokens
        ):
            return None, "timing_invariant_failed"
        if cumulative_tokens - previous_count != 1:
            exact_stream = False
        if index:
            event_itl_ms.append((observed_ns - previous_ns) / 1_000_000.0)
        previous_ns = observed_ns
        previous_count = cumulative_tokens

    first_ns = events[0][0]
    last_ns, final_tokens = events[-1]
    if (
        first_ns < issued_ns
        or first_ns < backend_submitted_ns
        or final_tokens != generated_tokens
    ):
        return None, "timing_invariant_failed"
    exact_stream = exact_stream and len(events) == generated_tokens
    return (
        GenerationTimingSample(
            source=source,
            request_ttft_ms=(first_ns - issued_ns) / 1_000_000.0,
            backend_ttft_ms=(first_ns - backend_submitted_ns) / 1_000_000.0,
            request_mean_tpot_ms=(
                None
                if generated_tokens <= 1
                else (last_ns - first_ns)
                / (generated_tokens - 1)
                / 1_000_000.0
            ),
            generated_tokens=generated_tokens,
            stream_event_itl_ms=(
                tuple(event_itl_ms) if exact_stream else ()
            ),
            exact_stream=exact_stream,
        ),
        None,
    )


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
                "p85": None,
                "p90": None,
                "p95": None,
                "p97": None,
                "p99": None,
                "p99_9": None,
            }
        values = np.frombuffer(self.values, dtype=np.float64)
        percentiles = np.percentile(
            values,
            PERCENTILES,
            method=PERCENTILE_METHOD,
        )
        return {
            "count": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "sum": float(values.sum()),
            "p50": float(percentiles[0]),
            "p85": float(percentiles[1]),
            "p90": float(percentiles[2]),
            "p95": float(percentiles[3]),
            "p97": float(percentiles[4]),
            "p99": float(percentiles[5]),
            "p99_9": float(percentiles[6]),
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
        "legacy_queue_transitions",
        "queue_sequence_high_water",
        "queue_latched_missing_ranges",
        "outcome_accounting_dirty",
        "outcomes",
        "next_attempt_token",
        "next_legacy_outcome",
        "terminal_times",
        "worker_busy_ns",
        "worker_batches",
        "worker_samples",
        "batch_sizes",
        "timings",
        "warnings",
        "error_types",
        "error_request_examples",
        "generation_timing_sources",
    )

    def __init__(self, started_ns: int):
        self.lock = Lock()
        self.started_ns = started_ns
        self.has_events = False
        self.counters = {}
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
        self.legacy_queue_transitions = []
        self.queue_sequence_high_water = 0
        self.queue_latched_missing_ranges = []
        self.outcome_accounting_dirty = False
        self.outcomes = {}
        self.next_attempt_token = 0
        self.next_legacy_outcome = -1
        self.terminal_times = {}
        self.worker_busy_ns = {}
        self.worker_batches = {}
        self.worker_samples = {}
        self.batch_sizes = []
        self.timings = {
            name: []
            for name in (
                "scheduler_delay",
                "submit_wait",
                "queue_wait",
                "service_time",
                "completion_overhead",
                "e2e_latency",
                "ttft_event",
                "reported_ttft",
                "reported_tpot",
                "generation_request_ttft",
                "generation_backend_ttft",
                "generation_request_mean_tpot",
                "generation_tokens_per_request",
                "generation_stream_event_itl",
            )
        }
        self.warnings = set()
        self.error_types = {}
        self.error_request_examples = {}
        self.generation_timing_sources = {}


_SEALED_ACCOUNTING_REGISTRY = {}
_SEALED_ACCOUNTING_REGISTRY_LOCK = RLock()


def _discard_sealed_accounting(reference, identity: int) -> None:
    with _SEALED_ACCOUNTING_REGISTRY_LOCK:
        current = _SEALED_ACCOUNTING_REGISTRY.get(identity)
        if current is not None and current[0] is reference:
            _SEALED_ACCOUNTING_REGISTRY.pop(identity, None)


def _register_sealed_accounting(metrics, started_ns: int):
    started_ns = _exact_int(started_ns)
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


def _exact_int(value) -> int:
    converted = int(value)
    return converted if type(converted) is int else int(str(converted))


def _exact_float(value) -> float:
    converted = float(value)
    return converted if type(converted) is float else float(str(converted))


def _exact_str(value) -> str:
    converted = str(value)
    if type(converted) is str:
        return converted
    return converted.encode("utf-8").decode("utf-8")


def _increment(mapping, key, amount=1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


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


def _summarize_values(values):
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "sum": 0.0,
            "p50": None,
            "p85": None,
            "p90": None,
            "p95": None,
            "p97": None,
            "p99": None,
            "p99_9": None,
        }
    data = np.asarray(values, dtype=np.float64)
    percentiles = np.percentile(
        data,
        PERCENTILES,
        method=PERCENTILE_METHOD,
    )
    return {
        "count": int(data.size),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "sum": float(data.sum()),
        "p50": float(percentiles[0]),
        "p85": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p97": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "p99_9": float(percentiles[6]),
    }


def _summarize_gauge_events(events, started_ns, end_ns):
    last_ns = started_ns
    value = 0
    area = 0
    minimum = 0
    maximum = 0
    for next_value, observed_ns in events:
        effective_ns = max(observed_ns, last_ns)
        area += value * (effective_ns - last_ns)
        value = next_value
        last_ns = effective_ns
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    area += value * (max(end_ns, last_ns) - last_ns)
    return {
        "min": minimum,
        "max": maximum,
        "mean": area / max(1, end_ns - started_ns),
    }


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


def _merge_sequence_ranges(ranges):
    merged = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def _queue_summary_locked(state, end_ns):
    sequences = sorted(state.queue_transitions)
    evidence = set(sequences)
    evidence.update(state.queue_failed_sequences)
    maximum = max(state.queue_sequence_high_water, max(evidence, default=0))
    current_missing = _missing_sequence_ranges(sequences, maximum)
    if current_missing:
        state.queue_latched_missing_ranges = _merge_sequence_ranges(
            state.queue_latched_missing_ranges + current_missing
        )
    missing = [list(item) for item in state.queue_latched_missing_ranges]
    sequence_valid = not (
        missing
        or state.queue_failed_sequences
        or state.queue_duplicate_conflict
        or bool(sequences and state.legacy_queue_events)
    )
    if not sequence_valid:
        state.invalid_reasons.add("metrics_unavailable")
    if sequences and sequence_valid:
        summary = _summarize_gauge_events(
            [state.queue_transitions[sequence] for sequence in sequences],
            state.started_ns,
            end_ns,
        )
    elif sequence_valid and not state.queue_failed_sequences:
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
        state.legacy_queue_transitions.append(
            ("explicit", None, (depth, now_ns))
        )
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


def _next_outcome_key_locked(state):
    key = state.next_legacy_outcome
    state.next_legacy_outcome -= 1
    return key


def _apply_accepted_outcome_locked(state, record) -> None:
    (
        _request_id,
        now_ns,
        queue_depth,
        sequence,
        depth,
        observed_ns,
    ) = record
    _increment(state.counters, "accepted")
    if sequence is None:
        state.legacy_queue_events += 1
        _update_queue_depth_locked(state, queue_depth, now_ns)
    else:
        _record_queue_depth_event_locked(
            state,
            depth,
            observed_ns,
            sequence,
        )
    _update_inflight_locked(
        state,
        state.inflight_value + 1,
        now_ns,
    )


def _apply_rejected_outcome_locked(state, record) -> None:
    _request_id, reason, evidence = record
    _increment(state.counters, "rejected")
    _increment(state.counters, f"rejected:{reason}")
    state.invalid_reasons.add(evidence)


def _apply_terminal_inflight_locked(state, completed_ns: int) -> None:
    _update_inflight_locked(
        state,
        state.inflight_value - 1,
        completed_ns,
    )


def _rebuild_outcome_accounting_locked(state) -> None:
    state.outcome_accounting_dirty = True
    accepted = []
    rejected = []
    for kind, record in state.outcomes.values():
        if kind == "accepted":
            accepted.append(record)
        else:
            rejected.append(record)

    counters = {
        key: value
        for key, value in state.counters.items()
        if key not in {"accepted", "rejected"}
        and not key.startswith("rejected:")
    }
    if accepted:
        counters["accepted"] = len(accepted)
    if rejected:
        counters["rejected"] = len(rejected)

    invalid_reasons = set(state.invalid_reasons)
    invalid_reasons.discard("request_rejected")
    for _request_id, reason, evidence in rejected:
        _increment(counters, f"rejected:{reason}")
        invalid_reasons.add(evidence)

    accepted_sequences = {
        record[3] for record in accepted if record[3] is not None
    }
    queue_transitions = {
        sequence: transition
        for sequence, transition in state.queue_transitions.items()
        if sequence not in accepted_sequences
    }
    queue_transitions.update(
        {
            sequence: (depth, observed_ns)
            for (
                _request_id,
                _now_ns,
                _queue_depth,
                sequence,
                depth,
                observed_ns,
            ) in accepted
            if sequence is not None
        }
    )
    queue_sequence_high_water = state.queue_sequence_high_water
    if accepted_sequences:
        queue_sequence_high_water = max(
            queue_sequence_high_water,
            max(accepted_sequences),
        )

    legacy_queue_transitions = []
    queue_last_ns = state.started_ns
    queue_value = 0
    queue_area = 0
    queue_minimum = 0
    queue_maximum = 0
    for source, outcome_key, payload in state.legacy_queue_transitions:
        if source == "accepted":
            if state.outcomes.get(outcome_key) != ("accepted", payload):
                continue
            next_value = payload[2]
            observed_ns = payload[1]
        else:
            next_value, observed_ns = payload
        legacy_queue_transitions.append((source, outcome_key, payload))
        effective_ns = max(observed_ns, queue_last_ns)
        queue_area += queue_value * (effective_ns - queue_last_ns)
        queue_value = next_value
        queue_last_ns = effective_ns
        queue_minimum = min(queue_minimum, queue_value)
        queue_maximum = max(queue_maximum, queue_value)

    inflight_last_ns = state.started_ns
    inflight_value = 0
    inflight_area = 0
    inflight_minimum = 0
    inflight_maximum = 0
    events = [(record[1], 1) for record in accepted]
    events.extend((when, -1) for when in state.terminal_times.values())
    for when, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        effective_ns = max(when, inflight_last_ns)
        inflight_area += inflight_value * (effective_ns - inflight_last_ns)
        inflight_value += delta
        inflight_last_ns = effective_ns
        inflight_minimum = min(inflight_minimum, inflight_value)
        inflight_maximum = max(inflight_maximum, inflight_value)

    state.counters = counters
    state.invalid_reasons = invalid_reasons
    state.queue_transitions = queue_transitions
    state.queue_sequence_high_water = queue_sequence_high_water
    state.legacy_queue_transitions = legacy_queue_transitions
    state.legacy_queue_events = len(legacy_queue_transitions)
    state.queue_last_ns = queue_last_ns
    state.queue_value = queue_value
    state.queue_area = queue_area
    state.queue_minimum = queue_minimum
    state.queue_maximum = queue_maximum
    state.inflight_last_ns = inflight_last_ns
    state.inflight_value = inflight_value
    state.inflight_area = inflight_area
    state.inflight_minimum = inflight_minimum
    state.inflight_maximum = inflight_maximum
    state.outcome_accounting_dirty = False


def _accounting_outcome_internal(metrics, attempt_token: int):
    attempt_token = _exact_int(attempt_token)
    state = _sealed_accounting(metrics)
    with state.lock:
        outcome = state.outcomes.get(attempt_token)
        return None if outcome is None else outcome[0]


def _allocate_attempt_token_internal(metrics) -> int:
    state = _sealed_accounting(metrics)
    with state.lock:
        token = state.next_attempt_token
        state.next_attempt_token += 1
        return token


def _resolve_accounting_internal(metrics) -> None:
    state = _sealed_accounting(metrics)
    with state.lock:
        if state.outcome_accounting_dirty:
            _rebuild_outcome_accounting_locked(state)


def _record_queue_sequence_allocated(metrics, sequence: int) -> None:
    sequence = _exact_int(sequence)
    state = _sealed_accounting(metrics)
    with state.lock:
        state.has_events = True
        state.queue_sequence_high_water = max(
            state.queue_sequence_high_water,
            sequence,
        )


def _record_queue_sequence_failed_internal(metrics, sequence: int) -> None:
    sequence = _exact_int(sequence)
    state = _sealed_accounting(metrics)
    with state.lock:
        state.has_events = True
        state.queue_sequence_high_water = max(
            state.queue_sequence_high_water,
            sequence,
        )
        state.queue_failed_sequences.add(sequence)
        state.invalid_reasons.add("metrics_unavailable")


def _commit_acceptance_internal(
    metrics,
    now_ns: int,
    queue_depth: int,
    queue_transition=None,
    attempt_token: int | None = None,
    request_id: int | None = None,
) -> None:
    now_ns = _exact_int(now_ns)
    queue_depth = _exact_int(queue_depth)
    normalized_transition = None
    if queue_transition is not None:
        normalized_transition = (
            _exact_int(queue_transition.sequence),
            _exact_int(queue_transition.depth),
            _exact_int(queue_transition.now_ns),
        )
    normalized_attempt_token = (
        None if attempt_token is None else _exact_int(attempt_token)
    )
    normalized_request_id = (
        None if request_id is None else _exact_int(request_id)
    )
    state = _sealed_accounting(metrics)
    with state.lock:
        if state.outcome_accounting_dirty:
            _rebuild_outcome_accounting_locked(state)
        state.has_events = True
        key = (
            _next_outcome_key_locked(state)
            if normalized_attempt_token is None
            else normalized_attempt_token
        )
        effective_request_id = (
            key if normalized_request_id is None else normalized_request_id
        )
        existing = state.outcomes.get(key)
        if existing is not None and existing[0] != "accepted":
            raise RuntimeError("request already has rejected accounting")
        if existing is None:
            if normalized_transition is None:
                record = (
                    effective_request_id,
                    now_ns,
                    queue_depth,
                    None,
                    queue_depth,
                    now_ns,
                )
            else:
                sequence, depth, observed_ns = normalized_transition
                record = (
                    effective_request_id,
                    now_ns,
                    queue_depth,
                    sequence,
                    depth,
                    observed_ns,
                )
            state.outcome_accounting_dirty = True
            if record[3] is None:
                state.legacy_queue_transitions.append(
                    ("accepted", key, record)
                )
            state.outcomes[key] = ("accepted", record)
            _apply_accepted_outcome_locked(state, record)
            state.outcome_accounting_dirty = False


def _record_rejected_internal(
    metrics,
    reason: str,
    attempt_token: int | None = None,
    request_id: int | None = None,
) -> None:
    reason = _exact_str(reason)
    normalized_attempt_token = (
        None if attempt_token is None else _exact_int(attempt_token)
    )
    normalized_request_id = (
        None if request_id is None else _exact_int(request_id)
    )
    state = _sealed_accounting(metrics)
    with state.lock:
        if state.outcome_accounting_dirty:
            _rebuild_outcome_accounting_locked(state)
        state.has_events = True
        key = (
            _next_outcome_key_locked(state)
            if normalized_attempt_token is None
            else normalized_attempt_token
        )
        effective_request_id = (
            key if normalized_request_id is None else normalized_request_id
        )
        existing = state.outcomes.get(key)
        if existing is not None and existing[0] != "rejected":
            raise RuntimeError("request already has accepted accounting")
        if existing is None:
            record = (effective_request_id, reason, "request_rejected")
            state.outcome_accounting_dirty = True
            state.outcomes[key] = ("rejected", record)
            _apply_rejected_outcome_locked(state, record)
            state.outcome_accounting_dirty = False


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
            "generation_request_ttft": TimingDistribution(),
            "generation_backend_ttft": TimingDistribution(),
            "generation_request_mean_tpot": TimingDistribution(),
            "generation_tokens_per_request": TimingDistribution(),
            "generation_stream_event_itl": TimingDistribution(),
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

    def _try_begin_measurement(self, started_ns: int) -> bool:
        started_ns = _exact_int(started_ns)
        state = _sealed_accounting(self)
        with state.lock:
            if state.has_events:
                return False
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
            state.legacy_queue_transitions = []
            state.queue_sequence_high_water = 0
            state.queue_latched_missing_ranges = []
            state.outcome_accounting_dirty = False
            state.next_attempt_token = 0
            self.inflight = TimeWeightedGauge(started_ns)
            _reset_inflight_locked(state, started_ns)
            _reset_queue_depth_locked(state, started_ns)
            return True

    def begin_measurement(self, started_ns: int) -> None:
        if not self._try_begin_measurement(started_ns):
            raise RuntimeError("measurement already contains events")

    def try_begin_measurement(self, started_ns: int) -> bool:
        return self._try_begin_measurement(started_ns)

    def record_submitted(self) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            _increment(state.counters, "submitted")

    def claim_acceptance(self, queue_transition=None):
        if getattr(self._acceptance_local, "claim", None) is not None:
            raise RuntimeError("acceptance claim already active")
        state = _sealed_accounting(self)
        with state.lock:
            accepted_before = state.counters.get("accepted", 0)
        claim = _AcceptanceClaim(accepted_before, queue_transition)
        self._acceptance_local.claim = claim
        return claim

    def finish_acceptance(self, claim) -> bool:
        active = getattr(self._acceptance_local, "claim", None)
        if active is not claim or claim.closed:
            raise RuntimeError("acceptance claim is not active")
        state = _sealed_accounting(self)
        with state.lock:
            if state.counters.get("accepted", 0) > claim.accepted_before:
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
        depth = _exact_int(depth)
        now_ns = _exact_int(now_ns)
        sequence = None if sequence is None else _exact_int(sequence)
        state = _sealed_accounting(self)
        with state.lock:
            if sequence is None and state.outcome_accounting_dirty:
                _rebuild_outcome_accounting_locked(state)
            state.has_events = True
            if sequence is None:
                state.outcome_accounting_dirty = True
            _record_queue_depth_event_locked(state, depth, now_ns, sequence)
            if sequence is None:
                state.outcome_accounting_dirty = False

    def record_queue_depth_failure(self, sequence: int) -> None:
        _record_queue_sequence_failed_internal(self, sequence)

    def record_queue_full(self) -> None:
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            _increment(state.counters, "queue_full_events")

    def record_worker_busy(
        self,
        worker_id: int,
        started_ns: int,
        finished_ns: int,
        batch_size: int = 1,
        sample_count: int | None = None,
    ) -> None:
        worker_id = _exact_int(worker_id)
        started_ns = _exact_int(started_ns)
        finished_ns = _exact_int(finished_ns)
        batch_size = _exact_int(batch_size)
        sample_count = (
            None if sample_count is None else _exact_int(sample_count)
        )
        state = _sealed_accounting(self)
        with state.lock:
            if state.outcome_accounting_dirty:
                _rebuild_outcome_accounting_locked(state)
            state.has_events = True
            if finished_ns < started_ns:
                state.invalid_reasons.add("timing_invariant_failed")
                return
            _increment(
                state.worker_busy_ns,
                worker_id,
                finished_ns - started_ns,
            )
            _increment(state.worker_batches, worker_id)
            _increment(
                state.worker_samples,
                worker_id,
                batch_size if sample_count is None else sample_count
            )
            state.batch_sizes.append(batch_size)

    def add_invalid_reason(self, reason: str) -> None:
        reason = _exact_str(reason)
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.invalid_reasons.add(reason)

    def add_warning(self, warning: str) -> None:
        warning = _exact_str(warning)
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            state.warnings.add(warning)

    def record_first_token(self, request, event) -> None:
        issued_ns = _exact_int(request.issued_ns)
        first_token_ns = _exact_int(event.first_token_ns)
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            if first_token_ns < issued_ns:
                state.invalid_reasons.add("timing_invariant_failed")
                return
            _increment(state.counters, "first_token_events")
            state.timings["ttft_event"].append(
                (first_token_ns - issued_ns) / 1_000_000.0
            )

    def record_generation(
        self,
        generated_tokens: int,
        timing_ms,
        *,
        observation: GenerationObservation | None = None,
        requests: tuple[InferenceRequest, ...] = (),
    ) -> None:
        generated_tokens = _exact_int(generated_tokens)
        if generated_tokens <= 0:
            return
        timing = None
        if isinstance(timing_ms, dict):
            reported_ttft = timing_ms.get("ttft_ms")
            reported_tpot = timing_ms.get("tpot_ms")
            timing = (
                None if reported_ttft is None else _exact_float(reported_ttft),
                None if reported_tpot is None else _exact_float(reported_tpot),
                _exact_str(timing_ms.get("timing_source", "unknown")),
            )
        generation_sample, generation_diagnostic = derive_generation_timing(
            generated_tokens,
            observation,
            requests,
        )
        state = _sealed_accounting(self)
        with state.lock:
            state.has_events = True
            _increment(state.counters, "completed_tokens", generated_tokens)
            _increment(
                state.counters,
                "generation_stream_applicable_requests",
            )
            if timing is not None:
                reported_ttft, reported_tpot, timing_source = timing
                if reported_ttft is not None:
                    state.timings["reported_ttft"].append(reported_ttft)
                if reported_tpot is not None:
                    state.timings["reported_tpot"].append(reported_tpot)
                _increment(state.generation_timing_sources, timing_source)
            elif generation_sample is not None:
                _increment(
                    state.generation_timing_sources,
                    generation_sample.source,
                )

            if generation_diagnostic == "generation_timing_batch_ambiguous":
                state.warnings.add(generation_diagnostic)
            elif generation_diagnostic is not None:
                state.invalid_reasons.add(generation_diagnostic)
            if generation_sample is None:
                _increment(
                    state.counters,
                    "generation_stream_unobservable_requests",
                )
                state.warnings.add("generation_stream_itl_incomplete")
                return

            _increment(state.counters, "generation_observed_requests")
            state.timings["generation_request_ttft"].append(
                generation_sample.request_ttft_ms
            )
            state.timings["generation_backend_ttft"].append(
                generation_sample.backend_ttft_ms
            )
            state.timings["generation_tokens_per_request"].append(
                generation_sample.generated_tokens
            )
            if generation_sample.request_mean_tpot_ms is not None:
                state.timings["generation_request_mean_tpot"].append(
                    generation_sample.request_mean_tpot_ms
                )
            if generation_sample.exact_stream:
                _increment(state.counters, "generation_stream_exact_requests")
                _increment(
                    state.counters,
                    "generation_stream_itl_samples",
                    len(generation_sample.stream_event_itl_ms),
                )
                state.timings["generation_stream_event_itl"].extend(
                    generation_sample.stream_event_itl_ms
                )
            else:
                _increment(
                    state.counters,
                    "generation_stream_unobservable_requests",
                )
                state.warnings.add("generation_stream_itl_incomplete")

    def record_terminal(self, trace: RequestTrace) -> None:
        request_id = _exact_int(trace.request_id)
        sample_count = _exact_int(trace.sample_count)
        status = _exact_str(trace.status.value)
        error_type = None if not trace.error_type else _exact_str(trace.error_type)
        completed_ns = _exact_int(trace.completed_ns)
        timed_out = bool(trace.timed_out)
        failed = trace.status is TerminalStatus.FAILED
        latency_slo_ms = (
            None
            if self.latency_slo_ms is None
            else _exact_float(self.latency_slo_ms)
        )
        timestamps = tuple(
            _exact_int(value)
            for value in (
                trace.scheduled_ns,
                trace.issued_ns,
                trace.enqueued_ns,
                trace.runtime_started_ns,
                trace.runtime_finished_ns,
                trace.completed_ns,
            )
        )
        state = _sealed_accounting(self)
        with state.lock:
            if state.outcome_accounting_dirty:
                _rebuild_outcome_accounting_locked(state)
            state.has_events = True
            _increment(state.counters, status)
            _increment(state.counters, f"{status}_samples", sample_count)
            _increment(state.counters, "terminal")
            if failed:
                state.invalid_reasons.add("request_failed")
            if timed_out:
                _increment(state.counters, "timed_out")
                state.invalid_reasons.add("request_timeout")
            if error_type:
                _increment(state.error_types, error_type)
                examples = state.error_request_examples.setdefault(
                    error_type,
                    [],
                )
                if len(examples) < 5:
                    examples.append(request_id)
            if request_id not in state.terminal_times:
                state.outcome_accounting_dirty = True
                state.terminal_times[request_id] = completed_ns
                _apply_terminal_inflight_locked(state, completed_ns)
                state.outcome_accounting_dirty = False
            else:
                state.terminal_times[request_id] = completed_ns
                state.outcome_accounting_dirty = True
            if any(
                earlier_ns > later_ns
                for earlier_ns, later_ns in zip(timestamps, timestamps[1:])
            ):
                state.invalid_reasons.add("timing_invariant_failed")
                return

            ns_to_ms = 1.0 / 1_000_000.0
            values = {
                "scheduler_delay": timestamps[1] - timestamps[0],
                "submit_wait": timestamps[2] - timestamps[1],
                "queue_wait": timestamps[3] - timestamps[2],
                "service_time": timestamps[4] - timestamps[3],
                "completion_overhead": timestamps[5] - timestamps[4],
                "e2e_latency": timestamps[5] - timestamps[1],
            }
            for name, value_ns in values.items():
                state.timings[name].append(value_ns * ns_to_ms)
            if (
                latency_slo_ms is not None
                and values["e2e_latency"] * ns_to_ms > latency_slo_ms
            ):
                _increment(state.counters, "over_latency_slo")
            timing_sum = (
                values["submit_wait"]
                + values["queue_wait"]
                + values["service_time"]
                + values["completion_overhead"]
            )
            if abs(values["e2e_latency"] - timing_sum) > 50_000:
                state.invalid_reasons.add("timing_invariant_failed")

    def finalize(self, end_ns: int) -> Dict[str, Dict[str, Any]]:
        end_ns = _exact_int(end_ns)
        worker_count = _exact_int(self.worker_count)
        state = _sealed_accounting(self)
        with state.lock:
            _rebuild_outcome_accounting_locked(state)
            emitted_counts = dict(state.counters)
            counters = dict(emitted_counts)
            for key in (
                "submitted",
                "accepted",
                "rejected",
                "completed",
                "completed_samples",
                "failed",
                "timed_out",
                "over_latency_slo",
                "completed_tokens",
                "generation_observed_requests",
                "generation_stream_applicable_requests",
                "generation_stream_exact_requests",
                "generation_stream_unobservable_requests",
                "generation_stream_itl_samples",
                "queue_full_events",
            ):
                counters.setdefault(key, 0)
            submitted = counters["submitted"]
            accepted = counters["accepted"]
            rejected = counters["rejected"]
            completed = counters["completed"]
            completed_samples = counters["completed_samples"]
            failed = counters["failed"]
            outstanding = accepted - completed - failed
            emitted_counts["outstanding"] = outstanding
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
            queue = _queue_summary_locked(state, end_ns)
            inflight = _inflight_summary_locked(state, end_ns)
            timing_values = {
                name: tuple(values) for name, values in state.timings.items()
            }
            worker_busy = dict(state.worker_busy_ns)
            worker_batches = dict(state.worker_batches)
            worker_samples = dict(state.worker_samples)
            batch_sizes = tuple(state.batch_sizes)
            warnings = tuple(sorted(state.warnings))
            error_types = dict(state.error_types)
            error_examples = {
                key: tuple(values)
                for key, values in state.error_request_examples.items()
            }
            generation_sources = dict(state.generation_timing_sources)
            invalid_reasons = tuple(sorted(state.invalid_reasons))
            started_ns = state.started_ns
            total_busy = sum(worker_busy.values())
            worker_slots = max(1, worker_count)
            worker_capacity_ns = worker_slots * duration_ns
            if total_busy > worker_capacity_ns or any(
                busy_ns > duration_ns for busy_ns in worker_busy.values()
            ):
                state.invalid_reasons.add("timing_invariant_failed")
                invalid_reasons = tuple(sorted(state.invalid_reasons))
            utilization = min(
                1.0,
                total_busy / worker_capacity_ns,
            )
        timing = {
            name: _summarize_values(values)
            for name, values in timing_values.items()
        }
        batch_size = _summarize_values(batch_sizes)
        generation_observed_requests = counters[
            "generation_observed_requests"
        ]
        generation_stream_applicable_requests = counters[
            "generation_stream_applicable_requests"
        ]
        generation_exact_stream_requests = counters[
            "generation_stream_exact_requests"
        ]
        generation_stream_itl_coverage = (
            None
            if generation_stream_applicable_requests == 0
            else generation_exact_stream_requests
            / generation_stream_applicable_requests
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
                "async_generation_observed_requests": (
                    generation_observed_requests
                ),
        }
        for percentile in ("p50", "p85", "p90", "p95", "p99"):
            summary[
                f"async_generation_request_ttft_{percentile}_ms"
            ] = timing["generation_request_ttft"][percentile]
            summary[
                f"async_generation_request_mean_tpot_{percentile}_ms"
            ] = timing["generation_request_mean_tpot"][percentile]
        if generation_stream_itl_coverage == 1.0:
            for percentile in ("p50", "p85", "p90", "p95", "p99"):
                summary[
                    f"async_generation_stream_itl_{percentile}_ms"
                ] = timing["generation_stream_event_itl"][percentile]
        details = {
                "measurement_duration_sec": duration_sec,
                "measurement": {
                    "started_monotonic_ns": started_ns,
                    "ended_monotonic_ns": end_ns,
                    "duration_sec": duration_sec,
                },
                "statistics": {
                    "percentile_method": (
                        "numpy.percentile(method=linear)"
                    ),
                },
                "invalid_reasons": list(invalid_reasons),
                "warnings": list(warnings),
                "counter_invariants": {
                    "valid": invariant_valid,
                    "submitted_equals_accepted_plus_rejected": (
                        submitted == accepted + rejected
                    ),
                    "accepted_equals_terminal_plus_outstanding": (
                        accepted == completed + failed + outstanding
                    ),
                },
                "counts": emitted_counts,
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
                    "busy_ns": worker_busy,
                    "batches": worker_batches,
                    "samples": worker_samples,
                },
                "batch_size": batch_size,
                "failure_types": error_types,
                "failure_request_examples": {
                    error_type: list(request_ids)
                    for error_type, request_ids in error_examples.items()
                },
                "generation": {
                    "definitions": {
                        "request_ttft_ms": (
                            "issued_to_first_nonempty_stream_output"
                        ),
                        "backend_ttft_ms": (
                            "backend_submit_to_first_nonempty_stream_output"
                        ),
                        "request_mean_tpot_ms": (
                            "first_to_last_output_divided_by_generated_tokens_minus_one"
                        ),
                        "stream_event_itl_ms": (
                            "adjacent_single_token_python_stream_events"
                        ),
                    },
                    "completed_tokens": counters["completed_tokens"],
                    "timing_sources": generation_sources,
                    "event_ttft_ms": timing["ttft_event"],
                    "reported_ttft_ms": timing["reported_ttft"],
                    "reported_tpot_ms": timing["reported_tpot"],
                    "applicable_requests": (
                        generation_stream_applicable_requests
                    ),
                    "observed_requests": generation_observed_requests,
                    "exact_stream_requests": (
                        generation_exact_stream_requests
                    ),
                    "unobservable_stream_requests": counters[
                        "generation_stream_unobservable_requests"
                    ],
                    "stream_event_itl_samples": counters[
                        "generation_stream_itl_samples"
                    ],
                    "stream_event_itl_coverage": (
                        generation_stream_itl_coverage
                    ),
                    "request_ttft_ms": timing[
                        "generation_request_ttft"
                    ],
                    "backend_ttft_ms": timing[
                        "generation_backend_ttft"
                    ],
                    "request_mean_tpot_ms": timing[
                        "generation_request_mean_tpot"
                    ],
                    "generated_tokens_per_request": timing[
                        "generation_tokens_per_request"
                    ],
                    "stream_event_itl_ms": timing[
                        "generation_stream_event_itl"
                    ],
                },
        }
        return {"summary": summary, "details": details}

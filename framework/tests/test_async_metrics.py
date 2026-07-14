from concurrent.futures import ThreadPoolExecutor
import gc
import weakref

import pytest

import core.async_inference.metrics as metrics_module
from core.async_inference.metrics import (
    AsyncMetricsCollector,
    _SEALED_ACCOUNTING_REGISTRY,
    _record_queue_sequence_allocated,
)
from core.async_inference.types import (
    FirstTokenEvent,
    InferenceRequest,
    RequestTrace,
    TerminalStatus,
)


def make_trace(
    request_id,
    issued_ns,
    started_ns,
    finished_ns,
    completed_ns,
    *,
    sample_count=1,
):
    return RequestTrace(
        request_id=request_id,
        sample_index=request_id,
        status=TerminalStatus.COMPLETED,
        scheduled_ns=issued_ns,
        issued_ns=issued_ns,
        enqueued_ns=issued_ns,
        runtime_started_ns=started_ns,
        runtime_finished_ns=finished_ns,
        completed_ns=completed_ns,
        worker_id=0,
        batch_size=1,
        timed_out=False,
        sample_count=sample_count,
    )


def test_outcome_identity_is_normalized_before_sealed_lock():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    state = metrics_module._sealed_accounting(metrics)

    class GuardedInt(int):
        conversions = 0

        def __int__(self):
            type(self).conversions += 1
            assert not state.lock.locked()
            return super().__int__()

    metrics_module._commit_acceptance_internal(
        metrics,
        now_ns=1,
        queue_depth=1,
        attempt_token=GuardedInt(10),
        request_id=GuardedInt(20),
    )
    metrics_module._record_rejected_internal(
        metrics,
        "invalid_request",
        attempt_token=GuardedInt(11),
        request_id=GuardedInt(21),
    )

    assert GuardedInt.conversions == 4
    assert metrics_module._accounting_outcome_internal(metrics, 10) == "accepted"
    assert metrics_module._accounting_outcome_internal(metrics, 11) == "rejected"


def test_metrics_compute_exact_latency_decomposition_and_percentiles():
    metrics = AsyncMetricsCollector(
        started_ns=0,
        worker_count=1,
        latency_slo_ms=5.0,
    )
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_queue_depth(depth=0, now_ns=1_000_000)
    metrics.record_worker_busy(
        worker_id=0,
        started_ns=2_000_000,
        finished_ns=5_000_000,
    )
    metrics.record_terminal(
        make_trace(0, 0, 2_000_000, 5_000_000, 6_000_000)
    )

    result = metrics.finalize(end_ns=10_000_000)

    assert result["summary"]["async_completed_requests"] == 1
    assert result["details"]["timing_ms"]["queue_wait"]["p50"] == pytest.approx(
        2.0
    )
    assert result["details"]["timing_ms"]["service_time"]["mean"] == pytest.approx(
        3.0
    )
    assert result["details"]["timing_ms"]["e2e_latency"]["max"] == pytest.approx(
        6.0
    )
    assert result["details"]["queue"]["depth_mean"] == pytest.approx(0.1)
    assert result["details"]["workers"]["utilization"] == pytest.approx(0.3)
    assert result["summary"]["async_issued_requests_per_sec"] == pytest.approx(100.0)
    assert result["summary"]["async_over_latency_slo_requests"] == 1
    assert result["details"]["queue"]["submit_block_total_ms"] == pytest.approx(0.0)


def test_timing_distribution_reports_every_percentile_count_and_sum():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    for request_id, latency_ms in enumerate((1, 2, 3, 4)):
        metrics.record_submitted()
        metrics.record_accepted(now_ns=0, queue_depth=1)
        latency_ns = latency_ms * 1_000_000
        metrics.record_terminal(
            make_trace(request_id, 0, 0, latency_ns, latency_ns)
        )

    result = metrics.finalize(end_ns=5_000_000)

    e2e = result["details"]["timing_ms"]["e2e_latency"]
    assert e2e["count"] == 4
    assert e2e["sum"] == pytest.approx(10.0)
    assert {
        key: e2e[key]
        for key in ("p50", "p90", "p95", "p97", "p99", "p99_9")
    } == pytest.approx(
        {
            "p50": 2.5,
            "p90": 3.7,
            "p95": 3.85,
            "p97": 3.91,
            "p99": 3.97,
            "p99_9": 3.997,
        }
    )


def test_inflight_gauge_reports_exact_time_weighted_mean():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=2_000_000, queue_depth=2)
    metrics.record_terminal(
        make_trace(0, 0, 1_000_000, 4_000_000, 5_000_000)
    )
    metrics.record_terminal(
        make_trace(1, 2_000_000, 3_000_000, 8_000_000, 9_000_000)
    )

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["inflight_min"] == 0
    assert queue["inflight_max"] == 2
    assert queue["inflight_mean"] == pytest.approx(1.2)


def test_public_counter_snapshot_cannot_mutate_sealed_accounting():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    public_counts = metrics.counters
    public_counts["accepted"] = 99
    public_counts["terminal"] = 99

    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_terminal(
        make_trace(0, 0, 1_000_000, 2_000_000, 3_000_000)
    )
    result = metrics.finalize(end_ns=4_000_000)

    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True


def test_registry_normalizes_extension_values_and_releases_collector():
    class SelfReferencingReason(str):
        pass

    class ExtendedInt(int):
        pass

    metrics = AsyncMetricsCollector(started_ns=ExtendedInt(0), worker_count=1)
    identity = id(metrics)
    reference = weakref.ref(metrics)
    reason = SelfReferencingReason("external_error")
    reason.owner = metrics
    metrics.add_invalid_reason(reason)
    metrics.record_queue_depth(
        ExtendedInt(1),
        ExtendedInt(2),
        sequence=ExtendedInt(1),
    )
    state = _SEALED_ACCOUNTING_REGISTRY[identity][1]
    with state.lock:
        assert type(state.started_ns) is int
        assert type(state.counters) is dict
        assert type(state.invalid_reasons) is set
        assert all(type(item) is str for item in state.invalid_reasons)
        assert type(state.queue_transitions) is dict
        assert all(
            type(item) is int
            for sequence, transition in state.queue_transitions.items()
            for item in (sequence, *transition)
        )
    del reason
    del state
    del metrics
    gc.collect()

    assert reference() is None
    assert identity not in _SEALED_ACCOUNTING_REGISTRY


def test_public_aggregate_replacements_are_not_dispatched_under_sealed_lock():
    class ForbiddenAggregate:
        def __getattr__(self, name):
            raise AssertionError(f"public aggregate dispatch: {name}")

        def __getitem__(self, key):
            raise AssertionError(f"public aggregate lookup: {key}")

        def __setitem__(self, key, value):
            raise AssertionError(f"public aggregate write: {key}")

        def values(self):
            raise AssertionError("public aggregate values")

        def summary(self):
            raise AssertionError("public aggregate summary")

    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.worker_busy_ns = ForbiddenAggregate()
    metrics.worker_batches = ForbiddenAggregate()
    metrics.worker_samples = ForbiddenAggregate()
    metrics.batch_sizes = ForbiddenAggregate()
    metrics.timings = ForbiddenAggregate()
    metrics.error_types = ForbiddenAggregate()
    metrics.error_request_examples = ForbiddenAggregate()

    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_worker_busy(0, 1_000_000, 2_000_000)
    metrics.record_terminal(
        make_trace(0, 0, 1_000_000, 2_000_000, 3_000_000)
    )
    result = metrics.finalize(end_ns=4_000_000)

    assert result["summary"]["async_completed_requests"] == 1
    assert result["details"]["workers"]["busy_ns"] == {0: 1_000_000}
    assert result["details"]["batch_size"]["count"] == 1
    assert result["details"]["timing_ms"]["e2e_latency"]["count"] == 1


def test_timeout_is_diagnostic_subset_not_extra_terminal_count():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    trace = make_trace(0, 0, 1_000_000, 2_000_000, 3_000_000)
    metrics.record_terminal(RequestTrace(**{**trace.__dict__, "timed_out": True}))

    result = metrics.finalize(end_ns=4_000_000)

    summary = result["summary"]
    assert summary["async_completed_requests"] == 1
    assert summary["async_timed_out_requests"] == 1
    assert summary["async_outstanding_requests"] == 0
    assert "request_timeout" in result["details"]["invalid_reasons"]


def test_rejected_request_preserves_counter_invariants():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_rejected("queue_full")

    result = metrics.finalize(end_ns=1_000_000)

    assert result["details"]["counter_invariants"]["valid"] is True
    assert result["summary"]["async_rejected_requests"] == 1
    assert "request_rejected" in result["details"]["invalid_reasons"]


def test_begin_measurement_excludes_engine_startup_time():
    metrics = AsyncMetricsCollector(started_ns=1, worker_count=1)
    metrics.begin_measurement(started_ns=1_000_000_000)
    metrics.record_submitted()
    metrics.record_rejected("queue_full")

    result = metrics.finalize(end_ns=2_000_000_000)

    assert result["summary"]["async_issued_requests_per_sec"] == pytest.approx(1.0)


def test_begin_measurement_rejects_preexisting_worker_and_batch_state():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_worker_busy(
        worker_id=0,
        started_ns=0,
        finished_ns=1_000_000,
        batch_size=2,
    )

    with pytest.raises(RuntimeError, match="measurement already contains events"):
        metrics.begin_measurement(started_ns=2_000_000)


def test_out_of_order_gauge_observation_does_not_regress_timestamp():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth(depth=1, now_ns=5_000_000)
    metrics.record_queue_depth(depth=0, now_ns=3_000_000)

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["depth_mean"] == pytest.approx(0.0)
    assert queue["depth_min"] == 0
    assert queue["depth_max"] == 1


def test_queue_transition_sequence_restores_actual_event_time_and_order():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth(
        depth=0,
        now_ns=5_000_000,
        sequence=2,
    )
    metrics.record_queue_depth(
        depth=1,
        now_ns=2_000_000,
        sequence=1,
    )

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["depth_mean"] == pytest.approx(0.3)
    assert queue["depth_min"] == 0
    assert queue["depth_max"] == 1


def test_missing_queue_sequence_invalidates_and_omits_depth_summary():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth(
        depth=1,
        now_ns=2_000_000,
        sequence=1,
    )
    metrics.record_queue_depth(
        depth=0,
        now_ns=5_000_000,
        sequence=3,
    )

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["sequence_valid"] is False
    assert queue["event_count"] == 2
    assert queue["missing_sequence_ranges"] == [[2, 2]]
    assert queue["depth_min"] is None
    assert queue["depth_max"] is None
    assert queue["depth_mean"] is None
    assert result["summary"]["async_queue_depth_max"] is None
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]


def test_identical_queue_sequence_duplicate_is_detected_without_corruption():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth(1, 2_000_000, sequence=1)
    metrics.record_queue_depth(1, 2_000_000, sequence=1)
    metrics.record_queue_depth(0, 5_000_000, sequence=2)

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["sequence_valid"] is True
    assert queue["event_count"] == 2
    assert queue["duplicate_same"] == 1
    assert queue["duplicate_conflict"] == 0
    assert queue["depth_mean"] == pytest.approx(0.3)


def test_conflicting_queue_sequence_duplicate_invalidates_depth_summary():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth(1, 2_000_000, sequence=1)
    metrics.record_queue_depth(2, 3_000_000, sequence=1)

    result = metrics.finalize(end_ns=10_000_000)

    queue = result["details"]["queue"]
    assert queue["sequence_valid"] is False
    assert queue["event_count"] == 1
    assert queue["duplicate_same"] == 0
    assert queue["duplicate_conflict"] == 1
    assert queue["depth_mean"] is None
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]


def test_failed_queue_sequence_is_explicit_without_pending_followup_growth():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_queue_depth_failure(sequence=1)
    for sequence in range(2, 102):
        metrics.record_queue_depth(
            depth=sequence % 2,
            now_ns=sequence * 1_000_000,
            sequence=sequence,
        )

    result = metrics.finalize(end_ns=103_000_000)

    queue = result["details"]["queue"]
    assert queue["sequence_valid"] is False
    assert queue["event_count"] == 100
    assert queue["failed_sequences"] == [1]
    assert queue["missing_sequence_ranges"] == [[1, 1]]
    assert queue["depth_mean"] is None


def test_trailing_allocated_sequence_latches_missing_evidence_across_finalize():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    _record_queue_sequence_allocated(metrics, sequence=1)

    first = metrics.finalize(end_ns=10_000_000)
    second = metrics.finalize(end_ns=10_000_000)

    metrics.record_queue_depth(
        depth=1,
        now_ns=2_000_000,
        sequence=1,
    )
    after_late_delivery = metrics.finalize(end_ns=10_000_000)

    for result in (first, second, after_late_delivery):
        queue = result["details"]["queue"]
        assert queue["sequence_valid"] is False
        assert queue["sequence_high_water"] == 1
        assert queue["missing_sequence_ranges"] == [[1, 1]]
        assert queue["depth_min"] is None
        assert queue["depth_max"] is None
        assert queue["depth_mean"] is None
        assert "metrics_unavailable" in result["details"]["invalid_reasons"]


def test_request_sample_and_token_counts_remain_distinct():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=2)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_worker_busy(
        worker_id=1,
        started_ns=1_000_000,
        finished_ns=3_000_000,
        batch_size=1,
        sample_count=3,
    )
    metrics.record_generation(
        generated_tokens=7,
        timing_ms={
            "ttft_ms": 1.25,
            "tpot_ms": 0.5,
            "timing_source": "runtime",
        },
    )
    request = InferenceRequest(
        request_id=0,
        sample_index=0,
        sample={},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=0,
        sample_count=3,
    )
    metrics.record_first_token(request, FirstTokenEvent(0, 1_500_000))
    metrics.record_terminal(
        make_trace(
            0,
            0,
            1_000_000,
            3_000_000,
            4_000_000,
            sample_count=3,
        )
    )

    result = metrics.finalize(end_ns=10_000_000)

    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_completed_samples"] == 3
    assert result["summary"]["async_completed_samples_per_sec"] == pytest.approx(300.0)
    assert result["summary"]["async_completed_tokens_per_sec"] == pytest.approx(700.0)
    assert result["details"]["workers"]["samples"] == {1: 3}
    assert result["details"]["generation"]["completed_tokens"] == 7
    assert result["details"]["generation"]["event_ttft_ms"]["mean"] == pytest.approx(
        1.5
    )
    assert result["details"]["generation"]["reported_ttft_ms"][
        "mean"
    ] == pytest.approx(1.25)
    assert result["details"]["generation"]["reported_tpot_ms"][
        "mean"
    ] == pytest.approx(0.5)
    assert result["details"]["generation"]["timing_sources"] == {"runtime": 1}


def test_invalid_timing_and_counter_states_are_reported():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    request = InferenceRequest(
        request_id=7,
        sample_index=7,
        sample={},
        scheduled_ns=10,
        issued_ns=10,
        enqueued_ns=10,
    )
    metrics.record_first_token(request, FirstTokenEvent(7, 9))
    metrics.record_submitted()
    metrics.add_warning("observer_lag")
    metrics.add_invalid_reason("external_error")
    metrics.record_queue_full()

    result = metrics.finalize(end_ns=1_000_000_000)

    assert result["details"]["counter_invariants"]["valid"] is False
    assert result["details"]["queue"]["full_events"] == 1
    assert result["details"]["warnings"] == ["observer_lag"]
    assert result["details"]["invalid_reasons"] == [
        "counter_invariant_failed",
        "external_error",
        "timing_invariant_failed",
    ]


def test_invalid_request_timestamp_order_is_excluded_from_timing_distributions():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    trace = make_trace(0, 0, 2_000_000, 3_000_000, 4_000_000)
    metrics.record_terminal(
        RequestTrace(**{**trace.__dict__, "enqueued_ns": 2_500_000})
    )

    result = metrics.finalize(end_ns=5_000_000)

    assert result["summary"]["async_completed_requests"] == 1
    assert "timing_invariant_failed" in result["details"]["invalid_reasons"]
    for timing_name in (
        "scheduler_delay",
        "submit_wait",
        "queue_wait",
        "service_time",
        "completion_overhead",
        "e2e_latency",
    ):
        assert result["details"]["timing_ms"][timing_name]["count"] == 0


def test_failure_examples_are_capped_and_returned_as_isolated_snapshots():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    for request_id in range(7):
        metrics.record_submitted()
        metrics.record_accepted(now_ns=0, queue_depth=1)
        trace = make_trace(
            request_id,
            0,
            0,
            1_000_000,
            2_000_000,
        )
        metrics.record_terminal(
            RequestTrace(
                **{
                    **trace.__dict__,
                    "status": TerminalStatus.FAILED,
                    "error_type": "RuntimeError",
                }
            )
        )

    first = metrics.finalize(end_ns=3_000_000)
    first["details"]["failure_types"]["RuntimeError"] = 0
    first["details"]["failure_request_examples"]["RuntimeError"].append(99)
    second = metrics.finalize(end_ns=3_000_000)

    assert second["details"]["failure_types"] == {"RuntimeError": 7}
    assert second["details"]["failure_request_examples"] == {
        "RuntimeError": [0, 1, 2, 3, 4]
    }


def test_inverted_worker_interval_is_invalid_and_not_recorded():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_worker_busy(
        worker_id=0,
        started_ns=2_000_000,
        finished_ns=1_000_000,
        batch_size=2,
        sample_count=3,
    )

    result = metrics.finalize(end_ns=10_000_000)

    assert "timing_invariant_failed" in result["details"]["invalid_reasons"]
    assert result["details"]["workers"]["busy_ns"] == {}
    assert result["details"]["workers"]["batches"] == {}
    assert result["details"]["workers"]["samples"] == {}
    assert result["details"]["batch_size"]["count"] == 0


def test_over_capacity_worker_utilization_is_bounded_and_invalid():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    metrics.record_worker_busy(
        worker_id=0,
        started_ns=0,
        finished_ns=15_000_000,
    )

    result = metrics.finalize(end_ns=10_000_000)

    assert result["summary"]["async_worker_utilization"] == pytest.approx(1.0)
    assert result["details"]["workers"]["utilization"] == pytest.approx(1.0)
    assert "timing_invariant_failed" in result["details"]["invalid_reasons"]


def test_metrics_updates_are_thread_safe():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)

    def reject_requests(count):
        for _ in range(count):
            metrics.record_submitted()
            metrics.record_rejected("queue_full")

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(reject_requests, 250) for _ in range(8)]
        for future in futures:
            future.result()

    result = metrics.finalize(end_ns=1_000_000_000)

    assert result["summary"]["async_submitted_requests"] == 2_000
    assert result["summary"]["async_rejected_requests"] == 2_000
    assert result["details"]["counter_invariants"]["valid"] is True


def test_finalize_returns_exact_summary_and_detail_schema():
    result = AsyncMetricsCollector(started_ns=0, worker_count=1).finalize(
        end_ns=1_000_000_000
    )

    assert set(result) == {"summary", "details"}
    assert set(result["summary"]) == {
        "async_submitted_requests",
        "async_accepted_requests",
        "async_completed_requests",
        "async_completed_samples",
        "async_failed_requests",
        "async_rejected_requests",
        "async_timed_out_requests",
        "async_over_latency_slo_requests",
        "async_outstanding_requests",
        "async_issued_requests_per_sec",
        "async_completed_samples_per_sec",
        "async_completed_tokens_per_sec",
        "async_queue_depth_max",
        "async_worker_utilization",
        "async_e2e_latency_p50_ms",
        "async_e2e_latency_p95_ms",
        "async_e2e_latency_p99_ms",
        "async_queue_wait_p99_ms",
        "async_service_time_p99_ms",
    }
    assert set(result["details"]) == {
        "measurement_duration_sec",
        "measurement",
        "invalid_reasons",
        "warnings",
        "counter_invariants",
        "counts",
        "timing_ms",
        "queue",
        "workers",
        "batch_size",
        "failure_types",
        "failure_request_examples",
        "generation",
    }
    assert result["details"]["counts"] == {}

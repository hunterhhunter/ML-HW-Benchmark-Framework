from concurrent.futures import ThreadPoolExecutor

import pytest

from core.async_inference.metrics import AsyncMetricsCollector
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

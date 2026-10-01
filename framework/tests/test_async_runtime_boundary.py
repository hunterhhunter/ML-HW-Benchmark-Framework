"""Runtime-call boundary regressions for asynchronous benchmarking."""

import pytest

from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference import types
from core.async_inference.types import (
    RequestTrace,
    TerminalStatus,
)


def _completed_trace(*, finished_ns=3_000_000_000, completed_ns=9_000_000_000):
    return RequestTrace(
        request_id=1,
        sample_index=0,
        status=TerminalStatus.COMPLETED,
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=0,
        runtime_started_ns=1_000_000_000,
        runtime_finished_ns=finished_ns,
        completed_ns=completed_ns,
        worker_id=0,
        batch_size=2,
        timed_out=False,
        sample_count=2,
    )


def test_runtime_throughput_ends_at_last_call_or_producer_not_flush():
    metrics = AsyncMetricsCollector(
        started_ns=0,
        worker_count=1,
        latency_slo_ms=4_000,
        pass_kind=types.AsyncPassKind.RUNTIME_ONLY,
    )
    metrics.record_submitted()
    metrics.record_accepted(now_ns=0, queue_depth=1)
    metrics.record_runtime_completion(
        request_count=1,
        sample_count=2,
        generated_tokens=7,
        finished_ns=3_000_000_000,
    )
    metrics.record_generation(7, timing_ms=None)
    metrics.record_terminal(_completed_trace())

    result = metrics.finalize(
        end_ns=10_000_000_000,
        producer_finished_ns=5_000_000_000,
    )
    summary = result["summary"]
    assert summary["async_runtime_measurement_duration_sec"] == pytest.approx(5)
    assert summary["async_runtime_completed_samples_per_sec"] == pytest.approx(0.4)
    assert summary["async_runtime_completed_tokens_per_sec"] == pytest.approx(1.4)
    assert summary["async_runtime_ready_latency_p99_ms"] == pytest.approx(3_000)
    assert summary["async_over_latency_slo_requests"] == 0
    assert "async_completed_samples_per_sec" not in summary
    assert "async_completed_tokens_per_sec" not in summary
    assert result["details"]["runtime_measurement_duration_sec"] == pytest.approx(5)
    assert "runtime_terminal_mismatch" not in result["details"]["invalid_reasons"]


def test_runtime_success_without_terminal_invalidates_result():
    metrics = AsyncMetricsCollector(
        started_ns=0,
        worker_count=1,
        pass_kind=types.AsyncPassKind.RUNTIME_ONLY,
    )
    metrics.record_runtime_completion(
        request_count=1,
        sample_count=2,
        generated_tokens=0,
        finished_ns=3_000_000_000,
    )

    result = metrics.finalize(
        end_ns=10_000_000_000,
        producer_finished_ns=5_000_000_000,
    )
    assert "runtime_terminal_mismatch" in result["details"]["invalid_reasons"]

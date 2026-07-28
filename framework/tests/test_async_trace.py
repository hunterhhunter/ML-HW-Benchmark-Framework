import pytest

from core.async_inference.trace import _trace_row
from core.async_inference.types import RequestTrace, TerminalStatus
from core.runtime_executor import GenerationOutputEvent


def make_trace(**changes):
    values = {
        "request_id": 7,
        "sample_index": 9,
        "status": TerminalStatus.COMPLETED,
        "scheduled_ns": 90_000_000,
        "issued_ns": 100_000_000,
        "enqueued_ns": 110_000_000,
        "runtime_started_ns": 115_000_000,
        "runtime_finished_ns": 205_000_000,
        "completed_ns": 210_000_000,
        "worker_id": 0,
        "batch_size": 1,
        "timed_out": False,
        "sample_count": 1,
    }
    values.update(changes)
    return RequestTrace(**values)


def test_trace_row_persists_raw_generation_evidence_and_derived_latency():
    row = _trace_row(
        make_trace(
            generated_tokens=3,
            backend_submitted_ns=120_000_000,
            generation_events=(
                GenerationOutputEvent(150_000_000, 1),
                GenerationOutputEvent(170_000_000, 2),
                GenerationOutputEvent(200_000_000, 3),
            ),
            generation_timing_source="furiosa_async_python_stream",
        )
    )

    assert row["generated_tokens"] == 3
    assert row["backend_submitted_ns"] == 120_000_000
    assert row["generation_events"] == [
        {"observed_ns": 150_000_000, "cumulative_tokens": 1},
        {"observed_ns": 170_000_000, "cumulative_tokens": 2},
        {"observed_ns": 200_000_000, "cumulative_tokens": 3},
    ]
    assert row["generation_timing_source"] == "furiosa_async_python_stream"
    assert row["request_ttft_ms"] == pytest.approx(50.0)
    assert row["backend_ttft_ms"] == pytest.approx(30.0)
    assert row["request_mean_tpot_ms"] == pytest.approx(25.0)


def test_trace_row_persists_mobilint_one_token_per_event_evidence():
    row = _trace_row(
        make_trace(
            generated_tokens=3,
            backend_submitted_ns=120_000_000,
            generation_events=(
                GenerationOutputEvent(150_000_000, 1),
                GenerationOutputEvent(170_000_000, 2),
                GenerationOutputEvent(200_000_000, 3),
            ),
            generation_timing_source="mobilint_transformers_streamer",
        )
    )

    assert row["generation_events"] == [
        {"observed_ns": 150_000_000, "cumulative_tokens": 1},
        {"observed_ns": 170_000_000, "cumulative_tokens": 2},
        {"observed_ns": 200_000_000, "cumulative_tokens": 3},
    ]
    assert row["generation_timing_source"] == "mobilint_transformers_streamer"
    assert row["request_ttft_ms"] == pytest.approx(50.0)
    assert row["backend_ttft_ms"] == pytest.approx(30.0)
    assert row["request_mean_tpot_ms"] == pytest.approx(25.0)


def test_trace_row_preserves_mobilint_grouped_callback_as_one_raw_event():
    row = _trace_row(
        make_trace(
            generated_tokens=3,
            backend_submitted_ns=120_000_000,
            generation_events=(
                GenerationOutputEvent(150_000_000, 2),
                GenerationOutputEvent(200_000_000, 3),
            ),
            generation_timing_source="mobilint_transformers_streamer",
        )
    )

    assert row["generation_events"] == [
        {"observed_ns": 150_000_000, "cumulative_tokens": 2},
        {"observed_ns": 200_000_000, "cumulative_tokens": 3},
    ]
    assert row["generation_timing_source"] == "mobilint_transformers_streamer"
    assert row["request_ttft_ms"] == pytest.approx(50.0)
    assert row["backend_ttft_ms"] == pytest.approx(30.0)
    assert row["request_mean_tpot_ms"] == pytest.approx(25.0)


def test_trace_row_uses_empty_generation_fields_for_failed_request():
    row = _trace_row(
        make_trace(
            status=TerminalStatus.FAILED,
            error_type="VendorError",
            error_message="failed",
        )
    )

    assert row["generated_tokens"] == 0
    assert row["backend_submitted_ns"] is None
    assert row["generation_events"] == []
    assert row["generation_timing_source"] is None
    assert row["request_ttft_ms"] is None
    assert row["backend_ttft_ms"] is None
    assert row["request_mean_tpot_ms"] is None


def test_trace_row_preserves_generated_tokens_when_stream_is_unobserved():
    row = _trace_row(make_trace(generated_tokens=4))

    assert row["generated_tokens"] == 4
    assert row["generation_events"] == []
    assert row["request_ttft_ms"] is None
    assert row["backend_ttft_ms"] is None
    assert row["request_mean_tpot_ms"] is None


def test_trace_row_rejects_more_than_4096_generation_events():
    events = tuple(
        GenerationOutputEvent(index, index + 1)
        for index in range(4097)
    )

    with pytest.raises(ValueError, match="4096"):
        _trace_row(
            make_trace(
                generated_tokens=len(events),
                backend_submitted_ns=0,
                generation_events=events,
                generation_timing_source="test_stream",
            )
        )


@pytest.mark.parametrize(
    ("backend_submitted_ns", "source"),
    [(-1, "test_stream"), (0, "x" * 129)],
)
def test_trace_row_revalidates_generation_observation_bounds(
    backend_submitted_ns,
    source,
):
    with pytest.raises(ValueError, match="generation evidence"):
        _trace_row(
            make_trace(
                generated_tokens=1,
                backend_submitted_ns=backend_submitted_ns,
                generation_events=(GenerationOutputEvent(150_000_000, 1),),
                generation_timing_source=source,
            )
        )

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import threading
import time

import numpy as np
import pytest

from core.async_inference.completion import (
    CompletionCoordinator,
    FirstTokenTracker,
)
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import (
    BatchCompletion,
    FirstTokenEvent,
    InferenceRequest,
    TerminalStatus,
)


class FakePipeline:
    def prepare_eval_labels(self, collated):
        return collated["label"]


class RecordingEvaluator:
    def __init__(self):
        self.calls = []

    def add_batch(self, outputs, labels, timing_ms):
        self.calls.append(
            (threading.get_ident(), outputs["output"].tolist(), labels, timing_ms)
        )


def request(request_id):
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={"input": np.array([request_id]), "label": request_id},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
    )


def completion(req):
    return BatchCompletion(
        requests=[req],
        collated={"label": [req.sample_index]},
        outputs={"output": np.array([[req.sample_index]])},
        timing_ms=1.0,
        runtime_started_ns=2,
        runtime_finished_ns=3,
        worker_id=0,
        batch_size=1,
    )


def test_completion_runs_evaluator_on_coordinator_thread_and_drains():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True

    assert len(evaluator.calls) == 1
    assert evaluator.calls[0][0] != threading.get_ident()


def test_duplicate_completion_marks_run_invalid_without_double_evaluation():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=2,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    item = completion(req)
    coordinator.submit(item)
    assert coordinator.wait_for_all(timeout=1.0) is True
    coordinator.submit(item)
    assert coordinator.stop(timeout=1.0) is True

    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert len(evaluator.calls) == 1
    assert "duplicate_completion" in details["invalid_reasons"]


def test_mixed_duplicate_batch_fails_new_member_without_double_evaluation():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=2,
    )
    coordinator.start()
    first = request(0)
    second = request(1)
    coordinator.register(first)
    coordinator.register(second)
    coordinator.submit(completion(first))
    with coordinator.condition:
        assert coordinator.condition.wait_for(
            lambda: bool(coordinator.terminal[0]),
            timeout=1.0,
        )
    coordinator.submit(replace(completion(second), requests=[first, second]))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert len(evaluator.calls) == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert "duplicate_completion" in result["details"]["invalid_reasons"]


def test_duplicate_members_within_one_batch_terminalize_request_once():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(replace(completion(req), requests=[req, req]))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert evaluator.calls == []
    assert result["details"]["counts"]["terminal"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert "duplicate_completion" in result["details"]["invalid_reasons"]


def test_concurrent_duplicate_submissions_evaluate_exactly_once():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=4,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    item = completion(req)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(coordinator.submit, item) for _ in range(16)]
        for future in futures:
            future.result(timeout=1.0)

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert len(evaluator.calls) == 1
    assert result["details"]["counts"]["terminal"] == 1
    assert "duplicate_completion" in result["details"]["invalid_reasons"]


def test_mixed_unknown_batch_fails_known_member_without_evaluation():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    known = request(0)
    unknown = request(99)
    coordinator.register(known)
    coordinator.submit(
        replace(completion(known), requests=[known, unknown])
    )

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert evaluator.calls == []
    assert result["summary"]["async_failed_requests"] == 1
    assert result["details"]["failure_types"] == {
        "InvalidCompletionMembership": 1
    }
    assert "unknown_completion" in result["details"]["invalid_reasons"]


def test_unknown_completion_is_ignored_and_recorded():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    coordinator.submit(completion(request(99)))

    assert coordinator.stop(timeout=1.0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert evaluator.calls == []
    assert details["counts"].get("terminal", 0) == 0
    assert "unknown_completion" in details["invalid_reasons"]


def test_completed_request_id_cannot_be_registered_again():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    req = request(0)
    try:
        coordinator.register(req)
        coordinator.submit(completion(req))
        assert coordinator.wait_for_all(timeout=1.0) is True

        with pytest.raises(ValueError, match="duplicate request_id"):
            coordinator.register(req)
    finally:
        assert coordinator.stop(timeout=1.0) is True


def test_negative_request_id_is_rejected_before_registration():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )

    with pytest.raises(ValueError, match="non-negative"):
        coordinator.register(request(-1))


def test_concurrent_submitters_are_serialized_onto_one_evaluator_thread():
    evaluator = RecordingEvaluator()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=4,
    )
    requests = [request(request_id) for request_id in range(12)]
    for req in requests:
        coordinator.register(req)
    coordinator.start()

    def submit(req):
        coordinator.submit(completion(req))
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=6) as executor:
        producer_threads = set(executor.map(submit, requests))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    evaluator_threads = {call[0] for call in evaluator.calls}
    assert len(evaluator.calls) == len(requests)
    assert len(evaluator_threads) == 1
    assert evaluator_threads.isdisjoint(producer_threads)


def test_decoder_output_reaches_evaluator_on_coordinator_thread():
    calls = []

    class Decoder:
        def decode(self, outputs):
            calls.append(("decoder", threading.get_ident()))
            return {"output": outputs["output"] + 10}

    class Evaluator(RecordingEvaluator):
        def add_batch(self, outputs, labels, timing_ms):
            calls.append(("evaluator", threading.get_ident()))
            super().add_batch(outputs, labels, timing_ms)

    evaluator = Evaluator()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=Decoder(),
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    assert [name for name, _ in calls] == ["decoder", "evaluator"]
    assert len({thread_id for _, thread_id in calls}) == 1
    assert evaluator.calls[0][1] == [[10]]


def test_decoder_error_fails_batch_with_normalized_bounded_evidence():
    class Decoder:
        def decode(self, _outputs):
            raise ValueError("  decoder\n\tfailed  " + "x" * 600)

    evaluator = RecordingEvaluator()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=Decoder(),
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert evaluator.calls == []
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].error_type == "ValueError"
    assert traces[0].error_message.startswith("decoder failed x")
    assert "\n" not in traces[0].error_message
    assert len(traces[0].error_message) == 512
    assert "request_failed" in details["invalid_reasons"]
    assert details["failure_types"] == {"ValueError": 1}


def test_evaluator_error_fails_only_that_batch_and_coordinator_continues():
    class FailOnceEvaluator(RecordingEvaluator):
        def add_batch(self, outputs, labels, timing_ms):
            super().add_batch(outputs, labels, timing_ms)
            if len(self.calls) == 1:
                raise RuntimeError(" evaluator  failed ")

    evaluator = FailOnceEvaluator()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=2,
        trace_callback=traces.append,
    )
    coordinator.start()
    first = request(0)
    second = request(1)
    coordinator.register(first)
    coordinator.register(second)
    coordinator.submit(completion(first))
    coordinator.submit(completion(second))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert len(evaluator.calls) == 2
    assert [trace.status for trace in traces] == [
        TerminalStatus.FAILED,
        TerminalStatus.COMPLETED,
    ]
    assert traces[0].error_message == "evaluator failed"
    assert details["failure_types"] == {"RuntimeError": 1}


def test_runtime_error_skips_decoder_and_evaluator_and_normalizes_message():
    class Decoder:
        def decode(self, _outputs):
            raise AssertionError("decoder must not run")

    evaluator = RecordingEvaluator()
    traces = []
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=Decoder(),
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
        trace_callback=traces.append,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(
        replace(
            completion(req),
            outputs=None,
            error_type="RuntimeError",
            error_message="  runtime\n failed  ",
        )
    )

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    assert evaluator.calls == []
    assert traces[0].status is TerminalStatus.FAILED
    assert traces[0].error_type == "RuntimeError"
    assert traces[0].error_message == "runtime failed"


def test_request_timeout_is_diagnostic_and_preserves_completed_status():
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        request_timeout_ms=1.0,
        trace_callback=traces.append,
        clock_ns=lambda: 2_000_000,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=3_000_000)
    assert traces[0].status is TerminalStatus.COMPLETED
    assert traces[0].timed_out is True
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_timed_out_requests"] == 1
    assert "request_timeout" in result["details"]["invalid_reasons"]


def test_wait_timeout_records_flush_timeout_without_terminalizing_request():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.start()
    coordinator.register(request(0))

    assert coordinator.wait_for_all(timeout=0.0) is False
    assert coordinator.stop(timeout=1.0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "flush_timeout" in details["invalid_reasons"]
    assert details["counts"].get("terminal", 0) == 0


def test_trace_callback_failure_warns_and_does_not_block_terminalization():
    def fail_trace(_trace):
        raise OSError("trace sink unavailable")

    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=fail_trace,
    )
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert result["summary"]["async_completed_requests"] == 1
    assert "request_trace_write_failed" in result["details"]["warnings"]


def test_trace_callback_receives_metadata_without_request_or_output_payloads():
    traces = []
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
        trace_callback=traces.append,
    )
    coordinator.start()
    req = replace(
        request(0),
        sample={"input": "private-input", "label": "private-label"},
    )
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.stop(timeout=1.0) is True
    serialized = json.dumps(asdict(traces[0]))
    assert "private-input" not in serialized
    assert "private-label" not in serialized
    assert not hasattr(traces[0], "sample")
    assert not hasattr(traces[0], "outputs")


def test_first_token_contract_rejects_duplicate_and_final_before_event():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    first = request(0)
    tracker.register(first)
    event = FirstTokenEvent(request_id=0, first_token_ns=2)
    assert tracker.record(event) is True
    assert tracker.record(event) is False
    assert tracker.finalize(request_id=0, generated_tokens=1) is True

    second = request(1)
    tracker.register(second)
    assert tracker.finalize(request_id=1, generated_tokens=1) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "timing_invariant_failed" in details["invalid_reasons"]
    assert details["generation"]["event_ttft_ms"]["count"] == 1


def test_first_token_event_before_issue_is_rejected():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    req = replace(request(0), issued_ns=10)
    tracker.register(req)

    assert tracker.record(FirstTokenEvent(request_id=0, first_token_ns=9)) is False
    assert tracker.finalize(request_id=0, generated_tokens=1) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert details["generation"]["event_ttft_ms"]["count"] == 0
    assert "timing_invariant_failed" in details["invalid_reasons"]


def test_first_token_event_requires_positive_token_count():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    tracker.register(request(0))

    assert tracker.record(
        FirstTokenEvent(request_id=0, first_token_ns=2, token_count=0)
    ) is False
    assert tracker.finalize(request_id=0, generated_tokens=0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert details["generation"]["event_ttft_ms"]["count"] == 0
    assert "timing_invariant_failed" in details["invalid_reasons"]


@pytest.mark.parametrize(
    ("event_tokens", "generated_tokens"),
    [(1, 0), (2, 1)],
)
def test_final_token_count_cannot_precede_recorded_first_token_count(
    event_tokens,
    generated_tokens,
):
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    tracker.register(request(0))
    assert tracker.record(
        FirstTokenEvent(request_id=0, first_token_ns=2, token_count=event_tokens)
    ) is True

    assert tracker.finalize(0, generated_tokens=generated_tokens) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "timing_invariant_failed" in details["invalid_reasons"]


def test_completion_thread_failure_unblocks_waiter_immediately():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    def crash(_completion):
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = crash
    coordinator.start()
    req = request(0)
    coordinator.register(req)
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "completion_thread_failed" in details["invalid_reasons"]


def test_completion_thread_failure_prevents_successful_empty_flush():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    def crash(_completion):
        raise RuntimeError("planned empty coordinator crash")

    coordinator._handle = crash
    coordinator.start()
    coordinator.submit(completion(request(99)))
    with coordinator.condition:
        assert coordinator.condition.wait_for(
            lambda: coordinator.thread_error is not None,
            timeout=1.0,
        )

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False

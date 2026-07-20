from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, replace
import json
import queue
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


class FailingEvaluator:
    def __init__(self, primary):
        self.primary = primary

    def add_batch(self, outputs, labels, timing_ms):
        del outputs, labels, timing_ms
        raise self.primary


class GatedInvalidReasonMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.entered = threading.Event()
        self.release = threading.Event()

    def add_invalid_reason(self, reason):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        super().add_invalid_reason(reason)


class WaitTrackingCondition(threading.Condition):
    def __init__(self, expected_waiters=1, after_wake=None):
        super().__init__()
        self.expected_waiters = expected_waiters
        self.after_wake = after_wake
        self.wait_path_entered = threading.Event()
        self.tracking = False
        self.waiting = 0

    def wait(self, timeout=None):
        tracked = self.tracking
        pause_after_wake = False
        if tracked:
            self.waiting += 1
            if self.waiting == self.expected_waiters:
                self.wait_path_entered.set()
            pause_after_wake = (
                self.after_wake is not None
                and self.waiting <= self.expected_waiters
            )
        try:
            result = super().wait(timeout)
            if pause_after_wake:
                self.release()
                try:
                    assert self.after_wake.wait(timeout=1.0)
                finally:
                    self.acquire()
            return result
        finally:
            if tracked:
                self.waiting -= 1


class SentinelTrackingQueue(queue.Queue):
    def __init__(self, maxsize):
        super().__init__(maxsize=maxsize)
        self.sentinel_put_started = threading.Event()
        self.sentinel_enqueued = threading.Event()

    def put(self, item, block=True, timeout=None):
        is_sentinel = not isinstance(item, BatchCompletion)
        if is_sentinel:
            self.sentinel_put_started.set()
        result = super().put(item, block=block, timeout=timeout)
        if is_sentinel:
            self.sentinel_enqueued.set()
        return result


class CoordinatorReentrantMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.coordinator = None
        self.reentered = threading.Event()

    def add_invalid_reason(self, reason):
        assert not self.coordinator.condition._is_owned(), (
            "public metrics called under coordinator condition"
        )
        with self.coordinator.condition:
            self.coordinator.snapshot_outstanding()
        self.reentered.set()
        super().add_invalid_reason(reason)


def request(request_id, *, sample_count=1):
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={"input": np.array([request_id]), "label": request_id},
        scheduled_ns=0,
        issued_ns=0,
        enqueued_ns=1,
        sample_count=sample_count,
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


def test_inline_completion_uses_membership_without_queue_or_thread():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(0, 1),
        queue_capacity=None,
        raise_callback_errors=True,
    )
    req = request(0)
    coordinator.start()
    coordinator.register(req)
    coordinator.submit(completion(req))
    assert coordinator.queue is None
    assert coordinator.thread is None
    assert coordinator.snapshot_outstanding() == ()
    assert coordinator.stop(timeout=0.0) is True


def test_inline_completion_commits_failure_then_reraises_same_error():
    primary = ValueError("quality failure")
    evaluator = FailingEvaluator(primary)
    metrics = AsyncMetricsCollector(0, 1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=None,
        raise_callback_errors=True,
    )
    req = request(0)
    coordinator.register(req)
    with pytest.raises(ValueError) as raised:
        coordinator.submit(completion(req))
    assert raised.value is primary
    assert coordinator.snapshot_outstanding() == ()
    assert metrics.finalize(10)["summary"]["async_failed_requests"] == 1


def test_inline_completion_rejects_operation_key_without_terminalizing():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(0, 1),
        queue_capacity=None,
        raise_callback_errors=True,
    )
    req = request(0)
    coordinator.register(req)

    with pytest.raises(ValueError, match="operation_key"):
        coordinator.submit(completion(req), operation_key=object())

    assert coordinator.snapshot_outstanding() == (0,)
    coordinator.submit(completion(req))
    assert coordinator.stop(timeout=0.0) is True


def test_inline_completion_dirty_stop_preserves_membership_for_retry():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(0, 1),
        queue_capacity=None,
        raise_callback_errors=True,
    )
    req = request(0)
    coordinator.register(req)

    assert coordinator.stop(timeout=0.0) is False
    assert coordinator.snapshot_outstanding() == (0,)

    coordinator.submit(completion(req))
    assert coordinator.snapshot_outstanding() == ()
    assert coordinator.stop(timeout=0.0) is True


def test_timeout_metrics_can_reenter_coordinator_condition():
    metrics = CoordinatorReentrantMetrics(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    metrics.coordinator = coordinator
    coordinator.register(request(0))

    assert coordinator.wait_for_requests((0,), timeout=0.0) is False
    assert metrics.reentered.is_set()


def test_never_started_stop_releases_condition_before_metrics_and_preserves_reservation():
    metrics = CoordinatorReentrantMetrics(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    metrics.coordinator = coordinator
    req = replace(request(7), submission_token=77)
    coordinator.reserve_registration(req, attempt_token=77)

    with ThreadPoolExecutor(max_workers=1) as executor:
        stopped = executor.submit(coordinator.stop, 0.1)
        assert stopped.result(timeout=1.0) is False

    assert metrics.reentered.is_set()
    with coordinator.condition:
        assert tuple(coordinator.reservations) == (7,)
    assert coordinator.abort_registration(7, expected_token=76) is False
    assert coordinator.abort_registration(7, expected_token=77) is True
    assert coordinator.stop(timeout=0.1) is True


def test_registration_stages_are_idempotent_and_record_exact_terminal_token():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    req = replace(request(49), submission_token=900)

    coordinator.reserve_registration(req, attempt_token=900)
    coordinator.reserve_registration(req, attempt_token=900)
    coordinator.commit_registration(req, expected_token=900)
    coordinator.commit_registration(req, expected_token=900)

    assert coordinator.reservations == {}
    assert coordinator.terminal[49] == 0
    assert coordinator.terminal_tokens[49] == 900
    assert coordinator.outstanding[49].submission_token == 900
    with pytest.raises(TypeError):
        coordinator.terminal[49] = 1
    with pytest.raises(TypeError):
        coordinator.terminal_tokens[49] = 901


def test_registration_token_normalization_finishes_before_coordinator_lock():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    class GuardedInt(int):
        def __int__(self):
            assert not coordinator.condition._is_owned()
            return super().__int__()

    req = replace(
        request(50),
        request_id=GuardedInt(50),
        submission_token=GuardedInt(901),
    )
    coordinator.reserve_registration(
        req,
        attempt_token=GuardedInt(901),
    )
    coordinator.commit_registration(
        req,
        expected_token=GuardedInt(901),
    )

    assert type(coordinator.outstanding[50].request_id) is int
    assert type(coordinator.outstanding[50].submission_token) is int
    assert type(coordinator.terminal_tokens[50]) is int
    assert coordinator.terminal_tokens[50] == 901


def test_reservation_attempt_token_prevents_stale_abort_after_aba():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    first = request(40)
    replacement = replace(first, sample_index=41)

    coordinator.reserve_registration(first, attempt_token=100)
    assert coordinator.abort_registration(40, expected_token=100) is True
    coordinator.reserve_registration(replacement, attempt_token=101)

    assert coordinator.abort_registration(40, expected_token=100) is False
    coordinator.commit_registration(replacement, expected_token=101)
    assert coordinator.outstanding[40] is replacement


def test_reservation_identity_normalizes_before_coordinator_lock():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    class GuardedInt(int):
        conversions = 0

        def __int__(self):
            type(self).conversions += 1
            assert not coordinator.condition._is_owned()
            return super().__int__()

    req = replace(request(47), request_id=GuardedInt(47))
    coordinator.reserve_registration(req, attempt_token=GuardedInt(300))

    with coordinator.condition:
        reservation = coordinator.reservations[47]
        assert type(reservation.attempt_token) is int
        assert type(reservation.request.request_id) is int
    assert GuardedInt.conversions >= 2


def test_unregister_rejected_requires_matching_token_and_normalizes_prelock():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )

    class GuardedInt(int):
        conversions = 0

        def __int__(self):
            type(self).conversions += 1
            assert not coordinator.condition._is_owned()
            return super().__int__()

    replacement = replace(request(48), submission_token=501)
    coordinator.reserve_registration(replacement, attempt_token=501)

    assert coordinator.unregister_rejected(
        GuardedInt(48),
        GuardedInt(500),
    ) is False
    assert coordinator.reservations[48].attempt_token == 501
    coordinator.commit_registration(replacement, expected_token=501)
    assert coordinator.unregister_rejected(48, 500) is False
    assert coordinator.outstanding[48] is replacement
    assert coordinator.unregister_rejected(48, 501) is True
    assert coordinator.outstanding == {}
    assert GuardedInt.conversions == 2


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


def test_stale_same_id_completion_does_not_touch_replacement_attempt():
    evaluator = RecordingEvaluator()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    replacement = replace(
        request(51),
        sample_index=52,
        submission_token=1101,
    )

    class GuardedInt(int):
        def __int__(self):
            assert not coordinator.condition._is_owned()
            return super().__int__()

    stale = replace(
        replacement,
        sample_index=51,
        submission_token=GuardedInt(1100),
    )
    coordinator.register(replacement)

    coordinator._handle(completion(stale))

    assert evaluator.calls == []
    assert coordinator.outstanding[51] is replacement
    assert coordinator.terminal[51] == 0
    first = metrics.finalize(end_ns=time.monotonic_ns())
    assert "stale_completion" in first["details"]["invalid_reasons"]
    assert first["details"]["counts"].get("terminal", 0) == 0

    coordinator._handle(completion(replacement))

    assert len(evaluator.calls) == 1
    assert coordinator.outstanding == {}
    assert coordinator.terminal[51] == 2
    assert coordinator.terminal_tokens[51] == 1101
    final = metrics.finalize(end_ns=time.monotonic_ns())
    assert final["details"]["counts"]["terminal"] == 1


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


def test_wait_timeout_records_flush_timeout_and_stop_fails_outstanding_request():
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
    )
    coordinator.start()
    coordinator.register(request(0, sample_count=3))

    assert coordinator.wait_for_all(timeout=0.0) is False
    assert coordinator.stop(timeout=1.0) is True
    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "flush_timeout" in details["invalid_reasons"]
    assert details["counts"]["terminal"] == 1
    assert details["counts"]["failed"] == 1
    assert traces[0].batch_size == 3
    assert traces[0].sample_count == 3
    with coordinator.condition:
        assert coordinator.outstanding == {}


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


def test_duplicate_first_token_diagnostic_does_not_hold_tracker_lock():
    metrics = GatedInvalidReasonMetrics(started_ns=0, worker_count=1)
    tracker = FirstTokenTracker(metrics)
    event = FirstTokenEvent(request_id=0, first_token_ns=2)
    tracker.register(request(0))
    assert tracker.record(event) is True

    executor = ThreadPoolExecutor(max_workers=2)
    duplicate = executor.submit(tracker.record, event)
    finalized = None
    try:
        assert metrics.entered.wait(timeout=1.0)
        finalized = executor.submit(tracker.finalize, 0, 1)
        assert finalized.result(timeout=1.0) is True
    finally:
        metrics.release.set()
        assert duplicate.result(timeout=1.0) is False
        if finalized is not None:
            finalized.result(timeout=1.0)
        executor.shutdown(wait=True)


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


def test_completion_thread_failure_terminalizes_all_outstanding_requests_once():
    evaluator = RecordingEvaluator()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
        clock_ns=lambda: 10,
    )

    def crash(_completion):
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = crash
    requests = [
        request(request_id, sample_count=request_id + 2)
        for request_id in range(3)
    ]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False
    with coordinator.condition:
        assert coordinator.outstanding == {}
        assert all(coordinator.terminal[req.request_id] for req in requests)

    result = metrics.finalize(end_ns=11)
    assert evaluator.calls == []
    assert result["summary"]["async_failed_requests"] == len(requests)
    assert result["details"]["counts"]["terminal"] == len(requests)
    assert [trace.request_id for trace in traces] == [0, 1, 2]
    assert all(trace.status is TerminalStatus.FAILED for trace in traces)
    assert all(trace.error_type == "CompletionThreadError" for trace in traces)
    assert [trace.batch_size for trace in traces] == [2, 3, 4]


def test_failed_stop_reports_residual_registration_reservation():
    class StopLockCheckingMetrics(AsyncMetricsCollector):
        coordinator = None
        counter_recorded_outside_condition = False

        def add_invalid_reason(self, reason):
            if reason == "counter_invariant_failed":
                acquired = self.coordinator.condition.acquire(blocking=False)
                assert acquired
                self.coordinator.condition.release()
                self.counter_recorded_outside_condition = True
            super().add_invalid_reason(reason)

    metrics = StopLockCheckingMetrics(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    coordinator.condition = threading.Condition(threading.Lock())
    metrics.coordinator = coordinator

    def crash(_completion):
        raise RuntimeError("planned coordinator crash with reservation")

    coordinator._handle = crash
    reserved = replace(request(52), submission_token=1200)
    coordinator.reserve_registration(reserved, attempt_token=1200)
    coordinator.start()
    coordinator.submit(completion(reserved))

    with coordinator.condition:
        assert coordinator.condition.wait_for(
            lambda: coordinator.thread_error is not None,
            timeout=1.0,
        )
    assert coordinator.stop(timeout=1.0) is False
    with coordinator.condition:
        assert coordinator.reservations[52].attempt_token == 1200

    details = metrics.finalize(end_ns=time.monotonic_ns())["details"]
    assert "completion_thread_failed" in details["invalid_reasons"]
    assert "counter_invariant_failed" in details["invalid_reasons"]
    assert metrics.counter_recorded_outside_condition is True


def test_registration_after_crash_cleanup_is_rejected_without_leaking_request():
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )

    def crash(_completion):
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = crash
    first = request(0)
    coordinator.register(first)
    coordinator.start()
    coordinator.submit(completion(first))
    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False

    with pytest.raises(RuntimeError, match="failed"):
        coordinator.register(request(1))
    with coordinator.condition:
        assert coordinator.outstanding == {}


def test_partial_terminal_metric_failure_never_double_counts_request():
    class FailAfterFirstTerminal(AsyncMetricsCollector):
        def __init__(self):
            super().__init__(started_ns=0, worker_count=1)
            self.fail_once = True

        def record_terminal(self, trace):
            super().record_terminal(trace)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("failure after terminal metric mutation")

    evaluator = RecordingEvaluator()
    metrics = FailAfterFirstTerminal()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        clock_ns=lambda: 10,
    )
    requests = [request(0), request(1)]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(
        replace(
            completion(requests[0]),
            requests=requests,
            collated={"label": [0, 1]},
            outputs={"output": np.array([[0], [1]])},
            batch_size=2,
        )
    )

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False
    result = metrics.finalize(end_ns=11)
    with coordinator.condition:
        assert coordinator.outstanding == {}
    assert len(evaluator.calls) == 1
    assert result["details"]["counts"]["terminal"] == len(requests)
    assert (
        result["summary"]["async_completed_requests"]
        + result["summary"]["async_failed_requests"]
        == len(requests)
    )
    assert "completion_thread_failed" in result["details"]["invalid_reasons"]


def test_blocked_submitters_fail_and_queue_releases_payloads_after_crash():
    handler_entered = threading.Event()
    release_crash = threading.Event()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )
    tracked_condition = WaitTrackingCondition(expected_waiters=2)
    coordinator.condition = tracked_condition

    def controlled_crash(_completion):
        handler_entered.set()
        assert release_crash.wait(timeout=1.0)
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = controlled_crash
    requests = [request(request_id) for request_id in range(4)]
    for req in requests:
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))
    assert handler_entered.wait(timeout=1.0)
    coordinator.submit(completion(requests[1]))

    submit_started = [threading.Event(), threading.Event()]

    def submit_and_capture(item, started):
        started.set()
        try:
            coordinator.submit(item)
        except RuntimeError as exc:
            return str(exc)
        return None

    executor = ThreadPoolExecutor(max_workers=2)
    tracked_condition.tracking = True
    futures = [
        executor.submit(
            submit_and_capture,
            completion(req),
            started,
        )
        for req, started in zip(requests[2:], submit_started)
    ]
    for started in submit_started:
        assert started.wait(timeout=1.0)
    assert tracked_condition.wait_path_entered.wait(timeout=1.0)

    release_crash.set()
    assert coordinator.wait_for_all(timeout=1.0) is False
    done, not_done = wait(futures, timeout=1.0)
    try:
        assert not not_done, "completion submitters remained blocked after crash"
        errors = [future.result() for future in done]
        assert all(error and "coordinator failed" in error for error in errors)
        assert coordinator.stop(timeout=1.0) is False
        assert coordinator.queue.empty()
        assert coordinator.queue.unfinished_tasks == 0
        with coordinator.condition:
            assert coordinator.outstanding == {}
    finally:
        if not_done:
            for _ in range(1 + len(not_done)):
                queued = coordinator.queue.get(timeout=1.0)
                del queued
                coordinator.queue.task_done()
            for future in futures:
                future.result(timeout=1.0)
        executor.shutdown(wait=True)


def test_stop_atomically_closes_blocked_submission_before_sentinel():
    handler_entered = threading.Event()
    release_handler = threading.Event()
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    tracked_queue = SentinelTrackingQueue(maxsize=1)
    tracked_condition = WaitTrackingCondition(
        after_wake=tracked_queue.sentinel_enqueued
    )
    coordinator.queue = tracked_queue
    coordinator.condition = tracked_condition
    original_handle = coordinator._handle

    def gated_handle(item):
        if item.requests[0].request_id == 0:
            handler_entered.set()
            assert release_handler.wait(timeout=1.0)
        original_handle(item)

    coordinator._handle = gated_handle
    requests = [request(request_id) for request_id in range(3)]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))
    assert handler_entered.wait(timeout=1.0)
    coordinator.submit(completion(requests[1]))

    def submit_last():
        try:
            coordinator.submit(completion(requests[2]))
        except RuntimeError:
            return "rejected"
        return "accepted"

    tracked_condition.tracking = True
    with ThreadPoolExecutor(max_workers=2) as executor:
        submit_future = executor.submit(submit_last)
        assert tracked_condition.wait_path_entered.wait(timeout=1.0)
        stop_future = executor.submit(coordinator.stop, 2.0)
        assert tracked_queue.sentinel_put_started.wait(timeout=1.0)
        release_handler.set()
        submit_result = submit_future.result(timeout=2.0)
        stop_result = stop_future.result(timeout=2.0)

    with coordinator.condition:
        outstanding_ids = tuple(coordinator.outstanding)
    assert (
        submit_result,
        stop_result,
        coordinator.queue.qsize(),
        coordinator.queue.unfinished_tasks,
        outstanding_ids,
    ) == ("rejected", True, 0, 0, ())


def test_stop_timeout_after_sentinel_enqueue_eventually_finalizes_requests():
    handler_entered = threading.Event()
    release_handler = threading.Event()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
    )
    tracked_queue = SentinelTrackingQueue(maxsize=1)
    coordinator.queue = tracked_queue
    original_handle = coordinator._handle

    def gated_handle(item):
        handler_entered.set()
        assert release_handler.wait(timeout=1.0)
        original_handle(item)

    coordinator._handle = gated_handle
    requests = [request(0), request(1)]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))
    assert handler_entered.wait(timeout=1.0)

    assert coordinator.stop(timeout=0.0) is False
    assert tracked_queue.sentinel_enqueued.is_set()
    release_handler.set()
    coordinator.thread.join(timeout=1.0)

    assert not coordinator.thread.is_alive()
    with coordinator.condition:
        assert coordinator.state == "stopped"
        assert coordinator.outstanding == {}
    assert coordinator.queue.empty()
    assert coordinator.queue.unfinished_tasks == 0
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert result["details"]["counts"]["terminal"] == len(requests)
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert [trace.status for trace in traces] == [
        TerminalStatus.COMPLETED,
        TerminalStatus.FAILED,
    ]


def test_crash_while_stop_waits_for_queue_space_does_not_strand_sentinel():
    handler_entered = threading.Event()
    release_crash = threading.Event()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
    )
    tracked_queue = SentinelTrackingQueue(maxsize=1)
    coordinator.queue = tracked_queue

    def controlled_crash(_completion):
        handler_entered.set()
        assert release_crash.wait(timeout=1.0)
        raise RuntimeError("planned coordinator crash")

    coordinator._handle = controlled_crash
    requests = [request(0), request(1)]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))
    assert handler_entered.wait(timeout=1.0)
    coordinator.submit(completion(requests[1]))

    with ThreadPoolExecutor(max_workers=1) as executor:
        stop_future = executor.submit(coordinator.stop, 1.0)
        assert tracked_queue.sentinel_put_started.wait(timeout=1.0)
        release_crash.set()
        assert stop_future.result(timeout=1.0) is False

    coordinator.thread.join(timeout=1.0)
    assert not coordinator.thread.is_alive()
    with coordinator.condition:
        assert coordinator.state == "failed"
        assert coordinator.outstanding == {}
    assert not tracked_queue.sentinel_enqueued.is_set()
    assert coordinator.queue.empty()
    assert coordinator.queue.unfinished_tasks == 0
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert result["details"]["counts"]["terminal"] == len(requests)
    assert result["summary"]["async_failed_requests"] == len(requests)
    assert len(traces) == len(requests)
    assert all(trace.status is TerminalStatus.FAILED for trace in traces)


def test_full_queue_stop_deadline_eventually_reaches_failed_terminal_state():
    handler_entered = threading.Event()
    release_handler = threading.Event()
    traces = []
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
        trace_callback=traces.append,
    )
    original_handle = coordinator._handle

    def gated_handle(item):
        if item.requests[0].request_id == 0:
            handler_entered.set()
            assert release_handler.wait(timeout=1.0)
        original_handle(item)

    coordinator._handle = gated_handle
    requests = [request(0), request(1)]
    for req in requests:
        metrics.record_submitted()
        metrics.record_accepted(now_ns=req.enqueued_ns, queue_depth=0)
        coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(requests[0]))
    assert handler_entered.wait(timeout=1.0)
    coordinator.submit(completion(requests[1]))

    assert coordinator.stop(timeout=0.0) is False
    release_handler.set()
    coordinator.thread.join(timeout=1.0)

    assert not coordinator.thread.is_alive()
    with coordinator.condition:
        assert coordinator.state == "failed"
        assert coordinator.outstanding == {}
    assert coordinator.queue.empty()
    assert coordinator.queue.unfinished_tasks == 0
    result = metrics.finalize(end_ns=time.monotonic_ns())
    assert result["details"]["counts"]["terminal"] == len(requests)
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert [trace.status for trace in traces] == [
        TerminalStatus.COMPLETED,
        TerminalStatus.FAILED,
    ]


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


def test_failure_finalization_notifies_handoff_terminal_outside_condition():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    callback_called = threading.Event()
    observed = []
    operation_key = object()
    req = request(98)

    def observe_terminal_handoffs():
        observed.append(
            (
                coordinator.condition._is_owned(),
                coordinator.state,
                coordinator.completion_handoff_state(operation_key),
            )
        )
        callback_called.set()

    def crash(_completion):
        raise RuntimeError("planned finalization notification crash")

    coordinator.handoff_ack_callback = observe_terminal_handoffs
    coordinator._handle = crash
    coordinator.register(req)
    coordinator.start()
    coordinator.submit(
        completion(req),
        timeout=1.0,
        operation_key=operation_key,
    )

    assert callback_called.wait(timeout=1.0)
    coordinator.thread.join(timeout=1.0)
    assert observed == [(False, "failed", "ACKED")]
    assert not coordinator.thread.is_alive()


def test_handoff_terminal_callback_failure_does_not_kill_coordinator():
    metrics = AsyncMetricsCollector(started_ns=0, worker_count=1)
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=metrics,
        queue_capacity=1,
    )
    req = request(97)
    operation_key = object()
    callback_called = threading.Event()

    def fail_callback():
        assert not coordinator.condition._is_owned()
        callback_called.set()
        raise RuntimeError("planned handoff callback failure")

    coordinator.handoff_ack_callback = fail_callback
    coordinator.register(req)
    coordinator.start()
    coordinator.submit(
        completion(req),
        timeout=1.0,
        operation_key=operation_key,
    )

    assert callback_called.wait(timeout=1.0)
    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.thread_error is None
    assert coordinator.acknowledge_completion_handoff(operation_key) is True
    assert coordinator.stop(timeout=1.0) is True


def test_prefixed_completion_thread_error_is_normalized_to_512_characters():
    traces = []
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=RecordingEvaluator(),
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
        trace_callback=traces.append,
        clock_ns=lambda: 10,
    )

    def crash(_completion):
        raise RuntimeError("  planned\n crash  " + "x" * 700)

    coordinator._handle = crash
    req = request(0)
    coordinator.register(req)
    coordinator.start()
    coordinator.submit(completion(req))

    assert coordinator.wait_for_all(timeout=1.0) is False
    assert coordinator.stop(timeout=1.0) is False
    assert len(coordinator.thread_error) <= 512
    assert "\n" not in coordinator.thread_error
    assert len(traces) == 1
    assert len(traces[0].error_message) <= 512
    assert traces[0].error_message == coordinator.thread_error


def test_handoff_dequeue_can_win_before_producer_enqueued_cas():
    class GateAfterPutQueue(queue.Queue):
        def __init__(self):
            super().__init__(maxsize=1)
            self.put_mutated = threading.Event()
            self.release_put = threading.Event()
            self.gated = False

        def put(self, item, block=True, timeout=None):
            result = super().put(item, block=block, timeout=timeout)
            if not self.gated:
                self.gated = True
                self.put_mutated.set()
                assert self.release_put.wait(timeout=2.0)
            return result

    evaluator = RecordingEvaluator()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )
    coordinator.queue = GateAfterPutQueue()
    handled = threading.Event()
    original_handle = coordinator._handle

    def observed_handle(item):
        handled.set()
        return original_handle(item)

    coordinator._handle = observed_handle
    req = request(204)
    coordinator.register(req)
    coordinator.start()
    operation_key = object()

    with ThreadPoolExecutor(max_workers=1) as executor:
        submitted = executor.submit(
            coordinator.submit,
            completion(req),
            1.0,
            operation_key=operation_key,
        )
        assert coordinator.queue.put_mutated.wait(timeout=1.0)
        try:
            assert handled.wait(timeout=1.0)
            assert coordinator.completion_handoff_state(operation_key) == "ACKED"
        finally:
            coordinator.queue.release_put.set()
        assert submitted.result(timeout=1.0) is None

    assert len(evaluator.calls) == 1
    assert coordinator.acknowledge_completion_handoff(operation_key) is True
    assert coordinator.completion_handoff_count == 0
    assert coordinator.stop(timeout=1.0) is True


@pytest.mark.parametrize("fault_stage", ["put", "dequeue"])
def test_handoff_inner_queue_mutation_fault_is_recovered_once(fault_stage):
    class FaultAfterMutationQueue(queue.Queue):
        def __init__(self):
            super().__init__(maxsize=1)
            self.fired = threading.Event()

        def put(self, item, block=True, timeout=None):
            result = super().put(item, block=block, timeout=timeout)
            if fault_stage == "put" and not self.fired.is_set():
                self.fired.set()
                raise RuntimeError("after completion queue put")
            return result

        def _get(self):
            item = super()._get()
            if fault_stage == "dequeue" and not self.fired.is_set():
                self.fired.set()
                raise RuntimeError("after completion queue dequeue")
            return item

    evaluator = RecordingEvaluator()
    coordinator = CompletionCoordinator(
        pipeline=FakePipeline(),
        evaluator=evaluator,
        decoder=None,
        metrics=AsyncMetricsCollector(started_ns=0, worker_count=1),
        queue_capacity=1,
    )
    coordinator.queue = FaultAfterMutationQueue()
    req = request(205)
    coordinator.register(req)
    coordinator.start()
    operation_key = object()

    coordinator.submit(
        completion(req),
        timeout=1.0,
        operation_key=operation_key,
    )

    assert coordinator.wait_for_all(timeout=1.0) is True
    assert coordinator.queue.fired.is_set()
    assert coordinator.completion_handoff_state(operation_key) == "ACKED"
    assert len(evaluator.calls) == 1
    assert coordinator.acknowledge_completion_handoff(operation_key) is True
    assert coordinator.stop(timeout=1.0) is True
    assert coordinator.queue.unfinished_tasks == 0

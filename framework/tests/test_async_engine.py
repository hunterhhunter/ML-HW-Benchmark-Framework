from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import queue
import threading
import time
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

import core.async_inference.engine as engine_module
import core.async_inference.metrics as metrics_module
from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine, _RequestQueue
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import (
    AsyncInferenceConfig,
    BatchCompletion,
    EngineState,
    InferenceRequest,
)
from core.inference_pipeline import InferencePipeline


class Loader:
    def __init__(self, *, static_batched=False):
        self.static_batched = static_batched

    def get_metadata(self):
        return {
            "is_static_batched": self.static_batched,
            "total_samples": 8,
        }


class Runtime:
    compiled_model = None

    def __init__(
        self,
        *,
        fail=False,
        max_batch_size=None,
        dynamic_batching=True,
        max_workers=1,
    ):
        self.fail = fail
        self._max_batch_size = max_batch_size
        self.dynamic_batching = dynamic_batching
        self.max_workers = max_workers
        self.batch_sizes = []

    def supports_generate(self):
        return False

    def max_concurrent_workers(self):
        return self.max_workers

    def supports_dynamic_batching(self):
        return self.dynamic_batching

    def max_dynamic_batch_size(self):
        return self._max_batch_size

    def supports_batch_generation(self):
        return False

    def run(self, inputs):
        if self.fail:
            raise RuntimeError("planned failure")
        values = next(iter(inputs.values()))
        self.batch_sizes.append(len(values))
        return {"output": np.asarray(values)}


class WeakrefRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.input_refs = []
        self.output_refs = []

    def run(self, inputs):
        values = next(iter(inputs.values()))
        outputs = np.array(values, copy=True)
        self.input_refs.append(weakref.ref(values))
        self.output_refs.append(weakref.ref(outputs))
        return {"output": outputs}


class LLMRuntime(Runtime):
    def supports_generate(self):
        return True

    def supports_batch_generation(self):
        return True

    def generate(
        self,
        inputs,
        max_new_tokens=256,
        stop_token_ids=None,
    ):
        values = next(iter(inputs.values()))
        self.batch_sizes.append(len(values))
        return SimpleNamespace(
            generated_ids=np.asarray(values),
            generated_lengths=None,
            total_ms=1.0,
            ttft_ms=0.5,
            tpot_ms=0.5,
            timing_mode="reported",
            uses_kv_cache=False,
            timing_source="test",
            num_tokens=len(values),
        )


class BlockingRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, inputs):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return super().run(inputs)


class PermanentlyBlockingRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def run(self, inputs):
        self.entered.set()
        self.release.wait()
        return super().run(inputs)


class WorkerAbort(BaseException):
    pass


class ExplodingArray:
    def __array__(self, *args, **kwargs):
        raise WorkerAbort("planned batch-key abort")


class GatedArray:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.entered = threading.Event()
        self.release = threading.Event()

    def __array__(self, *args, **kwargs):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return np.asarray(self.values, *args, **kwargs)


class AbortingRuntime(BlockingRuntime):
    def run(self, inputs):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        raise WorkerAbort("planned worker abort")


class TwoStageRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.entered = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]
        self.call_lock = threading.Lock()
        self.calls = 0

    def run(self, inputs):
        with self.call_lock:
            call = self.calls
            self.calls += 1
        self.entered[call].set()
        assert self.release[call].wait(timeout=2.0)
        return super().run(inputs)


class Evaluator:
    def __init__(self):
        self.samples = 0
        self.calls = 0
        self.lock = threading.Lock()

    def add_batch(self, outputs, labels, timing_ms):
        with self.lock:
            self.calls += 1
            self.samples += len(labels)


class BlockingEvaluator(Evaluator):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def add_batch(self, outputs, labels, timing_ms):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        super().add_batch(outputs, labels, timing_ms)


class OutOfOrderRuntime(Runtime):
    def __init__(self):
        super().__init__(max_workers=2)
        self.entered = [threading.Event(), threading.Event()]
        self.release = [threading.Event(), threading.Event()]

    def run(self, inputs):
        values = next(iter(inputs.values()))
        request_id = int(values.reshape(-1)[0])
        self.entered[request_id].set()
        assert self.release[request_id].wait(timeout=2.0)
        return super().run(inputs)


class PartialRaiseAcceptedMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.called = False

    def commit_acceptance(self, now_ns, queue_depth):
        self.called = True
        with self.lock:
            self._has_events = True
            self.counters["accepted"] += 1
        raise RuntimeError("accepted metrics failed after partial mutation")


class FailBeforeAcceptedMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.failed = False
        self.called = False

    def commit_acceptance(self, now_ns, queue_depth):
        self.called = True
        if not self.failed:
            self.failed = True
            raise RuntimeError("accepted metrics failed before mutation")
        super().commit_acceptance(now_ns, queue_depth)


class BlockingPublicAcceptanceHooksMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.request_queue = None
        self.hook_entered = threading.Event()
        self.release = threading.Event()

    def _blocked_public_hook(self):
        self.hook_entered.set()
        reentry_finished = threading.Event()

        def reenter_queue():
            self.request_queue.qsize()
            reentry_finished.set()

        threading.Thread(target=reenter_queue, daemon=True).start()
        assert self.release.wait(timeout=2.0)
        assert reentry_finished.wait(timeout=1.0)
        raise AssertionError("public acceptance hook must not run")

    def claim_acceptance(self, queue_transition=None):
        self._blocked_public_hook()

    def commit_acceptance(self, now_ns, queue_depth):
        self._blocked_public_hook()

    def finish_acceptance(self, claim):
        self._blocked_public_hook()


class PermanentlyBlockedAcceptanceMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.entered = threading.Event()
        self.never_release = threading.Event()

    def preflight_acceptance(self, _request):
        self.entered.set()
        self.never_release.wait()


class PublicLockHoldingPreflightMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.entered = threading.Event()
        self.release = threading.Event()

    def preflight_acceptance(self, _request):
        self.lock.acquire()
        try:
            self.entered.set()
            self.release.wait()
        finally:
            self.lock.release()


class ForbiddenPublicInflight:
    def __init__(self):
        self.called = threading.Event()

    def update(self, *_args):
        self.called.set()
        raise AssertionError("public inflight update must not run")

    def summary(self, *_args):
        self.called.set()
        raise AssertionError("public inflight summary must not run")


class ControlLockReentrantMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.engine = None
        self.worker_shutdown_recorded = threading.Event()

    def add_invalid_reason(self, reason):
        if reason == "worker_shutdown_failed":
            acquired = self.engine._control_lock.acquire(blocking=False)
            if not acquired:
                raise AssertionError("metrics callback ran under control lock")
            self.engine._control_lock.release()
            self.worker_shutdown_recorded.set()
        super().add_invalid_reason(reason)


class GatedQueueDepthMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.entered = threading.Event()
        self.release = threading.Event()

    def record_queue_depth(self, depth, now_ns, sequence=None):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        super().record_queue_depth(depth, now_ns, sequence=sequence)


class FailingQueueDepthMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count, *, fail_on_call):
        super().__init__(started_ns, worker_count)
        self.fail_on_call = fail_on_call
        self.calls = 0

    def record_queue_depth(self, depth, now_ns, sequence=None):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("queue depth metrics unavailable")
        super().record_queue_depth(depth, now_ns, sequence=sequence)


class ReentrantQueueDepthMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.request_queue = None
        self.reentered = threading.Event()

    def record_queue_depth(self, depth, now_ns, sequence=None):
        assert self.request_queue.qsize() >= 0
        self.reentered.set()
        super().record_queue_depth(depth, now_ns, sequence=sequence)


class GatedAcceptanceMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.entered = threading.Event()
        self.release = threading.Event()

    def preflight_acceptance(self, _request):
        self.entered.set()
        assert self.release.wait(timeout=2.0)

    def record_accepted(self, now_ns, queue_depth):
        self.preflight_acceptance(None)
        super().record_accepted(now_ns, queue_depth)


class ReentrantAcceptanceMetrics(AsyncMetricsCollector):
    def __init__(self, started_ns, worker_count):
        super().__init__(started_ns, worker_count)
        self.request_queue = None
        self.entered = threading.Event()
        self.reentered = threading.Event()

    def preflight_acceptance(self, _request):
        self.entered.set()
        assert self.request_queue.qsize() >= 0
        self.reentered.set()

    def record_accepted(self, now_ns, queue_depth):
        self.preflight_acceptance(None)
        super().record_accepted(now_ns, queue_depth)


class TrackingSlots:
    def __init__(self, wrapped):
        self.wrapped = wrapped
        self.blocking_acquire_started = threading.Event()

    def acquire(self, blocking=True, timeout=None):
        if blocking:
            self.blocking_acquire_started.set()
        return self.wrapped.acquire(blocking=blocking, timeout=timeout)

    def release(self):
        return self.wrapped.release()


class FlushTrackingCondition(threading.Condition):
    def __init__(self):
        super().__init__()
        self.flush_wait_started = threading.Event()

    def wait(self, timeout=None):
        if threading.current_thread().name == "engine-flush-test":
            self.flush_wait_started.set()
        return super().wait(timeout)


class GateGetQueue(_RequestQueue):
    def __init__(self, maxsize):
        super().__init__(maxsize=maxsize)
        self.blocking_get_entered = threading.Event()
        self.allow_blocking_get = threading.Event()
        self.item_published = threading.Event()

    def get(self, block=True, timeout=None):
        if block:
            self.blocking_get_entered.set()
            assert self.allow_blocking_get.wait(timeout=2.0)
        return super().get(block=block, timeout=timeout)

    def take(self, block=True, timeout=None):
        if block:
            self.blocking_get_entered.set()
            assert self.allow_blocking_get.wait(timeout=2.0)
        return super().take(
            block=block,
            timeout=timeout,
        )

    def put(self, item, block=True, timeout=None):
        result = super().put(item, block=block, timeout=timeout)
        self.item_published.set()
        return result


class GateDequeuedStopQueue(_RequestQueue):
    def __init__(self, maxsize, *, expected_waiters=1):
        super().__init__(maxsize=maxsize)
        self.expected_waiters = expected_waiters
        self.waiter_lock = threading.Lock()
        self.waiter_count = 0
        self.all_waiting = threading.Event()
        self.stop_dequeued = threading.Event()
        self.allow_stop_return = threading.Event()

    def take(self, *args, **kwargs):
        with self.waiter_lock:
            self.waiter_count += 1
            if self.waiter_count >= self.expected_waiters:
                self.all_waiting.set()
        item = super().take(*args, **kwargs)
        value = item[0] if isinstance(item, tuple) else item
        if value is engine_module._STOP:
            self.stop_dequeued.set()
            assert self.allow_stop_return.wait(timeout=2.0)
        return item


def build(
    config,
    runtime=None,
    *,
    static_batched=False,
    force_llm=False,
    trace_callback=None,
    evaluator=None,
    metrics=None,
):
    runtime = runtime or Runtime()
    pipeline = InferencePipeline(
        Loader(static_batched=static_batched),
        runtime,
    )
    if force_llm:
        pipeline.is_llm = True
    metrics = metrics or AsyncMetricsCollector(
        time.monotonic_ns(),
        config.worker_count,
    )
    evaluator = evaluator or Evaluator()
    coordinator = CompletionCoordinator(
        pipeline,
        evaluator,
        None,
        metrics,
        queue_capacity=config.worker_count,
        trace_callback=trace_callback,
    )
    engine = AsyncInferenceEngine(
        runtime,
        pipeline,
        config,
        coordinator,
        metrics,
    )
    return engine, runtime, evaluator, metrics


def make_request(request_id, *, input_size=1, sample_count=1):
    now = time.monotonic_ns()
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={
            "input": np.full(input_size, request_id, dtype=np.float32),
            "label": np.arange(sample_count),
        },
        scheduled_ns=now,
        issued_ns=now,
        enqueued_ns=0,
        sample_count=sample_count,
    )


def crash_completion(request_id=999):
    request = make_request(request_id)
    now = time.monotonic_ns()
    return BatchCompletion(
        requests=[replace(request, enqueued_ns=now)],
        collated={},
        outputs=None,
        timing_ms=None,
        runtime_started_ns=now,
        runtime_finished_ns=now,
        worker_id=-1,
        batch_size=1,
        error_type="InjectedError",
        error_message="trigger coordinator handler",
    )


def assert_slots_fully_released(engine, capacity):
    for _ in range(capacity):
        assert engine.slots.acquire(blocking=False)
    assert not engine.slots.acquire(blocking=False)
    for _ in range(capacity):
        engine.slots.release()


def test_engine_dynamically_batches_and_drains_every_request():
    config = AsyncInferenceConfig(
        queue_capacity=8,
        max_batch_size=4,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(config)
    engine.start()
    for request_id in range(8):
        assert engine.submit(make_request(request_id), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 8
    assert sum(runtime.batch_sizes) == 8
    assert max(runtime.batch_sizes) == 4
    assert all(size <= config.max_batch_size for size in runtime.batch_sizes)
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["queue"]["inflight_min"] >= 0
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_terminal_flush_releases_idle_worker_and_coordinator_payload_locals():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = WeakrefRuntime()
    engine, _, _, _ = build(config, runtime)
    request = make_request(0)
    request_ref = weakref.ref(request)
    payload_ref = weakref.ref(request.sample["input"])

    engine.start()
    assert engine.submit(request, block=True)
    del request
    engine.close_submission()
    assert engine.flush() is True
    gc.collect()

    assert request_ref() is None
    assert payload_ref() is None
    assert all(reference() is None for reference in runtime.input_refs)
    assert all(reference() is None for reference in runtime.output_refs)
    assert engine.shutdown() is True


def test_runtime_failure_does_not_deadlock_flush():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, evaluator, metrics = build(config, Runtime(fail=True))
    engine.start()
    assert engine.submit(make_request(0), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["details"]["failure_types"] == {"RuntimeError": 1}
    assert result["details"]["failure_request_examples"] == {
        "RuntimeError": [0]
    }
    assert evaluator.samples == 0


def test_engine_rejects_batch_size_above_runtime_capability():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        min_samples=1,
    )
    with pytest.raises(ValueError, match="exceeds runtime capability"):
        build(config, Runtime(max_batch_size=1))


def test_engine_rejects_dynamic_batching_without_runtime_support():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        min_samples=1,
    )
    with pytest.raises(ValueError, match="does not support dynamic batching"):
        build(config, Runtime(dynamic_batching=False))


def test_engine_rejects_worker_count_above_runtime_capability():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        worker_count=2,
        min_samples=1,
    )
    with pytest.raises(ValueError, match="worker_count=2 exceeds"):
        build(config, Runtime(max_workers=1))


@pytest.mark.parametrize(
    ("worker_count", "completion_capacity", "message"),
    [
        (1, 0, "completion queue must be strictly bounded"),
        (2, 1, "completion queue capacity=1 must equal worker_count=2"),
    ],
)
def test_engine_validates_completion_queue_worker_contract(
    worker_count,
    completion_capacity,
    message,
):
    config = AsyncInferenceConfig(
        queue_capacity=2,
        worker_count=worker_count,
        min_samples=1,
    )
    runtime = Runtime(max_workers=worker_count)
    pipeline = InferencePipeline(Loader(), runtime)
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count)
    coordinator = CompletionCoordinator(
        pipeline,
        Evaluator(),
        None,
        metrics,
        queue_capacity=completion_capacity,
    )

    with pytest.raises(ValueError, match=message):
        AsyncInferenceEngine(
            runtime,
            pipeline,
            config,
            coordinator,
            metrics,
        )


def test_engine_lifecycle_rejects_invalid_start_and_submit_states():
    config = AsyncInferenceConfig(queue_capacity=1, min_samples=1)
    engine, _, _, _ = build(config)
    with pytest.raises(RuntimeError, match="cannot submit in created"):
        engine.submit(make_request(0), block=False)
    with pytest.raises(RuntimeError, match="before start"):
        engine.shutdown()

    engine.start()
    with pytest.raises(RuntimeError, match="cannot start engine in running"):
        engine.start()
    engine.close_submission()
    with pytest.raises(RuntimeError, match="cannot submit in draining"):
        engine.submit(make_request(0), block=False)
    assert engine.flush() is True
    assert engine.shutdown() is True


def test_duplicate_registration_is_rejected_once_without_counter_drift():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, _, metrics = build(config, runtime)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered.wait(timeout=1.0)

    with pytest.raises(ValueError, match="duplicate request_id"):
        engine.submit(make_request(0), block=True)

    engine.close_submission()
    runtime.release.set()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_cancel_queued_terminalizes_requests_before_flush():
    config = AsyncInferenceConfig(queue_capacity=4, min_samples=1)
    engine, _, _, metrics = build(config)
    engine.coordinator.start()

    queued = make_request(0)
    now = time.monotonic_ns()
    queued = replace(queued, enqueued_ns=now)
    engine.coordinator.register(queued)
    metrics.record_submitted()
    metrics.record_accepted(now_ns=now, queue_depth=1)
    engine.requests.put_nowait(queued)
    engine.slots.acquire(blocking=False)

    assert engine.cancel_queued("KeyboardInterrupt") == 1
    assert engine.flush() is True
    assert engine.coordinator.stop(timeout=1.0) is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["failure_types"] == {"CancelledError": 1}
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_cancel_completion_batch_size_is_sum_of_request_sample_counts():
    config = AsyncInferenceConfig(queue_capacity=2, min_samples=1)
    traces = []
    engine, _, _, metrics = build(config, trace_callback=traces.append)
    engine.coordinator.start()

    for request_id, sample_count in enumerate((2, 3)):
        queued = replace(
            make_request(request_id, sample_count=sample_count),
            enqueued_ns=time.monotonic_ns(),
        )
        engine.coordinator.register(queued)
        metrics.record_submitted()
        metrics.record_accepted(
            now_ns=queued.enqueued_ns,
            queue_depth=request_id + 1,
        )
        engine.requests.put_nowait(queued)
        assert engine.slots.acquire(blocking=False)

    assert engine.cancel_queued("KeyboardInterrupt") == 2
    assert engine.flush() is True
    assert engine.coordinator.stop(timeout=1.0) is True

    result = metrics.finalize(time.monotonic_ns())
    assert [trace.batch_size for trace in traces] == [5, 5]
    assert result["summary"]["async_failed_requests"] == 2
    assert result["details"]["counts"]["failed_samples"] == 5
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_cancel_queued_claims_incompatible_worker_pending_request():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, evaluator, metrics = build(config, runtime)
    engine.start()
    assert engine.submit(make_request(0, input_size=1), block=True)
    assert engine.submit(make_request(1, input_size=2), block=True)
    assert runtime.entered.wait(timeout=1.0)

    try:
        assert engine.cancel_queued("KeyboardInterrupt") == 1
    finally:
        runtime.release.set()
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 1
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["failure_types"] == {"CancelledError": 1}
    assert result["details"]["counts"]["terminal"] == 2
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_cancel_claims_candidate_while_compatibility_key_is_blocked():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    gated_input = GatedArray(np.ones(2, dtype=np.float32))
    candidate = replace(
        make_request(1),
        sample={"input": gated_input, "label": np.asarray([1])},
    )
    engine, runtime, evaluator, metrics = build(config)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert engine.submit(candidate, block=True)
    assert gated_input.entered.wait(timeout=1.0)

    try:
        assert engine.cancel_queued("KeyboardInterrupt") == 1
    finally:
        gated_input.release.set()
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == [1]
    assert evaluator.samples == 1
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counts"]["terminal"] == 2
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_nonblocking_submit_rejects_when_bounded_queue_is_full():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, evaluator, metrics = build(config, runtime)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered.wait(timeout=1.0)
    assert engine.submit(make_request(1), block=True)

    assert engine.submit(make_request(2), block=False) is False

    engine.close_submission()
    runtime.release.set()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 3
    assert result["summary"]["async_accepted_requests"] == 2
    assert result["summary"]["async_rejected_requests"] == 1
    assert evaluator.samples == 2
    assert result["details"]["queue"]["depth_max"] <= 1


def test_static_batched_request_is_atomic_and_skips_dynamic_batch_capability():
    config = AsyncInferenceConfig(
        queue_capacity=4,
        max_batch_size=4,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = Runtime(dynamic_batching=False)
    traces = []
    engine, _, evaluator, metrics = build(
        config,
        runtime,
        static_batched=True,
        trace_callback=traces.append,
    )
    engine.start()
    assert engine.submit(make_request(0, input_size=2, sample_count=2), True)
    assert engine.submit(make_request(1, input_size=2, sample_count=2), True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == [2, 2]
    assert evaluator.samples == 4
    assert result["summary"]["async_completed_requests"] == 2
    assert result["summary"]["async_completed_samples"] == 4
    assert result["details"]["batch_size"]["mean"] == 2
    assert [trace.batch_size for trace in traces] == [2, 2]


def test_incompatible_input_shapes_are_sealed_into_separate_batches():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, _ = build(config)
    engine.start()
    assert engine.submit(make_request(0, input_size=1), block=True)
    assert engine.submit(make_request(1, input_size=2), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    assert runtime.batch_sizes == [1, 1]
    assert evaluator.samples == 2


def test_batch_key_includes_task_options_dtype_and_non_batch_shape():
    config = AsyncInferenceConfig(queue_capacity=2, min_samples=1)
    engine, _, _, _ = build(config)
    first = replace(
        make_request(0),
        sample={"input": np.zeros((1, 3), dtype=np.float32), "label": [0]},
        task="nlp_generation",
        generation_options={"temperature": 0.0, "stop_token_ids": [1, 2]},
        batch_axis=0,
    )
    same_non_batch_shape = replace(
        first,
        request_id=1,
        sample={"input": np.zeros((2, 3), dtype=np.float32), "label": [1]},
    )
    different_shape = replace(
        first,
        request_id=2,
        sample={"input": np.zeros((1, 4), dtype=np.float32), "label": [2]},
    )
    different_dtype = replace(
        first,
        request_id=3,
        sample={"input": np.zeros((1, 3), dtype=np.int32), "label": [3]},
    )
    different_task = replace(first, request_id=4, task="classification")
    different_options = replace(
        first,
        request_id=5,
        generation_options={"temperature": 0.8, "stop_token_ids": [1, 2]},
    )

    assert engine._batch_key(first) == engine._batch_key(same_non_batch_shape)
    for incompatible in (
        different_shape,
        different_dtype,
        different_task,
        different_options,
    ):
        assert engine._batch_key(first) != engine._batch_key(incompatible)


def test_prebatched_requests_concatenate_along_declared_batch_axis():
    config = AsyncInferenceConfig(
        queue_capacity=3,
        max_batch_size=3,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(config)
    first = replace(
        make_request(0, sample_count=1),
        sample={
            "input": np.zeros((1, 3), dtype=np.float32),
            "label": np.asarray([0]),
        },
        batch_axis=0,
    )
    second = replace(
        make_request(1, sample_count=2),
        sample={
            "input": np.ones((2, 3), dtype=np.float32),
            "label": np.asarray([1, 2]),
        },
        batch_axis=0,
    )

    engine.start()
    assert engine.submit(first, block=True)
    assert engine.submit(second, block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == [3]
    assert evaluator.samples == 3
    assert result["details"]["batch_size"]["mean"] == 3


def test_submit_rejects_declared_batch_axis_sample_count_mismatch():
    config = AsyncInferenceConfig(
        queue_capacity=3,
        max_batch_size=3,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(config)
    request = replace(
        make_request(0, sample_count=1),
        sample={
            "input": np.zeros((2, 3), dtype=np.float32),
            "label": np.asarray([0, 1]),
        },
        batch_axis=0,
    )

    engine.start()
    assert engine.submit(request, block=True) is False
    engine.close_submission()
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == []
    assert evaluator.samples == 0
    assert result["summary"]["async_submitted_requests"] == 1
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counts"]["rejected:invalid_request"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_submit_rejects_single_request_above_configured_batch_cap():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, metrics = build(config)
    request = replace(
        make_request(0, sample_count=3),
        sample={
            "input": np.zeros((3, 2), dtype=np.float32),
            "label": np.asarray([0, 1, 2]),
        },
        batch_axis=0,
    )

    engine.start()
    assert engine.submit(request, block=True) is False
    engine.close_submission()
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == []
    assert evaluator.samples == 0
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_static_request_rejects_sample_count_above_runtime_capability():
    config = AsyncInferenceConfig(
        queue_capacity=3,
        max_batch_size=3,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = Runtime(
        max_batch_size=2,
        dynamic_batching=False,
    )
    engine, _, evaluator, metrics = build(
        config,
        runtime,
        static_batched=True,
    )

    engine.start()
    assert engine.submit(
        make_request(0, input_size=3, sample_count=3),
        block=True,
    ) is False
    engine.close_submission()
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == []
    assert evaluator.samples == 0
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_prebatched_requests_respect_max_actual_batch_size():
    config = AsyncInferenceConfig(
        queue_capacity=3,
        max_batch_size=3,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, runtime, evaluator, _ = build(config)
    requests = [
        replace(
            make_request(request_id, sample_count=2),
            sample={
                "input": np.full(
                    (2, 3),
                    request_id,
                    dtype=np.float32,
                ),
                "label": np.asarray([0, 1]),
            },
            batch_axis=0,
        )
        for request_id in range(2)
    ]

    engine.start()
    for request in requests:
        assert engine.submit(request, block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert runtime.batch_sizes == [2, 2]
    assert evaluator.samples == 4


def test_llm_requests_with_different_generation_options_do_not_batch():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = LLMRuntime()
    engine, _, evaluator, _ = build(
        config,
        runtime,
        force_llm=True,
    )
    first = replace(
        make_request(0),
        task="nlp_generation",
        generation_options={"temperature": 0.0},
    )
    second = replace(
        make_request(1),
        task="nlp_generation",
        generation_options={"temperature": 0.8},
    )
    engine.start()
    assert engine.submit(first, block=True)
    assert engine.submit(second, block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    assert runtime.batch_sizes == [1, 1]
    assert evaluator.samples == 2


def test_worker_base_exception_terminalizes_owned_and_queued_requests():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = AbortingRuntime()
    engine, _, evaluator, metrics = build(config, runtime)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered.wait(timeout=1.0)
    assert engine.submit(make_request(1), block=True)
    engine.close_submission()

    runtime.release.set()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert engine.state is EngineState.STOPPED
    assert evaluator.samples == 0
    assert result["summary"]["async_failed_requests"] == 2
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counts"]["terminal"] == 2
    assert result["details"]["failure_types"] == {"WorkerAbort": 2}
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_batch_assembly_failure_keeps_candidate_owned_until_terminal():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=0.2,
    )
    engine, _, _, metrics = build(config)
    bad_request = replace(
        make_request(1),
        sample={"input": ExplodingArray(), "label": np.arange(1)},
    )
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert engine.submit(bad_request, block=True)
    engine.close_submission()

    try:
        assert engine.flush() is True
        assert engine.shutdown() is True
    finally:
        if any(worker.is_alive() for worker in engine.workers):
            engine.shutdown()

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 2
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counts"]["terminal"] == 2
    assert result["details"]["failure_types"] == {"WorkerAbort": 2}
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_completion_crash_unblocks_submitter_and_releases_queued_payloads():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        submit_timeout_sec=1.0,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, _, metrics = build(config, runtime)

    def crash(_completion):
        raise RuntimeError("planned coordinator crash")

    engine.coordinator._handle = crash
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered.wait(timeout=1.0)
    assert engine.submit(make_request(1), block=True)
    tracked_slots = TrackingSlots(engine.slots)
    engine.slots = tracked_slots
    submit_started = threading.Event()

    def blocked_submit():
        submit_started.set()
        return engine.submit(make_request(2), block=True)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(blocked_submit)
        assert submit_started.wait(timeout=1.0)
        assert tracked_slots.blocking_acquire_started.wait(timeout=1.0)
        engine.coordinator.submit(crash_completion())
        assert future.result(timeout=1.0) is False

    assert engine.flush() is False
    runtime.release.set()
    assert engine.shutdown() is False
    for worker in engine.workers:
        worker.join(timeout=1.0)

    result = metrics.finalize(time.monotonic_ns())
    assert engine.state is EngineState.FAILED
    assert result["summary"]["async_submitted_requests"] == 3
    assert result["summary"]["async_accepted_requests"] == 2
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 2
    assert result["summary"]["async_outstanding_requests"] == 0
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_registration_commit_is_atomic_with_completion_crash_cleanup():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = GatedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    handler_entered = threading.Event()
    release_crash = threading.Event()

    def crash(_completion):
        handler_entered.set()
        assert release_crash.wait(timeout=2.0)
        raise RuntimeError("planned coordinator crash during registration")

    engine.coordinator._handle = crash
    engine.start()
    engine.coordinator.submit(crash_completion())
    assert handler_entered.wait(timeout=1.0)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(engine.submit, make_request(0), True)
    try:
        assert metrics.entered.wait(timeout=1.0)
        release_crash.set()
        with engine.coordinator.condition:
            assert engine.coordinator.condition.wait_for(
                lambda: engine.coordinator.thread_error is not None,
                timeout=1.0,
            )
        metrics.release.set()
        assert future.result(timeout=1.0) is False
    finally:
        metrics.release.set()
        future.result(timeout=1.0)
        executor.shutdown(wait=True)

    engine.close_submission()
    assert engine.flush() is False
    assert engine.shutdown() is False
    for worker in engine.workers:
        worker.join(timeout=1.0)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 0
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["queue"]["inflight_min"] >= 0
    assert result["details"]["counts"].get("terminal", 0) == 0
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_public_commit_override_is_bypassed_by_sealed_acceptance_accounting():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = PartialRaiseAcceptedMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, runtime, evaluator, _ = build(config, metrics=metrics)

    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == [1]
    assert evaluator.samples == 1
    assert result["summary"]["async_submitted_requests"] == 1
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_completed_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 0
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
    assert "counter_invariant_failed" not in result["details"]["invalid_reasons"]
    assert metrics.called is False
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_failing_public_commit_override_cannot_rollback_sealed_publication():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailBeforeAcceptedMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, runtime, evaluator, _ = build(config, metrics=metrics)
    engine.start()

    assert engine.submit(make_request(0), block=True) is True
    assert engine.submit(make_request(1), block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert runtime.batch_sizes == [1, 1]
    assert evaluator.samples == 2
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_accepted_requests"] == 2
    assert result["summary"]["async_rejected_requests"] == 0
    assert result["summary"]["async_completed_requests"] == 2
    assert result["details"]["queue"]["sequence_valid"] is True
    assert result["details"]["queue"]["failed_sequences"] == []
    assert result["details"]["queue"]["missing_sequence_ranges"] == []
    assert metrics.called is False
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_locked_acceptance_commit_never_dispatches_public_acceptance_hooks():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    metrics = BlockingPublicAcceptanceHooksMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    metrics.request_queue = engine.requests
    engine.start()
    result = []
    finished = threading.Event()

    def submit_request():
        try:
            result.append(engine.submit(make_request(0), block=True))
        finally:
            finished.set()

    thread = threading.Thread(target=submit_request, daemon=True)
    thread.start()
    try:
        assert finished.wait(timeout=1.0)
        assert result == [True]
        assert not metrics.hook_entered.is_set()
        engine.close_submission()
        assert engine.flush() is True
        assert engine.shutdown() is True
    finally:
        metrics.release.set()
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


def test_acceptance_preflight_can_reenter_request_queue_without_mutex_deadlock():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = ReentrantAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    metrics.request_queue = engine.requests
    engine.start()
    submitted = []
    finished = threading.Event()

    def submit_request():
        try:
            submitted.append(engine.submit(make_request(0), block=True))
        finally:
            finished.set()

    thread = threading.Thread(target=submit_request, daemon=True)
    thread.start()
    assert metrics.entered.wait(timeout=1.0)
    assert metrics.reentered.wait(timeout=1.0)
    assert finished.wait(timeout=1.0)
    assert submitted == [True]

    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_blocked_acceptance_preflight_does_not_serialize_shutdown_deadline():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    metrics = GatedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    executor = ThreadPoolExecutor(max_workers=2)
    submit = executor.submit(engine.submit, make_request(0), True)
    shutdown = None
    try:
        assert metrics.entered.wait(timeout=1.0)
        shutdown = executor.submit(engine.shutdown)
        assert shutdown.result(timeout=1.0) is False
        assert engine.state is EngineState.FAILED
        assert engine.requests.empty()
        assert engine.requests.unfinished_tasks == 0
        assert_slots_fully_released(engine, config.queue_capacity)
        before_release = metrics.finalize(time.monotonic_ns())
        assert "metrics_unavailable" in before_release["details"][
            "invalid_reasons"
        ]
        assert before_release["summary"]["async_submitted_requests"] == 1
        assert before_release["summary"]["async_accepted_requests"] == 0
        assert before_release["summary"]["async_rejected_requests"] == 1
        assert before_release["details"]["counter_invariants"]["valid"] is True
        assert engine._submission_transactions == {}
        assert engine.coordinator.reservations == {}
    finally:
        metrics.release.set()
        submit_result = submit.result(timeout=1.0)
        if shutdown is not None:
            shutdown.result(timeout=1.0)
        executor.shutdown(wait=True)

    assert submit_result is False
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 1
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True


def test_shutdown_commits_rejection_for_preflight_that_never_returns():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    metrics = PermanentlyBlockedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    submit = threading.Thread(
        target=engine.submit,
        args=(make_request(0), True),
        daemon=True,
    )
    submit.start()
    assert metrics.entered.wait(timeout=1.0)

    assert engine.shutdown() is False

    snapshot = metrics.finalize(time.monotonic_ns())
    assert submit.is_alive()
    assert engine.state is EngineState.FAILED
    assert snapshot["summary"]["async_submitted_requests"] == 1
    assert snapshot["summary"]["async_accepted_requests"] == 0
    assert snapshot["summary"]["async_rejected_requests"] == 1
    assert snapshot["summary"]["async_outstanding_requests"] == 0
    assert snapshot["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_public_metrics_lock_and_inflight_cannot_capture_shutdown_accounting():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    metrics = PublicLockHoldingPreflightMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    forbidden_inflight = ForbiddenPublicInflight()
    metrics.inflight = forbidden_inflight
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    executor = ThreadPoolExecutor(max_workers=2)
    submit = executor.submit(engine.submit, make_request(0), True)
    shutdown = None
    try:
        assert metrics.entered.wait(timeout=1.0)
        shutdown = executor.submit(engine.shutdown)
        assert shutdown.result(timeout=1.0) is False
        snapshot = executor.submit(
            metrics.finalize,
            time.monotonic_ns(),
        ).result(timeout=1.0)
        assert snapshot["summary"]["async_submitted_requests"] == 1
        assert snapshot["summary"]["async_accepted_requests"] == 0
        assert snapshot["summary"]["async_rejected_requests"] == 1
        assert snapshot["summary"]["async_outstanding_requests"] == 0
        assert snapshot["details"]["counter_invariants"]["valid"] is True
        assert forbidden_inflight.called.is_set() is False
    finally:
        metrics.release.set()
        assert submit.result(timeout=1.0) is False
        if shutdown is not None:
            shutdown.result(timeout=1.0)
        executor.shutdown(wait=True)


def test_replaced_public_inflight_cannot_break_accepted_commit_or_snapshot():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = AsyncMetricsCollector(time.monotonic_ns(), config.worker_count)
    forbidden_inflight = ForbiddenPublicInflight()
    metrics.inflight = forbidden_inflight
    engine, _, evaluator, _ = build(config, metrics=metrics)

    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    snapshot = metrics.finalize(time.monotonic_ns())
    assert snapshot["summary"]["async_accepted_requests"] == 1
    assert snapshot["summary"]["async_completed_requests"] == 1
    assert snapshot["summary"]["async_rejected_requests"] == 0
    assert snapshot["summary"]["async_outstanding_requests"] == 0
    assert snapshot["details"]["counter_invariants"]["valid"] is True
    assert snapshot["details"]["queue"]["inflight_max"] == 1
    assert evaluator.samples == 1
    assert forbidden_inflight.called.is_set() is False


def test_accepted_base_exception_resolves_from_authoritative_membership(
    monkeypatch,
):
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, evaluator, metrics = build(config)
    original = metrics_module._update_inflight_locked
    injected = False

    def interrupt_after_counter(state, value, now_ns):
        nonlocal injected
        original(state, value, now_ns)
        if not injected:
            injected = True
            raise WorkerAbort("after accepted counter")

    monkeypatch.setattr(
        metrics_module,
        "_update_inflight_locked",
        interrupt_after_counter,
    )
    engine.start()

    with pytest.raises(WorkerAbort, match="after accepted counter"):
        engine.submit(make_request(0), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 0
    assert result["summary"]["async_submitted_requests"] == 1
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 0
    assert result["summary"]["async_failed_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_rejection_base_exception_resolves_before_releasing_ownership(
    monkeypatch,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned preflight failure")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = engine_module._record_rejected_internal
    injected = False

    def interrupt_after_rejection(*args, **kwargs):
        nonlocal injected
        original(*args, **kwargs)
        if not injected:
            injected = True
            raise WorkerAbort("after rejection record")

    monkeypatch.setattr(
        engine_module,
        "_record_rejected_internal",
        interrupt_after_rejection,
    )
    engine.start()

    assert engine.submit(make_request(0), block=True) is False
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 1
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)


def test_same_request_id_rejected_attempts_count_independently():
    class RejectEveryPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = RejectEveryPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()

    assert engine.submit(make_request(7), block=True) is False
    assert engine.submit(make_request(7), block=True) is False
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_rejected_requests"] == 2
    assert result["details"]["counts"]["rejected:metrics_unavailable"] == 2
    assert result["details"]["counter_invariants"]["valid"] is True


def test_rejected_request_id_can_be_accepted_by_later_attempt():
    class RejectFirstPreflight(AsyncMetricsCollector):
        def __init__(self, started_ns, worker_count):
            super().__init__(started_ns, worker_count)
            self.calls = 0

        def preflight_acceptance(self, _request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("planned first rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = RejectFirstPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, evaluator, _ = build(config, metrics=metrics)
    engine.start()

    assert engine.submit(make_request(9), block=True) is False
    assert engine.submit(make_request(9), block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 1
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["summary"]["async_completed_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


def test_duplicate_reserved_request_is_rejected_without_aborting_owner():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = GatedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, evaluator, _ = build(config, metrics=metrics)
    engine.start()
    executor = ThreadPoolExecutor(max_workers=1)
    owner = executor.submit(engine.submit, make_request(11), True)
    try:
        assert metrics.entered.wait(timeout=1.0)
        with pytest.raises(ValueError, match="duplicate request_id"):
            engine.submit(make_request(11), block=True)
        assert 11 in engine.coordinator.reservations
        metrics.release.set()
        assert owner.result(timeout=1.0) is True
    finally:
        metrics.release.set()
        executor.shutdown(wait=True)

    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert evaluator.samples == 1
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True


@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_base_exception_around_coordinator_commit_finishes_accepted_stages(
    monkeypatch,
    fault_timing,
):
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, metrics = build(config)
    original = engine.coordinator._commit_registration_locked
    injected = False

    def interrupt_after_commit(request, *args):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before coordinator commit")
        original(request, *args)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after coordinator commit")

    monkeypatch.setattr(
        engine.coordinator,
        "_commit_registration_locked",
        interrupt_after_commit,
    )
    engine.start()

    with pytest.raises(WorkerAbort, match=f"{fault_timing} coordinator commit"):
        engine.submit(make_request(3), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 0
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}


@pytest.mark.parametrize(
    ("fault_timing", "accepted", "rejected"),
    [("before", 0, 1), ("after", 1, 0)],
)
def test_base_exception_around_queue_publish_resolves_matching_outcome(
    monkeypatch,
    fault_timing,
    accepted,
    rejected,
):
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, metrics = build(config)
    original = engine.requests.publish_accepted
    injected = False

    def interrupt_after_publish(request, collector, attempt_token):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before queue publish")
        queued = original(request, collector, attempt_token)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after queue publish")
        return queued

    monkeypatch.setattr(engine.requests, "publish_accepted", interrupt_after_publish)
    engine.start()

    with pytest.raises(WorkerAbort, match=f"{fault_timing} queue publish"):
        engine.submit(make_request(4), block=True)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == accepted
    assert result["summary"]["async_rejected_requests"] == rejected
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}


@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_rejection_cleanup_reenters_around_slot_release_exception(
    monkeypatch,
    fault_timing,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = engine._release_slot_once
    injected = False

    def interrupt_after_release(transaction):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before slot release")
        original(transaction)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after slot release")

    monkeypatch.setattr(engine, "_release_slot_once", interrupt_after_release)
    engine.start()

    with pytest.raises(WorkerAbort, match=f"{fault_timing} slot release"):
        engine.submit(make_request(5), block=True)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


@pytest.mark.parametrize(
    "stage_name",
    ["_mark_submission_terminal", "_remove_submission_transaction"],
)
@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_rejection_cleanup_reenters_around_transaction_terminal_stages(
    monkeypatch,
    stage_name,
    fault_timing,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = getattr(engine, stage_name)
    injected = False

    def interrupt(transaction, *args):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort(f"before {stage_name}")
        original(transaction, *args)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort(f"after {stage_name}")

    monkeypatch.setattr(engine, stage_name, interrupt)
    engine.start()

    with pytest.raises(WorkerAbort, match=f"{fault_timing} {stage_name}"):
        engine.submit(make_request(6), block=True)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


@pytest.mark.parametrize(
    "stage_name",
    ["_mark_submission_terminal", "_remove_submission_transaction"],
)
@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_acceptance_cleanup_reenters_around_transaction_terminal_stages(
    monkeypatch,
    stage_name,
    fault_timing,
):
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, _, metrics = build(config, runtime)
    original = getattr(engine, stage_name)
    injected = False

    def interrupt(transaction, *args):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort(f"before accepted {stage_name}")
        original(transaction, *args)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort(f"after accepted {stage_name}")

    monkeypatch.setattr(engine, stage_name, interrupt)
    engine.start()

    with pytest.raises(
        WorkerAbort,
        match=f"{fault_timing} accepted {stage_name}",
    ):
        engine.submit(make_request(12), block=True)

    runtime.release.set()
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_rejected_requests"] == 0
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}


@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_rejection_cleanup_resolves_reservation_abort_ambiguity(
    monkeypatch,
    fault_timing,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = engine.coordinator.abort_registration
    injected = False

    def interrupt(request_id, *args, **kwargs):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before reservation abort")
        original(request_id, *args, **kwargs)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after reservation abort")

    monkeypatch.setattr(engine.coordinator, "abort_registration", interrupt)
    engine.start()

    with pytest.raises(
        WorkerAbort,
        match=f"{fault_timing} reservation abort",
    ):
        engine.submit(make_request(8), block=True)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


@pytest.mark.parametrize("fault_timing", ["before", "after"])
def test_rejection_outcome_rebuild_restores_reason_and_evidence(
    monkeypatch,
    fault_timing,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = metrics_module._rebuild_outcome_accounting_locked
    injected = False

    def interrupt(state):
        nonlocal injected
        if not injected and fault_timing == "before":
            injected = True
            raise WorkerAbort("before outcome rebuild")
        original(state)
        if not injected and fault_timing == "after":
            injected = True
            raise WorkerAbort("after outcome rebuild")

    monkeypatch.setattr(
        metrics_module,
        "_rebuild_outcome_accounting_locked",
        interrupt,
    )
    engine.start()

    assert engine.submit(make_request(10), block=True) is False

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_rejected_requests"] == 1
    assert result["details"]["counts"]["rejected:metrics_unavailable"] == 1
    assert "request_rejected" in result["details"]["invalid_reasons"]
    assert result["details"]["counter_invariants"]["valid"] is True
    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


def test_queue_publish_rolls_back_if_put_raises_after_mutation(monkeypatch):
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count=1)
    request_queue = _RequestQueue(maxsize=1, transition_metrics=metrics)
    transaction = engine_module._SubmissionTransaction(0, 20)
    original = request_queue._put

    def interrupt_after_put(item):
        original(item)
        raise WorkerAbort("after queue put")

    monkeypatch.setattr(request_queue, "_put", interrupt_after_put)

    with pytest.raises(WorkerAbort, match="after queue put"):
        request_queue.publish_accepted(make_request(20), metrics, transaction)

    assert request_queue.empty()
    assert request_queue.unfinished_tasks == 0
    assert metrics_module._accounting_outcome_internal(metrics, 0) is None


def test_queue_publish_rolls_back_task_and_marks_allocated_sequence(monkeypatch):
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count=1)
    request_queue = _RequestQueue(maxsize=1, transition_metrics=metrics)
    transaction = engine_module._SubmissionTransaction(0, 21)
    original = request_queue._capture_transition

    def interrupt_after_transition(*args, **kwargs):
        original(*args, **kwargs)
        raise WorkerAbort("after transition allocation")

    monkeypatch.setattr(
        request_queue,
        "_capture_transition",
        interrupt_after_transition,
    )

    with pytest.raises(WorkerAbort, match="after transition allocation"):
        request_queue.publish_accepted(make_request(21), metrics, transaction)

    result = metrics.finalize(time.monotonic_ns())
    assert request_queue.empty()
    assert request_queue.unfinished_tasks == 0
    assert result["details"]["queue"]["failed_sequences"] == [1]
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]


def test_direct_queue_publish_marks_allocated_sequence_failed_on_rollback(
    monkeypatch,
):
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count=1)
    request_queue = _RequestQueue(maxsize=1, transition_metrics=metrics)
    original = request_queue._capture_transition

    def interrupt_after_transition(*args, **kwargs):
        original(*args, **kwargs)
        raise WorkerAbort("after direct transition allocation")

    monkeypatch.setattr(
        request_queue,
        "_capture_transition",
        interrupt_after_transition,
    )

    with pytest.raises(WorkerAbort, match="after direct transition allocation"):
        request_queue.publish(make_request(22))

    result = metrics.finalize(time.monotonic_ns())
    assert request_queue.empty()
    assert request_queue.unfinished_tasks == 0
    assert result["details"]["queue"]["failed_sequences"] == [1]
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]


def test_queue_publish_rolls_back_payload_evidence_fault():
    class InterruptingTransaction:
        attempt_token = 0

        def __init__(self):
            self.injected = False
            self.queued_request = None

        def __setattr__(self, name, value):
            object.__setattr__(self, name, value)
            if (
                name == "queued_request"
                and value is not None
                and not self.injected
            ):
                object.__setattr__(self, "injected", True)
                raise WorkerAbort("after payload evidence")

    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count=1)
    request_queue = _RequestQueue(maxsize=1, transition_metrics=metrics)
    transaction = InterruptingTransaction()

    with pytest.raises(WorkerAbort, match="after payload evidence"):
        request_queue.publish_accepted(make_request(22), metrics, transaction)

    assert request_queue.empty()
    assert request_queue.unfinished_tasks == 0
    assert transaction.queued_request is None
    assert metrics_module._accounting_outcome_internal(metrics, 0) is None


def test_preflight_keyboard_interrupt_is_reraised_after_rejection_cleanup():
    class InterruptingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise KeyboardInterrupt("planned preflight interrupt")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = InterruptingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()

    with pytest.raises(KeyboardInterrupt, match="planned preflight interrupt"):
        engine.submit(make_request(23), block=True)

    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


def test_terminal_bitmap_completes_accepted_recovery_without_masking_interrupt(
    monkeypatch,
):
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    engine, _, _, metrics = build(config, runtime)
    original = engine.coordinator._commit_registration_locked

    def terminalize_then_interrupt(request, *args):
        original(request, *args)
        engine.coordinator.terminal[request.request_id] = 2
        engine.coordinator.outstanding.pop(request.request_id, None)
        engine.coordinator.condition.notify_all()
        raise KeyboardInterrupt("after terminal pop")

    monkeypatch.setattr(
        engine.coordinator,
        "_commit_registration_locked",
        terminalize_then_interrupt,
    )
    engine.start()

    with pytest.raises(KeyboardInterrupt, match="after terminal pop"):
        engine.submit(make_request(24), block=True)

    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert engine.coordinator.outstanding == {}
    runtime.release.set()
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    assert metrics.finalize(time.monotonic_ns())["summary"][
        "async_accepted_requests"
    ] == 1


def test_slot_pool_release_is_idempotent_after_inner_mutation_fault(monkeypatch):
    pool = engine_module._SlotLeasePool(capacity=1)
    assert pool.acquire_lease(30, blocking=False) is True
    original = pool._release_lease_locked
    injected = False

    def interrupt_after_remove(token):
        nonlocal injected
        released = original(token)
        if not injected:
            injected = True
            raise WorkerAbort("after lease removal")
        return released

    monkeypatch.setattr(pool, "_release_lease_locked", interrupt_after_remove)

    with pytest.raises(WorkerAbort, match="after lease removal"):
        pool.release_lease(30)

    assert pool.contains(30) is False
    assert pool.release_lease(30) is False
    assert pool.acquire_lease(31, blocking=False) is True
    assert pool.held_count == 1


def test_slot_pool_concurrent_release_removes_one_membership():
    pool = engine_module._SlotLeasePool(capacity=1)
    assert pool.acquire_lease(32, blocking=False) is True
    ready = threading.Barrier(3)

    def release():
        ready.wait(timeout=1.0)
        return pool.release_lease(32)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(release)
        second = executor.submit(release)
        ready.wait(timeout=1.0)
        results = [first.result(timeout=1.0), second.result(timeout=1.0)]

    assert sorted(results) == [False, True]
    assert pool.held_count == 0


def test_abort_system_exit_after_mutation_is_reraised_with_clean_state(
    monkeypatch,
):
    class FailingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise RuntimeError("planned rejection")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = engine.coordinator.abort_registration
    injected = False

    def interrupt_after_abort(request_id, *args, **kwargs):
        nonlocal injected
        result = original(request_id, *args, **kwargs)
        if not injected:
            injected = True
            raise SystemExit("after reservation removal")
        return result

    monkeypatch.setattr(
        engine.coordinator,
        "abort_registration",
        interrupt_after_abort,
    )
    engine.start()

    with pytest.raises(SystemExit, match="after reservation removal"):
        engine.submit(make_request(33), block=True)

    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert_slots_fully_released(engine, config.queue_capacity)
    engine.close_submission()
    assert engine.shutdown() is True


def test_two_engines_sharing_metrics_allocate_distinct_attempt_tokens():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = AsyncMetricsCollector(time.monotonic_ns(), config.worker_count)
    first, _, _, _ = build(config, metrics=metrics)
    second, _, _, _ = build(config, metrics=metrics)
    first.start()
    second.start()

    assert first.submit(make_request(41), block=True) is True
    assert second.submit(make_request(42), block=True) is True
    first.close_submission()
    second.close_submission()
    assert first.flush() is True
    assert second.flush() is True
    assert first.shutdown() is True
    assert second.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_submitted_requests"] == 2
    assert result["summary"]["async_accepted_requests"] == 2
    assert result["summary"]["async_completed_requests"] == 2
    assert result["details"]["counter_invariants"]["valid"] is True


def test_request_identity_is_normalized_before_lifecycle_locks():
    class ExactIdentityMetrics(AsyncMetricsCollector):
        def preflight_acceptance(self, request):
            assert type(request.request_id) is int
            assert type(request.submission_token) is int

    class GuardedInt(int):
        engine = None
        conversions = 0

        def __int__(self):
            type(self).conversions += 1
            assert not type(self).engine.state_condition._is_owned()
            assert not type(self).engine.coordinator.condition._is_owned()
            return super().__int__()

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = ExactIdentityMetrics(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    GuardedInt.engine = engine
    request = replace(make_request(43), request_id=GuardedInt(43))
    engine.start()

    assert engine.submit(request, block=True) is True
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    assert GuardedInt.conversions >= 1


def test_original_keyboard_interrupt_survives_abort_recovery_fault(monkeypatch):
    class InterruptingPreflight(AsyncMetricsCollector):
        def preflight_acceptance(self, _request):
            raise KeyboardInterrupt("original preflight interrupt")

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = InterruptingPreflight(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    original = engine.coordinator.abort_registration
    injected = False

    def interrupt_after_abort(*args, **kwargs):
        nonlocal injected
        result = original(*args, **kwargs)
        if not injected:
            injected = True
            raise SystemExit("recovery abort fault")
        return result

    monkeypatch.setattr(
        engine.coordinator,
        "abort_registration",
        interrupt_after_abort,
    )
    engine.start()

    with pytest.raises(KeyboardInterrupt, match="original preflight interrupt"):
        engine.submit(make_request(44), block=True)

    assert engine._submission_transactions == {}
    assert engine.coordinator.reservations == {}
    assert engine._slot_pool.held_count == 0
    engine.close_submission()
    assert engine.shutdown() is True


def test_accepted_recovery_never_commits_replacement_reservation():
    config = AsyncInferenceConfig(queue_capacity=1, min_samples=1)
    engine, _, _, metrics = build(config)
    original = replace(
        make_request(45),
        enqueued_ns=time.monotonic_ns(),
        submission_token=200,
    )
    replacement = replace(original, sample_index=46, submission_token=201)
    transition = engine_module._QueueTransition(
        depth=1,
        now_ns=original.enqueued_ns,
        sequence=1,
    )
    metrics_module._commit_acceptance_internal(
        metrics,
        original.enqueued_ns,
        1,
        queue_transition=transition,
        attempt_token=200,
        request_id=45,
    )
    engine.coordinator.reserve_registration(replacement, attempt_token=201)
    transaction = engine_module._SubmissionTransaction(200, 45)
    transaction.queued_request = original
    engine._submission_transactions[45] = transaction

    with pytest.raises(RuntimeError, match="ownership missing"):
        engine._complete_accepted_submission(transaction)

    with engine.coordinator.condition:
        reservation = engine.coordinator.reservations[45]
        assert reservation.attempt_token == 201
        assert engine.coordinator.outstanding == {}
    assert engine._submission_transactions[45] is transaction


def test_concurrent_rejection_cleanup_releases_one_authoritative_lease():
    config = AsyncInferenceConfig(queue_capacity=1, min_samples=1)
    engine, _, _, _ = build(config)
    transaction = engine_module._SubmissionTransaction(301, 46)
    engine._submission_transactions[46] = transaction
    assert engine._slot_pool.acquire_lease(301, blocking=False) is True
    ready = threading.Barrier(3)

    def cleanup():
        ready.wait(timeout=1.0)
        engine._complete_rejected_submission(transaction)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(cleanup)
        second = executor.submit(cleanup)
        ready.wait(timeout=1.0)
        assert first.result(timeout=1.0) is None
        assert second.result(timeout=1.0) is None

    assert engine._slot_pool.held_count == 0
    assert engine._submission_transactions == {}
    assert transaction.terminal_state == "rejected"


def test_blocked_trailing_queue_transition_latches_missing_snapshot():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = GatedQueueDepthMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    assert metrics.entered.wait(timeout=1.0)

    try:
        first = metrics.finalize(time.monotonic_ns())
        second = metrics.finalize(time.monotonic_ns())
        for snapshot in (first, second):
            queue_metrics = snapshot["details"]["queue"]
            assert queue_metrics["sequence_valid"] is False
            assert queue_metrics["sequence_high_water"] == 2
            assert queue_metrics["missing_sequence_ranges"] == [[2, 2]]
            assert queue_metrics["depth_min"] is None
            assert queue_metrics["depth_max"] is None
            assert queue_metrics["depth_mean"] is None
            assert "metrics_unavailable" in snapshot["details"][
                "invalid_reasons"
            ]
    finally:
        metrics.release.set()

    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    after_delivery = metrics.finalize(time.monotonic_ns())
    assert after_delivery["details"]["queue"]["sequence_valid"] is False
    assert after_delivery["details"]["queue"]["missing_sequence_ranges"] == [
        [2, 2]
    ]
    assert after_delivery["details"]["queue"]["depth_mean"] is None


def test_close_submission_can_cancel_stale_acceptance_preflight():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = GatedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    executor = ThreadPoolExecutor(max_workers=2)
    submit = executor.submit(engine.submit, make_request(0), True)
    close = None
    try:
        assert metrics.entered.wait(timeout=1.0)
        close = executor.submit(engine.close_submission)
        assert close.result(timeout=1.0) is None
        assert engine.state is EngineState.DRAINING
        metrics.release.set()
        assert submit.result(timeout=1.0) is False
    finally:
        metrics.release.set()
        submit.result(timeout=1.0)
        if close is not None:
            close.result(timeout=1.0)
        executor.shutdown(wait=True)

    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == 0
    assert result["summary"]["async_rejected_requests"] == 1
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_enqueued_timestamp_and_depth_linearize_at_queue_publication():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = BlockingRuntime()
    metrics = GatedAcceptanceMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, runtime, metrics=metrics)
    engine.start()
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(engine.submit, make_request(0), True)
    try:
        assert metrics.entered.wait(timeout=1.0)
        assert engine.requests.empty()
        publication_not_before_ns = time.monotonic_ns()
        metrics.release.set()
        assert future.result(timeout=1.0) is True
    finally:
        metrics.release.set()
        future.result(timeout=1.0)
        executor.shutdown(wait=True)

    assert runtime.entered.wait(timeout=1.0)
    with engine.coordinator.condition:
        accepted = engine.coordinator.outstanding[0]
    assert accepted.enqueued_ns >= publication_not_before_ns

    engine.close_submission()
    runtime.release.set()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["details"]["queue"]["depth_max"] == 1
    assert result["details"]["queue"]["inflight_min"] >= 0


def test_request_queue_captures_dequeue_before_concurrent_publish_without_locking_callback():
    request_queue = _RequestQueue(maxsize=1)
    depth_events = []
    dequeue_entered = threading.Event()
    release_dequeue = threading.Event()
    publish_started = threading.Event()
    publish_completed = threading.Event()
    first = make_request(0)
    second = make_request(1)

    _, publication = request_queue.publish(first)
    depth_events.append(publication.depth)

    def record_dequeue(transition):
        depth_events.append(transition.depth)
        dequeue_entered.set()
        assert release_dequeue.wait(timeout=2.0)

    def publish_second():
        publish_started.set()

        queued, transition = request_queue.publish(second)
        depth_events.append(transition.depth)
        publish_completed.set()
        return queued

    executor = ThreadPoolExecutor(max_workers=2)
    item, transition = request_queue.take()
    take_future = executor.submit(record_dequeue, transition)
    publish_future = None
    try:
        assert dequeue_entered.wait(timeout=1.0)
        publish_future = executor.submit(publish_second)
        assert publish_started.wait(timeout=1.0)
        assert publish_completed.wait(timeout=1.0)
        release_dequeue.set()
        assert take_future.result(timeout=1.0) is None
        assert publish_future.result(timeout=1.0).request_id == 1
    finally:
        release_dequeue.set()
        take_future.result(timeout=1.0)
        if publish_future is not None:
            publish_future.result(timeout=1.0)
        executor.shutdown(wait=True)

    assert item.request_id == 0
    assert depth_events == [1, 0, 1]
    assert transition.sequence == 2
    request_queue.task_done()
    request_queue.get_nowait()
    request_queue.task_done()
    assert request_queue.unfinished_tasks == 0


def test_candidate_transition_timestamp_precedes_ownership_lock_contention(
    monkeypatch,
):
    now_ns = [10]
    monkeypatch.setattr(
        engine_module.time,
        "monotonic_ns",
        lambda: now_ns[0],
    )
    request_queue = _RequestQueue(maxsize=1)
    queued, _ = request_queue.publish(make_request(0))
    claim_entered = threading.Event()
    release_claim = threading.Event()

    def gated_claim(_request):
        claim_entered.set()
        assert release_claim.wait(timeout=2.0)

    now_ns[0] = 20
    with ThreadPoolExecutor(max_workers=1) as executor:
        candidate = executor.submit(
            request_queue.get_candidate,
            gated_claim,
        )
        assert claim_entered.wait(timeout=1.0)
        now_ns[0] = 30
        release_claim.set()
        item, transition = candidate.result(timeout=1.0)

    assert item.request_id == queued.request_id
    assert transition.now_ns == 20
    request_queue.task_done()
    assert request_queue.unfinished_tasks == 0


def test_blocked_dequeue_metrics_does_not_hold_queue_mutex_past_shutdown_deadline():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    metrics = GatedQueueDepthMetrics(time.monotonic_ns(), config.worker_count)
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert metrics.entered.wait(timeout=1.0)
    engine.close_submission()

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(engine.shutdown).result(timeout=1.0) is False

    metrics.release.set()
    for worker in engine.workers:
        worker.join(timeout=1.0)
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_dequeue_metrics_can_reenter_request_queue_without_deadlock():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = ReentrantQueueDepthMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    metrics.request_queue = engine.requests
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert metrics.reentered.wait(timeout=1.0)
    engine.close_submission()

    assert engine.flush() is True
    assert engine.shutdown() is True
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


@pytest.mark.parametrize("fail_on_call, request_count", [(1, 1), (2, 2)])
def test_dequeue_metrics_failure_terminalizes_every_worker_owned_request(
    fail_on_call,
    request_count,
):
    config = AsyncInferenceConfig(
        queue_capacity=2,
        max_batch_size=2,
        batch_timeout_ms=100,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingQueueDepthMetrics(
        time.monotonic_ns(),
        config.worker_count,
        fail_on_call=fail_on_call,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.start()
    for request_id in range(request_count):
        assert engine.submit(make_request(request_id), block=True)
    engine.close_submission()

    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == request_count
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["queue"]["sequence_valid"] is False
    assert len(result["details"]["queue"]["failed_sequences"]) == 1
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_drain_metrics_failure_preserves_cancellation_cleanup():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = FailingQueueDepthMetrics(
        time.monotonic_ns(),
        config.worker_count,
        fail_on_call=1,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    engine.coordinator.start()
    assert engine.slots.acquire(blocking=False)
    request = make_request(0)
    queued, transition = engine.requests.publish(request)
    metrics.record_accepted(
        queued.enqueued_ns,
        transition.depth,
    )
    engine.coordinator.register(queued)

    assert engine._cancel_queued("test cancellation", 1.0) == 1
    assert engine.coordinator.wait_for_requests((0,), timeout=1.0)
    assert engine.coordinator.stop(timeout=1.0)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert "metrics_unavailable" in result["details"]["invalid_reasons"]
    assert len(result["details"]["queue"]["failed_sequences"]) == 1
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_drain_metrics_can_reenter_request_queue_without_deadlock():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    metrics = ReentrantQueueDepthMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    metrics.request_queue = engine.requests
    engine.coordinator.start()
    assert engine.slots.acquire(blocking=False)
    request = make_request(0)
    queued, transition = engine.requests.publish(request)
    metrics.record_accepted(
        queued.enqueued_ns,
        transition.depth,
    )
    engine.coordinator.register(queued)

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(
            engine._cancel_queued,
            "test cancellation",
            1.0,
        ).result(timeout=1.0) == 1

    assert metrics.reentered.is_set()
    assert engine.coordinator.wait_for_requests((0,), timeout=1.0)
    assert engine.coordinator.stop(timeout=1.0)
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_late_stop_owner_does_not_republish_after_terminal_shutdown_cleanup():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    engine, _, _, _ = build(config)
    engine.requests = GateDequeuedStopQueue(maxsize=1)
    engine.start()

    with ThreadPoolExecutor(max_workers=1) as executor:
        shutdown = executor.submit(engine.shutdown)
        assert engine.requests.stop_dequeued.wait(timeout=1.0)
        assert shutdown.result(timeout=1.0) is False

    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    engine.requests.allow_stop_return.set()
    for worker in engine.workers:
        worker.join(timeout=1.0)
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


def test_multi_worker_terminal_close_wakes_idle_waiter_behind_late_stop_owner():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        worker_count=2,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    runtime = Runtime(max_workers=2)
    engine, _, _, _ = build(config, runtime)
    engine.requests = GateDequeuedStopQueue(
        maxsize=1,
        expected_waiters=2,
    )
    engine.start()
    assert engine.requests.all_waiting.wait(timeout=1.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        shutdown = executor.submit(engine.shutdown)
        assert engine.requests.stop_dequeued.wait(timeout=1.0)
        assert shutdown.result(timeout=1.0) is False

    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    engine.requests.allow_stop_return.set()
    for worker in engine.workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


def test_blocked_runtime_has_finite_flush_and_shutdown_cleanup():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    runtime = PermanentlyBlockingRuntime()
    engine, _, _, metrics = build(config, runtime)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered.wait(timeout=1.0)
    engine.close_submission()

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(engine.flush).result(timeout=1.0) is False
        assert executor.submit(engine.shutdown).result(timeout=1.0) is False

    with engine.coordinator.condition:
        assert engine.coordinator.condition.wait_for(
            lambda: not engine.coordinator.outstanding,
            timeout=1.0,
        )
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert "flush_timeout" in result["details"]["invalid_reasons"]
    assert engine.requests.empty()
    assert_slots_fully_released(engine, config.queue_capacity)

    runtime.release.set()
    for worker in engine.workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()


def test_flush_waits_only_for_requests_accepted_at_invocation():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = TwoStageRuntime()
    engine, _, _, metrics = build(config, runtime)
    tracking_condition = FlushTrackingCondition()
    engine.coordinator.condition = tracking_condition
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert runtime.entered[0].wait(timeout=1.0)

    def flush_on_named_thread():
        threading.current_thread().name = "engine-flush-test"
        return engine.flush()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(flush_on_named_thread)
    try:
        assert tracking_condition.flush_wait_started.wait(timeout=1.0)
        assert engine.submit(make_request(1), block=True)
        runtime.release[0].set()
        assert runtime.entered[1].wait(timeout=1.0)
        assert future.result(timeout=0.5) is True
    finally:
        runtime.release[0].set()
        runtime.release[1].set()
        future.result(timeout=1.0)
        executor.shutdown(wait=True)

    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True
    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_completed_requests"] == 2
    assert result["summary"]["async_outstanding_requests"] == 0


def test_full_completion_queue_cannot_make_shutdown_infinite():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    engine, _, _, _ = build(config)
    handler_entered = threading.Event()
    release_handler = threading.Event()
    worker_submit_blocked = threading.Event()
    original_handle = engine.coordinator._handle
    original_submit = engine.coordinator.submit

    def blocked_handle(completion):
        handler_entered.set()
        assert release_handler.wait(timeout=2.0)
        original_handle(completion)

    def tracked_submit(completion, *args, **kwargs):
        if completion.requests[0].request_id == 2:
            worker_submit_blocked.set()
        return original_submit(completion, *args, **kwargs)

    engine.coordinator._handle = blocked_handle
    engine.coordinator.submit = tracked_submit
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert handler_entered.wait(timeout=1.0)
    assert engine.submit(make_request(1), block=True)
    assert engine.submit(make_request(2), block=True)
    assert worker_submit_blocked.wait(timeout=1.0)
    assert engine.submit(make_request(3), block=True)
    engine.close_submission()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(engine.shutdown)
    try:
        assert future.result(timeout=1.0) is False
    finally:
        release_handler.set()
        future.result(timeout=1.0)
        executor.shutdown(wait=True)

    for worker in engine.workers:
        worker.join(timeout=1.0)
        assert not worker.is_alive()
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


def test_multi_worker_completion_is_deterministically_out_of_order():
    config = AsyncInferenceConfig(
        queue_capacity=2,
        worker_count=2,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    runtime = OutOfOrderRuntime()
    traces = []
    second_terminal = threading.Event()

    def record_trace(trace):
        traces.append(trace)
        if trace.request_id == 1:
            second_terminal.set()

    engine, _, evaluator, metrics = build(
        config,
        runtime,
        trace_callback=record_trace,
    )
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert engine.submit(make_request(1), block=True)
    engine.close_submission()
    assert runtime.entered[0].wait(timeout=1.0)
    assert runtime.entered[1].wait(timeout=1.0)

    runtime.release[1].set()
    assert second_terminal.wait(timeout=1.0)
    runtime.release[0].set()
    assert engine.flush() is True
    assert engine.shutdown() is True

    result = metrics.finalize(time.monotonic_ns())
    assert [trace.request_id for trace in traces] == [1, 0]
    assert evaluator.samples == 2
    assert result["summary"]["async_completed_requests"] == 2
    assert result["summary"]["async_outstanding_requests"] == 0


def test_blocked_external_completion_preserves_outstanding_until_gate_releases():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.05,
    )
    evaluator = BlockingEvaluator()
    engine, _, _, metrics = build(config, evaluator=evaluator)
    engine.start()
    assert engine.submit(make_request(0), block=True)
    assert evaluator.entered.wait(timeout=1.0)
    engine.close_submission()

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(engine.shutdown).result(timeout=1.0) is False

    before_release = metrics.finalize(time.monotonic_ns())
    assert engine.state is EngineState.FAILED
    assert engine.outstanding_request_ids() == (0,)
    assert before_release["summary"]["async_completed_requests"] == 0
    assert before_release["summary"]["async_failed_requests"] == 0
    assert before_release["summary"]["async_outstanding_requests"] == 1

    evaluator.release.set()
    engine.coordinator.thread.join(timeout=1.0)
    assert not engine.coordinator.thread.is_alive()
    assert engine.outstanding_request_ids() == ()
    after_release = metrics.finalize(time.monotonic_ns())
    assert after_release["summary"]["async_completed_requests"] == 1
    assert after_release["summary"]["async_failed_requests"] == 0
    assert after_release["summary"]["async_outstanding_requests"] == 0
    assert after_release["details"]["counts"]["terminal"] == 1


def test_cancel_queued_cannot_steal_concurrent_shutdown_sentinel():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, _ = build(config)
    gated_queue = GateGetQueue(maxsize=1)
    engine.requests = gated_queue
    engine.start()
    assert gated_queue.blocking_get_entered.wait(timeout=1.0)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(engine.shutdown)
    try:
        assert gated_queue.item_published.wait(timeout=1.0)
        assert engine.cancel_queued("KeyboardInterrupt") == 0
        gated_queue.allow_blocking_get.set()
        assert future.result(timeout=1.0) is True
    finally:
        gated_queue.allow_blocking_get.set()
        future.result(timeout=2.0)
        executor.shutdown(wait=True)

    assert engine.state is EngineState.STOPPED
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


def test_shutdown_deadline_starts_before_waiting_for_control_lock(monkeypatch):
    class GateLock:
        def __init__(self):
            self.lock = threading.Lock()
            self.lock.acquire()
            self.acquire_attempted = threading.Event()

        def __enter__(self):
            self.acquire_attempted.set()
            self.lock.acquire()
            return self

        def __exit__(self, *_args):
            self.lock.release()

        def release(self):
            self.lock.release()

    class ShutdownProbe(RuntimeError):
        def __init__(self, deadline):
            super().__init__(deadline)
            self.deadline = deadline

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, _ = build(config)
    engine.state = EngineState.RUNNING
    gate = GateLock()
    engine._control_lock = gate
    now = [100.0]
    monkeypatch.setattr(
        engine_module,
        "time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )

    def probe_flush(deadline):
        raise ShutdownProbe(deadline)

    engine._flush_until = probe_flush
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(engine.shutdown)
        assert gate.acquire_attempted.wait(timeout=1.0)
        now[0] = 200.0
        gate.release()
        with pytest.raises(ShutdownProbe) as error:
            future.result(timeout=1.0)

    assert error.value.deadline == pytest.approx(101.0)


def test_shutdown_records_full_stop_queue_after_releasing_control_lock():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=0.01,
    )
    metrics = ControlLockReentrantMetrics(
        time.monotonic_ns(),
        config.worker_count,
    )
    engine, _, _, _ = build(config, metrics=metrics)
    metrics.engine = engine
    engine.state = EngineState.RUNNING
    engine.requests.put_nowait(engine_module._STOP)
    engine._flush_until = lambda _deadline: True
    engine._join_workers = lambda _deadline: True
    engine.coordinator.stop = lambda _timeout: True
    engine._stop_completion_monitor = lambda: None
    engine.completion_monitor = SimpleNamespace(
        join=lambda timeout: None,
        is_alive=lambda: False,
    )

    assert engine.shutdown() is False

    assert metrics.worker_shutdown_recorded.is_set()
    snapshot = metrics.finalize(time.monotonic_ns())
    assert "worker_shutdown_failed" in snapshot["details"]["invalid_reasons"]
    assert engine.state is EngineState.FAILED
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0


def test_blocked_cancel_completion_does_not_serialize_shutdown_timeout():
    class ShutdownProbe(RuntimeError):
        pass

    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, _ = build(config)
    engine.state = EngineState.RUNNING
    cancel_entered = threading.Event()
    release_cancel = threading.Event()
    shutdown_progressed = threading.Event()

    def blocked_cancel(_reason, _timeout):
        cancel_entered.set()
        assert release_cancel.wait(timeout=2.0)
        return 0

    def probe_flush(_deadline):
        shutdown_progressed.set()
        raise ShutdownProbe

    engine._cancel_queued = blocked_cancel
    engine._flush_until = probe_flush
    executor = ThreadPoolExecutor(max_workers=2)
    cancel_future = executor.submit(engine.cancel_queued, "KeyboardInterrupt")
    shutdown_future = None
    try:
        assert cancel_entered.wait(timeout=1.0)
        shutdown_future = executor.submit(engine.shutdown)
        assert shutdown_progressed.wait(timeout=1.0)
    finally:
        release_cancel.set()
        assert cancel_future.result(timeout=1.0) == 0
        if shutdown_future is not None:
            with pytest.raises(ShutdownProbe):
                shutdown_future.result(timeout=1.0)
        executor.shutdown(wait=True)


def test_cancel_that_started_first_cannot_drain_shutdown_sentinel():
    config = AsyncInferenceConfig(
        queue_capacity=1,
        min_samples=1,
        flush_timeout_sec=1.0,
    )
    engine, _, _, _ = build(config)
    gated_queue = GateGetQueue(maxsize=1)
    engine.requests = gated_queue
    cancel_ready_to_drain = threading.Event()
    allow_cancel_drain = threading.Event()
    original_cancel = engine._cancel_queued

    def gated_cancel(reason, timeout):
        cancel_ready_to_drain.set()
        assert allow_cancel_drain.wait(timeout=2.0)
        return original_cancel(reason, timeout)

    engine._cancel_queued = gated_cancel
    engine.start()
    assert gated_queue.blocking_get_entered.wait(timeout=1.0)
    executor = ThreadPoolExecutor(max_workers=2)
    cancel_future = executor.submit(engine.cancel_queued, "KeyboardInterrupt")
    shutdown_future = None
    try:
        assert cancel_ready_to_drain.wait(timeout=1.0)
        shutdown_future = executor.submit(engine.shutdown)
        assert gated_queue.item_published.wait(timeout=1.0)
        allow_cancel_drain.set()
        assert cancel_future.result(timeout=1.0) == 0
        gated_queue.allow_blocking_get.set()
        assert shutdown_future.result(timeout=1.0) is True
    finally:
        allow_cancel_drain.set()
        gated_queue.allow_blocking_get.set()
        cancel_future.result(timeout=2.0)
        if shutdown_future is not None:
            shutdown_future.result(timeout=2.0)
        executor.shutdown(wait=True)

    assert engine.state is EngineState.STOPPED
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine
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


def build(config, runtime=None, *, static_batched=False):
    runtime = runtime or Runtime()
    pipeline = InferencePipeline(
        Loader(static_batched=static_batched),
        runtime,
    )
    metrics = AsyncMetricsCollector(time.monotonic_ns(), config.worker_count)
    evaluator = Evaluator()
    coordinator = CompletionCoordinator(
        pipeline,
        evaluator,
        None,
        metrics,
        queue_capacity=config.worker_count,
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
    engine, _, evaluator, metrics = build(
        config,
        runtime,
        static_batched=True,
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
    engine, _, _, metrics = build(config)
    handler_entered = threading.Event()
    release_crash = threading.Event()
    registered = threading.Event()
    release_register_return = threading.Event()
    original_register = engine.coordinator.register

    def crash(_completion):
        handler_entered.set()
        assert release_crash.wait(timeout=2.0)
        raise RuntimeError("planned coordinator crash during registration")

    def gated_register(request, *args, **kwargs):
        original_register(request, *args, **kwargs)
        registered.set()
        assert release_register_return.wait(timeout=2.0)

    engine.coordinator._handle = crash
    engine.coordinator.register = gated_register
    engine.start()
    engine.coordinator.submit(crash_completion())
    assert handler_entered.wait(timeout=1.0)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(engine.submit, make_request(0), True)
    try:
        assert registered.wait(timeout=1.0)
        release_crash.set()
        with engine.coordinator.condition:
            assert engine.coordinator.condition.wait_for(
                lambda: engine.coordinator.thread_error is not None,
                timeout=1.0,
            )
        release_register_return.set()
        assert future.result(timeout=1.0) is True
    finally:
        release_register_return.set()
        future.result(timeout=1.0)
        executor.shutdown(wait=True)

    engine.close_submission()
    assert engine.flush() is False
    assert engine.shutdown() is False
    for worker in engine.workers:
        worker.join(timeout=1.0)

    result = metrics.finalize(time.monotonic_ns())
    assert result["summary"]["async_accepted_requests"] == 1
    assert result["summary"]["async_failed_requests"] == 1
    assert result["summary"]["async_outstanding_requests"] == 0
    assert result["details"]["queue"]["inflight_min"] >= 0
    assert result["details"]["counts"]["terminal"] == 1
    assert engine.requests.empty()
    assert engine.requests.unfinished_tasks == 0
    assert_slots_fully_released(engine, config.queue_capacity)


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

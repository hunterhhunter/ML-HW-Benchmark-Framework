import gc
import math
import threading
import time
import weakref
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real

import numpy as np
import pytest

import core.runtime_executor as runtime_executor_module
from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import (
    AsyncInferenceConfig,
    InferenceRequest,
    TerminalStatus,
)
from core.inference_pipeline import InferencePipeline
from core.runtime_executor import (
    NativeAsyncOutcome,
    NativeAsyncRuntimeExecutor,
    RuntimeExecution,
)


class DeviceSubmitError(RuntimeError):
    pass


class FakeNativeBackend:
    def __init__(
        self,
        *,
        inline_outcome=None,
        submit_error=None,
        raise_after_callback=None,
    ):
        self.inline_outcome = inline_outcome
        self.submit_error = submit_error
        self.raise_after_callback = raise_after_callback
        self.condition = threading.Condition()
        self.jobs = {}
        self.submitted = []
        self.next_job = 1

    def submit_async(self, inputs, callback):
        with self.condition:
            if self.submit_error is not None:
                raise self.submit_error
            job_id = f"job-{self.next_job}"
            self.next_job += 1
            self.jobs[job_id] = (inputs, callback)
            self.submitted.append(job_id)
            self.condition.notify_all()
        if self.inline_outcome is not None:
            outcome = (
                self.inline_outcome(inputs)
                if callable(self.inline_outcome)
                else self.inline_outcome
            )
            callback(outcome)
        if self.raise_after_callback is not None:
            raise self.raise_after_callback
        return job_id

    def wait_for_jobs(self, count, timeout=1.0):
        with self.condition:
            assert self.condition.wait_for(
                lambda: len(self.submitted) >= count,
                timeout=timeout,
            )
            return tuple(self.submitted)

    def complete(self, job_id, outcome):
        with self.condition:
            _, callback = self.jobs[job_id]
        callback(outcome)

    def inputs_for(self, job_id):
        with self.condition:
            return self.jobs[job_id][0]

    def release(self, job_id):
        with self.condition:
            self.jobs.pop(job_id, None)


class GatedSubmitReturnBackend:
    def __init__(self, *, callback_before_gate):
        self.callback_before_gate = callback_before_gate
        self.entered = threading.Event()
        self.release = threading.Event()

    def submit_async(self, inputs, callback):
        outcome = NativeAsyncOutcome(
            outputs={"output": np.array(inputs["input"], copy=True)},
            timing_ms=1.0,
        )
        if self.callback_before_gate:
            callback(outcome)
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        if not self.callback_before_gate:
            callback(outcome)
        return "gated-job"


class StatelessInlineBackend:
    def __init__(self):
        self.submissions = 0

    def submit_async(self, inputs, callback):
        self.submissions += 1
        callback(
            NativeAsyncOutcome(
                outputs={"output": np.array(inputs["input"], copy=True)},
                timing_ms=1.0,
            )
        )
        return self.submissions


class VendorIdBackend:
    def __init__(self, vendor_job_id):
        self.vendor_job_id = vendor_job_id

    def submit_async(self, inputs, callback):
        callback(
            NativeAsyncOutcome(
                outputs={"output": np.array(inputs["input"], copy=True)},
                timing_ms=1.0,
            )
        )
        return self.vendor_job_id


class RetainingString(str):
    def __getitem__(self, key):
        del key
        return self


class LyingLongString(str):
    def __len__(self):
        return 1


class SecretFloat(float):
    def __new__(cls, value, secret):
        instance = super().__new__(cls, value)
        instance.secret = secret
        return instance


class RaisingInt(int):
    def __int__(self):
        raise RuntimeError("vendor integer conversion must not run")


class RaisingMaxInflightInt(int):
    def __int__(self):
        raise RuntimeError("max_inflight conversion must be normalized")


class RaisingRegisteredIntegral:
    def __int__(self):
        raise RuntimeError("registered integral conversion must be normalized")

    def __le__(self, other):
        return False


class OneShotIntegral:
    def __init__(self, value):
        self.value = value
        self.conversions = 0

    def __int__(self):
        self.conversions += 1
        if self.conversions > 1:
            raise RuntimeError("max_inflight was converted more than once")
        return self.value

    def __le__(self, other):
        return int(self) <= other


Integral.register(RaisingRegisteredIntegral)
Integral.register(OneShotIntegral)


class RaisingFloat(float):
    def __float__(self):
        raise RuntimeError("timeout conversion must be normalized")


class RaisingReal:
    def __float__(self):
        raise RuntimeError("real timeout conversion must be normalized")


Real.register(RaisingReal)


class OneShotFloat(float):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.conversions = 0
        return instance

    def __float__(self):
        self.conversions += 1
        if self.conversions > 1:
            raise RuntimeError("timeout was converted more than once")
        return float.__float__(self)


class LyingLargeMapping(Mapping):
    def __init__(self, count=33):
        self.data = {
            f"metric-{index}": float(index)
            for index in range(count)
        }

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return 0

    def items(self):
        return self.data.items()


class GatedTimingMapping(Mapping):
    def __init__(self):
        self.data = {"total_ms": 1.0}
        self.entered = threading.Event()
        self.release = threading.Event()

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def items(self):
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        return self.data.items()


class WeakTimingMapping(dict):
    pass


def traceback_bearing_error():
    try:
        raise RuntimeError("timing payload must not retain this traceback")
    except RuntimeError as error:
        return error


class NativeRuntimeCapabilities:
    compiled_model = None

    def max_concurrent_workers(self):
        return 2

    def max_dynamic_batch_size(self):
        return 1

    def supports_dynamic_batching(self):
        return False

    def supports_batch_generation(self):
        return False

    def supports_generate(self):
        return False


class NativeLoader:
    def get_metadata(self):
        return {"is_static_batched": False, "total_samples": 8}


class RecordingEvaluator:
    def __init__(self):
        self.lock = threading.Lock()
        self.pairs = []

    def add_batch(self, outputs, labels, timing_ms):
        del timing_ms
        predictions = np.asarray(outputs["output"]).reshape(-1)
        expected = np.asarray(labels).reshape(-1)
        with self.lock:
            self.pairs.extend(
                (float(prediction), float(label))
                for prediction, label in zip(predictions, expected)
            )


class TraceRecorder:
    def __init__(self):
        self.condition = threading.Condition()
        self.traces = []

    def __call__(self, trace):
        with self.condition:
            self.traces.append(trace)
            self.condition.notify_all()

    def wait_for(self, count, timeout=2.0):
        with self.condition:
            assert self.condition.wait_for(
                lambda: len(self.traces) >= count,
                timeout=timeout,
            )
            return tuple(self.traces)


class RecordingPermit:
    def __init__(self):
        self.timeouts = []

    def acquire(self, *, timeout):
        self.timeouts.append(timeout)
        return False


class CountingBoundedPermit:
    def __init__(self, value):
        self.semaphore = threading.BoundedSemaphore(value)
        self.releases = 0

    def acquire(self, *, timeout):
        return self.semaphore.acquire(timeout=timeout)

    def release(self):
        self.releases += 1
        self.semaphore.release()


@pytest.fixture(autouse=True)
def no_native_test_async_thread_leaks():
    before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("async-")
    }
    yield
    leaked = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith("async-") and thread.ident not in before
    ]
    assert leaked == []


def make_request(request_id):
    now = time.monotonic_ns()
    return InferenceRequest(
        request_id=request_id,
        sample_index=request_id,
        sample={
            "input": np.asarray([request_id], dtype=np.float32),
            "label": np.asarray(request_id, dtype=np.float32),
        },
        scheduled_ns=now,
        issued_ns=now,
        enqueued_ns=0,
    )


def build_native_engine(
    backend,
    *,
    worker_count=1,
    max_inflight=None,
    completion_timeout_sec=1.0,
):
    config = AsyncInferenceConfig(
        queue_capacity=max(2, worker_count),
        worker_count=worker_count,
        max_batch_size=1,
        batch_timeout_ms=0,
        submit_timeout_sec=1.0,
        flush_timeout_sec=2.0,
        min_samples=1,
    )
    runtime = NativeRuntimeCapabilities()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=max_inflight or worker_count,
        completion_timeout_sec=completion_timeout_sec,
    )
    pipeline = InferencePipeline(
        NativeLoader(),
        runtime,
        runtime_executor=executor,
    )
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count)
    evaluator = RecordingEvaluator()
    traces = TraceRecorder()
    coordinator = CompletionCoordinator(
        pipeline,
        evaluator,
        None,
        metrics,
        queue_capacity=worker_count,
        trace_callback=traces,
    )
    engine = AsyncInferenceEngine(
        runtime,
        pipeline,
        config,
        coordinator,
        metrics,
        executor=executor,
    )
    return engine, executor, evaluator, metrics, traces


def assert_accounting(metrics, *, completed, failed, rejected=0):
    result = metrics.finalize(time.monotonic_ns())
    summary = result["summary"]
    details = result["details"]
    submitted = completed + failed + rejected
    accepted = completed + failed
    assert summary["async_submitted_requests"] == submitted
    assert summary["async_accepted_requests"] == accepted
    assert summary["async_completed_requests"] == completed
    assert summary["async_failed_requests"] == failed
    assert summary["async_rejected_requests"] == rejected
    assert summary["async_outstanding_requests"] == 0
    assert details["counter_invariants"][
        "submitted_equals_accepted_plus_rejected"
    ] is True
    assert details["counter_invariants"][
        "accepted_equals_terminal_plus_outstanding"
    ] is True
    return result


def test_native_executor_accepts_inline_callback_before_vendor_id_return():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.array([[7]])}, timing_ms=1.0
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )

    execution = executor.execute({"input": np.array([[7]])})

    np.testing.assert_array_equal(execution.outputs["output"], [[7]])
    assert execution.vendor_job_id == "job-1"
    assert execution.dispatch_token is not None
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0


def test_native_executor_matches_out_of_order_callbacks_to_dispatches():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=2, completion_timeout_sec=1.0
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, {"input": np.array([[1]])})
        second = pool.submit(executor.execute, {"input": np.array([[2]])})
        job_1, job_2 = backend.wait_for_jobs(2)
        backend.complete(
            job_2,
            NativeAsyncOutcome(
                outputs={"output": backend.inputs_for(job_2)["input"] * 10},
                timing_ms=2.0,
            ),
        )
        backend.complete(
            job_1,
            NativeAsyncOutcome(
                outputs={"output": backend.inputs_for(job_1)["input"] * 10},
                timing_ms=3.0,
            ),
        )
        execution_1 = first.result(timeout=1.0)
        execution_2 = second.result(timeout=1.0)

    np.testing.assert_array_equal(execution_1.outputs["output"], [[10]])
    np.testing.assert_array_equal(execution_2.outputs["output"], [[20]])
    assert execution_1.dispatch_token != execution_2.dispatch_token
    executor.acknowledge(execution_1)
    executor.acknowledge(execution_2)


def test_native_executor_ignores_duplicate_callback_and_keeps_first_result():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, {"input": np.array([[1]])})
        job_id = backend.wait_for_jobs(1)[0]
        backend.complete(
            job_id,
            NativeAsyncOutcome(outputs={"output": np.array([[1]])}, timing_ms=1.0),
        )
        backend.complete(
            job_id,
            NativeAsyncOutcome(outputs={"output": np.array([[99]])}, timing_ms=2.0),
        )
        execution = future.result(timeout=1.0)

    np.testing.assert_array_equal(execution.outputs["output"], [[1]])
    assert executor.snapshot().duplicate_callbacks == 1
    executor.acknowledge(execution)


def test_native_executor_submit_failure_releases_permit_on_ack():
    backend = FakeNativeBackend(submit_error=DeviceSubmitError("submit failed"))
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )

    failed = executor.execute({"input": np.array([[1]])})
    assert failed.error_type == "DeviceSubmitError"
    assert failed.outputs is None
    assert executor.snapshot().submit_failures == 1
    executor.acknowledge(failed)

    backend.submit_error = None
    backend.inline_outcome = NativeAsyncOutcome(
        outputs={"output": np.array([[2]])}, timing_ms=1.0
    )
    succeeded = executor.execute({"input": np.array([[2]])})
    np.testing.assert_array_equal(succeeded.outputs["output"], [[2]])
    executor.acknowledge(succeeded)


def test_native_executor_timeout_and_late_callback_are_safe():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=0.01
    )

    permits = CountingBoundedPermit(1)
    executor._permits = permits

    execution = executor.execute({"input": np.array([[1]])})
    job_id = backend.wait_for_jobs(1)[0]
    dispatch = executor._dispatches[execution.dispatch_token]
    assert execution.error_type == "NativeAsyncTimeout"
    assert executor.snapshot().timeouts == 1
    assert executor.shutdown(timeout=0.0) is False

    executor.acknowledge(execution)
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 1
    assert executor.shutdown(timeout=0.0) is False
    assert permits.releases == 0

    backend.complete(
        job_id,
        NativeAsyncOutcome(outputs={"output": np.array([[99]])}, timing_ms=2.0),
    )

    snapshot = executor.snapshot()
    assert snapshot.inflight == 0
    assert snapshot.late_callbacks == 1
    assert dispatch.inputs is None
    assert dispatch.outcome is None
    assert permits.releases == 1
    executor.acknowledge(execution)
    assert permits.releases == 1
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_acknowledge_is_idempotent_but_unknown_token_is_error():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(outputs={"output": np.array([[1]])})
    )
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )
    execution = executor.execute({"input": np.array([[1]])})

    unknown = type(execution)(
        outputs=None,
        timing_ms=None,
        dispatch_token=9999,
    )
    with pytest.raises(
        RuntimeError,
        match="^unknown native async dispatch token$",
    ):
        executor.acknowledge(unknown)
    blocked = executor.execute({"input": np.array([[2]])}, timeout=0.0)
    assert blocked.error_type == "NativeAsyncBackpressureTimeout"
    executor.acknowledge(blocked)

    executor.acknowledge(execution)
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    "unknown_token",
    [10**100_000, LyingLongString("secret" * 2_000)],
    ids=("huge-int", "adversarial-string"),
)
def test_native_executor_unknown_ack_error_is_fixed_and_bounded(unknown_token):
    executor = NativeAsyncRuntimeExecutor(
        StatelessInlineBackend(),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError) as raised:
        executor.acknowledge(
            RuntimeExecution(
                outputs=None,
                timing_ms=None,
                dispatch_token=unknown_token,
            )
        )

    assert str(raised.value) == "unknown native async dispatch token"
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_ack_history_does_not_grow_with_completed_dispatches():
    backend = StatelessInlineBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    baseline_entries = sum(
        len(value)
        for value in vars(executor).values()
        if isinstance(value, (dict, list, set))
    )

    for value in range(10_000):
        execution = executor.execute({"input": np.asarray([[value]])})
        executor.acknowledge(execution)
        executor.acknowledge(execution)

    retained_entries = sum(
        len(value)
        for value in vars(executor).values()
        if isinstance(value, (dict, list, set))
    )
    assert retained_entries == baseline_entries
    assert executor.snapshot().inflight == 0
    for unknown_token in (0, 10_001):
        with pytest.raises(
            RuntimeError,
            match="^unknown native async dispatch token$",
        ):
            executor.acknowledge(
                RuntimeExecution(
                    outputs=None,
                    timing_ms=None,
                    dispatch_token=unknown_token,
                )
            )
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    ("max_inflight", "completion_timeout_sec"),
    [
        (True, 1.0),
        (1.5, 1.0),
        ("1", 1.0),
        (0, 1.0),
        (1, True),
        (1, 0.0),
        (1, math.inf),
        (1, math.nan),
        (1, "1.0"),
    ],
)
def test_native_executor_rejects_malformed_constructor_settings(
    max_inflight,
    completion_timeout_sec,
):
    with pytest.raises(ValueError):
        NativeAsyncRuntimeExecutor(
            FakeNativeBackend(),
            max_inflight=max_inflight,
            completion_timeout_sec=completion_timeout_sec,
        )


@pytest.mark.parametrize(
    "max_inflight",
    [RaisingMaxInflightInt(1), RaisingRegisteredIntegral()],
    ids=("int-subclass", "registered-integral"),
)
def test_native_executor_normalizes_adversarial_max_inflight_conversion(
    max_inflight,
):
    with pytest.raises(ValueError, match="max_inflight"):
        NativeAsyncRuntimeExecutor(
            FakeNativeBackend(),
            max_inflight=max_inflight,
            completion_timeout_sec=1.0,
        )


def test_native_executor_converts_max_inflight_exactly_once():
    max_inflight = OneShotIntegral(2)

    executor = NativeAsyncRuntimeExecutor(
        FakeNativeBackend(),
        max_inflight=max_inflight,
        completion_timeout_sec=1.0,
    )

    assert executor.max_inflight == 2
    assert max_inflight.conversions == 1
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_accepts_numpy_integer_max_inflight():
    executor = NativeAsyncRuntimeExecutor(
        FakeNativeBackend(),
        max_inflight=np.int64(2),
        completion_timeout_sec=1.0,
    )

    assert type(executor.max_inflight) is int
    assert executor.max_inflight == 2
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_rejects_constructor_timeout_max_overflow():
    too_large = threading.TIMEOUT_MAX + 1.0
    with pytest.raises(ValueError, match="completion_timeout_sec"):
        NativeAsyncRuntimeExecutor(
            FakeNativeBackend(),
            max_inflight=1,
            completion_timeout_sec=too_large,
        )


def test_native_executor_rejects_execute_timeout_max_before_dispatch():
    too_large = threading.TIMEOUT_MAX + 1.0
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    with pytest.raises(ValueError, match="timeout"):
        executor.execute({"input": np.asarray([[1]])}, timeout=too_large)
    assert backend.submitted == []
    assert executor.snapshot().inflight == 0

    backend.inline_outcome = NativeAsyncOutcome(
        outputs={"output": np.asarray([[2]])},
        timing_ms=1.0,
    )
    execution = executor.execute({"input": np.asarray([[2]])})
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_rejects_shutdown_timeout_max_without_closing():
    too_large = threading.TIMEOUT_MAX + 1.0
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[3]])},
            timing_ms=1.0,
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with pytest.raises(ValueError, match="timeout"):
        executor.shutdown(timeout=too_large)

    execution = executor.execute({"input": np.asarray([[3]])})
    assert execution.error_type is None
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    "bad_timeout",
    [RaisingFloat(1.0), RaisingReal(), 10**10_000],
    ids=("float-subclass", "registered-real", "huge-int"),
)
@pytest.mark.parametrize("boundary", ["constructor", "execute", "shutdown"])
def test_native_executor_normalizes_adversarial_timeout_conversion(
    bad_timeout,
    boundary,
):
    if boundary == "constructor":
        with pytest.raises(ValueError, match="completion_timeout_sec"):
            NativeAsyncRuntimeExecutor(
                FakeNativeBackend(),
                max_inflight=1,
                completion_timeout_sec=bad_timeout,
            )
        return

    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    with pytest.raises(ValueError, match="timeout"):
        if boundary == "execute":
            executor.execute(
                {"input": np.asarray([[1]])},
                timeout=bad_timeout,
            )
        else:
            executor.shutdown(timeout=bad_timeout)
    assert backend.submitted == []
    assert executor.snapshot().inflight == 0

    backend.inline_outcome = NativeAsyncOutcome(
        outputs={"output": np.asarray([[1]])},
        timing_ms=1.0,
    )
    execution = executor.execute({"input": np.asarray([[1]])})
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_converts_timeout_exactly_once():
    timeout = OneShotFloat(1.0)

    executor = NativeAsyncRuntimeExecutor(
        FakeNativeBackend(),
        max_inflight=1,
        completion_timeout_sec=timeout,
    )

    assert executor.completion_timeout_sec == 1.0
    assert timeout.conversions == 1
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_callback_wins_when_submit_raises_after_callback():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[11]], dtype=np.int64)},
            timing_ms=1.0,
        ),
        raise_after_callback=DeviceSubmitError("raise after callback"),
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute(
        {"input": np.asarray([[11]], dtype=np.int64)}
    )

    np.testing.assert_array_equal(execution.outputs["output"], [[11]])
    assert execution.error_type is None
    assert executor.snapshot().submit_failures == 0
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_callback_after_deadline_is_timeout_and_late():
    backend = GatedSubmitReturnBackend(callback_before_gate=False)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.asarray([[4]], dtype=np.float32)},
            0.01,
        )
        assert backend.entered.wait(timeout=1.0)
        assert threading.Event().wait(timeout=0.03) is False
        backend.release.set()
        execution = future.result(timeout=1.0)

    assert execution.outputs is None
    assert execution.error_type == "NativeAsyncTimeout"
    snapshot = executor.snapshot()
    assert snapshot.timeouts == 1
    assert snapshot.late_callbacks == 1
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_callback_observation_is_after_payload_validation():
    timing = GatedTimingMapping()
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[4]])},
            timing_ms=timing,
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.asarray([[4]])},
            0.01,
        )
        assert timing.entered.wait(timeout=1.0)
        assert threading.Event().wait(timeout=0.03) is False
        timing.release.set()
        execution = future.result(timeout=1.0)

    assert execution.outputs is None
    assert execution.error_type == "NativeAsyncTimeout"
    snapshot = executor.snapshot()
    assert snapshot.timeouts == 1
    assert snapshot.late_callbacks == 1
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_ack_waits_for_callback_normalization_to_finish():
    timing = GatedTimingMapping()
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=0.01,
    )
    permits = CountingBoundedPermit(1)
    executor._permits = permits

    with ThreadPoolExecutor(max_workers=2) as pool:
        execution_future = pool.submit(
            executor.execute,
            {"input": np.asarray([[4]])},
        )
        job_id = backend.wait_for_jobs(1)[0]
        callback_future = pool.submit(
            backend.complete,
            job_id,
            NativeAsyncOutcome(
                outputs={"output": np.asarray([[4]])},
                timing_ms=timing,
            ),
        )
        assert timing.entered.wait(timeout=1.0)
        execution = execution_future.result(timeout=1.0)

        assert execution.error_type == "NativeAsyncTimeout"
        executor.acknowledge(execution)
        assert executor.snapshot().inflight == 1
        assert executor.shutdown(timeout=0.0) is False
        assert permits.releases == 0

        timing.release.set()
        callback_future.result(timeout=1.0)

    snapshot = executor.snapshot()
    assert snapshot.inflight == 0
    assert snapshot.late_callbacks == 1
    assert permits.releases == 1
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_permit_wait_uses_remaining_single_deadline(monkeypatch):
    executor = NativeAsyncRuntimeExecutor(
        FakeNativeBackend(),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    permit = RecordingPermit()
    executor._permits = permit
    clock_values = iter((100.0, 100.004))
    monkeypatch.setattr(
        runtime_executor_module.time,
        "monotonic",
        lambda: next(clock_values),
    )

    execution = executor.execute({"input": np.asarray([[1]])}, timeout=0.01)

    assert execution.error_type == "NativeAsyncBackpressureTimeout"
    assert permit.timeouts == [pytest.approx(0.006)]
    assert executor.snapshot().inflight == 0


def test_native_executor_callback_before_deadline_survives_late_submit_return():
    backend = GatedSubmitReturnBackend(callback_before_gate=True)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.asarray([[5]], dtype=np.float32)},
            0.01,
        )
        assert backend.entered.wait(timeout=1.0)
        assert threading.Event().wait(timeout=0.03) is False
        backend.release.set()
        execution = future.result(timeout=1.0)

    np.testing.assert_array_equal(execution.outputs["output"], [[5]])
    assert execution.error_type is None
    snapshot = executor.snapshot()
    assert snapshot.timeouts == 0
    assert snapshot.late_callbacks == 0
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_invalid_callback_payload_is_first_terminal_failure():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            executor.execute,
            {"input": np.asarray([[3]], dtype=np.int64)},
        )
        job_id = backend.wait_for_jobs(1)[0]
        backend.complete(job_id, {"output": np.asarray([[3]])})
        backend.complete(
            job_id,
            NativeAsyncOutcome(
                outputs={"output": np.asarray([[99]])},
                timing_ms=2.0,
            ),
        )
        execution = future.result(timeout=1.0)

    assert execution.outputs is None
    assert execution.error_type == "NativeAsyncProtocolError"
    assert executor.snapshot().duplicate_callbacks == 1
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    "timing_ms",
    [
        True,
        -1.0,
        math.inf,
        math.nan,
        np.asarray([1.0]),
        traceback_bearing_error(),
        {"total_ms": -1.0},
        {"total_ms": math.inf},
        {"nested": {"total_ms": 1.0}},
        {"array": np.asarray([1.0])},
        {"error": traceback_bearing_error()},
        {f"metric-{index}": float(index) for index in range(33)},
        {"k" * 129: 1.0},
        {"timing_source": "s" * 513},
    ],
)
def test_native_executor_rejects_untrusted_timing_payloads(timing_ms):
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[1]])},
            timing_ms=timing_ms,
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.outputs is None
    assert execution.timing_ms is None
    assert execution.error_type == "NativeAsyncProtocolError"
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_copies_flat_llm_timing_without_retaining_payload():
    timing = WeakTimingMapping(
        total_ms=4.0,
        ttft_ms=1.0,
        tpot_ms=3.0,
        timing_mode="native",
        uses_kv_cache=True,
        timing_source="measured",
        optional_value=None,
    )
    outcome = NativeAsyncOutcome(
        outputs={"output": np.asarray([[8]])},
        timing_ms=timing,
        generated_tokens=np.int64(2),
    )
    timing_ref = weakref.ref(timing)
    outcome_ref = weakref.ref(outcome)
    backend = FakeNativeBackend(inline_outcome=outcome)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[8]])})
    backend.inline_outcome = None
    del outcome
    del timing
    gc.collect()

    assert outcome_ref() is None
    assert timing_ref() is None
    assert execution.timing_ms == {
        "total_ms": 4.0,
        "ttft_ms": 1.0,
        "tpot_ms": 3.0,
        "timing_mode": "native",
        "uses_kv_cache": True,
        "timing_source": "measured",
        "optional_value": None,
    }
    assert type(execution.timing_ms) is dict
    assert execution.generated_tokens == 2
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_rejects_mapping_that_lies_about_length():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[1]])},
            timing_ms=LyingLargeMapping(),
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.outputs is None
    assert execution.error_type == "NativeAsyncProtocolError"
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize("payload_location", ["error", "timing"])
def test_native_executor_rejects_lying_long_string_subclasses(payload_location):
    secret = LyingLongString("secret" * 2_000)
    outcome = NativeAsyncOutcome(
        outputs={"output": np.asarray([[1]])},
        timing_ms=(
            {"timing_source": secret}
            if payload_location == "timing"
            else 1.0
        ),
        error_type=("DeviceError" if payload_location == "error" else None),
        error_message=(secret if payload_location == "error" else None),
    )
    backend = FakeNativeBackend(inline_outcome=outcome)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.outputs is None
    assert execution.timing_ms is None
    assert execution.error_type == "NativeAsyncProtocolError"
    assert "secret" not in execution.error_message
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_accepts_nonnegative_scalar_timing_copy():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(
            outputs={"output": np.asarray([[6]])},
            timing_ms=np.float32(1.25),
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[6]])})

    assert execution.timing_ms == pytest.approx(1.25)
    assert type(execution.timing_ms) is float
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_retains_input_and_permit_until_acknowledgement():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    input_array = np.asarray([[5]], dtype=np.float32)
    input_ref = weakref.ref(input_array)
    output_array = np.asarray([[50]], dtype=np.float32)
    output_ref = weakref.ref(output_array)
    inputs = {"input": input_array}

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(executor.execute, inputs)
        job_id = backend.wait_for_jobs(1)[0]
        backend.complete(
            job_id,
            NativeAsyncOutcome(
                outputs={"output": output_array},
                timing_ms=1.0,
            ),
        )
        execution = future.result(timeout=1.0)

    token = execution.dispatch_token
    backend.release(job_id)
    del input_array
    del output_array
    del inputs
    del execution
    del future
    gc.collect()
    assert input_ref() is not None
    assert output_ref() is not None
    assert executor.snapshot().inflight == 1
    assert executor.shutdown(timeout=0.0) is False

    executor.acknowledge(
        RuntimeExecution(
            outputs=None,
            timing_ms=None,
            dispatch_token=token,
        )
    )
    gc.collect()
    assert input_ref() is None
    assert output_ref() is None
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_bounded_slot_timeout_is_acknowledgeable_noop():
    backend = FakeNativeBackend(
        inline_outcome=lambda inputs: NativeAsyncOutcome(
            outputs={"output": np.array(inputs["input"], copy=True)},
            timing_ms=1.0,
        )
    )
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    held = executor.execute({"input": np.asarray([[1]])})

    blocked = executor.execute(
        {"input": np.asarray([[2]])},
        timeout=0.0,
    )

    assert blocked.error_type == "NativeAsyncBackpressureTimeout"
    assert blocked.dispatch_token is None
    executor.acknowledge(blocked)
    assert executor.snapshot().inflight == 1
    executor.acknowledge(held)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_shutdown_before_submit_returns_normalized_failures():
    backend = FakeNativeBackend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    assert executor.shutdown(timeout=0.0) is True

    for value in (1, 2):
        execution = executor.execute({"input": np.asarray([[value]])})
        assert execution.error_type == "NativeAsyncShutdown"
        assert execution.dispatch_token is None
        executor.acknowledge(execution)

    assert backend.submitted == []
    assert executor.snapshot().inflight == 0


def test_native_executor_submit_failure_diagnostics_are_sanitized():
    primary = DeviceSubmitError("secret tensor address 0xdeadbeef")
    backend = FakeNativeBackend(submit_error=primary)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.error_type == "DeviceSubmitError"
    assert execution.error_message == "native async submission failed"
    assert "secret" not in execution.error_message
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_summarizes_huge_integer_vendor_id():
    huge_vendor_id = 10**100_000
    executor = NativeAsyncRuntimeExecutor(
        VendorIdBackend(huge_vendor_id),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert type(execution.vendor_job_id) is str
    assert len(execution.vendor_job_id) <= 512
    assert "bits" in execution.vendor_job_id
    assert execution.vendor_job_id != huge_vendor_id
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    "vendor_job_id",
    ["v" * 10_000],
)
def test_native_executor_bounds_every_vendor_id_string(vendor_job_id):
    executor = NativeAsyncRuntimeExecutor(
        VendorIdBackend(vendor_job_id),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert type(execution.vendor_job_id) is str
    assert len(execution.vendor_job_id) == 512
    executor.acknowledge(execution)
    assert executor.shutdown(timeout=0.0) is True


@pytest.mark.parametrize(
    "vendor_job_id",
    [
        RetainingString("secret" * 2_000),
        SecretFloat(1.5, secret=object()),
    ],
)
def test_native_executor_summarizes_vendor_primitive_subclasses(vendor_job_id):
    executor = NativeAsyncRuntimeExecutor(
        VendorIdBackend(vendor_job_id),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert type(execution.vendor_job_id) is str
    assert len(execution.vendor_job_id) <= 512
    assert type(vendor_job_id).__name__ in execution.vendor_job_id
    assert "secret" not in execution.vendor_job_id
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_vendor_sanitizer_error_cannot_leak_dispatch():
    executor = NativeAsyncRuntimeExecutor(
        VendorIdBackend(RaisingInt(7)),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.error_type is None
    assert type(execution.vendor_job_id) is str
    assert "RaisingInt" in execution.vendor_job_id
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_bounds_submit_exception_type_name():
    long_error_type = type("E" * 10_000, (RuntimeError,), {})
    executor = NativeAsyncRuntimeExecutor(
        FakeNativeBackend(submit_error=long_error_type("submit failed")),
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    execution = executor.execute({"input": np.asarray([[1]])})

    assert execution.outputs is None
    assert type(execution.error_type) is str
    assert len(execution.error_type) == 256
    assert execution.error_message == "native async submission failed"
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_real_queue_preserves_reverse_completion_identity():
    backend = FakeNativeBackend()
    engine, executor, evaluator, metrics, traces = build_native_engine(
        backend,
        worker_count=2,
    )
    engine.start()
    assert engine.submit(make_request(0), block=True) is True
    assert engine.submit(make_request(1), block=True) is True
    job_1, job_2 = backend.wait_for_jobs(2)

    for job_id in (job_2, job_1):
        runtime_inputs = backend.inputs_for(job_id)
        backend.complete(
            job_id,
            NativeAsyncOutcome(
                outputs={"output": runtime_inputs["input"] * 10},
                timing_ms=1.0,
            ),
        )

    observed = traces.wait_for(2)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert sorted(evaluator.pairs) == [(0.0, 0.0), (10.0, 1.0)]
    assert sorted(trace.request_id for trace in observed) == [0, 1]
    assert all(trace.status is TerminalStatus.COMPLETED for trace in observed)
    assert_accounting(metrics, completed=2, failed=0)
    assert executor.snapshot().inflight == 0


def test_native_executor_real_queue_duplicate_is_exactly_once():
    backend = FakeNativeBackend()
    engine, executor, evaluator, metrics, traces = build_native_engine(backend)
    engine.start()
    assert engine.submit(make_request(1), block=True) is True
    job_id = backend.wait_for_jobs(1)[0]
    backend.complete(
        job_id,
        NativeAsyncOutcome(
            outputs={"output": np.asarray([[10]], dtype=np.float32)},
            timing_ms=1.0,
        ),
    )
    backend.complete(
        job_id,
        NativeAsyncOutcome(
            outputs={"output": np.asarray([[99]], dtype=np.float32)},
            timing_ms=2.0,
        ),
    )

    observed = traces.wait_for(1)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert evaluator.pairs == [(10.0, 1.0)]
    assert [trace.request_id for trace in observed] == [1]
    assert observed[0].status is TerminalStatus.COMPLETED
    assert executor.snapshot().duplicate_callbacks == 1
    assert executor.snapshot().inflight == 0
    assert_accounting(metrics, completed=1, failed=0)


def test_native_executor_real_queue_timeout_late_callback_is_exactly_once():
    backend = FakeNativeBackend()
    engine, executor, evaluator, metrics, traces = build_native_engine(
        backend,
        completion_timeout_sec=0.02,
    )
    engine.start()
    assert engine.submit(make_request(2), block=True) is True
    job_id = backend.wait_for_jobs(1)[0]

    observed = traces.wait_for(1)
    backend.complete(
        job_id,
        NativeAsyncOutcome(
            outputs={"output": np.asarray([[20]], dtype=np.float32)},
            timing_ms=1.0,
        ),
    )
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert evaluator.pairs == []
    assert [trace.request_id for trace in observed] == [2]
    assert observed[0].status is TerminalStatus.FAILED
    assert observed[0].error_type == "NativeAsyncTimeout"
    snapshot = executor.snapshot()
    assert snapshot.timeouts == 1
    assert snapshot.late_callbacks == 1
    assert snapshot.inflight == 0
    assert_accounting(metrics, completed=0, failed=1)


def test_native_executor_real_queue_submit_failure_is_exactly_once():
    backend = FakeNativeBackend(
        submit_error=DeviceSubmitError("device unavailable")
    )
    engine, executor, evaluator, metrics, traces = build_native_engine(backend)
    engine.start()
    assert engine.submit(make_request(3), block=True) is True

    observed = traces.wait_for(1)
    engine.close_submission()
    assert engine.flush() is True
    assert engine.shutdown() is True

    assert evaluator.pairs == []
    assert [trace.request_id for trace in observed] == [3]
    assert observed[0].status is TerminalStatus.FAILED
    assert observed[0].error_type == "DeviceSubmitError"
    snapshot = executor.snapshot()
    assert snapshot.submit_failures == 1
    assert snapshot.inflight == 0
    assert_accounting(metrics, completed=0, failed=1)

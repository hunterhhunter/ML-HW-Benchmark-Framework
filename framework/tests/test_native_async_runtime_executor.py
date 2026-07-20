import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from core.runtime_executor import (
    NativeAsyncOutcome,
    NativeAsyncRuntimeExecutor,
)


class DeviceSubmitError(RuntimeError):
    pass


class FakeNativeBackend:
    def __init__(self, *, inline_outcome=None, submit_error=None):
        self.inline_outcome = inline_outcome
        self.submit_error = submit_error
        self.condition = threading.Condition()
        self.jobs = {}
        self.submitted = []
        self.next_job = 1

    def submit_async(self, inputs, callback):
        if self.submit_error is not None:
            raise self.submit_error
        with self.condition:
            job_id = f"job-{self.next_job}"
            self.next_job += 1
            self.jobs[job_id] = (inputs, callback)
            self.submitted.append(job_id)
            self.condition.notify_all()
        if self.inline_outcome is not None:
            callback(self.inline_outcome)
        return job_id

    def wait_for_jobs(self, count, timeout=1.0):
        with self.condition:
            assert self.condition.wait_for(
                lambda: len(self.submitted) >= count,
                timeout=timeout,
            )
            return tuple(self.submitted)

    def complete(self, job_id, outcome):
        _, callback = self.jobs[job_id]
        callback(outcome)


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
            NativeAsyncOutcome(outputs={"output": np.array([[20]])}, timing_ms=2.0),
        )
        backend.complete(
            job_1,
            NativeAsyncOutcome(outputs={"output": np.array([[10]])}, timing_ms=3.0),
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

    execution = executor.execute({"input": np.array([[1]])})
    job_id = backend.wait_for_jobs(1)[0]
    assert execution.error_type == "NativeAsyncTimeout"
    assert executor.snapshot().timeouts == 1
    assert executor.shutdown(timeout=0.0) is False

    executor.acknowledge(execution)
    backend.complete(
        job_id,
        NativeAsyncOutcome(outputs={"output": np.array([[99]])}, timing_ms=2.0),
    )

    snapshot = executor.snapshot()
    assert snapshot.inflight == 0
    assert snapshot.late_callbacks == 1
    assert executor.shutdown(timeout=0.0) is True


def test_native_executor_acknowledge_is_idempotent_but_unknown_token_is_error():
    backend = FakeNativeBackend(
        inline_outcome=NativeAsyncOutcome(outputs={"output": np.array([[1]])})
    )
    executor = NativeAsyncRuntimeExecutor(
        backend, max_inflight=1, completion_timeout_sec=1.0
    )
    execution = executor.execute({"input": np.array([[1]])})

    executor.acknowledge(execution)
    executor.acknowledge(execution)

    unknown = type(execution)(
        outputs=None,
        timing_ms=None,
        dispatch_token=9999,
    )
    try:
        executor.acknowledge(unknown)
    except RuntimeError as exc:
        assert "9999" in str(exc)
    else:
        raise AssertionError("unknown dispatch token must fail")

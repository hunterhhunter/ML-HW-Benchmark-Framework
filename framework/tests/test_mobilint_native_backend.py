import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import runtimes.mobilint_rt as mobilint_rt_module
from core.runtime_executor import NativeAsyncRuntimeExecutor
from runtimes.mobilint_rt import MobilintNativeBackend, MobilintRuntime


class FakeFuture:
    def __init__(self, outputs=None, error=None, release=None):
        self.outputs = outputs
        self.error = error
        self.release = release
        self.get_calls = 0

    def get(self):
        self.get_calls += 1
        if self.release is not None:
            self.release.wait(timeout=2.0)
        if self.error is not None:
            raise self.error
        return self.outputs


class FakeModel:
    def __init__(self, futures):
        self.futures = list(futures)
        self.calls = []

    def infer_async(self, inputs):
        self.calls.append(inputs)
        future = self.futures.pop(0)
        if isinstance(future, BaseException):
            raise future
        return future


def _runtime(model, slots=2):
    def ordered(inputs):
        return [
            np.ascontiguousarray(inputs["first"]),
            np.ascontiguousarray(inputs["second"]),
        ]

    def normalize(outputs, *, expected_batch_size=None):
        assert expected_batch_size == 1
        if outputs is None or len(outputs) != 1:
            raise RuntimeError("invalid output count")
        return {"output": np.asarray(outputs[0])}

    return SimpleNamespace(
        _model=model,
        async_pipeline_enabled=True,
        max_concurrent_workers=lambda: slots,
        _ordered_inputs=ordered,
        _normalize_outputs=normalize,
    )


def _inputs(value=1):
    return {
        "second": np.full((1, 2), value + 1, dtype=np.float32),
        "first": np.full((1, 2), value, dtype=np.float32),
    }


def test_future_get_and_callback_each_happen_once():
    future = FakeFuture([np.array([[7.0]], dtype=np.float32)])
    model = FakeModel([future])
    backend = MobilintNativeBackend(_runtime(model))
    completed = []
    done = threading.Event()

    job_id = backend.submit_async(
        _inputs(),
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert job_id.startswith("mobilint-")
    assert done.wait(timeout=1.0)
    assert future.get_calls == 1
    assert len(completed) == 1
    np.testing.assert_array_equal(completed[0].outputs["output"], [[7.0]])
    np.testing.assert_array_equal(model.calls[0][0], [[1.0, 1.0]])
    np.testing.assert_array_equal(model.calls[0][1], [[2.0, 2.0]])
    assert backend.shutdown(timeout=1.0) is True


def test_jobs_may_complete_out_of_order_without_duplicate_callbacks():
    first_release = threading.Event()
    first = FakeFuture([np.array([[1]])], release=first_release)
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(_runtime(FakeModel([first, second])))
    completed = []
    done = threading.Event()

    first_job_id = backend.submit_async(
        _inputs(1), lambda outcome: completed.append(1)
    )
    second_job_id = backend.submit_async(
        _inputs(2),
        lambda outcome: (completed.append(2), done.set()),
    )
    assert done.wait(timeout=1.0)
    first_release.set()
    assert backend.shutdown(timeout=1.0) is True

    assert completed == [2, 1]
    assert (first_job_id, second_job_id) == ("mobilint-1", "mobilint-2")
    assert first.get_calls == 1
    assert second.get_calls == 1


def test_sdk_error_is_sanitized_before_one_callback():
    future = FakeFuture(error=RuntimeError("secret tensor values 12345"))
    backend = MobilintNativeBackend(_runtime(FakeModel([future])))
    completed = []
    done = threading.Event()

    backend.submit_async(
        _inputs(),
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    assert completed[0].error_type == "RuntimeError"
    assert completed[0].error_message == "Mobilint asynchronous inference failed."
    assert "secret" not in completed[0].error_message
    assert future.get_calls == 1
    assert backend.shutdown(timeout=1.0) is True


def test_future_outputs_use_runtime_yolo_shape_contract():
    future = FakeFuture(
        [
            np.empty((1, 20, 20, 255), dtype=np.float32),
            np.empty((1, 40, 40, 255), dtype=np.float32),
            np.empty((1, 81, 80, 255), dtype=np.float32),
        ]
    )
    model = FakeModel([future])
    runtime = MobilintRuntime(
        expected_family="aries",
        async_pipeline_enabled=True,
        vision_profile_id="mobilint-yolov5m-default",
        expected_input_dtype="uint8",
        expected_input_layout="NHWC",
        expected_unbatched_input_shape=[640, 640, 3],
        max_input_batch_size=1,
        expected_unbatched_output_shapes=[
            [20, 20, 255],
            [40, 40, 255],
            [80, 80, 255],
        ],
    )
    runtime._model = model
    runtime._input_names = ("images",)
    runtime._output_names = ("stride32", "stride16", "stride8")
    backend = MobilintNativeBackend(runtime)
    completed = []
    done = threading.Event()

    backend.submit_async(
        {"images": np.zeros((1, 640, 640, 3), dtype=np.uint8)},
        lambda outcome: (completed.append(outcome), done.set()),
    )

    assert done.wait(timeout=1.0)
    assert completed[0].error_type == "RuntimeError"
    assert completed[0].error_message == "Mobilint asynchronous inference failed."
    assert future.get_calls == 1
    assert backend.shutdown(timeout=1.0) is True


def test_synchronous_submission_error_has_no_callback():
    backend = MobilintNativeBackend(
        _runtime(FakeModel([RuntimeError("secret input")]))
    )
    completed = []

    with pytest.raises(RuntimeError, match="secret input"):
        backend.submit_async(_inputs(), completed.append)

    assert completed == []
    assert backend.shutdown(timeout=1.0) is True


def test_batch_dimension_other_than_one_is_rejected_before_sdk():
    model = FakeModel([])
    backend = MobilintNativeBackend(_runtime(model))
    inputs = _inputs()
    inputs["first"] = np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="batch dimension N=1"):
        backend.submit_async(inputs, lambda outcome: None)

    assert model.calls == []


def test_waiter_capacity_is_bounded_without_an_extra_request_queue():
    release = threading.Event()
    first = FakeFuture([np.array([[1]])], release=release)
    model = FakeModel([first])
    backend = MobilintNativeBackend(_runtime(model, slots=1))
    backend.submit_async(_inputs(), lambda outcome: None)

    with pytest.raises(RuntimeError, match="waiter capacity is exhausted"):
        backend.submit_async(_inputs(2), lambda outcome: None)

    assert len(model.calls) == 1
    release.set()
    assert backend.shutdown(timeout=1.0) is True


def test_terminal_future_releases_slot_before_callback_returns():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture([np.array([[1]])])
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )

    def blocked_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    first_job = backend.submit_async(_inputs(1), blocked_callback)
    assert callback_entered.wait(timeout=1.0)
    assert first_job in backend._jobs
    assert backend._jobs[first_job].inputs

    second_submission_succeeded = False
    try:
        second_job = backend.submit_async(
            _inputs(2), lambda outcome: second_done.set()
        )
        second_submission_succeeded = True

        assert second_done.wait(timeout=1.0)
        assert first_job in backend._jobs
    finally:
        release_callback.set()
        shutdown_succeeded = backend.shutdown(timeout=1.0)
    if second_submission_succeeded:
        assert shutdown_succeeded is True
    assert (first_job, second_job) == ("mobilint-1", "mobilint-2")
    assert first.get_calls == 1
    assert second.get_calls == 1


def test_terminal_output_is_detached_before_sdk_slot_reuse():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()

    class ReusingOutputModel:
        def __init__(self):
            self.shared_output = np.zeros((1, 1), dtype=np.int64)
            self.calls = 0

        def infer_async(self, inputs):
            self.calls += 1
            self.shared_output.fill(self.calls)
            return FakeFuture([self.shared_output])

    model = ReusingOutputModel()
    backend = MobilintNativeBackend(_runtime(model, slots=1))
    first_outcomes = []

    def blocked_callback(outcome):
        first_outcomes.append(outcome)
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(1), blocked_callback)
    assert callback_entered.wait(timeout=1.0)

    try:
        backend.submit_async(_inputs(2), lambda outcome: second_done.set())
        assert second_done.wait(timeout=1.0)
        np.testing.assert_array_equal(first_outcomes[0].outputs["output"], [[1]])
    finally:
        release_callback.set()

    assert backend.shutdown(timeout=1.0) is True


def test_failed_future_releases_slot_before_error_callback_returns():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture(error=RuntimeError("private SDK failure"))
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )
    outcomes = []

    def blocked_callback(outcome):
        outcomes.append(outcome)
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(1), blocked_callback)
    assert callback_entered.wait(timeout=1.0)

    second_submission_succeeded = False
    try:
        backend.submit_async(_inputs(2), lambda outcome: second_done.set())
        second_submission_succeeded = True

        assert second_done.wait(timeout=1.0)
        assert outcomes[0].error_type == "RuntimeError"
        assert outcomes[0].error_message == "Mobilint asynchronous inference failed."
    finally:
        release_callback.set()
        shutdown_succeeded = backend.shutdown(timeout=1.0)
    if second_submission_succeeded:
        assert shutdown_succeeded is True


def test_callback_failure_does_not_delay_next_sdk_submission():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture([np.array([[1]])])
    second = FakeFuture([np.array([[2]])])
    backend = MobilintNativeBackend(
        _runtime(FakeModel([first, second]), slots=1)
    )

    def failing_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)
        raise RuntimeError("consumer failed")

    backend.submit_async(_inputs(1), failing_callback)
    assert callback_entered.wait(timeout=1.0)

    second_submission_succeeded = False
    try:
        backend.submit_async(_inputs(2), lambda outcome: second_done.set())
        second_submission_succeeded = True

        assert second_done.wait(timeout=1.0)
    finally:
        release_callback.set()
        shutdown_succeeded = backend.shutdown(timeout=1.0)
    if second_submission_succeeded:
        assert shutdown_succeeded is True
    assert first.get_calls == 1
    assert second.get_calls == 1


def test_normalization_holds_slot_but_callback_does_not():
    normalization_entered = threading.Event()
    release_normalization = threading.Event()
    callback_entered = threading.Event()
    release_callback = threading.Event()
    second_done = threading.Event()
    first = FakeFuture([np.array([[1]])])
    second = FakeFuture([np.array([[2]])])
    runtime = _runtime(FakeModel([first, second]), slots=1)
    normalize_outputs = runtime._normalize_outputs

    def blocked_normalize(outputs, *, expected_batch_size=None):
        normalization_entered.set()
        assert release_normalization.wait(timeout=2.0)
        return normalize_outputs(outputs, expected_batch_size=expected_batch_size)

    runtime._normalize_outputs = blocked_normalize
    backend = MobilintNativeBackend(runtime)

    def blocked_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(1), blocked_callback)
    sequence_succeeded = False
    try:
        assert normalization_entered.wait(timeout=1.0)
        with pytest.raises(RuntimeError, match="waiter capacity is exhausted"):
            backend.submit_async(_inputs(2), lambda outcome: second_done.set())

        release_normalization.set()
        assert callback_entered.wait(timeout=1.0)
        backend.submit_async(_inputs(2), lambda outcome: second_done.set())
        assert second_done.wait(timeout=1.0)
        assert first.get_calls == 1
        assert second.get_calls == 1
        sequence_succeeded = True
    finally:
        release_normalization.set()
        release_callback.set()
        try:
            shutdown_succeeded = backend.shutdown(timeout=1.0)
        except BaseException:
            if sequence_succeeded:
                raise
    if sequence_succeeded:
        assert shutdown_succeeded is True


def test_shutdown_waits_for_callback_after_sdk_slot_is_available():
    callback_entered = threading.Event()
    release_callback = threading.Event()
    backend = MobilintNativeBackend(
        _runtime(
            FakeModel([FakeFuture([np.array([[1]])])]),
            slots=1,
        )
    )

    def blocked_callback(outcome):
        callback_entered.set()
        assert release_callback.wait(timeout=2.0)

    backend.submit_async(_inputs(), blocked_callback)
    sequence_succeeded = False
    try:
        assert callback_entered.wait(timeout=1.0)
        assert backend._slots.acquire(blocking=False) is True
        backend._slots.release()

        assert backend.shutdown(timeout=0.01) is False
        sequence_succeeded = True
    finally:
        release_callback.set()
        try:
            shutdown_succeeded = backend.shutdown(timeout=1.0)
        except BaseException:
            if sequence_succeeded:
                raise
    if sequence_succeeded:
        assert shutdown_succeeded is True


def test_shutdown_refuses_quiescence_until_future_finishes():
    release = threading.Event()
    future = FakeFuture([np.array([[3]])], release=release)
    backend = MobilintNativeBackend(_runtime(FakeModel([future]), slots=1))
    backend.submit_async(_inputs(), lambda outcome: None)

    assert backend.shutdown(timeout=0.01) is False
    with pytest.raises(RuntimeError, match="shutting down"):
        backend.submit_async(_inputs(), lambda outcome: None)
    release.set()
    assert backend.shutdown(timeout=1.0) is True


def test_shutdown_tracks_an_infer_async_call_that_is_itself_blocked():
    entered = threading.Event()
    release_submit = threading.Event()

    class BlockingSubmitModel:
        def infer_async(self, inputs):
            entered.set()
            release_submit.wait(timeout=2.0)
            return FakeFuture([np.array([[4]])])

    backend = MobilintNativeBackend(_runtime(BlockingSubmitModel(), slots=1))
    submitter = threading.Thread(
        target=backend.submit_async,
        args=(_inputs(), lambda outcome: None),
    )
    submitter.start()
    assert entered.wait(timeout=1.0)
    assert backend.shutdown(timeout=0.01) is False
    release_submit.set()
    submitter.join(timeout=1.0)
    assert not submitter.is_alive()
    assert backend.shutdown(timeout=1.0) is True


def test_callback_exception_does_not_leak_job_or_waiter_capacity():
    future = FakeFuture([np.array([[5]])])
    backend = MobilintNativeBackend(_runtime(FakeModel([future]), slots=1))
    called = threading.Event()

    def failing_callback(outcome):
        called.set()
        raise RuntimeError("consumer failed")

    backend.submit_async(_inputs(), failing_callback)

    assert called.wait(timeout=1.0)
    assert backend.shutdown(timeout=1.0) is True
    assert future.get_calls == 1


@pytest.mark.parametrize(
    "failure_stage",
    ["construction", "start", "start_without_liveness", "start_then_raise"],
)
def test_waiter_startup_failure_after_sdk_acceptance_falls_back_exactly_once(
    monkeypatch, failure_stage
):
    first = FakeFuture([np.array([[6]])])
    second = FakeFuture([np.array([[7]])])
    model = FakeModel([first, second])
    backend = MobilintNativeBackend(_runtime(model, slots=1))
    original_thread = threading.Thread
    startup_attempts = 0
    completed = []
    second_done = threading.Event()

    def thread_factory(*args, **kwargs):
        nonlocal startup_attempts
        startup_attempts += 1
        assert len(model.calls) == startup_attempts
        if startup_attempts == 1 and failure_stage == "construction":
            raise RuntimeError("waiter construction failed")
        if startup_attempts == 1 and failure_stage == "start_without_liveness":
            class OpaqueThread:
                def start(self):
                    raise RuntimeError("opaque waiter start failed")

            return OpaqueThread()
        thread = original_thread(*args, **kwargs)
        if startup_attempts == 1 and failure_stage == "start":
            thread.start = lambda: (_ for _ in ()).throw(
                RuntimeError("waiter start failed")
            )
        elif startup_attempts == 1 and failure_stage == "start_then_raise":
            original_start = thread.start

            def start_then_raise():
                original_start()
                raise RuntimeError("waiter start reported failure")

            thread.start = start_then_raise
        return thread

    monkeypatch.setattr(mobilint_rt_module.threading, "Thread", thread_factory)

    first_job = backend.submit_async(_inputs(1), completed.append)
    deadline = time.monotonic() + 1.0
    while first_job in backend._jobs and time.monotonic() < deadline:
        time.sleep(0.001)
    assert first_job not in backend._jobs
    second_job = backend.submit_async(
        _inputs(2),
        lambda outcome: (completed.append(outcome), second_done.set()),
    )

    assert second_done.wait(timeout=1.0)
    assert (first_job, second_job) == ("mobilint-1", "mobilint-2")
    assert first.get_calls == 1
    assert second.get_calls == 1
    assert len(completed) == 2
    assert backend.shutdown(timeout=1.0) is True
    assert backend._active_submissions == 0
    assert backend._jobs == {}
    assert backend._threads == set()


def test_zero_timeout_shutdown_succeeds_with_retained_dead_waiter():
    future = FakeFuture([np.array([[8]])])
    backend = MobilintNativeBackend(_runtime(FakeModel([future]), slots=1))
    callback_done = threading.Event()
    backend.submit_async(_inputs(), lambda outcome: callback_done.set())

    assert callback_done.wait(timeout=1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with backend._condition:
            retained_threads = tuple(backend._threads)
            if not backend._jobs and all(
                not thread.is_alive() for thread in retained_threads
            ):
                break
        time.sleep(0.001)

    assert retained_threads
    assert all(not thread.is_alive() for thread in retained_threads)
    assert backend.shutdown(timeout=0) is True
    assert backend._threads == set()


def test_existing_executor_retains_late_job_until_callback_and_ack():
    release = threading.Event()
    future = FakeFuture([np.array([[9]])], release=release)
    backend = MobilintNativeBackend(_runtime(FakeModel([future]), slots=1))
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=0.005,
    )

    execution = executor.execute(_inputs())
    assert execution.error_type == "NativeAsyncTimeout"
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 1
    release.set()
    deadline = time.monotonic() + 1.0
    while executor.snapshot().inflight and time.monotonic() < deadline:
        time.sleep(0.005)

    assert executor.snapshot().inflight == 0
    assert executor.snapshot().late_callbacks == 1
    assert backend.shutdown(timeout=1.0) is True

import gc
import subprocess
import sys
import threading
import time
import warnings
import weakref
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from core.async_inference.completion import CompletionCoordinator
from core.async_inference.engine import AsyncInferenceEngine
from core.async_inference.metrics import AsyncMetricsCollector
from core.async_inference.types import AsyncInferenceConfig, RunStatus
from core.inference_engine import InferenceEngine
from core.inference_pipeline import InferencePipeline
from core.runtime_executor import NativeAsyncOutcome, NativeAsyncRuntimeExecutor
from rbln_test_utils import fake_rebel, loaded_runtime, valid_inputs


def test_module_cli_help_exposes_rbln_target_and_backend():
    framework_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "-m", "src.main", "--help"],
        cwd=framework_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "rbln" in completed.stdout
    assert "rbln-static" in completed.stdout


def _wait_for_count(items, count, timeout=1.0):
    event = threading.Event()

    def append(value):
        items.append(value)
        if len(items) >= count:
            event.set()

    return append, event


def _wait_for_backend_jobs(backend, count, timeout=1.0):
    with backend._condition:
        return backend._condition.wait_for(
            lambda: len(backend._jobs) == count,
            timeout=timeout,
        )


def _wait_for_executor_inflight(executor, count, timeout=1.0):
    with executor._condition:
        return executor._condition.wait_for(
            lambda: executor.snapshot().inflight == count,
            timeout=timeout,
        )


class _RblnAsyncLoader:
    def __init__(self):
        template = valid_inputs()
        self.samples = [
            {
                "input": {
                    name: np.array(value[0], copy=True)
                    for name, value in template.items()
                },
                "label": sample_index,
            }
            for sample_index in range(2)
        ]
        self.current_idx = 0

    def get_metadata(self):
        return {"is_static_batched": False, "total_samples": 2}

    def load_batch(self, batch_size):
        batch = self.samples[self.current_idx:self.current_idx + batch_size]
        self.current_idx += len(batch)
        return batch

    def load_by_index(self, index):
        return self.samples[index]


class _CountingEvaluator:
    def __init__(self):
        self.samples = 0

    def add_batch(self, outputs, labels, timing_ms):
        del outputs, timing_ms
        self.samples += len(labels)

    def compute(self):
        return {"Total Samples": self.samples}


class _ObservedBoundedPermit:
    def __init__(self, capacity):
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self.acquire_calls = 0
        self.second_acquire_entered = threading.Event()

    def acquire(self, *, timeout):
        with self._lock:
            self.acquire_calls += 1
            if self.acquire_calls == 2:
                self.second_acquire_entered.set()
        return self._semaphore.acquire(timeout=timeout)

    def release(self):
        self._semaphore.release()


class _DrainWaitCondition(threading.Condition):
    def __init__(self, wait_entered):
        super().__init__(threading.RLock())
        self.wait_entered = wait_entered

    def wait(self, timeout=None):
        self.wait_entered.set()
        return super().wait(timeout=timeout)


class _ConcurrentTimingMapping(Mapping):
    def __init__(self):
        self._data = {"total_ms": 1.0}
        self._barrier = threading.Barrier(2)
        self._lock = threading.Lock()
        self.item_threads = []
        self.overlapped = threading.Event()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def items(self):
        with self._lock:
            self.item_threads.append(threading.get_ident())
        self._barrier.wait(timeout=1.0)
        self.overlapped.set()
        return self._data.items()


def test_native_backend_constructs_and_executes_on_owner_loop(
    loaded_runtime, fake_rebel
):
    loaded_runtime.async_parallel = 2
    backend = loaded_runtime.create_native_backend()
    outcomes = []
    callback, callback_done = _wait_for_count(outcomes, 1)

    job_id = backend.submit_async(valid_inputs(), callback)

    assert job_id == "rbln-1"
    assert not callback_done.is_set()
    assert backend._thread.daemon is True
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_done.wait(timeout=1.0)
    owner_ident = backend.owner_thread_ident
    assert fake_rebel.async_constructor_thread == owner_ident
    assert fake_rebel.async_run_threads == [owner_ident]
    assert fake_rebel.async_runtime_calls == [
        (
            str(loaded_runtime.compiled_model.artifact_path),
            {
                "device": 0,
                "tensor_type": "np",
                "parallel": 2,
                "timeout": 17.5,
            },
        )
    ]
    assert loaded_runtime.create_native_backend() is backend
    assert len(fake_rebel.async_runtime_calls) == 1
    assert list(outcomes[0].outputs) == ["logits"]
    assert outcomes[0].error_type is None
    assert backend.shutdown(timeout=1.0) is True


def test_submit_validates_synchronously_and_ids_are_monotonic(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()

    with pytest.raises(ValueError, match="missing required inputs"):
        backend.submit_async({"input_ids": valid_inputs()["input_ids"]}, lambda _: None)

    first = backend.submit_async(valid_inputs(), lambda _: None)
    second = backend.submit_async(valid_inputs(), lambda _: None)

    assert (first, second) == ("rbln-1", "rbln-2")
    assert fake_rebel.wait_for_async_calls(2)
    assert fake_rebel.release_call(1)
    assert fake_rebel.release_call(2)
    assert backend.shutdown(timeout=1.0) is True


def test_submit_coroutines_share_owner_loop_and_complete_out_of_order(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    outcomes = []
    callback, callbacks_done = _wait_for_count(outcomes, 2)

    first = backend.submit_async(
        valid_inputs(), lambda outcome: callback(("first", outcome))
    )
    second = backend.submit_async(
        valid_inputs(), lambda outcome: callback(("second", outcome))
    )

    assert (first, second) == ("rbln-1", "rbln-2")
    assert fake_rebel.wait_for_async_calls(2)
    assert fake_rebel.async_run_threads == [
        backend.owner_thread_ident,
        backend.owner_thread_ident,
    ]
    assert fake_rebel.release_call(2)
    assert fake_rebel.release_call(1)
    assert callbacks_done.wait(timeout=1.0)
    assert [name for name, _ in outcomes] == ["second", "first"]
    assert backend.shutdown(timeout=1.0) is True


def test_submit_uses_sync_output_normalization_contract(
    loaded_runtime, fake_rebel
):
    logits = np.array([[0.75, 0.25]], dtype=np.float32)
    fake_rebel.async_outputs[1] = [logits]
    backend = loaded_runtime.create_native_backend()
    outcomes = []
    callback, callback_done = _wait_for_count(outcomes, 1)

    backend.submit_async(valid_inputs(), callback)
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_done.wait(timeout=1.0)

    assert outcomes[0].outputs == {"logits": logits}
    assert outcomes[0].outputs["logits"] is logits
    assert backend.shutdown(timeout=1.0) is True


@pytest.mark.parametrize(
    ("sdk_output", "sdk_error", "expected_type"),
    [
        (
            None,
            type("VendorError" * 20, (RuntimeError,), {})("secret"),
            ("VendorError" * 20)[:64],
        ),
        (np.ones((1, 2), dtype=np.float64), None, "RuntimeError"),
    ],
)
def test_submit_bounds_sdk_and_normalization_exceptions(
    loaded_runtime,
    fake_rebel,
    sdk_output,
    sdk_error,
    expected_type,
):
    if sdk_output is not None:
        fake_rebel.async_outputs[1] = sdk_output
    if sdk_error is not None:
        fake_rebel.async_errors[1] = sdk_error
    backend = loaded_runtime.create_native_backend()
    outcomes = []
    callback, callback_done = _wait_for_count(outcomes, 1)
    inputs = valid_inputs()
    input_references = [weakref.ref(value) for value in inputs.values()]

    backend.submit_async(inputs, callback)
    del inputs
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_done.wait(timeout=1.0)

    outcome = outcomes[0]
    assert outcome.error_type == expected_type
    assert len(outcome.error_type) <= 64
    assert outcome.error_type.replace("_", "").isalnum()
    assert outcome.error_message == "RBLN asynchronous inference failed."
    assert "secret" not in outcome.error_message
    assert len(outcome.error_message) <= 512
    assert backend.shutdown(timeout=1.0) is True
    assert backend._jobs == {}
    assert fake_rebel.async_input_refs == [()]
    if sdk_error is not None:
        sdk_error.__traceback__ = None
    gc.collect()
    assert all(reference() is None for reference in input_references)


def test_callback_exception_is_attempted_once_and_releases_job_inputs(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    inputs = valid_inputs()
    input_references = [weakref.ref(value) for value in inputs.values()]
    callback_count = 0
    callback_done = threading.Event()

    def failing_callback(outcome):
        nonlocal callback_count
        del outcome
        callback_count += 1
        callback_done.set()
        raise RuntimeError("consumer detail must not retain inputs")

    backend.submit_async(inputs, failing_callback)
    del inputs
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_done.wait(timeout=1.0)
    assert backend.shutdown(timeout=1.0) is True
    gc.collect()

    assert callback_count == 1
    assert backend._jobs == {}
    assert all(reference() is None for reference in input_references)


def test_sync_and_native_async_modes_are_exclusive(loaded_runtime, fake_rebel):
    loaded_runtime.run(valid_inputs())

    with pytest.raises(RuntimeError, match="sync mode"):
        loaded_runtime.create_native_backend()

    assert fake_rebel.async_runtime_calls == []
    loaded_runtime.unload()


def test_native_async_and_sync_modes_are_exclusive(loaded_runtime, fake_rebel):
    backend = loaded_runtime.create_native_backend()

    with pytest.raises(RuntimeError, match="native async mode"):
        loaded_runtime.run(valid_inputs())

    assert fake_rebel.runtime_calls == []
    assert backend.shutdown(timeout=1.0) is True


def test_native_async_warmup_uses_owner_runtime_without_measured_jobs(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    warmup_errors = []

    def run_warmup():
        try:
            loaded_runtime.warmup(valid_inputs(), num_runs=1)
        except BaseException as exc:
            warmup_errors.append(exc)

    warmup_thread = threading.Thread(target=run_warmup, daemon=True)
    warmup_thread.start()
    assert fake_rebel.wait_for_async_calls(1)
    assert backend._jobs == {}
    with backend._condition:
        assert backend._condition.wait_for(
            lambda: len(backend._warmup_futures) == 1,
            timeout=1.0,
        )
    assert fake_rebel.release_call(1)
    warmup_thread.join(timeout=1.0)

    assert not warmup_thread.is_alive()
    assert warmup_errors == []
    outcomes = []
    callback, callback_done = _wait_for_count(outcomes, 1)
    assert backend.submit_async(valid_inputs(), callback) == "rbln-1"
    assert fake_rebel.wait_for_async_calls(2)
    assert fake_rebel.release_call(2)
    assert callback_done.wait(timeout=1.0)
    assert fake_rebel.async_run_threads == [
        backend.owner_thread_ident,
        backend.owner_thread_ident,
    ]
    assert backend.shutdown(timeout=1.0) is True


def test_warmup_timeout_remains_tracked_until_physical_completion(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()

    with pytest.raises(TimeoutError):
        backend.run_warmup_blocking(valid_inputs(), timeout=0.001)

    assert fake_rebel.wait_for_async_calls(1)
    assert len(backend._warmup_futures) == 1
    assert backend.shutdown(timeout=0.001) is False
    assert backend.owner_thread_alive
    assert backend._async_runtime is not None
    assert fake_rebel.release_call(1)
    assert backend.shutdown(timeout=1.0) is True


def test_unload_publishes_closing_while_native_warmup_is_active(
    loaded_runtime, fake_rebel, monkeypatch
):
    loaded_runtime.shutdown_timeout_sec = 0.01
    backend = loaded_runtime.create_native_backend()
    warmup_errors = []
    unload_errors = []
    unload_invoked = threading.Event()
    unload_finished = threading.Event()
    shutdown_entered = threading.Event()
    real_shutdown = backend.shutdown

    def observe_shutdown(timeout):
        shutdown_entered.set()
        return real_shutdown(timeout)

    monkeypatch.setattr(backend, "shutdown", observe_shutdown)

    def run_warmup():
        try:
            loaded_runtime.warmup(valid_inputs(), num_runs=1)
        except BaseException as exc:
            warmup_errors.append(exc)

    def unload_runtime():
        unload_invoked.set()
        try:
            loaded_runtime.unload()
        except BaseException as exc:
            unload_errors.append(exc)
        finally:
            unload_finished.set()

    warmup_thread = threading.Thread(target=run_warmup, daemon=True)
    unload_thread = threading.Thread(target=unload_runtime, daemon=True)
    warmup_thread.start()
    assert fake_rebel.wait_for_async_calls(1)
    with backend._condition:
        assert backend._condition.wait_for(
            lambda: len(backend._warmup_futures) == 1,
            timeout=1.0,
        )

    unload_thread.start()
    assert unload_invoked.wait(timeout=1.0)
    shutdown_reached_active_warmup = shutdown_entered.wait(timeout=1.0)
    closing_published = False
    if shutdown_reached_active_warmup:
        with backend._condition:
            closing_published = backend._condition.wait_for(
                lambda: backend._closing,
                timeout=1.0,
            )
    submission_error = None
    accepted_job = None
    try:
        accepted_job = backend.submit_async(valid_inputs(), lambda _: None)
    except RuntimeError as exc:
        submission_error = exc

    try:
        assert shutdown_reached_active_warmup is True
        assert closing_published is True
        assert unload_finished.wait(timeout=1.0)
        assert len(unload_errors) == 1
        assert "cleanup pending" in str(unload_errors[0])
        assert submission_error is not None
        assert "shutting down" in str(submission_error)
        assert accepted_job is None
        with backend._condition:
            assert backend._closing is True
            assert backend._jobs == {}
            assert len(backend._warmup_futures) == 1
        assert loaded_runtime._cleanup_pending is True
        assert loaded_runtime._native_backend is backend
        assert warmup_thread.is_alive()
        assert backend.owner_thread_alive
    finally:
        if accepted_job is not None:
            assert fake_rebel.wait_for_async_calls(2)
            assert fake_rebel.release_call(2)
        assert fake_rebel.release_call(1)
        warmup_thread.join(timeout=1.0)
        unload_thread.join(timeout=1.0)
        assert real_shutdown(timeout=1.0) is True

    assert warmup_errors == []
    loaded_runtime.unload()


def test_submit_publication_failure_closes_unscheduled_coroutine(
    loaded_runtime, fake_rebel, monkeypatch
):
    backend = loaded_runtime.create_native_backend()
    inputs = valid_inputs()
    input_references = [weakref.ref(value) for value in inputs.values()]
    errors = []

    def fail_publication(coroutine, loop):
        del coroutine, loop
        raise RuntimeError("publication failed")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with monkeypatch.context() as publication_patch:
            publication_patch.setattr(
                "runtimes.rbln_rt.asyncio.run_coroutine_threadsafe",
                fail_publication,
            )
            try:
                backend.submit_async(inputs, lambda _: None)
            except RuntimeError as exc:
                errors.append(str(exc))
                exc.__traceback__ = None
        del inputs
        gc.collect()

    assert errors == ["publication failed"]
    assert backend._jobs == {}
    assert all(reference() is None for reference in input_references)
    assert not [
        warning
        for warning in caught
        if "was never awaited" in str(warning.message)
    ]
    assert backend.shutdown(timeout=1.0) is True


def test_warmup_publication_failure_closes_unscheduled_coroutine(
    loaded_runtime, fake_rebel, monkeypatch
):
    backend = loaded_runtime.create_native_backend()
    inputs = valid_inputs()
    input_references = [weakref.ref(value) for value in inputs.values()]
    errors = []

    def fail_publication(coroutine, loop):
        del coroutine, loop
        raise RuntimeError("publication failed")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with monkeypatch.context() as publication_patch:
            publication_patch.setattr(
                "runtimes.rbln_rt.asyncio.run_coroutine_threadsafe",
                fail_publication,
            )
            try:
                backend.run_warmup_blocking(inputs, timeout=1.0)
            except RuntimeError as exc:
                errors.append(str(exc))
                exc.__traceback__ = None
        del inputs
        gc.collect()

    assert errors == ["publication failed"]
    assert backend._warmup_futures == set()
    assert all(reference() is None for reference in input_references)
    assert not [
        warning
        for warning in caught
        if "was never awaited" in str(warning.message)
    ]
    assert backend.shutdown(timeout=1.0) is True


def test_shutdown_rejects_new_submissions_after_deadline(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    backend.submit_async(valid_inputs(), lambda outcome: None)
    assert fake_rebel.wait_for_async_calls(1)

    assert backend.shutdown(timeout=0.001) is False
    with pytest.raises(RuntimeError, match="shutting down"):
        backend.submit_async(valid_inputs(), lambda outcome: None)

    assert fake_rebel.release_call(1)
    assert backend.shutdown(timeout=1.0) is True


def test_submit_shutdown_overlap_tracks_accepted_job_and_rejects_after_closing(
    loaded_runtime, fake_rebel, monkeypatch
):
    from runtimes import rbln_rt

    backend = loaded_runtime.create_native_backend()
    real_publication = rbln_rt.asyncio.run_coroutine_threadsafe
    publication_entered = threading.Event()
    publication_release = threading.Event()
    callback_done = threading.Event()
    submit_results = []
    submit_errors = []
    shutdown_results = []

    def gated_publication(coroutine, loop):
        publication_entered.set()
        assert publication_release.wait(timeout=1.0)
        return real_publication(coroutine, loop)

    monkeypatch.setattr(
        "runtimes.rbln_rt.asyncio.run_coroutine_threadsafe",
        gated_publication,
    )

    def submit_before_closing():
        try:
            submit_results.append(
                backend.submit_async(
                    valid_inputs(), lambda _: callback_done.set()
                )
            )
        except BaseException as exc:
            submit_errors.append(exc)

    submit_thread = threading.Thread(target=submit_before_closing, daemon=True)
    submit_thread.start()
    assert publication_entered.wait(timeout=1.0)
    with backend._condition:
        assert tuple(backend._jobs) == ("rbln-1",)
        assert backend._jobs["rbln-1"].future is None
        assert backend._closing is False

    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(backend.shutdown(timeout=1.0)),
        daemon=True,
    )
    shutdown_thread.start()
    with backend._condition:
        assert backend._condition.wait_for(
            lambda: backend._closing,
            timeout=1.0,
        )
        assert tuple(backend._jobs) == ("rbln-1",)

    with pytest.raises(RuntimeError, match="shutting down"):
        backend.submit_async(valid_inputs(), lambda _: None)
    with backend._condition:
        assert tuple(backend._jobs) == ("rbln-1",)
        assert backend._next_job_id == 2

    publication_release.set()
    submit_thread.join(timeout=1.0)
    assert submit_results == ["rbln-1"]
    assert submit_errors == []
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_done.wait(timeout=1.0)
    shutdown_thread.join(timeout=1.0)

    assert shutdown_results == [True]
    assert not submit_thread.is_alive()
    assert not shutdown_thread.is_alive()
    assert backend._jobs == {}
    assert not backend.owner_thread_alive


def test_shutdown_waits_for_callback_cleanup_and_owner_release(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    owner_ident = backend.owner_thread_ident
    callback_entered = threading.Event()
    callback_release = threading.Event()
    shutdown_results = []

    def blocking_callback(outcome):
        del outcome
        callback_entered.set()
        assert callback_release.wait(timeout=1.0)

    backend.submit_async(valid_inputs(), blocking_callback)
    assert fake_rebel.wait_for_async_calls(1)
    assert fake_rebel.release_call(1)
    assert callback_entered.wait(timeout=1.0)
    shutdown_thread = threading.Thread(
        target=lambda: shutdown_results.append(backend.shutdown(timeout=1.0)),
        daemon=True,
    )
    shutdown_thread.start()
    time.sleep(0.01)

    assert shutdown_thread.is_alive()
    assert backend._jobs
    assert backend._async_runtime is not None
    callback_release.set()
    shutdown_thread.join(timeout=1.0)

    assert shutdown_results == [True]
    assert backend._jobs == {}
    assert backend._async_runtime is None
    assert backend._loop_stopped is True
    assert not backend.owner_thread_alive
    assert fake_rebel.destruction_threads == [owner_ident]


def test_shutdown_timeout_retains_runtime_until_late_completion(
    loaded_runtime, fake_rebel
):
    loaded_runtime.shutdown_timeout_sec = 0.001
    backend = loaded_runtime.create_native_backend()
    outcomes = []
    backend.submit_async(valid_inputs(), outcomes.append)
    assert fake_rebel.wait_for_async_calls(1)

    assert backend.shutdown(timeout=0.001) is False
    assert backend.owner_thread_alive
    with pytest.raises(RuntimeError, match="cleanup pending"):
        loaded_runtime.unload()

    assert loaded_runtime.compiled_model is not None
    assert loaded_runtime._native_backend is backend
    assert fake_rebel.release_call(1)
    assert backend.shutdown(timeout=1.0) is True
    assert len(outcomes) == 1
    assert outcomes[0].error_type is None
    loaded_runtime.unload()
    assert loaded_runtime.execution_mode == "unloaded"


def test_shutdown_is_idempotent_after_success(loaded_runtime, fake_rebel):
    backend = loaded_runtime.create_native_backend()

    assert backend.shutdown(timeout=1.0) is True
    assert backend.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True


def test_shutdown_retry_joins_delayed_owner_finalization(
    loaded_runtime, fake_rebel
):
    destruction_gate = threading.Event()
    fake_rebel.async_destruction_gate = destruction_gate
    backend = loaded_runtime.create_native_backend()

    assert backend.shutdown(timeout=0.001) is False
    assert fake_rebel.async_destruction_entered.wait(timeout=1.0)
    assert backend._async_runtime is None
    assert backend.owner_thread_alive
    destruction_gate.set()

    assert backend.shutdown(timeout=1.0) is True
    assert not backend.owner_thread_alive


def test_startup_constructor_failure_rolls_back_to_loaded(
    loaded_runtime, fake_rebel
):
    fake_rebel.async_runtime_error = RuntimeError("secret constructor detail")

    with pytest.raises(RuntimeError, match="initialization failed") as caught:
        loaded_runtime.create_native_backend()

    assert "secret" not in str(caught.value)
    assert loaded_runtime._native_backend is None
    assert loaded_runtime.get_device_spec()["execution_mode"] == "loaded"
    fake_rebel.async_runtime_error = None
    backend = loaded_runtime.create_native_backend()
    assert backend.shutdown(timeout=1.0) is True


def test_concurrent_factory_callers_never_observe_partial_backend(
    loaded_runtime, fake_rebel
):
    constructor_gate = threading.Event()
    fake_rebel.async_constructor_gate = constructor_gate
    backends = []
    errors = []

    def create_backend():
        try:
            backends.append(loaded_runtime.create_native_backend())
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_backend, daemon=True)
    second = threading.Thread(target=create_backend, daemon=True)
    first.start()
    assert fake_rebel.async_constructor_entered.wait(timeout=1.0)
    second.start()
    second_returned_during_startup = not second.is_alive()
    if not second_returned_during_startup:
        second_returned_during_startup = second.join(timeout=0.02) is None and not second.is_alive()
    constructor_gate.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)

    assert second_returned_during_startup is False
    assert errors == []
    assert len(backends) == 2
    assert backends[0] is backends[1]
    assert len(fake_rebel.async_runtime_calls) == 1
    assert backends[0].shutdown(timeout=1.0) is True


def test_sync_run_waits_for_native_startup_before_rejecting_mode(
    loaded_runtime, fake_rebel
):
    constructor_gate = threading.Event()
    fake_rebel.async_constructor_gate = constructor_gate
    backend_results = []
    run_errors = []
    run_done = threading.Event()

    factory_thread = threading.Thread(
        target=lambda: backend_results.append(
            loaded_runtime.create_native_backend()
        ),
        daemon=True,
    )
    factory_thread.start()
    assert fake_rebel.async_constructor_entered.wait(timeout=1.0)

    def run_sync():
        try:
            loaded_runtime.run(valid_inputs())
        except BaseException as exc:
            run_errors.append(exc)
        finally:
            run_done.set()

    run_thread = threading.Thread(target=run_sync, daemon=True)
    run_thread.start()
    run_returned_during_startup = run_done.wait(timeout=0.02)
    constructor_gate.set()
    factory_thread.join(timeout=1.0)
    run_thread.join(timeout=1.0)

    assert run_returned_during_startup is False
    assert len(run_errors) == 1
    assert "native async mode" in str(run_errors[0])
    assert fake_rebel.runtime_calls == []
    assert backend_results[0].shutdown(timeout=1.0) is True


def test_owner_loop_setup_failure_is_published_and_retryable(
    loaded_runtime, fake_rebel, monkeypatch
):
    from runtimes import rbln_rt

    real_new_event_loop = rbln_rt.asyncio.new_event_loop
    monkeypatch.setattr("runtimes.rbln_rt._RBLN_STARTUP_TIMEOUT_SEC", 0.01)
    monkeypatch.setattr(
        "runtimes.rbln_rt.asyncio.new_event_loop",
        lambda: (_ for _ in ()).throw(RuntimeError("loop setup detail")),
    )

    with pytest.raises(RuntimeError, match="initialization failed") as caught:
        loaded_runtime.create_native_backend()

    assert "loop setup detail" not in str(caught.value)
    assert loaded_runtime._native_backend is None
    assert loaded_runtime._cleanup_pending is False
    assert loaded_runtime.execution_mode == "loaded"
    monkeypatch.setattr(
        "runtimes.rbln_rt.asyncio.new_event_loop", real_new_event_loop
    )
    backend = loaded_runtime.create_native_backend()
    assert backend.shutdown(timeout=1.0) is True


def test_startup_timeout_retains_cleanup_owner_until_constructor_exits(
    loaded_runtime, fake_rebel, monkeypatch
):
    constructor_gate = threading.Event()
    fake_rebel.async_constructor_gate = constructor_gate
    monkeypatch.setattr("runtimes.rbln_rt._RBLN_STARTUP_TIMEOUT_SEC", 0.001)

    try:
        with pytest.raises(TimeoutError, match="startup timed out"):
            loaded_runtime.create_native_backend()

        assert loaded_runtime._cleanup_pending is True
        assert loaded_runtime._native_backend is not None
        assert loaded_runtime.execution_mode == "native_async"
    finally:
        constructor_gate.set()

    loaded_runtime.unload()
    assert loaded_runtime.execution_mode == "unloaded"
    assert fake_rebel.destruction_threads == [
        fake_rebel.async_constructor_thread
    ]


def test_executor_capacity_and_reverse_rbln_completion_preserve_identity(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=2,
        completion_timeout_sec=1.0,
    )
    fake_rebel.async_outputs[1] = np.array(
        [[0.9, 0.1]], dtype=np.float32
    )
    fake_rebel.async_outputs[2] = np.array(
        [[0.2, 0.8]], dtype=np.float32
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(1)
        second_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(2)
        assert executor.snapshot().inflight == 2

        blocked = executor.execute(valid_inputs(), timeout=0.0)
        assert blocked.error_type == "NativeAsyncBackpressureTimeout"
        assert blocked.dispatch_token is None
        executor.acknowledge(blocked)
        assert len(fake_rebel.async_run_threads) == 2

        assert fake_rebel.release_call(2)
        second = second_future.result(timeout=1.0)
        assert fake_rebel.release_call(1)
        first = first_future.result(timeout=1.0)

    np.testing.assert_array_equal(
        first.outputs["logits"],
        np.array([[0.9, 0.1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        second.outputs["logits"],
        np.array([[0.2, 0.8]], dtype=np.float32),
    )
    assert (first.vendor_job_id, second.vendor_job_id) == (
        "rbln-1",
        "rbln-2",
    )
    assert first.dispatch_token != second.dispatch_token
    assert _wait_for_backend_jobs(backend, 0)
    executor.acknowledge(first)
    executor.acknowledge(second)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True


@pytest.mark.parametrize(
    ("completion_kind", "expected_error"),
    [
        ("success", None),
        ("sdk_failure", "RuntimeError"),
        ("malformed_output", "RuntimeError"),
        ("callback_failure", None),
    ],
)
def test_executor_terminal_paths_release_one_rbln_adapter_job(
    loaded_runtime,
    fake_rebel,
    monkeypatch,
    completion_kind,
    expected_error,
):
    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    if completion_kind == "sdk_failure":
        fake_rebel.async_errors[1] = RuntimeError("vendor detail")
    elif completion_kind == "malformed_output":
        fake_rebel.async_outputs[1] = np.ones((1, 2), dtype=np.float64)
    elif completion_kind == "callback_failure":
        real_submit = backend.submit_async

        def submit_with_failure_after_executor_terminal(inputs, callback):
            def callback_then_fail(outcome):
                callback(outcome)
                raise RuntimeError("consumer callback failed")

            return real_submit(inputs, callback_then_fail)

        monkeypatch.setattr(
            backend,
            "submit_async",
            submit_with_failure_after_executor_terminal,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(1)
        with backend._condition:
            assert tuple(backend._jobs) == ("rbln-1",)
        assert fake_rebel.release_call(1)
        execution = execution_future.result(timeout=1.0)

    assert execution.error_type == expected_error
    assert _wait_for_backend_jobs(backend, 0)
    assert backend._next_job_id == 2
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True


def test_logical_timeout_keeps_rbln_physical_ownership(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        timed_out_future = pool.submit(
            executor.execute,
            valid_inputs(),
            0.1,
        )
        assert fake_rebel.wait_for_async_calls(1)
        execution = timed_out_future.result(timeout=1.0)

    assert execution.error_type == "NativeAsyncTimeout"
    assert executor.snapshot().inflight == 1
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 1

    blocked = executor.execute(valid_inputs(), timeout=0.0)
    assert blocked.error_type == "NativeAsyncBackpressureTimeout"
    assert blocked.dispatch_token is None
    assert len(fake_rebel.async_run_threads) == 1

    assert fake_rebel.release_call(1)
    assert _wait_for_executor_inflight(executor, 0)
    snapshot = executor.snapshot()
    assert snapshot.timeouts == 1
    assert snapshot.late_callbacks == 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        admitted_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(2)
        assert fake_rebel.release_call(2)
        admitted = admitted_future.result(timeout=1.0)

    assert admitted.error_type is None
    assert admitted.vendor_job_id == "rbln-2"
    executor.acknowledge(admitted)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True


def test_duplicate_rbln_boundary_completion_cannot_replace_first_terminal(
    loaded_runtime, fake_rebel, monkeypatch
):
    backend = loaded_runtime.create_native_backend()
    callback_delivered = threading.Event()
    timing = _ConcurrentTimingMapping()
    callback_errors = []
    callback_threads = []
    real_submit = backend.submit_async

    def capture_executor_callback(inputs, callback):
        def recorded_callback(outcome):
            del outcome
            candidates = (
                NativeAsyncOutcome(
                    outputs={
                        "logits": np.array(
                            [[0.7, 0.3]], dtype=np.float32
                        )
                    },
                    timing_ms=timing,
                ),
                NativeAsyncOutcome(
                    outputs={
                        "logits": np.array(
                            [[0.2, 0.8]], dtype=np.float32
                        )
                    },
                    timing_ms=timing,
                ),
            )

            def invoke(candidate):
                try:
                    callback(candidate)
                except BaseException as exc:
                    callback_errors.append(exc)

            for candidate_index, candidate in enumerate(candidates):
                callback_thread = threading.Thread(
                    target=invoke,
                    args=(candidate,),
                    name=f"rbln-duplicate-callback-{candidate_index}",
                    daemon=True,
                )
                callback_threads.append(callback_thread)
                callback_thread.start()
            for callback_thread in callback_threads:
                callback_thread.join(timeout=1.0)
            callback_delivered.set()

        return real_submit(inputs, recorded_callback)

    monkeypatch.setattr(backend, "submit_async", capture_executor_callback)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        execution_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(1)
        assert fake_rebel.release_call(1)
        execution = execution_future.result(timeout=1.0)

    assert callback_delivered.wait(timeout=1.0)
    assert timing.overlapped.is_set()
    assert len(set(timing.item_threads)) == 2
    assert callback_errors == []
    assert all(not thread.is_alive() for thread in callback_threads)
    winner = execution.outputs["logits"]
    assert any(
        np.array_equal(winner, candidate)
        for candidate in (
            np.array([[0.7, 0.3]], dtype=np.float32),
            np.array([[0.2, 0.8]], dtype=np.float32),
        )
    )
    assert executor.snapshot().duplicate_callbacks == 1
    assert _wait_for_backend_jobs(backend, 0)
    assert executor.snapshot().inflight == 1
    executor.acknowledge(execution)
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True


def test_executor_shutdown_closes_admission_then_rbln_backend_drains(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=1.0,
    )
    permits = _ObservedBoundedPermit(1)
    executor._permits = permits
    executor_drain_wait_entered = threading.Event()
    backend_drain_wait_entered = threading.Event()
    executor._condition = _DrainWaitCondition(
        executor_drain_wait_entered
    )
    backend._condition = _DrainWaitCondition(backend_drain_wait_entered)
    shutdown_results = []
    backend_shutdown_results = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(executor.execute, valid_inputs())
        assert fake_rebel.wait_for_async_calls(1)
        blocked_future = pool.submit(
            executor.execute,
            valid_inputs(),
            1.0,
        )
        assert permits.second_acquire_entered.wait(timeout=1.0)

        shutdown_thread = threading.Thread(
            target=lambda: shutdown_results.append(
                executor.shutdown(timeout=1.0)
            ),
            daemon=True,
        )
        shutdown_thread.start()
        executor_wait_observed = executor_drain_wait_entered.wait(
            timeout=1.0
        )
        if executor_wait_observed:
            with executor._condition:
                assert executor._closed is True
                assert executor.snapshot().inflight == 1

        backend_shutdown_thread = threading.Thread(
            target=lambda: backend_shutdown_results.append(
                backend.shutdown(timeout=1.0)
            ),
            daemon=True,
        )
        backend_shutdown_thread.start()
        backend_wait_observed = backend_drain_wait_entered.wait(
            timeout=1.0
        )
        if backend_wait_observed:
            with backend._condition:
                assert backend._closing is True
                assert tuple(backend._jobs) == ("rbln-1",)
                assert backend_shutdown_results == []

        assert fake_rebel.release_call(1)
        first = first_future.result(timeout=1.0)
        executor.acknowledge(first)
        rejected = blocked_future.result(timeout=1.0)
        assert rejected.error_type == "NativeAsyncShutdown"
        assert rejected.dispatch_token is None
        assert len(fake_rebel.async_run_threads) == 1
        assert backend._next_job_id == 2
        shutdown_thread.join(timeout=1.0)
        backend_shutdown_thread.join(timeout=1.0)

    assert shutdown_results == [True]
    assert backend_shutdown_results == [True]
    assert not shutdown_thread.is_alive()
    assert not backend_shutdown_thread.is_alive()
    assert backend._jobs == {}
    assert executor_wait_observed is True
    assert backend_wait_observed is True


def test_rbln_async_engine_reuses_one_runtime_and_fixed_threads(
    loaded_runtime, fake_rebel, monkeypatch
):
    from core.async_inference import runner as runner_module

    backend = loaded_runtime.create_native_backend()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=loaded_runtime.max_concurrent_workers(),
        completion_timeout_sec=1.0,
    )
    evaluator = _CountingEvaluator()
    engine = InferenceEngine(
        _RblnAsyncLoader(),
        loaded_runtime,
        evaluator,
        runtime_executor=executor,
    )
    config = AsyncInferenceConfig(
        queue_capacity=1,
        worker_count=1,
        max_batch_size=1,
        batch_timeout_ms=0,
        submit_timeout_sec=1.0,
        flush_timeout_sec=2.0,
        min_samples=2,
        max_samples=2,
    )
    captured_async_engines = []
    real_async_engine = runner_module.AsyncInferenceEngine

    def capture_async_engine(*args, **kwargs):
        async_engine = real_async_engine(*args, **kwargs)
        captured_async_engines.append(async_engine)
        return async_engine

    monkeypatch.setattr(
        runner_module,
        "AsyncInferenceEngine",
        capture_async_engine,
    )
    real_thread_start = threading.Thread.start
    started_thread_names = []
    start_lock = threading.Lock()

    def record_thread_start(thread, *args, **kwargs):
        with start_lock:
            started_thread_names.append(thread.name)
        return real_thread_start(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", record_thread_start)
    results = []
    run_errors = []

    def run_engine():
        try:
            results.append(engine.run_async(config, warmup_runs=1))
        except BaseException as exc:
            run_errors.append(exc)

    driver = threading.Thread(
        target=run_engine,
        name="rbln-engine-test-driver",
        daemon=True,
    )
    driver.start()
    assert fake_rebel.wait_for_async_calls(1)
    assert len(fake_rebel.async_runtime_calls) == 1
    assert fake_rebel.release_call(1)

    assert fake_rebel.wait_for_async_calls(2)
    with start_lock:
        starts_at_first_request = Counter(started_thread_names)
    assert fake_rebel.release_call(2)

    assert fake_rebel.wait_for_async_calls(3)
    with start_lock:
        starts_at_second_request = Counter(started_thread_names)
    assert starts_at_second_request == starts_at_first_request
    assert fake_rebel.release_call(3)
    driver.join(timeout=2.0)

    assert not driver.is_alive()
    assert run_errors == []
    assert len(results) == 1
    result = results[0]

    assert result.status is RunStatus.VALID
    assert result.metrics["async_accepted_requests"] == 2
    assert result.metrics["async_completed_requests"] == 2
    assert result.metrics["async_failed_requests"] == 0
    assert result.metrics["Total Samples"] == 2
    assert evaluator.samples == 2
    assert engine.completion.queue.maxsize == config.worker_count
    assert len(captured_async_engines) == 1
    scheduling_engine = captured_async_engines[0]
    assert scheduling_engine.requests.maxsize == config.queue_capacity
    assert scheduling_engine.slots.capacity == config.queue_capacity
    assert starts_at_first_request == Counter(
        {
            "rbln-engine-test-driver": 1,
            "async-completion": 1,
            "async-completion-monitor": 1,
            "async-worker-0": 1,
        }
    )
    with start_lock:
        all_started_threads = Counter(started_thread_names)
    assert all_started_threads == starts_at_second_request + Counter(
        {"async-callback-evaluator_compute-1": 1}
    )
    assert "rbln-native-loop" not in all_started_threads
    assert len(fake_rebel.async_runtime_calls) == 1
    assert fake_rebel.async_run_threads == [backend.owner_thread_ident] * 3
    assert backend.shutdown(timeout=1.0) is True


def test_rbln_four_worker_capability_and_generic_dynamic_batch_gate(
    loaded_runtime, fake_rebel
):
    backend = loaded_runtime.create_native_backend()
    assert loaded_runtime.max_async_inflight == 4
    assert loaded_runtime.max_concurrent_workers() == 4
    native_batch_limit = loaded_runtime.native_async_max_batch_size()
    assert type(native_batch_limit) is int
    assert native_batch_limit == 1

    loader = _RblnAsyncLoader()
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=4,
        completion_timeout_sec=1.0,
    )
    pipeline = InferencePipeline(
        loader,
        loaded_runtime,
        runtime_executor=executor,
    )
    metrics = AsyncMetricsCollector(time.monotonic_ns(), worker_count=4)
    coordinator = CompletionCoordinator(
        pipeline,
        _CountingEvaluator(),
        None,
        metrics,
        queue_capacity=4,
    )
    accepted_config = AsyncInferenceConfig(
        queue_capacity=4,
        worker_count=4,
        max_batch_size=1,
        min_samples=1,
    )

    accepted = AsyncInferenceEngine(
        loaded_runtime,
        pipeline,
        accepted_config,
        coordinator,
        metrics,
        executor=executor,
    )
    assert len(accepted.workers) == 4

    # Native limit enforcement belongs to the Task 6 async-runtime factory.
    # This assertion separately characterizes the generic engine gate for a
    # runtime that does not support dynamic batching.
    with pytest.raises(ValueError, match="does not support dynamic batching"):
        AsyncInferenceEngine(
            loaded_runtime,
            pipeline,
            AsyncInferenceConfig(
                queue_capacity=4,
                worker_count=4,
                max_batch_size=2,
                min_samples=1,
            ),
            coordinator,
            metrics,
            executor=executor,
        )

    assert executor.shutdown(timeout=0.0) is True
    assert backend.shutdown(timeout=1.0) is True

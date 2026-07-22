import gc
import threading
import time
import warnings
import weakref

import numpy as np
import pytest

from rbln_test_utils import fake_rebel, loaded_runtime, valid_inputs


def _wait_for_count(items, count, timeout=1.0):
    event = threading.Event()

    def append(value):
        items.append(value)
        if len(items) >= count:
            event.set()

    return append, event


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

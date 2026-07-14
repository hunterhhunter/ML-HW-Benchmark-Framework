import logging
import queue
import threading
import time
from dataclasses import replace

import numpy as np

from .types import BatchCompletion, EngineState


_STOP = object()
LOGGER = logging.getLogger(__name__)


class _SubmissionClosed(RuntimeError):
    pass


class _AcceptanceMetricsError(RuntimeError):
    pass


class _RequestQueue(queue.Queue):
    def publish(self, request, on_published):
        with self.not_full:
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                raise queue.Full
            queued = replace(request, enqueued_ns=time.monotonic_ns())
            self._put(queued)
            self.unfinished_tasks += 1
            depth = self._qsize()
            try:
                on_published(queued, depth)
            except BaseException:
                self.queue.pop()
                self.unfinished_tasks -= 1
                self.not_full.notify()
                raise
            self.not_empty.notify()
            return queued

    def take(self, on_dequeued, block=True, timeout=None):
        return self._take(
            on_dequeued=on_dequeued,
            block=block,
            timeout=timeout,
        )

    def get_candidate(
        self,
        on_claim,
        on_dequeued,
        block=True,
        timeout=None,
    ):
        return self._take(
            on_dequeued=on_dequeued,
            on_claim=on_claim,
            block=block,
            timeout=timeout,
        )

    def _take(
        self,
        on_dequeued,
        on_claim=None,
        block=True,
        timeout=None,
    ):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise queue.Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                endtime = time.monotonic() + timeout
                while not self._qsize():
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            if item is not _STOP:
                if on_claim is not None:
                    on_claim(item)
                on_dequeued(item, self._qsize())
            self.not_full.notify()
            return item

    def drain_requests(self, on_drained):
        with self.not_full:
            drained = []
            retained = []
            while self._qsize():
                item = self._get()
                if item is _STOP:
                    retained.append(item)
                else:
                    drained.append(item)
            for item in retained:
                self._put(item)
            if drained:
                self.unfinished_tasks -= len(drained)
                if self.unfinished_tasks < 0:
                    raise ValueError("task_done() called too many times")
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                on_drained(0)
                self.not_full.notify_all()
            return drained


class AsyncInferenceEngine:
    def __init__(self, runtime, pipeline, config, coordinator, metrics):
        config.validate()
        runtime_worker_limit = runtime.max_concurrent_workers()
        runtime_batch_limit = runtime.max_dynamic_batch_size()
        if config.worker_count > runtime_worker_limit:
            raise ValueError(
                f"worker_count={config.worker_count} exceeds runtime capability "
                f"{runtime_worker_limit}"
            )
        if config.max_batch_size > 1 and not pipeline.is_static_batched:
            if not runtime.supports_dynamic_batching():
                raise ValueError("runtime does not support dynamic batching")
            if (
                runtime_batch_limit is not None
                and config.max_batch_size > runtime_batch_limit
            ):
                raise ValueError(
                    f"max_batch_size={config.max_batch_size} exceeds runtime "
                    f"capability {runtime_batch_limit}"
                )
        if pipeline.is_llm and config.max_batch_size > 1:
            if not runtime.supports_batch_generation():
                raise ValueError("runtime does not support batch generation")
        completion_capacity = getattr(
            getattr(coordinator, "queue", None),
            "maxsize",
            0,
        )
        if completion_capacity <= 0:
            raise ValueError("completion queue must be strictly bounded")
        if completion_capacity != config.worker_count:
            raise ValueError(
                f"completion queue capacity={completion_capacity} must equal "
                f"worker_count={config.worker_count}"
            )

        self.runtime = runtime
        self.pipeline = pipeline
        self.config = config
        self.runtime_batch_limit = runtime_batch_limit
        self.coordinator = coordinator
        self.metrics = metrics
        self.state = EngineState.CREATED
        self.state_lock = threading.Lock()
        self.state_condition = threading.Condition(self.state_lock)
        self._active_submitters = 0
        self._stop_requested = threading.Event()
        self._completion_monitor_stop = threading.Event()
        self._pending_lock = threading.Lock()
        self._pending_by_worker = {}
        self._control_lock = threading.Lock()
        self._shutdown_started = False

        self.requests = _RequestQueue(maxsize=config.queue_capacity)
        self.slots = threading.BoundedSemaphore(config.queue_capacity)
        self.workers = [
            threading.Thread(
                target=self._worker,
                args=(worker_id,),
                name=f"async-worker-{worker_id}",
                daemon=True,
            )
            for worker_id in range(config.worker_count)
        ]
        self.completion_monitor = threading.Thread(
            target=self._watch_completion,
            name="async-completion-monitor",
            daemon=True,
        )

    def start(self) -> None:
        with self.state_condition:
            if self.state is not EngineState.CREATED:
                raise RuntimeError(f"cannot start engine in {self.state.value}")
            self.state = EngineState.RUNNING
            self.state_condition.notify_all()

        try:
            self.coordinator.start()
            self.completion_monitor.start()
            for worker in self.workers:
                worker.start()
        except BaseException:
            self._mark_failed("worker_shutdown_failed")
            self._stop_requested.set()
            raise

    def submit(self, request, block: bool) -> bool:
        with self.state_condition:
            if self.state is not EngineState.RUNNING:
                raise RuntimeError(f"cannot submit in {self.state.value}")
            self._active_submitters += 1

        submitted = False
        acquired = False
        registered = False
        accepted = False
        rejected = False
        try:
            self.metrics.record_submitted()
            submitted = True
            try:
                self._validate_request(request)
            except (TypeError, ValueError):
                self.metrics.record_rejected("invalid_request")
                rejected = True
                return False
            if block:
                acquired = self.slots.acquire(
                    blocking=True,
                    timeout=self.config.submit_timeout_sec,
                )
            else:
                acquired = self.slots.acquire(blocking=False)
            if not acquired:
                self.metrics.record_queue_full()
                self.metrics.record_rejected("queue_full")
                rejected = True
                if block:
                    self.metrics.add_invalid_reason("queue_submit_timeout")
                return False

            with self.state_condition:
                accepting = self.state is EngineState.RUNNING
            if not accepting:
                self.slots.release()
                acquired = False
                self.metrics.record_rejected("submission_closed")
                rejected = True
                return False

            def commit_registration():
                with self.state_condition:
                    if self.state is not EngineState.RUNNING:
                        raise _SubmissionClosed(
                            f"cannot submit in {self.state.value}"
                        )

                    def record_publication(queued, depth) -> None:
                        claim = self.metrics.claim_acceptance()
                        try:
                            self.metrics.record_accepted(
                                now_ns=queued.enqueued_ns,
                                queue_depth=depth,
                            )
                        except BaseException as exc:
                            committed = self.metrics.finish_acceptance(claim)
                            self.metrics.add_invalid_reason(
                                "counter_invariant_failed"
                            )
                            if committed:
                                LOGGER.exception(
                                    "accepted metrics failed after commit"
                                )
                                return
                            raise _AcceptanceMetricsError(
                                "accepted metrics failed before commit"
                            ) from exc
                        if not self.metrics.finish_acceptance(claim):
                            self.metrics.add_invalid_reason(
                                "counter_invariant_failed"
                            )
                            raise _AcceptanceMetricsError(
                                "accepted metrics did not commit"
                            )

                    return self.requests.publish(
                        request,
                        record_publication,
                    )

            try:
                self.coordinator.register(
                    request,
                    on_registered=commit_registration,
                )
                registered = True
                accepted = True
            except _SubmissionClosed:
                self.slots.release()
                acquired = False
                self.metrics.record_rejected("submission_closed")
                rejected = True
                return False
            except _AcceptanceMetricsError:
                self.slots.release()
                acquired = False
                self.metrics.record_rejected("metrics_unavailable")
                rejected = True
                return False
            except RuntimeError:
                self.slots.release()
                acquired = False
                self.metrics.record_rejected("completion_unavailable")
                rejected = True
                self._mark_failed("completion_thread_failed")
                return False
            except BaseException:
                self.slots.release()
                acquired = False
                self.metrics.record_rejected("registration_failed")
                rejected = True
                raise

            return True
        except BaseException as exc:
            if acquired:
                self.slots.release()
            if registered and not accepted:
                self.coordinator.unregister_rejected(request.request_id)
            if submitted and not accepted and not rejected:
                self.metrics.record_rejected("submission_interrupted")
            elif accepted:
                self._submit_failure(
                    [replace(request, enqueued_ns=time.monotonic_ns())],
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    worker_id=-1,
                    timeout=self.config.flush_timeout_sec,
                )
            raise
        finally:
            with self.state_condition:
                self._active_submitters -= 1
                self.state_condition.notify_all()

    def close_submission(self) -> None:
        with self.state_condition:
            if self.state is EngineState.RUNNING:
                self.state = EngineState.DRAINING
            self.state_condition.notify_all()

    def cancel_queued(self, reason: str) -> int:
        with self._control_lock:
            if self._shutdown_started:
                return 0
        return self._cancel_queued(reason, self.config.flush_timeout_sec)

    def _cancel_queued(self, reason: str, timeout: float) -> int:
        requests = self._drain_request_queue()
        requests.extend(self._claim_all_pending())
        count = len(requests)
        if not count:
            return 0
        try:
            self._submit_failure(
                requests,
                error_type="CancelledError",
                error_message=reason,
                worker_id=-1,
                timeout=timeout,
            )
        except (RuntimeError, TimeoutError):
            self._mark_failed("completion_thread_failed")
        finally:
            del requests
        return count

    def flush(self) -> bool:
        deadline = time.monotonic() + self.config.flush_timeout_sec
        return self._flush_until(deadline)

    def outstanding_request_ids(self):
        return tuple(sorted(self.coordinator.snapshot_outstanding()))

    def shutdown(self) -> bool:
        deadline = time.monotonic() + self.config.flush_timeout_sec
        with self.state_condition:
            if self.state is EngineState.STOPPED:
                return True
            if self.state is EngineState.CREATED:
                raise RuntimeError("cannot shutdown engine before start")
        with self._control_lock:
            self._shutdown_started = True

        self.close_submission()
        flushed = self._flush_until(deadline)
        submitters_stopped = self._wait_for_submitters(deadline)
        ok = flushed and submitters_stopped
        if not flushed:
            self._mark_failed("flush_timeout")
            self._cancel_queued(
                "engine shutdown after flush failure",
                max(0.0, deadline - time.monotonic()),
            )

        self._stop_requested.set()
        stop_enqueued = self._enqueue_stop(deadline)
        ok = stop_enqueued and ok

        if flushed:
            workers_stopped = self._join_workers(deadline)
            ok = workers_stopped and ok
            coordinator_stopped = self.coordinator.stop(
                max(0.0, deadline - time.monotonic())
            )
        else:
            coordinator_stopped = self.coordinator.stop(
                max(0.0, deadline - time.monotonic())
            )
            workers_stopped = self._join_workers(deadline)
            ok = workers_stopped and ok
        ok = coordinator_stopped and ok

        self._stop_completion_monitor()
        self.completion_monitor.join(
            timeout=max(0.0, deadline - time.monotonic())
        )
        if self.completion_monitor.is_alive():
            ok = False

        abandoned = self._drain_request_queue()
        if abandoned:
            ok = False
            del abandoned
        self._discard_stop_tokens()

        with self.state_condition:
            self.state = EngineState.STOPPED if ok else EngineState.FAILED
            self.state_condition.notify_all()
        return ok

    def _flush_until(self, deadline: float) -> bool:
        request_ids = self.coordinator.snapshot_outstanding()
        ok = self.coordinator.wait_for_requests(
            request_ids,
            max(0.0, deadline - time.monotonic()),
        )
        if not ok and self.coordinator.thread_error is not None:
            self._mark_failed("completion_thread_failed")
            abandoned = self._drain_request_queue()
            del abandoned
        return ok

    def _wait_for_submitters(self, deadline: float) -> bool:
        with self.state_condition:
            while self._active_submitters:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.state_condition.wait(timeout=remaining)
        return True

    def _worker(self, worker_id: int) -> None:
        has_pending = False
        owned = []
        consecutive_failures = 0
        try:
            while True:
                stop_after_batch = False
                owned = []
                if has_pending:
                    first = self._take_pending(worker_id)
                    has_pending = False
                    if first is None:
                        continue
                    owned = [first]
                else:
                    first = self.requests.take(self._request_dequeued)
                    if first is _STOP:
                        self._pass_stop_token()
                        return
                    owned = [first]

                batch = [first]
                if (
                    self.config.max_batch_size > 1
                    and not self.pipeline.is_static_batched
                ):
                    deadline_ns = time.monotonic_ns() + int(
                        self.config.batch_timeout_ms * 1_000_000
                    )
                    while (
                        self._dynamic_batch_size(batch)
                        < self.config.max_batch_size
                    ):
                        remaining_sec = (
                            deadline_ns - time.monotonic_ns()
                        ) / 1_000_000_000
                        if remaining_sec <= 0:
                            break
                        try:
                            candidate = self.requests.get_candidate(
                                lambda request: self._publish_pending(
                                    worker_id,
                                    request,
                                ),
                                self._request_dequeued,
                                timeout=remaining_sec,
                            )
                        except queue.Empty:
                            break
                        if candidate is _STOP:
                            self._pass_stop_token()
                            stop_after_batch = True
                            break
                        has_pending = True
                        compatible = not (
                            self._batch_key(candidate) != self._batch_key(first)
                            or self._dynamic_batch_size(batch)
                            + self._request_batch_size(candidate)
                            > self.config.max_batch_size
                        )
                        if not compatible:
                            candidate = None
                            break
                        claimed = self._take_pending(worker_id)
                        has_pending = False
                        if claimed is None:
                            candidate = None
                            break
                        owned.append(claimed)
                        batch.append(claimed)

                collated = {}
                started_ns = None
                actual_batch_size = (
                    sum(request.sample_count for request in batch)
                    if self.pipeline.is_static_batched
                    else self._dynamic_batch_size(batch)
                )
                try:
                    source = (
                        batch[0].sample
                        if self.pipeline.is_static_batched
                        else [item.sample for item in batch]
                    )
                    if batch[0].batch_axis is None:
                        collated = self.pipeline.collate_batch(source)
                    else:
                        collated = self._collate_prebatched(
                            source,
                            batch[0].batch_axis,
                        )
                    runtime_input = self.pipeline.prepare_runtime_input(
                        collated["input"]
                    )
                    started_ns = time.monotonic_ns()
                    invocation = self.pipeline.invoke(runtime_input)
                    finished_ns = time.monotonic_ns()
                    completion = BatchCompletion(
                        requests=tuple(batch),
                        collated=collated,
                        outputs=invocation.outputs,
                        timing_ms=invocation.timing_ms,
                        runtime_started_ns=started_ns,
                        runtime_finished_ns=finished_ns,
                        worker_id=worker_id,
                        batch_size=actual_batch_size,
                        generated_tokens=invocation.generated_tokens,
                    )
                    consecutive_failures = 0
                except Exception as exc:
                    finished_ns = time.monotonic_ns()
                    if started_ns is None:
                        started_ns = finished_ns
                    LOGGER.exception(
                        "async runtime batch failed on worker %s",
                        worker_id,
                    )
                    consecutive_failures += 1
                    completion = BatchCompletion(
                        requests=tuple(batch),
                        collated=collated,
                        outputs=None,
                        timing_ms=None,
                        runtime_started_ns=started_ns,
                        runtime_finished_ns=finished_ns,
                        worker_id=worker_id,
                        batch_size=actual_batch_size,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    if consecutive_failures >= 3:
                        self._mark_failed("request_failed")

                self.metrics.record_worker_busy(
                    worker_id,
                    started_ns,
                    finished_ns,
                    actual_batch_size,
                    sum(request.sample_count for request in batch),
                )
                self.coordinator.submit(
                    completion,
                    timeout=self.config.flush_timeout_sec,
                )
                for _batch_index in range(len(batch)):
                    self.requests.task_done()
                owned = []
                completion = None
                invocation = None
                runtime_input = None
                collated = None
                source = None
                batch = None
                first = None
                candidate = None

                if stop_after_batch or self._stop_requested.is_set():
                    if has_pending:
                        pending_request = self._take_pending(worker_id)
                        has_pending = False
                    else:
                        pending_request = None
                    if pending_request is not None:
                        owned = [pending_request]
                        raise RuntimeError(
                            "worker stopped with a pending accepted request"
                        )
                    return
        except BaseException as exc:
            LOGGER.exception("async worker %s terminated unexpectedly", worker_id)
            failed = list(owned)
            if has_pending:
                pending_request = self._take_pending(worker_id)
                has_pending = False
                if pending_request is not None:
                    failed.append(pending_request)
            for _ in failed:
                self.requests.task_done()
            drained = self._drain_request_queue()
            failed.extend(drained)
            self._stop_requested.set()
            self._mark_failed("request_failed")
            if failed:
                try:
                    self._submit_failure(
                        failed,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        worker_id=worker_id,
                        timeout=self.config.flush_timeout_sec,
                    )
                except (RuntimeError, TimeoutError):
                    self._mark_failed("completion_thread_failed")
            del failed
            del drained

    def _watch_completion(self) -> None:
        with self.coordinator.condition:
            while (
                self.coordinator.thread_error is None
                and not self._completion_monitor_stop.is_set()
            ):
                self.coordinator.condition.wait()
            failed = self.coordinator.thread_error is not None
        if failed:
            self._mark_failed("completion_thread_failed")
            abandoned = self._drain_request_queue()
            del abandoned

    def _stop_completion_monitor(self) -> None:
        self._completion_monitor_stop.set()
        with self.coordinator.condition:
            self.coordinator.condition.notify_all()

    def _mark_failed(self, reason: str) -> None:
        with self.state_condition:
            if self.state is not EngineState.STOPPED:
                self.state = EngineState.FAILED
            self.state_condition.notify_all()
        self.metrics.add_invalid_reason(reason)

    def _request_dequeued(self, _request, depth: int) -> None:
        self.slots.release()
        self.metrics.record_queue_depth(
            depth,
            time.monotonic_ns(),
        )

    def _publish_pending(self, worker_id: int, request) -> None:
        with self._pending_lock:
            if worker_id in self._pending_by_worker:
                raise RuntimeError(f"worker {worker_id} already has a pending request")
            self._pending_by_worker[worker_id] = request

    def _take_pending(self, worker_id: int):
        with self._pending_lock:
            return self._pending_by_worker.pop(worker_id, None)

    def _claim_all_pending(self):
        with self._pending_lock:
            pending = list(self._pending_by_worker.values())
            self._pending_by_worker.clear()
        for _pending_index in range(len(pending)):
            self.requests.task_done()
        return pending

    def _drain_request_queue(self):
        drained = self.requests.drain_requests(
            lambda depth: self.metrics.record_queue_depth(
                depth,
                time.monotonic_ns(),
            )
        )
        for _drained_index in range(len(drained)):
            self.slots.release()
        return drained

    def _submit_failure(
        self,
        requests,
        *,
        error_type: str,
        error_message: str,
        worker_id: int,
        timeout: float,
    ) -> None:
        if not requests:
            return
        now_ns = time.monotonic_ns()
        self.coordinator.submit(
            BatchCompletion(
                requests=tuple(requests),
                collated={},
                outputs=None,
                timing_ms=None,
                runtime_started_ns=now_ns,
                runtime_finished_ns=now_ns,
                worker_id=worker_id,
                batch_size=sum(
                    request.sample_count for request in requests
                ),
                error_type=error_type,
                error_message=error_message,
            ),
            timeout=timeout,
        )

    def _enqueue_stop(self, deadline: float) -> bool:
        try:
            self.requests.put(
                _STOP,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except queue.Full:
            self.metrics.add_invalid_reason("worker_shutdown_failed")
            return False
        return True

    def _pass_stop_token(self) -> None:
        self.requests.task_done()
        try:
            self.requests.put_nowait(_STOP)
        except queue.Full:
            pass

    def _join_workers(self, deadline: float) -> bool:
        ok = True
        for worker in self.workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
            if worker.is_alive():
                ok = False
                self.metrics.add_invalid_reason("worker_shutdown_failed")
        return ok

    def _discard_stop_tokens(self) -> None:
        while True:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                self.slots.release()
            self.requests.task_done()

    def _batch_key(self, request):
        value = request.sample["input"]
        if isinstance(value, dict):
            input_signature = tuple(
                (
                    name,
                    np.asarray(array).dtype.str,
                    self._non_batch_shape(
                        np.asarray(array).shape,
                        request.batch_axis,
                    ),
                    request.batch_axis,
                )
                for name, array in sorted(value.items())
            )
        else:
            array = np.asarray(value)
            input_signature = (
                (
                    self.pipeline.input_name,
                    array.dtype.str,
                    self._non_batch_shape(array.shape, request.batch_axis),
                    request.batch_axis,
                ),
            )

        task = request.task
        if task is None:
            compiled_model = getattr(self.runtime, "compiled_model", None)
            spec = getattr(compiled_model, "spec", None)
            task = getattr(spec, "task", None)
        task = getattr(task, "value", task)

        generation_options = {}
        if self.pipeline.is_llm:
            generation_options.update(
                {
                    "max_new_tokens": self.pipeline.max_new_tokens,
                    "stop_token_ids": self.pipeline.stop_token_ids,
                }
            )
        if request.generation_options:
            generation_options.update(request.generation_options)
        return (
            task,
            self._freeze_option(generation_options),
            input_signature,
        )

    def _validate_request(self, request) -> None:
        sample_count = request.sample_count
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, (int, np.integer))
            or sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        if sample_count > self.config.max_batch_size:
            raise ValueError(
                f"request sample_count={sample_count} exceeds "
                f"max_batch_size={self.config.max_batch_size}"
            )
        if (
            self.runtime_batch_limit is not None
            and sample_count > self.runtime_batch_limit
        ):
            raise ValueError(
                f"request sample_count={sample_count} exceeds runtime "
                f"capability {self.runtime_batch_limit}"
            )

        value = request.sample["input"]
        arrays = value.values() if isinstance(value, dict) else (value,)
        if request.batch_axis is not None:
            for input_value in arrays:
                array = np.asarray(input_value)
                axis = self._normalize_batch_axis(
                    request.batch_axis,
                    array.ndim,
                )
                if array.shape[axis] != sample_count:
                    raise ValueError(
                        "sample_count does not match declared batch-axis "
                        "length"
                    )
        elif self.pipeline.is_static_batched:
            for input_value in arrays:
                array = np.asarray(input_value)
                if array.ndim == 0 or array.shape[0] != sample_count:
                    raise ValueError(
                        "sample_count does not match static batch length"
                    )
        elif sample_count != 1:
            raise ValueError(
                "non-batched requests must have sample_count=1"
            )

    @staticmethod
    def _request_batch_size(request):
        if request.batch_axis is not None:
            return request.sample_count
        return 1

    @classmethod
    def _dynamic_batch_size(cls, requests):
        return sum(cls._request_batch_size(request) for request in requests)

    @staticmethod
    def _collate_prebatched(samples, batch_axis):
        collated = {}
        for key in samples[0]:
            values = [sample[key] for sample in samples]
            if key == "input":
                first_input = values[0]
                if isinstance(first_input, dict):
                    collated[key] = {
                        name: np.concatenate(
                            [np.asarray(value[name]) for value in values],
                            axis=batch_axis,
                        )
                        for name in first_input
                    }
                else:
                    collated[key] = np.concatenate(
                        [np.asarray(value) for value in values],
                        axis=batch_axis,
                    )
                continue

            if all(isinstance(value, np.ndarray) for value in values):
                collated[key] = np.concatenate(values, axis=0)
            elif all(isinstance(value, (list, tuple)) for value in values):
                collated[key] = [
                    item
                    for value in values
                    for item in value
                ]
            else:
                collated[key] = values
        return collated

    @staticmethod
    def _non_batch_shape(shape, batch_axis):
        shape = tuple(shape)
        if batch_axis is None:
            return shape
        axis = AsyncInferenceEngine._normalize_batch_axis(
            batch_axis,
            len(shape),
        )
        return shape[:axis] + shape[axis + 1 :]

    @staticmethod
    def _normalize_batch_axis(batch_axis, rank):
        axis = batch_axis if batch_axis >= 0 else rank + batch_axis
        if axis < 0 or axis >= rank:
            raise ValueError(
                f"batch_axis={batch_axis} is invalid for input rank {rank}"
            )
        return axis

    @classmethod
    def _freeze_option(cls, value):
        if isinstance(value, dict):
            return tuple(
                (name, cls._freeze_option(item))
                for name, item in sorted(value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_option(item) for item in value)
        if isinstance(value, np.ndarray):
            return (
                value.dtype.str,
                tuple(value.shape),
                tuple(value.reshape(-1).tolist()),
            )
        if isinstance(value, np.generic):
            return value.item()
        return value

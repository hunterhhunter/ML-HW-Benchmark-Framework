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


class AsyncInferenceEngine:
    def __init__(self, runtime, pipeline, config, coordinator, metrics):
        config.validate()
        runtime_worker_limit = runtime.max_concurrent_workers()
        if config.worker_count > runtime_worker_limit:
            raise ValueError(
                f"worker_count={config.worker_count} exceeds runtime capability "
                f"{runtime_worker_limit}"
            )
        if config.max_batch_size > 1 and not pipeline.is_static_batched:
            if not runtime.supports_dynamic_batching():
                raise ValueError("runtime does not support dynamic batching")
            runtime_batch_limit = runtime.max_dynamic_batch_size()
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

        self.runtime = runtime
        self.pipeline = pipeline
        self.config = config
        self.coordinator = coordinator
        self.metrics = metrics
        self.state = EngineState.CREATED
        self.state_lock = threading.Lock()
        self.state_condition = threading.Condition(self.state_lock)
        self._active_submitters = 0
        self._stop_requested = threading.Event()
        self._completion_monitor_stop = threading.Event()

        self.requests = queue.Queue(maxsize=config.queue_capacity)
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

            queued = replace(request, enqueued_ns=time.monotonic_ns())

            def commit_registration() -> None:
                with self.state_condition:
                    if self.state is not EngineState.RUNNING:
                        raise _SubmissionClosed(
                            f"cannot submit in {self.state.value}"
                        )
                    self.metrics.record_accepted(
                        now_ns=queued.enqueued_ns,
                        queue_depth=self.requests.qsize() + 1,
                    )
                    self.requests.put_nowait(queued)

            try:
                self.coordinator.register(
                    queued,
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
        return self._cancel_queued(reason, self.config.flush_timeout_sec)

    def _cancel_queued(self, reason: str, timeout: float) -> int:
        requests = self._drain_request_queue()
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

    def shutdown(self) -> bool:
        with self.state_condition:
            if self.state is EngineState.STOPPED:
                return True
            if self.state is EngineState.CREATED:
                raise RuntimeError("cannot shutdown engine before start")

        deadline = time.monotonic() + self.config.flush_timeout_sec
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
        pending = None
        owned = []
        consecutive_failures = 0
        try:
            while True:
                stop_after_batch = False
                first = pending
                pending = None
                owned = []
                if first is None:
                    first = self.requests.get()
                    if first is _STOP:
                        self._pass_stop_token()
                        return
                    owned = [first]
                    self._request_dequeued()
                else:
                    owned = [first]

                batch = [first]
                if (
                    self.config.max_batch_size > 1
                    and not self.pipeline.is_static_batched
                ):
                    deadline_ns = time.monotonic_ns() + int(
                        self.config.batch_timeout_ms * 1_000_000
                    )
                    while len(batch) < self.config.max_batch_size:
                        remaining_sec = (
                            deadline_ns - time.monotonic_ns()
                        ) / 1_000_000_000
                        if remaining_sec <= 0:
                            break
                        try:
                            candidate = self.requests.get(timeout=remaining_sec)
                        except queue.Empty:
                            break
                        if candidate is _STOP:
                            self._pass_stop_token()
                            stop_after_batch = True
                            break
                        owned.append(candidate)
                        self._request_dequeued()
                        if self._batch_key(candidate) != self._batch_key(first):
                            pending = candidate
                            break
                        batch.append(candidate)

                collated = {}
                started_ns = None
                try:
                    source = (
                        batch[0].sample
                        if self.pipeline.is_static_batched
                        else [item.sample for item in batch]
                    )
                    collated = self.pipeline.collate_batch(source)
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
                        batch_size=len(batch),
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
                        batch_size=len(batch),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                    if consecutive_failures >= 3:
                        self._mark_failed("request_failed")

                self.metrics.record_worker_busy(
                    worker_id,
                    started_ns,
                    finished_ns,
                    len(batch),
                    sum(request.sample_count for request in batch),
                )
                self.coordinator.submit(
                    completion,
                    timeout=self.config.flush_timeout_sec,
                )
                for _ in batch:
                    self.requests.task_done()
                owned = [pending] if pending is not None else []

                if stop_after_batch or self._stop_requested.is_set():
                    if pending is not None:
                        raise RuntimeError(
                            "worker stopped with a pending accepted request"
                        )
                    return
        except BaseException as exc:
            LOGGER.exception("async worker %s terminated unexpectedly", worker_id)
            failed = list(owned)
            if pending is not None and all(
                request.request_id != pending.request_id for request in failed
            ):
                failed.append(pending)
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

    def _request_dequeued(self) -> None:
        self.slots.release()
        self.metrics.record_queue_depth(
            self.requests.qsize(),
            time.monotonic_ns(),
        )

    def _drain_request_queue(self):
        drained = []
        while True:
            try:
                item = self.requests.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP:
                self.slots.release()
                drained.append(item)
            self.requests.task_done()
        if drained:
            self.metrics.record_queue_depth(0, time.monotonic_ns())
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
                batch_size=len(requests),
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

    @staticmethod
    def _batch_key(request):
        value = request.sample["input"]
        if isinstance(value, dict):
            return tuple(
                (
                    name,
                    np.asarray(array).dtype.str,
                    tuple(np.asarray(array).shape),
                )
                for name, array in sorted(value.items())
            )
        array = np.asarray(value)
        return (array.dtype.str, tuple(array.shape))

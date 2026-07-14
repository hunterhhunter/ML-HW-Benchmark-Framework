import logging
import queue
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from .completion import _reconcile_registration_internal
from .metrics import (
    _accounting_outcome_internal,
    _allocate_attempt_token_internal,
    _commit_acceptance_internal,
    _record_queue_sequence_allocated,
    _record_queue_sequence_failed_internal,
    _record_rejected_internal,
    _resolve_accounting_internal,
)
from .types import BatchCompletion, EngineState


_STOP = object()
_CLOSED = object()
_OUTCOME_UNKNOWN = object()
_LEASE_UNKNOWN = object()
LOGGER = logging.getLogger(__name__)


class _SubmissionClosed(RuntimeError):
    pass


def _exact_int(value) -> int:
    converted = int(value)
    return converted if type(converted) is int else int(str(converted))


def _query_accounting_outcome(metrics, attempt_token):
    try:
        return _accounting_outcome_internal(metrics, attempt_token)
    except BaseException:
        return _OUTCOME_UNKNOWN


def _slot_membership_internal(pool, attempt_token):
    with pool.condition:
        return attempt_token in pool._held


def _query_slot_membership(pool, attempt_token):
    try:
        return _slot_membership_internal(pool, attempt_token)
    except BaseException:
        return _LEASE_UNKNOWN


class _SlotLeasePool:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.condition = threading.Condition()
        self._held = set()
        self._next_compatibility_token = -1
        self._local = threading.local()

    @property
    def held_count(self) -> int:
        with self.condition:
            return len(self._held)

    def prepare_lease(self, token: int) -> None:
        self._local.requested_token = int(token)

    def clear_prepared_lease(self) -> None:
        if hasattr(self._local, "requested_token"):
            del self._local.requested_token

    def _acquire_lease_locked(
        self,
        token: int,
        blocking: bool,
        timeout: float | None,
    ) -> bool:
        if token in self._held:
            raise RuntimeError(f"slot lease {token} is already held")
        if not blocking:
            if timeout is not None:
                raise ValueError("can't specify timeout for non-blocking acquire")
            if len(self._held) >= self.capacity:
                return False
        elif timeout is None:
            while len(self._held) >= self.capacity:
                self.condition.wait()
        else:
            deadline = time.monotonic() + max(0.0, timeout)
            while len(self._held) >= self.capacity:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.condition.wait(timeout=remaining)
        self._held.add(token)
        return True

    def acquire_lease(
        self,
        token: int,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        token = int(token)
        with self.condition:
            return self._acquire_lease_locked(token, blocking, timeout)

    def acquire(
        self,
        blocking: bool = True,
        timeout: float | None = None,
    ) -> bool:
        requested = getattr(self._local, "requested_token", None)
        if requested is not None:
            return self.acquire_lease(requested, blocking, timeout)
        with self.condition:
            token = self._next_compatibility_token
            self._next_compatibility_token -= 1
            acquired = self._acquire_lease_locked(token, blocking, timeout)
        if acquired:
            stack = getattr(self._local, "compatibility_tokens", None)
            if stack is None:
                stack = []
                self._local.compatibility_tokens = stack
            stack.append(token)
        return acquired

    def _release_lease_locked(self, token: int) -> bool:
        if token not in self._held:
            return False
        self._held.remove(token)
        self.condition.notify()
        return True

    def release_lease(self, token: int | None) -> bool:
        with self.condition:
            if token is None:
                token = next(
                    (item for item in self._held if item < 0),
                    None,
                )
                if token is None:
                    return False
            else:
                token = int(token)
            return self._release_lease_locked(token)

    def release(self) -> None:
        stack = getattr(self._local, "compatibility_tokens", None)
        if not stack:
            raise ValueError("slot pool released too many times")
        token = stack.pop()
        if not self.release_lease(token):
            raise ValueError("slot lease is no longer held")

    def contains(self, token: int) -> bool:
        token = int(token)
        with self.condition:
            return token in self._held


@dataclass(frozen=True)
class _QueueTransition:
    depth: int
    now_ns: int
    sequence: int


@dataclass
class _SubmissionTransaction:
    attempt_token: int
    request_id: int
    reservation_owned: bool = False
    terminal_state: str | None = None
    rejection_reason: str | None = None
    queued_request: object | None = None
    coordinator_committed: bool = False
    registry_removed: bool = False
    reservation_aborted: bool = False
    queue_publication_uncertain: bool = False
    queue_item_preserved: bool = False
    queue_sequences: tuple[int, ...] = ()
    queue_visible: bool = False
    queue_visibility_notified: bool = False
    recovery_unresolved: bool = False


class _RequestQueue(queue.Queue):
    def __init__(self, maxsize=0, transition_metrics=None):
        super().__init__(maxsize=maxsize)
        self._transition_sequence = 0
        self._transition_metrics = transition_metrics
        self._closed = False
        self._prepared_entries = {}

    def _entry_state(self, item):
        entry = self._prepared_entries.get(id(item))
        if entry is None or entry[0] is not item:
            return None
        return entry[1]

    def _set_entry_state(self, item, state) -> None:
        self._prepared_entries[id(item)] = (item, state)

    def _clear_entry_state(self, item) -> None:
        entry = self._prepared_entries.get(id(item))
        if entry is not None and entry[0] is item:
            self._prepared_entries.pop(id(item), None)

    def _head_is_visible(self) -> bool:
        if not self._qsize():
            return False
        item = self.queue[0]
        return item is _STOP or self._entry_state(item) is None

    def _logical_depth(self) -> int:
        depth = 0
        for item in self.queue:
            if item is _STOP:
                continue
            state = self._entry_state(item)
            if state is None or state == "accepted_prepared":
                depth += 1
        return depth

    @property
    def closed(self):
        with self.mutex:
            return self._closed

    def close(self) -> None:
        with self.mutex:
            self._closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()

    def _capture_transition(
        self,
        depth: int,
        now_ns: int | None = None,
        *,
        record_allocation: bool = True,
    ):
        self._transition_sequence += 1
        if record_allocation and self._transition_metrics is not None:
            _record_queue_sequence_allocated(
                self._transition_metrics,
                self._transition_sequence,
            )
        return _QueueTransition(
            depth=depth,
            now_ns=time.monotonic_ns() if now_ns is None else now_ns,
            sequence=self._transition_sequence,
        )

    def publish(self, request):
        return self._publish(request, acceptance_metrics=None)

    def publish_accepted(self, request, metrics, transaction):
        return self._publish(
            request,
            acceptance_metrics=metrics,
            attempt_token=transaction.attempt_token,
            submission_transaction=transaction,
        )

    def _publish(
        self,
        request,
        acceptance_metrics,
        attempt_token=None,
        submission_transaction=None,
    ):
        with self.not_full:
            if self._closed:
                raise _SubmissionClosed("request queue is closed")
            if self.maxsize > 0 and self._qsize() >= self.maxsize:
                raise queue.Full
            queued = replace(request, enqueued_ns=time.monotonic_ns())
            unfinished_before = self.unfinished_tasks
            sequence_before = self._transition_sequence
            try:
                self._put(queued)
                if acceptance_metrics is not None:
                    self._set_entry_state(queued, "prepared")
                if submission_transaction is not None:
                    submission_transaction.queued_request = queued
                    submission_transaction.queue_item_preserved = True
                self.unfinished_tasks += 1
                transition = self._capture_transition(
                    (
                        self._logical_depth()
                        if acceptance_metrics is None
                        else self._logical_depth() + 1
                    ),
                    now_ns=queued.enqueued_ns,
                    record_allocation=acceptance_metrics is None,
                )
                if acceptance_metrics is not None:
                    _commit_acceptance_internal(
                        acceptance_metrics,
                        queued.enqueued_ns,
                        transition.depth,
                        queue_transition=transition,
                        attempt_token=attempt_token,
                        request_id=queued.request_id,
                    )
                    self._set_entry_state(queued, "accepted_prepared")
                else:
                    self.not_empty.notify()
                return queued, transition
            except BaseException:
                outcome = None
                if acceptance_metrics is not None:
                    outcome = _query_accounting_outcome(
                        acceptance_metrics,
                        attempt_token,
                    )
                queued_present = any(item is queued for item in self.queue)
                sequences = tuple(
                    range(sequence_before + 1, self._transition_sequence + 1)
                )
                if outcome is _OUTCOME_UNKNOWN or outcome == "accepted":
                    if queued_present and acceptance_metrics is not None:
                        self._set_entry_state(
                            queued,
                            (
                                "accepted_prepared"
                                if outcome == "accepted"
                                else "prepared"
                            ),
                        )
                    if (
                        queued_present
                        and self.unfinished_tasks == unfinished_before
                    ):
                        self.unfinished_tasks += 1
                    if submission_transaction is not None:
                        try:
                            submission_transaction.queued_request = (
                                queued if queued_present else None
                            )
                            submission_transaction.queue_publication_uncertain = True
                            submission_transaction.queue_item_preserved = queued_present
                            submission_transaction.queue_sequences = sequences
                        except BaseException:
                            pass
                    if outcome == "accepted":
                        try:
                            _resolve_accounting_internal(acceptance_metrics)
                        except BaseException:
                            pass
                    raise
                for index in range(len(self.queue) - 1, -1, -1):
                    if self.queue[index] is queued:
                        del self.queue[index]
                        break
                self._clear_entry_state(queued)
                self.unfinished_tasks = unfinished_before
                if submission_transaction is not None:
                    try:
                        submission_transaction.queued_request = queued
                        submission_transaction.queue_publication_uncertain = bool(
                            sequences
                        )
                        submission_transaction.queue_item_preserved = False
                        submission_transaction.queue_sequences = sequences
                    except BaseException:
                        pass
                failure_metrics = acceptance_metrics or self._transition_metrics
                if failure_metrics is not None:
                    evidence_complete = True
                    for sequence in sequences:
                        try:
                            _record_queue_sequence_failed_internal(
                                failure_metrics,
                                sequence,
                            )
                        except BaseException:
                            evidence_complete = False
                            break
                    if evidence_complete and submission_transaction is not None:
                        submission_transaction.queued_request = None
                        submission_transaction.queue_publication_uncertain = False
                        submission_transaction.queue_sequences = ()
                try:
                    self.not_full.notify()
                except BaseException:
                    pass
                raise

    def rollback_uncertain_publication(self, transaction) -> bool:
        with self.not_full:
            queued = transaction.queued_request
            removed = False
            if transaction.queue_item_preserved:
                for index in range(len(self.queue) - 1, -1, -1):
                    if self.queue[index] is queued:
                        del self.queue[index]
                        self._clear_entry_state(queued)
                        removed = True
                        break
            if transaction.queue_item_preserved and not removed:
                return False
            if removed:
                self.unfinished_tasks -= 1
                if self.unfinished_tasks < 0:
                    raise ValueError("task_done() called too many times")
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                transaction.queue_item_preserved = False
                self.not_full.notify()
                if self._head_is_visible():
                    self.not_empty.notify_all()
            for sequence in transaction.queue_sequences:
                _record_queue_sequence_failed_internal(
                    self._transition_metrics,
                    sequence,
                )
            transaction.queued_request = None
            transaction.queue_publication_uncertain = False
            transaction.queue_sequences = ()
            self.not_full.notify()
            return True

    def restore_uncertain_visibility(self, transaction) -> None:
        with self.not_empty:
            if not transaction.queue_visible:
                queued = transaction.queued_request
                if not any(item is queued for item in self.queue):
                    raise RuntimeError("accepted queue ownership missing")
                self._clear_entry_state(queued)
                transaction.queue_visible = True
            if not transaction.queue_visibility_notified:
                self.not_empty.notify()
                transaction.queue_visibility_notified = True
            transaction.queue_publication_uncertain = False
            transaction.queue_item_preserved = False
            transaction.queue_sequences = ()

    def take(self, block=True, timeout=None):
        return self._take(
            block=block,
            timeout=timeout,
        )

    def get_candidate(
        self,
        on_claim,
        block=True,
        timeout=None,
    ):
        return self._take(
            on_claim=on_claim,
            block=block,
            timeout=timeout,
        )

    def _take(
        self,
        on_claim=None,
        block=True,
        timeout=None,
    ):
        with self.not_empty:
            if not block:
                if not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    raise queue.Empty
            elif timeout is None:
                while not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                endtime = time.monotonic() + timeout
                while not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            if item is _STOP:
                if self._head_is_visible():
                    self.not_empty.notify_all()
                self.unfinished_tasks -= 1
                if self.unfinished_tasks < 0:
                    raise ValueError("task_done() called too many times")
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                transition = None
            else:
                self._clear_entry_state(item)
                if self._head_is_visible():
                    self.not_empty.notify_all()
                transition = self._capture_transition(self._logical_depth())
                if on_claim is not None:
                    on_claim(item)
            self.not_full.notify()
            return item, transition

    def drain_requests(self):
        with self.not_full:
            drained = []
            retained = []
            while self._qsize():
                item = self._get()
                if item is _STOP or self._entry_state(item) is not None:
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
                transition = self._capture_transition(self._logical_depth())
                self.not_full.notify_all()
            else:
                transition = None
            return drained, transition

    def discard_stop_tokens(self) -> None:
        with self.not_full:
            retained = []
            discarded = 0
            while self._qsize():
                item = self._get()
                if item is _STOP:
                    discarded += 1
                else:
                    retained.append(item)
            for item in retained:
                self._put(item)
            if discarded:
                self.unfinished_tasks -= discarded
                if self.unfinished_tasks < 0:
                    raise ValueError("task_done() called too many times")
                if self.unfinished_tasks == 0:
                    self.all_tasks_done.notify_all()
                self.not_full.notify_all()


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
        self._shutdown_terminal = False
        self._submission_transactions = {}
        self._unresolved_submissions = {}

        self.requests = _RequestQueue(
            maxsize=config.queue_capacity,
            transition_metrics=metrics,
        )
        self._slot_pool = _SlotLeasePool(config.queue_capacity)
        self.slots = self._slot_pool
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
        identity_error = None
        try:
            request_id = _exact_int(request.request_id)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            request_id = -1
            identity_error = exc
        attempt_token = _allocate_attempt_token_internal(self.metrics)
        with self.state_condition:
            if self.state is not EngineState.RUNNING:
                raise RuntimeError(f"cannot submit in {self.state.value}")
            self._active_submitters += 1

        submitted = False
        acquired = False
        transaction = None
        accepted = False
        rejected = False
        recovery_unknown = False
        try:
            self.metrics.record_submitted()
            submitted = True
            if identity_error is not None:
                _record_rejected_internal(
                    self.metrics,
                    "invalid_request",
                    attempt_token=attempt_token,
                    request_id=request_id,
                )
                rejected = True
                return False
            request = replace(
                request,
                request_id=request_id,
                submission_token=attempt_token,
            )
            try:
                self._validate_request(request)
            except (TypeError, ValueError):
                _record_rejected_internal(
                    self.metrics,
                    "invalid_request",
                    attempt_token=attempt_token,
                    request_id=getattr(request, "request_id", -1),
                )
                rejected = True
                return False
            self._slot_pool.prepare_lease(attempt_token)
            if block:
                try:
                    acquired = self.slots.acquire(
                        blocking=True,
                        timeout=self.config.submit_timeout_sec,
                    )
                finally:
                    self._slot_pool.clear_prepared_lease()
            else:
                try:
                    acquired = self.slots.acquire(blocking=False)
                finally:
                    self._slot_pool.clear_prepared_lease()
            if not acquired:
                self.metrics.record_queue_full()
                _record_rejected_internal(
                    self.metrics,
                    "queue_full",
                    attempt_token=attempt_token,
                    request_id=request.request_id,
                )
                rejected = True
                if block:
                    self.metrics.add_invalid_reason("queue_submit_timeout")
                return False

            try:
                transaction = _SubmissionTransaction(
                    attempt_token,
                    request.request_id,
                )
                with self.state_condition:
                    if self.state is not EngineState.RUNNING:
                        raise _SubmissionClosed(
                            f"cannot submit in {self.state.value}"
                        )
                    with self.coordinator.condition:
                        self.coordinator.reserve_registration(
                            request,
                            attempt_token=attempt_token,
                        )
                    transaction.reservation_owned = True
                    self._submission_transactions[
                        request.request_id
                    ] = transaction
            except _SubmissionClosed:
                _record_rejected_internal(
                    self.metrics,
                    "submission_closed",
                    attempt_token=attempt_token,
                    request_id=request.request_id,
                )
                self._refresh_reservation_ownership(transaction)
                self._complete_rejected_submission(transaction)
                acquired = False
                rejected = True
                return False
            except RuntimeError:
                _record_rejected_internal(
                    self.metrics,
                    "completion_unavailable",
                    attempt_token=attempt_token,
                    request_id=request.request_id,
                )
                self._refresh_reservation_ownership(transaction)
                self._complete_rejected_submission(transaction)
                acquired = False
                rejected = True
                self._mark_failed("completion_thread_failed")
                return False

            try:
                self.metrics.preflight_acceptance(request)
            except Exception:
                newly_rejected = self._reject_submission(
                    transaction,
                    "metrics_unavailable",
                )
                acquired = False
                if newly_rejected:
                    self.metrics.add_invalid_reason("metrics_unavailable")
                rejected = True
                return False

            publication_transition = None

            try:
                with self.state_condition:
                    if (
                        self.state is not EngineState.RUNNING
                        or transaction.terminal_state is not None
                        or self._submission_transactions.get(
                            request.request_id
                        )
                        is not transaction
                    ):
                        raise _SubmissionClosed(
                            f"cannot submit in {self.state.value}"
                        )
                    with self.coordinator.condition:
                        self.coordinator._validate_registration_locked(
                            request.request_id,
                            transaction.attempt_token,
                        )
                        (
                            queued,
                            publication_transition,
                        ) = self.requests.publish_accepted(
                            request,
                            self.metrics,
                            transaction,
                        )
                        self.coordinator._commit_registration_locked(
                            queued,
                            transaction.attempt_token,
                        )
                        transaction.coordinator_committed = True
                self._complete_accepted_submission(transaction)
                acquired = False
                accepted = True
            except _SubmissionClosed:
                self._reject_submission(transaction, "submission_closed")
                acquired = False
                rejected = True
                return False
            except RuntimeError:
                self._reject_submission(transaction, "completion_unavailable")
                acquired = False
                rejected = True
                self._mark_failed("completion_thread_failed")
                return False

            return True
        except BaseException as exc:
            if transaction is not None and not accepted:
                recovery_unknown = True
                for _recovery_attempt in range(2):
                    try:
                        recovery = self._recover_submission(transaction)
                    except BaseException:
                        continue
                    if recovery is _OUTCOME_UNKNOWN:
                        continue
                    accepted, rejected = recovery
                    recovery_unknown = False
                    break
                if recovery_unknown:
                    self._preserve_unresolved_submission(transaction)
                acquired = False
            else:
                lease_membership = _query_slot_membership(
                    self._slot_pool,
                    attempt_token,
                )
                if lease_membership is _LEASE_UNKNOWN:
                    transaction = _SubmissionTransaction(
                        attempt_token,
                        request_id,
                    )
                    with self.state_condition:
                        self._submission_transactions.setdefault(
                            request_id,
                            transaction,
                        )
                    self._preserve_unresolved_submission(transaction)
                    recovery_unknown = True
                elif lease_membership:
                    try:
                        self._slot_pool.release_lease(attempt_token)
                    except BaseException:
                        try:
                            self._slot_pool.release_lease(attempt_token)
                        except BaseException:
                            pass
            if (
                submitted
                and not accepted
                and not rejected
                and not recovery_unknown
            ):
                try:
                    _record_rejected_internal(
                        self.metrics,
                        "submission_interrupted",
                        attempt_token=attempt_token,
                        request_id=request_id,
                    )
                except BaseException:
                    pass
            elif accepted:
                try:
                    self._submit_failure(
                        [replace(request, enqueued_ns=time.monotonic_ns())],
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        worker_id=-1,
                        timeout=self.config.flush_timeout_sec,
                    )
                except BaseException:
                    pass
            raise
        finally:
            with self.state_condition:
                self._active_submitters -= 1
                self.state_condition.notify_all()

    def _recover_submission(self, transaction):
        outcome = _query_accounting_outcome(
            self.metrics,
            transaction.attempt_token,
        )
        if outcome is _OUTCOME_UNKNOWN:
            return _OUTCOME_UNKNOWN
        if outcome == "accepted":
            self._complete_accepted_submission(transaction)
            return True, False
        if outcome == "rejected":
            self._complete_rejected_submission(transaction)
            return False, True
        if transaction.queue_publication_uncertain:
            if not self.requests.rollback_uncertain_publication(transaction):
                return _OUTCOME_UNKNOWN
        newly_rejected = self._reject_submission(
            transaction,
            "submission_interrupted",
        )
        if not newly_rejected and transaction.terminal_state is None:
            _record_rejected_internal(
                self.metrics,
                "submission_interrupted",
                attempt_token=transaction.attempt_token,
                request_id=transaction.request_id,
            )
            self._refresh_reservation_ownership(transaction)
            self._complete_rejected_submission(transaction)
        return False, transaction.terminal_state == "rejected"

    def _preserve_unresolved_submission(self, transaction) -> None:
        with self.state_condition:
            transaction.recovery_unresolved = True
            self._unresolved_submissions[
                transaction.attempt_token
            ] = transaction
            self.state = EngineState.FAILED
            self.state_condition.notify_all()
        try:
            self.metrics.add_invalid_reason("metrics_unavailable")
        except BaseException:
            pass

    def _complete_accepted_submission(self, transaction) -> None:
        if not transaction.coordinator_committed:
            with self.coordinator.condition:
                try:
                    _reconcile_registration_internal(
                        self.coordinator,
                        transaction.queued_request,
                        transaction.attempt_token,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "accepted registration ownership missing"
                    ) from exc
                transaction.coordinator_committed = True
        with self.state_condition:
            transaction.reservation_owned = False
        self.requests.restore_uncertain_visibility(transaction)
        self._mark_submission_terminal(transaction, "accepted")
        self._remove_submission_transaction(transaction)

    def _refresh_reservation_ownership(self, transaction) -> None:
        with self.coordinator.condition:
            transaction.reservation_owned = bool(
                self.coordinator._reservation_matches_locked(
                    transaction.request_id,
                    transaction.attempt_token,
                )
            )

    def _mark_submission_terminal(self, transaction, terminal_state) -> None:
        with self.state_condition:
            if transaction.terminal_state is None:
                transaction.terminal_state = terminal_state
            elif transaction.terminal_state != terminal_state:
                raise RuntimeError("submission transaction outcome mismatch")
            self.state_condition.notify_all()

    def _remove_submission_transaction(self, transaction) -> None:
        if transaction.registry_removed:
            return
        with self.state_condition:
            current = self._submission_transactions.get(transaction.request_id)
            if current is transaction:
                self._submission_transactions.pop(transaction.request_id, None)
            if (
                self._unresolved_submissions.get(transaction.attempt_token)
                is transaction
            ):
                self._unresolved_submissions.pop(
                    transaction.attempt_token,
                    None,
                )
            transaction.registry_removed = True
            self.state_condition.notify_all()

    def _release_slot_once(self, transaction) -> None:
        self._slot_pool.release_lease(transaction.attempt_token)

    def _complete_rejected_submission(self, transaction) -> None:
        first_error = None
        if transaction.reservation_owned and not transaction.reservation_aborted:
            try:
                self.coordinator.abort_registration(
                    transaction.request_id,
                    expected_token=transaction.attempt_token,
                )
            except BaseException as exc:
                first_error = exc
                with self.coordinator.condition:
                    reservation_remains = (
                        self.coordinator._reservation_matches_locked(
                            transaction.request_id,
                            transaction.attempt_token,
                        )
                    )
                if reservation_remains:
                    try:
                        self.coordinator.abort_registration(
                            transaction.request_id,
                            expected_token=transaction.attempt_token,
                        )
                    except BaseException:
                        pass
            with self.coordinator.condition:
                reservation_remains = (
                    self.coordinator._reservation_matches_locked(
                        transaction.request_id,
                        transaction.attempt_token,
                    )
                )
            if not reservation_remains:
                transaction.reservation_aborted = True
                transaction.reservation_owned = False
        try:
            self._release_slot_once(transaction)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            lease_membership = _query_slot_membership(
                self._slot_pool,
                transaction.attempt_token,
            )
            if lease_membership is _LEASE_UNKNOWN:
                raise first_error
            if lease_membership:
                try:
                    self._release_slot_once(transaction)
                except BaseException:
                    pass
        try:
            self._mark_submission_terminal(transaction, "rejected")
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            try:
                self._mark_submission_terminal(transaction, "rejected")
            except BaseException:
                pass
        try:
            self._remove_submission_transaction(transaction)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            try:
                self._remove_submission_transaction(transaction)
            except BaseException:
                pass
        if first_error is not None:
            raise first_error

    def _reject_submission(self, transaction, reason: str) -> bool:
        if transaction is None:
            return False
        with self.state_condition:
            if transaction.terminal_state is not None:
                return False
            current = self._submission_transactions.get(transaction.request_id)
            if current is not transaction:
                return False
        try:
            _record_rejected_internal(
                self.metrics,
                reason,
                attempt_token=transaction.attempt_token,
                request_id=transaction.request_id,
            )
        except BaseException:
            outcome = _query_accounting_outcome(
                self.metrics,
                transaction.attempt_token,
            )
            if outcome is _OUTCOME_UNKNOWN:
                raise
            if outcome != "rejected":
                _record_rejected_internal(
                    self.metrics,
                    reason,
                    attempt_token=transaction.attempt_token,
                    request_id=transaction.request_id,
                )
            _resolve_accounting_internal(self.metrics)
        transaction.rejection_reason = reason
        self._complete_rejected_submission(transaction)
        return True

    def _cancel_preflight_submissions(self) -> None:
        with self.state_condition:
            transactions = list(self._submission_transactions.values())
        for transaction in transactions:
            self._reject_submission(transaction, "submission_closed")

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
        with self.state_condition:
            unknown = {
                transaction.request_id
                for transaction in self._unresolved_submissions.values()
                if transaction.recovery_unresolved
            }
        return tuple(
            sorted(unknown.union(self.coordinator.snapshot_outstanding()))
        )

    def shutdown(self) -> bool:
        deadline = time.monotonic() + self.config.flush_timeout_sec
        with self.state_condition:
            if self.state is EngineState.STOPPED:
                return True
            if self.state is EngineState.CREATED:
                raise RuntimeError("cannot shutdown engine before start")
        with self._control_lock:
            self._shutdown_started = True
            self._shutdown_terminal = False

        self.close_submission()
        flushed = self._flush_until(deadline)
        submitters_stopped = self._wait_for_submitters(deadline)
        ok = flushed and submitters_stopped
        if not submitters_stopped:
            self._cancel_preflight_submissions()
            self.metrics.add_invalid_reason("metrics_unavailable")
        if not flushed:
            self._mark_failed("flush_timeout")
            self._cancel_queued(
                "engine shutdown after flush failure",
                max(0.0, deadline - time.monotonic()),
            )

        self._stop_requested.set()
        with self._control_lock:
            self._shutdown_terminal = True
            stop_enqueued = self._enqueue_stop(deadline)
            self.requests.close()
        if not stop_enqueued:
            self.metrics.add_invalid_reason("worker_shutdown_failed")
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
            if any(
                transaction.recovery_unresolved
                for transaction in self._unresolved_submissions.values()
            ):
                ok = False
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
                    first, transition = self.requests.take()
                    if first is _CLOSED:
                        return
                    if first is _STOP:
                        self._pass_stop_token()
                        return
                    owned = [first]
                    self._request_dequeued(first, transition)

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
                            candidate, transition = self.requests.get_candidate(
                                lambda request: self._publish_pending(
                                    worker_id,
                                    request,
                                ),
                                timeout=remaining_sec,
                            )
                        except queue.Empty:
                            break
                        if candidate is _STOP or candidate is _CLOSED:
                            if candidate is _STOP:
                                self._pass_stop_token()
                            stop_after_batch = True
                            break
                        has_pending = True
                        self._request_dequeued(candidate, transition)
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

    def _request_dequeued(self, request, transition: _QueueTransition) -> None:
        self._slot_pool.release_lease(request.submission_token)
        try:
            self.metrics.record_queue_depth(
                transition.depth,
                transition.now_ns,
                sequence=transition.sequence,
            )
        except BaseException:
            self.metrics.record_queue_depth_failure(transition.sequence)
            raise

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
        drained, transition = self.requests.drain_requests()
        for request in drained:
            self._slot_pool.release_lease(request.submission_token)
        if transition is not None:
            try:
                self.metrics.record_queue_depth(
                    transition.depth,
                    transition.now_ns,
                    sequence=transition.sequence,
                )
            except BaseException:
                self.metrics.record_queue_depth_failure(transition.sequence)
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
        if self.requests.closed:
            return True
        try:
            self.requests.put(
                _STOP,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except queue.Full:
            return False
        return True

    def _pass_stop_token(self) -> None:
        with self._control_lock:
            if self._shutdown_terminal:
                return
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
        self.requests.discard_stop_tokens()

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

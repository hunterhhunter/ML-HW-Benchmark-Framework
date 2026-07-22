import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from ..runtime_executor import (
    BlockingRuntimeExecutor,
    NativeAsyncRuntimeExecutor,
)
from .completion import (
    _COORDINATOR_RUNNING,
    _TERMINAL_PENDING,
    _reconcile_registration_internal,
)
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
_WAKE = object()
_OUTCOME_UNKNOWN = object()
_LEASE_UNKNOWN = object()
LOGGER = logging.getLogger(__name__)


class _SubmissionClosed(RuntimeError):
    pass


class _DrainDeadlineExceeded(TimeoutError):
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


@dataclass(frozen=True)
class _TransitionAllocation:
    transition: _QueueTransition
    evidence_recorded: bool = False


@dataclass(frozen=True)
class _TransitionState:
    next_sequence: int = 1
    allocations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allocations",
            MappingProxyType(dict(self.allocations)),
        )


@dataclass(eq=False)
class _QueueEntry:
    payload: object
    task_token: object = field(default_factory=object)
    state: object | None = None
    task_balanced: bool = False
    publication_committed: bool = False


@dataclass(frozen=True)
class _QueueOperationReservation:
    operation_key: object
    operation: object


@dataclass(eq=False)
class _DequeueOperation:
    operation_key: object
    worker_id: int | None
    entry: _QueueEntry
    request_id: int
    attempt_token: int | None
    transition_depth: int
    transition: _QueueTransition | None = None
    reservation_committed: bool = False
    physical_removed: bool = False
    prepared_state_cleared: bool = False
    handoff_committed: bool = False
    handoff_recovered: bool = False
    slot_released: bool = False
    transition_delivered: bool = False
    transition_failed: bool = False
    task_balanced: bool = False
    pending_owned_cleared: bool = False
    completion_operation_key: object | None = None
    completion_handoff_committed: bool = False
    stage_lock: object = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @property
    def request(self):
        return self.entry.payload

    @property
    def task_token(self):
        return self.entry.task_token


@dataclass(eq=False)
class _DrainOperation:
    operation_key: object
    entries: tuple[_QueueEntry, ...]
    request_ids: tuple[int, ...]
    attempt_tokens: tuple[int | None, ...]
    transition: _QueueTransition
    reservation_committed: bool = False
    physical_removed: bool = False
    prepared_states_cleared: bool = False
    task_balanced: bool = False
    released_indexes: set[int] = field(default_factory=set)
    transition_delivered: bool = False
    transition_failed: bool = False
    failure_completion_delivered: bool = False
    cancellation_completed: bool = False
    completion_operation_key: object = field(default_factory=object)
    cancellation_error_type: str | None = None
    cancellation_error_message: str | None = None
    cancellation_completion: BatchCompletion | None = None
    stage_lock: object = field(
        default_factory=threading.RLock,
        repr=False,
    )

    @property
    def requests(self):
        return tuple(entry.payload for entry in self.entries)

    @property
    def task_tokens(self):
        return tuple(entry.task_token for entry in self.entries)


@dataclass(eq=False)
class _StopOperation:
    operation_key: object
    worker_id: int | None
    entry: _QueueEntry
    physical_removed: bool = False
    task_balanced: bool = False


@dataclass(eq=False)
class _StopPublicationOperation:
    operation_key: object
    entry: _QueueEntry


@dataclass(eq=False)
class _CompatibilityOperation:
    entry: _QueueEntry
    operation_key: object = field(default_factory=object)
    reservation_committed: bool = False
    physical_removed: bool = False
    reservation_cleared: bool = False
    returned: bool = False
    task_balanced: bool = False
    retired: bool = False


@dataclass(eq=False)
class _CancellationOperation:
    requests: tuple
    completion_operation_key: object
    drain_operation_key: object | None
    error_type: str
    error_message: str
    completion: BatchCompletion
    active_cancellation_generation: object | None = None
    active_drain_operation_key: object | None = None
    active_drain_generation: object | None = None
    drain_operation: _DrainOperation | None = None
    phase: str = "ACTIVE_PRODUCER"
    retirement_error_type: str | None = None
    retirement_error_message: str | None = None
    dequeue_retired: bool = False
    drain_retirement_started: bool = False
    drain_retired: bool = False
    handoff_ack_started: bool = False
    handoff_retired: bool = False


@dataclass(frozen=True)
class _TerminalQueueTombstone:
    request_id: int
    attempt_token: int


@dataclass
class _SubmissionTransaction:
    attempt_token: int
    request_id: int
    reservation_owned: bool = False
    terminal_state: str | None = None
    rejection_reason: str | None = None
    queue_entry: _QueueEntry | None = None
    coordinator_committed: bool = False
    registry_removed: bool = False
    reservation_aborted: bool = False
    queue_publication_uncertain: bool = False
    queue_item_preserved: bool = False
    queue_sequences: tuple[int, ...] = ()
    queue_operation_key: object = field(default_factory=object)
    queue_visible: bool = False
    queue_visibility_notified: bool = False
    terminal_queue_tombstoned: bool = False
    terminal_queue_removed: bool = False
    terminal_queue_state_cleared: bool = False
    terminal_queue_task_balanced: bool = False
    terminal_queue_operation_key: object = field(default_factory=object)
    terminal_queue_transition: _QueueTransition | None = None
    terminal_queue_transition_recorded: bool = False
    terminal_queue_depth_recorded: bool = False
    terminal_slot_released: bool = False
    recovery_deadline: float | None = None
    recovery_unresolved: bool = False

    @property
    def queued_request(self):
        return None if self.queue_entry is None else self.queue_entry.payload

    @queued_request.setter
    def queued_request(self, value) -> None:
        if value is None:
            self.queue_entry = None
        elif self.queue_entry is None:
            self.queue_entry = _QueueEntry(payload=value, task_balanced=True)
        elif self.queue_entry.payload is not value:
            raise RuntimeError("submission queue entry ownership changed")

    @property
    def queue_task_token(self):
        return None if self.queue_entry is None else self.queue_entry.task_token


class _RequestQueue(queue.Queue):
    def __init__(self, maxsize=0, transition_metrics=None):
        super().__init__(maxsize=maxsize)
        self._transition_state = _TransitionState()
        self._transition_metrics = transition_metrics
        self._closed = False
        self._dequeue_operations = {}
        self._drain_operations = {}
        self._stop_operations = {}
        self._stop_publication_operations = {}
        self._terminal_task_operations = {}
        self._compatibility_operations = {}
        self._task_local = threading.local()
        self._put_state = None
        self._wake_generation = 0
        self._worker_wake_generations = {}

    @property
    def task_token_count(self) -> int:
        with self.mutex:
            return self._live_task_entry_count_locked()

    @property
    def live_task_entry_count(self) -> int:
        with self.mutex:
            return self._live_task_entry_count_locked()

    @property
    def transition_allocation_count(self) -> int:
        with self.mutex:
            return len(self._transition_state.allocations)

    @property
    def transition_next_sequence(self) -> int:
        with self.mutex:
            return self._transition_state.next_sequence

    def _owned_entries_locked(self):
        entries = list(self.queue)
        entries.extend(
            operation.entry for operation in self._dequeue_operations.values()
        )
        for operation in self._drain_operations.values():
            entries.extend(operation.entries)
        entries.extend(
            operation.entry for operation in self._stop_operations.values()
        )
        entries.extend(
            operation.entry
            for operation in self._stop_publication_operations.values()
            if operation.entry.publication_committed
        )
        entries.extend(self._terminal_task_operations.values())
        entries.extend(
            operation.entry
            for operation in self._compatibility_operations.values()
        )
        return tuple(dict.fromkeys(entries))

    def _live_task_entry_count_locked(self) -> int:
        return sum(
            not entry.task_balanced
            for entry in self._owned_entries_locked()
        )

    def _reconcile_unfinished_locked(self) -> None:
        self.unfinished_tasks = self._live_task_entry_count_locked()
        if self.unfinished_tasks == 0:
            self.all_tasks_done.notify_all()

    def _put_entry_locked(self, entry):
        try:
            self.queue.append(entry)
        finally:
            if any(queued is entry for queued in self.queue):
                entry.publication_committed = True
            self._reconcile_unfinished_locked()
        return entry

    def _put(self, item):
        entry = getattr(self._task_local, "prepared_put_entry", None)
        if entry is None:
            entry = _QueueEntry(payload=item, state=self._put_state)
        elif entry.payload is not item:
            raise RuntimeError("prepared queue entry payload changed")
        return self._put_entry_locked(entry)

    def _get(self):
        entry = self.queue[0]
        internal_get = getattr(self._task_local, "internal_get", False)
        if internal_get:
            return self.queue.popleft().payload
        if entry.state is not None:
            raise RuntimeError("compatibility queue head is not visible")
        operation = _CompatibilityOperation(entry=entry)
        reservation = _QueueOperationReservation(
            operation.operation_key,
            operation,
        )
        entry.state = reservation
        self._compatibility_operations[entry.task_token] = operation
        operation.reservation_committed = True
        self._task_local.compatibility_get_operation = operation
        return self._resume_compatibility_get_locked(operation)

    def _resume_compatibility_get_locked(self, operation):
        entry = operation.entry
        if not operation.physical_removed:
            index = next(
                (
                    index
                    for index, queued in enumerate(self.queue)
                    if queued is entry
                ),
                None,
            )
            if index is not None:
                if index != 0:
                    raise RuntimeError(
                        "compatibility queue entry order changed"
                    )
                state = entry.state
                if not (
                    isinstance(state, _QueueOperationReservation)
                    and state.operation is operation
                ):
                    raise RuntimeError(
                        "compatibility reservation ownership changed"
                    )
                try:
                    removed = self.queue.popleft()
                except BaseException:
                    if not any(queued is entry for queued in self.queue):
                        operation.physical_removed = True
                        self.not_full.notify_all()
                        if self._head_is_visible():
                            self.not_empty.notify_all()
                    self._reconcile_unfinished_locked()
                    raise
                if removed is not entry:
                    raise RuntimeError(
                        "compatibility queue entry identity changed"
                    )
            operation.physical_removed = True
            if self._head_is_visible():
                self.not_empty.notify_all()
        if not operation.reservation_cleared:
            state = entry.state
            if isinstance(state, _QueueOperationReservation):
                if state.operation is not operation:
                    raise RuntimeError(
                        "compatibility reservation ownership changed"
                    )
                entry.state = None
            elif state is not None:
                raise RuntimeError("compatibility reservation state changed")
            operation.reservation_cleared = True
        if not operation.returned:
            operations = getattr(
                self._task_local,
                "compatibility_operations",
                None,
            )
            if operations is None:
                operations = []
                self._task_local.compatibility_operations = operations
            operations.append(operation)
            operation.returned = True
        self._task_local.compatibility_get_operation = None
        return entry.payload

    def get(self, block=True, timeout=None):
        with self.not_empty:
            retry_operation = getattr(
                self._task_local,
                "compatibility_get_operation",
                None,
            )
            if retry_operation is not None:
                item = self._resume_compatibility_get_locked(
                    retry_operation
                )
                self.not_full.notify()
                return item
            if not block:
                if not self._head_is_visible():
                    raise queue.Empty
            elif timeout is None:
                while not self._head_is_visible():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                endtime = time.monotonic() + timeout
                while not self._head_is_visible():
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item

    def put(self, item, block=True, timeout=None) -> None:
        self._put_payload(
            item,
            block=block,
            timeout=timeout,
        )

    def _put_payload(
        self,
        item,
        *,
        block,
        timeout,
    ) -> None:
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self._qsize() >= self.maxsize:
                        raise queue.Full
                elif timeout is None:
                    while self._qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("timeout must be a non-negative number")
                else:
                    endtime = time.monotonic() + timeout
                    while self._qsize() >= self.maxsize:
                        remaining = endtime - time.monotonic()
                        if remaining <= 0.0:
                            raise queue.Full
                        self.not_full.wait(remaining)
            try:
                self._put(item)
            finally:
                self._reconcile_unfinished_locked()
                if self._qsize():
                    self.not_empty.notify()

    def prepare_stop_publication(self):
        with self.mutex:
            operation = _StopPublicationOperation(
                operation_key=object(),
                entry=_QueueEntry(payload=_STOP),
            )
            self._stop_publication_operations[
                operation.operation_key
            ] = operation
            return operation

    def publish_stop(self, operation, *, timeout) -> None:
        with self.mutex:
            current = self._stop_publication_operations.get(
                operation.operation_key
            )
            if current is not operation:
                if operation.entry.publication_committed:
                    return
                raise RuntimeError("stop publication ownership missing")
        if operation.entry.publication_committed:
            return
        self._task_local.prepared_put_entry = operation.entry
        try:
            self.put(
                _STOP,
                block=True,
                timeout=timeout,
            )
        finally:
            self._task_local.prepared_put_entry = None

    def stop_publication_committed(self, operation) -> bool:
        with self.mutex:
            return bool(operation.entry.publication_committed)

    def abort_stop_publication(self, operation) -> None:
        with self.all_tasks_done:
            if operation.entry.publication_committed:
                raise RuntimeError("committed stop publication cannot abort")
            current = self._stop_publication_operations.get(
                operation.operation_key
            )
            if current is operation:
                self._stop_publication_operations.pop(
                    operation.operation_key,
                    None,
                )
            operation.entry.task_balanced = True
            self._reconcile_unfinished_locked()

    def _retire_stop_publication_entry_locked(self, entry) -> None:
        for operation_key, operation in tuple(
            self._stop_publication_operations.items()
        ):
            if operation.entry is entry:
                if not entry.task_balanced:
                    raise RuntimeError("stop publication task is not balanced")
                self._stop_publication_operations.pop(operation_key, None)
                return

    def _remove_entry_at_locked(self, index: int, *, use_queue_get=False):
        entry = self.queue[index]
        if index == 0 and use_queue_get:
            self._task_local.internal_get = True
            try:
                removed_payload = self._get()
            finally:
                self._task_local.internal_get = False
            if removed_payload is not entry.payload:
                raise RuntimeError("queue entry identity changed")
        else:
            del self.queue[index]
        return entry

    def _balance_task_entry_locked(self, entry, *, strict=False) -> bool:
        balanced = not entry.task_balanced
        if balanced:
            entry.task_balanced = True
        elif strict:
            raise ValueError("task_done() called too many times")
        self._reconcile_unfinished_locked()
        return balanced

    def _balance_task_token_locked(self, task_token, *, strict=False) -> bool:
        entry = next(
            (
                candidate
                for candidate in self._owned_entries_locked()
                if candidate.task_token is task_token
            ),
            None,
        )
        if entry is None:
            if strict:
                raise ValueError("task_done() called too many times")
            self._reconcile_unfinished_locked()
            return False
        return self._balance_task_entry_locked(entry, strict=strict)

    def task_done(self) -> None:
        with self.all_tasks_done:
            operations = getattr(
                self._task_local,
                "compatibility_operations",
                None,
            )
            compatibility_operation = None
            if operations:
                compatibility_operation = operations[-1]
                entry = compatibility_operation.entry
            else:
                operation = next(
                    (
                        operation
                        for operation in self._dequeue_operations.values()
                        if operation.worker_id is None
                        and operation.handoff_committed
                        and not operation.task_balanced
                    ),
                    None,
                )
                if operation is None:
                    raise ValueError("task_done() called too many times")
                entry = operation.entry
            if compatibility_operation is not None:
                if not compatibility_operation.task_balanced:
                    try:
                        self._balance_task_entry_locked(entry, strict=True)
                    except BaseException:
                        if entry.task_balanced:
                            compatibility_operation.task_balanced = True
                        raise
                    compatibility_operation.task_balanced = True
                if not compatibility_operation.retired:
                    try:
                        self._compatibility_operations.pop(
                            entry.task_token,
                            None,
                        )
                    except BaseException:
                        if (
                            self._compatibility_operations.get(entry.task_token)
                            is not compatibility_operation
                        ):
                            compatibility_operation.retired = True
                        raise
                    compatibility_operation.retired = True
                operations.pop()
                return
            self._balance_task_entry_locked(entry, strict=True)
            operation = next(
                (
                    operation
                    for operation in self._dequeue_operations.values()
                    if operation.entry is entry
                ),
                None,
            )
            if operation is not None:
                operation.task_balanced = True

    def _entry_state(self, item):
        entry = self._find_entry_locked(item)
        return None if entry is None else entry.state

    def _find_entry_locked(self, item):
        return next(
            (
                entry
                for entry in self._owned_entries_locked()
                if entry.payload is item
            ),
            None,
        )

    def _set_entry_state(self, item, state) -> None:
        entry = self._find_entry_locked(item)
        if entry is None:
            raise RuntimeError("queue entry ownership missing")
        entry.state = state

    def _clear_entry_state(self, item) -> None:
        entry = self._find_entry_locked(item)
        if entry is not None:
            entry.state = None

    def _head_is_visible(self) -> bool:
        if not self._qsize():
            return False
        entry = self.queue[0]
        return entry.state is None

    def _logical_depth(self) -> int:
        depth = 0
        for entry in self.queue:
            if entry.payload is _STOP:
                continue
            state = entry.state
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

    def wake_waiters(self) -> None:
        with self.not_empty:
            self._wake_generation += 1
            self.not_empty.notify_all()

    def _capture_transition(
        self,
        depth: int,
        now_ns: int | None = None,
        *,
        record_allocation: bool = True,
        operation_key=None,
    ):
        if operation_key is None:
            operation_key = object()
        state = self._transition_state
        allocation = state.allocations.get(operation_key)
        if allocation is None:
            normalized_depth = _exact_int(depth)
            normalized_now_ns = _exact_int(
                time.monotonic_ns() if now_ns is None else now_ns
            )
            transition = _QueueTransition(
                depth=normalized_depth,
                now_ns=normalized_now_ns,
                sequence=state.next_sequence,
            )
            allocation = _TransitionAllocation(transition=transition)
            allocations = dict(state.allocations)
            allocations[operation_key] = allocation
            self._swap_transition_state_locked(
                _TransitionState(
                    next_sequence=state.next_sequence + 1,
                    allocations=allocations,
                )
            )
        transition = allocation.transition
        if (
            record_allocation
            and self._transition_metrics is not None
            and not allocation.evidence_recorded
        ):
            _record_queue_sequence_allocated(
                self._transition_metrics,
                transition.sequence,
            )
            current = self._transition_state
            current_allocation = current.allocations.get(operation_key)
            if current_allocation is not None:
                allocations = dict(current.allocations)
                allocations[operation_key] = replace(
                    current_allocation,
                    evidence_recorded=True,
                )
                self._swap_transition_state_locked(
                    _TransitionState(
                        next_sequence=current.next_sequence,
                        allocations=allocations,
                    )
                )
        return transition

    def _swap_transition_state_locked(self, state: _TransitionState) -> None:
        self._transition_state = state

    def _retire_transition_locked(self, operation_key) -> None:
        state = self._transition_state
        if operation_key not in state.allocations:
            return
        allocations = dict(state.allocations)
        allocations.pop(operation_key, None)
        self._swap_transition_state_locked(
            _TransitionState(
                next_sequence=state.next_sequence,
                allocations=allocations,
            )
        )

    def _transition_evidence_recorded_locked(self, operation_key) -> bool:
        allocation = self._transition_state.allocations.get(operation_key)
        return bool(allocation is not None and allocation.evidence_recorded)

    def retire_transition(self, operation_key) -> None:
        with self.mutex:
            self._retire_transition_locked(operation_key)

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
            operation_key = (
                None
                if submission_transaction is None
                else getattr(submission_transaction, "queue_operation_key", None)
            )
            if operation_key is None:
                operation_key = object()
            entry = None
            try:
                self._put_state = (
                    "prepared" if acceptance_metrics is not None else None
                )
                try:
                    entry = self._put(queued)
                finally:
                    self._put_state = None
                if entry is None:
                    entry = self._find_entry_locked(queued)
                if submission_transaction is not None:
                    if hasattr(submission_transaction, "queue_entry"):
                        submission_transaction.queue_entry = entry
                    submission_transaction.queued_request = queued
                    submission_transaction.queue_item_preserved = True
                transition = self._capture_transition(
                    (
                        self._logical_depth()
                        if acceptance_metrics is None
                        else self._logical_depth() + 1
                    ),
                    now_ns=queued.enqueued_ns,
                    record_allocation=acceptance_metrics is None,
                    operation_key=operation_key,
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
                    self._retire_transition_locked(operation_key)
                return queued, transition
            except BaseException:
                self._put_state = None
                if entry is None:
                    entry = self._find_entry_locked(queued)
                outcome = None
                if acceptance_metrics is not None:
                    outcome = _query_accounting_outcome(
                        acceptance_metrics,
                        attempt_token,
                    )
                queued_present = bool(entry is not None and entry in self.queue)
                failed_allocation = self._transition_state.allocations.get(
                    operation_key
                )
                sequences = (
                    ()
                    if failed_allocation is None
                    else (failed_allocation.transition.sequence,)
                )
                if outcome is _OUTCOME_UNKNOWN or outcome == "accepted":
                    if queued_present and acceptance_metrics is not None:
                        entry.state = (
                            "accepted_prepared"
                            if outcome == "accepted"
                            else "prepared"
                        )
                    self._reconcile_unfinished_locked()
                    if submission_transaction is not None:
                        try:
                            if hasattr(submission_transaction, "queue_entry"):
                                submission_transaction.queue_entry = (
                                    entry if queued_present else None
                                )
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
                if entry is not None:
                    for index in range(len(self.queue) - 1, -1, -1):
                        if self.queue[index] is entry:
                            self._remove_entry_at_locked(index)
                            break
                    entry.state = None
                    self._balance_task_entry_locked(entry)
                else:
                    self._reconcile_unfinished_locked()
                if submission_transaction is not None:
                    try:
                        if hasattr(submission_transaction, "queue_entry"):
                            submission_transaction.queue_entry = entry
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
                    if evidence_complete:
                        self._retire_transition_locked(operation_key)
                else:
                    self._retire_transition_locked(operation_key)
                try:
                    self.not_full.notify()
                except BaseException:
                    pass
                raise

    def rollback_uncertain_publication(self, transaction) -> bool:
        with self.not_full:
            queued = transaction.queued_request
            entry = transaction.queue_entry
            removed = False
            if transaction.queue_item_preserved:
                for index in range(len(self.queue) - 1, -1, -1):
                    if self.queue[index] is entry:
                        self._remove_entry_at_locked(index)
                        entry.state = None
                        removed = True
                        break
            if transaction.queue_item_preserved and not removed:
                return False
            if removed:
                self._balance_task_entry_locked(entry)
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
            self._retire_transition_locked(transaction.queue_operation_key)
            self.not_full.notify()
            return True

    def restore_uncertain_visibility(self, transaction) -> None:
        with self.not_empty:
            if not transaction.queue_visible:
                queued = transaction.queued_request
                entry = transaction.queue_entry
                if not any(candidate is entry for candidate in self.queue):
                    raise RuntimeError("accepted queue ownership missing")
                entry.state = None
                transaction.queue_visible = True
            if not transaction.queue_visibility_notified:
                self.not_empty.notify()
                transaction.queue_visibility_notified = True
            transaction.queue_publication_uncertain = False
            transaction.queue_item_preserved = False
            transaction.queue_sequences = ()
            self._retire_transition_locked(transaction.queue_operation_key)

    @staticmethod
    def _terminal_tombstone(transaction):
        return _TerminalQueueTombstone(
            request_id=transaction.request_id,
            attempt_token=transaction.attempt_token,
        )

    def _tombstone_terminal_identity_locked(self, transaction) -> None:
        queued = transaction.queued_request
        entry = transaction.queue_entry
        if transaction.terminal_queue_tombstoned:
            return
        tombstone = self._terminal_tombstone(transaction)
        for index in range(len(self.queue) - 1, -1, -1):
            if self.queue[index] is not entry:
                continue
            state = entry.state
            if state == "accepted_prepared":
                entry.state = tombstone
            elif state != tombstone:
                raise RuntimeError("terminal queue item is not accepted-prepared")
            transaction.terminal_queue_tombstoned = True
            return
        if entry is not None and entry.state == tombstone:
            transaction.terminal_queue_tombstoned = True
            return
        raise RuntimeError("accepted terminal queue ownership missing")

    def _remove_terminal_identity_locked(self, transaction):
        queued = transaction.queued_request
        entry = transaction.queue_entry
        if transaction.terminal_queue_removed:
            return queued
        tombstone = self._terminal_tombstone(transaction)
        if entry is None or entry.state != tombstone:
            raise RuntimeError("terminal queue tombstone ownership missing")
        self._terminal_task_operations[
            transaction.terminal_queue_operation_key
        ] = entry
        for index in range(len(self.queue) - 1, -1, -1):
            if self.queue[index] is not entry:
                continue
            removed = self._remove_entry_at_locked(index)
            if removed is not entry:
                raise RuntimeError("terminal queue entry changed")
            transaction.terminal_queue_removed = True
            self.not_full.notify()
            if self._head_is_visible():
                self.not_empty.notify_all()
            return queued
        transaction.terminal_queue_removed = True
        return queued

    def _clear_terminal_state_locked(self, transaction) -> None:
        if transaction.terminal_queue_state_cleared:
            self._retire_transition_locked(transaction.queue_operation_key)
            return
        if not transaction.terminal_queue_removed:
            raise RuntimeError("terminal queue identity is not removed")
        queued = transaction.queued_request
        entry = transaction.queue_entry
        tombstone = self._terminal_tombstone(transaction)
        state = entry.state
        if state == tombstone:
            self._clear_entry_state(queued)
        elif state is not None:
            raise RuntimeError("terminal queue tombstone ownership missing")
        transaction.terminal_queue_state_cleared = True
        transaction.queue_item_preserved = False
        transaction.queue_publication_uncertain = False
        transaction.queue_sequences = ()
        self._retire_transition_locked(transaction.queue_operation_key)

    def _balance_terminal_task_locked(self, transaction) -> None:
        if transaction.terminal_queue_task_balanced:
            return
        if not transaction.terminal_queue_removed:
            raise RuntimeError("terminal queue identity is not removed")
        if not transaction.terminal_queue_state_cleared:
            raise RuntimeError("terminal queue state is not cleared")
        entry = transaction.queue_entry
        if entry is None:
            raise RuntimeError("terminal queue entry ownership missing")
        self._balance_task_entry_locked(entry)
        transaction.terminal_queue_task_balanced = True
        self._terminal_task_operations.pop(
            transaction.terminal_queue_operation_key,
            None,
        )

    def _capture_terminal_transition_locked(self, transaction):
        if not transaction.terminal_queue_task_balanced:
            raise RuntimeError("terminal queue task is not balanced")
        if transaction.terminal_queue_transition is not None:
            return transaction.terminal_queue_transition
        transition = self._capture_transition(
            self._logical_depth(),
            operation_key=transaction.terminal_queue_operation_key,
        )
        transaction.terminal_queue_transition = transition
        transaction.terminal_queue_transition_recorded = (
            self._transition_evidence_recorded_locked(
                transaction.terminal_queue_operation_key
            )
        )
        return transition

    def remove_terminal_accepted(self, transaction):
        with self.not_full:
            self._tombstone_terminal_identity_locked(transaction)
            queued = self._remove_terminal_identity_locked(transaction)
            self._clear_terminal_state_locked(transaction)
            self._balance_terminal_task_locked(transaction)
            transition = self._capture_terminal_transition_locked(transaction)
            return queued, transition

    @staticmethod
    def _operation_identity(request):
        request_id = _exact_int(request.request_id)
        attempt_token = getattr(request, "submission_token", None)
        if attempt_token is not None:
            attempt_token = _exact_int(attempt_token)
        return request_id, attempt_token

    def _prepare_dequeue_operation_locked(
        self,
        entry,
        worker_id,
        operation_key,
    ):
        normalized_worker_id = (
            None if worker_id is None else _exact_int(worker_id)
        )
        request_id, attempt_token = self._operation_identity(entry.payload)
        transition_depth = self._logical_depth() - 1
        operation = _DequeueOperation(
            operation_key=operation_key,
            worker_id=normalized_worker_id,
            entry=entry,
            request_id=request_id,
            attempt_token=attempt_token,
            transition_depth=transition_depth,
        )
        reservation = _QueueOperationReservation(operation_key, operation)
        try:
            entry.state = reservation
            self._dequeue_operations[operation_key] = operation
            operation.reservation_committed = True
        except BaseException:
            if entry.state is reservation:
                entry.state = None
            self._dequeue_operations.pop(operation_key, None)
            self._retire_transition_locked(operation_key)
            raise
        try:
            operation.transition = self._capture_transition(
                operation.transition_depth,
                operation_key=operation_key,
            )
        except BaseException:
            allocation = self._transition_state.allocations.get(operation_key)
            if allocation is not None:
                operation.transition = allocation.transition
            else:
                if entry.state is reservation:
                    entry.state = None
                self._dequeue_operations.pop(operation_key, None)
                operation.reservation_committed = False
                if self._head_is_visible():
                    self.not_empty.notify_all()
            raise
        return operation

    def _ensure_dequeue_transition_locked(self, operation):
        allocation = self._transition_state.allocations.get(
            operation.operation_key
        )
        if allocation is not None:
            if (
                operation.transition is not None
                and operation.transition is not allocation.transition
            ):
                raise RuntimeError("dequeue transition ownership changed")
            operation.transition = self._capture_transition(
                operation.transition_depth,
                operation_key=operation.operation_key,
            )
            return operation.transition
        if operation.transition is not None:
            raise RuntimeError("dequeue transition allocation missing")
        operation.transition = self._capture_transition(
            operation.transition_depth,
            operation_key=operation.operation_key,
        )
        return operation.transition

    def _find_dequeue_operation_locked(self, request):
        for operation in self._dequeue_operations.values():
            if operation.request is request:
                return operation
        return None

    def dequeue_operation(self, request):
        with self.mutex:
            return self._find_dequeue_operation_locked(request)

    def dequeue_operations(self):
        with self.mutex:
            return tuple(self._dequeue_operations.values())

    def bind_dequeue_completion_handoff(
        self,
        request,
        operation_key,
    ) -> None:
        with self.mutex:
            operation = self._find_dequeue_operation_locked(request)
            if operation is None:
                raise RuntimeError("dequeue operation ownership missing")
            if (
                operation.completion_operation_key is not None
                and operation.completion_operation_key is not operation_key
            ):
                raise RuntimeError("dequeue completion ownership changed")
            operation.completion_operation_key = operation_key

    def _remove_dequeue_operation_locked(
        self,
        operation,
        *,
        use_queue_get,
    ) -> None:
        if not operation.physical_removed:
            index = next(
                (
                    index
                    for index, queued in enumerate(self.queue)
                    if queued is operation.entry
                ),
                None,
            )
            if index is not None:
                state = operation.entry.state
                if not (
                    isinstance(state, _QueueOperationReservation)
                    and state.operation is operation
                ):
                    raise RuntimeError("dequeue reservation ownership changed")
                removed = self._remove_entry_at_locked(
                    index,
                    use_queue_get=use_queue_get,
                )
                if removed is not operation.entry:
                    raise RuntimeError("dequeue operation identity changed")
            operation.physical_removed = True
        if not operation.prepared_state_cleared:
            state = operation.entry.state
            if isinstance(state, _QueueOperationReservation):
                if state.operation is not operation:
                    raise RuntimeError("dequeue reservation ownership changed")
                operation.entry.state = None
            elif state is not None:
                raise RuntimeError("dequeue reservation state changed")
            operation.prepared_state_cleared = True
        if self._head_is_visible():
            self.not_empty.notify_all()
        self.not_full.notify()

    def recover_worker_dequeues(self, worker_id):
        normalized_worker_id = _exact_int(worker_id)
        with self.not_full:
            operations = [
                operation
                for operation in self._dequeue_operations.values()
                if operation.worker_id == normalized_worker_id
            ]
            for operation in operations:
                self._ensure_dequeue_transition_locked(operation)
                self._remove_dequeue_operation_locked(
                    operation,
                    use_queue_get=False,
                )
                operation.handoff_recovered = True
            return tuple(operations)

    def complete_dequeue(self, request) -> None:
        with self.all_tasks_done:
            operation = self._find_dequeue_operation_locked(request)
            if operation is None:
                raise RuntimeError("dequeue operation ownership missing")
            if not operation.task_balanced:
                if not operation.physical_removed:
                    raise RuntimeError("dequeue operation is not removed")
                if not operation.prepared_state_cleared:
                    raise RuntimeError("dequeue prepared state is not cleared")
                self._balance_task_token_locked(operation.task_token)
                operation.task_balanced = True

    def mark_dequeue_owned(self, request) -> None:
        with self.mutex:
            operation = self._find_dequeue_operation_locked(request)
            if operation is None:
                raise RuntimeError("dequeue operation ownership missing")
            operation.pending_owned_cleared = True

    def mark_dequeue_completion_handoff(
        self,
        request,
        operation_key,
    ) -> None:
        with self.mutex:
            operation = self._find_dequeue_operation_locked(request)
            if operation is None:
                raise RuntimeError("dequeue operation ownership missing")
            if (
                operation.completion_operation_key is not None
                and operation.completion_operation_key is not operation_key
            ):
                raise RuntimeError("dequeue completion ownership changed")
            operation.completion_operation_key = operation_key
            operation.completion_handoff_committed = True

    def retire_dequeue(self, request) -> bool:
        with self.mutex:
            operation = self._find_dequeue_operation_locked(request)
            if operation is None:
                return True
            stages_complete = bool(
                operation.slot_released
                and (
                    operation.transition_delivered
                    or operation.transition_failed
                )
                and operation.task_balanced
                and operation.pending_owned_cleared
                and operation.completion_handoff_committed
            )
            if not stages_complete:
                return False
            self._dequeue_operations.pop(operation.operation_key, None)
            self._retire_transition_locked(operation.operation_key)
            return True

    def take(self, block=True, timeout=None, *, worker_id=None):
        return self._take(
            block=block,
            timeout=timeout,
            worker_id=worker_id,
        )

    def get_candidate(
        self,
        on_claim,
        block=True,
        timeout=None,
        *,
        worker_id=None,
    ):
        return self._take(
            on_claim=on_claim,
            block=block,
            timeout=timeout,
            worker_id=worker_id,
        )

    def _take(
        self,
        on_claim=None,
        block=True,
        timeout=None,
        worker_id=None,
    ):
        with self.not_empty:
            normalized_worker_id = (
                None if worker_id is None else _exact_int(worker_id)
            )
            retry_operation = next(
                (
                    operation
                    for operation in self._dequeue_operations.values()
                    if operation.worker_id == normalized_worker_id
                    and not operation.handoff_committed
                ),
                None,
            )
            if retry_operation is not None:
                transition = self._ensure_dequeue_transition_locked(
                    retry_operation
                )
                self._remove_dequeue_operation_locked(
                    retry_operation,
                    use_queue_get=False,
                )
                if on_claim is not None:
                    on_claim(retry_operation.request)
                retry_operation.handoff_committed = True
                retry_operation.handoff_recovered = True
                return retry_operation.request, transition
            retry_stop = next(
                (
                    operation
                    for operation in self._stop_operations.values()
                    if operation.worker_id == normalized_worker_id
                ),
                None,
            )
            if retry_stop is not None:
                self._recover_stop_operation_locked(retry_stop)
                return _STOP, None
            wake_generation = self._wake_generation
            if normalized_worker_id is not None:
                wake_generation = self._worker_wake_generations.setdefault(
                    normalized_worker_id,
                    self._wake_generation,
                )
                if self._wake_generation != wake_generation:
                    self._worker_wake_generations[normalized_worker_id] = (
                        self._wake_generation
                    )
                    return _WAKE, None
            operation_key = object()
            if not block:
                if not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    raise queue.Empty
            elif timeout is None:
                while not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    if self._wake_generation != wake_generation:
                        if normalized_worker_id is not None:
                            self._worker_wake_generations[
                                normalized_worker_id
                            ] = self._wake_generation
                        return _WAKE, None
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                endtime = time.monotonic() + timeout
                while not self._head_is_visible():
                    if self._closed:
                        return _CLOSED, None
                    if self._wake_generation != wake_generation:
                        if normalized_worker_id is not None:
                            self._worker_wake_generations[
                                normalized_worker_id
                            ] = self._wake_generation
                        return _WAKE, None
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
            entry = self.queue[0]
            item = entry.payload
            operation = None
            if item is not _STOP:
                operation = self._prepare_dequeue_operation_locked(
                    entry,
                    worker_id,
                    operation_key,
                )
            if item is _STOP:
                stop_operation = _StopOperation(
                    operation_key=operation_key,
                    worker_id=normalized_worker_id,
                    entry=entry,
                )
                reservation = _QueueOperationReservation(
                    operation_key,
                    stop_operation,
                )
                entry.state = reservation
                self._stop_operations[operation_key] = stop_operation
                removed = self._remove_entry_at_locked(0, use_queue_get=True)
                if removed is not entry:
                    raise RuntimeError("stop-token queue identity changed")
                stop_operation.physical_removed = True
                entry.state = None
                if self._head_is_visible():
                    self.not_empty.notify_all()
                self._balance_task_entry_locked(entry)
                stop_operation.task_balanced = True
                self._stop_operations.pop(operation_key, None)
                self._retire_stop_publication_entry_locked(entry)
                transition = None
            else:
                transition = self._ensure_dequeue_transition_locked(operation)
                self._remove_dequeue_operation_locked(
                    operation,
                    use_queue_get=True,
                )
                if on_claim is not None:
                    on_claim(item)
                operation.handoff_committed = True
            self.not_full.notify()
            return item, transition

    def _recover_stop_operation_locked(self, operation) -> None:
        if not operation.physical_removed:
            index = next(
                (
                    index
                    for index, entry in enumerate(self.queue)
                    if entry is operation.entry
                ),
                None,
            )
            if index is not None:
                removed = self._remove_entry_at_locked(index)
                if removed is not operation.entry:
                    raise RuntimeError("stop-token operation identity changed")
            operation.physical_removed = True
        state = operation.entry.state
        if isinstance(state, _QueueOperationReservation):
            if state.operation is not operation:
                raise RuntimeError("stop-token reservation ownership changed")
            operation.entry.state = None
        elif state is not None:
            raise RuntimeError("stop-token reservation state changed")
        if not operation.task_balanced:
            self._balance_task_entry_locked(operation.entry)
            operation.task_balanced = True
        self._stop_operations.pop(operation.operation_key, None)
        self._retire_stop_publication_entry_locked(operation.entry)
        self.not_full.notify_all()
        if self._head_is_visible():
            self.not_empty.notify_all()

    def recover_stop_dequeues(self, worker_id):
        normalized_worker_id = _exact_int(worker_id)
        with self.not_full:
            operations = tuple(
                operation
                for operation in self._stop_operations.values()
                if operation.worker_id == normalized_worker_id
            )
            for operation in operations:
                self._recover_stop_operation_locked(operation)
            return operations

    def _prepare_drain_operation_locked(self, operation_key):
        entries = tuple(
            entry
            for entry in self.queue
            if entry.payload is not _STOP and entry.state is None
        )
        requests = tuple(entry.payload for entry in entries)
        if not requests:
            return None
        identities = tuple(
            self._operation_identity(request)
            for request in requests
        )
        request_ids = tuple(identity[0] for identity in identities)
        attempt_tokens = tuple(identity[1] for identity in identities)
        transition = self._capture_transition(
            self._logical_depth() - len(requests),
            operation_key=operation_key,
        )
        operation = _DrainOperation(
            operation_key=operation_key,
            entries=entries,
            request_ids=request_ids,
            attempt_tokens=attempt_tokens,
            transition=transition,
        )
        reservations = [
            (entry, _QueueOperationReservation(operation_key, operation))
            for entry in entries
        ]
        try:
            for entry, reservation in reservations:
                entry.state = reservation
            self._drain_operations[operation_key] = operation
            operation.reservation_committed = True
        except BaseException:
            for entry, reservation in reservations:
                if entry.state is reservation:
                    entry.state = None
            self._drain_operations.pop(operation_key, None)
            self._retire_transition_locked(operation_key)
            raise
        return operation

    def _remove_drain_operation_locked(
        self,
        operation,
        *,
        use_queue_get,
    ) -> None:
        if not operation.physical_removed:
            for entry in operation.entries:
                index = next(
                    (
                        index
                        for index, queued in enumerate(self.queue)
                        if queued is entry
                    ),
                    None,
                )
                if index is None:
                    continue
                state = entry.state
                if not (
                    isinstance(state, _QueueOperationReservation)
                    and state.operation is operation
                ):
                    raise RuntimeError("drain reservation ownership changed")
                removed = self._remove_entry_at_locked(
                    index,
                    use_queue_get=use_queue_get,
                )
                if removed is not entry:
                    raise RuntimeError("drain operation identity changed")
            operation.physical_removed = True
        if not operation.prepared_states_cleared:
            for entry in operation.entries:
                state = entry.state
                if isinstance(state, _QueueOperationReservation):
                    if state.operation is not operation:
                        raise RuntimeError("drain reservation ownership changed")
                    entry.state = None
                elif state is not None:
                    raise RuntimeError("drain reservation state changed")
            operation.prepared_states_cleared = True
        if not operation.task_balanced:
            if not operation.physical_removed:
                raise RuntimeError("drain operation is not removed")
            for entry in operation.entries:
                self._balance_task_entry_locked(entry)
            operation.task_balanced = True
        self.not_full.notify_all()
        if self._head_is_visible():
            self.not_empty.notify_all()

    def drain_operation(self, operation_key):
        with self.mutex:
            return self._drain_operations.get(operation_key)

    def finish_drain_operation(self, operation) -> bool:
        with self.mutex:
            current = self._drain_operations.get(operation.operation_key)
            if current is not operation:
                return current is None
            stages_complete = bool(
                operation.physical_removed
                and operation.prepared_states_cleared
                and operation.task_balanced
                and len(operation.released_indexes) == len(operation.requests)
                and (
                    operation.transition_delivered
                    or operation.transition_failed
                )
                and operation.failure_completion_delivered
            )
            if not stages_complete:
                return False
            self._drain_operations.pop(operation.operation_key, None)
            self._retire_transition_locked(operation.operation_key)
            return True

    def has_unresolved_operations(self) -> bool:
        with self.mutex:
            return bool(
                self._dequeue_operations
                or self._drain_operations
                or self._stop_operations
                or self._stop_publication_operations
                or self._terminal_task_operations
                or self._compatibility_operations
                or self._transition_state.allocations
            )

    def drain_requests(self, operation_key=None):
        if operation_key is None:
            operation_key = object()
        with self.not_full:
            operation = self._drain_operations.get(operation_key)
            resumed = operation is not None
            if operation is None:
                operation = self._prepare_drain_operation_locked(operation_key)
            if operation is None:
                return [], None
            self._remove_drain_operation_locked(
                operation,
                use_queue_get=not resumed,
            )
            return list(operation.requests), operation.transition

    def discard_stop_tokens(self) -> None:
        with self.not_full:
            discarded_entries = []
            for index in range(len(self.queue) - 1, -1, -1):
                if self.queue[index].payload is _STOP:
                    discarded_entries.append(
                        self._remove_entry_at_locked(index)
                    )
            for operation in tuple(self._stop_operations.values()):
                self._recover_stop_operation_locked(operation)
            if discarded_entries:
                for entry in discarded_entries:
                    entry.state = None
                    self._balance_task_entry_locked(entry)
                    self._retire_stop_publication_entry_locked(entry)
                self.not_full.notify_all()


class AsyncInferenceEngine:
    def __init__(
        self,
        runtime,
        pipeline,
        config,
        coordinator,
        metrics,
        executor=None,
    ):
        config.validate()
        runtime_batch_limit = runtime.max_dynamic_batch_size()
        if executor is None or isinstance(executor, BlockingRuntimeExecutor):
            runtime_worker_limit = runtime.max_concurrent_workers()
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
        self.executor = (
            executor
            if executor is not None
            else BlockingRuntimeExecutor(
                runtime,
                is_llm=pipeline.is_llm,
                max_new_tokens=pipeline.max_new_tokens,
                stop_token_ids=pipeline.stop_token_ids,
            )
        )
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
        self._handoff_retirement_lock = threading.RLock()
        self._execution_by_handoff = {}
        self._worker_local_handoffs = set()
        self._flush_retired_worker_handoffs = set()
        self._deferred_worker_handoffs = set()
        self._control_lock = threading.Lock()
        self._active_drain_lock = threading.RLock()
        self._cancellation_retirement_lock = threading.RLock()
        self._deferred_cancellations = {}
        self._active_cancellation_generation = None
        self._active_drain_operation_key = None
        self._active_drain_generation = None
        self._active_cancellation_requests = ()
        self._active_cancellation_completion_key = None
        self._active_cancellation_error_type = None
        self._active_cancellation_error_message = None
        self._active_cancellation_operation = None
        self._active_stop_publication_operation = None
        self._shutdown_started = False
        self._shutdown_terminal = False
        self._executor_shutdown_attempted = False
        self._executor_shutdown_ok = False
        self._submission_transactions = {}
        self._unresolved_submissions = {}

        self.requests = _RequestQueue(
            maxsize=config.queue_capacity,
            transition_metrics=metrics,
        )
        self.coordinator.handoff_ack_callback = (
            self._on_completion_handoff_terminal
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
        self._coordinator_started = False
        self._completion_monitor_started = False
        self._workers_started = [False] * len(self.workers)

    def start(self) -> None:
        with self.state_condition:
            if self.state is not EngineState.CREATED:
                raise RuntimeError(f"cannot start engine in {self.state.value}")
            self.state = EngineState.RUNNING
            self.state_condition.notify_all()

        try:
            self.coordinator.start()
            self._coordinator_started = True
            self.completion_monitor.start()
            self._completion_monitor_started = True
            for worker_id, worker in enumerate(self.workers):
                worker.start()
                self._workers_started[worker_id] = True
        except BaseException:
            self._coordinator_started = bool(
                self._coordinator_started
                or self.coordinator.thread.ident is not None
            )
            self._completion_monitor_started = bool(
                self._completion_monitor_started
                or self.completion_monitor.ident is not None
            )
            self._workers_started = [
                started or worker.ident is not None
                for started, worker in zip(
                    self._workers_started,
                    self.workers,
                )
            ]
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
                outcome = _query_accounting_outcome(
                    self.metrics,
                    transaction.attempt_token,
                )
                if outcome is _OUTCOME_UNKNOWN or outcome == "accepted":
                    raise
                if outcome == "rejected":
                    self._complete_rejected_submission(transaction)
                else:
                    self._reject_submission(
                        transaction,
                        "completion_unavailable",
                    )
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
        terminal_removal = None
        with self.coordinator.condition:
            terminal_state = self.coordinator._terminal_state_locked(
                transaction.request_id,
                transaction.attempt_token,
            )
            terminal_state = self._wait_for_terminal_cleanup_locked(
                transaction,
                terminal_state,
            )
            if not transaction.coordinator_committed:
                if (
                    terminal_state is not None
                    and terminal_state != _TERMINAL_PENDING
                ):
                    self.coordinator._remove_reservation_locked(
                        transaction.request_id,
                        transaction.attempt_token,
                    )
                else:
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
                terminal_state = self.coordinator._terminal_state_locked(
                    transaction.request_id,
                    transaction.attempt_token,
                )
            if terminal_state is None:
                raise RuntimeError("accepted registration ownership missing")
            terminal_state = self._wait_for_terminal_cleanup_locked(
                transaction,
                terminal_state,
            )
            if terminal_state == _TERMINAL_PENDING:
                self.requests.restore_uncertain_visibility(transaction)
            else:
                try:
                    terminal_removal = self.requests.remove_terminal_accepted(
                        transaction
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "accepted terminal queue ownership missing"
                    ) from exc
        with self.state_condition:
            transaction.reservation_owned = False
        if terminal_removal is not None:
            self._deliver_terminal_depth_once(transaction)
            self._release_terminal_slot_once(transaction)
        self._mark_terminal_submission_once(transaction)
        self._pop_terminal_submission_once(transaction)

    def _wait_for_terminal_cleanup_locked(self, transaction, terminal_state):
        while (
            terminal_state == _TERMINAL_PENDING
            and self.coordinator.state != _COORDINATOR_RUNNING
        ):
            if transaction.recovery_deadline is None:
                transaction.recovery_deadline = (
                    time.monotonic() + self.config.flush_timeout_sec
                )
            remaining = transaction.recovery_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("accepted terminal cleanup timed out")
            self.coordinator.condition.wait(timeout=remaining)
            terminal_state = self.coordinator._terminal_state_locked(
                transaction.request_id,
                transaction.attempt_token,
            )
            if terminal_state is None:
                raise RuntimeError("accepted registration ownership missing")
        return terminal_state

    def _deliver_terminal_depth_once(self, transaction) -> None:
        if transaction.terminal_queue_depth_recorded:
            return
        transition = transaction.terminal_queue_transition
        if transition is None or not transaction.terminal_queue_transition_recorded:
            raise RuntimeError("terminal queue transition is incomplete")
        self.metrics.record_queue_depth(
            transition.depth,
            transition.now_ns,
            sequence=transition.sequence,
        )
        transaction.terminal_queue_depth_recorded = True
        self.requests.retire_transition(
            transaction.terminal_queue_operation_key
        )

    def _release_terminal_slot_once(self, transaction) -> None:
        if transaction.terminal_slot_released:
            return
        self._release_slot_once(transaction)
        membership = _query_slot_membership(
            self._slot_pool,
            transaction.attempt_token,
        )
        if membership is _LEASE_UNKNOWN:
            raise RuntimeError("terminal slot membership is unknown")
        if membership:
            raise RuntimeError("terminal slot release did not commit")
        transaction.terminal_slot_released = True

    def _mark_terminal_submission_once(self, transaction) -> None:
        if transaction.terminal_state is None:
            self._mark_submission_terminal(transaction, "accepted")
        elif transaction.terminal_state != "accepted":
            raise RuntimeError("submission transaction outcome mismatch")

    def _pop_terminal_submission_once(self, transaction) -> None:
        if not transaction.registry_removed:
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
            if transaction.recovery_unresolved:
                continue
            outcome = _query_accounting_outcome(
                self.metrics,
                transaction.attempt_token,
            )
            pending_preflight = bool(
                transaction.queued_request is None
                and not transaction.coordinator_committed
                and not transaction.queue_publication_uncertain
                and not transaction.queue_item_preserved
                and not transaction.queue_visible
                and not transaction.terminal_queue_tombstoned
            )
            if outcome is None and pending_preflight:
                self._reject_submission(transaction, "submission_closed")

    def _close_submission_internal(self) -> None:
        with self.state_condition:
            if self.state is EngineState.RUNNING:
                self.state = EngineState.DRAINING
            self.state_condition.notify_all()

    def close_submission(self) -> None:
        self._close_submission_internal()

    def cancel_queued(self, reason: str) -> int:
        with self._control_lock:
            if self._shutdown_started:
                return 0
        return self._cancel_queued(reason, self.config.flush_timeout_sec)

    def _acquire_active_drain_until(self, deadline: float) -> bool:
        remaining = max(0.0, deadline - time.monotonic())
        return self._active_drain_lock.acquire(timeout=remaining)

    def _cancel_queued(
        self,
        reason: str,
        timeout: float | None = None,
        *,
        deadline: float | None = None,
    ) -> int:
        if deadline is None:
            deadline = time.monotonic() + max(0.0, timeout or 0.0)
        if not self._acquire_active_drain_until(deadline):
            self._mark_failed("request_failed")
            return 0
        try:
            return self._cancel_queued_locked(reason, deadline)
        finally:
            self._active_drain_lock.release()

    def _cancel_queued_locked(self, reason: str, deadline: float) -> int:
        with self._cancellation_retirement_lock:
            active_cancellation = self._active_cancellation_operation
            reconcile_deferred = bool(
                active_cancellation is not None
                and active_cancellation.phase == "DEFERRED_TERMINAL_WAIT"
            )
        if reconcile_deferred:
            self._retire_deferred_cancellations()
        with self._cancellation_retirement_lock:
            active_cancellation = self._active_cancellation_operation
            if (
                active_cancellation is not None
                and active_cancellation.phase == "DEFERRED_TERMINAL_WAIT"
            ):
                operation_key = (
                    active_cancellation.completion_operation_key
                )
                if (
                    self._deferred_cancellations.get(operation_key)
                    is active_cancellation
                ):
                    self._deferred_cancellations.pop(operation_key, None)
                active_cancellation.phase = "ACTIVE_PRODUCER"
            elif (
                active_cancellation is not None
                and active_cancellation.phase == "RETIRING"
            ):
                return len(active_cancellation.requests)
        active_at_entry = active_cancellation is not None
        if active_cancellation is None:
            requests, drain_operations = self._drain_request_queue(
                return_operations=True,
                error_type="CancelledError",
                error_message=reason,
                retain_empty_operation_key=True,
                deadline=deadline,
            )
            requests.extend(self._claim_all_pending())
        else:
            requests = list(active_cancellation.requests)
            drain_operation = (
                None
                if active_cancellation.drain_operation_key is None
                else self.requests.drain_operation(
                    active_cancellation.drain_operation_key
                )
            )
            drain_operations = (
                [] if drain_operation is None else [drain_operation]
            )
        request_by_identity = {}
        for request in requests:
            request_by_identity.setdefault(id(request), request)
        requests = list(request_by_identity.values())
        count = len(requests)
        if not count:
            with self._cancellation_retirement_lock:
                self._active_drain_operation_key = None
                self._active_drain_generation = None
            return 0
        if active_cancellation is None:
            operation_key = (
                drain_operations[0].completion_operation_key
                if drain_operations
                else object()
            )
            error_type = "CancelledError"
            error_message = reason
            if drain_operations:
                error_type = (
                    drain_operations[0].cancellation_error_type
                    or error_type
                )
                error_message = (
                    drain_operations[0].cancellation_error_message
                    or error_message
                )
            completion = self._make_failure_completion(
                requests,
                error_type=error_type,
                error_message=error_message,
                worker_id=-1,
            )
            with self._cancellation_retirement_lock:
                cancellation_generation = object()
                active_cancellation = _CancellationOperation(
                    requests=tuple(requests),
                    completion_operation_key=operation_key,
                    drain_operation_key=(
                        None
                        if not drain_operations
                        else drain_operations[0].operation_key
                    ),
                    error_type=error_type,
                    error_message=error_message,
                    completion=completion,
                    active_cancellation_generation=(
                        cancellation_generation
                    ),
                    active_drain_operation_key=(
                        self._active_drain_operation_key
                    ),
                    active_drain_generation=self._active_drain_generation,
                    drain_operation=(
                        None
                        if not drain_operations
                        else drain_operations[0]
                    ),
                )
                self._active_cancellation_generation = (
                    cancellation_generation
                )
                self._active_cancellation_operation = active_cancellation
        else:
            requests = list(active_cancellation.requests)
            count = len(requests)
            operation_key = active_cancellation.completion_operation_key
            error_type = active_cancellation.error_type
            error_message = active_cancellation.error_message
            completion = active_cancellation.completion
        with self._cancellation_retirement_lock:
            self._active_cancellation_completion_key = operation_key
            self._active_cancellation_requests = tuple(requests)
            self._active_cancellation_error_type = error_type
            self._active_cancellation_error_message = error_message
        for drain_operation in drain_operations:
            if drain_operation.cancellation_completion is None:
                drain_operation.cancellation_completion = completion
            elif drain_operation.cancellation_completion is not completion:
                raise RuntimeError("drain cancellation completion changed")
        dequeue_requests = [
            request
            for request in requests
            if self.requests.dequeue_operation(request) is not None
        ]
        dequeue_stages_complete = True
        for request in dequeue_requests:
            operation = self.requests.dequeue_operation(request)
            if operation is not None:
                dequeue_stages_complete = (
                    self._cleanup_dequeue_operation(operation)
                    and dequeue_stages_complete
                )
        if not dequeue_stages_complete:
            self._mark_failed("request_failed")
        self._bind_dequeue_handoff(dequeue_requests, operation_key)
        try:
            self._submit_failure(
                requests,
                error_type=error_type,
                error_message=error_message,
                worker_id=-1,
                timeout=max(0.0, deadline - time.monotonic()),
                operation_key=operation_key,
                completion=completion,
            )
        except BaseException:
            handoff_query_error = None
            try:
                handoff_acked = (
                    self.coordinator.completion_handoff_state(operation_key)
                    == "ACKED"
                )
            except BaseException as exc:
                handoff_acked = False
                handoff_query_error = exc
            if not handoff_acked:
                transfer_error = None
                try:
                    self._transfer_cancellation_to_deferred(
                        active_cancellation
                    )
                except BaseException as exc:
                    transfer_error = exc
                self._mark_failed("completion_thread_failed")
                if transfer_error is not None:
                    raise transfer_error
                if handoff_query_error is not None:
                    raise handoff_query_error
                return count
        try:
            dequeue_retired = self._finalize_dequeue_handoff(
                dequeue_requests,
                operation_key,
            )
            drain_retired = True
            for operation in drain_operations:
                operation.failure_completion_delivered = True
                operation.cancellation_completed = True
                drain_retired = (
                    self._retire_drain_operation(operation, deadline)
                    and drain_retired
                )
            if (
                dequeue_stages_complete
                and dequeue_retired
                and drain_retired
            ):
                if not self._acknowledge_completion_handoff(operation_key):
                    raise RuntimeError(
                        "completion handoff retirement is incomplete"
                    )
                active_cancellation.dequeue_retired = True
                active_cancellation.drain_retired = True
                active_cancellation.handoff_retired = True
                self._clear_retired_cancellation(active_cancellation)
        except BaseException:
            self._mark_failed("request_failed")
            raise
        finally:
            del requests
            del drain_operations
        with self._cancellation_retirement_lock:
            active_cancellation_retired = (
                self._active_cancellation_operation is None
            )
        if (
            active_at_entry
            and active_cancellation_retired
            and time.monotonic() < deadline
        ):
            count += self._cancel_queued_locked(reason, deadline)
        return count

    def _resume_active_drain_cancellation(
        self,
        deadline: float,
    ) -> bool | None:
        if not self._active_drain_lock.acquire(blocking=False):
            return None
        try:
            with self._cancellation_retirement_lock:
                operation_key = self._active_drain_operation_key
            if operation_key is None:
                return True
            operation = self.requests.drain_operation(operation_key)
            with self._cancellation_retirement_lock:
                if (
                    operation is None
                    and not self._active_cancellation_requests
                ):
                    self._active_drain_operation_key = None
                    self._active_drain_generation = None
                    return True
                error_type = (
                    self._active_cancellation_error_type
                    if operation is None
                    else operation.cancellation_error_type
                )
                error_message = (
                    self._active_cancellation_error_message
                    if operation is None
                    else operation.cancellation_error_message
                )
            if error_type is None:
                return False
            self._cancel_queued(
                error_message or "engine shutdown resumed queue cancellation",
                deadline=deadline,
            )
            with self._cancellation_retirement_lock:
                return self._active_drain_operation_key is None
        finally:
            self._active_drain_lock.release()

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
            close_submission = self.state is EngineState.RUNNING
        with self._control_lock:
            self._shutdown_started = True
            self._shutdown_terminal = False

        try:
            resumed_drain = self._resume_active_drain_cancellation(deadline)
        except BaseException:
            resumed_drain = False
            self._mark_failed("request_failed")
        if close_submission:
            self.close_submission()
        flushed = self._flush_until(deadline)
        submitters_stopped = self._wait_for_submitters(deadline)
        ok = resumed_drain is not False and flushed and submitters_stopped
        if not submitters_stopped:
            self._cancel_preflight_submissions()
            self.metrics.add_invalid_reason("metrics_unavailable")
        if not flushed:
            self._mark_failed("flush_timeout")
            self._cancel_queued(
                "engine shutdown after flush failure",
                deadline=deadline,
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
        if self._completion_monitor_started:
            self.completion_monitor.join(
                timeout=max(0.0, deadline - time.monotonic())
            )
        if self.completion_monitor.is_alive():
            ok = False

        try:
            abandoned = self._drain_request_queue(deadline=deadline)
        except _DrainDeadlineExceeded:
            abandoned = []
            ok = False
        if abandoned:
            ok = False
            del abandoned
        self._discard_stop_tokens()
        if self.requests.has_unresolved_operations():
            ok = False
            self.metrics.add_invalid_reason("request_failed")
        if self.coordinator.completion_handoff_count:
            ok = False
            self.metrics.add_invalid_reason("request_failed")
        with self._handoff_retirement_lock:
            has_unretired_worker_handoffs = bool(
                self._worker_local_handoffs
                or self._flush_retired_worker_handoffs
                or self._deferred_worker_handoffs
            )
        if has_unretired_worker_handoffs:
            ok = False
            self.metrics.add_invalid_reason("request_failed")
        if self.requests.live_task_entry_count:
            ok = False
            self.metrics.add_invalid_reason("request_failed")
        with self._cancellation_retirement_lock:
            has_unretired_cancellation = bool(
                self._active_cancellation_operation is not None
                or self._deferred_cancellations
            )
        if has_unretired_cancellation:
            ok = False
            self.metrics.add_invalid_reason("request_failed")

        completion_handoffs_retired = bool(
            workers_stopped
            and coordinator_stopped
            and not self.coordinator.completion_handoff_count
            and not has_unretired_worker_handoffs
        )
        if completion_handoffs_retired:
            if not self._executor_shutdown_attempted:
                self._executor_shutdown_attempted = True
                try:
                    self._executor_shutdown_ok = bool(
                        self.executor.shutdown(
                            max(0.0, deadline - time.monotonic())
                        )
                    )
                except BaseException:
                    LOGGER.exception("runtime executor shutdown failed")
                    self._executor_shutdown_ok = False
            if not self._executor_shutdown_ok:
                ok = False
                self.metrics.add_invalid_reason("worker_shutdown_failed")
        else:
            ok = False

        with self._handoff_retirement_lock:
            has_unacknowledged_executions = bool(
                self._execution_by_handoff
            )
        if has_unacknowledged_executions:
            ok = False
            self.metrics.add_invalid_reason("request_failed")

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
            abandoned = self._drain_request_queue(deadline=deadline)
            del abandoned
        if ok:
            self._retire_acked_dequeue_operations()
        return ok

    def _retire_acked_dequeue_operations(self) -> None:
        retirement_failed = False
        with self._handoff_retirement_lock:
            handoff_groups = {}
            for operation in self.requests.dequeue_operations():
                operation_key = operation.completion_operation_key
                if (
                    operation_key is not None
                    and self.coordinator.completion_handoff_state(
                        operation_key
                    )
                    == "ACKED"
                ):
                    handoff_groups.setdefault(operation_key, []).append(
                        operation.request
                    )
            for operation_key, requests in handoff_groups.items():
                if not self._finalize_dequeue_handoff(
                    requests,
                    operation_key,
                ):
                    retirement_failed = True
                    continue
                if not self._acknowledge_completion_handoff(
                    operation_key,
                    preserve_worker_retirement=True,
                ):
                    retirement_failed = True
        if retirement_failed:
            self._mark_failed("request_failed")

    def _wait_for_submitters(self, deadline: float) -> bool:
        with self.state_condition:
            while self._active_submitters:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.state_condition.wait(timeout=remaining)
        return True

    def _submit_completion_handoff(
        self,
        completion,
        operation_key,
        timeout: float,
        *,
        wait_for_ack: bool = True,
    ) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        try:
            self.coordinator.submit(
                completion,
                timeout=max(0.0, deadline - time.monotonic()),
                operation_key=operation_key,
            )
        except BaseException:
            if self.coordinator.completion_handoff_state(operation_key) == "ACKED":
                return
            raise
        if wait_for_ack:
            if not self.coordinator.wait_for_completion_handoff(
                operation_key,
                max(0.0, deadline - time.monotonic()),
            ):
                raise TimeoutError("completion handoff acknowledgement timed out")

    def _retire_worker_handoffs(
        self,
        handoffs,
        *,
        deadline=None,
    ):
        with self._handoff_retirement_lock:
            remaining_handoffs = []
            for operation_key in handoffs:
                if operation_key in self._flush_retired_worker_handoffs:
                    self._flush_retired_worker_handoffs.remove(operation_key)
                    self._worker_local_handoffs.discard(operation_key)
                    continue
                if (
                    self.coordinator.completion_handoff_state(operation_key)
                    != "ACKED"
                ):
                    if deadline is None or not self.coordinator.wait_for_completion_handoff(
                        operation_key,
                        max(0.0, deadline - time.monotonic()),
                    ):
                        remaining_handoffs.append(operation_key)
                        continue
                requests = [
                    operation.request
                    for operation in self.requests.dequeue_operations()
                    if operation.completion_operation_key is operation_key
                ]
                if not self._finalize_dequeue_handoff(
                    requests,
                    operation_key,
                ):
                    raise RuntimeError("dequeue cleanup is incomplete")
                if not self._acknowledge_completion_handoff(operation_key):
                    raise RuntimeError(
                        "completion handoff retirement is incomplete"
                    )
            return remaining_handoffs

    def _acknowledge_completion_handoff(
        self,
        operation_key,
        *,
        preserve_worker_retirement: bool = False,
    ) -> bool:
        state_before = self.coordinator.completion_handoff_state(
            operation_key
        )
        if state_before is None:
            return False
        try:
            retired = self.coordinator.acknowledge_completion_handoff(
                operation_key
            )
        except BaseException:
            if (
                state_before == "ACKED"
                and self.coordinator.completion_handoff_state(operation_key)
                is None
            ):
                completed = True
            else:
                raise
        else:
            completed = bool(
                retired
                or (
                    state_before == "ACKED"
                    and self.coordinator.completion_handoff_state(
                        operation_key
                    )
                    is None
                )
            )
        execution_ack_failed = False
        if completed:
            with self._handoff_retirement_lock:
                if (
                    preserve_worker_retirement
                    and operation_key in self._worker_local_handoffs
                ):
                    self._flush_retired_worker_handoffs.add(operation_key)
                else:
                    self._worker_local_handoffs.discard(operation_key)
                    self._flush_retired_worker_handoffs.discard(
                        operation_key
                    )
                self._deferred_worker_handoffs.discard(operation_key)
                execution = self._execution_by_handoff.get(operation_key)
                if execution is not None:
                    try:
                        self.executor.acknowledge(execution)
                    except BaseException:
                        LOGGER.exception(
                            "runtime execution acknowledgement failed"
                        )
                        execution_ack_failed = True
                    else:
                        if (
                            self._execution_by_handoff.get(operation_key)
                            is execution
                        ):
                            self._execution_by_handoff.pop(
                                operation_key,
                                None,
                            )
        if execution_ack_failed:
            self._mark_failed("request_failed")
        return completed

    def _register_worker_local_handoff(self, operation_key) -> None:
        with self._handoff_retirement_lock:
            self._worker_local_handoffs.add(operation_key)

    def _transfer_worker_handoffs_to_deferred(self, handoffs) -> None:
        with self._handoff_retirement_lock:
            for operation_key in handoffs:
                self._deferred_worker_handoffs.add(operation_key)
                self._worker_local_handoffs.discard(operation_key)
        self._retire_deferred_worker_handoffs()

    def _retire_deferred_worker_handoffs(self) -> None:
        retirement_failed = False
        with self._handoff_retirement_lock:
            handoffs = tuple(self._deferred_worker_handoffs)
            if not handoffs:
                return
            try:
                remaining = self._retire_worker_handoffs(
                    handoffs,
                    deadline=time.monotonic(),
                )
            except BaseException:
                LOGGER.exception("deferred completion handoff retirement failed")
                retirement_failed = True
            else:
                remaining_ids = {id(item) for item in remaining}
                for operation_key in handoffs:
                    if id(operation_key) not in remaining_ids:
                        self._deferred_worker_handoffs.discard(operation_key)
        if retirement_failed:
            self._mark_failed("request_failed")

    def _clear_retired_cancellation(
        self,
        operation: _CancellationOperation,
    ) -> None:
        operation_key = operation.completion_operation_key
        try:
            with self._cancellation_retirement_lock:
                if self._deferred_cancellations.get(operation_key) is operation:
                    self._deferred_cancellations.pop(operation_key, None)
                owns_active_generation = bool(
                    self._active_cancellation_operation is operation
                    or (
                        self._active_cancellation_operation is None
                        and operation.active_cancellation_generation is not None
                        and self._active_cancellation_generation
                        is operation.active_cancellation_generation
                    )
                )
                if owns_active_generation:
                    self._active_cancellation_requests = ()
                    self._active_cancellation_completion_key = None
                    self._active_cancellation_error_type = None
                    self._active_cancellation_error_message = None
                    self._active_cancellation_operation = None
                    self._active_cancellation_generation = None
                if (
                    operation.active_drain_generation is not None
                    and self._active_drain_operation_key
                    is operation.active_drain_operation_key
                    and self._active_drain_generation
                    is operation.active_drain_generation
                ):
                    self._active_drain_operation_key = None
                    self._active_drain_generation = None
                operation.phase = "RETIRED"
                operation.retirement_error_type = None
                operation.retirement_error_message = None
        except BaseException as exc:
            self._reconcile_cancellation_clear_failure(operation, exc)
            raise

    @staticmethod
    def _record_cancellation_retirement_error(
        operation: _CancellationOperation,
        exc: BaseException,
    ) -> None:
        operation.retirement_error_type = type(exc).__name__[:128]
        try:
            message = str(exc)
        except BaseException:
            message = "retirement error message unavailable"
        operation.retirement_error_message = message[:512]

    def _reconcile_cancellation_clear_failure(
        self,
        operation: _CancellationOperation,
        exc: BaseException,
    ) -> bool:
        operation_key = operation.completion_operation_key
        with self._cancellation_retirement_lock:
            owns_active_generation = bool(
                self._active_cancellation_operation is operation
                or (
                    self._active_cancellation_operation is None
                    and operation.active_cancellation_generation is not None
                    and self._active_cancellation_generation
                    is operation.active_cancellation_generation
                )
            )
            owns_drain_generation = bool(
                operation.active_drain_generation is not None
                and self._active_drain_operation_key
                is operation.active_drain_operation_key
                and self._active_drain_generation
                is operation.active_drain_generation
            )
            journal_owner = (
                self._deferred_cancellations.get(operation_key)
            )
            clear_committed = bool(
                operation.phase == "RETIRED"
                and journal_owner is not operation
                and not owns_active_generation
                and not owns_drain_generation
            )
            if clear_committed:
                return True
            if journal_owner is not None and journal_owner is not operation:
                raise RuntimeError(
                    "deferred cancellation ownership changed"
                ) from exc
            operation.phase = "DEFERRED_TERMINAL_WAIT"
            self._deferred_cancellations[operation_key] = operation
            self._record_cancellation_retirement_error(operation, exc)
            return False

    def _transfer_cancellation_to_deferred(
        self,
        operation: _CancellationOperation,
    ) -> None:
        operation_key = operation.completion_operation_key
        with self._cancellation_retirement_lock:
            if operation.phase == "RETIRED":
                return
            if operation.phase == "ACTIVE_PRODUCER":
                operation.phase = "DEFERRED_TERMINAL_WAIT"
            if operation.phase != "DEFERRED_TERMINAL_WAIT":
                return
            current = self._deferred_cancellations.get(operation_key)
            if current is not None and current is not operation:
                raise RuntimeError("deferred cancellation ownership changed")
            self._deferred_cancellations[operation_key] = operation
        self._retire_deferred_cancellations()

    def _retire_deferred_cancellations(self) -> None:
        retirement_failed = False
        with self._cancellation_retirement_lock:
            operations = tuple(self._deferred_cancellations.values())
        for operation in operations:
            operation_key = operation.completion_operation_key
            try:
                handoff_state = self.coordinator.completion_handoff_state(
                    operation_key
                )
            except BaseException as exc:
                with self._cancellation_retirement_lock:
                    if (
                        self._deferred_cancellations.get(operation_key)
                        is operation
                    ):
                        self._record_cancellation_retirement_error(
                            operation,
                            exc,
                        )
                retirement_failed = True
                continue
            if not (
                handoff_state == "ACKED"
                or (
                    handoff_state is None
                    and operation.handoff_ack_started
                )
            ):
                continue
            with self._cancellation_retirement_lock:
                if (
                    self._deferred_cancellations.get(operation_key)
                    is not operation
                    or operation.phase != "DEFERRED_TERMINAL_WAIT"
                ):
                    continue
                operation.phase = "RETIRING"
            try:
                if not operation.dequeue_retired:
                    dequeue_requests = [
                        request
                        for request in operation.requests
                        if (
                            self.requests.dequeue_operation(request)
                            is not None
                        )
                    ]
                    if not self._finalize_dequeue_handoff(
                        dequeue_requests,
                        operation_key,
                    ):
                        raise RuntimeError("dequeue cleanup is incomplete")
                    operation.dequeue_retired = True
                if not operation.drain_retired:
                    canonical_drain = operation.drain_operation
                    if canonical_drain is None:
                        operation.drain_retirement_started = True
                        operation.drain_retired = True
                    else:
                        current_drain = self.requests.drain_operation(
                            canonical_drain.operation_key
                        )
                        if not operation.drain_retirement_started:
                            if current_drain is not canonical_drain:
                                raise RuntimeError(
                                    "canonical cancellation drain ownership "
                                    "changed before retirement"
                                )
                            operation.drain_retirement_started = True
                            with canonical_drain.stage_lock:
                                canonical_drain.failure_completion_delivered = (
                                    True
                                )
                                canonical_drain.cancellation_completed = True
                        current_drain = self.requests.drain_operation(
                            canonical_drain.operation_key
                        )
                        if current_drain is canonical_drain:
                            if not self.requests.finish_drain_operation(
                                canonical_drain
                            ):
                                raise RuntimeError(
                                    "cancellation drain retirement is "
                                    "incomplete"
                                )
                        operation.drain_retired = True
                if not operation.handoff_retired:
                    if (
                        operation.handoff_ack_started
                        and handoff_state is None
                    ):
                        operation.handoff_retired = True
                    else:
                        operation.handoff_ack_started = True
                        if not self._acknowledge_completion_handoff(
                            operation_key
                        ):
                            raise RuntimeError(
                                "completion handoff retirement is incomplete"
                            )
                        operation.handoff_retired = True
                self._clear_retired_cancellation(operation)
            except BaseException as exc:
                LOGGER.exception("deferred cancellation retirement failed")
                self._reconcile_cancellation_clear_failure(
                    operation,
                    exc,
                )
                retirement_failed = True
        if retirement_failed:
            self._mark_failed("request_failed")

    def _on_completion_handoff_terminal(self) -> None:
        try:
            self.requests.wake_waiters()
            self._retire_deferred_worker_handoffs()
            self._retire_deferred_cancellations()
        except BaseException:
            LOGGER.exception("completion handoff terminal notification failed")
            self._mark_failed("request_failed")

    def _bind_execution_handoff(self, operation_key, execution) -> None:
        with self._handoff_retirement_lock:
            current = self._execution_by_handoff.get(operation_key)
            if current is not None and current is not execution:
                raise RuntimeError("runtime execution ownership changed")
            self._execution_by_handoff[operation_key] = execution

    def _bind_dequeue_handoff(self, requests, operation_key) -> None:
        for request in requests:
            if self.requests.dequeue_operation(request) is not None:
                self.requests.bind_dequeue_completion_handoff(
                    request,
                    operation_key,
                )

    def _finalize_dequeue_handoff(self, requests, operation_key) -> bool:
        for request in requests:
            if self.requests.dequeue_operation(request) is not None:
                self.requests.mark_dequeue_completion_handoff(
                    request,
                    operation_key,
                )
        for request in requests:
            if self.requests.dequeue_operation(request) is not None:
                self.requests.complete_dequeue(request)
        retired = True
        for request in requests:
            retired = self.requests.retire_dequeue(request) and retired
        return retired

    def _cleanup_dequeue_operation(
        self,
        operation,
        *,
        attempts: int = 3,
    ) -> bool:
        for _attempt in range(attempts):
            try:
                self._request_dequeued(
                    operation.request,
                    operation.transition,
                )
            except BaseException:
                if (
                    operation.slot_released
                    and (
                        operation.transition_delivered
                        or operation.transition_failed
                    )
                ):
                    return True
                continue
            return True
        return bool(
            operation.slot_released
            and (
                operation.transition_delivered
                or operation.transition_failed
            )
        )

    def _worker(self, worker_id: int) -> None:
        has_pending = False
        owned = []
        pending_handoffs = []
        consecutive_failures = 0
        completion = None
        completion_operation_key = None
        execution = None
        try:
            while True:
                pending_handoffs = self._retire_worker_handoffs(
                    pending_handoffs
                )
                stop_after_batch = False
                owned = []
                completion = None
                completion_operation_key = None
                execution = None
                if has_pending:
                    first = self._take_pending(worker_id)
                    has_pending = False
                    if first is None:
                        continue
                    owned = [first]
                    self.requests.mark_dequeue_owned(first)
                else:
                    first, transition = self.requests.take(worker_id=worker_id)
                    if first is _WAKE:
                        continue
                    if first is _CLOSED:
                        pending_handoffs = self._retire_worker_handoffs(
                            pending_handoffs,
                            deadline=(
                                time.monotonic()
                                + self.config.flush_timeout_sec
                            ),
                        )
                        if pending_handoffs:
                            raise TimeoutError(
                                "worker completion acknowledgement timed out"
                            )
                        return
                    if first is _STOP:
                        self._pass_stop_token()
                        pending_handoffs = self._retire_worker_handoffs(
                            pending_handoffs,
                            deadline=(
                                time.monotonic()
                                + self.config.flush_timeout_sec
                            ),
                        )
                        if pending_handoffs:
                            raise TimeoutError(
                                "worker completion acknowledgement timed out"
                            )
                        return
                    owned = [first]
                    self._request_dequeued(first, transition)
                    self.requests.mark_dequeue_owned(first)

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
                                worker_id=worker_id,
                            )
                        except queue.Empty:
                            break
                        if candidate is _STOP or candidate is _CLOSED:
                            if candidate is _STOP:
                                self._pass_stop_token()
                            stop_after_batch = True
                            break
                        if candidate is _WAKE:
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
                        self.requests.mark_dequeue_owned(claimed)
                        owned.append(claimed)
                        batch.append(claimed)

                collated = {}
                started_ns = None
                actual_batch_size = (
                    sum(request.sample_count for request in batch)
                    if self.pipeline.is_static_batched
                    else self._dynamic_batch_size(batch)
                )
                completion_operation_key = object()
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
                    execution = self.executor.execute(
                        runtime_input,
                        timeout=self.config.flush_timeout_sec,
                    )
                    try:
                        finished_ns = time.monotonic_ns()
                    finally:
                        self._bind_execution_handoff(
                            completion_operation_key,
                            execution,
                        )
                    self._bind_dequeue_handoff(
                        batch,
                        completion_operation_key,
                    )
                    completion = BatchCompletion(
                        requests=tuple(batch),
                        collated=collated,
                        outputs=execution.outputs,
                        timing_ms=execution.timing_ms,
                        runtime_started_ns=started_ns,
                        runtime_finished_ns=finished_ns,
                        worker_id=worker_id,
                        batch_size=actual_batch_size,
                        generated_tokens=execution.generated_tokens,
                        error_type=execution.error_type,
                        error_message=execution.error_message,
                        generation_observation=(
                            execution.generation_observation
                        ),
                    )
                    if execution.error_type is None:
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= 3:
                            self._mark_failed("request_failed")
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

                self._bind_dequeue_handoff(
                    batch,
                    completion_operation_key,
                )
                if execution is not None:
                    self._bind_execution_handoff(
                        completion_operation_key,
                        execution,
                    )
                self._register_worker_local_handoff(
                    completion_operation_key
                )
                self.metrics.record_worker_busy(
                    worker_id,
                    started_ns,
                    finished_ns,
                    actual_batch_size,
                    sum(request.sample_count for request in batch),
                )
                self._submit_completion_handoff(
                    completion,
                    completion_operation_key,
                    self.config.flush_timeout_sec,
                    wait_for_ack=False,
                )
                pending_handoffs.append(completion_operation_key)
                pending_handoffs = self._retire_worker_handoffs(
                    pending_handoffs
                )
                if pending_handoffs and isinstance(
                    self.executor,
                    NativeAsyncRuntimeExecutor,
                ):
                    self._transfer_worker_handoffs_to_deferred(
                        pending_handoffs
                    )
                    pending_handoffs = []
                owned = []
                completion = None
                completion_operation_key = None
                execution = None
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
                        self.requests.mark_dequeue_owned(pending_request)
                        owned = [pending_request]
                        raise RuntimeError(
                            "worker stopped with a pending accepted request"
                        )
                    pending_handoffs = self._retire_worker_handoffs(
                        pending_handoffs,
                        deadline=(
                            time.monotonic()
                            + self.config.flush_timeout_sec
                        ),
                    )
                    if pending_handoffs:
                        raise TimeoutError(
                            "worker completion acknowledgement timed out"
                        )
                    return
        except BaseException as exc:
            LOGGER.exception("async worker %s terminated unexpectedly", worker_id)
            try:
                pending_handoffs = self._retire_worker_handoffs(
                    pending_handoffs,
                    deadline=(
                        time.monotonic() + self.config.flush_timeout_sec
                    ),
                )
            except BaseException:
                self._mark_failed("request_failed")
            current_owned_request_ids = {
                id(request) for request in owned
            }
            failed = list(owned)
            pending_request = self._take_pending(worker_id)
            has_pending = False
            if pending_request is not None:
                self.requests.mark_dequeue_owned(pending_request)
                failed.append(pending_request)
            recovered_operations = self.requests.recover_worker_dequeues(worker_id)
            for operation in recovered_operations:
                self.requests.mark_dequeue_owned(operation.request)
                if not self._cleanup_dequeue_operation(operation):
                    self._mark_failed("request_failed")
                if not any(
                    request is operation.request
                    for request in failed
                ):
                    failed.append(operation.request)

            current_execution_bound = False
            if (
                completion_operation_key is not None
                and execution is not None
            ):
                with self._handoff_retirement_lock:
                    current_execution_bound = bool(
                        self._execution_by_handoff.get(
                            completion_operation_key
                        )
                        is execution
                    )
            bound_execution_without_completion = bool(
                completion is None and current_execution_bound
            )
            current_handoff_requests = [
                operation.request
                for operation in recovered_operations
                if (
                    id(operation.request) in current_owned_request_ids
                    and (
                        operation.completion_operation_key is None
                        or operation.completion_operation_key
                        is completion_operation_key
                    )
                )
            ]
            if (
                completion_operation_key is not None
                and current_handoff_requests
                and (
                    completion is not None
                    or bound_execution_without_completion
                )
            ):
                self._bind_dequeue_handoff(
                    current_handoff_requests,
                    completion_operation_key,
                )

            handoff_groups = {}
            for operation in recovered_operations:
                if operation.completion_operation_key is not None:
                    handoff_groups.setdefault(
                        operation.completion_operation_key,
                        [],
                    ).append(operation.request)
            current_recovered_handoff = bool(
                completion_operation_key in handoff_groups
                and (
                    completion is not None
                    or bound_execution_without_completion
                )
            )
            if current_recovered_handoff:
                self._register_worker_local_handoff(
                    completion_operation_key
                )
                try:
                    if completion is None:
                        self._submit_failure(
                            handoff_groups[completion_operation_key],
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            worker_id=worker_id,
                            timeout=self.config.flush_timeout_sec,
                            operation_key=completion_operation_key,
                        )
                    else:
                        self._submit_completion_handoff(
                            completion,
                            completion_operation_key,
                            self.config.flush_timeout_sec,
                        )
                except BaseException:
                    pass
            completed_handoff_ids = set()
            deferred_handoffs = []
            for operation_key, requests in handoff_groups.items():
                try:
                    remaining = self._retire_worker_handoffs(
                        [operation_key],
                        deadline=(
                            time.monotonic()
                            + self.config.flush_timeout_sec
                        ),
                    )
                    if not remaining:
                        completed_handoff_ids.update(
                            id(request) for request in requests
                        )
                    else:
                        deferred_handoffs.extend(remaining)
                except BaseException:
                    self._mark_failed("request_failed")
                    deferred_handoffs.append(operation_key)
            if deferred_handoffs:
                self._transfer_worker_handoffs_to_deferred(
                    deferred_handoffs
                )
            if completed_handoff_ids:
                failed = [
                    request
                    for request in failed
                    if id(request) not in completed_handoff_ids
                ]
            failed = [
                request
                for request in failed
                if (
                    self.requests.dequeue_operation(request) is None
                    or self.requests.dequeue_operation(
                        request
                    ).completion_operation_key
                    is None
                )
            ]

            try:
                drained, drain_operations = self._drain_request_queue(
                    return_operations=True,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except BaseException:
                drained = []
                drain_operations = []
                self._mark_failed("request_failed")
            failed.extend(drained)
            unique_failed = []
            seen_failed = set()
            for request in failed:
                identity = id(request)
                if identity not in seen_failed:
                    seen_failed.add(identity)
                    unique_failed.append(request)
            failed = unique_failed
            self._stop_requested.set()
            self._mark_failed("request_failed")
            if failed:
                failure_operation_key = (
                    drain_operations[0].completion_operation_key
                    if drain_operations
                    else object()
                )
                dequeue_failed = [
                    request
                    for request in failed
                    if self.requests.dequeue_operation(request) is not None
                ]
                self._bind_dequeue_handoff(
                    dequeue_failed,
                    failure_operation_key,
                )
                try:
                    self._submit_failure(
                        failed,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                        worker_id=worker_id,
                        timeout=self.config.flush_timeout_sec,
                        operation_key=failure_operation_key,
                    )
                    dequeue_retired = self._finalize_dequeue_handoff(
                        dequeue_failed,
                        failure_operation_key,
                    )
                    drain_retired = True
                    for operation in drain_operations:
                        operation.failure_completion_delivered = True
                        drain_retired = (
                            self._retire_drain_operation(operation)
                            and drain_retired
                        )
                    if dequeue_retired and drain_retired:
                        self._acknowledge_completion_handoff(
                            failure_operation_key
                        )
                except BaseException:
                    if (
                        self.coordinator.completion_handoff_state(
                            failure_operation_key
                        )
                        == "ACKED"
                    ):
                        try:
                            dequeue_retired = self._finalize_dequeue_handoff(
                                dequeue_failed,
                                failure_operation_key,
                            )
                            drain_retired = True
                            for operation in drain_operations:
                                operation.failure_completion_delivered = True
                                drain_retired = (
                                    self._retire_drain_operation(operation)
                                    and drain_retired
                                )
                            if dequeue_retired and drain_retired:
                                self._acknowledge_completion_handoff(
                                    failure_operation_key
                                )
                        except BaseException:
                            self._mark_failed("request_failed")
                    else:
                        self._mark_failed("completion_thread_failed")
                if (
                    self.coordinator.completion_handoff_state(
                        failure_operation_key
                    )
                    != "ACKED"
                    and self.coordinator.thread_error is not None
                ):
                    self._mark_failed("completion_thread_failed")
            del failed
            del drained
            del drain_operations

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
        operation = self.requests.dequeue_operation(request)
        if operation is None:
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
            return
        if operation.transition is not transition:
            raise RuntimeError("dequeue transition ownership changed")
        with operation.stage_lock:
            if not operation.slot_released:
                try:
                    self._slot_pool.release_lease(operation.attempt_token)
                except BaseException:
                    membership = _query_slot_membership(
                        self._slot_pool,
                        operation.attempt_token,
                    )
                    if membership is False:
                        operation.slot_released = True
                    raise
                membership = _query_slot_membership(
                    self._slot_pool,
                    operation.attempt_token,
                )
                if membership is _LEASE_UNKNOWN:
                    raise RuntimeError("dequeue slot membership is unknown")
                if membership:
                    raise RuntimeError("dequeue slot release did not commit")
                operation.slot_released = True
            if not operation.transition_delivered:
                try:
                    self.metrics.record_queue_depth(
                        transition.depth,
                        transition.now_ns,
                        sequence=transition.sequence,
                    )
                except BaseException:
                    operation.transition_failed = True
                    self.metrics.record_queue_depth_failure(transition.sequence)
                    raise
                operation.transition_delivered = True

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
        for request in pending:
            self.requests.mark_dequeue_owned(request)
        return pending

    def _complete_drain_operation(self, operation) -> None:
        with operation.stage_lock:
            for index, request in enumerate(operation.requests):
                if index in operation.released_indexes:
                    continue
                attempt_token = operation.attempt_tokens[index]
                try:
                    self._slot_pool.release_lease(attempt_token)
                except BaseException:
                    if attempt_token is not None:
                        membership = _query_slot_membership(
                            self._slot_pool,
                            attempt_token,
                        )
                        if membership is False:
                            operation.released_indexes.add(index)
                    raise
                if attempt_token is not None:
                    membership = _query_slot_membership(
                        self._slot_pool,
                        attempt_token,
                    )
                    if membership is _LEASE_UNKNOWN:
                        raise RuntimeError("drain slot membership is unknown")
                    if membership:
                        raise RuntimeError("drain slot release did not commit")
                operation.released_indexes.add(index)
            if not operation.transition_delivered:
                try:
                    self.metrics.record_queue_depth(
                        operation.transition.depth,
                        operation.transition.now_ns,
                        sequence=operation.transition.sequence,
                    )
                except BaseException:
                    operation.transition_failed = True
                    self.metrics.record_queue_depth_failure(
                        operation.transition.sequence
                    )
                else:
                    operation.transition_delivered = True

    def _reconcile_drain_publication_generation(
        self,
        *,
        operation_key,
        reserved_generation,
        previous_generation,
        previous_owner,
        published_operation,
    ) -> None:
        if reserved_generation is None:
            return
        with self._cancellation_retirement_lock:
            if not (
                self._active_drain_operation_key is operation_key
                and self._active_drain_generation is reserved_generation
            ):
                return
            if published_operation is not None:
                return
            if (
                previous_owner is not None
                and previous_owner.phase == "RETIRED"
            ):
                self._active_drain_operation_key = None
                self._active_drain_generation = None
            else:
                self._active_drain_generation = previous_generation

    def _drain_request_queue(
        self,
        *,
        operation_key=None,
        return_operations=False,
        error_type=None,
        error_message=None,
        retain_empty_operation_key=False,
        deadline=None,
    ):
        if deadline is None:
            deadline = time.monotonic() + self.config.flush_timeout_sec
        if not self._acquire_active_drain_until(deadline):
            raise _DrainDeadlineExceeded("active queue drain deadline expired")
        try:
            with self._cancellation_retirement_lock:
                if self._active_drain_operation_key is not None:
                    operation_key = self._active_drain_operation_key
                elif operation_key is None:
                    operation_key = object()
                    self._active_drain_operation_key = operation_key
                    self._active_drain_generation = object()
                else:
                    self._active_drain_operation_key = operation_key
                    self._active_drain_generation = object()
                if self._active_drain_generation is None:
                    self._active_drain_generation = object()
                previous_generation = self._active_drain_generation
                previous_owner = self._active_cancellation_operation
                if not (
                    previous_owner is not None
                    and previous_owner.active_drain_operation_key
                    is operation_key
                    and previous_owner.active_drain_generation
                    is previous_generation
                ):
                    previous_owner = None
            previous_drain_operation = self.requests.drain_operation(
                operation_key
            )
            reserved_generation = None
            if previous_drain_operation is None:
                reserved_generation = object()
                with self._cancellation_retirement_lock:
                    current_drain_key = self._active_drain_operation_key
                    if (
                        current_drain_key is not None
                        and current_drain_key is not operation_key
                    ):
                        raise RuntimeError(
                            "active drain publication ownership changed"
                        )
                    self._active_drain_operation_key = operation_key
                    self._active_drain_generation = reserved_generation
            try:
                drained, _transition = self.requests.drain_requests(
                    operation_key
                )
            except BaseException as primary:
                try:
                    drained, _transition = self.requests.drain_requests(
                        operation_key
                    )
                except BaseException:
                    published_operation = self.requests.drain_operation(
                        operation_key
                    )
                    self._reconcile_drain_publication_generation(
                        operation_key=operation_key,
                        reserved_generation=reserved_generation,
                        previous_generation=previous_generation,
                        previous_owner=previous_owner,
                        published_operation=published_operation,
                    )
                    self._mark_failed("request_failed")
                    raise primary
            operation = self.requests.drain_operation(operation_key)
            self._reconcile_drain_publication_generation(
                operation_key=operation_key,
                reserved_generation=reserved_generation,
                previous_generation=previous_generation,
                previous_owner=previous_owner,
                published_operation=operation,
            )
            operations = [] if operation is None else [operation]
            if operation is None:
                with self._cancellation_retirement_lock:
                    if (
                        not retain_empty_operation_key
                        and self._active_drain_operation_key is operation_key
                    ):
                        self._active_drain_operation_key = None
                        self._active_drain_generation = None
            else:
                if (
                    error_type is not None
                    and operation.cancellation_error_type is None
                ):
                    operation.cancellation_error_type = str(error_type)
                if (
                    error_message is not None
                    and operation.cancellation_error_message is None
                ):
                    operation.cancellation_error_message = str(error_message)
                try:
                    self._complete_drain_operation(operation)
                except BaseException as primary:
                    try:
                        self._complete_drain_operation(operation)
                    except BaseException:
                        self._mark_failed("request_failed")
                        raise primary
            if return_operations:
                return drained, operations
            return drained
        finally:
            self._active_drain_lock.release()

    def _retire_drain_operation(self, operation, deadline=None) -> bool:
        retired = self.requests.finish_drain_operation(operation)
        if retired:
            if deadline is None:
                deadline = time.monotonic() + self.config.flush_timeout_sec
            if not self._acquire_active_drain_until(deadline):
                return False
            try:
                with self._cancellation_retirement_lock:
                    if (
                        not self._active_cancellation_requests
                        and self._active_drain_operation_key
                        is operation.operation_key
                    ):
                        self._active_drain_operation_key = None
                        self._active_drain_generation = None
            finally:
                self._active_drain_lock.release()
        return retired

    def _submit_failure(
        self,
        requests,
        *,
        error_type: str,
        error_message: str,
        worker_id: int,
        timeout: float,
        operation_key=None,
        completion=None,
    ) -> None:
        if not requests:
            return
        if completion is None:
            completion = self._make_failure_completion(
                requests,
                error_type=error_type,
                error_message=error_message,
                worker_id=worker_id,
            )
        else:
            canonical_requests = tuple(completion.requests)
            submitted_requests = tuple(requests)
            if len(canonical_requests) != len(submitted_requests) or any(
                canonical is not submitted
                for canonical, submitted in zip(
                    canonical_requests,
                    submitted_requests,
                )
            ):
                raise RuntimeError(
                    "failure completion request ownership changed"
                )
        if operation_key is None:
            self.coordinator.submit(completion, timeout=timeout)
            return
        self._submit_completion_handoff(
            completion,
            operation_key,
            timeout,
        )

    @staticmethod
    def _make_failure_completion(
        requests,
        *,
        error_type: str,
        error_message: str,
        worker_id: int,
    ):
        now_ns = time.monotonic_ns()
        return BatchCompletion(
            requests=tuple(requests),
            collated={},
            outputs=None,
            timing_ms=None,
            runtime_started_ns=now_ns,
            runtime_finished_ns=now_ns,
            worker_id=worker_id,
            batch_size=sum(request.sample_count for request in requests),
            error_type=error_type,
            error_message=error_message,
        )

    def _enqueue_stop(self, deadline: float) -> bool:
        if self.requests.closed:
            return True
        operation = self._active_stop_publication_operation
        if operation is None:
            operation = self.requests.prepare_stop_publication()
            self._active_stop_publication_operation = operation
        try:
            self.requests.publish_stop(
                operation,
                timeout=max(0.0, deadline - time.monotonic()),
            )
        except queue.Full:
            self.requests.abort_stop_publication(operation)
            self._active_stop_publication_operation = None
            return False
        except BaseException:
            if self.requests.stop_publication_committed(operation):
                self._active_stop_publication_operation = None
                return True
            self.requests.abort_stop_publication(operation)
            self._active_stop_publication_operation = None
            return False
        self._active_stop_publication_operation = None
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
        for started, worker in zip(self._workers_started, self.workers):
            if not started:
                continue
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

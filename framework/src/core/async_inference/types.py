from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Sequence


class AsyncScenario(str, Enum):
    OFFLINE = "offline"
    SERVER_LIKE = "server_like"


class EngineState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    FAILED = "failed"
    STOPPED = "stopped"


class TerminalStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class RunStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


@dataclass(frozen=True)
class AsyncInferenceConfig:
    scenario: AsyncScenario = AsyncScenario.OFFLINE
    queue_capacity: int = 256
    worker_count: int = 1
    max_batch_size: int = 1
    batch_timeout_ms: float = 1.0
    submit_timeout_sec: float = 30.0
    flush_timeout_sec: float = 300.0
    request_timeout_ms: float = 0.0
    min_samples: int = 100
    min_duration_sec: float = 0.0
    max_samples: Optional[int] = None
    target_qps: Optional[float] = None
    schedule_seed: int = 0
    latency_slo_ms: Optional[float] = None

    def validate(self) -> None:
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.queue_capacity < self.max_batch_size:
            raise ValueError("queue_capacity must be >= max_batch_size")
        if self.batch_timeout_ms < 0:
            raise ValueError("batch_timeout_ms must be >= 0")
        if self.submit_timeout_sec <= 0 or self.flush_timeout_sec <= 0:
            raise ValueError("submit_timeout_sec and flush_timeout_sec must be > 0")
        if self.request_timeout_ms < 0:
            raise ValueError("request_timeout_ms must be >= 0")
        if self.min_samples < 1 or self.min_duration_sec < 0:
            raise ValueError("minimum run constraints are invalid")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be >= 1")
        if self.scenario is AsyncScenario.SERVER_LIKE:
            if self.target_qps is None or self.target_qps <= 0:
                raise ValueError("server_like requires target_qps > 0")
        elif self.target_qps is not None:
            raise ValueError("target_qps is only valid for server_like")
        if self.latency_slo_ms is not None and self.latency_slo_ms <= 0:
            raise ValueError("latency_slo_ms must be > 0")


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    sample_index: int
    sample: Dict[str, Any]
    scheduled_ns: int
    issued_ns: int
    enqueued_ns: int
    sample_count: int = 1


@dataclass(frozen=True)
class FirstTokenEvent:
    request_id: int
    first_token_ns: int
    token_count: int = 1


@dataclass(frozen=True)
class BatchCompletion:
    requests: Sequence[InferenceRequest]
    collated: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    runtime_started_ns: int
    runtime_finished_ns: int
    worker_id: int
    batch_size: int
    generated_tokens: int = 0
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class RequestTrace:
    request_id: int
    sample_index: int
    status: TerminalStatus
    scheduled_ns: int
    issued_ns: int
    enqueued_ns: int
    runtime_started_ns: int
    runtime_finished_ns: int
    completed_ns: int
    worker_id: int
    batch_size: int
    timed_out: bool
    sample_count: int = 1
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@dataclass(frozen=True)
class AsyncBenchmarkResult:
    metrics: Dict[str, Any]
    details: Dict[str, Any]
    status: RunStatus
    invalid_reasons: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

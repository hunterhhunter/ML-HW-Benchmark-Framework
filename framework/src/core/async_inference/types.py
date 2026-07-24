import math
from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral, Real
from typing import Any, Dict, Optional, Sequence

from ..runtime_executor import GenerationObservation, GenerationOutputEvent


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


def _is_positive_integral(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and value > 0
    )


def _is_integral(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral)


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and math.isfinite(float(value))
    )


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
        if type(self.scenario) is not AsyncScenario:
            raise ValueError("scenario must be an AsyncScenario")
        if not _is_positive_integral(self.queue_capacity):
            raise ValueError("queue_capacity must be a positive integer")
        if not _is_positive_integral(self.worker_count):
            raise ValueError("worker_count must be a positive integer")
        if not _is_positive_integral(self.max_batch_size):
            raise ValueError("max_batch_size must be a positive integer")
        if self.queue_capacity < self.max_batch_size:
            raise ValueError("queue_capacity must be >= max_batch_size")
        if not _is_finite_real(self.batch_timeout_ms) or self.batch_timeout_ms < 0:
            raise ValueError("batch_timeout_ms must be >= 0")
        if (
            not _is_finite_real(self.submit_timeout_sec)
            or self.submit_timeout_sec <= 0
        ):
            raise ValueError("submit_timeout_sec must be > 0")
        if (
            not _is_finite_real(self.flush_timeout_sec)
            or self.flush_timeout_sec <= 0
        ):
            raise ValueError("flush_timeout_sec must be > 0")
        if (
            not _is_finite_real(self.request_timeout_ms)
            or self.request_timeout_ms < 0
        ):
            raise ValueError("request_timeout_ms must be >= 0")
        if not _is_positive_integral(self.min_samples):
            raise ValueError("min_samples must be a positive integer")
        if (
            not _is_finite_real(self.min_duration_sec)
            or self.min_duration_sec < 0
        ):
            raise ValueError("min_duration_sec must be a finite value >= 0")
        if self.max_samples is not None and not _is_positive_integral(
            self.max_samples
        ):
            raise ValueError("max_samples must be a positive integer")
        if not _is_integral(self.schedule_seed):
            raise ValueError("schedule_seed must be an integer")
        if self.target_qps is not None and not _is_finite_real(
            self.target_qps
        ):
            raise ValueError("target_qps must be a finite real number")
        if self.scenario is AsyncScenario.SERVER_LIKE:
            if self.target_qps is None or self.target_qps <= 0:
                raise ValueError("server_like requires target_qps > 0")
        elif self.target_qps is not None:
            raise ValueError("target_qps is only valid for server_like")
        if self.latency_slo_ms is not None and (
            not _is_finite_real(self.latency_slo_ms)
            or self.latency_slo_ms <= 0
        ):
            raise ValueError("latency_slo_ms must be a finite value > 0")


@dataclass(frozen=True)
class InferenceRequest:
    request_id: int
    sample_index: int
    sample: Dict[str, Any]
    scheduled_ns: int
    issued_ns: int
    enqueued_ns: int
    sample_count: int = 1
    task: Optional[str] = None
    generation_options: Optional[Dict[str, Any]] = None
    batch_axis: Optional[int] = None
    submission_token: Optional[int] = None


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
    generation_observation: GenerationObservation | None = None


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
    generated_tokens: int = 0
    backend_submitted_ns: Optional[int] = None
    generation_events: tuple[GenerationOutputEvent, ...] = ()
    generation_timing_source: Optional[str] = None


@dataclass(frozen=True)
class AsyncBenchmarkResult:
    metrics: Dict[str, Any]
    details: Dict[str, Any]
    status: RunStatus
    invalid_reasons: Sequence[str] = field(default_factory=tuple)
    warnings: Sequence[str] = field(default_factory=tuple)

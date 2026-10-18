from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Dict, List, Literal, Optional
from enum import Enum


class BenchmarkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelProfileResponse(BaseModel):
    """모델 프로필 정보"""
    model_name: str
    task: str
    backends: List[str]
    default_model_path: Optional[str] = None
    default_dataset_path: Optional[str] = None


class ProfileListResponse(BaseModel):
    """사용 가능한 모델 프로필 목록"""
    profiles: List[ModelProfileResponse]


class TargetResponse(BaseModel):
    """벤치마크 실행 target 정보"""
    target_id: str
    label: str
    runtime_name: str
    device: str
    compiler_name: Optional[str] = None
    monitor_names: List[str]
    artifact_format: str
    accelerator_vendor: str
    accelerator_name: str
    device_selector: str
    capabilities: List[str]
    description: str = ""


class TargetListResponse(BaseModel):
    """사용 가능한 target 목록"""
    targets: List[TargetResponse]


class BenchmarkRunRequest(BaseModel):
    """CLI 옵션의 미입력과 명시 입력을 구분하는 웹 실행 요청."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    target_id: Optional[str] = None
    backend: Optional[str] = None
    device: Optional[str] = None
    onnx_path: Optional[str] = None
    hef_path: Optional[str] = None
    artifact_path: Optional[str] = None
    mobilint_bert_weights: Optional[str] = None
    fxb_path: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    dataset_path: Optional[str] = None
    image_dir: Optional[str] = None
    label_dir: Optional[str] = None
    layout: Optional[Literal["NCHW", "NHWC"]] = None
    image_preprocess_mode: Optional[Literal["auto", "normalized", "raw"]] = None
    image_preprocess_profile: Optional[str] = None
    image_resize_mode: Optional[Literal["auto", "direct", "letterbox"]] = None
    batch_size: Optional[int] = Field(default=None, ge=1)
    warmup: Optional[int] = Field(default=None, ge=0)
    max_steps: Optional[int] = Field(default=None, ge=1)
    max_new_tokens: Optional[int] = Field(default=None, ge=1)
    max_model_len: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    compile: Optional[bool] = None
    compile_options: Dict[str, str] = Field(default_factory=dict)
    runtime_options: Dict[str, str] = Field(default_factory=dict)
    enforce_eager: Optional[bool] = None
    results_path: Optional[str] = None
    debug: Optional[bool] = None
    monitor: Optional[bool] = None
    monitor_interval: Optional[float] = Field(default=None, gt=0)
    inference_mode: Optional[Literal["e2e", "async_queue"]] = None
    scenario: Optional[Literal["offline", "server_like"]] = None
    async_pass: Optional[Literal["quality-and-runtime", "runtime-only"]] = None
    target_qps: Optional[float] = Field(default=None, gt=0)
    queue_capacity: Optional[int] = Field(default=None, ge=1)
    worker_count: Optional[int] = Field(default=None, ge=1)
    batch_timeout_ms: Optional[float] = Field(default=None, ge=0)
    submit_timeout_sec: Optional[float] = Field(default=None, gt=0)
    flush_timeout_sec: Optional[float] = Field(default=None, gt=0)
    request_timeout_ms: Optional[float] = Field(default=None, gt=0)
    min_samples: Optional[int] = Field(default=None, ge=1)
    min_duration_sec: Optional[float] = Field(default=None, ge=0)
    max_samples: Optional[int] = Field(default=None, ge=1)
    schedule_seed: Optional[int] = None
    latency_slo_ms: Optional[float] = Field(default=None, gt=0)
    save_request_trace: Optional[bool] = None
    async_preload_inputs: Optional[bool] = None
    runner_profile_id: Optional[str] = None
    result_source_id: Optional[str] = None
    timeout_sec: int = Field(default=1800, ge=10, le=24 * 3600)

    @model_validator(mode="after")
    def validate_mode(self):
        async_fields = (
            "scenario", "async_pass", "target_qps", "queue_capacity", "worker_count",
            "batch_timeout_ms", "submit_timeout_sec", "flush_timeout_sec",
            "request_timeout_ms", "min_samples", "min_duration_sec", "max_samples",
            "schedule_seed", "latency_slo_ms", "save_request_trace",
            "async_preload_inputs",
        )
        if self.inference_mode != "async_queue":
            if any(getattr(self, name) not in (None, False) for name in async_fields):
                raise ValueError("async_queue 옵션에는 inference_mode=async_queue가 필요합니다")
        elif self.max_steps is not None:
            raise ValueError("async_queue에서는 max_steps 대신 max_samples를 사용합니다")
        if self.scenario == "server_like" and self.target_qps is None:
            raise ValueError("server_like에는 target_qps가 필요합니다")
        for group in (self.compile_options, self.runtime_options):
            for key in group:
                if not key or "=" in key or key.strip() != key:
                    raise ValueError("옵션 key는 공백 없는 key=value 형식이어야 합니다")
        return self


class BenchmarkJobResponse(BaseModel):
    """벤치마크 실행 작업 응답"""
    job_id: str
    status: BenchmarkStatus
    model: str
    backend: str
    device: str
    target_id: Optional[str] = None
    message: str


class BenchmarkJobStatusResponse(BaseModel):
    """벤치마크 작업 상태 응답"""
    job_id: str
    status: BenchmarkStatus
    model: str
    backend: str
    device: str
    target_id: Optional[str] = None
    output: str
    error: Optional[str] = None
    run_id: Optional[str] = None

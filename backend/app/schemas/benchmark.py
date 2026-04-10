from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


class BenchmarkStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


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


class BenchmarkRunRequest(BaseModel):
    """벤치마크 실행 요청"""
    model: str
    backend: str = "onnxruntime"
    device: str = "cpu"
    batch_size: int = Field(default=1, ge=1)
    warmup: int = Field(default=2, ge=0)
    max_steps: Optional[int] = Field(default=None, ge=1)
    layout: str = Field(default="NCHW", pattern="^(NCHW|NHWC)$")
    max_new_tokens: int = Field(default=256, ge=1)
    max_model_len: Optional[int] = None
    gpu_memory_utilization: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    enforce_eager: bool = False
    debug: bool = False


class BenchmarkJobResponse(BaseModel):
    """벤치마크 실행 작업 응답"""
    job_id: str
    status: BenchmarkStatus
    model: str
    backend: str
    device: str
    message: str


class BenchmarkJobStatusResponse(BaseModel):
    """벤치마크 작업 상태 응답"""
    job_id: str
    status: BenchmarkStatus
    model: str
    backend: str
    device: str
    output: str
    error: Optional[str] = None
    run_id: Optional[str] = None

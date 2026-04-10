from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class BenchmarkResultResponse(BaseModel):
    """단일 벤치마크 결과 응답"""
    run_id: str
    timestamp: str
    model_name: str
    task: str
    backend: str
    device: str
    batch_size: int
    warmup_runs: int
    max_steps: Optional[int] = None
    metrics: Dict[str, Any]


class BenchmarkResultListResponse(BaseModel):
    """벤치마크 결과 목록 응답"""
    total: int
    results: List[BenchmarkResultResponse]


class DeleteResultResponse(BaseModel):
    """결과 삭제 응답"""
    success: bool
    message: str

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.runtime_executor import GenerationObservation


@dataclasses.dataclass(frozen=True)
class GenerationResult:
    """자기회귀 생성 결과 컨테이너."""
    generated_ids: np.ndarray  # shape (num_tokens,) 또는 (batch, max_num_tokens), dtype int64
    ttft_ms: float | None      # Time To First Token; None when the runtime cannot measure it
    tpot_ms: float | None      # Time Per Output Token; None when unavailable
    total_ms: float            # 전체 생성 시간
    num_tokens: int            # 생성된 총 토큰 수
    timing_mode: str = "unknown"       # no_kv_full_context | kv_cache | unknown
    uses_kv_cache: bool = False
    timing_source: str = "measured"    # measured | estimated_from_total
    generated_lengths: np.ndarray | None = None  # batch별 실제 생성 길이
    generation_observation: GenerationObservation | None = None

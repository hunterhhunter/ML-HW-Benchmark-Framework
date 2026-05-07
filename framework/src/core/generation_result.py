import dataclasses
import numpy as np

@dataclasses.dataclass(frozen=True)
class GenerationResult:
    """자기회귀 생성 결과 컨테이너."""
    generated_ids: np.ndarray  # shape (num_tokens,) 또는 (batch, max_num_tokens), dtype int64
    ttft_ms: float             # Time To First Token
    tpot_ms: float             # Time Per Output Token / decode-step timing
    total_ms: float            # 전체 생성 시간
    num_tokens: int            # 생성된 총 토큰 수
    timing_mode: str = "unknown"       # no_kv_full_context | kv_cache | unknown
    uses_kv_cache: bool = False
    timing_source: str = "measured"    # measured | estimated_from_total
    generated_lengths: np.ndarray | None = None  # batch별 실제 생성 길이

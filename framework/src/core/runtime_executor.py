import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RuntimeExecution:
    outputs: Optional[Dict[str, Any]]
    timing_ms: float | Dict[str, Any] | None
    generated_tokens: int = 0
    dispatch_token: Optional[int] = None
    vendor_job_id: Any = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class RuntimeExecutor(ABC):
    @abstractmethod
    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        raise NotImplementedError

    @abstractmethod
    def acknowledge(self, execution: RuntimeExecution) -> None:
        raise NotImplementedError

    @abstractmethod
    def shutdown(self, timeout: float) -> bool:
        raise NotImplementedError


class BlockingRuntimeExecutor(RuntimeExecutor):
    def __init__(
        self,
        runtime,
        *,
        is_llm: bool,
        max_new_tokens: int = 256,
        stop_token_ids=None,
    ):
        self.runtime = runtime
        self.is_llm = is_llm
        self.max_new_tokens = max_new_tokens
        self.stop_token_ids = stop_token_ids

    def execute(self, inputs, timeout=None) -> RuntimeExecution:
        if self.is_llm:
            result = self.runtime.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                stop_token_ids=self.stop_token_ids,
            )
            outputs = {"generated_ids": result.generated_ids}
            if result.generated_lengths is not None:
                outputs["generated_lengths"] = result.generated_lengths
            timing = {
                "total_ms": result.total_ms,
                "ttft_ms": result.ttft_ms,
                "tpot_ms": result.tpot_ms,
                "timing_mode": result.timing_mode,
                "uses_kv_cache": result.uses_kv_cache,
                "timing_source": result.timing_source,
            }
            return RuntimeExecution(
                outputs=outputs,
                timing_ms=timing,
                generated_tokens=result.num_tokens,
            )

        started = time.perf_counter()
        outputs = self.runtime.run(inputs)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RuntimeExecution(outputs=outputs, timing_ms=elapsed_ms)

    def acknowledge(self, execution: RuntimeExecution) -> None:
        return None

    def shutdown(self, timeout: float) -> bool:
        return True

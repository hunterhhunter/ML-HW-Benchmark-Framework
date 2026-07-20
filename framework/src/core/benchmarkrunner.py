from typing import Any, Dict

from dataloader.base import DataLoader
from evaluators.base import Evaluator
from runtimes.base import Runtime

from .inference_engine import InferenceEngine


class BenchmarkRunner:
    """
    DataLoader(데이터 공급) -> Runtime(추론 실행) -> Decoder(출력 해석) -> Evaluator(결과 평가)
    전체 파이프라인을 일관되게 관리하는 오케스트레이터 클래스입니다.

    스트리밍 평가(Streaming Evaluation) 패턴을 채택합니다.
    배치마다 Evaluator.add_batch()를 호출하여 무거운 출력 텐서를 즉시 처리·폐기하고,
    루프 종료 후 Evaluator.compute()로 최종 메트릭을 산출합니다.
    이로써 수백만 샘플을 처리해도 RAM 사용량이 선형으로 폭발하지 않습니다.
    """

    def __init__(
        self,
        dataloader: DataLoader,
        runtime: Runtime,
        evaluator: Evaluator,
        max_new_tokens: int = 256,
        monitor=None,
        decoder=None,
    ):
        self.dataloader = dataloader
        self.runtime = runtime
        self.evaluator = evaluator
        self.decoder = decoder
        self._max_new_tokens = max_new_tokens
        self._monitor = monitor
        self._has_run = False

        self._replace_engine()

    def _replace_engine(self) -> None:
        self.engine = InferenceEngine(
            self.dataloader,
            self.runtime,
            self.evaluator,
            decoder=self.decoder,
            max_new_tokens=self._max_new_tokens,
        )

        # Keep legacy private state available to existing integrations.
        self._pipeline = self.engine.pipeline
        self.is_static_batched = self.engine.pipeline.is_static_batched
        self._stop_token_ids = self.engine.pipeline.stop_token_ids

    def _collate_batch(self, batch_list: Any) -> Dict[str, Any]:
        """List of single sample dicts -> Batched dict (np.stack)."""
        return self.engine.pipeline.collate_batch(batch_list)

    def _prepare_runtime_input(
        self,
        collated_input: Any,
        fallback_name: str,
    ) -> Dict[str, Any]:
        """로더가 던진 input이 딕셔너리(Multi-input)면 통째로 반환, 아니면 단일 노드(Single-input) 이름으로 래핑합니다."""
        del fallback_name
        return self.engine.pipeline.prepare_runtime_input(collated_input)

    def _prepare_eval_labels(self, collated: Dict[str, Any]) -> Any:
        """Attach optional per-sample preprocessing metadata for evaluators that need coordinate recovery."""
        return self.engine.pipeline.prepare_eval_labels(collated)

    def _handle_engine_event(self, event: str, **details: Any) -> None:
        if event == "limit_reached":
            max_steps = details["max_steps"]
            print(
                "[BenchmarkRunner] 🛑 사용자가 요청한 리미터에 "
                f"도달했습니다! ({max_steps} steps) - "
                "즉각 탈출하여 결과를 채점합니다."
            )
            return

        if event == "before_compute":
            print("[BenchmarkRunner] 🏆 Computing final metrics...")
            return

        if event != "batch_complete":
            return

        batch_idx = details["batch_idx"]
        if batch_idx % 10 != 0:
            return

        latency_ms = details["timing_ms"]
        if isinstance(latency_ms, dict):
            latency_display = (
                f"total={latency_ms.get('total_ms', 0):.2f} ms, "
                f"ttft={latency_ms.get('ttft_ms', 0):.2f} ms, "
                f"tpot={latency_ms.get('tpot_ms', 0):.2f} ms, "
                f"mode={latency_ms.get('timing_mode', 'unknown')}, "
                f"source={latency_ms.get('timing_source', 'measured')}"
            )
        else:
            latency_display = f"{latency_ms:.2f} ms"
        actual_batch_size = details["actual_batch_size"]
        print(
            f"  - Completed batch {batch_idx} ({actual_batch_size} samples), "
            f"Latency: {latency_display}"
        )

    def run(
        self,
        warmup_runs: int = 1,
        batch_size: int = 1,
        max_steps: int = None,
    ) -> Dict[str, Any]:
        """
        주입된 컴포넌트들을 연결하여 벤치마크 테스트 전체 루프를 수행합니다.

        Args:
            warmup_runs (int): 본 측정 전 Runtime 엔진을 예열하기 위한 횟수
            batch_size (int): 한 번에 묶어서 추론을 보낼 갯수
            max_steps (int): 옵션 - 지정된 횟수만큼만 루프를 돌고 탈출(테스트/리미터용)

        Returns:
            Dict[str, Any]: 최종 성능 종합 메트릭 리포트 (Evaluator.compute() 반환값)
        """
        print("[BenchmarkRunner] 🚀 Starts benchmarking...")

        if self._has_run:
            self._replace_engine()
        self._has_run = True

        if warmup_runs > 0:
            print(f"[BenchmarkRunner] 🌡️ Warming up {warmup_runs} times...")
        self.engine.warmup(runs=warmup_runs, batch_size=batch_size)

        if self.engine.pipeline.is_llm:
            print(
                "[BenchmarkRunner] 🤖 LLM 감지 (NLP_GENERATION) — "
                "generate() 경로 사용 "
                f"(max_new_tokens={self._max_new_tokens})"
            )

        # 하드웨어 모니터링 시작 (warmup 제외, inference만 측정)
        if self._monitor:
            self._monitor.start()

        print("[BenchmarkRunner] ⚡ Running inference loop (streaming evaluation)...")
        try:
            metrics = self.engine.run_e2e(
                batch_size=batch_size,
                max_steps=max_steps,
                event_callback=self._handle_engine_event,
            )
        finally:
            if self._monitor:
                self._monitor.stop()

        # 하드웨어 메트릭 병합 (hw_ prefix로 키 충돌 없음)
        if self._monitor:
            hw_metrics = self._monitor.summary()
            metrics.update(hw_metrics)

        return metrics

    def _reset_dataloader_cursor(self) -> None:
        """DataLoader 구현별 cursor 이름 차이를 흡수해 warmup 후 처음부터 재순회합니다."""
        self.engine.pipeline.reset_dataloader_cursor()

"""Furiosa-LLM runtime for explicit or SDK-resolved RNGD artifacts."""

from __future__ import annotations

import asyncio
from concurrent.futures import wait as wait_futures
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from transformers import BatchEncoding

from core.compiled_model import CompiledModel
from core.generation_result import GenerationResult
from core.runtime_executor import (
    GenerationObservation,
    GenerationOutputEvent,
    NativeAsyncOutcome,
)
from .base import Runtime


class FuriosaNativeBackend:
    """Callback backend backed by Furiosa's ``AsyncLLMEngine`` stream API."""

    def __init__(
        self,
        runtime: "FuriosaLlmRuntime",
        *,
        max_new_tokens: int,
        stop_token_ids=None,
    ):
        if runtime._llm is None or runtime._sampling_params_cls is None:
            raise RuntimeError("Furiosa-LLM engine must be loaded before native async setup.")
        self.runtime = runtime
        self.max_new_tokens = int(max_new_tokens)
        self.stop_token_ids = (
            None
            if stop_token_ids is None
            else [int(token_id) for token_id in stop_token_ids]
        )
        self._lock = threading.RLock()
        self._futures = {}
        self._aborted_request_ids = set()
        self._shutdown_request_ids = set()
        self._closing = False
        self._ready = threading.Event()
        self._loop = None
        self._async_engine = None
        self._startup_error = None
        self._thread = threading.Thread(
            target=self._run_loop,
            name="furiosa-native-loop",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise TimeoutError("Furiosa native async loop failed to start.")
        if self._startup_error is not None:
            raise RuntimeError("Furiosa AsyncLLMEngine initialization failed.") from self._startup_error

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            from furiosa_llm import AsyncLLMEngine

            self._async_engine = AsyncLLMEngine.from_llm(self.runtime._llm)
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def submit_async(self, inputs, callback):
        prompt_batches = self.runtime._trim_prompt_tokens(inputs)
        if len(prompt_batches) != 1:
            raise ValueError(
                "Furiosa native async accepts one request at a time; "
                "continuous batching is owned by Furiosa-LLM."
            )
        request_id = f"rngd-{uuid.uuid4().hex}"
        sampling_params = self.runtime._sampling_params_cls(
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            stop_token_ids=self.stop_token_ids,
        )
        started_ns = time.monotonic_ns()
        with self._lock:
            if self._closing:
                raise RuntimeError("Furiosa native backend is shutting down.")
            future = asyncio.run_coroutine_threadsafe(
                self._consume_request(
                    prompt_batches[0],
                    sampling_params,
                    request_id,
                    started_ns,
                    callback,
                ),
                self._loop,
            )
            self._futures[request_id] = future

            def retire(_):
                with self._lock:
                    self._futures.pop(request_id, None)

            future.add_done_callback(retire)
        return request_id

    async def _abort_request(self, request_id: str) -> None:
        with self._lock:
            if request_id in self._aborted_request_ids:
                return
            self._aborted_request_ids.add(request_id)
        try:
            await self._async_engine.abort(request_id)
        except BaseException:
            with self._lock:
                self._aborted_request_ids.discard(request_id)
            raise

    async def _abort_requests(self, request_ids: Sequence[str]) -> None:
        for request_id in request_ids:
            await self._abort_request(request_id)

    async def _consume_request(
        self,
        prompt_token_ids,
        sampling_params,
        request_id,
        started_ns,
        callback,
    ) -> None:
        emitted = False

        def emit_once(outcome: NativeAsyncOutcome) -> None:
            nonlocal emitted
            if emitted:
                return
            emitted = True
            callback(outcome)

        first_token_ns = None
        final_output_ns = None
        final_output = None
        previous_cumulative_tokens = 0
        generation_events = []
        try:
            stream = self._async_engine.generate(
                {"prompt_token_ids": prompt_token_ids},
                sampling_params,
                request_id,
            )
            async for request_output in stream:
                final_output = request_output
                token_ids = self._extract_token_ids(request_output)
                cumulative_tokens = len(token_ids)
                if cumulative_tokens < previous_cumulative_tokens:
                    raise RuntimeError(
                        "Furiosa cumulative stream token count decreased."
                    )
                if cumulative_tokens > previous_cumulative_tokens:
                    observed_ns = time.monotonic_ns()
                    generation_events.append(
                        GenerationOutputEvent(
                            observed_ns=observed_ns,
                            cumulative_tokens=cumulative_tokens,
                        )
                    )
                    if first_token_ns is None:
                        first_token_ns = observed_ns
                    final_output_ns = observed_ns
                    previous_cumulative_tokens = cumulative_tokens

            with self._lock:
                shutdown_requested = request_id in self._shutdown_request_ids
            if shutdown_requested and final_output is None:
                emit_once(NativeAsyncOutcome(
                    error_type="FuriosaAsyncShutdown",
                    error_message="Furiosa async generation stopped during shutdown.",
                ))
                return

            finished_ns = final_output_ns or time.monotonic_ns()
            if final_output is None:
                generated_ids = np.zeros((1, 0), dtype=np.int64)
                generated_lengths = np.zeros((1,), dtype=np.int64)
            else:
                generated_ids, generated_lengths = self.runtime._normalize_outputs(
                    final_output
                )
            generated_tokens = int(generated_lengths.sum())
            timing_ms: Dict[str, Any] = {
                "total_ms": (finished_ns - started_ns) / 1_000_000.0,
                "timing_mode": "kv_cache",
                "uses_kv_cache": True,
                "timing_source": "furiosa_async_python_stream",
            }
            if first_token_ns is not None:
                timing_ms["ttft_ms"] = (
                    first_token_ns - started_ns
                ) / 1_000_000.0
                timing_ms["tpot_ms"] = (
                    None
                    if generated_tokens <= 1
                    else (finished_ns - first_token_ns)
                    / (generated_tokens - 1)
                    / 1_000_000.0
                )
            emit_once(NativeAsyncOutcome(
                outputs={
                    "generated_ids": generated_ids,
                    "generated_lengths": generated_lengths,
                },
                timing_ms=timing_ms,
                generated_tokens=generated_tokens,
                generation_observation=GenerationObservation(
                    backend_submitted_ns=started_ns,
                    events=tuple(generation_events),
                    source="furiosa_async_python_stream",
                ),
            ))
        except BaseException as exc:
            try:
                await self._abort_request(request_id)
            except BaseException:
                pass
            with self._lock:
                shutdown_requested = request_id in self._shutdown_request_ids
            error_type = (
                "FuriosaAsyncShutdown"
                if shutdown_requested
                else type(exc).__name__
            )
            error_message = (
                "Furiosa async generation stopped during shutdown."
                if shutdown_requested
                else "Furiosa async generation failed "
                f"({type(exc).__name__})."
            )
            emit_once(NativeAsyncOutcome(
                error_type=error_type,
                error_message=error_message,
            ))
        finally:
            with self._lock:
                self._aborted_request_ids.discard(request_id)
                self._shutdown_request_ids.discard(request_id)

    @staticmethod
    def _extract_token_ids(request_output) -> list[int]:
        completions = getattr(request_output, "outputs", None)
        if not completions:
            return []
        return [int(token) for token in completions[0].token_ids]

    def shutdown(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._closing = True
            if not self._thread.is_alive():
                return True
            requests = tuple(self._futures.items())
            self._shutdown_request_ids.update(
                request_id for request_id, _ in requests
            )

        loop = self._loop
        if requests:
            request_ids = tuple(request_id for request_id, _ in requests)
            abort_future = asyncio.run_coroutine_threadsafe(
                self._abort_requests(request_ids),
                loop,
            )
            remaining = max(0.0, deadline - time.monotonic())
            done, pending = wait_futures((abort_future,), timeout=remaining)
            if pending:
                abort_future.cancel()
                return False
            try:
                next(iter(done)).result()
            except BaseException:
                return False

            futures = tuple(future for _, future in requests)
            for future in futures:
                future.cancel()
            remaining = max(0.0, deadline - time.monotonic())
            _, pending = wait_futures(futures, timeout=remaining)
            if pending:
                return False

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive()


class FuriosaLlmRuntime(Runtime):
    """Run Hugging Face models with an explicit or SDK-resolved artifact."""

    def __init__(self, **runtime_options):
        self.device = runtime_options.get("device", "npu:0")
        self.devices = runtime_options.get("devices", self.device)
        self.data_parallel_size = runtime_options.get("data_parallel_size")
        self.pipeline_parallel_size = runtime_options.get("pipeline_parallel_size")
        self.max_io_memory_mb = runtime_options.get("max_io_memory_mb", 2048)
        self.seed = runtime_options.get("seed")
        self.cache_dir = runtime_options.get("cache_dir")
        self.npu_queue_limit = runtime_options.get("npu_queue_limit")
        self.max_processing_samples = runtime_options.get("max_processing_samples")
        self.spare_blocks_ratio = runtime_options.get("spare_blocks_ratio")

        self._llm = None
        self._sampling_params_cls = None
        self.compiled_model: CompiledModel | None = None
        self._model_path = ""
        self._native_backend: FuriosaNativeBackend | None = None

    def load(self, compiled_model: CompiledModel) -> None:
        try:
            from furiosa_llm import LLM, SamplingParams
            from furiosa_llm.api import SchedulerConfig
        except ImportError as exc:
            raise ImportError(
                "Furiosa RNGD 실행에는 furiosa-llm 전용 환경이 필요합니다. "
                "Furiosa SDK 2026.3 설치 문서를 따라 furiosa-llm을 설치하세요."
            ) from exc

        model_path = compiled_model.spec.model_paths.get("hf_model")
        if not model_path:
            raise ValueError(
                "Furiosa-LLM requires CompiledModel.spec.model_paths['hf_model']."
            )

        scheduler_values = {
            "npu_queue_limit": self.npu_queue_limit,
            "max_processing_samples": self.max_processing_samples,
            "spare_blocks_ratio": self.spare_blocks_ratio,
        }
        scheduler_values = {
            name: value for name, value in scheduler_values.items() if value is not None
        }

        llm_kwargs: Dict[str, Any] = {
            "devices": self.devices,
            "max_io_memory_mb": self.max_io_memory_mb,
        }
        if compiled_model.artifact_path is not None:
            llm_kwargs["fxb"] = str(compiled_model.artifact_path)
        optional_values = {
            "data_parallel_size": self.data_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "seed": self.seed,
            "cache_dir": self.cache_dir,
        }
        llm_kwargs.update(
            {name: value for name, value in optional_values.items() if value is not None}
        )
        if scheduler_values:
            llm_kwargs["scheduler_config"] = SchedulerConfig(**scheduler_values)

        self._llm = LLM(str(model_path), **llm_kwargs)
        self._sampling_params_cls = SamplingParams
        self.compiled_model = compiled_model
        self._model_path = str(model_path)

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = self.generate(inputs, max_new_tokens=1)
        outputs = {"generated_ids": result.generated_ids}
        if result.generated_lengths is not None:
            outputs["generated_lengths"] = result.generated_lengths
        return outputs

    def supports_generate(self) -> bool:
        return True

    def native_async_max_batch_size(self) -> int:
        return 1

    def supports_batch_generation(self) -> bool:
        return True

    def supports_streaming_generate(self) -> bool:
        return True

    def generate(
        self,
        inputs: Dict[str, np.ndarray],
        max_new_tokens: int = 256,
        stop_token_ids: Optional[List[int]] = None,
    ) -> GenerationResult:
        if self._llm is None or self._sampling_params_cls is None:
            raise RuntimeError("Furiosa-LLM engine is not loaded. Call load() first.")

        prompt_token_ids = self._trim_prompt_tokens(inputs)
        normalized_stop_ids = None
        if stop_token_ids is not None:
            normalized_stop_ids = [int(token_id) for token_id in stop_token_ids]
        sampling_params = self._sampling_params_cls(
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=normalized_stop_ids,
        )
        batch_encoding = BatchEncoding({"input_ids": prompt_token_ids})

        started = time.perf_counter()
        raw_outputs = self._llm.generate(
            [""] * len(prompt_token_ids),
            sampling_params=sampling_params,
            prompt_token_ids=batch_encoding,
        )
        total_ms = (time.perf_counter() - started) * 1000.0

        generated_ids, generated_lengths = self._normalize_outputs(raw_outputs)
        return GenerationResult(
            generated_ids=generated_ids,
            generated_lengths=generated_lengths,
            ttft_ms=None,
            tpot_ms=None,
            total_ms=total_ms,
            num_tokens=int(generated_lengths.sum()),
            timing_mode="kv_cache",
            uses_kv_cache=True,
            timing_source="wall_clock_total_only",
        )

    @staticmethod
    def _trim_prompt_tokens(inputs: Dict[str, np.ndarray]) -> list[list[int]]:
        if "input_ids" not in inputs:
            raise ValueError("Furiosa-LLM inputs must contain 'input_ids'.")

        input_ids = np.asarray(inputs["input_ids"])
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 1D or 2D, got shape={input_ids.shape}."
            )

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            return [[int(token) for token in row] for row in input_ids]

        attention_mask = np.asarray(attention_mask)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.reshape(1, -1)
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask shape must match input_ids shape: "
                f"{attention_mask.shape} != {input_ids.shape}."
            )
        return [
            [int(token) for token in row[mask.astype(bool)]]
            for row, mask in zip(input_ids, attention_mask)
        ]

    @staticmethod
    def _normalize_outputs(raw_outputs: Any) -> tuple[np.ndarray, np.ndarray]:
        request_outputs: Sequence[Any]
        if isinstance(raw_outputs, (list, tuple)):
            request_outputs = raw_outputs
        else:
            request_outputs = [raw_outputs]

        token_batches: list[list[int]] = []
        for request_output in request_outputs:
            completions = getattr(request_output, "outputs", None)
            if not completions:
                token_batches.append([])
                continue
            token_batches.append([int(token) for token in completions[0].token_ids])

        lengths = np.asarray([len(tokens) for tokens in token_batches], dtype=np.int64)
        max_length = int(lengths.max()) if len(lengths) else 0
        padded = np.zeros((len(token_batches), max_length), dtype=np.int64)
        for index, tokens in enumerate(token_batches):
            if tokens:
                padded[index, : len(tokens)] = tokens
        return padded, lengths

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(max(0, num_runs)):
            self.generate(inputs, max_new_tokens=1)

    def create_native_backend(
        self,
        *,
        max_new_tokens: int,
        stop_token_ids=None,
    ) -> FuriosaNativeBackend:
        if self._native_backend is not None:
            return self._native_backend
        self._native_backend = FuriosaNativeBackend(
            self,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        return self._native_backend

    def unload(self) -> None:
        native_backend = self._native_backend
        if native_backend is not None:
            if not native_backend.shutdown(timeout=5.0):
                raise RuntimeError(
                    "Furiosa native async backend did not stop; "
                    "LLM shutdown was skipped."
                )
            self._native_backend = None
        llm = self._llm
        if llm is not None:
            llm.shutdown()
        self._llm = None
        self._sampling_params_cls = None
        self.compiled_model = None
        self._model_path = ""

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "furiosa_llm",
            "device": self.device,
            "devices": self.devices,
            "accelerator_vendor": "FuriosaAI",
            "accelerator_name": "RNGD",
            "model_path": self._model_path,
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        if compiled_model.backend_name.lower() in {
            "furiosa_llm",
            "furiosa",
            "rngd",
        }:
            return True
        artifact_path = compiled_model.artifact_path
        return (
            artifact_path is not None
            and artifact_path.suffix.lower() == ".fxb"
        )

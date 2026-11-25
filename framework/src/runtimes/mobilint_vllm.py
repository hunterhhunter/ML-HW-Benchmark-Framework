"""Mobilint Llama generation through the official vLLM platform plugin."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future
from contextlib import aclosing
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np

from core.compiled_model import CompiledModel
from core.generation_result import GenerationResult
from core.model_spec import Task
from core.runtime_executor import (
    GenerationObservation,
    GenerationOutputEvent,
    NativeAsyncOutcome,
)
from .base import Runtime

_TIMING_SOURCE = "mobilint_vllm_async_python_stream"
_ENGINE_START_LOCK = threading.Lock()


@dataclass(frozen=True)
class _GenerationOptions:
    max_tokens: int
    stop_token_ids: tuple[int, ...]

    @classmethod
    def from_request(cls, max_new_tokens, stop_token_ids):
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        stops = () if stop_token_ids is None else tuple(int(token) for token in stop_token_ids)
        return cls(max_new_tokens, stops)


def _prompt_token_ids(inputs: dict[str, np.ndarray]) -> list[int]:
    ids = np.asarray(inputs.get("input_ids"))
    if ids.ndim == 2 and ids.shape[0] == 1:
        ids = ids[0]
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("Mobilint vLLM requires one integer input_ids sequence")
    if "attention_mask" in inputs:
        mask = np.asarray(inputs["attention_mask"])
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.shape != ids.shape or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("attention_mask must be binary and match input_ids")
        ids = ids[mask.astype(bool)]
    if ids.size == 0 or np.any(ids < 0):
        raise ValueError("Mobilint vLLM requires a nonempty prompt with valid token IDs")
    return ids.tolist()


class _Engine:
    """Own the vLLM engine and all its requests on one asyncio loop."""

    def __init__(self, engine_options: dict, request_timeout: float, shutdown_timeout: float):
        self._options = engine_options
        self._request_timeout = request_timeout
        self._shutdown_timeout = shutdown_timeout
        self._thread = threading.Thread(target=self._run, name="mobilint-vllm", daemon=True)
        self._ready: Future[dict] = Future()
        self._lock = threading.Lock()
        self._closing = False
        self._loop = None
        self._stop = None
        self._shutdown_error = None
        self._requests: set[asyncio.Task] = set()

    def start(self, timeout: float) -> dict:
        self._thread.start()
        try:
            return self._ready.result(timeout)
        except TimeoutError as exc:
            if self._ready.done():
                raise
            raise TimeoutError("Mobilint vLLM engine startup timed out") from exc

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            if self._ready.done():
                self._shutdown_error = exc
            else:
                self._ready.set_exception(exc)

    async def _serve(self) -> None:
        try:
            plugin_version = version("vllm-mblt")
            vllm_version = version("vllm")
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.platforms import current_platform
            from vllm.sampling_params import RequestOutputKind
            from vllm.v1.engine.async_llm import AsyncLLM
            from vllm_mblt.mblt_platform import MbltPlatform
        except ImportError as exc:
            raise RuntimeError(
                "Mobilint vLLM requires the dedicated vllm-mblt environment"
            ) from exc
        if not isinstance(current_platform, MbltPlatform):
            raise RuntimeError(
                "vLLM did not activate the Mobilint platform. "
                "Use a dedicated environment with the mblt and mblt_models plugins enabled."
            )

        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        with self._lock:
            self._loop = loop
            self._stop = stop
            if self._closing:
                raise RuntimeError("Mobilint vLLM startup was cancelled")

        # vLLM starts its EngineCore from this thread. Spawn avoids forking
        # the framework's active NumPy, tokenizer and Torch state.
        with _ENGINE_START_LOCK:
            method_var = "VLLM_WORKER_MULTIPROC_METHOD"
            previous = os.environ.get(method_var)
            os.environ[method_var] = "spawn"
            try:
                engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**self._options))
            finally:
                if previous is None:
                    os.environ.pop(method_var, None)
                else:
                    os.environ[method_var] = previous
        try:
            self._engine = engine
            self._sampling_params = SamplingParams
            self._output_kind = RequestOutputKind.CUMULATIVE
            scheduler = engine.vllm_config.scheduler_config
            self._ready.set_result({
                "vllm_version": vllm_version,
                "vllm_mblt_version": plugin_version,
                "max_num_seqs": scheduler.max_num_seqs,
                "max_num_batched_tokens": scheduler.max_num_batched_tokens,
                "enable_chunked_prefill": scheduler.enable_chunked_prefill,
                "enable_prefix_caching": engine.vllm_config.cache_config.enable_prefix_caching,
                "max_model_len": engine.vllm_config.model_config.max_model_len,
            })
            await stop.wait()
        finally:
            requests = tuple(self._requests)
            for request in requests:
                request.cancel()
            try:
                await asyncio.gather(*requests, return_exceptions=True)
            finally:
                try:
                    engine.shutdown(timeout=self._shutdown_timeout)
                except BaseException as exc:
                    self._shutdown_error = exc
                    raise

    def submit(self, prompt: list[int], options: _GenerationOptions):
        request_id = uuid.uuid4().hex
        with self._lock:
            if self._closing or not self._thread.is_alive():
                raise RuntimeError("Mobilint vLLM engine is not accepting requests")
            submitted_ns = time.monotonic_ns()
            request = self._generate(request_id, prompt, options, submitted_ns)
            try:
                future = asyncio.run_coroutine_threadsafe(request, self._loop)
            except RuntimeError:
                request.close()
                raise
        return request_id, future

    async def _generate(self, request_id, prompt, options, submitted_ns) -> GenerationResult:
        task = asyncio.current_task()
        self._requests.add(task)
        events = []
        tokens = ()
        final_output = None
        try:
            sampling = self._sampling_params(
                max_tokens=options.max_tokens,
                temperature=0.0,
                stop_token_ids=list(options.stop_token_ids),
                detokenize=False,
                output_kind=self._output_kind,
            )
            async with asyncio.timeout(self._request_timeout):
                stream = self._engine.generate(
                    {"prompt_token_ids": prompt}, sampling, request_id
                )
                async with aclosing(stream):
                    async for output in stream:
                        final_output = output
                        if not output.outputs:
                            continue
                        tokens = output.outputs[0].token_ids
                        previous_count = events[-1].cumulative_tokens if events else 0
                        if len(tokens) < previous_count:
                            raise RuntimeError("vLLM cumulative token count decreased")
                        if len(tokens) > previous_count:
                            events.append(GenerationOutputEvent(time.monotonic_ns(), len(tokens)))
            finished_ns = time.monotonic_ns()
        finally:
            self._requests.discard(task)
        if final_output is None or not final_output.finished or not final_output.outputs:
            raise RuntimeError("Mobilint vLLM stream ended without a final completion")

        finish_reason = final_output.outputs[0].finish_reason
        if finish_reason in ("abort", "error"):
            raise RuntimeError(f"Mobilint vLLM request finished with reason: {finish_reason}")

        count = len(tokens)
        return GenerationResult(
            generated_ids=np.asarray(tokens, dtype=np.int64).reshape(1, -1),
            generated_lengths=np.asarray([count], dtype=np.int64),
            ttft_ms=(events[0].observed_ns - submitted_ns) / 1e6 if events else None,
            tpot_ms=(
                (events[-1].observed_ns - events[0].observed_ns) / (count - 1) / 1e6
                if count > 1 and events[0].cumulative_tokens == 1
                else None
            ),
            total_ms=(finished_ns - submitted_ns) / 1e6,
            num_tokens=count,
            timing_mode="kv_cache",
            uses_kv_cache=True,
            timing_source=_TIMING_SOURCE,
            generation_observation=GenerationObservation(
                backend_submitted_ns=submitted_ns,
                events=tuple(events),
                source=_TIMING_SOURCE,
            ),
        )

    def shutdown(self) -> None:
        with self._lock:
            self._closing = True
            if self._loop is not None:
                try:
                    self._loop.call_soon_threadsafe(self._stop.set)
                except RuntimeError:
                    # A failed startup or previous shutdown can already have closed it.
                    if not self._loop.is_closed():
                        raise
        if self._thread.ident is not None:
            self._thread.join(self._shutdown_timeout)
        if self._thread.is_alive():
            raise TimeoutError("Mobilint vLLM engine shutdown timed out")
        if self._shutdown_error is not None:
            raise RuntimeError("Mobilint vLLM engine shutdown failed") from self._shutdown_error


@dataclass(frozen=True)
class _NativeBackend:
    engine: _Engine
    options: _GenerationOptions

    def submit_async(self, inputs, callback):
        if not callable(callback):
            raise ValueError("callback must be callable")
        request_id, future = self.engine.submit(_prompt_token_ids(inputs), self.options)

        def complete(result_future):
            try:
                result = result_future.result()
            except Exception as exc:
                outcome = NativeAsyncOutcome(
                    error_type=type(exc).__name__[:256],
                    error_message=str(exc)[:512] or f"Mobilint vLLM request {type(exc).__name__}",
                )
            else:
                outcome = NativeAsyncOutcome(
                    outputs=_generation_outputs(result),
                    timing_ms={
                        "ttft_ms": result.ttft_ms,
                        "tpot_ms": result.tpot_ms,
                        "total_ms": result.total_ms,
                        "timing_mode": result.timing_mode,
                        "uses_kv_cache": result.uses_kv_cache,
                        "timing_source": result.timing_source,
                    },
                    generated_tokens=result.num_tokens,
                    generation_observation=result.generation_observation,
                )
            callback(outcome)

        future.add_done_callback(complete)
        return request_id


def _generation_outputs(result: GenerationResult) -> dict[str, np.ndarray]:
    return {
        "generated_ids": result.generated_ids,
        "generated_lengths": result.generated_lengths,
    }


class MobilintVllmRuntime(Runtime):
    """Run a local Mobilint Model Zoo Llama on ARIES device 0."""

    def __init__(
        self, device="0", device_id=0, expected_family="aries", *,
        tokenizer_path=None, max_model_len=None, max_num_seqs=None,
        enable_prefix_caching=None, startup_timeout_sec=600.0,
        shutdown_timeout_sec=300.0, request_timeout_sec=240.0,
    ):
        if str(device) != "0" or type(device_id) is not int or device_id != 0 or expected_family != "aries":
            raise ValueError("Mobilint vLLM requires ARIES device 0")
        for name, value in (("max_model_len", max_model_len), ("max_num_seqs", max_num_seqs)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if enable_prefix_caching is not None and type(enable_prefix_caching) is not bool:
            raise ValueError("enable_prefix_caching must be a boolean")
        for name, value in (
            ("startup_timeout_sec", startup_timeout_sec),
            ("shutdown_timeout_sec", shutdown_timeout_sec),
            ("request_timeout_sec", request_timeout_sec),
        ):
            if type(value) not in (int, float) or not 0 < value <= threading.TIMEOUT_MAX:
                raise ValueError(f"{name} must be a positive finite number")
        self._tokenizer_path = tokenizer_path
        self._engine_options = {
            key: value for key, value in {
                "max_model_len": max_model_len,
                "max_num_seqs": max_num_seqs,
                "enable_prefix_caching": enable_prefix_caching,
            }.items() if value is not None
        }
        self._startup_timeout = startup_timeout_sec
        self._shutdown_timeout = shutdown_timeout_sec
        self._request_timeout = request_timeout_sec
        self._lock = threading.Lock()
        self._engine = None
        self._engine_diagnostics = {}

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        path = compiled_model.artifact_path
        return compiled_model.spec.task is Task.NLP_GENERATION and path is not None and path.is_dir()

    def load(self, compiled_model: CompiledModel) -> None:
        with self._lock:
            if self._engine is not None:
                raise RuntimeError("Mobilint vLLM is already loaded or requires unload")
            if not self.is_compatible(compiled_model):
                raise ValueError("Mobilint vLLM requires a local generation model directory")
            path = compiled_model.artifact_path.resolve()
            try:
                config = json.loads((path / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("Mobilint vLLM requires a valid model config.json") from exc
            if not isinstance(config, dict) or config.get("model_type") != "mobilint-llama":
                raise ValueError("Mobilint vLLM requires a Mobilint Model Zoo Llama artifact")
            if not any(mxq.is_file() for mxq in path.rglob("*.mxq")):
                raise ValueError("Mobilint vLLM model directory contains no MXQ artifacts")

            visible = os.environ.get("QBRUNTIME_VISIBLE_DEVICES")
            if visible is not None and visible.split(",")[0].strip() != "0":
                raise ValueError("QBRUNTIME_VISIBLE_DEVICES must keep ARIES device 0 first")
            tokenizer = Path(self._tokenizer_path or path).expanduser().resolve()
            if not tokenizer.is_dir():
                raise ValueError("Mobilint vLLM requires a local tokenizer directory")
            options = {
                "model": str(path),
                "tokenizer": str(tokenizer),
                "trust_remote_code": True,
                "model_loader_extra_config": {"dev_no": 0},
                **self._engine_options,
            }
            engine = _Engine(options, self._request_timeout, self._shutdown_timeout)
            self._engine = engine
            try:
                self._engine_diagnostics = engine.start(self._startup_timeout)
            except BaseException as startup_error:
                try:
                    engine.shutdown()
                except Exception as shutdown_error:
                    raise RuntimeError(
                        f"Mobilint vLLM startup failed and cleanup is incomplete: {shutdown_error}"
                    ) from startup_error
                self._engine = None
                raise

    def _loaded_engine(self) -> _Engine:
        with self._lock:
            if not self._engine_diagnostics:
                raise RuntimeError("Mobilint vLLM must be loaded before generating")
            return self._engine

    def supports_generate(self) -> bool:
        return True

    def native_async_max_batch_size(self) -> int:
        return 1

    def create_native_backend(self, *, max_new_tokens, stop_token_ids=None):
        return _NativeBackend(
            self._loaded_engine(),
            _GenerationOptions.from_request(max_new_tokens, stop_token_ids),
        )

    def generate(self, inputs, max_new_tokens=256, stop_token_ids=None) -> GenerationResult:
        engine = self._loaded_engine()
        options = _GenerationOptions.from_request(max_new_tokens, stop_token_ids)
        _, future = engine.submit(_prompt_token_ids(inputs), options)
        try:
            return future.result()
        except KeyboardInterrupt:
            future.cancel()
            raise

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return _generation_outputs(self.generate(inputs, max_new_tokens=1))

    def warmup(self, inputs: dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(num_runs):
            self.generate(inputs, max_new_tokens=1)

    def unload(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.shutdown()
                self._engine = None
                self._engine_diagnostics = {}

    def get_device_spec(self) -> dict:
        return {
            "backend": "mobilint_vllm",
            "device": "0",
            "expected_family": "aries",
            **self._engine_diagnostics,
        }

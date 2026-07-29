"""In-process vLLM RBLN generation runtime."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import wait as wait_futures
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from core.compiled_model import CompiledModel
from core.generation_result import GenerationResult
from core.runtime_executor import (
    GenerationObservation,
    GenerationOutputEvent,
    NativeAsyncOutcome,
)

from .base import Runtime
from .vllm_rt import VllmRuntime

_DEVICE_COUNT_ENV = "VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"
_ENGINE_ENV_LOCK = threading.Lock()
_GIB = 1024**3
_MANIFEST_NAME = "rbln-vllm-manifest.json"


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in int")
    return value


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a built-in bool")
    return value


def _decoder_batch_sizes(value: object, max_num_seqs: int) -> list[int] | None:
    if value is None or value == "":
        return None
    if type(value) is str:
        values: Sequence[object] = tuple(
            part.strip() for part in value.split(",") if part.strip()
        )
        try:
            values = tuple(int(part) for part in values)
        except ValueError as exc:
            raise ValueError(
                "decoder_batch_sizes must contain positive integers"
            ) from exc
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError(
            "decoder_batch_sizes must be a list or comma-separated integers"
        )
    result = [_positive_int(item, "decoder_batch_sizes") for item in values]
    if not result or any(item > max_num_seqs for item in result):
        raise ValueError(
            "decoder_batch_sizes values must not exceed max_num_seqs"
        )
    return sorted(set(result), reverse=True)


def _json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"invalid {description}: expected a JSON object")
    return payload


def _recognized_model_kinds(*values: object) -> set[str]:
    identity = " ".join(
        value.casefold() for value in values if isinstance(value, str)
    )
    return {
        model_kind
        for model_kind in ("llama-3.2-3b", "llama-3.1-8b")
        if model_kind in identity
    }


def _resolve_model_kind(
    compiled_model: CompiledModel,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
) -> str:
    requested = _recognized_model_kinds(compiled_model.spec.name)
    artifact = _recognized_model_kinds(
        config.get("_name_or_path"),
        manifest.get("model") if manifest is not None else None,
        manifest.get("model_id") if manifest is not None else None,
    )
    if len(requested) > 1 or len(artifact) > 1:
        raise ValueError("RBLN vLLM model identity is ambiguous")
    requested_kind = next(iter(requested), None)
    artifact_kind = next(iter(artifact), None)
    if (
        requested_kind is not None
        and artifact_kind is not None
        and requested_kind != artifact_kind
    ):
        raise ValueError(
            "RBLN vLLM model identity mismatch: requested "
            f"{requested_kind}, artifact is {artifact_kind}"
        )
    return artifact_kind or "other"


def _memory_bytes(value: object) -> int | None:
    if type(value) is int:
        return value if value >= 0 else None
    if type(value) is str:
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


class RblnVllmNativeBackend:
    """Callback adapter over one vLLM `AsyncLLMEngine`."""

    def __init__(
        self,
        runtime: RblnVllmRuntime,
        *,
        max_new_tokens: int,
        stop_token_ids=None,
    ):
        if runtime.compiled_model is None:
            raise RuntimeError("RBLN vLLM runtime must be loaded first")
        self.runtime = runtime
        self.max_new_tokens = _positive_int(
            max_new_tokens, "max_new_tokens"
        )
        self.stop_token_ids = (
            None
            if stop_token_ids is None
            else [int(token_id) for token_id in stop_token_ids]
        )
        self._lock = threading.RLock()
        self._futures: dict[str, Any] = {}
        self._aborted_request_ids: set[str] = set()
        self._shutdown_request_ids: set[str] = set()
        self._closing = False
        self._ready = threading.Event()
        self._loop = None
        self._async_engine = None
        self._sampling_params_cls = None
        self._startup_error: BaseException | None = None
        self._shutdown_error: BaseException | None = None
        self._started = False
        self._thread = threading.Thread(
            target=self._run_loop,
            name="rbln-vllm-native-loop",
            daemon=True,
        )

    def start(self, timeout: float) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("RBLN vLLM native backend is shutting down")
            if not self._started:
                self._started = True
                self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError("RBLN vLLM async loop failed to start")
        if self._startup_error is not None:
            raise RuntimeError(
                "RBLN vLLM AsyncLLMEngine initialization failed"
            ) from self._startup_error

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._futures)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            with self.runtime._engine_environment():
                try:
                    from vllm import (
                        AsyncEngineArgs,
                        AsyncLLMEngine,
                        SamplingParams,
                    )
                except ImportError:
                    from vllm import SamplingParams
                    from vllm.engine.arg_utils import AsyncEngineArgs
                    from vllm.engine.async_llm_engine import AsyncLLMEngine

                engine_args = AsyncEngineArgs(
                    **self.runtime._engine_kwargs()
                )
                self._async_engine = AsyncLLMEngine.from_engine_args(
                    engine_args
                )
                self._sampling_params_cls = SamplingParams
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            loop.close()
            return
        with self._lock:
            close_after_startup = self._closing
        self._ready.set()
        if close_after_startup:
            try:
                loop.run_until_complete(self._shutdown_async_engine())
            except BaseException as exc:
                self._shutdown_error = exc
            finally:
                loop.close()
            return
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
        return self._submit_async(
            inputs,
            callback,
            max_new_tokens=self.max_new_tokens,
        )

    def _submit_async(self, inputs, callback, *, max_new_tokens: int):
        if not callable(callback):
            raise ValueError("callback must be callable")
        prompt_batches = self.runtime._trim_prompt_tokens(inputs)
        if len(prompt_batches) != 1:
            raise ValueError(
                "RBLN vLLM native async accepts one framework request at "
                "a time; vLLM owns continuous batching"
            )
        request_id = f"rbln-vllm-{uuid.uuid4().hex}"
        sampling_params = self._sampling_params_cls(
            max_tokens=_positive_int(max_new_tokens, "max_new_tokens"),
            temperature=0.0,
            stop_token_ids=self.stop_token_ids,
        )
        started_ns = time.monotonic_ns()
        with self._lock:
            if self._closing:
                raise RuntimeError("RBLN vLLM native backend is shutting down")
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

    def run_warmup_blocking(
        self,
        inputs,
        *,
        num_runs: int,
        timeout: float,
    ) -> None:
        deadline = time.monotonic() + timeout
        for _ in range(num_runs):
            completed = threading.Event()
            outcomes: list[NativeAsyncOutcome] = []

            def callback(outcome: NativeAsyncOutcome) -> None:
                outcomes.append(outcome)
                completed.set()

            self._submit_async(inputs, callback, max_new_tokens=1)
            if not completed.wait(
                timeout=max(0.0, deadline - time.monotonic())
            ):
                raise TimeoutError("RBLN vLLM async warmup timed out")
            outcome = outcomes[0]
            if outcome.error_type is not None:
                raise RuntimeError(
                    "RBLN vLLM async warmup failed: "
                    f"{outcome.error_type}: {outcome.error_message}"
                )

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
        events = []
        try:
            stream = self._async_engine.generate(
                {"prompt_token_ids": prompt_token_ids},
                sampling_params,
                request_id,
            )
            async for request_output in stream:
                final_output = request_output
                cumulative_tokens = len(
                    self._extract_token_ids(request_output)
                )
                if cumulative_tokens < previous_cumulative_tokens:
                    raise RuntimeError(
                        "RBLN vLLM cumulative token count decreased"
                    )
                if cumulative_tokens > previous_cumulative_tokens:
                    observed_ns = time.monotonic_ns()
                    events.append(
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
                shutdown_requested = (
                    request_id in self._shutdown_request_ids
                )
            if shutdown_requested:
                emit_once(
                    NativeAsyncOutcome(
                        error_type="RblnVllmAsyncShutdown",
                        error_message=(
                            "RBLN vLLM async generation stopped during shutdown"
                        ),
                    )
                )
                return

            finished_ns = final_output_ns or time.monotonic_ns()
            if final_output is None:
                generated_ids = np.zeros((1, 0), dtype=np.int64)
                generated_lengths = np.zeros((1,), dtype=np.int64)
            else:
                generated_ids, generated_lengths = (
                    self.runtime._normalize_outputs([final_output])
                )
            generated_tokens = int(generated_lengths.sum())
            timing_ms: dict[str, Any] = {
                "total_ms": (finished_ns - started_ns) / 1_000_000.0,
                "timing_mode": "kv_cache",
                "uses_kv_cache": True,
                "timing_source": "rbln_vllm_async_python_stream",
            }
            if first_token_ns is not None:
                timing_ms["ttft_ms"] = (
                    first_token_ns - started_ns
                ) / 1_000_000.0
                timing_ms["tpot_ms"] = (
                    0.0
                    if generated_tokens <= 1
                    else (finished_ns - first_token_ns)
                    / (generated_tokens - 1)
                    / 1_000_000.0
                )
            emit_once(
                NativeAsyncOutcome(
                    outputs={
                        "generated_ids": generated_ids,
                        "generated_lengths": generated_lengths,
                    },
                    timing_ms=timing_ms,
                    generated_tokens=generated_tokens,
                    generation_observation=GenerationObservation(
                        backend_submitted_ns=started_ns,
                        events=tuple(events),
                        source="rbln_vllm_async_python_stream",
                    ),
                )
            )
        except BaseException as exc:
            try:
                await self._abort_request(request_id)
            except BaseException:
                pass
            with self._lock:
                shutdown_requested = (
                    request_id in self._shutdown_request_ids
                )
            emit_once(
                NativeAsyncOutcome(
                    error_type=(
                        "RblnVllmAsyncShutdown"
                        if shutdown_requested
                        else type(exc).__name__[:128]
                    ),
                    error_message=(
                        "RBLN vLLM async generation stopped during shutdown"
                        if shutdown_requested
                        else "RBLN vLLM async generation failed"
                    ),
                )
            )
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

    async def _shutdown_async_engine(self) -> None:
        result = self._async_engine.shutdown()
        if inspect.isawaitable(result):
            await result

    def shutdown(self, timeout: float = 5.0) -> bool:
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(timeout_value) or timeout_value < 0.0:
            return False
        deadline = time.monotonic() + timeout_value
        with self._lock:
            self._closing = True
            if not self._started:
                return True
            if not self._thread.is_alive():
                return self._shutdown_error is None
            requests = tuple(self._futures.items())
            self._shutdown_request_ids.update(
                request_id for request_id, _ in requests
            )
        if not self._ready.wait(
            timeout=max(0.0, deadline - time.monotonic())
        ):
            return False
        if self._startup_error is not None:
            self._thread.join(
                timeout=max(0.0, deadline - time.monotonic())
            )
            return not self._thread.is_alive()
        if not self._thread.is_alive():
            return self._shutdown_error is None
        loop = self._loop
        if requests:
            abort_future = asyncio.run_coroutine_threadsafe(
                self._abort_requests(
                    tuple(request_id for request_id, _ in requests)
                ),
                loop,
            )
            done, pending = wait_futures(
                (abort_future,),
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if pending:
                abort_future.cancel()
                return False
            try:
                next(iter(done)).result()
            except BaseException:
                return False
            _, pending = wait_futures(
                tuple(future for _, future in requests),
                timeout=max(0.0, deadline - time.monotonic()),
            )
            if pending:
                return False

        shutdown_future = asyncio.run_coroutine_threadsafe(
            self._shutdown_async_engine(), loop
        )
        done, pending = wait_futures(
            (shutdown_future,),
            timeout=max(0.0, deadline - time.monotonic()),
        )
        if pending:
            shutdown_future.cancel()
            return False
        try:
            next(iter(done)).result()
        except BaseException:
            return False
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        self._thread.join(timeout=max(0.0, deadline - time.monotonic()))
        return not self._thread.is_alive()


class RblnVllmRuntime(Runtime):
    """Run prepared Optimum RBLN directories through vLLM RBLN."""

    def __init__(self, **runtime_options):
        self.device = str(runtime_options.get("device", "0"))
        self.block_size = _positive_int(
            runtime_options.get("block_size"), "block_size"
        )
        self.max_model_len = _optional_positive_int(
            runtime_options.get("max_model_len"), "max_model_len"
        )
        if (
            self.max_model_len is not None
            and self.max_model_len % self.block_size != 0
        ):
            raise ValueError("block_size must divide max_model_len")
        self.max_num_seqs = _positive_int(
            runtime_options.get("max_num_seqs", 1), "max_num_seqs"
        )
        self.tensor_parallel_size = _positive_int(
            runtime_options.get("tensor_parallel_size", 1),
            "tensor_parallel_size",
        )
        if self.tensor_parallel_size != 1:
            raise ValueError("vLLM RBLN tensor_parallel_size is fixed at 1")
        self.num_devices = _positive_int(
            runtime_options.get("num_devices", 1), "num_devices"
        )
        self.allow_unsupported_single_npu = _bool(
            runtime_options.get("allow_unsupported_single_npu", False),
            "allow_unsupported_single_npu",
        )
        self.shutdown_timeout_sec = float(
            runtime_options.get("shutdown_timeout_sec", 300.0)
        )
        if (
            not np.isfinite(self.shutdown_timeout_sec)
            or self.shutdown_timeout_sec <= 0.0
        ):
            raise ValueError("shutdown_timeout_sec must be positive and finite")
        self.startup_timeout_sec = float(
            runtime_options.get("startup_timeout_sec", 600.0)
        )
        if (
            not np.isfinite(self.startup_timeout_sec)
            or self.startup_timeout_sec <= 0.0
        ):
            raise ValueError("startup_timeout_sec must be positive and finite")
        self.decoder_batch_sizes = _decoder_batch_sizes(
            runtime_options.get("decoder_batch_sizes"), self.max_num_seqs
        )
        self.dtype = runtime_options.get("dtype", "auto")
        self.seed = runtime_options.get("seed")
        self.trust_remote_code = _bool(
            runtime_options.get("trust_remote_code", False),
            "trust_remote_code",
        )
        self.enable_warm_up = runtime_options.get("enable_warm_up")
        if self.enable_warm_up is not None:
            self.enable_warm_up = _bool(
                self.enable_warm_up, "enable_warm_up"
            )
        self.rbln_sampler = runtime_options.get("rbln_sampler")
        if self.rbln_sampler is not None:
            self.rbln_sampler = _bool(self.rbln_sampler, "rbln_sampler")
        self.cache_root = runtime_options.get("cache_root")
        self.tokenizer_path = runtime_options.get("tokenizer_path")
        self._inventory_provider: Callable[[], Mapping[str, Any]] = (
            runtime_options.get("inventory_provider")
            or self._read_rbln_inventory
        )
        if not callable(self._inventory_provider):
            raise ValueError("inventory_provider must be callable")

        supplied_additional = runtime_options.get("additional_config") or {}
        if not isinstance(supplied_additional, Mapping):
            raise ValueError("additional_config must be a mapping")
        self.additional_config = {
            key: value for key, value in supplied_additional.items()
        }
        if self.decoder_batch_sizes is not None:
            rbln_config = self.additional_config.get("rbln_config", {})
            if not isinstance(rbln_config, Mapping):
                raise ValueError("additional_config.rbln_config must be a mapping")
            self.additional_config["rbln_config"] = {
                **rbln_config,
                "decoder_batch_sizes": list(self.decoder_batch_sizes),
            }

        self.compiled_model: CompiledModel | None = None
        self._model_path: Path | None = None
        self._tokenizer_path: Path | None = None
        self._model_kind = "other"
        self._support_classification = "unverified"
        self._inventory: dict[str, Any] = {}
        self._sync_engine = None
        self._sampling_params_cls = None
        self._mode: str | None = None
        self._native_backend = None

    def load(self, compiled_model: CompiledModel) -> None:
        if self.compiled_model is not None:
            raise RuntimeError("RBLN vLLM runtime is already loaded")
        model_path = compiled_model.artifact_path
        if model_path is None or not model_path.is_dir():
            raise ValueError(
                "RBLN vLLM requires a prepared model directory artifact"
            )
        config_path = model_path / "config.json"
        if not config_path.is_file():
            raise ValueError("prepared RBLN model must contain config.json")
        if not any(path.is_file() for path in model_path.rglob("*.rbln")):
            raise ValueError(
                "prepared RBLN model must contain at least one .rbln file"
            )
        tokenizer_path = Path(self.tokenizer_path or model_path).expanduser()
        if (
            not tokenizer_path.is_dir()
            or not (tokenizer_path / "tokenizer_config.json").is_file()
            or not any(
                (tokenizer_path / name).is_file()
                for name in ("tokenizer.json", "tokenizer.model")
            )
        ):
            raise ValueError(
                "RBLN vLLM requires a local tokenizer directory with "
                "tokenizer_config.json and tokenizer.json or tokenizer.model"
            )
        config = _json_mapping(config_path, "model config")
        if str(config.get("model_type", "")).casefold() != "llama":
            raise ValueError("rbln-vllm currently supports Llama models only")

        manifest_path = model_path / _MANIFEST_NAME
        if manifest_path.is_file():
            manifest = _json_mapping(manifest_path, "RBLN vLLM manifest")
            resolved_max_model_len = self._validate_manifest_contract(
                manifest
            )
        else:
            manifest = None
            resolved_max_model_len = self.max_model_len

        model_kind = _resolve_model_kind(compiled_model, config, manifest)
        if model_kind == "llama-3.1-8b" and self.num_devices == 1:
            if manifest is None:
                raise ValueError(
                    "single-NPU Llama 3.1 8B requires "
                    f"{_MANIFEST_NAME} to verify its compiled device, "
                    "context, and batch contract"
                )
            if (
                dict.get(manifest, "support_classification")
                != "unsupported_single_npu_experiment"
            ):
                raise ValueError(
                    "single-NPU Llama 3.1 8B manifest "
                    "support_classification must be "
                    "unsupported_single_npu_experiment"
                )

        support_classification = self._validate_model_device_contract(
            model_kind,
            resolved_max_model_len,
        )
        inventory = self._load_inventory()
        devices = inventory["devices"]
        if len(devices) < self.num_devices:
            raise RuntimeError(
                f"RBLN vLLM requested {self.num_devices} devices but found "
                f"{len(devices)}"
            )
        selected = devices[: self.num_devices]
        unhealthy = [
            item.get("npu")
            for item in selected
            if str(item.get("status", "")).casefold() != "normal"
        ]
        if unhealthy:
            raise RuntimeError(f"RBLN devices are not normal: {unhealthy}")
        if (
            model_kind in {"llama-3.2-3b", "llama-3.1-8b"}
            and self.num_devices == 1
        ):
            minimum_gib = 15 if model_kind == "llama-3.1-8b" else 8
            total_bytes = _memory_bytes(
                selected[0].get("memory", {}).get("total")
                if isinstance(selected[0].get("memory"), Mapping)
                else None
            )
            if total_bytes is None:
                raise ValueError(
                    f"single-NPU {model_kind} requires readable device "
                    "memory.total"
                )
            if total_bytes < minimum_gib * _GIB:
                raise ValueError(
                    f"single-NPU {model_kind} requires at least "
                    f"{minimum_gib} GiB for weights and runtime reserve"
                )

        self.compiled_model = compiled_model
        self._model_path = model_path.resolve()
        self._tokenizer_path = tokenizer_path.resolve()
        self._model_kind = model_kind
        self._support_classification = support_classification
        self._inventory = dict(inventory)
        self.max_model_len = resolved_max_model_len

    def _validate_manifest_contract(
        self, manifest: Mapping[str, Any]
    ) -> int:
        try:
            manifest_devices = _positive_int(
                manifest.get("num_devices"), "manifest num_devices"
            )
            manifest_block_size = _positive_int(
                manifest.get("block_size"), "manifest block_size"
            )
            manifest_max_seq_len = _positive_int(
                manifest.get("max_seq_len"), "manifest max_seq_len"
            )
            manifest_batch_size = _positive_int(
                manifest.get("batch_size"), "manifest batch_size"
            )
            manifest_decoder_batch_sizes = _decoder_batch_sizes(
                manifest.get("decoder_batch_sizes"),
                manifest_batch_size,
            )
        except ValueError as exc:
            raise ValueError(f"invalid RBLN vLLM manifest: {exc}") from exc
        if manifest_decoder_batch_sizes is None:
            raise ValueError(
                "invalid RBLN vLLM manifest: decoder_batch_sizes are required"
            )
        if manifest_batch_size not in manifest_decoder_batch_sizes:
            raise ValueError(
                "invalid RBLN vLLM manifest: decoder_batch_sizes must "
                "include batch_size"
            )
        if manifest_devices != self.num_devices:
            raise ValueError(
                f"prepared model was compiled for {manifest_devices} devices "
                f"but runtime requested {self.num_devices}"
            )
        if manifest_block_size != self.block_size:
            raise ValueError(
                f"prepared model block_size={manifest_block_size} but runtime "
                f"block_size={self.block_size}"
            )
        if manifest_max_seq_len % manifest_block_size != 0:
            raise ValueError(
                "invalid RBLN vLLM manifest: block_size must divide "
                "max_seq_len"
            )
        if (
            self.max_model_len is not None
            and manifest_max_seq_len != self.max_model_len
        ):
            raise ValueError(
                f"prepared model max_seq_len={manifest_max_seq_len} but "
                f"runtime max_model_len={self.max_model_len}"
            )
        if manifest_batch_size != self.max_num_seqs:
            raise ValueError(
                f"prepared model batch_size={manifest_batch_size} but runtime "
                f"max_num_seqs={self.max_num_seqs}"
            )
        if self.decoder_batch_sizes is not None:
            unavailable = sorted(
                set(self.decoder_batch_sizes)
                - set(manifest_decoder_batch_sizes)
            )
            if unavailable:
                raise ValueError(
                    "runtime decoder_batch_sizes were not compiled into the "
                    f"prepared model: {unavailable}"
                )
        return manifest_max_seq_len

    def _validate_model_device_contract(
        self,
        model_kind: str,
        resolved_max_model_len: int | None,
    ) -> str:
        if model_kind in {"llama-3.1-8b", "llama-3.2-3b"}:
            if self.num_devices == 8:
                return "official"
            if self.num_devices == 1 and self.allow_unsupported_single_npu:
                if self.max_num_seqs != 1:
                    raise ValueError(
                        f"single-NPU {model_kind} requires max_num_seqs "
                        "exactly 1"
                    )
                max_single_npu_context = (
                    512 if model_kind == "llama-3.1-8b" else 1024
                )
                if (
                    resolved_max_model_len is None
                    or resolved_max_model_len > max_single_npu_context
                ):
                    raise ValueError(
                        f"single-NPU {model_kind} max_model_len must be "
                        "explicitly resolved and at most "
                        f"{max_single_npu_context}"
                    )
                return "unsupported_single_npu_experiment"
            raise ValueError(
                f"{model_kind} is officially supported with 8 NPUs; set "
                "allow_unsupported_single_npu=true for the bounded "
                "one-NPU experiment"
            )
        if self.num_devices == 1:
            raise ValueError(
                "single-NPU RBLN vLLM requires artifact identity "
                "Llama 3.2 3B or Llama 3.1 8B"
            )
        return "unverified"

    def _load_inventory(self) -> dict[str, Any]:
        try:
            payload = self._inventory_provider()
        except BaseException as exc:
            raise RuntimeError("failed to read RBLN device inventory") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("RBLN device inventory must be a mapping")
        devices = payload.get("devices")
        if not isinstance(devices, list) or any(
            not isinstance(item, Mapping) for item in devices
        ):
            raise RuntimeError("RBLN device inventory has invalid devices")
        return {**payload, "devices": [dict(item) for item in devices]}

    @staticmethod
    def _read_rbln_inventory() -> Mapping[str, Any]:
        executable = shutil.which("rbln-smi")
        if not executable:
            raise RuntimeError("rbln-smi executable was not found")
        completed = subprocess.run(
            [executable, "-j"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict):
            raise RuntimeError("rbln-smi -j returned a non-object payload")
        return payload

    @contextmanager
    def _engine_environment(self):
        updates = {_DEVICE_COUNT_ENV: str(self.num_devices)}
        if self.enable_warm_up is not None:
            updates["VLLM_RBLN_ENABLE_WARM_UP"] = str(
                self.enable_warm_up
            ).lower()
        if self.rbln_sampler is not None:
            updates["VLLM_RBLN_SAMPLER"] = str(self.rbln_sampler).lower()
        if self.cache_root is not None:
            updates["VLLM_CACHE_ROOT"] = str(self.cache_root)
        with _ENGINE_ENV_LOCK:
            previous = {name: os.environ.get(name) for name in updates}
            os.environ.update(updates)
            try:
                yield
            finally:
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value

    def _engine_kwargs(self) -> dict[str, Any]:
        if self._model_path is None or self._tokenizer_path is None:
            raise RuntimeError("RBLN vLLM runtime is not loaded")
        kwargs: dict[str, Any] = {
            "model": str(self._model_path),
            "tokenizer": str(self._tokenizer_path),
            "block_size": self.block_size,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.additional_config:
            kwargs["additional_config"] = self.additional_config
        return kwargs

    def _ensure_sync_engine(self):
        if self.compiled_model is None:
            raise RuntimeError("RBLN vLLM runtime is not loaded. Call load() first.")
        if self._mode == "async":
            raise RuntimeError(
                "RBLN vLLM runtime already owns an async engine; sync and "
                "async engine modes cannot be mixed"
            )
        if self._sync_engine is not None:
            return self._sync_engine
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "rbln-vllm requires vllm-rbln and its compatible vLLM wheel"
            ) from exc
        with self._engine_environment():
            engine = LLM(**self._engine_kwargs())
        self._sync_engine = engine
        self._sampling_params_cls = SamplingParams
        self._mode = "sync"
        return engine

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
        inputs: dict[str, np.ndarray],
        max_new_tokens: int = 256,
        stop_token_ids: list[int] | None = None,
    ) -> GenerationResult:
        prompt_batches = self._trim_prompt_tokens(inputs)
        if len(prompt_batches) > self.max_num_seqs:
            raise ValueError(
                f"prompt batch {len(prompt_batches)} exceeds max_num_seqs "
                f"{self.max_num_seqs}"
            )
        max_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        normalized_stop_ids = (
            None
            if stop_token_ids is None
            else [int(token_id) for token_id in stop_token_ids]
        )
        engine = self._ensure_sync_engine()
        sampling_params = self._sampling_params_cls(
            max_tokens=max_tokens,
            temperature=0.0,
            stop_token_ids=normalized_stop_ids,
        )
        prompts = [
            {"prompt_token_ids": prompt} for prompt in prompt_batches
        ]
        started = time.perf_counter()
        raw_outputs = engine.generate(prompts, sampling_params=sampling_params)
        total_ms = (time.perf_counter() - started) * 1000.0
        generated_ids, generated_lengths = self._normalize_outputs(raw_outputs)
        generated_tokens = int(generated_lengths.sum())
        ttft_ms, tpot_ms, timing_source = (
            VllmRuntime._extract_timing_from_vllm_metrics(
                raw_outputs,
                total_ms=total_ms,
                num_tokens=generated_tokens,
            )
        )
        return GenerationResult(
            generated_ids=generated_ids,
            generated_lengths=generated_lengths,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            num_tokens=generated_tokens,
            timing_mode="kv_cache",
            uses_kv_cache=True,
            timing_source=timing_source,
        )

    @staticmethod
    def _trim_prompt_tokens(
        inputs: dict[str, np.ndarray],
    ) -> list[list[int]]:
        if "input_ids" not in inputs:
            raise ValueError("RBLN vLLM inputs must contain input_ids")
        input_ids = np.asarray(inputs["input_ids"])
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must be 1D or 2D, got shape={input_ids.shape}"
            )
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            prompts = [list(map(int, row)) for row in input_ids]
        else:
            mask = np.asarray(attention_mask)
            if mask.ndim == 1:
                mask = mask.reshape(1, -1)
            if mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask shape must match input_ids shape: "
                    f"{mask.shape} != {input_ids.shape}"
                )
            prompts = [
                [int(token) for token in row[selected.astype(bool)]]
                for row, selected in zip(input_ids, mask)
            ]
        if any(not prompt for prompt in prompts):
            raise ValueError("RBLN vLLM prompts must contain at least one token")
        return prompts

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
            token_batches.append(
                [int(token) for token in completions[0].token_ids]
            )
        lengths = np.asarray(
            [len(tokens) for tokens in token_batches], dtype=np.int64
        )
        max_length = int(lengths.max()) if len(lengths) else 0
        padded = np.zeros((len(token_batches), max_length), dtype=np.int64)
        for index, tokens in enumerate(token_batches):
            if tokens:
                padded[index, : len(tokens)] = tokens
        return padded, lengths

    def warmup(
        self, inputs: dict[str, np.ndarray], num_runs: int = 1
    ) -> None:
        if type(num_runs) is not int or num_runs < 0:
            raise ValueError("num_runs must be a nonnegative int")
        if self._mode == "async":
            if self._native_backend is None:
                raise RuntimeError("RBLN vLLM async backend is unavailable")
            self._native_backend.run_warmup_blocking(
                inputs,
                num_runs=num_runs,
                timeout=self.shutdown_timeout_sec,
            )
            return
        for _ in range(num_runs):
            self.generate(inputs, max_new_tokens=1)

    def create_native_backend(
        self,
        *,
        max_new_tokens: int,
        stop_token_ids=None,
    ) -> RblnVllmNativeBackend:
        if self.compiled_model is None:
            raise RuntimeError("RBLN vLLM runtime is not loaded. Call load() first.")
        if self._mode == "sync":
            raise RuntimeError(
                "RBLN vLLM runtime already owns a sync engine; sync and "
                "async engine modes cannot be mixed"
            )
        if self._native_backend is not None:
            return self._native_backend
        backend = RblnVllmNativeBackend(
            self,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
        self._native_backend = backend
        self._mode = "async"
        try:
            backend.start(timeout=self.startup_timeout_sec)
        except BaseException:
            if backend.shutdown(timeout=self.shutdown_timeout_sec):
                self._native_backend = None
                self._mode = None
            raise
        return backend

    @staticmethod
    def _shutdown_engine(engine) -> None:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        inner_engine = getattr(engine, "llm_engine", None)
        inner_shutdown = getattr(inner_engine, "shutdown", None)
        if callable(inner_shutdown):
            inner_shutdown()

    def unload(self) -> None:
        native_backend = self._native_backend
        if native_backend is not None:
            if not native_backend.shutdown(timeout=self.shutdown_timeout_sec):
                raise RuntimeError(
                    "RBLN vLLM native async backend did not stop; engine "
                    "references were preserved"
                )
            self._native_backend = None
        if self._sync_engine is not None:
            self._shutdown_engine(self._sync_engine)
        self._sync_engine = None
        self._sampling_params_cls = None
        self.compiled_model = None
        self._model_path = None
        self._tokenizer_path = None
        self._mode = None
        self._inventory = {}

    def get_device_spec(self) -> dict[str, Any]:
        devices = self._inventory.get("devices", [])
        return {
            "backend": "rbln_vllm",
            "device": self.device,
            "num_devices": self.num_devices,
            "tensor_parallel_size": self.tensor_parallel_size,
            "model_path": str(self._model_path or ""),
            "model_kind": self._model_kind,
            "support_classification": self._support_classification,
            "available_devices": len(devices),
            "accelerator_vendor": "Rebellions",
            "accelerator_name": (
                devices[0].get("name", "RBLN NPU")
                if devices
                else "RBLN NPU"
            ),
        }

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        path = compiled_model.artifact_path
        return bool(
            path is not None
            and path.is_dir()
            and (path / "config.json").is_file()
            and any(item.is_file() for item in path.rglob("*.rbln"))
            and compiled_model.backend_name.casefold()
            in {"rbln_vllm", "rbln-vllm"}
        )

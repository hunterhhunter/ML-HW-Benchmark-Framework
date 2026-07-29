"""Mobilint Model Zoo generation runtime for ARIES device 0."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.compiled_model import CompiledModel
from core.generation_result import GenerationResult
from core.model_spec import Task
from core.runtime_executor import GenerationObservation, GenerationOutputEvent
from mobilint_device import MobilintDeviceSession
from .base import Runtime


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class _TimingStreamer:
    """Record actual producer callbacks without inventing token timestamps."""

    def __init__(self, clock_ns):
        self._clock_ns = clock_ns
        self._first_put = True
        self._tokens: list[int] = []
        self.events: list[GenerationOutputEvent] = []

    def put(self, value) -> None:
        tokens = _to_numpy(value).reshape(-1)
        if self._first_put:
            self._first_put = False
            return
        self._tokens.extend(int(token) for token in tokens)
        self.events.append(
            GenerationOutputEvent(
                observed_ns=self._clock_ns(),
                cumulative_tokens=len(self._tokens),
            )
        )

    def end(self) -> None:
        return None


class MobilintLlmRuntime(Runtime):
    """Run a locally prepared Model Zoo LLM on Mobilint ARIES device 0."""

    def __init__(
        self,
        device="0",
        device_id=0,
        expected_family="aries",
        **options,
    ):
        if type(device_id) is not int or device_id != 0 or expected_family != "aries":
            raise ValueError(
                "Mobilint LLM runtime currently supports ARIES device 0 only."
            )
        self.device = str(device)
        self.device_id = 0
        self.expected_family = "aries"
        self.compiled_model: CompiledModel | None = None
        self._model = None
        self._device_session: MobilintDeviceSession | None = None
        self._device_info = None
        self._cleanup_pending = False
        self._max_batch_size = 1
        self._clock_ns = options.get("clock_ns", time.monotonic_ns)
        if not callable(self._clock_ns):
            raise ValueError("clock_ns must be callable.")

    def supports_generate(self) -> bool:
        return True

    def supports_batch_generation(self) -> bool:
        return self._max_batch_size > 1

    def supports_dynamic_batching(self) -> bool:
        return self._max_batch_size > 1

    def max_dynamic_batch_size(self) -> int:
        return self._max_batch_size

    def native_async_max_batch_size(self) -> None:
        return None

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        return (
            compiled_model.spec.task is Task.NLP_GENERATION
            and compiled_model.artifact_path.is_dir()
        )

    @staticmethod
    def _read_artifact_capacity(model_dir: Path) -> int:
        config_path = model_dir / "config.json"
        try:
            config_text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                "Mobilint LLM model directory requires a readable config.json."
            ) from exc
        try:
            config = json.loads(config_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Mobilint LLM config.json must contain valid JSON."
            ) from exc
        if not isinstance(config, dict) or "max_batch_size" not in config:
            raise ValueError(
                "Mobilint LLM config.json requires max_batch_size."
            )
        capacity = config["max_batch_size"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(
                "Mobilint LLM max_batch_size must be a positive integer."
            )
        return capacity

    def _cleanup_resources(self) -> None:
        if self._model is not None:
            self._model.dispose()
            self._model = None
            self.compiled_model = None

        if self._device_session is not None:
            self._device_session.release()
            self._device_session = None
            self._device_info = None
        self._cleanup_pending = False

    def load(self, compiled_model: CompiledModel) -> None:
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint LLM cleanup is incomplete; call unload() to retry."
            )
        if self._model is not None or self._device_session is not None:
            raise RuntimeError("Mobilint LLM runtime is already loaded.")
        if not self.is_compatible(compiled_model):
            raise ValueError(
                "Mobilint LLM requires a local NLP generation model directory."
            )
        artifact_capacity = self._read_artifact_capacity(
            compiled_model.artifact_path
        )

        session = MobilintDeviceSession(0, "aries")
        self._device_session = session
        self._cleanup_pending = True
        try:
            self._device_info = session.acquire()
        except BaseException as acquire_error:
            self._device_info = None
            if session.module is not None:
                raise RuntimeError(
                    "Mobilint Model Zoo model load failed and rollback cleanup "
                    f"is incomplete ({type(acquire_error).__name__}: "
                    f"{acquire_error}); call unload() to retry cleanup."
                ) from acquire_error
            self._device_session = None
            self._cleanup_pending = False
            if isinstance(acquire_error, (ImportError, ModuleNotFoundError)):
                raise
            raise RuntimeError(
                "Mobilint Model Zoo model load failed."
            ) from acquire_error

        try:
            from mblt_model_zoo.hf_transformers.models.llama import (
                modeling_llama as _modeling_llama_registration,  # noqa: F401
            )
            from transformers import AutoModelForCausalLM
        except BaseException as import_error:
            try:
                self._cleanup_resources()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Mobilint Model Zoo model load failed and rollback cleanup "
                    f"is incomplete ({type(cleanup_error).__name__}: "
                    f"{cleanup_error}); call unload() to retry cleanup."
                ) from import_error
            if isinstance(import_error, (ImportError, ModuleNotFoundError)):
                raise ImportError(
                    "Mobilint Model Zoo LLM loading requires the optional "
                    "'mblt-model-zoo[transformers]' package with Transformers "
                    "support."
                ) from import_error
            raise RuntimeError(
                "Mobilint Model Zoo model load failed."
            ) from import_error

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(compiled_model.artifact_path),
                dev_no=0,
            )
        except BaseException as load_error:
            try:
                self._cleanup_resources()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Mobilint Model Zoo model load failed and rollback cleanup "
                    f"is incomplete ({type(cleanup_error).__name__}: "
                    f"{cleanup_error}); call unload() to retry cleanup."
                ) from load_error
            raise RuntimeError("Mobilint Model Zoo model load failed.") from load_error

        self.compiled_model = compiled_model
        self._max_batch_size = artifact_capacity
        self._cleanup_pending = False

    def generate(self, inputs, max_new_tokens=256, stop_token_ids=None):
        if self._cleanup_pending:
            raise RuntimeError(
                "Mobilint LLM cleanup is incomplete; call unload() to retry."
            )
        if self._model is None:
            raise RuntimeError("Mobilint LLM model is not loaded. Call load() first.")

        if "input_ids" not in inputs:
            raise ValueError("Mobilint LLM generation requires input_ids.")
        input_ids = np.asarray(inputs["input_ids"])
        if input_ids.ndim == 1:
            input_ids = input_ids.reshape(1, -1)
        if input_ids.ndim != 2:
            raise ValueError("Mobilint LLM input_ids must be a rank-2 token batch.")
        batch_size = int(input_ids.shape[0])
        if not 1 <= batch_size <= self._max_batch_size:
            suffix = (
                "; the standard artifact supports batch size 1 only"
                if self._max_batch_size == 1
                else ""
            )
            raise ValueError(
                f"Mobilint LLM actual batch size {batch_size} exceeds artifact "
                f"capacity {self._max_batch_size}{suffix}."
            )

        if "attention_mask" in inputs:
            attention_mask = np.asarray(inputs["attention_mask"])
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.reshape(1, -1)
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "attention_mask must have the same shape as input_ids."
                )
            real_token_mask = attention_mask.astype(bool, copy=False)
        else:
            attention_mask = np.ones(input_ids.shape, dtype=np.int64)
            real_token_mask = attention_mask.astype(bool, copy=False)

        if np.any(real_token_mask.sum(axis=1) == 0):
            raise ValueError(
                "Every Mobilint LLM prompt must contain at least one token."
            )
        if batch_size == 1:
            prompt = input_ids[0, real_token_mask[0]].astype(
                np.int64,
                copy=False,
            ).reshape(1, -1)
            prompt_mask = np.ones(prompt.shape, dtype=np.int64)
        else:
            prompt_lengths = real_token_mask.sum(axis=1).astype(
                np.int64,
                copy=False,
            )
            grouped_width = int(prompt_lengths.max())
            prompt = np.empty(
                (batch_size, grouped_width),
                dtype=np.int64,
            )
            prompt_mask = np.zeros(prompt.shape, dtype=np.int64)
            for row_index in range(batch_size):
                row_mask = real_token_mask[row_index]
                tokens = input_ids[row_index, row_mask].astype(
                    np.int64,
                    copy=False,
                )
                padding = input_ids[row_index, ~row_mask]
                pad_token_id = int(padding[0]) if padding.size else 0
                prompt[row_index].fill(pad_token_id)
                prompt[row_index, -tokens.size :] = tokens
                prompt_mask[row_index, -tokens.size :] = 1
        prompt_width = int(prompt.shape[1])

        try:
            import torch
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError(
                "Mobilint Model Zoo generation requires the optional torch package."
            ) from exc

        input_ids_tensor = torch.as_tensor(
            prompt,
            dtype=torch.long,
            device="cpu",
        )
        attention_mask_tensor = torch.as_tensor(
            prompt_mask,
            dtype=torch.long,
            device="cpu",
        )
        normalized_stop_ids = (
            None
            if stop_token_ids is None
            else [int(stop_token_ids)]
            if isinstance(stop_token_ids, (int, np.integer))
            else [int(token_id) for token_id in stop_token_ids]
        )
        streamer = _TimingStreamer(self._clock_ns)

        submitted_ns = self._clock_ns()
        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=normalized_stop_ids,
                streamer=streamer,
            )
        completed_ns = self._clock_ns()

        if hasattr(output, "sequences"):
            output = output.sequences
        sequences = _to_numpy(output)
        if sequences.ndim == 1 and batch_size == 1:
            sequences = sequences.reshape(1, -1)
        if sequences.ndim != 2 or sequences.shape[0] != batch_size:
            raise RuntimeError(
                "Mobilint Model Zoo generate() returned an unexpected batch shape."
            )
        if sequences.shape[1] < prompt_width:
            raise RuntimeError(
                "Mobilint Model Zoo output is shorter than the input prompt."
            )
        continuations = np.asarray(sequences[:, prompt_width:])
        continuation_width = int(continuations.shape[1])
        generated_lengths = np.full(
            batch_size,
            continuation_width,
            dtype=np.int64,
        )
        if normalized_stop_ids:
            stop_ids = np.asarray(normalized_stop_ids, dtype=np.int64)
            for row_index, row in enumerate(continuations):
                matches = np.flatnonzero(np.isin(row, stop_ids))
                if matches.size:
                    generated_lengths[row_index] = int(matches[0]) + 1
        generated_token_count = int(generated_lengths.sum())
        streamed_length = (
            streamer.events[-1].cumulative_tokens if streamer.events else 0
        )
        streamed_output_count = int(continuations.size)
        if streamed_length != streamed_output_count:
            raise RuntimeError(
                "Mobilint Model Zoo streamer/output mismatch: "
                f"streamed {streamed_length} tokens but returned "
                f"{streamed_output_count}."
            )

        events = tuple(streamer.events)
        ttft_ms = (
            None
            if not events
            else (events[0].observed_ns - submitted_ns) / 1_000_000.0
        )
        tpot_ms = (
            None
            if (
                (batch_size == 1 and generated_token_count <= 1)
                or (batch_size > 1 and len(events) <= 1)
            )
            else (events[-1].observed_ns - events[0].observed_ns)
            / (
                generated_token_count - 1
                if batch_size == 1
                else len(events) - 1
            )
            / 1_000_000.0
        )
        source = "mobilint_transformers_streamer"
        generated_ids = (
            continuations[0]
            if batch_size == 1
            else continuations
        )
        return GenerationResult(
            generated_ids=generated_ids.astype(np.int64, copy=False),
            generated_lengths=generated_lengths,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=(completed_ns - submitted_ns) / 1_000_000.0,
            num_tokens=generated_token_count,
            timing_mode="kv_cache",
            uses_kv_cache=True,
            timing_source=source,
            generation_observation=GenerationObservation(
                backend_submitted_ns=submitted_ns,
                events=events,
                source=source,
            ),
        )

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = self.generate(inputs, max_new_tokens=1)
        return {"generated_ids": result.generated_ids}

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        for _ in range(max(0, int(num_runs))):
            self.generate(inputs, max_new_tokens=1)

    def unload(self) -> None:
        if self._model is None and self._device_session is None:
            self.compiled_model = None
            self._device_info = None
            self._cleanup_pending = False
            return
        self._cleanup_pending = True
        self._cleanup_resources()

    def get_device_spec(self) -> Dict[str, Any]:
        return {
            "backend": "mobilint_llm",
            "device": self.device,
            "device_id": self.device_id,
            "expected_family": self.expected_family,
            "detected_family": getattr(self._device_info, "family", None),
            "device_type": getattr(self._device_info, "device_type", None),
            "accelerator_vendor": "Mobilint",
            "accelerator_name": "ARIES",
            "max_batch_size": self._max_batch_size,
        }

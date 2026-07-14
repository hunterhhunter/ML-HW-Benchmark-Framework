import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .model_spec import Task


@dataclass(frozen=True)
class RuntimeInvocation:
    outputs: Dict[str, Any]
    timing_ms: float | Dict[str, Any]
    generated_tokens: int = 0


class InferencePipeline:
    def __init__(self, dataloader, runtime, max_new_tokens: int = 256):
        self.dataloader = dataloader
        self.runtime = runtime
        self.max_new_tokens = max_new_tokens
        metadata = dataloader.get_metadata()
        self.is_static_batched = bool(metadata.get("is_static_batched", False))
        self.stop_token_ids = metadata.get("stop_token_ids")

        self.input_name = "input"
        compiled_model = getattr(runtime, "compiled_model", None)
        if compiled_model is not None:
            self.input_name = next(iter(compiled_model.spec.input_shapes))

        spec = getattr(compiled_model, "spec", None)
        self.is_llm = bool(
            spec is not None
            and spec.task == Task.NLP_GENERATION
            and runtime.supports_generate()
        )

    def collate_batch(self, batch_list: Any) -> Dict[str, Any]:
        if self.is_static_batched:
            return batch_list
        collated: Dict[str, Any] = {}
        for key in batch_list[0]:
            if key != "input":
                collated[key] = [item[key] for item in batch_list]
                continue
            first_input = batch_list[0][key]
            if isinstance(first_input, dict):
                collated[key] = {
                    name: np.stack(
                        [item[key][name] for item in batch_list],
                        axis=0,
                    )
                    for name in first_input
                }
            else:
                collated[key] = np.stack(
                    [item[key] for item in batch_list],
                    axis=0,
                )
        return collated

    def prepare_runtime_input(self, collated_input: Any) -> Dict[str, Any]:
        if isinstance(collated_input, dict):
            return collated_input
        return {self.input_name: collated_input}

    def prepare_eval_labels(self, collated: Dict[str, Any]) -> Any:
        labels = collated["label"]
        contexts = collated.get("preprocess_context")
        if not isinstance(labels, list) or not isinstance(contexts, list):
            return labels
        if len(labels) != len(contexts):
            return labels
        return [
            {"label": label, "preprocess_context": context}
            for label, context in zip(labels, contexts)
        ]

    def invoke(self, runtime_input: Dict[str, Any]) -> RuntimeInvocation:
        if self.is_llm:
            result = self.runtime.generate(
                runtime_input,
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
            return RuntimeInvocation(
                outputs=outputs,
                timing_ms=timing,
                generated_tokens=result.num_tokens,
            )

        started = time.perf_counter()
        outputs = self.runtime.run(runtime_input)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return RuntimeInvocation(outputs=outputs, timing_ms=elapsed_ms)

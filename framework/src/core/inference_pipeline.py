from typing import Any, Dict

import numpy as np

from .model_spec import Task
from .runtime_executor import BlockingRuntimeExecutor, RuntimeExecution


RuntimeInvocation = RuntimeExecution


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
        self._compat_executor = BlockingRuntimeExecutor(
            runtime,
            is_llm=self.is_llm,
            max_new_tokens=max_new_tokens,
            stop_token_ids=self.stop_token_ids,
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

    def batch_size(self, collated: Dict[str, Any]) -> int:
        inputs = collated["input"]
        if isinstance(inputs, dict):
            inputs = next(iter(inputs.values()))
        return len(inputs)

    def reset_dataloader_cursor(self) -> None:
        if hasattr(self.dataloader, "current_idx"):
            self.dataloader.current_idx = 0
        elif hasattr(self.dataloader, "_current_idx"):
            self.dataloader._current_idx = 0

    def invoke(self, runtime_input):
        return self._compat_executor.execute(runtime_input)

"""Pure task contracts and model helpers for Mobilint BERT compilation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class TensorContract:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model_id: str
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    max_length: int
    source_outputs: tuple[str, ...]
    verified_runtime_outputs: tuple[str, ...]
    calibration_samples: int = 32
    target_device: str = "aries-rb"
    compiler_inputs: tuple[TensorContract, ...] = (
        TensorContract("input_ids", (1, -1), "int64"),
        TensorContract("attention_mask", (1, -1), "int64"),
        TensorContract("token_type_ids", (1, -1), "int64"),
    )
    mxq_inputs: tuple[TensorContract, ...] = (
        TensorContract("embeddings", (1, -1, 768), "float32"),
    )


TASK_SPECS = MappingProxyType({
    "sst2": TaskSpec(
        name="sst2",
        model_id="textattack/bert-base-uncased-SST-2",
        dataset_name="glue",
        dataset_config="sst2",
        dataset_split="validation",
        max_length=128,
        source_outputs=("logits",),
        verified_runtime_outputs=("logits",),
    ),
    "squad1": TaskSpec(
        name="squad1",
        model_id="csarron/bert-base-uncased-squad-v1",
        dataset_name="squad",
        dataset_config=None,
        dataset_split="validation",
        max_length=384,
        source_outputs=("start_logits", "end_logits"),
        verified_runtime_outputs=("end_logits", "start_logits"),
    ),
})


def get_task_spec(task: str) -> TaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as error:
        raise ValueError(
            f"unsupported Mobilint BERT compile task: {task}"
        ) from error


def _tensor_to_dict(tensor: TensorContract) -> dict[str, object]:
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
    }


def contract_to_dict(spec: TaskSpec) -> dict[str, object]:
    return {
        "task": spec.name,
        "model_id": spec.model_id,
        "dataset": {
            "name": spec.dataset_name,
            "config": spec.dataset_config,
            "split": spec.dataset_split,
        },
        "max_length": spec.max_length,
        "calibration_samples": spec.calibration_samples,
        "target_device": spec.target_device,
        "compiler_inputs": [
            _tensor_to_dict(tensor) for tensor in spec.compiler_inputs
        ],
        "mxq_inputs": [_tensor_to_dict(tensor) for tensor in spec.mxq_inputs],
        "source_outputs": list(spec.source_outputs),
        "verified_runtime_outputs": list(spec.verified_runtime_outputs),
    }


def extract_embedding_weights(model: object) -> dict[str, object]:
    """Copy the fine-tuned model's host-side BERT embedding boundary."""
    embeddings = model.bert.embeddings
    return {
        "word_embeddings": (
            embeddings.word_embeddings.weight.detach().cpu().clone()
        ),
        "token_type_embeddings": (
            embeddings.token_type_embeddings.weight.detach().cpu().clone()
        ),
        "position_embeddings": (
            embeddings.position_embeddings.weight.detach().cpu().clone()
        ),
        "layernorm_weight": embeddings.LayerNorm.weight.detach().cpu().clone(),
        "layernorm_bias": embeddings.LayerNorm.bias.detach().cpu().clone(),
    }


def make_compiler_model(task: str, model: object) -> object:
    if task == "sst2":
        return model
    if task != "squad1":
        raise ValueError(f"unsupported Mobilint BERT compile task: {task}")

    from tools.mobilint_bert_compile.compiler_models import (
        BertQuestionAnsweringForCompiler,
    )

    return BertQuestionAnsweringForCompiler(model)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Server-verified BERT model adapters for the Furiosa Torch runtime."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from core.model_spec import Model_Spec, Task


@dataclass(frozen=True)
class HuggingFaceSourceContract:
    directory_name: str
    architecture: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    minimum_position_embeddings: int
    num_labels: int | None = None


@dataclass(frozen=True)
class TorchModelAdapter:
    model_name: str
    task: Task
    input_names: tuple[str, ...]
    input_shapes: Mapping[str, tuple[int, ...]]
    input_dtypes: Mapping[str, str]
    output_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int, ...]]
    tactic_hint: str
    loader: Callable[[Path], object]
    source_contract: HuggingFaceSourceContract

    def validate_source(self, path: Path) -> None:
        _validate_huggingface_source(Path(path), self.source_contract)

    def validate_spec(self, spec: Model_Spec) -> None:
        errors = []
        if spec.name != self.model_name:
            errors.append(f"model name={spec.name!r}")
        if spec.task is not self.task:
            errors.append(f"task={spec.task!r}")
        if tuple(spec.input_shapes) != self.input_names:
            errors.append(f"input names={tuple(spec.input_shapes)!r}")
        if {
            name: tuple(shape) for name, shape in spec.input_shapes.items()
        } != dict(self.input_shapes):
            errors.append(f"input shapes={spec.input_shapes!r}")
        if dict(spec.input_dtype) != dict(self.input_dtypes):
            errors.append(f"input dtypes={spec.input_dtype!r}")
        if tuple(spec.output_shapes) != self.output_names:
            errors.append(f"output names={tuple(spec.output_shapes)!r}")
        if {
            name: tuple(shape) for name, shape in spec.output_shapes.items()
        } != dict(self.output_shapes):
            errors.append(f"output shapes={spec.output_shapes!r}")
        if errors:
            raise ValueError(
                f"Unverified Furiosa Torch contract for {self.model_name}: "
                + "; ".join(errors)
            )


def _validate_huggingface_source(
    path: Path,
    contract: HuggingFaceSourceContract,
) -> None:
    if not path.is_dir():
        raise ValueError(f"Hugging Face model directory does not exist: {path}")
    if path.name != contract.directory_name:
        raise ValueError(
            "Unverified Hugging Face model directory name: "
            f"expected {contract.directory_name!r}, got {path.name!r}"
        )

    config_path = path / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except FileNotFoundError:
        raise ValueError(f"Hugging Face config.json is missing: {config_path}") from None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Hugging Face config.json: {config_path}") from exc

    expected_fields = {
        "model_type": contract.model_type,
        "hidden_size": contract.hidden_size,
        "intermediate_size": contract.intermediate_size,
        "num_attention_heads": contract.num_attention_heads,
        "num_hidden_layers": contract.num_hidden_layers,
        "vocab_size": contract.vocab_size,
    }
    mismatches = [
        f"{name}={config.get(name)!r} (expected {expected!r})"
        for name, expected in expected_fields.items()
        if config.get(name) != expected
    ]
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or contract.architecture not in architectures:
        mismatches.append(
            f"architectures={architectures!r} "
            f"(expected {contract.architecture!r})"
        )
    max_positions = config.get("max_position_embeddings")
    if not isinstance(max_positions, int) or (
        max_positions < contract.minimum_position_embeddings
    ):
        mismatches.append(
            f"max_position_embeddings={max_positions!r} "
            f"(minimum {contract.minimum_position_embeddings})"
        )
    if contract.num_labels is not None:
        label_count = config.get("num_labels")
        if label_count is None and isinstance(config.get("id2label"), dict):
            label_count = len(config["id2label"])
        # Older Transformers configs omit both fields; BertConfig then uses
        # its two-label default. Reject only an explicit conflicting value.
        if label_count is not None and label_count != contract.num_labels:
            mismatches.append(
                f"label count={label_count!r} (expected {contract.num_labels})"
            )
    if mismatches:
        raise ValueError(
            "Unverified Hugging Face model config: " + "; ".join(mismatches)
        )

    has_weights = any(
        (path / filename).is_file()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
        )
    ) or any(candidate.is_file() for candidate in path.glob("model-*.safetensors"))
    if not has_weights:
        raise ValueError(f"Hugging Face model weights are missing: {path}")


_SST2_SOURCE = HuggingFaceSourceContract(
    directory_name="textattack_bert-base-uncased-SST-2",
    architecture="BertForSequenceClassification",
    model_type="bert",
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    vocab_size=30522,
    minimum_position_embeddings=128,
    num_labels=2,
)

_SQUAD_SOURCE = HuggingFaceSourceContract(
    directory_name="csarron_bert-base-uncased-squad-v1",
    architecture="BertForQuestionAnswering",
    model_type="bert",
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    vocab_size=30522,
    minimum_position_embeddings=384,
    num_labels=2,
)


def _validate_loaded_label_count(model: object, expected: int = 2) -> None:
    loaded_label_count = getattr(getattr(model, "config", None), "num_labels", None)
    if loaded_label_count != expected:
        raise ValueError(
            "Unverified loaded model label count: "
            f"expected {expected}, got {loaded_label_count!r}"
        )


def _load_bert_classification(path: Path):
    _validate_huggingface_source(path, _SST2_SOURCE)
    import torch
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    _validate_loaded_label_count(base)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0]

    return Wrapper(base).eval()


def _load_bert_qa(path: Path):
    _validate_huggingface_source(path, _SQUAD_SOURCE)
    import torch
    from transformers import AutoModelForQuestionAnswering

    base = AutoModelForQuestionAnswering.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    _validate_loaded_label_count(base)

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask, token_type_ids):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=False,
            )
            return output[0], output[1]

    return Wrapper(base).eval()


_DEFAULT_TACTIC_HINT = "Default"

_ADAPTERS = {
    "bert-base-uncased": TorchModelAdapter(
        model_name="bert-base-uncased",
        task=Task.NLP_CLASSIFICATION,
        input_names=("input_ids", "attention_mask"),
        input_shapes={"input_ids": (1, 128), "attention_mask": (1, 128)},
        input_dtypes={"input_ids": "int64", "attention_mask": "int64"},
        output_names=("logits",),
        output_shapes={"logits": (1, 2)},
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_bert_classification,
        source_contract=_SST2_SOURCE,
    ),
    "bert-base-uncased-squad-v1": TorchModelAdapter(
        model_name="bert-base-uncased-squad-v1",
        task=Task.QUESTION_ANSWERING,
        input_names=("input_ids", "attention_mask", "token_type_ids"),
        input_shapes={
            "input_ids": (1, 384),
            "attention_mask": (1, 384),
            "token_type_ids": (1, 384),
        },
        input_dtypes={
            "input_ids": "int64",
            "attention_mask": "int64",
            "token_type_ids": "int64",
        },
        output_names=("start_logits", "end_logits"),
        output_shapes={
            "start_logits": (1, 384),
            "end_logits": (1, 384),
        },
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_bert_qa,
        source_contract=_SQUAD_SOURCE,
    ),
}


def get_torch_model_adapter(model_name: str) -> TorchModelAdapter:
    try:
        return _ADAPTERS[model_name]
    except KeyError:
        supported = sorted(_ADAPTERS)
        raise ValueError(
            f"No Furiosa Torch adapter for '{model_name}'. Supported: {supported}"
        ) from None


__all__ = [
    "HuggingFaceSourceContract",
    "TorchModelAdapter",
    "get_torch_model_adapter",
]

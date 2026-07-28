"""Local PyTorch model construction contracts for the Furiosa Torch runtime."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class TorchModelAdapter:
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    tactic_hint: str
    loader: Callable[[Path], object]


def _load_resnet(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local ResNet50 ONNX model not found: {path}")
    if path.suffix.lower() != ".onnx":
        raise ValueError(f"ResNet50 Furiosa model must be an ONNX file: {path}")

    import torch
    import onnx
    from onnx import version_converter
    from onnx2torch import convert

    onnx_model = onnx.load(str(path))
    upgraded_model = version_converter.convert_version(onnx_model, 13)
    base = convert(upgraded_model).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            output = self.model(images)
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

    return Wrapper(base).eval()


def _load_yolov5(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local YOLOv5 checkpoint not found: {path}")
    if path.name != "yolov5mu.pt":
        raise ValueError(
            "Furiosa YOLOv5m uses the Ultralytics YOLOv5u-medium contract. "
            f"Expected checkpoint filename 'yolov5mu.pt', got: {path.name}"
        )

    import torch
    from ultralytics import YOLO

    base = YOLO(str(path)).model.eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            output = self.model(images)
            if isinstance(output, (tuple, list)):
                return output[0]
            return output

    return Wrapper(base).eval()


def _load_bert_classification(path: Path):
    import torch
    from transformers import AutoModelForSequenceClassification

    base = AutoModelForSequenceClassification.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()

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
    import torch
    from transformers import AutoModelForQuestionAnswering

    base = AutoModelForQuestionAnswering.from_pretrained(
        path,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()

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


def _load_patchtst_etth1(path: Path):
    import torch
    from transformers import PatchTSTForPrediction

    base = PatchTSTForPrediction.from_pretrained(
        path,
        local_files_only=True,
    ).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, past_values, past_observed_mask):
            output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                return_dict=True,
            )
            return output.prediction_outputs

    return Wrapper(base).eval()


def _load_patchtst_fm(path: Path):
    import torch
    try:
        from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
    except ImportError as exc:
        raise RuntimeError(
            "PatchTST-FM-R1 requires IBM granite-tsfm 0.3.6. The package "
            "declares transformers<5 while Furiosa Torch 2026.3 environments "
            "may use Transformers 5.x, so install its code without dependency "
            "resolution and then run the import smoke test: "
            "uv pip install --no-deps granite-tsfm==0.3.6"
        ) from exc

    base = PatchTSTFMForPrediction.from_pretrained(
        path,
        local_files_only=True,
    ).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, past_values, past_observed_mask):
            output = self.model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                prediction_length=96,
                return_dict=True,
            )
            return output.prediction_outputs

    return Wrapper(base).eval()


_DEFAULT_TACTIC_HINT = "Default"

_ADAPTERS = {
    "resnet50": TorchModelAdapter(
        input_names=("input",),
        output_names=("logits",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_resnet,
    ),
    "yolov5m": TorchModelAdapter(
        input_names=("input",),
        output_names=("output",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_yolov5,
    ),
    "bert-base-uncased": TorchModelAdapter(
        input_names=("input_ids", "attention_mask"),
        output_names=("logits",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_bert_classification,
    ),
    "bert-base-uncased-squad-v1": TorchModelAdapter(
        input_names=("input_ids", "attention_mask", "token_type_ids"),
        output_names=("start_logits", "end_logits"),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_bert_qa,
    ),
    "patchtst-fm-r1": TorchModelAdapter(
        input_names=("past_values", "past_observed_mask"),
        output_names=("predictions",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_patchtst_fm,
    ),
    "patchtst-etth1": TorchModelAdapter(
        input_names=("past_values", "past_observed_mask"),
        output_names=("predictions",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_patchtst_etth1,
    ),
}


def get_torch_model_adapter(name: str) -> TorchModelAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(f"No Furiosa Torch adapter for model: {name}") from None


__all__ = ["TorchModelAdapter", "get_torch_model_adapter"]

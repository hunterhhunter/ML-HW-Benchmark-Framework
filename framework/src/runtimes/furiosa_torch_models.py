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
    import torch
    from transformers import AutoModelForImageClassification

    base = AutoModelForImageClassification.from_pretrained(
        path,
        local_files_only=True,
    ).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(pixel_values=pixel_values, return_dict=False)[0]

    return Wrapper(base).eval()


def _load_yolov5(path: Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Local YOLOv5 checkpoint not found: {path}")

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
    ).eval()

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, attention_mask):
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )
            return output[0], output[1]

    return Wrapper(base).eval()


def _load_patchtst(path: Path):
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


_DEFAULT_TACTIC_HINT = "Default"

_ADAPTERS = {
    "resnet50": TorchModelAdapter(
        input_names=("input",),
        output_names=("logits",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_resnet,
    ),
    "yolov5m": TorchModelAdapter(
        input_names=("images",),
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
        input_names=("input_ids", "attention_mask"),
        output_names=("start_logits", "end_logits"),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_bert_qa,
    ),
    "patchtst-fm-r1": TorchModelAdapter(
        input_names=("past_values", "past_observed_mask"),
        output_names=("predictions",),
        tactic_hint=_DEFAULT_TACTIC_HINT,
        loader=_load_patchtst,
    ),
}


def get_torch_model_adapter(name: str) -> TorchModelAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError:
        raise ValueError(f"No Furiosa Torch adapter for model: {name}") from None


__all__ = ["TorchModelAdapter", "get_torch_model_adapter"]

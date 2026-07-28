import subprocess
import sys
import builtins
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from core.model_profiles import SUPPORTED_PROFILES
from runtimes.furiosa_torch_models import get_torch_model_adapter


@pytest.mark.parametrize(
    ("name", "inputs", "outputs"),
    [
        ("resnet50", ("input",), ("logits",)),
        ("yolov5m", ("input",), ("output",)),
        ("bert-base-uncased", ("input_ids", "attention_mask"), ("logits",)),
        (
            "bert-base-uncased-squad-v1",
            ("input_ids", "attention_mask", "token_type_ids"),
            ("start_logits", "end_logits"),
        ),
        (
            "patchtst-fm-r1",
            ("past_values", "past_observed_mask"),
            ("predictions",),
        ),
        (
            "patchtst-etth1",
            ("past_values", "past_observed_mask"),
            ("predictions",),
        ),
    ],
)
def test_adapter_contract(name, inputs, outputs):
    adapter = get_torch_model_adapter(name)

    assert adapter.input_names == inputs
    assert adapter.output_names == outputs
    assert adapter.tactic_hint == "Default"


def test_unknown_model_fails_closed():
    with pytest.raises(ValueError, match="Furiosa Torch adapter"):
        get_torch_model_adapter("unknown")


def test_adapter_lookup_does_not_import_model_or_vendor_packages():
    source_root = Path(__file__).resolve().parent.parent / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(source_root)!r})
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {{
        'torch', 'transformers', 'ultralytics', 'furiosa', 'onnx', 'onnx2torch'
    }}:
        raise AssertionError(f'unexpected eager import: {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from runtimes.furiosa_torch_models import get_torch_model_adapter
get_torch_model_adapter('resnet50')
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("name", "onnx_path", "torch_path"),
    [
        (
            "resnet50",
            "models/Kalray_resnet50/resnet50-v1-7s.onnx",
            "models/Kalray_resnet50/resnet50-v1-7s.onnx",
        ),
        (
            "yolov5m",
            "models/yolov5m/yolov5m.onnx",
            "models/yolov5m/yolov5mu.pt",
        ),
        (
            "bert-base-uncased",
            "models/bert-base-uncased/bert_sst2.onnx",
            "models/textattack_bert-base-uncased-SST-2",
        ),
        (
            "bert-base-uncased-squad-v1",
            "models/bert-base-uncased-squad-v1/squad.onnx",
            "models/csarron_bert-base-uncased-squad-v1",
        ),
        (
            "patchtst-fm-r1",
            "models/ibm-research_patchtst-fm-r1-ONNX/model.onnx",
            "models/ibm-research_patchtst-fm-r1",
        ),
        (
            "patchtst-etth1",
            "models/ibm-granite_granite-timeseries-patchtst-ONNX/model.onnx",
            "models/ibm-granite_granite-timeseries-patchtst",
        ),
    ],
)
def test_profiles_preserve_onnx_and_add_explicit_torch_source(
    name, onnx_path, torch_path
):
    profile = SUPPORTED_PROFILES[name]

    assert profile["default_model_path"] == onnx_path
    assert profile["default_torch_model_path"] == torch_path


class _FakeHuggingFaceModel(torch.nn.Module):
    def __init__(self, result):
        super().__init__()
        self.result = result
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.result


def _install_fake_transformers(monkeypatch, class_name, base, load_calls):
    module = ModuleType("transformers")

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            load_calls.append({"path": path, "kwargs": kwargs})
            return base

    setattr(module, class_name, FakeAutoModel)
    monkeypatch.setitem(sys.modules, "transformers", module)


def test_resnet_loader_converts_local_onnx_and_returns_logits(monkeypatch, tmp_path):
    logits = torch.randn(1, 1000)
    base = _FakeHuggingFaceModel((logits, "ignored"))
    onnx_graph = object()
    upgraded_onnx_graph = object()
    onnx_load_calls = []
    version_conversion_calls = []
    convert_calls = []

    onnx = ModuleType("onnx")

    def load(path):
        onnx_load_calls.append(path)
        return onnx_graph

    onnx.load = load

    def convert_version(graph, target_version):
        version_conversion_calls.append((graph, target_version))
        return upgraded_onnx_graph

    onnx.version_converter = SimpleNamespace(convert_version=convert_version)
    onnx2torch = ModuleType("onnx2torch")

    def convert(graph):
        convert_calls.append(graph)
        return base

    onnx2torch.convert = convert
    monkeypatch.setitem(sys.modules, "onnx", onnx)
    monkeypatch.setitem(sys.modules, "onnx2torch", onnx2torch)
    model_path = tmp_path / "resnet50-v1-7s.onnx"
    model_path.touch()

    wrapper = get_torch_model_adapter("resnet50").loader(model_path)
    images = torch.randn(1, 3, 224, 224)

    assert wrapper(images) is logits
    assert onnx_load_calls == [str(model_path)]
    assert version_conversion_calls == [(onnx_graph, 13)]
    assert convert_calls == [upgraded_onnx_graph]
    assert base.calls == [{"args": (images,), "kwargs": {}}]
    assert wrapper.training is False
    assert base.training is False


def _guard_onnx_imports(monkeypatch):
    imported = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"onnx", "onnx2torch"}:
            imported.append(name)
            raise AssertionError(f"unexpected conversion import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return imported


def test_resnet_loader_rejects_missing_onnx_before_conversion(monkeypatch, tmp_path):
    imported = _guard_onnx_imports(monkeypatch)
    missing_path = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="Local ResNet50 ONNX model"):
        get_torch_model_adapter("resnet50").loader(missing_path)

    assert imported == []


def test_resnet_loader_rejects_onnx_directory_before_conversion(
    monkeypatch, tmp_path
):
    imported = _guard_onnx_imports(monkeypatch)
    directory_path = tmp_path / "directory.onnx"
    directory_path.mkdir()

    with pytest.raises(FileNotFoundError, match="Local ResNet50 ONNX model"):
        get_torch_model_adapter("resnet50").loader(directory_path)

    assert imported == []


def test_resnet_loader_rejects_non_onnx_file_before_conversion(monkeypatch, tmp_path):
    imported = _guard_onnx_imports(monkeypatch)
    checkpoint_path = tmp_path / "resnet50.pt"
    checkpoint_path.touch()

    with pytest.raises(ValueError, match="must be an ONNX file"):
        get_torch_model_adapter("resnet50").loader(checkpoint_path)

    assert imported == []


def test_bert_classification_loader_returns_only_logits(monkeypatch, tmp_path):
    logits = torch.randn(1, 2)
    base = _FakeHuggingFaceModel((logits, "ignored"))
    load_calls = []
    _install_fake_transformers(
        monkeypatch, "AutoModelForSequenceClassification", base, load_calls
    )
    model_path = tmp_path / "sst2"
    model_path.mkdir()

    wrapper = get_torch_model_adapter("bert-base-uncased").loader(model_path)
    input_ids = torch.ones((1, 128), dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)

    assert wrapper(input_ids, attention_mask) is logits
    assert load_calls == [
        {
            "path": model_path,
            "kwargs": {
                "local_files_only": True,
                "attn_implementation": "eager",
            },
        }
    ]
    assert base.calls == [
        {
            "args": (),
            "kwargs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": False,
            },
        }
    ]


def test_bert_qa_loader_returns_raw_start_and_end_logits(monkeypatch, tmp_path):
    start_logits = torch.randn(1, 384)
    end_logits = torch.randn(1, 384)
    base = _FakeHuggingFaceModel((start_logits, end_logits, "ignored"))
    load_calls = []
    _install_fake_transformers(
        monkeypatch, "AutoModelForQuestionAnswering", base, load_calls
    )
    model_path = tmp_path / "squad"
    model_path.mkdir()

    wrapper = get_torch_model_adapter("bert-base-uncased-squad-v1").loader(
        model_path
    )
    input_ids = torch.ones((1, 384), dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)

    result = wrapper(input_ids, attention_mask, token_type_ids)
    assert result[0] is start_logits
    assert result[1] is end_logits
    assert load_calls == [
        {
            "path": model_path,
            "kwargs": {
                "local_files_only": True,
                "attn_implementation": "eager",
            },
        }
    ]
    assert base.calls == [
        {
            "args": (),
            "kwargs": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "return_dict": False,
            },
        }
    ]


def test_yolov5_loader_returns_raw_detection_tensor(monkeypatch, tmp_path):
    raw_detections = torch.randn(1, 84, 8400)
    auxiliary_outputs = [torch.randn(1, 3, 80, 80, 85)]

    class FakeYoloModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, images):
            self.inputs.append(images)
            return raw_detections, auxiliary_outputs

    base = FakeYoloModel()
    load_calls = []
    fuse_calls = []
    ultralytics = ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, path):
            load_calls.append(path)
            self.model = base

        def fuse(self):
            fuse_calls.append(self)
            return self

    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    model_path = tmp_path / "yolov5mu.pt"
    model_path.touch()

    wrapper = get_torch_model_adapter("yolov5m").loader(model_path)
    images = torch.randn(1, 3, 640, 640)

    assert wrapper(images) is raw_detections
    assert load_calls == [str(model_path)]
    assert len(fuse_calls) == 1
    assert base.inputs == [images]
    assert wrapper.training is False
    assert base.training is False


def test_yolov5_loader_rejects_missing_checkpoint_before_ultralytics_load(
    monkeypatch, tmp_path
):
    load_calls = []
    ultralytics = ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, path):
            load_calls.append(path)

    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    missing_path = tmp_path / "missing-yolov5m.pt"

    with pytest.raises(FileNotFoundError, match="Local YOLOv5 checkpoint"):
        get_torch_model_adapter("yolov5m").loader(missing_path)

    assert load_calls == []


def test_yolov5_loader_rejects_legacy_checkpoint_contract(monkeypatch, tmp_path):
    load_calls = []
    ultralytics = ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, path):
            load_calls.append(path)

    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    legacy_path = tmp_path / "yolov5m.pt"
    legacy_path.touch()

    with pytest.raises(ValueError, match="yolov5mu.pt"):
        get_torch_model_adapter("yolov5m").loader(legacy_path)

    assert load_calls == []


def test_yolov5mu_cpu_forward_matches_static_output_contract():
    model_path = Path(SUPPORTED_PROFILES["yolov5m"]["default_torch_model_path"])
    if not model_path.is_file():
        pytest.skip(f"YOLOv5u medium checkpoint is not available: {model_path}")

    wrapper = get_torch_model_adapter("yolov5m").loader(model_path)
    with torch.inference_mode():
        output = wrapper(torch.zeros(1, 3, 640, 640))

    assert tuple(output.shape) == (1, 84, 8400)


def test_patchtst_etth1_loader_passes_mask_and_returns_predictions(monkeypatch, tmp_path):
    predictions = torch.randn(1, 96, 7)
    output = SimpleNamespace(prediction_outputs=predictions)
    base = _FakeHuggingFaceModel(output)
    load_calls = []
    _install_fake_transformers(
        monkeypatch, "PatchTSTForPrediction", base, load_calls
    )
    model_path = tmp_path / "patchtst"
    model_path.mkdir()

    wrapper = get_torch_model_adapter("patchtst-etth1").loader(model_path)
    past_values = torch.randn(1, 512, 7)
    past_observed_mask = torch.ones((1, 512, 7), dtype=torch.bool)

    assert wrapper(past_values, past_observed_mask) is predictions
    assert load_calls == [
        {"path": model_path, "kwargs": {"local_files_only": True}}
    ]
    assert base.calls == [
        {
            "args": (),
            "kwargs": {
                "past_values": past_values,
                "past_observed_mask": past_observed_mask,
                "return_dict": True,
            },
        }
    ]


def test_patchtst_fm_loader_uses_exact_tsfm_architecture(monkeypatch, tmp_path):
    predictions_bhc = torch.randn(1, 96, 7)
    output = SimpleNamespace(prediction_outputs=predictions_bhc)
    base = _FakeHuggingFaceModel(output)
    load_calls = []

    module = ModuleType("tsfm_public.models.patchtst_fm")

    class FakePatchTSTFMForPrediction:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            load_calls.append({"path": path, "kwargs": kwargs})
            return base

    module.PatchTSTFMForPrediction = FakePatchTSTFMForPrediction
    monkeypatch.setitem(sys.modules, "tsfm_public", ModuleType("tsfm_public"))
    monkeypatch.setitem(sys.modules, "tsfm_public.models", ModuleType("tsfm_public.models"))
    monkeypatch.setitem(sys.modules, "tsfm_public.models.patchtst_fm", module)
    model_path = tmp_path / "patchtst-fm-r1"
    model_path.mkdir()

    wrapper = get_torch_model_adapter("patchtst-fm-r1").loader(model_path)
    past_values = torch.randn(1, 512, 7)
    past_observed_mask = torch.ones((1, 512, 7), dtype=torch.bool)

    result = wrapper(past_values, past_observed_mask)

    assert result.shape == (1, 96, 7)
    assert result is predictions_bhc
    assert load_calls == [
        {"path": model_path, "kwargs": {"local_files_only": True}}
    ]
    assert base.calls == [
        {
            "args": (),
            "kwargs": {
                "past_values": past_values,
                "past_observed_mask": past_observed_mask,
                "prediction_length": 96,
                "return_dict": True,
            },
        }
    ]


def test_patchtst_fm_loader_explains_optional_dependency(monkeypatch, tmp_path):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "tsfm_public.models.patchtst_fm":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model_path = tmp_path / "patchtst-fm-r1"
    model_path.mkdir()

    with pytest.raises(RuntimeError, match="--no-deps granite-tsfm==0.3.6"):
        get_torch_model_adapter("patchtst-fm-r1").loader(model_path)

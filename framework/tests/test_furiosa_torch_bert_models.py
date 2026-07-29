import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from runtimes.furiosa_torch_models import get_torch_model_adapter


class _FakeHuggingFaceModel(torch.nn.Module):
    def __init__(self, result, *, num_labels=2):
        super().__init__()
        self.result = result
        self.calls = []
        self.config = SimpleNamespace(num_labels=num_labels)

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


def _write_model_source(root: Path, directory_name: str, architecture: str):
    source = root / directory_name
    source.mkdir()
    config = {
        "model_type": "bert",
        "architectures": [architecture],
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 30522,
        "max_position_embeddings": 512,
    }
    if architecture == "BertForSequenceClassification":
        config["id2label"] = {"0": "LABEL_0", "1": "LABEL_1"}
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").touch()
    return source


def test_adapter_lookup_does_not_import_torch_transformers_or_furiosa():
    source_root = Path(__file__).resolve().parent.parent / "src"
    script = f"""
import builtins
import sys

sys.path.insert(0, {str(source_root)!r})
original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {{'torch', 'transformers', 'furiosa'}}:
        raise AssertionError(f'unexpected eager import: {{name}}')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from runtimes.furiosa_torch_models import get_torch_model_adapter
get_torch_model_adapter('bert-base-uncased')
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_bert_classification_loader_uses_eager_attention_and_returns_logits(
    monkeypatch, tmp_path
):
    logits = torch.randn(1, 2)
    base = _FakeHuggingFaceModel((logits, "ignored"))
    load_calls = []
    _install_fake_transformers(
        monkeypatch,
        "AutoModelForSequenceClassification",
        base,
        load_calls,
    )
    model_path = _write_model_source(
        tmp_path,
        "textattack_bert-base-uncased-SST-2",
        "BertForSequenceClassification",
    )

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
    assert wrapper.training is False
    assert base.training is False


def test_bert_qa_loader_passes_token_types_and_returns_raw_logits(
    monkeypatch, tmp_path
):
    start_logits = torch.randn(1, 384)
    end_logits = torch.randn(1, 384)
    base = _FakeHuggingFaceModel((start_logits, end_logits, "ignored"))
    load_calls = []
    _install_fake_transformers(
        monkeypatch,
        "AutoModelForQuestionAnswering",
        base,
        load_calls,
    )
    model_path = _write_model_source(
        tmp_path,
        "csarron_bert-base-uncased-squad-v1",
        "BertForQuestionAnswering",
    )

    wrapper = get_torch_model_adapter("bert-base-uncased-squad-v1").loader(
        model_path
    )
    input_ids = torch.ones((1, 384), dtype=torch.int64)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)

    output = wrapper(input_ids, attention_mask, token_type_ids)

    assert output == (start_logits, end_logits)
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


@pytest.mark.parametrize(
    ("model_name", "directory_name", "architecture"),
    [
        (
            "bert-base-uncased",
            "textattack_bert-base-uncased-SST-2",
            "BertForSequenceClassification",
        ),
        (
            "bert-base-uncased-squad-v1",
            "csarron_bert-base-uncased-squad-v1",
            "BertForQuestionAnswering",
        ),
    ],
)
def test_adapter_rejects_unverified_source_identity_and_architecture(
    tmp_path, model_name, directory_name, architecture
):
    adapter = get_torch_model_adapter(model_name)
    wrong_name = _write_model_source(tmp_path, "renamed-model", architecture)
    with pytest.raises(ValueError, match="directory name"):
        adapter.validate_source(wrong_name)

    wrong_architecture = _write_model_source(
        tmp_path,
        directory_name,
        "BertForMaskedLM",
    )
    with pytest.raises(ValueError, match="architectures"):
        adapter.validate_source(wrong_architecture)


def test_classification_adapter_rejects_wrong_label_count(tmp_path):
    source = _write_model_source(
        tmp_path,
        "textattack_bert-base-uncased-SST-2",
        "BertForSequenceClassification",
    )
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["id2label"] = {"0": "A", "1": "B", "2": "C"}
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="label count"):
        get_torch_model_adapter("bert-base-uncased").validate_source(source)


def test_classification_adapter_accepts_legacy_config_default_label_count(tmp_path):
    source = _write_model_source(
        tmp_path,
        "textattack_bert-base-uncased-SST-2",
        "BertForSequenceClassification",
    )
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config.pop("id2label")
    config_path.write_text(json.dumps(config))

    get_torch_model_adapter("bert-base-uncased").validate_source(source)


def test_classification_loader_rejects_loaded_head_with_wrong_label_count(
    monkeypatch, tmp_path
):
    base = _FakeHuggingFaceModel((torch.randn(1, 3),), num_labels=3)
    _install_fake_transformers(
        monkeypatch,
        "AutoModelForSequenceClassification",
        base,
        [],
    )
    source = _write_model_source(
        tmp_path,
        "textattack_bert-base-uncased-SST-2",
        "BertForSequenceClassification",
    )

    with pytest.raises(ValueError, match="loaded model label count"):
        get_torch_model_adapter("bert-base-uncased").loader(source)


def test_qa_adapter_rejects_explicit_wrong_label_count(tmp_path):
    source = _write_model_source(
        tmp_path,
        "csarron_bert-base-uncased-squad-v1",
        "BertForQuestionAnswering",
    )
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["num_labels"] = 3
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="label count"):
        get_torch_model_adapter("bert-base-uncased-squad-v1").validate_source(
            source
        )


def test_qa_loader_rejects_loaded_head_with_wrong_label_count(monkeypatch, tmp_path):
    base = _FakeHuggingFaceModel(
        (torch.randn(1, 384), torch.randn(1, 384)),
        num_labels=3,
    )
    _install_fake_transformers(
        monkeypatch,
        "AutoModelForQuestionAnswering",
        base,
        [],
    )
    source = _write_model_source(
        tmp_path,
        "csarron_bert-base-uncased-squad-v1",
        "BertForQuestionAnswering",
    )

    with pytest.raises(ValueError, match="loaded model label count"):
        get_torch_model_adapter("bert-base-uncased-squad-v1").loader(source)

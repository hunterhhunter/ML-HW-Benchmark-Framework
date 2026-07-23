import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

from core.model_profiles import SUPPORTED_PROFILES
from preprocessor.bert_qa_preprocessor import BertQAPreprocessor


def test_bert_qa_profile_declares_three_int64_inputs():
    profile = SUPPORTED_PROFILES["bert-base-uncased-squad-v1"]

    assert profile["input_shapes"] == {
        "input_ids": (1, 384),
        "attention_mask": (1, 384),
        "token_type_ids": (1, 384),
    }
    assert profile["input_dtype"] == {
        "input_ids": "int64",
        "attention_mask": "int64",
        "token_type_ids": "int64",
    }


def test_bert_qa_preprocessor_requires_token_type_ids_for_cache(tmp_path):
    for file_name in (
        "input_ids.npy",
        "attention_mask.npy",
        "start_positions.npy",
        "end_positions.npy",
    ):
        np.save(tmp_path / file_name, np.zeros((1,), dtype=np.int64))

    assert not BertQAPreprocessor().is_preprocessed(str(tmp_path))


def test_bert_qa_preprocessor_saves_token_type_ids(
    tmp_path, monkeypatch
):
    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            assert model_id == "csarron/bert-base-uncased-squad-v1"

            def tokenize(question, context, **kwargs):
                assert question == "Where?"
                assert context == "Answer here."
                assert kwargs == {
                    "max_length": 4,
                    "padding": "max_length",
                    "truncation": "only_second",
                    "return_offsets_mapping": True,
                }
                return {
                    "input_ids": [101, 102, 103, 0],
                    "attention_mask": [1, 1, 1, 0],
                    "token_type_ids": [0, 0, 1, 0],
                    "offset_mapping": [(0, 0), (0, 0), (0, 6), (0, 0)],
                }

            return tokenize

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: [
        {
            "question": "Where?",
            "context": "Answer here.",
            "answers": {"text": ["Answer"], "answer_start": [0]},
        }
    ]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    BertQAPreprocessor(seq_len=4).preprocess_dataset(str(tmp_path))

    np.testing.assert_array_equal(
        np.load(tmp_path / "token_type_ids.npy"),
        np.array([[0, 0, 1, 0]], dtype=np.int64),
    )
    assert BertQAPreprocessor(seq_len=4).is_preprocessed(str(tmp_path))


def test_prepare_squad_numpy_script_saves_token_type_ids(
    tmp_path, monkeypatch
):
    class FakeEncoding(dict):
        def sequence_ids(self):
            return [None, 0, 1, None]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            def tokenize(question, context, **kwargs):
                return FakeEncoding(
                    input_ids=[101, 102, 103, 0],
                    attention_mask=[1, 1, 1, 0],
                    token_type_ids=[0, 0, 1, 0],
                    offset_mapping=[(0, 0), (0, 0), (0, 6), (0, 0)],
                )

            return tokenize

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *args, **kwargs: [
        {
            "question": "Where?",
            "context": "Answer here.",
            "answers": {"text": ["Answer"], "answer_start": [0]},
        }
    ]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_squad_numpy.py",
            "--seq-len",
            "4",
            "--output-dir",
            str(tmp_path),
        ],
    )

    module_path = (
        Path(__file__).resolve().parent.parent
        / "datasets"
        / "prepare_squad_numpy.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_squad_numpy_contract_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.main()

    np.testing.assert_array_equal(
        np.load(tmp_path / "token_type_ids.npy"),
        np.array([[0, 0, 1, 0]], dtype=np.int64),
    )


def test_bert_qa_onnx_export_uses_three_named_inputs(monkeypatch):
    captured = {}
    input_markers = {
        "input_ids": object(),
        "attention_mask": object(),
        "token_type_ids": object(),
    }

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            return lambda *args, **kwargs: input_markers

    class FakeModel:
        def eval(self):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id):
            return FakeModel()

    class NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    torch = types.ModuleType("torch")
    torch.no_grad = NoGrad
    torch.onnx = types.SimpleNamespace(
        export=lambda *args, **kwargs: captured.update(
            {"args": args, "kwargs": kwargs}
        )
    )
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForQuestionAnswering = FakeAutoModel
    transformers.AutoTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    module_path = (
        Path(__file__).resolve().parent.parent
        / "models"
        / "prepare_bert_squad.py"
    )
    spec = importlib.util.spec_from_file_location(
        "prepare_bert_squad_contract_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.os, "makedirs", lambda *args, **kwargs: None)

    module.export_bert_squad()

    assert captured["args"][1] == (
        input_markers["input_ids"],
        input_markers["attention_mask"],
        input_markers["token_type_ids"],
    )
    assert captured["kwargs"]["input_names"] == [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    assert "token_type_ids" in captured["kwargs"]["dynamic_axes"]

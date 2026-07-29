import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from core.model_profiles import SUPPORTED_PROFILES, create_model_spec
from core.targets import get_target, resolve_target, validate_registry_graph
from runtimes import get_runtime_entry
import main as benchmark_main


BERT_MODELS = (
    "bert-base-uncased",
    "bert-base-uncased-squad-v1",
)


def _write_verified_source(root: Path, model_name: str) -> Path:
    if model_name == "bert-base-uncased":
        directory_name = "textattack_bert-base-uncased-SST-2"
        architecture = "BertForSequenceClassification"
    else:
        directory_name = "csarron_bert-base-uncased-squad-v1"
        architecture = "BertForQuestionAnswering"
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
    if model_name == "bert-base-uncased":
        config["id2label"] = {"0": "LABEL_0", "1": "LABEL_1"}
    (source / "config.json").write_text(json.dumps(config))
    (source / "model.safetensors").touch()
    return source


def test_furiosa_torch_target_is_strict_static_and_non_native_async():
    target = get_target("furiosa-rngd-torch")

    assert target.runtime_name == "furiosa_torch"
    assert target.device == "npu:0"
    assert target.artifact_format == "pytorch_model"
    assert target.monitor_names == ("system",)
    assert "sync" in target.capabilities
    assert "static_shape" in target.capabilities
    assert "native_async" not in target.capabilities
    assert "generation" not in target.capabilities
    assert validate_registry_graph([target], strict=True)["ok"] is True


def test_furiosa_torch_runtime_registry_is_lazy_and_has_explicit_aliases():
    entry = get_runtime_entry("furiosa_torch")

    assert entry.name == "furiosa_torch"
    assert get_runtime_entry("furiosa-torch") is entry
    assert get_runtime_entry("rngd_torch") is entry
    assert resolve_target(None, "furiosa_torch", "npu:0").target_id == (
        "furiosa-rngd-torch"
    )
    assert resolve_target(None, "rngd", "npu:0").target_id == "furiosa-rngd"


@pytest.mark.parametrize("model_name", BERT_MODELS)
def test_furiosa_torch_bert_profiles_have_local_huggingface_sources(model_name):
    profile = SUPPORTED_PROFILES[model_name]
    source = profile["default_torch_model_path"]

    assert source.startswith("models/")
    spec = create_model_spec(
        model_name,
        source,
        task=profile["task"],
        sniff_onnx=False,
        source_format="pytorch_model",
    )
    assert spec.model_paths == {"pytorch_model": source}
    assert all(shape[0] == 1 for shape in spec.input_shapes.values())


def test_only_server_verified_bert_models_are_registered():
    from runtimes.furiosa_torch_models import get_torch_model_adapter

    for model_name in BERT_MODELS:
        assert get_torch_model_adapter(model_name).input_names

    for unverified_model in ("resnet50", "yolov5m", "patchtst-fm-r1"):
        with pytest.raises(ValueError, match="Furiosa Torch adapter"):
            get_torch_model_adapter(unverified_model)


def test_furiosa_torch_profile_paths_match_server_validation_layout():
    assert SUPPORTED_PROFILES["bert-base-uncased"]["default_torch_model_path"] == (
        "models/textattack_bert-base-uncased-SST-2"
    )
    assert SUPPORTED_PROFILES["bert-base-uncased-squad-v1"][
        "default_torch_model_path"
    ] == "models/csarron_bert-base-uncased-squad-v1"


def test_parser_accepts_furiosa_torch_backend():
    args = benchmark_main.build_parser().parse_args(
        ["--model", "bert-base-uncased", "--backend", "furiosa_torch"]
    )

    assert args.backend == "furiosa_torch"


def _torch_args(model_path, **overrides):
    values = {
        "model": "bert-base-uncased",
        "model_path": str(model_path) if model_path is not None else None,
        "batch_size": 1,
        "worker_count": None,
        "compile": True,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("model_name", BERT_MODELS)
def test_validate_furiosa_torch_cli_accepts_server_verified_bert(
    tmp_path, model_name
):
    source = _write_verified_source(tmp_path, model_name)

    assert benchmark_main._validate_furiosa_torch_cli(
        _torch_args(source, model=model_name),
        SUPPORTED_PROFILES[model_name]["task"],
    ) == source.resolve()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"batch_size": 2}, "batch size exactly 1"),
        ({"worker_count": 2}, "worker count exactly 1"),
        ({"compile": False}, "does not support --no-compile"),
        ({"model": "resnet50"}, "no model adapter"),
        ({"model": "yolov5m"}, "no model adapter"),
        ({"model": "patchtst-fm-r1"}, "no model adapter"),
    ],
)
def test_validate_furiosa_torch_cli_rejects_unverified_contracts(
    tmp_path, overrides, message
):
    source = tmp_path / "model"
    source.mkdir()

    with pytest.raises(ValueError, match=message):
        benchmark_main._validate_furiosa_torch_cli(
            _torch_args(source, **overrides),
            SUPPORTED_PROFILES["bert-base-uncased"]["task"],
        )


def test_furiosa_torch_async_uses_framework_blocking_executor():
    args = benchmark_main.build_parser().parse_args(
        [
            "--model",
            "bert-base-uncased",
            "--target",
            "furiosa-rngd-torch",
            "--inference-mode",
            "async_queue",
            "--worker-count",
            "1",
            "--queue-capacity",
            "4",
            "--max-samples",
            "1",
        ]
    )
    config = benchmark_main.build_async_config(args)

    class RuntimeThatMustNotExposeNativeAsync:
        def create_native_backend(self):
            raise AssertionError("Furiosa Torch must use the framework async queue")

    executor = benchmark_main._build_async_runtime_executor(
        args,
        get_target("furiosa-rngd-torch"),
        RuntimeThatMustNotExposeNativeAsync(),
        object(),
        config,
    )

    assert executor is None


@pytest.mark.parametrize("model_name", BERT_MODELS)
@pytest.mark.parametrize("inference_mode", ["e2e", "async_queue"])
def test_furiosa_torch_main_routes_bert_to_common_pipeline(
    monkeypatch, tmp_path, model_name, inference_mode
):
    profile = SUPPORTED_PROFILES[model_name]
    model_path = _write_verified_source(tmp_path, model_name)
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    captured = {}
    spec = create_model_spec(
        model_name,
        str(model_path),
        task=profile["task"],
        sniff_onnx=False,
        source_format="pytorch_model",
    )

    class FakeRuntime:
        def load(self, compiled_model):
            captured["compiled_model"] = compiled_model

    class FakeLoader:
        def get_metadata(self):
            return {}

    def fake_create_model_spec(name, source, **kwargs):
        captured["spec_request"] = (name, source, kwargs)
        return spec

    def fake_create_dataloader(**kwargs):
        captured["loader_kwargs"] = kwargs
        return FakeLoader()

    def fake_create_runtime(backend, **kwargs):
        captured["runtime_request"] = (backend, kwargs)
        return FakeRuntime()

    import utils.dataset_resolver as dataset_resolver

    monkeypatch.setattr(benchmark_main, "create_model_spec", fake_create_model_spec)
    monkeypatch.setattr(benchmark_main, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(benchmark_main, "create_runtime", fake_create_runtime)
    monkeypatch.setattr(benchmark_main, "create_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark_main, "create_decoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark_main, "execute_benchmark", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        benchmark_main,
        "_run_prepare_script",
        lambda script: (_ for _ in ()).throw(
            AssertionError(f"unexpected model preparation: {script}")
        ),
    )
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    argv = [
        "main.py",
        "--model",
        model_name,
        "--target",
        "furiosa-rngd-torch",
        "--model-path",
        str(model_path),
        "--dataset",
        str(dataset_path),
        "--inference-mode",
        inference_mode,
    ]
    if inference_mode == "async_queue":
        argv.extend(
            [
                "--worker-count",
                "1",
                "--queue-capacity",
                "4",
                "--max-samples",
                "1",
            ]
        )
    else:
        argv.extend(["--max-steps", "1"])
    monkeypatch.setattr(sys, "argv", argv)

    assert benchmark_main.main() == 0
    assert captured["spec_request"] == (
        model_name,
        str(model_path.resolve()),
        {
            "task": profile["task"],
            "sniff_onnx": False,
            "source_format": "pytorch_model",
        },
    )
    assert captured["loader_kwargs"]["backend"] == "furiosa_torch"
    assert captured["runtime_request"][0] == "furiosa_torch"
    assert captured["compiled_model"].artifact_path == model_path.resolve()

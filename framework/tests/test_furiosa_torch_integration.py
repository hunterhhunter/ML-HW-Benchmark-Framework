import sys
from argparse import Namespace
from pathlib import Path

import pytest

import main as benchmark_main
import runtimes as runtime_registry
from core.model_profiles import SUPPORTED_PROFILES, create_model_spec
from core.model_spec import Task
from core.targets import get_target, resolve_target, validate_registry_graph
from runtimes import get_runtime_entry
from runtimes.furiosa_torch_models import get_torch_model_adapter


def test_registry_exposes_distinct_furiosa_torch_runtime_without_vendor_import(monkeypatch):
    for module_name in tuple(sys.modules):
        if module_name == "furiosa" or module_name.startswith("furiosa."):
            monkeypatch.delitem(sys.modules, module_name, raising=False)

    entry = get_runtime_entry("rngd_torch")

    assert entry.name == "furiosa_torch"
    assert get_runtime_entry("furiosa-torch") is entry
    assert runtime_registry.FuriosaTorchRuntime.__name__ == "FuriosaTorchRuntime"
    assert "furiosa" not in sys.modules
    assert "furiosa.torch" not in sys.modules


def test_registry_exposes_non_native_async_tensor_target():
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


def test_furiosa_torch_backend_is_preserved_in_failure_diagnostics():
    assert "furiosa_torch" in benchmark_main._SAFE_RUNTIME_BACKENDS


def test_legacy_backend_resolution_maps_only_explicit_torch_aliases():
    assert resolve_target(None, "furiosa_torch", "npu:0").target_id == "furiosa-rngd-torch"
    assert resolve_target(None, "rngd_torch", "npu:0").target_id == "furiosa-rngd-torch"
    assert resolve_target(None, "rngd", "npu:0").target_id == "furiosa-rngd"


def test_parser_accepts_furiosa_torch_backend():
    args = benchmark_main.build_parser().parse_args(
        ["--model", "resnet50", "--backend", "furiosa_torch"]
    )

    assert args.backend == "furiosa_torch"


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet50",
        "yolov5m",
        "bert-base-uncased",
        "bert-base-uncased-squad-v1",
        "patchtst-fm-r1",
        "patchtst-etth1",
    ],
)
def test_furiosa_torch_profiles_have_explicit_local_sources(model_name):
    source = SUPPORTED_PROFILES[model_name]["default_torch_model_path"]
    assert source.startswith("models/")


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet50",
        "yolov5m",
        "bert-base-uncased",
        "bert-base-uncased-squad-v1",
        "patchtst-fm-r1",
        "patchtst-etth1",
    ],
)
def test_furiosa_torch_profiles_match_adapter_static_io_contract(model_name):
    profile = SUPPORTED_PROFILES[model_name]
    spec = create_model_spec(
        model_name,
        profile["default_torch_model_path"],
        task=profile["task"],
        sniff_onnx=False,
        source_format="pytorch_model",
    )
    adapter = get_torch_model_adapter(model_name)

    assert tuple(spec.input_shapes) == adapter.input_names
    assert all(shape[0] == 1 for shape in spec.input_shapes.values())
    assert len(spec.output_shapes) == len(adapter.output_names)


def test_furiosa_torch_model_spec_does_not_sniff_onnx(tmp_path):
    model_path = tmp_path / "yolov5mu.pt"
    model_path.write_bytes(b"checkpoint")

    spec = create_model_spec(
        "yolov5m",
        str(model_path),
        sniff_onnx=False,
        source_format="pytorch_model",
    )

    assert spec.input_shapes == {"input": (1, 3, 640, 640)}
    assert spec.output_shapes == {"output": (1, 84, 8400)}
    assert spec.model_paths == {"pytorch_model": str(model_path)}


def _torch_args(model_path, **overrides):
    values = {
        "model": "resnet50",
        "model_path": str(model_path) if model_path is not None else None,
        "batch_size": 1,
        "worker_count": None,
    }
    values.update(overrides)
    return Namespace(**values)


def test_validate_furiosa_torch_cli_accepts_file_or_directory(tmp_path):
    source_file = tmp_path / "resnet.onnx"
    source_file.write_bytes(b"onnx")
    source_dir = tmp_path / "bert"
    source_dir.mkdir()

    assert benchmark_main._validate_furiosa_torch_cli(
        _torch_args(source_file), Task.IMAGE_CLASSIFICATION
    ) == source_file.resolve()
    assert benchmark_main._validate_furiosa_torch_cli(
        _torch_args(source_dir, model="bert-base-uncased"),
        Task.NLP_CLASSIFICATION,
    ) == source_dir.resolve()


@pytest.mark.parametrize(
    ("overrides", "task", "message"),
    [
        ({"batch_size": 2}, Task.IMAGE_CLASSIFICATION, "batch size exactly 1"),
        ({"worker_count": 2}, Task.IMAGE_CLASSIFICATION, "worker count exactly 1"),
        ({"compile": False}, Task.IMAGE_CLASSIFICATION, "does not support --no-compile"),
        ({"model": "llama-3.1-8b"}, Task.NLP_GENERATION, "does not support"),
        ({"model": "unknown"}, Task.IMAGE_CLASSIFICATION, "adapter"),
    ],
)
def test_validate_furiosa_torch_cli_rejects_unsupported_contracts(
    tmp_path, overrides, task, message
):
    source = tmp_path / "model"
    source.mkdir()

    with pytest.raises(ValueError, match=message):
        benchmark_main._validate_furiosa_torch_cli(
            _torch_args(source, **overrides),
            task,
        )


def test_validate_furiosa_torch_cli_requires_existing_local_source(tmp_path):
    with pytest.raises(ValueError, match="existing local"):
        benchmark_main._validate_furiosa_torch_cli(
            _torch_args(tmp_path / "missing"),
            Task.IMAGE_CLASSIFICATION,
        )


def test_furiosa_torch_patchtst_fm_disables_duplicate_loader_normalization():
    loader_kwargs = {"cache_dir": "/tmp/cache"}

    benchmark_main._apply_furiosa_torch_loader_contract(
        loader_kwargs,
        model_name="patchtst-fm-r1",
    )

    assert loader_kwargs == {
        "cache_dir": "/tmp/cache",
        "normalize": False,
    }


def test_furiosa_torch_standard_patchtst_keeps_loader_normalization():
    loader_kwargs = {"normalize": True}

    benchmark_main._apply_furiosa_torch_loader_contract(
        loader_kwargs,
        model_name="patchtst-etth1",
    )

    assert loader_kwargs == {"normalize": True}


def test_furiosa_torch_async_uses_framework_blocking_executor():
    args = benchmark_main.build_parser().parse_args(
        [
            "--model",
            "resnet50",
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


@pytest.mark.parametrize(
    "model_name",
    [
        "resnet50",
        "yolov5m",
        "bert-base-uncased",
        "bert-base-uncased-squad-v1",
        "patchtst-fm-r1",
        "patchtst-etth1",
    ],
)
@pytest.mark.parametrize("inference_mode", ["e2e", "async_queue"])
def test_furiosa_torch_main_routes_all_models_to_common_pipeline(
    monkeypatch, tmp_path, model_name, inference_mode
):
    profile = SUPPORTED_PROFILES[model_name]
    source_name = Path(profile["default_torch_model_path"]).name
    model_path = tmp_path / source_name
    if model_path.suffix in {".onnx", ".pt"}:
        model_path.write_bytes(b"checkpoint")
    else:
        model_path.mkdir()
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
    if model_name == "patchtst-fm-r1":
        assert captured["loader_kwargs"]["normalize"] is False

import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dataloader as dataloader_package
import main as benchmark_main
from core.model_spec import Model_Spec
from dataloader.mobilint_vision_profiles import (
    MOBILINT_RESNET50_IMAGENET1K_V2,
    MOBILINT_YOLOV5M_DEFAULT,
)


def _model_spec(task):
    if task is benchmark_main.Task.IMAGE_CLASSIFICATION:
        return Model_Spec(
            name="resnet50",
            task=task,
            input_shapes={"input": (1, 3, 224, 224)},
            input_dtype={"input": "float32"},
            output_shapes={"output": (1, 1000)},
        )
    return Model_Spec(
        name="yolov5m",
        task=task,
        input_shapes={"images": (1, 3, 640, 640)},
        input_dtype={"images": "float32"},
        output_shapes={"output": (1, 25200, 85)},
    )


@pytest.mark.parametrize(
    ("task", "loader_name", "profile"),
    [
        (
            benchmark_main.Task.IMAGE_CLASSIFICATION,
            "MobilintImageClassificationLoader",
            MOBILINT_RESNET50_IMAGENET1K_V2,
        ),
        (
            benchmark_main.Task.OBJECT_DETECTION,
            "MobilintObjectDetectionLoader",
            MOBILINT_YOLOV5M_DEFAULT,
        ),
    ],
)
def test_create_dataloader_routes_mobilint_vision_before_generic_loaders(
    monkeypatch, task, loader_name, profile
):
    selected = object()
    calls = []

    def fake_loader(model_spec, **kwargs):
        calls.append((model_spec, kwargs))
        return selected

    monkeypatch.setattr(
        dataloader_package,
        loader_name,
        fake_loader,
        raising=False,
    )
    spec = _model_spec(task)

    actual = dataloader_package.create_dataloader(
        spec,
        backend="mobilint",
        mobilint_vision_profile=profile,
    )

    assert actual is selected
    assert calls == [
        (
            spec,
            {
                "backend": "mobilint",
                "mobilint_vision_profile": profile,
            },
        )
    ]


@pytest.mark.parametrize(
    ("backend", "task", "loader_name"),
    [
        ("onnxruntime", benchmark_main.Task.IMAGE_CLASSIFICATION, "ImageClassificationLoader"),
        ("onnxruntime", benchmark_main.Task.OBJECT_DETECTION, "ObjectDetectionLoader"),
        ("hailort", benchmark_main.Task.IMAGE_CLASSIFICATION, "HailoImageClassificationLoader"),
        ("deepx", benchmark_main.Task.OBJECT_DETECTION, "DeepXDataLoader"),
        ("mobilint", benchmark_main.Task.NLP_GENERATION, "LlamaLoader"),
    ],
)
def test_create_dataloader_keeps_existing_backend_routing(
    monkeypatch, backend, task, loader_name
):
    selected = object()
    monkeypatch.setattr(
        dataloader_package,
        loader_name,
        lambda model_spec, **kwargs: selected,
    )

    actual = dataloader_package.create_dataloader(
        _model_spec(task),
        backend=backend,
    )

    assert actual is selected


@pytest.mark.parametrize(
    "task",
    [
        benchmark_main.Task.SEMANTIC_SEGMENTATION,
        benchmark_main.Task.INSTANCE_SEGMENTATION,
        benchmark_main.Task.POSE_ESTIMATION,
    ],
)
def test_create_dataloader_rejects_unsupported_mobilint_vision_tasks(task):
    with pytest.raises(ValueError, match=rf"Mobilint vision task {task.name}.*not supported"):
        dataloader_package.create_dataloader(_model_spec(task), backend="mobilint")


def test_dataloader_package_exports_mobilint_vision_loaders():
    assert "MobilintImageClassificationLoader" in dataloader_package.__all__
    assert "MobilintObjectDetectionLoader" in dataloader_package.__all__
    assert dataloader_package.MobilintImageClassificationLoader is not None
    assert dataloader_package.MobilintObjectDetectionLoader is not None


def test_resolve_framework_path_uses_framework_root_for_profile_paths():
    resolved = benchmark_main._resolve_framework_path("models/yolov5m/yolov5m.onnx")

    assert resolved == str(
        benchmark_main.FRAMEWORK_ROOT / "models" / "yolov5m" / "yolov5m.onnx"
    )


def test_resolve_framework_path_leaves_absolute_paths_unchanged(tmp_path):
    absolute_path = tmp_path / "model.onnx"

    assert benchmark_main._resolve_framework_path(str(absolute_path)) == str(absolute_path)


def test_run_auto_prepare_executes_profile_script_from_framework_root(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, check, cwd):
        calls.append({"cmd": cmd, "check": check, "cwd": cwd})

    monkeypatch.setattr(benchmark_main.subprocess, "run", fake_run)

    args = Namespace(
        backend="onnxruntime",
        hef=None,
        artifact=None,
        compile=True,
        onnx=str(tmp_path / "missing.onnx"),
        dataset=str(tmp_path),
    )
    target = SimpleNamespace(uses_compiler=False, artifact_format="onnx")
    profile = {"prepare_model_script": "models/prepare_yolov5m.py"}

    benchmark_main.run_auto_prepare(profile, args, target)

    assert calls == [
        {
            "cmd": [
                sys.executable,
                str(benchmark_main.FRAMEWORK_ROOT / "models" / "prepare_yolov5m.py"),
            ],
            "check": True,
            "cwd": str(benchmark_main.FRAMEWORK_ROOT),
        }
    ]


def test_hailo_classification_runtime_defaults_to_float32_outputs():
    runtime_kwargs = {"input_format_type": "uint8", "output_format_type": "auto"}

    benchmark_main._apply_hailo_task_runtime_defaults(
        runtime_kwargs,
        cli_runtime_options={},
        task_enum=benchmark_main.Task.IMAGE_CLASSIFICATION,
    )

    assert runtime_kwargs["output_format_type"] == "float32"


def test_hailo_runtime_default_respects_cli_output_format_override():
    runtime_kwargs = {"input_format_type": "uint8", "output_format_type": "auto"}

    benchmark_main._apply_hailo_task_runtime_defaults(
        runtime_kwargs,
        cli_runtime_options={"output_format_type": "uint8"},
        task_enum=benchmark_main.Task.IMAGE_CLASSIFICATION,
    )

    assert runtime_kwargs["output_format_type"] == "auto"


def test_parser_exposes_furiosa_backend_and_explicit_fxb():
    args = benchmark_main.build_parser().parse_args(
        [
            "--model",
            "llama-3.2-3b",
            "--backend",
            "furiosa_llm",
            "--fxb",
            "model.fxb",
        ]
    )

    assert args.backend == "furiosa_llm"
    assert args.fxb == "model.fxb"


def test_parser_accepts_explicit_mobilint_target_and_generic_artifact():
    args = benchmark_main.build_parser().parse_args(
        [
            "--model",
            "resnet50",
            "--target",
            "mobilint-regulus",
            "--artifact",
            "/opt/models/resnet50.mxq",
        ]
    )

    assert args.target == "mobilint-regulus"
    assert args.artifact == "/opt/models/resnet50.mxq"
    assert args.backend == "onnxruntime"


def test_parser_defaults_mobilint_image_preprocess_profile_to_auto():
    args = benchmark_main.build_parser().parse_args(
        ["--model", "resnet50"]
    )

    assert args.image_preprocess_profile == "auto"


def test_parser_help_mentions_explicit_mobilint_targets_and_mxq_artifacts():
    parser = benchmark_main.build_parser()
    help_by_option = {
        option: action.help
        for action in parser._actions
        for option in action.option_strings
    }

    assert "Mobilint .mxq" in help_by_option["--artifact"]
    assert "mobilint-aries" in help_by_option["--target"]
    assert "mobilint-regulus" in help_by_option["--target"]


@pytest.mark.parametrize(
    ("backend", "task"),
    [
        ("onnxruntime", benchmark_main.Task.IMAGE_CLASSIFICATION),
        ("mobilint", benchmark_main.Task.NLP_GENERATION),
    ],
)
def test_explicit_image_preprocess_profile_is_scoped_to_mobilint_raw_vision(
    backend, task
):
    with pytest.raises(
        ValueError,
        match="supported only for Mobilint raw vision targets",
    ):
        benchmark_main._validate_image_preprocess_profile_scope(
            "mobilint-resnet50-imagenet1k-v2",
            backend=backend,
            task=task,
        )


def test_auto_image_preprocess_profile_is_allowed_outside_mobilint_vision():
    benchmark_main._validate_image_preprocess_profile_scope(
        "auto",
        backend="onnxruntime",
        task=benchmark_main.Task.IMAGE_CLASSIFICATION,
    )


def test_mobilint_runtime_diagnostics_are_safe_for_async_details():
    assert "mobilint" in benchmark_main._SAFE_RUNTIME_BACKENDS


def test_mobilint_llm_runtime_diagnostics_are_safe_for_async_details():
    assert "mobilint_llm" in benchmark_main._SAFE_RUNTIME_BACKENDS


def test_local_hf_model_generation_target_validates_directory_and_defaults_tokenizer(
    tmp_path,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    target = benchmark_main.resolve_target(
        "mobilint-aries-llm", "onnxruntime", "cpu"
    )
    args = Namespace(model_path=str(model_path), tokenizer_path=None)

    assert benchmark_main._is_local_hf_generation_target(target) is True
    assert benchmark_main._validate_local_hf_generation_cli(
        args,
        target,
        benchmark_main.Task.NLP_GENERATION,
    ) == model_path
    assert args.tokenizer_path == str(model_path)


@pytest.mark.parametrize(
    ("task", "model_kind", "message"),
    [
        (benchmark_main.Task.IMAGE_CLASSIFICATION, "dir", "NLP_GENERATION"),
        (benchmark_main.Task.NLP_GENERATION, "file", "local directory"),
        (benchmark_main.Task.NLP_GENERATION, "missing", "local directory"),
    ],
)
def test_local_hf_model_generation_target_rejects_invalid_cli(
    tmp_path, task, model_kind, message
):
    model_path = tmp_path / "model"
    if model_kind == "dir":
        model_path.mkdir()
    elif model_kind == "file":
        model_path.write_text("not a directory", encoding="utf-8")
    target = benchmark_main.resolve_target(
        "mobilint-aries-llm", "onnxruntime", "cpu"
    )
    args = Namespace(model_path=str(model_path), tokenizer_path=None)

    with pytest.raises(ValueError, match=message):
        benchmark_main._validate_local_hf_generation_cli(args, target, task)


def test_vllm_targets_keep_local_hf_model_generation_routing(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    target = benchmark_main.resolve_target("vllm-cpu", "onnxruntime", "cpu")
    args = Namespace(model_path=str(model_path), tokenizer_path=None)

    assert benchmark_main._is_local_hf_generation_target(target) is True
    assert benchmark_main._validate_local_hf_generation_cli(
        args,
        target,
        benchmark_main.Task.NLP_GENERATION,
    ) == model_path
    assert args.tokenizer_path == str(model_path)


def test_mobilint_aries_llm_main_routes_hf_model_and_tokenizer(
    monkeypatch, tmp_path
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_create_model_spec(model_name, artifact_path, **kwargs):
        captured["model_name"] = model_name
        captured["artifact_path"] = artifact_path
        captured.update(kwargs)
        return SimpleNamespace(task=benchmark_main.Task.NLP_GENERATION)

    class StopAfterLoader(RuntimeError):
        pass

    def fake_create_dataloader(**kwargs):
        captured["loader_kwargs"] = kwargs
        raise StopAfterLoader

    import utils.dataset_resolver as dataset_resolver

    monkeypatch.setattr(benchmark_main, "create_model_spec", fake_create_model_spec)
    monkeypatch.setattr(benchmark_main, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(
        benchmark_main,
        "_run_prepare_script",
        lambda script: (_ for _ in ()).throw(
            AssertionError(f"unexpected prepare script: {script}")
        ),
    )
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "llama-3.2-3b",
            "--target",
            "mobilint-aries-llm",
            "--model-path",
            str(model_path),
            "--dataset",
            str(dataset_path),
        ],
    )

    with pytest.raises(StopAfterLoader):
        benchmark_main.main()

    assert captured["artifact_path"] == str(model_path)
    assert captured["source_format"] == "hf_model"
    assert captured["sniff_onnx"] is False
    assert captured["loader_kwargs"]["tokenizer_path"] == str(model_path)


@pytest.mark.parametrize("target_id", ["mobilint-aries", "mobilint-regulus"])
@pytest.mark.parametrize(
    ("model_name", "task", "artifact_name", "expected_profile"),
    [
        (
            "resnet50",
            benchmark_main.Task.IMAGE_CLASSIFICATION,
            "resnet50_IMAGENET1K_V2.mxq",
            MOBILINT_RESNET50_IMAGENET1K_V2,
        ),
        (
            "yolov5m",
            benchmark_main.Task.OBJECT_DETECTION,
            "yolov5m.mxq",
            MOBILINT_YOLOV5M_DEFAULT,
        ),
    ],
)
def test_mobilint_vision_main_resolves_one_profile_for_spec_loader_and_decoder(
    monkeypatch,
    tmp_path,
    capsys,
    target_id,
    model_name,
    task,
    artifact_name,
    expected_profile,
):
    artifact_path = tmp_path / artifact_name
    artifact_path.touch()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    captured = {}
    source_spec = _model_spec(task)

    class StopAfterDecoder(RuntimeError):
        pass

    class FakeLoader:
        def __init__(self, profile):
            self.profile = profile

        def get_metadata(self):
            if self.profile is None:
                return {}
            return {"runtime_options": self.profile.runtime_contract()}

    class FakeRuntime:
        def load(self, compiled_model):
            captured["compiled_model"] = compiled_model

    def fake_create_model_spec(name, artifact, **kwargs):
        captured["spec_request"] = (name, artifact, kwargs)
        return source_spec

    def fake_create_dataloader(**kwargs):
        captured["loader_kwargs"] = kwargs
        return FakeLoader(kwargs.get("mobilint_vision_profile"))

    def fake_create_runtime(backend, **kwargs):
        captured["runtime_request"] = (backend, kwargs)
        return FakeRuntime()

    def fake_create_decoder(model_spec, **kwargs):
        captured["decoder_spec"] = model_spec
        captured["decoder_kwargs"] = kwargs
        raise StopAfterDecoder

    import utils.dataset_resolver as dataset_resolver

    monkeypatch.setattr(benchmark_main, "create_model_spec", fake_create_model_spec)
    monkeypatch.setattr(benchmark_main, "create_dataloader", fake_create_dataloader)
    monkeypatch.setattr(benchmark_main, "create_runtime", fake_create_runtime)
    monkeypatch.setattr(benchmark_main, "create_evaluator", lambda *args, **kwargs: object())
    monkeypatch.setattr(benchmark_main, "create_decoder", fake_create_decoder)
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            model_name,
            "--target",
            target_id,
            "--artifact",
            str(artifact_path),
            "--dataset",
            str(dataset_path),
        ],
    )

    with pytest.raises(StopAfterDecoder):
        benchmark_main.main()

    loader_profile = captured["loader_kwargs"]["mobilint_vision_profile"]
    decoder_profile = captured["decoder_kwargs"]["mobilint_vision_profile"]
    compiled_model = captured["compiled_model"]
    assert loader_profile is expected_profile
    assert decoder_profile is loader_profile
    assert compiled_model.spec is captured["loader_kwargs"]["model_spec"]
    assert captured["decoder_spec"] is compiled_model.spec
    assert compiled_model.spec is not source_spec
    assert compiled_model.artifact_path == artifact_path
    assert next(iter(compiled_model.spec.input_shapes.values())) == (
        1,
        *expected_profile.unbatched_input_shape,
    )
    assert next(iter(compiled_model.spec.input_dtype.values())) == "uint8"
    assert captured["loader_kwargs"]["layout"] == "NHWC"
    if expected_profile.expected_output_shapes:
        assert set(compiled_model.spec.output_shapes.values()) == {
            (1, *shape) for shape in expected_profile.expected_output_shapes
        }
    assert "| Layout: NHWC" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("artifact_name", "extra_args", "message"),
    [
        ("renamed.mxq", [], "No Mobilint vision profile matches"),
        (
            "resnet50_IMAGENET1K_V2.mxq",
            ["--layout", "NCHW"],
            "requires input layout",
        ),
        pytest.param(
            "resnet50_IMAGENET1K_V2.mxq",
            ["--lay", "NCHW"],
            "requires input layout",
            id="abbreviated-explicit-nchw",
        ),
        (
            "resnet50_IMAGENET1K_V2.mxq",
            ["--image-preprocess-mode", "normalized"],
            "requires preprocess mode",
        ),
    ],
)
def test_mobilint_vision_main_rejects_invalid_artifact_layout_and_mode(
    monkeypatch, tmp_path, artifact_name, extra_args, message
):
    artifact_path = tmp_path / artifact_name
    artifact_path.touch()
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()

    import utils.dataset_resolver as dataset_resolver

    monkeypatch.setattr(
        benchmark_main,
        "create_model_spec",
        lambda *args, **kwargs: _model_spec(
            benchmark_main.Task.IMAGE_CLASSIFICATION
        ),
    )
    monkeypatch.setattr(
        benchmark_main,
        "create_dataloader",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid profile must fail before loader creation")
        ),
    )
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "resnet50",
            "--target",
            "mobilint-aries",
            "--artifact",
            str(artifact_path),
            "--dataset",
            str(dataset_path),
            *extra_args,
        ],
    )

    with pytest.raises(ValueError, match=message):
        benchmark_main.main()


def test_mobilint_llm_stays_explicit_only():
    target = benchmark_main.resolve_target(None, "mobilint_llm", "0")
    backend_action = next(
        action
        for action in benchmark_main.build_parser()._actions
        if "--backend" in action.option_strings
    )

    assert target.target_id == "mobilint_llm:0"
    assert target.target_id != "mobilint-aries-llm"
    assert "mobilint_llm" not in backend_action.choices


@pytest.mark.parametrize(
    ("source", "override"),
    [
        ("loader runtime_options", {"device_id": 1}),
        ("loader runtime_options", {"device_id": 0.0}),
        ("loader runtime_options", {"device_id": "0"}),
        ("loader runtime_options", {"expected_family": "regulus"}),
        ("CLI --runtime-option", {"device_id": 1}),
        ("CLI --runtime-option", {"device_id": False}),
        ("CLI --runtime-option", {"expected_family": "regulus"}),
        ("CLI --runtime-option", {"expected_family": " ARIES"}),
    ],
)
def test_mobilint_runtime_option_merge_rejects_locked_target_mismatch(
    source, override
):
    target = benchmark_main.resolve_target(
        "mobilint-aries", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)
    locked_key = next(iter(override))

    with pytest.raises(
        ValueError,
        match=rf"{source}.*{locked_key}.*mobilint-aries",
    ):
        benchmark_main._merge_target_runtime_options(
            runtime_options,
            override,
            target=target,
            source=source,
        )

    assert runtime_options == target.runtime_options


@pytest.mark.parametrize(
    ("target_id", "family"),
    [
        ("mobilint-aries", "aries"),
        ("mobilint-regulus", "regulus"),
        ("mobilint-aries-llm", "aries"),
    ],
)
def test_mobilint_runtime_option_merge_accepts_canonical_matches(
    target_id, family
):
    target = benchmark_main.resolve_target(
        target_id, "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)

    benchmark_main._merge_target_runtime_options(
        runtime_options,
        {
            "device_id": 0,
            "expected_family": family.upper(),
            "loader_option": "kept",
        },
        target=target,
        source="loader runtime_options",
    )
    benchmark_main._merge_target_runtime_options(
        runtime_options,
        {
            "device_id": 0,
            "expected_family": family.swapcase(),
            "cli_option": "kept",
        },
        target=target,
        source="CLI --runtime-option",
    )

    assert runtime_options["device_id"] == 0
    assert type(runtime_options["device_id"]) is int
    assert runtime_options["expected_family"] == family
    assert (
        runtime_options["device_id"]
        == target.monitor_options["mobilint"]["device_id"]
    )
    assert (
        runtime_options["expected_family"]
        == target.monitor_options["mobilint"]["expected_family"]
    )
    assert runtime_options["loader_option"] == "kept"
    assert runtime_options["cli_option"] == "kept"


@pytest.mark.parametrize(
    ("source", "override"),
    [
        ("loader runtime_options", {"device_id": 1}),
        ("loader runtime_options", {"expected_family": "regulus"}),
        ("CLI --runtime-option", {"device_id": 1}),
        ("CLI --runtime-option", {"expected_family": "regulus"}),
    ],
)
def test_mobilint_llm_runtime_option_merge_rejects_locked_target_mismatch(
    source, override
):
    target = benchmark_main.resolve_target(
        "mobilint-aries-llm", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)
    locked_key = next(iter(override))

    with pytest.raises(
        ValueError,
        match=rf"{source}.*{locked_key}.*mobilint-aries-llm",
    ):
        benchmark_main._merge_target_runtime_options(
            runtime_options,
            override,
            target=target,
            source=source,
        )

    assert runtime_options == target.runtime_options


def test_mobilint_runtime_option_merge_allows_absent_locked_options():
    target = benchmark_main.resolve_target(
        "mobilint-regulus", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)

    benchmark_main._merge_target_runtime_options(
        runtime_options,
        {"unlocked_option": "kept"},
        target=target,
        source="loader runtime_options",
    )

    assert runtime_options == {
        **target.runtime_options,
        "unlocked_option": "kept",
    }


def test_runtime_option_merge_leaves_other_targets_unrestricted():
    target = benchmark_main.resolve_target("cpu", "onnxruntime", "cpu")
    runtime_options = {"existing": "value"}
    overrides = {
        "device_id": "vendor-defined",
        "expected_family": "vendor-defined",
    }

    benchmark_main._merge_target_runtime_options(
        runtime_options,
        overrides,
        target=target,
        source="CLI --runtime-option",
    )

    assert runtime_options == {"existing": "value", **overrides}


@pytest.mark.parametrize(
    ("source", "loader_options", "cli_options"),
    [
        ("loader runtime_options", {"device_id": 1}, {}),
        ("CLI --runtime-option", {}, {"device_id": 1}),
    ],
)
def test_runtime_option_layers_route_both_sources_through_target_guard(
    source, loader_options, cli_options
):
    target = benchmark_main.resolve_target(
        "mobilint-aries", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)

    with pytest.raises(
        ValueError,
        match=rf"{source}.*device_id.*mobilint-aries",
    ):
        benchmark_main._merge_runtime_option_layers(
            runtime_options,
            target=target,
            loader_runtime_options=loader_options,
            cli_runtime_options=cli_options,
            backend="mobilint",
            task_enum=benchmark_main.Task.IMAGE_CLASSIFICATION,
        )

    assert runtime_options == target.runtime_options


@pytest.mark.parametrize(
    "protected_key",
    [
        "vision_profile_id",
        "expected_input_dtype",
        "expected_input_layout",
        "expected_unbatched_input_shape",
        "max_input_batch_size",
        "expected_unbatched_output_shapes",
    ],
)
def test_mobilint_vision_runtime_option_layers_reject_cli_contract_overrides(
    protected_key,
):
    target = benchmark_main.resolve_target(
        "mobilint-aries", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)

    with pytest.raises(
        ValueError,
        match=rf"CLI --runtime-option.*{protected_key}.*Mobilint vision",
    ):
        benchmark_main._merge_runtime_option_layers(
            runtime_options,
            target=target,
            loader_runtime_options={
                "vision_profile_id": "mobilint-resnet50-imagenet1k-v2",
            },
            cli_runtime_options={protected_key: "override"},
            backend="mobilint",
            task_enum=benchmark_main.Task.IMAGE_CLASSIFICATION,
        )

    assert runtime_options == target.runtime_options


def test_mobilint_vision_runtime_options_allow_runtime_tuning_keys():
    target = benchmark_main.resolve_target(
        "mobilint-regulus", "onnxruntime", "cpu"
    )
    runtime_options = dict(target.runtime_options)
    tuning_options = {
        "core_mode": "single",
        "activation_slots": 2,
        "async_pipeline_enabled": True,
    }

    benchmark_main._merge_runtime_option_layers(
        runtime_options,
        target=target,
        loader_runtime_options=MOBILINT_YOLOV5M_DEFAULT.runtime_contract(),
        cli_runtime_options=tuning_options,
        backend="mobilint",
        task_enum=benchmark_main.Task.OBJECT_DETECTION,
    )

    assert runtime_options == {
        **target.runtime_options,
        **MOBILINT_YOLOV5M_DEFAULT.runtime_contract(),
        **tuning_options,
    }


@pytest.mark.parametrize(
    ("target_id", "backend", "task"),
    [
        ("cpu", "onnxruntime", benchmark_main.Task.IMAGE_CLASSIFICATION),
        ("mobilint-aries-llm", "mobilint_llm", benchmark_main.Task.NLP_GENERATION),
    ],
)
def test_contract_named_runtime_options_remain_unrestricted_outside_mobilint_vision(
    target_id, backend, task
):
    target = benchmark_main.resolve_target(target_id, "onnxruntime", "cpu")
    runtime_options = dict(target.runtime_options)

    benchmark_main._merge_runtime_option_layers(
        runtime_options,
        target=target,
        loader_runtime_options={},
        cli_runtime_options={"vision_profile_id": "vendor-defined"},
        backend=backend,
        task_enum=task,
    )

    assert runtime_options["vision_profile_id"] == "vendor-defined"


def test_validate_furiosa_cli_accepts_artifact_fallback_and_defaults_tokenizer(
    tmp_path,
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    fxb_path = tmp_path / "model.fxb"
    fxb_path.write_bytes(b"fxb")
    args = Namespace(
        model_path=str(model_path),
        fxb=None,
        artifact=str(fxb_path),
        tokenizer_path=None,
    )

    benchmark_main._validate_furiosa_cli(args, benchmark_main.Task.NLP_GENERATION)

    assert args.fxb == str(fxb_path)
    assert args.artifact == str(fxb_path)
    assert args.tokenizer_path == str(model_path)


def test_validate_furiosa_cli_accepts_hub_model_without_fxb():
    args = Namespace(
        model_path="furiosa-ai/Llama-3.1-8B-Instruct",
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )

    benchmark_main._validate_furiosa_cli(
        args, benchmark_main.Task.NLP_GENERATION
    )

    assert args.model_path == "furiosa-ai/Llama-3.1-8B-Instruct"
    assert args.fxb is None
    assert args.artifact is None
    assert args.tokenizer_path == args.model_path


def test_validate_furiosa_cli_accepts_local_model_without_fxb(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    args = Namespace(
        model_path=str(model_path),
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )

    benchmark_main._validate_furiosa_cli(
        args, benchmark_main.Task.NLP_GENERATION
    )

    assert args.fxb is None
    assert args.artifact is None
    assert args.tokenizer_path == str(model_path)


@pytest.mark.parametrize("model_path", [None, "", "   "])
def test_validate_furiosa_cli_rejects_empty_model_reference(model_path):
    args = Namespace(
        model_path=model_path,
        fxb=None,
        artifact=None,
        tokenizer_path=None,
    )

    with pytest.raises(
        ValueError,
        match="repository ID or local model directory",
    ):
        benchmark_main._validate_furiosa_cli(
            args, benchmark_main.Task.NLP_GENERATION
        )


@pytest.mark.parametrize(
    ("task", "model_kind", "fxb_name", "message"),
    [
        (benchmark_main.Task.IMAGE_CLASSIFICATION, "dir", "model.fxb", "NLP_GENERATION"),
        (benchmark_main.Task.NLP_GENERATION, "file", "model.fxb", "directory"),
        (benchmark_main.Task.NLP_GENERATION, "dir", "model.bin", ".fxb"),
    ],
)
def test_validate_furiosa_cli_rejects_invalid_inputs(
    tmp_path, task, model_kind, fxb_name, message
):
    model_path = tmp_path / "model"
    if model_kind == "dir":
        model_path.mkdir()
    else:
        model_path.write_text("not a directory", encoding="utf-8")
    fxb_path = tmp_path / fxb_name
    fxb_path.write_bytes(b"artifact")
    args = Namespace(
        model_path=str(model_path),
        fxb=str(fxb_path),
        artifact=None,
        tokenizer_path=None,
    )

    with pytest.raises(ValueError, match=message):
        benchmark_main._validate_furiosa_cli(args, task)


def test_validate_furiosa_runtime_options_rejects_unknown_option():
    supported = {
        "devices": "npu:0",
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_io_memory_mb": 4096,
        "seed": 3,
        "cache_dir": "/tmp/cache",
        "npu_queue_limit": 2,
        "max_processing_samples": 32,
        "spare_blocks_ratio": 0.1,
    }
    benchmark_main._validate_furiosa_runtime_options(supported)

    with pytest.raises(ValueError, match="unsupported_option"):
        benchmark_main._validate_furiosa_runtime_options(
            {**supported, "unsupported_option": True}
        )

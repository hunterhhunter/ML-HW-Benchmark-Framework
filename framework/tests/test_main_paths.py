import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main


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


def test_mobilint_runtime_diagnostics_are_safe_for_async_details():
    assert "mobilint" in benchmark_main._SAFE_RUNTIME_BACKENDS


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

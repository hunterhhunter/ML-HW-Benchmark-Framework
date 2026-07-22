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


def test_parser_accepts_rbln_backend():
    args = benchmark_main.build_parser().parse_args(
        ["--model", "resnet50", "--backend", "rbln"]
    )

    assert args.backend == "rbln"


@pytest.mark.parametrize("backend", ["rebel", "rbln-static"])
def test_parser_rejects_runtime_aliases_that_bypass_static_target_contract(
    backend,
):
    with pytest.raises(SystemExit) as raised:
        benchmark_main.build_parser().parse_args(
            ["--model", "resnet50", "--backend", backend]
        )

    assert raised.value.code == 2


def test_parser_target_help_mentions_rbln_static():
    parser = benchmark_main.build_parser()
    target_action = next(
        action
        for action in parser._actions
        if "--target" in action.option_strings
    )

    assert "rbln-static" in target_action.help


def _rbln_target():
    return benchmark_main.resolve_target(
        "rbln-static", "onnxruntime", "cpu"
    )


def test_rbln_static_requires_precompiled_artifact():
    with pytest.raises(ValueError, match="--artifact.*rbln-static"):
        benchmark_main._validate_precompiled_artifact(
            _rbln_target(), None
        )


@pytest.mark.parametrize(
    "artifact_kind",
    [
        "missing",
        "directory",
        "symlink-to-directory",
        "onnx-file",
        "huggingface-directory",
    ],
)
def test_rbln_static_rejects_non_rbln_regular_files(
    tmp_path, artifact_kind
):
    if artifact_kind == "missing":
        artifact = tmp_path / "missing.rbln"
    elif artifact_kind == "directory":
        artifact = tmp_path / "compiled.rbln"
        artifact.mkdir()
    elif artifact_kind == "symlink-to-directory":
        directory = tmp_path / "compiled-directory"
        directory.mkdir()
        artifact = tmp_path / "compiled.rbln"
        artifact.symlink_to(directory, target_is_directory=True)
    elif artifact_kind == "onnx-file":
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"onnx")
    else:
        artifact = tmp_path / "huggingface-model"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match=r"rbln-static.*\.rbln"):
        benchmark_main._validate_precompiled_artifact(
            _rbln_target(), str(artifact)
        )


def test_rbln_static_accepts_resolved_case_insensitive_artifact(tmp_path):
    artifact = tmp_path / "MODEL.RBLN"
    artifact.write_bytes(b"compiled")

    resolved = benchmark_main._validate_precompiled_artifact(
        _rbln_target(), str(artifact)
    )

    assert resolved == artifact.resolve()
    assert resolved.is_file()


@pytest.mark.parametrize(
    "task",
    [
        benchmark_main.Task.IMAGE_CLASSIFICATION,
        benchmark_main.Task.OBJECT_DETECTION,
        benchmark_main.Task.NLP_CLASSIFICATION,
        benchmark_main.Task.QUESTION_ANSWERING,
        benchmark_main.Task.TIME_SERIES_FORECASTING,
    ],
)
def test_rbln_static_accepts_supported_static_tasks(task):
    benchmark_main._validate_target_task(
        _rbln_target(), task, batch_size=1
    )


def test_rbln_static_rejects_generation_before_runtime_creation():
    with pytest.raises(ValueError, match="rbln-vllm"):
        benchmark_main._validate_target_task(
            _rbln_target(),
            benchmark_main.Task.NLP_GENERATION,
            batch_size=1,
        )


@pytest.mark.parametrize(
    "batch_size",
    [0, 2, -1, True, False, 1.0],
)
def test_rbln_static_requires_exact_builtin_batch_size_one(batch_size):
    with pytest.raises(ValueError, match="batch size.*1"):
        benchmark_main._validate_target_task(
            _rbln_target(),
            benchmark_main.Task.IMAGE_CLASSIFICATION,
            batch_size=batch_size,
        )


@pytest.mark.parametrize(
    ("artifact_kind", "extra_args"),
    [
        ("directory", []),
        ("symlink-to-directory", []),
        ("onnx-file", []),
        ("huggingface-directory", []),
        ("missing", []),
        ("valid", ["--batch-size", "2"]),
        (
            "valid",
            ["--inference-mode", "async_queue", "--batch-size", "2"],
        ),
    ],
)
def test_rbln_main_rejects_invalid_artifact_or_batch_before_preparation(
    monkeypatch, tmp_path, artifact_kind, extra_args
):
    if artifact_kind == "directory":
        artifact = tmp_path / "model.rbln"
        artifact.mkdir()
    elif artifact_kind == "symlink-to-directory":
        directory = tmp_path / "compiled-directory"
        directory.mkdir()
        artifact = tmp_path / "model.rbln"
        artifact.symlink_to(directory, target_is_directory=True)
    elif artifact_kind == "onnx-file":
        artifact = tmp_path / "model.onnx"
        artifact.write_bytes(b"onnx")
    elif artifact_kind == "huggingface-directory":
        artifact = tmp_path / "hf-model"
        artifact.mkdir()
        (artifact / "config.json").write_text("{}", encoding="utf-8")
    elif artifact_kind == "missing":
        artifact = tmp_path / "missing.rbln"
    else:
        artifact = tmp_path / "model.rbln"
        artifact.write_bytes(b"compiled")

    forbidden_calls = []

    def forbidden(stage):
        def fail(*args, **kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"unexpected early call: {stage}")

        return fail

    monkeypatch.setattr(
        benchmark_main, "run_auto_prepare", forbidden("auto_prepare")
    )
    monkeypatch.setattr(
        benchmark_main, "create_model_spec", forbidden("model_spec")
    )
    monkeypatch.setattr(
        benchmark_main, "get_compiler", forbidden("compiler")
    )
    monkeypatch.setattr(
        benchmark_main, "create_runtime", forbidden("runtime")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "resnet50",
            "--target",
            "rbln-static",
            "--artifact",
            str(artifact),
            *extra_args,
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 1
    assert forbidden_calls == []


def test_rbln_generation_main_rejects_before_preparation_or_runtime(
    monkeypatch, tmp_path, capsys
):
    artifact = tmp_path / "llama.rbln"
    artifact.write_bytes(b"compiled")
    forbidden_calls = []

    def forbidden(stage):
        def fail(*args, **kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"unexpected early call: {stage}")

        return fail

    monkeypatch.setattr(
        benchmark_main, "run_auto_prepare", forbidden("auto_prepare")
    )
    monkeypatch.setattr(
        benchmark_main, "create_model_spec", forbidden("model_spec")
    )
    monkeypatch.setattr(
        benchmark_main, "get_compiler", forbidden("compiler")
    )
    monkeypatch.setattr(
        benchmark_main, "create_runtime", forbidden("runtime")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "llama-3.2-3b",
            "--target",
            "rbln-static",
            "--artifact",
            str(artifact),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 1
    assert forbidden_calls == []
    assert "rbln-vllm" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("model", "artifact_name", "extra_args", "expected_error"),
    [
        ("resnet50", None, [], ".rbln"),
        ("llama-3.2-3b", "model.rbln", [], "rbln-vllm"),
        (
            "resnet50",
            "model.rbln",
            ["--batch-size", "2"],
            "batch size",
        ),
    ],
)
def test_rbln_backend_device_zero_uses_static_target_preflight(
    monkeypatch,
    tmp_path,
    capsys,
    model,
    artifact_name,
    extra_args,
    expected_error,
):
    artifact = None
    if artifact_name is not None:
        artifact = tmp_path / artifact_name
        artifact.write_bytes(b"compiled")
    real_resolve_target = benchmark_main.resolve_target
    resolutions = []
    forbidden_calls = []

    def resolve(target_id, backend, device):
        target = real_resolve_target(target_id, backend, device)
        resolutions.append(
            (target_id, backend, device, target.target_id)
        )
        return target

    def forbidden(stage):
        def fail(*args, **kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"unexpected early call: {stage}")

        return fail

    monkeypatch.setattr(benchmark_main, "resolve_target", resolve)
    monkeypatch.setattr(
        benchmark_main, "run_auto_prepare", forbidden("auto_prepare")
    )
    monkeypatch.setattr(
        benchmark_main, "create_model_spec", forbidden("model_spec")
    )
    monkeypatch.setattr(
        benchmark_main, "get_compiler", forbidden("compiler")
    )
    monkeypatch.setattr(
        benchmark_main, "create_runtime", forbidden("runtime")
    )
    argv = [
        "main.py",
        "--model",
        model,
        "--backend",
        "rbln",
        "--device",
        "0",
    ]
    if artifact is not None:
        argv.extend(["--artifact", str(artifact)])
    argv.extend(extra_args)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 1
    assert resolutions == [(None, "rbln", "0", "rbln-static")]
    assert forbidden_calls == []
    assert expected_error in capsys.readouterr().out


def test_rbln_valid_main_runs_auto_prepare_without_model_script_or_compiler(
    monkeypatch, tmp_path
):
    import utils.dataset_resolver as dataset_resolver_module

    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    real_run_auto_prepare = benchmark_main.run_auto_prepare
    auto_prepare_calls = []
    forbidden_calls = []
    spec_calls = []
    runtime_calls = []
    fake_spec = object()
    fake_loader = SimpleNamespace(get_metadata=lambda: {})

    def forbidden(stage):
        def fail(*args, **kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"unexpected call: {stage}")

        return fail

    def create_model_spec(*args, **kwargs):
        spec_calls.append((args, kwargs))
        return fake_spec

    def observe_auto_prepare(profile, args, target):
        auto_prepare_calls.append(
            (
                target.target_id,
                args.artifact,
                args.dataset,
                profile["prepare_model_script"],
            )
        )
        return real_run_auto_prepare(profile, args, target)

    def stop_before_runtime_load(backend, **kwargs):
        runtime_calls.append((backend, kwargs))
        raise RuntimeError("controlled stop before SDK runtime creation")

    monkeypatch.setattr(
        benchmark_main, "run_auto_prepare", observe_auto_prepare
    )
    monkeypatch.setattr(
        benchmark_main,
        "_run_prepare_script",
        forbidden("model_prepare_script"),
    )
    monkeypatch.setattr(
        benchmark_main, "get_compiler", forbidden("compiler")
    )
    monkeypatch.setattr(
        benchmark_main, "create_model_spec", create_model_spec
    )
    monkeypatch.setattr(
        benchmark_main,
        "create_dataloader",
        lambda **kwargs: fake_loader,
    )
    monkeypatch.setattr(
        dataset_resolver_module,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setattr(
        benchmark_main, "create_runtime", stop_before_runtime_load
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "resnet50",
            "--target",
            "rbln-static",
            "--artifact",
            str(artifact),
            "--dataset",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 1
    assert auto_prepare_calls == [
        (
            "rbln-static",
            str(artifact.resolve()),
            str(tmp_path),
            "models/prepare_resnet50_kalray.py",
        )
    ]
    assert forbidden_calls == []
    assert spec_calls == [
        (
            ("resnet50", str(artifact.resolve())),
            {
                "task": benchmark_main.Task.IMAGE_CLASSIFICATION,
                "sniff_onnx": False,
                "source_format": "rbln",
            },
        )
    ]
    assert runtime_calls == [
        (
            "rbln",
            {
                "device": "0",
                **_rbln_target().runtime_options,
            },
        )
    ]


def test_rbln_auto_prepare_never_runs_model_prepare_script(
    monkeypatch, tmp_path
):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    calls = []
    args = Namespace(
        backend="rbln",
        hef=None,
        artifact=str(artifact),
        compile=True,
        onnx=str(tmp_path / "missing.onnx"),
        model_path=str(tmp_path / "missing-hf-model"),
        dataset=str(tmp_path),
    )
    profile = {
        "prepare_model_script": "models/download-or-compile.py",
        "prepare_dataset_script": "datasets/prepare.py",
    }
    monkeypatch.setattr(
        benchmark_main,
        "_run_prepare_script",
        lambda script: calls.append(script),
    )

    benchmark_main.run_auto_prepare(profile, args, _rbln_target())

    assert calls == []


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


@pytest.mark.parametrize(
    ("source", "override"),
    [
        ("loader runtime_options", {"device_id": 1}),
        ("loader runtime_options", {"device_id": True}),
        ("CLI --runtime-option", {"device_id": 1}),
        ("CLI --runtime-option", {"device_id": True}),
    ],
)
def test_rbln_runtime_option_merge_rejects_locked_target_mismatch(
    source, override
):
    target = _rbln_target()
    runtime_options = dict(target.runtime_options)

    with pytest.raises(
        ValueError,
        match=rf"{source}.*device_id.*rbln-static",
    ):
        benchmark_main._merge_target_runtime_options(
            runtime_options,
            override,
            target=target,
            source=source,
        )

    assert runtime_options == target.runtime_options


@pytest.mark.parametrize(
    "source", ["loader runtime_options", "CLI --runtime-option"]
)
def test_rbln_runtime_option_merge_accepts_and_normalizes_exact_device_zero(
    source,
):
    target = _rbln_target()
    runtime_options = dict(target.runtime_options)

    benchmark_main._merge_target_runtime_options(
        runtime_options,
        {"device_id": 0, "unlocked_option": "kept"},
        target=target,
        source=source,
    )

    assert runtime_options["device_id"] == 0
    assert type(runtime_options["device_id"]) is int
    assert runtime_options["unlocked_option"] == "kept"


@pytest.mark.parametrize("device_id", [1, True])
def test_rbln_cli_locked_device_override_fails_before_preparation(
    monkeypatch, tmp_path, device_id
):
    artifact = tmp_path / "model.rbln"
    artifact.write_bytes(b"compiled")
    forbidden_calls = []

    def forbidden(stage):
        def fail(*args, **kwargs):
            forbidden_calls.append(stage)
            raise AssertionError(f"unexpected early call: {stage}")

        return fail

    monkeypatch.setattr(
        benchmark_main, "run_auto_prepare", forbidden("auto_prepare")
    )
    monkeypatch.setattr(
        benchmark_main, "create_model_spec", forbidden("model_spec")
    )
    monkeypatch.setattr(
        benchmark_main, "create_runtime", forbidden("runtime")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--model",
            "resnet50",
            "--target",
            "rbln-static",
            "--artifact",
            str(artifact),
            "--runtime-option",
            f"device_id={device_id}",
        ],
    )

    with pytest.raises(SystemExit) as raised:
        benchmark_main.main()

    assert raised.value.code == 1
    assert forbidden_calls == []


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

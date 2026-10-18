"""웹 요청이 프레임워크 CLI의 실제 인자를 보존하는지 확인한다."""

import sys
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "framework" / "src"))

from app.schemas.benchmark import BenchmarkRunRequest
from main import build_parser


def build_cli_args(request):
    return importlib.import_module("app.services.command_builder").build_cli_args(request)


def test_omitted_values_keep_cli_defaults_and_target_has_priority():
    request = BenchmarkRunRequest(model="yolov8s-pose", target_id="hailo8")

    assert build_cli_args(request) == ["--model", "yolov8s-pose", "--target", "hailo8"]


def test_explicit_layout_and_async_options_are_forwarded_without_shell_joining():
    request = BenchmarkRunRequest(
        model="resnet50",
        target_id="cpu",
        layout="NHWC",
        inference_mode="async_queue",
        scenario="server_like",
        async_pass="runtime-only",
        target_qps=7.5,
        worker_count=2,
        max_samples=25,
        save_request_trace=True,
        runtime_options={"output_format_type": "uint8"},
        results_path="/tmp/results with spaces.csv",
        monitor=True,
    )

    assert build_cli_args(request) == [
        "--model", "resnet50", "--target", "cpu", "--layout", "NHWC",
        "--results-path", "/tmp/results with spaces.csv", "--monitor",
        "--runtime-option", "output_format_type=uint8",
        "--inference-mode", "async_queue", "--scenario", "server_like",
        "--async-pass", "runtime-only", "--target-qps", "7.5",
        "--worker-count", "2", "--max-samples", "25",
        "--save-request-trace",
    ]


def test_explicit_disable_compile_and_legacy_artifact_names():
    request = BenchmarkRunRequest(
        model="resnet50", backend="onnxruntime", device="cpu",
        onnx_path="/models/my model.onnx", artifact_path="/models/compiled.mxq",
        compile=False, compile_options={"opt_level": "2"},
    )

    assert build_cli_args(request) == [
        "--model", "resnet50", "--backend", "onnxruntime", "--device", "cpu",
        "--onnx", "/models/my model.onnx", "--artifact", "/models/compiled.mxq",
        "--no-compile", "--compile-option", "opt_level=2",
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"model": "resnet50", "target_id": "cpu", "worker_count": 2},
        {"model": "resnet50", "target_id": "cpu", "inference_mode": "async_queue", "max_steps": 2},
    ],
)
def test_mode_specific_fields_are_rejected(values):
    with pytest.raises(ValueError):
        BenchmarkRunRequest(**values)


def test_all_current_cli_flags_have_a_web_mapping():
    supported = importlib.import_module("app.services.command_builder").SUPPORTED_CLI_FLAGS
    parser_flags = {
        flag
        for action in build_parser()._actions
        for flag in action.option_strings
        if flag.startswith("--") and flag != "--help"
    }
    assert set(supported) == parser_flags

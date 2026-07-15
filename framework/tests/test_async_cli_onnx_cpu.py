import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from PIL import Image


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = FRAMEWORK_ROOT / "results"


def _create_resnet_like_model(
    path: Path,
    *,
    input_shape: list[str | int],
) -> None:
    batch_dimension = input_shape[0]
    input_info = helper.make_tensor_value_info(
        "input",
        TensorProto.FLOAT,
        input_shape,
    )
    output_info = helper.make_tensor_value_info(
        "output",
        TensorProto.FLOAT,
        [batch_dimension, 1000],
    )
    weights = np.zeros((3, 1000), dtype=np.float32)
    bias = np.zeros(1000, dtype=np.float32)
    bias[0] = 10.0
    graph = helper.make_graph(
        [
            helper.make_node(
                "GlobalAveragePool",
                inputs=["input"],
                outputs=["pooled"],
            ),
            helper.make_node(
                "Flatten",
                inputs=["pooled"],
                outputs=["features"],
                axis=1,
            ),
            helper.make_node(
                "Gemm",
                inputs=["features", "weights", "bias"],
                outputs=["output"],
            ),
        ],
        "resnet-like-cli-acceptance",
        [input_info],
        [output_info],
        initializer=[
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def _create_image_dataset(root: Path) -> Path:
    dataset = root / "dataset"
    image_dir = dataset / "val"
    image_dir.mkdir(parents=True)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (127, 127, 127),
    ]
    label_lines = []
    for index, color in enumerate(colors):
        filename = f"sample_{index}.png"
        Image.new("RGB", (256, 256), color).save(image_dir / filename)
        label_lines.append(f"{filename} 0")
    (dataset / "val_labels.txt").write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8",
    )
    return dataset


def _snapshot_default_results() -> dict[str, tuple[int, int]]:
    if not DEFAULT_RESULTS_ROOT.exists():
        return {}
    return {
        str(path.relative_to(DEFAULT_RESULTS_ROOT)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in DEFAULT_RESULTS_ROOT.rglob("*")
        if path.is_file()
    }


def _run_async_cli(
    tmp_path: Path,
    model_path: Path,
    dataset_path: Path,
    *,
    warmup: int,
    batch_size: int,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    results_path = tmp_path / "results" / "benchmark_results.csv"
    command = [
        sys.executable,
        "src/main.py",
        "--model",
        "resnet50",
        "--onnx",
        str(model_path),
        "--dataset",
        str(dataset_path),
        "--target",
        "cpu",
        "--inference-mode",
        "async_queue",
        "--scenario",
        "offline",
        "--max-samples",
        "4",
        "--min-samples",
        "4",
        "--batch-size",
        str(batch_size),
        "--queue-capacity",
        "4",
        "--worker-count",
        "1",
        "--batch-timeout-ms",
        "20",
        "--warmup",
        str(warmup),
        "--save-request-trace",
        "--debug",
        "--results-path",
        str(results_path),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "HF_DATASETS_CACHE": str(tmp_path / "hf-cache"),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    default_results_before = _snapshot_default_results()
    completed = subprocess.run(
        command,
        cwd=FRAMEWORK_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert _snapshot_default_results() == default_results_before
    return completed, results_path


def _run_ids(completed: subprocess.CompletedProcess[str]) -> tuple[list[str], list[str]]:
    reserved = re.findall(
        r"^RUN_ID_RESERVED=(\w+)$",
        completed.stdout,
        re.MULTILINE,
    )
    finished = re.findall(
        r"^RUN_ID=(\w+)$",
        completed.stdout,
        re.MULTILINE,
    )
    return reserved, finished


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_async_cli_runs_dynamic_onnx_on_actual_cpu(tmp_path):
    model_path = tmp_path / "dynamic-resnet-like.onnx"
    _create_resnet_like_model(
        model_path,
        input_shape=["batch", 3, 224, 224],
    )
    dataset_path = _create_image_dataset(tmp_path)

    completed, results_path = _run_async_cli(
        tmp_path,
        model_path,
        dataset_path,
        warmup=0,
        batch_size=2,
    )

    assert completed.returncode == 0, completed.stderr
    reserved, finished = _run_ids(completed)
    assert len(reserved) == len(finished) == 1
    assert reserved == finished

    rows = _read_csv(results_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == finished[0]
    assert rows[0]["async_run_status"] == "valid"

    details_path = results_path.parent / "details" / f"{finished[0]}.json"
    trace_path = results_path.parent / "traces" / f"{finished[0]}.jsonl"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    trace_rows = _read_jsonl(trace_path)

    assert details["counts"]["submitted"] == 4
    assert details["counts"]["accepted"] == 4
    assert details["counts"]["completed"] == 4
    assert details["counts"]["outstanding"] == 0
    assert details["batch_size"]["max"] == 2.0
    assert details["run"]["runtime_device_spec"]["active_providers"] == [
        "CPUExecutionProvider"
    ]
    assert len(trace_rows) == 4
    assert {row["status"] for row in trace_rows} == {"completed"}


def test_async_cli_persists_redacted_warmup_shape_failure(tmp_path):
    model_path = tmp_path / "fixed-spatial-resnet-like.onnx"
    _create_resnet_like_model(
        model_path,
        input_shape=[1, 3, 1, 1],
    )
    dataset_path = _create_image_dataset(tmp_path)

    completed, results_path = _run_async_cli(
        tmp_path,
        model_path,
        dataset_path,
        warmup=1,
        batch_size=1,
    )

    assert completed.returncode != 0
    reserved, finished = _run_ids(completed)
    assert len(reserved) == len(finished) == 1
    assert reserved == finished
    assert "Traceback (most recent call last)" in completed.stderr
    assert "Got invalid dimensions for input" in completed.stderr

    rows = _read_csv(results_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == finished[0]
    assert rows[0]["async_run_status"] == "invalid"
    assert rows[0]["async_invalid_reasons"] == "benchmark_exception"

    details_path = results_path.parent / "details" / f"{finished[0]}.json"
    details = json.loads(details_path.read_text(encoding="utf-8"))
    failure = details["failure"]
    assert details["run"]["measurement_started"] is False
    assert failure["phase"] == "warmup"
    assert failure["error_type"] == "InvalidArgument"
    assert failure["error_message"] == (
        "benchmark failed during warmup (InvalidArgument)"
    )

    serialized_details = json.dumps(details).lower()
    for forbidden in ("traceback", "input_tensor", "output_tensor", "prompt"):
        assert forbidden not in serialized_details

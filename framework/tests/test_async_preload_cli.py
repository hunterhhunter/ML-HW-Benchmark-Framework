import sys
from types import SimpleNamespace

import numpy as np
import pytest

import main as benchmark_main


def test_async_preload_is_opt_in_and_e2e_rejects_it():
    parser = benchmark_main.build_parser()
    assert parser.parse_args(["--model", "resnet50"]).async_preload_inputs is False
    args = parser.parse_args(["--model", "resnet50", "--async-preload-inputs"])
    with pytest.raises(ValueError, match="async_queue"):
        benchmark_main.validate_async_args(args)


def _run_main(monkeypatch, tmp_path, *, dynamic=False, load_failure=False):
    import utils.dataset_resolver as dataset_resolver
    artifact = tmp_path / "model.onnx"; artifact.touch()
    dataset = tmp_path / "dataset"; dataset.mkdir()
    events = []

    class Loader:
        total_samples = 3
        def get_metadata(self):
            return {"total_samples": 3}
        def load_by_index(self, index):
            events.append(("preload", index))
            value = {"x": np.ones(1)} if dynamic else np.full((2, 2, 3), index, dtype=np.uint8)
            return {"input": value, "label": index, "preprocess_context": {"pad_y": index}}
        def get_labels(self):
            return [0, 1, 2]
        def preprocess(self, raw):
            return raw

    class Runtime:
        def load(self, model):
            events.append(("load", None))
            if load_failure:
                raise RuntimeError("hardware load failed")

    monkeypatch.setattr(dataset_resolver, "resolve_dataset_paths", lambda *a: (None, None))
    monkeypatch.setattr(benchmark_main, "create_model_spec", lambda *a, **k: SimpleNamespace(
        name="resnet50", task=benchmark_main.Task.IMAGE_CLASSIFICATION,
        input_shapes={"images": (1, 3, 2, 2)}, input_dtype={"images": "uint8"},
        output_shapes={"output": (1, 1000)}))
    monkeypatch.setattr(benchmark_main, "create_dataloader", lambda **k: Loader())
    monkeypatch.setattr(benchmark_main, "create_runtime", lambda *a, **k: Runtime())
    monkeypatch.setattr(benchmark_main, "execute_benchmark", lambda args, **k: events.append(("measure", k["loader"])) or 0)
    monkeypatch.setattr(sys, "argv", ["main.py", "--model", "resnet50", "--target", "cpu",
        "--onnx", str(artifact), "--dataset", str(dataset), "--inference-mode", "async_queue",
        "--async-pass", "runtime-only", "--async-preload-inputs", "--max-samples", "3",
        "--min-samples", "3", "--results-path", str(tmp_path / "results.csv")])
    return events, benchmark_main.main()


def test_async_preload_finishes_before_runtime_load_and_measurement(monkeypatch, tmp_path):
    events, status = _run_main(monkeypatch, tmp_path)
    assert status == 0
    assert [event[0] for event in events] == ["preload", "preload", "preload", "load", "measure"]
    assert [event[1] for event in events[:3]] == [0, 1, 2]


def test_async_preload_sidecar_records_tensor_contract(monkeypatch, tmp_path):
    events, _ = _run_main(monkeypatch, tmp_path)
    args = benchmark_main.build_parser().parse_args(["--model", "resnet50", "--inference-mode",
        "async_queue", "--async-preload-inputs", "--max-samples", "3", "--min-samples", "3"])
    args._async_preload_metadata = events[-1][1].get_metadata()
    run = benchmark_main._async_run_metadata(args, "IMAGE_CLASSIFICATION", {}, {})
    assert run["async_preload_inputs"] is True
    assert run["async_preload_sample_count"] == 3
    assert run["async_preload_input_shape"] == [2, 2, 3]
    assert run["async_preload_input_dtype"] == "uint8"


def test_async_preload_rejects_dynamic_inputs_before_runtime_load(monkeypatch, tmp_path):
    with pytest.raises(TypeError):
        events, _ = _run_main(monkeypatch, tmp_path, dynamic=True)


def test_async_preload_cleans_mmap_when_runtime_load_fails(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="hardware load failed"):
        _run_main(monkeypatch, tmp_path, load_failure=True)
    assert list((tmp_path / ".preloaded-inputs").glob("*.npy")) == []

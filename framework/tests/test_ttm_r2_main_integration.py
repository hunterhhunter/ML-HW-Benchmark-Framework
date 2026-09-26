from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
from core.model_profiles import create_model_spec
from core.targets import get_target
from runtimes.mobilint_ttm_r2 import MOBILINT_TTM_R2_ADAPTER_ID


@pytest.fixture
def ttm_resources(tmp_path, monkeypatch):
    checkpoint = tmp_path / "ibm-granite_granite-timeseries-ttm-r2"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_bytes(b"config")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    dataset = tmp_path / "ETTh1.csv"
    dataset.write_bytes(b"dataset")
    rbln = tmp_path / "ttm-r2-core.rbln"
    rbln.write_bytes(b"rbln")
    mxq = tmp_path / "ttm-r2-core.mxq"
    mxq.write_bytes(b"mxq")
    monkeypatch.setattr(
        benchmark_main,
        "validate_checkpoint",
        lambda path: {
            "config_sha256": "config-sha",
            "model_sha256": "model-sha",
        },
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_main,
        "validate_dataset",
        lambda path: "dataset-sha",
        raising=False,
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        dataset=dataset,
        rbln=rbln,
        mxq=mxq,
    )


def _args(resources, **overrides):
    values = {
        "model": "ttm-r2",
        "batch_size": 1,
        "inference_mode": "e2e",
        "dataset": str(resources.dataset),
        "model_path": str(resources.checkpoint),
        "artifact": None,
        "max_steps": None,
        "max_samples": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("target_id", "artifact_name"),
    [
        ("furiosa-rngd-torch", "checkpoint"),
        ("rbln-static", "rbln"),
        ("mobilint-aries", "mxq"),
    ],
)
def test_ttm_r2_accepts_only_three_verified_targets(
    ttm_resources, target_id, artifact_name
):
    benchmark_main.validate_ttm_r2_execution(
        _args(ttm_resources),
        get_target(target_id),
        getattr(ttm_resources, artifact_name),
    )


def test_ttm_r2_rejects_non_unit_batch_before_runtime(ttm_resources):
    with pytest.raises(ValueError, match="batch size exactly 1"):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources, batch_size=2),
            get_target("mobilint-aries"),
            ttm_resources.mxq,
        )


@pytest.mark.parametrize(
    ("target_id", "artifact_name"),
    [
        ("furiosa-rngd-torch", "checkpoint"),
        ("rbln-static", "rbln"),
        ("mobilint-aries", "mxq"),
    ],
)
def test_ttm_r2_accepts_async_queue_for_verified_targets(
    ttm_resources, target_id, artifact_name
):
    benchmark_main.validate_ttm_r2_execution(
        _args(ttm_resources, inference_mode="async_queue"),
        get_target(target_id),
        getattr(ttm_resources, artifact_name),
    )


@pytest.mark.parametrize(
    ("target_id", "artifact_name", "replacement", "message"),
    [
        ("rbln-static", "rbln", "wrong.mxq", r"\.rbln"),
        ("mobilint-aries", "mxq", "wrong.rbln", r"\.mxq"),
        ("rbln-static", "rbln", "missing.rbln", "regular"),
    ],
)
def test_ttm_r2_rejects_invalid_precompiled_artifact(
    ttm_resources, target_id, artifact_name, replacement, message
):
    del artifact_name
    artifact = ttm_resources.dataset.parent / replacement
    if not replacement.startswith("missing"):
        artifact.write_bytes(b"artifact")

    with pytest.raises(ValueError, match=message):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources), get_target(target_id), artifact
        )


def test_ttm_r2_rejects_missing_artifact_argument(ttm_resources):
    with pytest.raises(ValueError, match=r"regular.*\.mxq"):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources), get_target("mobilint-aries"), None
        )


def test_ttm_r2_rejects_wrong_dataset_hash_before_runtime(
    ttm_resources, monkeypatch
):
    monkeypatch.setattr(
        benchmark_main,
        "validate_dataset",
        lambda path: (_ for _ in ()).throw(ValueError("dataset SHA-256 mismatch")),
    )

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources), get_target("rbln-static"), ttm_resources.rbln
        )


def test_ttm_r2_rejects_wrong_furiosa_checkpoint_hash(
    ttm_resources, monkeypatch
):
    monkeypatch.setattr(
        benchmark_main,
        "validate_checkpoint",
        lambda path: (_ for _ in ()).throw(
            ValueError("checkpoint SHA-256 mismatch")
        ),
    )

    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources),
            get_target("furiosa-rngd-torch"),
            ttm_resources.checkpoint,
        )


def test_ttm_r2_rejects_target_outside_verified_matrix(ttm_resources):
    with pytest.raises(ValueError, match="verified targets"):
        benchmark_main.validate_ttm_r2_execution(
            _args(ttm_resources), get_target("cpu"), ttm_resources.dataset
        )


def test_ttm_r2_metadata_distinguishes_full_and_smoke(
    ttm_resources, monkeypatch
):
    monkeypatch.setattr(
        benchmark_main,
        "artifact_evidence",
        lambda path: {"sha256": "artifact-sha", "size_bytes": 123},
        raising=False,
    )

    full = benchmark_main.ttm_r2_result_metadata(
        _args(ttm_resources), get_target("rbln-static"), ttm_resources.rbln
    )
    smoke = benchmark_main.ttm_r2_result_metadata(
        _args(ttm_resources, max_steps=1),
        get_target("mobilint-aries"),
        ttm_resources.mxq,
    )

    assert full == {
        "ttm_contract_id": benchmark_main.TTM_R2_CONTRACT_ID,
        "ttm_validation_scope": "full",
        "ttm_expected_windows": 240,
        "ttm_dataset_sha256": "dataset-sha",
        "ttm_checkpoint_config_sha256": benchmark_main.TTM_R2_CONFIG_SHA256,
        "ttm_checkpoint_model_sha256": benchmark_main.TTM_R2_MODEL_SHA256,
        "ttm_artifact_sha256": "artifact-sha",
        "ttm_artifact_size_bytes": 123,
    }
    assert smoke["ttm_validation_scope"] == "smoke"


def test_ttm_r2_async_metadata_uses_max_samples_for_validation_scope(
    ttm_resources, monkeypatch
):
    monkeypatch.setattr(
        benchmark_main,
        "artifact_evidence",
        lambda path: {"sha256": "artifact-sha", "size_bytes": 123},
        raising=False,
    )

    metadata = benchmark_main.ttm_r2_result_metadata(
        _args(
            ttm_resources,
            inference_mode="async_queue",
            max_samples=1,
        ),
        get_target("rbln-static"),
        ttm_resources.rbln,
    )

    assert metadata["ttm_validation_scope"] == "smoke"


def test_ttm_r2_aries_runtime_diagnostics_are_safely_persisted():
    runtime = SimpleNamespace(
        get_device_spec=lambda: {
            "backend": "mobilint",
            "tensor_boundary_adapter_id": MOBILINT_TTM_R2_ADAPTER_ID,
            "mobilint_quantization_status": "unsaturated",
            "mobilint_saturation_elements": 0,
            "mobilint_saturation_total": 122880,
            "mobilint_input_scale_mode": "per_last_axis",
            "mobilint_input_zero_point": 0,
        }
    )

    safe = benchmark_main._safe_runtime_diagnostics(runtime)

    assert benchmark_main._runtime_result_metadata(safe) == {
        "mobilint_quantization_status": "unsaturated",
        "mobilint_saturation_elements": 0,
        "mobilint_saturation_total": 122880,
        "mobilint_input_scale_mode": "per_last_axis",
        "mobilint_input_zero_point": 0,
    }


def test_ttm_r2_aries_runtime_diagnostics_reject_unbounded_values():
    diagnostics = {
        "backend": "mobilint",
        "tensor_boundary_adapter_id": MOBILINT_TTM_R2_ADAPTER_ID,
        "mobilint_quantization_status": "maybe",
        "mobilint_saturation_elements": True,
        "mobilint_saturation_total": 122880,
        "mobilint_input_scale_mode": "custom",
        "mobilint_input_zero_point": 1,
    }

    assert benchmark_main._runtime_result_metadata(diagnostics) == {}


@pytest.mark.parametrize(
    ("target_id", "artifact_name", "runtime_backend"),
    [
        ("furiosa-rngd-torch", "checkpoint", "furiosa_torch"),
        ("rbln-static", "rbln", "rbln"),
        ("mobilint-aries", "mxq", "mobilint"),
    ],
)
def test_ttm_r2_main_reaches_common_benchmark_pipeline(
    monkeypatch,
    ttm_resources,
    target_id,
    artifact_name,
    runtime_backend,
):
    import utils.dataset_resolver as dataset_resolver

    captured = {"execute_calls": 0}
    artifact = getattr(ttm_resources, artifact_name)

    class TTMR2ETTh1Loader:
        total_samples = 240

        def get_metadata(self):
            return {"contract_id": "ttm-r2-etth1-ot-512-96-v1"}

    class TTMR2Evaluator:
        pass

    class FakeRuntime:
        def load(self, compiled_model):
            captured["compiled_model"] = compiled_model

    def fake_loader(**kwargs):
        captured["loader_kwargs"] = kwargs
        return TTMR2ETTh1Loader()

    def fake_evaluator(spec, **kwargs):
        captured["evaluator_spec"] = spec
        captured["evaluator_kwargs"] = kwargs
        return TTMR2Evaluator()

    def fake_runtime(backend, **kwargs):
        captured["runtime_request"] = (backend, kwargs)
        return FakeRuntime()

    def fake_execute(*args, **kwargs):
        captured["execute_calls"] += 1
        captured["execution_kwargs"] = kwargs
        return 0

    validations = []
    monkeypatch.setattr(benchmark_main, "run_auto_prepare", lambda *args: None)
    monkeypatch.setattr(
        benchmark_main,
        "_validate_furiosa_torch_cli",
        lambda args, task: Path(args.model_path),
    )
    monkeypatch.setattr(
        benchmark_main,
        "validate_ttm_r2_execution",
        lambda args, target, path: validations.append(
            (target.target_id, Path(path))
        ),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_main,
        "ttm_r2_result_metadata",
        lambda args, target, path: {"ttm_contract_id": "contract"},
        raising=False,
    )
    monkeypatch.setattr(benchmark_main, "create_model_spec", create_model_spec)
    monkeypatch.setattr(benchmark_main, "create_dataloader", fake_loader)
    monkeypatch.setattr(benchmark_main, "create_evaluator", fake_evaluator)
    monkeypatch.setattr(benchmark_main, "create_runtime", fake_runtime)
    monkeypatch.setattr(benchmark_main, "create_decoder", lambda *a, **k: object())
    monkeypatch.setattr(benchmark_main, "execute_benchmark", fake_execute)
    monkeypatch.setattr(
        dataset_resolver,
        "resolve_dataset_paths",
        lambda *args, **kwargs: (None, None),
    )

    cli = [
        "main.py",
        "--model",
        "ttm-r2",
        "--target",
        target_id,
        "--dataset",
        str(ttm_resources.dataset),
    ]
    if target_id == "furiosa-rngd-torch":
        cli.extend(["--model-path", str(artifact)])
    else:
        cli.extend(["--artifact", str(artifact)])
    monkeypatch.setattr(sys, "argv", cli)

    assert benchmark_main.main() == 0

    assert validations == [(target_id, artifact.resolve())]
    assert captured["execute_calls"] == 1
    assert (
        captured["execution_kwargs"]["loader"].__class__.__name__
        == "TTMR2ETTh1Loader"
    )
    assert (
        captured["execution_kwargs"]["evaluator"].__class__.__name__
        == "TTMR2Evaluator"
    )
    assert captured["loader_kwargs"]["model_spec"].name == "ttm-r2"
    assert captured["evaluator_spec"].name == "ttm-r2"
    backend, runtime_kwargs = captured["runtime_request"]
    assert backend == runtime_backend
    if target_id == "mobilint-aries":
        assert runtime_kwargs["tensor_boundary_adapter_id"] == (
            MOBILINT_TTM_R2_ADAPTER_ID
        )
        assert not (
            benchmark_main._MOBILINT_TENSOR_CONTRACT_OPTIONS
            & runtime_kwargs.keys()
        )
    else:
        assert "tensor_boundary_adapter_id" not in runtime_kwargs
    assert captured["execution_kwargs"]["result_metadata"] == {
        "ttm_contract_id": "contract"
    }

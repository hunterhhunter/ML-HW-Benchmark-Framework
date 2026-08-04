"""Static ONNX-to-MBLT compilation and ARIES execution primitives for TTM-R1."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ttm_r1.contracts import TTMR1Contract


_TARGET_DEVICE = "aries-rb"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qbcompiler_module() -> Any:
    try:
        return importlib.import_module("qbcompiler")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Mobilint compilation requires qbcompiler and onnxruntime in the selected environment"
        ) from exc


def _qbruntime_module() -> Any:
    try:
        return importlib.import_module("qbruntime")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("Mobilint execution requires qbruntime") from exc


def export_core_onnx(
    core: Any,
    inputs: Sequence[Any],
    contract: TTMR1Contract,
    output_path: str | Path,
) -> Path:
    """Export one tensor-only TTM core with no dynamic axes."""
    import torch

    output_path = Path(output_path).resolve()
    if output_path.suffix != ".onnx":
        raise ValueError("Mobilint export path must use the .onnx suffix")
    if output_path.exists():
        raise FileExistsError(f"Mobilint ONNX path already exists: {output_path}")
    if len(inputs) != 1:
        raise ValueError("Mobilint ONNX export input count does not match the core ABI")
    value = inputs[0]
    expected = contract.core_inputs[0]
    if tuple(value.shape) != expected.shape or value.dtype != torch.float32:
        raise ValueError(f"Mobilint ONNX input does not match {expected.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    core.eval()
    torch.onnx.export(
        core,
        tuple(inputs),
        str(output_path),
        input_names=[expected.name],
        output_names=[contract.core_output.name],
        opset_version=17,
        dynamic_axes=None,
        dynamo=False,
        do_constant_folding=True,
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError("PyTorch ONNX export did not create a nonempty file")
    return output_path


def run_onnx_reference(
    onnx_path: str | Path,
    inputs: Sequence[np.ndarray],
    contract: TTMR1Contract,
    *,
    onnxruntime_module: Any | None = None,
) -> np.ndarray:
    """Run the exported static graph on ONNX Runtime CPU before MBLT compilation."""
    onnx_path = Path(onnx_path).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Mobilint ONNX file does not exist: {onnx_path}")
    ordered = _validate_inputs(inputs, contract)
    if onnxruntime_module is None:
        try:
            onnxruntime_module = importlib.import_module("onnxruntime")
        except (ImportError, ModuleNotFoundError) as exc:
            raise ImportError("Mobilint ONNX validation requires onnxruntime") from exc
    session = onnxruntime_module.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        [contract.core_output.name],
        {item.name: value for item, value in zip(contract.core_inputs, ordered)},
    )
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("ONNX Runtime returned an unexpected output count")
    output = np.asarray(outputs[0], dtype=np.float32)
    if output.shape != contract.core_output.shape:
        raise ValueError(f"ONNX Runtime output shape mismatch: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("ONNX Runtime output contains non-finite values")
    return output


def compile_mblt(
    onnx_path: str | Path,
    artifact: str | Path,
    *,
    qbcompiler_module: Any | None = None,
) -> dict[str, object]:
    """Compile one static ONNX core into a new ARIES-targeted MBLT artifact."""
    onnx_path = Path(onnx_path).resolve()
    artifact = Path(artifact).resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Mobilint ONNX file does not exist: {onnx_path}")
    if artifact.suffix != ".mblt":
        raise ValueError("Mobilint artifact must use the .mblt suffix")
    if artifact.exists():
        raise FileExistsError(f"Mobilint artifact already exists: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    compiler = qbcompiler_module or _qbcompiler_module()
    compiler.mblt_compile_V2(
        str(onnx_path),
        target_device=_TARGET_DEVICE,
        mblt_save_path=str(artifact),
        backend="onnx",
        device="cpu",
        cpu_offload=False,
    )
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError("Mobilint compiler did not create a nonempty MBLT artifact")
    return {
        "target_device": _TARGET_DEVICE,
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
    }


def _validate_inputs(inputs: Sequence[np.ndarray], contract: TTMR1Contract) -> list[np.ndarray]:
    if len(inputs) != len(contract.core_inputs):
        raise ValueError("Mobilint input count does not match the TTM core ABI")
    normalized = []
    for value, expected in zip(inputs, contract.core_inputs):
        array = np.asarray(value)
        if array.shape != expected.shape:
            raise ValueError(f"Mobilint input {expected.name} shape mismatch: {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"Mobilint input {expected.name} must use float32")
        normalized.append(np.ascontiguousarray(array))
    return normalized


def _validate_model_shapes(model: Any, contract: TTMR1Contract) -> None:
    if [tuple(shape) for shape in model.get_model_input_shape()] != [
        contract.core_inputs[0].shape
    ]:
        raise ValueError("Mobilint artifact input shapes do not match the TTM core ABI")
    if [tuple(shape) for shape in model.get_model_output_shape()] != [
        contract.core_output.shape
    ]:
        raise ValueError("Mobilint artifact output shapes do not match the TTM core ABI")


def run_mblt(
    artifact: str | Path,
    inputs: Sequence[np.ndarray],
    contract: TTMR1Contract,
    *,
    qbruntime_module: Any | None = None,
) -> np.ndarray:
    """Run one existing MBLT artifact on ARIES device zero and return FP32 data."""
    artifact = Path(artifact).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Mobilint artifact does not exist: {artifact}")
    ordered = _validate_inputs(inputs, contract)
    runtime = qbruntime_module or _qbruntime_module()
    if 0 not in runtime.get_available_device_numbers():
        raise RuntimeError("Mobilint ARIES device 0 is not available")
    model = runtime.load(str(artifact))
    _validate_model_shapes(model, contract)
    outputs = model.infer_to_float(ordered)
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("Mobilint runtime returned an unexpected output count")
    output = np.asarray(outputs[0], dtype=np.float32)
    if output.shape != contract.core_output.shape:
        raise ValueError(f"Mobilint output shape mismatch: {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("Mobilint output contains non-finite values")
    return output

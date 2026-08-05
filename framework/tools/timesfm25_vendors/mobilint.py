"""Local MXQ compilation and remote ARIES execution helpers for TimesFM 2.5."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from timesfm25.contracts import TimesFM25Contract

_TARGET = "aries-rb"


@dataclass(frozen=True)
class MXQRun:
    output: np.ndarray
    input_abi: tuple[int, ...]
    output_abi: tuple[int, ...]
    saturated_elements: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _qbcompiler_module() -> Any:
    try:
        return importlib.import_module("qbcompiler")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("MXQ compilation requires qbcompiler and onnxruntime") from exc


def _qbruntime_module() -> Any:
    try:
        return importlib.import_module("qbruntime")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("ARIES execution requires qbruntime") from exc


def export_core_onnx(core: Any, inputs: Sequence[Any], contract: TimesFM25Contract, path: str | Path) -> Path:
    """Export the fixed float32 core without dynamic axes or host operations."""
    import torch

    path = Path(path).resolve()
    if path.suffix != ".onnx" or path.exists():
        raise ValueError("Choose a new .onnx export path")
    if len(inputs) != 1 or tuple(inputs[0].shape) != (1, 1024) or inputs[0].dtype != torch.float32:
        raise ValueError("TimesFM ONNX export input must be float32 [1,1024]")
    path.parent.mkdir(parents=True, exist_ok=True)
    core.eval()
    torch.onnx.export(
        core, tuple(inputs), str(path),
        input_names=[contract.core_inputs[0].name], output_names=[contract.core_output.name],
        opset_version=17, dynamic_axes=None, dynamo=False, do_constant_folding=True,
    )
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("ONNX export did not create a nonempty artifact")
    return path


def compile_mxq(
    onnx_path: str | Path, artifact: str | Path, calibration_dir: str | Path, feed: np.ndarray,
    *, qbcompiler_module: Any | None = None,
) -> dict[str, object]:
    """Compile one new ARIES MXQ with explicit, finite calibration tensors."""
    onnx_path, artifact, calibration_dir = map(lambda p: Path(p).resolve(), (onnx_path, artifact, calibration_dir))
    if not onnx_path.is_file() or not calibration_dir.is_dir() or not list(calibration_dir.glob("*.npy")):
        raise ValueError("ONNX and nonempty calibration directory are required")
    if artifact.suffix != ".mxq" or artifact.exists():
        raise ValueError("Choose a new .mxq artifact path")
    feed = np.asarray(feed, dtype=np.float32)
    if feed.shape != (1, 1024) or not np.isfinite(feed).all():
        raise ValueError("MXQ feed must be finite float32 [1,1024]")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    compiler = qbcompiler_module or _qbcompiler_module()
    compiler.mxq_compile_V2(
        str(onnx_path), target_device=_TARGET, calib_data_path=str(calibration_dir),
        save_path=str(artifact), backend="onnx", feed_dict={"normalized_context": feed},
        device="cpu", cpu_offload=False, use_random_calib=False,
    )
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError("MXQ compiler did not create a nonempty artifact")
    return {"target_device": _TARGET, "artifact": {"path": str(artifact), "size_bytes": artifact.stat().st_size, "sha256": _sha256(artifact)}}


def run_onnx_reference(onnx_path: str | Path, value: np.ndarray, contract: TimesFM25Contract) -> np.ndarray:
    """Run the exported graph on CPU before trusting it as an MXQ input."""
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError("ONNX validation requires onnxruntime") from exc
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (1, 1024):
        raise ValueError("TimesFM ONNX input must be float32 [1,1024]")
    session = onnxruntime.InferenceSession(str(Path(onnx_path).resolve()), providers=["CPUExecutionProvider"])
    output = session.run([contract.core_output.name], {contract.core_inputs[0].name: value})[0]
    output = np.asarray(output, dtype=np.float32)
    if output.shape != contract.core_output.shape or not np.isfinite(output).all():
        raise ValueError("ONNX Runtime did not return finite float32 [1,128]")
    return output


def _scale_to_array(scale: Any, shape: tuple[int, ...]) -> np.ndarray:
    if bool(getattr(scale, "is_asymmetric", False)):
        raise ValueError("asymmetric ARIES input quantization requires an explicit zero-point ABI")
    if bool(getattr(scale, "is_uniform", False)):
        value = float(getattr(scale, "scale"))
        if value <= 0:
            raise ValueError("ARIES input scale must be positive")
        return np.full(shape, value, dtype=np.float32)
    values = np.asarray(getattr(scale, "scale_list"), dtype=np.float32)
    if values.size == shape[-1]:
        return np.broadcast_to(values.reshape((1,) * (len(shape) - 1) + (-1,)), shape)
    if values.size == int(np.prod(shape)):
        return values.reshape(shape)
    raise ValueError(f"unsupported ARIES input scale shape: {values.size} values for {shape}")


def _quantize_input(value: np.ndarray, scale: Any, abi: tuple[int, ...]) -> tuple[np.ndarray, int]:
    if value.shape != (1, 1024) or value.dtype != np.float32:
        raise ValueError("TimesFM ARIES input must be float32 [1,1024]")
    if int(np.prod(abi)) != 1024:
        raise ValueError(f"unexpected ARIES input ABI: {abi}")
    shaped = value.reshape(abi)
    quantized = np.rint(shaped * _scale_to_array(scale, abi))
    saturated = int(np.count_nonzero((quantized < -128) | (quantized > 127)))
    return np.clip(quantized, -128, 127).astype(np.int8), saturated


def run_mxq(artifact: str | Path, value: np.ndarray, contract: TimesFM25Contract, *, qbruntime_module: Any | None = None) -> MXQRun:
    """Run a transferred MXQ on ARIES device zero and dequantize its output."""
    artifact = Path(artifact).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"MXQ artifact is missing: {artifact}")
    runtime = qbruntime_module or _qbruntime_module()
    if 0 not in runtime.get_available_device_numbers():
        raise RuntimeError("Mobilint ARIES device 0 is unavailable")
    model = runtime.load(str(artifact))
    try:
        input_abi = tuple(model.get_model_input_shape()[0])
        output_abi = tuple(model.get_model_output_shape()[0])
        if int(np.prod(output_abi)) != int(np.prod(contract.core_output.shape)):
            raise ValueError(f"unexpected ARIES output ABI: {output_abi}")
        quantized, saturated = _quantize_input(np.asarray(value), model.get_input_scale()[0], input_abi)
        outputs = model.infer_to_float([quantized])
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise ValueError("ARIES runtime returned an unexpected output count")
        output = np.asarray(outputs[0], dtype=np.float32).reshape(contract.core_output.shape)
    finally:
        model.dispose()
    if not np.isfinite(output).all():
        raise ValueError("ARIES output contains non-finite values")
    return MXQRun(output, input_abi, output_abi, saturated)

"""Static RBLN-CA22 compilation and execution primitives for TTM-R1."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chronos_bolt.contracts import TensorContract
from ttm_r1.contracts import TTMR1Contract


def _field(value: object, name: str) -> object:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_descriptor(
    actual: object, expected: TensorContract, *, allow_unnamed: bool = False
) -> None:
    name = _field(actual, "name")
    if not (allow_unnamed and name in (None, "")) and name != expected.name:
        raise ValueError(
            f"RBLN tensor name mismatch: expected {expected.name!r}, got {name!r}"
        )
    if tuple(_field(actual, "shape") or ()) != expected.shape:
        raise ValueError(f"RBLN tensor shape mismatch for {expected.name}")
    dtype = _field(actual, "dtype")
    if not isinstance(dtype, str) or dtype.lower() != expected.dtype:
        raise ValueError(f"RBLN tensor dtype mismatch for {expected.name}")


def _validate_inspection(inspection: object, contract: TTMR1Contract) -> dict[str, object]:
    if _field(inspection, "npu") != "RBLN-CA22":
        raise ValueError("RBLN artifact target must be RBLN-CA22")
    inputs = _field(inspection, "inputs")
    outputs = _field(inspection, "outputs")
    if not isinstance(inputs, (list, tuple)) or len(inputs) != 1:
        raise ValueError("RBLN artifact must expose exactly one TTM input")
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise ValueError("RBLN artifact must expose exactly one TTM output")
    _validate_descriptor(inputs[0], contract.core_inputs[0])
    _validate_descriptor(outputs[0], contract.core_output, allow_unnamed=True)
    return {
        "npu": "RBLN-CA22",
        "compiler_version": _field(inspection, "compiler_version"),
        "inputs": [
            {"name": item.name, "shape": list(item.shape), "dtype": item.dtype}
            for item in contract.core_inputs
        ],
        "output": {
            "name": contract.core_output.name,
            "shape": list(contract.core_output.shape),
            "dtype": contract.core_output.dtype,
        },
    }


def _rebel_module() -> Any:
    try:
        return importlib.import_module("rebel")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError("RBLN compilation requires the rebel-compiler SDK") from exc


def compile_rbln(
    core: object, contract: TTMR1Contract, artifact: str | Path
) -> dict[str, object]:
    """Compile and inspect one new CA22 artifact for the fixed core ABI."""
    artifact = Path(artifact).resolve()
    if artifact.suffix != ".rbln":
        raise ValueError("RBLN artifact must use the .rbln suffix")
    if artifact.exists():
        raise FileExistsError(f"RBLN artifact already exists: {artifact}")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    rebel = _rebel_module()
    compiled = rebel.compile_from_torch(
        core,
        [(item.name, list(item.shape), item.dtype) for item in contract.core_inputs],
    )
    compiled.save(str(artifact))
    if not artifact.is_file() or artifact.stat().st_size == 0:
        raise ValueError("RBLN compiler did not create a nonempty artifact")
    inspection = rebel.RBLNCompiledModel.inspect(str(artifact))
    return {
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": _sha256(artifact),
        },
        "inspection": _validate_inspection(inspection, contract),
    }


def _validate_inputs(
    inputs: Sequence[np.ndarray], contract: TTMR1Contract
) -> tuple[np.ndarray, ...]:
    if len(inputs) != len(contract.core_inputs):
        raise ValueError("RBLN input count does not match the TTM core ABI")
    normalized = []
    for value, expected in zip(inputs, contract.core_inputs):
        array = np.asarray(value)
        if array.shape != expected.shape:
            raise ValueError(f"RBLN input {expected.name} shape mismatch: {array.shape}")
        if array.dtype != np.float32:
            raise ValueError(f"RBLN input {expected.name} must use float32")
        normalized.append(np.ascontiguousarray(array))
    return tuple(normalized)


def run_rbln_artifact(
    artifact: str | Path,
    inputs: Sequence[np.ndarray],
    contract: TTMR1Contract,
    *,
    device_id: int = 0,
    timeout_sec: int = 60,
) -> np.ndarray:
    """Execute one existing artifact on CA22 device zero and validate its forecast."""
    if type(device_id) is not int or device_id != 0:
        raise ValueError("RBLN-CA22 execution requires device_id=0")
    if type(timeout_sec) is not int or timeout_sec <= 0:
        raise ValueError("timeout_sec must be a positive integer")
    ordered = _validate_inputs(inputs, contract)
    runtime = _rebel_module().Runtime(
        str(Path(artifact).resolve()), device=device_id, tensor_type="np", timeout=timeout_sec
    )
    raw = runtime(*ordered)
    if isinstance(raw, (tuple, list)):
        if len(raw) != 1:
            raise ValueError("RBLN runtime returned an unexpected output count")
        raw = raw[0]
    output = np.asarray(raw)
    if output.shape != contract.core_output.shape:
        raise ValueError(f"RBLN output shape mismatch: {output.shape}")
    if output.dtype != np.float32:
        raise ValueError(f"RBLN output dtype must be float32, got {output.dtype}")
    if not np.isfinite(output).all():
        raise ValueError("RBLN output contains non-finite values")
    return output

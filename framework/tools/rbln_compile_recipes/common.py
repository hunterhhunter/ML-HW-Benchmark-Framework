"""Shared contracts and safe artifact finalization for RBLN recipes."""

import argparse
import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TensorContract:
    name: str | None
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class RecipeContract:
    recipe: str
    model_id: str
    inputs: tuple[TensorContract, ...]
    outputs: tuple[TensorContract, ...]
    allow_unnamed_outputs: bool = False
    notes: tuple[str, ...] = ()
    target_npu: str = "RBLN-CA22"


def contract_to_dict(contract: RecipeContract) -> dict[str, object]:
    """Return a JSON-safe, stable representation of a recipe ABI."""
    return {
        "recipe": contract.recipe,
        "model_id": contract.model_id,
        "target_npu": contract.target_npu,
        "inputs": [
            {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
            for tensor in contract.inputs
        ],
        "outputs": [
            {"name": tensor.name, "shape": list(tensor.shape), "dtype": tensor.dtype}
            for tensor in contract.outputs
        ],
        "allow_unnamed_outputs": contract.allow_unnamed_outputs,
        "notes": list(contract.notes),
    }


def prepare_output_path(value: str | Path) -> Path:
    """Resolve an explicit new .rbln output path without overwriting artifacts."""
    output = Path(value).expanduser().resolve()
    if output.suffix != ".rbln":
        raise ValueError("RBLN compile output must use the .rbln suffix")
    if output.exists():
        raise FileExistsError(f"RBLN compile output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def create_parser() -> argparse.ArgumentParser:
    """Create the common explicit-output interface for a recipe command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--describe", action="store_true")
    return parser


def emit_description_or_require_output(
    args: argparse.Namespace, contract: RecipeContract
) -> Path | None:
    """Print a contract for discovery or return a new explicit output path."""
    if args.describe:
        print(json.dumps(contract_to_dict(contract), sort_keys=True))
        return None
    if args.output is None:
        raise ValueError("RBLN compilation requires an explicit --output .rbln path")
    return prepare_output_path(args.output)


def _field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _normalize_shape(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"RBLN {label} shape must be a sequence of positive dimensions")
    shape = tuple(value)
    if not shape or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f"RBLN {label} shape must contain only positive dimensions")
    return shape


def _normalize_dtype(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"RBLN {label} dtype is missing")
    return value.lower()


def _validate_tensors(
    kind: str,
    expected: tuple[TensorContract, ...],
    actual: object,
    *,
    allow_unnamed_outputs: bool = False,
) -> None:
    if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
        raise ValueError(f"RBLN inspection {kind}s must be a sequence")
    if len(actual) != len(expected):
        raise ValueError(
            f"RBLN {kind} count mismatch: expected {len(expected)}, got {len(actual)}"
        )
    for index, (wanted, observed) in enumerate(zip(expected, actual)):
        label = f"{kind} {index}"
        actual_name = _field(observed, "name")
        if kind == "input" or actual_name is not None:
            if actual_name != wanted.name:
                raise ValueError(
                    f"RBLN {label} name mismatch: expected {wanted.name!r}, "
                    f"got {actual_name!r}"
                )
        elif not allow_unnamed_outputs:
            raise ValueError(f"RBLN {label} name is required")
        actual_shape = _normalize_shape(_field(observed, "shape"), label)
        if actual_shape != wanted.shape:
            raise ValueError(
                f"RBLN {label} shape mismatch: expected {wanted.shape}, got {actual_shape}"
            )
        actual_dtype = _normalize_dtype(_field(observed, "dtype"), label)
        expected_dtype = _normalize_dtype(wanted.dtype, label)
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"RBLN {label} dtype mismatch: expected {expected_dtype!r}, "
                f"got {actual_dtype!r}"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_batch_one(contract: RecipeContract) -> None:
    for kind, tensors in (("input", contract.inputs), ("output", contract.outputs)):
        for index, tensor in enumerate(tensors):
            shape = _normalize_shape(tensor.shape, f"{kind} contract {index}")
            if shape[0] != 1:
                raise ValueError(
                    f"RBLN {kind} contract {index} must use fixed batch size 1"
                )


def save_and_validate(
    compiled_model: object, output: Path, contract: RecipeContract
) -> dict[str, object]:
    """Save once, inspect lazily, and verify the compiled fixed-batch ABI."""
    if contract.target_npu != "RBLN-CA22":
        raise ValueError("RBLN compile recipes target RBLN-CA22 only")
    _require_batch_one(contract)
    artifact = prepare_output_path(output)
    compiled_model.save(str(artifact))
    if not artifact.is_file():
        raise ValueError(f"RBLN compiler did not create output: {artifact}")
    size_bytes = artifact.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"RBLN compiler produced an empty output: {artifact}")

    rebel = importlib.import_module("rebel")
    inspection = rebel.RBLNCompiledModel.inspect(str(artifact))
    actual_npu = _field(inspection, "npu")
    if actual_npu != contract.target_npu:
        raise ValueError(
            f"RBLN NPU mismatch: expected {contract.target_npu!r}, got {actual_npu!r}"
        )
    _validate_tensors("input", contract.inputs, _field(inspection, "inputs"))
    _validate_tensors(
        "output",
        contract.outputs,
        _field(inspection, "outputs"),
        allow_unnamed_outputs=contract.allow_unnamed_outputs,
    )

    report = contract_to_dict(contract)
    report.update(
        {
            "output": str(artifact),
            "compiler_version": _field(inspection, "compiler_version"),
            "size_bytes": size_bytes,
            "sha256": _sha256(artifact),
        }
    )
    return report

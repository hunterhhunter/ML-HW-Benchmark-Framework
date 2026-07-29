"""Ordered tensor contracts for precompiled Mobilint MXQ artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .model_spec import Model_Spec


def _normalize_dtype(value: object, *, field: str) -> str:
    try:
        return np.dtype(value).name
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid NumPy dtype.") from exc


def _unbatched_shape(
    value: object,
    *,
    field: str,
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{field} must include a batch axis and tensor dimensions.")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, Integral)
        or dimension <= 0
        for dimension in value
    ):
        raise ValueError(f"{field} must contain positive integer dimensions.")
    return tuple(int(dimension) for dimension in value[1:])


@dataclass(frozen=True)
class MobilintTensor:
    name: str
    dtype: str | None
    unbatched_shape: tuple[int, ...]


@dataclass(frozen=True)
class MobilintTensorContract:
    profile_id: str
    inputs: tuple[MobilintTensor, ...]
    outputs: tuple[MobilintTensor, ...]
    max_batch_size: int
    native_async_supported: bool = False

    def runtime_contract(self) -> dict[str, object]:
        return {
            "artifact_profile_id": self.profile_id,
            "expected_input_names": [tensor.name for tensor in self.inputs],
            "expected_input_dtypes": [tensor.dtype for tensor in self.inputs],
            "expected_unbatched_input_shapes": [
                list(tensor.unbatched_shape) for tensor in self.inputs
            ],
            "expected_output_names": [tensor.name for tensor in self.outputs],
            "expected_unbatched_output_shapes": [
                list(tensor.unbatched_shape) for tensor in self.outputs
            ],
            "max_input_batch_size": self.max_batch_size,
            "native_async_supported": self.native_async_supported,
        }


def build_mobilint_tensor_contract(
    spec: Model_Spec,
    *,
    max_batch_size: int,
) -> MobilintTensorContract:
    """Derive the ordered qb Runtime boundary from an existing ModelSpec."""
    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, Integral)
        or max_batch_size <= 0
    ):
        raise ValueError("max_batch_size must be a positive integer.")

    inputs = tuple(
        MobilintTensor(
            name=name,
            dtype=_normalize_dtype(
                spec.input_dtype[name],
                field=f"input dtype for {name}",
            ),
            unbatched_shape=_unbatched_shape(
                shape,
                field=f"input {name!r} shape batch axis",
            ),
        )
        for name, shape in spec.input_shapes.items()
    )
    outputs = tuple(
        MobilintTensor(
            name=name,
            dtype=None,
            unbatched_shape=_unbatched_shape(
                shape,
                field=f"output {name!r} shape batch axis",
            ),
        )
        for name, shape in spec.output_shapes.items()
    )
    normalized_name = "-".join(
        part for part in str(spec.name).strip().casefold().replace("_", "-").split("-")
        if part
    )
    if not normalized_name:
        raise ValueError("ModelSpec name must be non-empty.")
    return MobilintTensorContract(
        profile_id=f"mobilint-{normalized_name}-tensor-v1",
        inputs=inputs,
        outputs=outputs,
        max_batch_size=int(max_batch_size),
        native_async_supported=False,
    )

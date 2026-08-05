"""Immutable fixed tensor contract for the TimesFM 2.5 point forecast."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TensorContract:
    """One named, static, floating-point tensor boundary."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor contract name must not be empty")
        if not self.shape or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("tensor contract dimensions must be positive")
        if self.dtype != "float32":
            raise ValueError("TimesFM 2.5 tensor contract dtype must be float32")


@dataclass(frozen=True)
class TimesFM25Contract:
    """Fixed single-series public and static-core boundary descriptions."""

    external_input: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract
    external_output: TensorContract

    @classmethod
    def fixed(cls) -> "TimesFM25Contract":
        point_forecast = TensorContract("point_forecast", (1, 128), "float32")
        return cls(
            external_input=TensorContract("context", (1, 1024), "float32"),
            core_inputs=(TensorContract("normalized_context", (1, 1024), "float32"),),
            core_output=point_forecast,
            external_output=point_forecast,
        )

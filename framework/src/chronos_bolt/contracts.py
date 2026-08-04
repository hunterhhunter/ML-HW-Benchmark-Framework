"""Immutable tensor contracts shared by every Chronos-Bolt compiler path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TensorContract:
    """One ordered tensor in an externally inspectable ABI."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tensor contract name must not be empty")
        if not self.shape or any(type(size) is not int or size <= 0 for size in self.shape):
            raise ValueError(f"tensor contract {self.name!r} has an invalid shape")
        if self.dtype != "float32":
            raise ValueError(f"tensor contract {self.name!r} must use float32")


class CompileStatus(str, Enum):
    """Evidence states; only the final value proves a device execution."""

    NOT_ATTEMPTED = "not_attempted"
    PREREQUISITE_MISSING = "prerequisite_missing"
    PARITY_FAILED = "parity_failed"
    COMPILE_BLOCKED_NO_DEVICE = "compile_blocked_no_device"
    COMPILE_FAILED = "compile_failed"
    NOT_RUNNABLE = "not_runnable"
    COMPILED = "compiled"
    DEVICE_VERIFIED = "device_verified"


@dataclass(frozen=True)
class ChronosBoltContract:
    """The fixed Tiny external and Transformer-core ABI."""

    d_model: int
    external_input: TensorContract
    external_output: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract
    quantile_levels: tuple[float, ...]

    @classmethod
    def tiny(cls, d_model: int) -> "ChronosBoltContract":
        """Build the official Tiny horizon-64 ABI from a checkpoint dimension."""
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        output = TensorContract("quantile_preds", (1, 9, 64), "float32")
        return cls(
            d_model=d_model,
            external_input=TensorContract("context", (1, 512), "float32"),
            external_output=output,
            core_inputs=(
                TensorContract("input_embeds", (1, 32, d_model), "float32"),
                TensorContract("attention_mask", (1, 32), "float32"),
                TensorContract("decoder_input_embeds", (1, 1, d_model), "float32"),
            ),
            core_output=output,
            quantile_levels=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        )

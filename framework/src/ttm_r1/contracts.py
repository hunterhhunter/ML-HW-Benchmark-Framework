"""Immutable tensor contract for the fixed univariate TTM-R1 benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from chronos_bolt.contracts import TensorContract


@dataclass(frozen=True)
class TTMR1Contract:
    """One public and core ABI shared by every TTM-R1 vendor artifact."""

    external_input: TensorContract
    external_output: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract

    def __post_init__(self) -> None:
        expected_input = TensorContract("context", (1, 512, 1), "float32")
        expected_output = TensorContract("forecast", (1, 96, 1), "float32")
        expected_core_inputs = (
            TensorContract("past_values", (1, 512, 1), "float32"),
        )
        if self.external_input != expected_input:
            raise ValueError("TTM-R1 context must use the fixed [1, 512, 1] ABI")
        if self.external_output != expected_output:
            raise ValueError("TTM-R1 forecast must use the fixed [1, 96, 1] ABI")
        if self.core_inputs != expected_core_inputs:
            raise ValueError("TTM-R1 past_values must use the fixed [1, 512, 1] ABI")
        if self.core_output != expected_output:
            raise ValueError("TTM-R1 core forecast must use the fixed [1, 96, 1] ABI")

    @classmethod
    def fixed(cls) -> "TTMR1Contract":
        """Return the only ABI accepted for cross-vendor comparison."""
        context = TensorContract("context", (1, 512, 1), "float32")
        forecast = TensorContract("forecast", (1, 96, 1), "float32")
        return cls(
            external_input=context,
            external_output=forecast,
            core_inputs=(TensorContract("past_values", (1, 512, 1), "float32"),),
            core_output=forecast,
        )

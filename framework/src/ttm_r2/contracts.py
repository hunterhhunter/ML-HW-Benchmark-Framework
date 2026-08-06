"""Immutable tensor contract for the fixed univariate TTM-R2 benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from chronos_bolt.contracts import TensorContract


@dataclass(frozen=True)
class TTMR2Contract:
    """One public and core ABI shared by all TTM-R2 vendor artifacts."""

    external_input: TensorContract
    external_output: TensorContract
    core_inputs: tuple[TensorContract, ...]
    core_output: TensorContract

    def __post_init__(self) -> None:
        context = TensorContract("context", (1, 512, 1), "float32")
        forecast = TensorContract("forecast", (1, 96, 1), "float32")
        core_inputs = (TensorContract("past_values", (1, 512, 1), "float32"),)
        if (self.external_input, self.external_output, self.core_inputs, self.core_output) != (
            context,
            forecast,
            core_inputs,
            forecast,
        ):
            raise ValueError("TTM-R2 requires the fixed float32 [1,512,1] to [1,96,1] ABI")

    @classmethod
    def fixed(cls) -> "TTMR2Contract":
        context = TensorContract("context", (1, 512, 1), "float32")
        forecast = TensorContract("forecast", (1, 96, 1), "float32")
        return cls(
            external_input=context,
            external_output=forecast,
            core_inputs=(TensorContract("past_values", (1, 512, 1), "float32"),),
            core_output=forecast,
        )

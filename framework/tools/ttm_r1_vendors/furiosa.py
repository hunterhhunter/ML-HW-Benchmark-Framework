"""Strict first-call Furiosa RNGD compilation primitive for TTM-R1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ttm_r1.contracts import TTMR1Contract


@dataclass(frozen=True)
class FuriosaDependencies:
    """Lazy SDK imports, injectable only by SDK-free unit tests."""

    torch: Any
    furiosa_torch: Any
    CompilerConfig: Any
    TacticHintConfig: Any


def _load_dependencies() -> FuriosaDependencies:
    import torch
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig

    return FuriosaDependencies(torch, furiosa_torch, CompilerConfig, TacticHintConfig)


def _validate_inputs(inputs: Sequence[Any], contract: TTMR1Contract, torch: Any) -> None:
    if len(inputs) != len(contract.core_inputs):
        raise ValueError("Furiosa input count does not match the TTM core ABI")
    for value, expected in zip(inputs, contract.core_inputs):
        if tuple(value.shape) != expected.shape:
            raise ValueError(f"Furiosa input {expected.name} shape mismatch")
        if value.dtype != torch.float32:
            raise ValueError(f"Furiosa input {expected.name} must use float32")


def run_furiosa_core(
    core: Any,
    inputs: Sequence[Any],
    contract: TTMR1Contract,
    *,
    device: str = "furiosa:0",
    dependencies: FuriosaDependencies | None = None,
) -> Any:
    """Strict-compile one static TTM graph and run the first call on RNGD."""
    dependencies = dependencies or _load_dependencies()
    torch = dependencies.torch
    _validate_inputs(inputs, contract, torch)
    torch_device = torch.device(device)
    model = core.eval().to(torch_device)
    device_inputs = tuple(value.to(torch_device) for value in inputs)
    compiler_config = dependencies.CompilerConfig(
        tactic_hint=dependencies.TacticHintConfig.Default
    )
    backend = dependencies.furiosa_torch.backend.with_config(
        compiler_config, eager_fallback=False
    )
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        output = compiled(*device_inputs)
    if isinstance(output, (tuple, list)):
        if len(output) != 1:
            raise ValueError("Furiosa runtime returned an unexpected output count")
        output = output[0]
    if tuple(output.shape) != contract.core_output.shape:
        raise ValueError(f"Furiosa output shape mismatch: {tuple(output.shape)}")
    if output.dtype != torch.float32:
        raise ValueError(f"Furiosa output dtype must be float32, got {output.dtype}")
    if not bool(torch.isfinite(output).all()):
        raise ValueError("Furiosa output contains non-finite values")
    return output

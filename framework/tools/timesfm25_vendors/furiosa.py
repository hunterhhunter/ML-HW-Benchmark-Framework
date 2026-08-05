"""Strict static Furiosa RNGD compilation primitive for TimesFM 2.5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from timesfm25.contracts import TimesFM25Contract


@dataclass(frozen=True)
class FuriosaDependencies:
    torch: Any
    furiosa_torch: Any
    CompilerConfig: Any
    TacticHintConfig: Any


def _load_dependencies() -> FuriosaDependencies:
    import torch
    import furiosa.torch as furiosa_torch
    from furiosa.torch.config import CompilerConfig, TacticHintConfig

    return FuriosaDependencies(torch, furiosa_torch, CompilerConfig, TacticHintConfig)


def _validate_inputs(inputs: Sequence[Any], contract: TimesFM25Contract, torch: Any) -> None:
    if len(inputs) != 1:
        raise ValueError("Furiosa input count does not match the TimesFM core ABI")
    value = inputs[0]
    if tuple(value.shape) != contract.core_inputs[0].shape or value.dtype != torch.float32:
        raise ValueError("Furiosa TimesFM input must be float32 [1, 1024]")


def compile_furiosa_runner(
    core: Any,
    contract: TimesFM25Contract,
    *,
    device: str = "furiosa:0",
    dependencies: FuriosaDependencies | None = None,
) -> Callable[[Sequence[Any]], Any]:
    """Build one strict static compiler runner; compilation occurs on its first call."""
    dependencies = dependencies or _load_dependencies()
    torch = dependencies.torch
    model = core.eval().to(torch.device(device))
    config = dependencies.CompilerConfig(tactic_hint=dependencies.TacticHintConfig.Default)
    backend = dependencies.furiosa_torch.backend.with_config(config, eager_fallback=False)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    def run(inputs: Sequence[Any]) -> Any:
        _validate_inputs(inputs, contract, torch)
        device_inputs = tuple(value.to(torch.device(device)) for value in inputs)
        with torch.inference_mode():
            output = compiled(*device_inputs)
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError("Furiosa runtime returned an unexpected output count")
            output = output[0]
        if tuple(output.shape) != contract.core_output.shape or output.dtype != torch.float32:
            raise ValueError("Furiosa output must be float32 [1, 128]")
        if not bool(torch.isfinite(output).all()):
            raise ValueError("Furiosa output contains non-finite values")
        return output

    return run

"""CPU reference preflight for TTM-R1 before any vendor compiler runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .contracts import TTMR1Contract
from .core import TTMR1Core, load_ttm_r1_model
from .host_adapter import TTMR1HostAdapter


@dataclass(frozen=True)
class TTMR1Preflight:
    """All static tensors and parity evidence required by a device runner."""

    core: TTMR1Core
    contract: TTMR1Contract
    core_inputs: dict[str, tuple[torch.Tensor]]
    core_outputs: dict[str, torch.Tensor]
    host_parity: dict[str, dict[str, float | list[int] | str]]


def _reference_contexts() -> dict[str, torch.Tensor]:
    finite = torch.linspace(-4.0, 7.0, steps=512, dtype=torch.float32).reshape(1, 512, 1)
    with_nan = finite.clone()
    with_nan[:, 17, :] = torch.nan
    with_nan[:, 289, :] = torch.nan
    return {"finite": finite, "nan": with_nan}


def run_preflight(model_path: str | Path) -> TTMR1Preflight:
    """Prove host and wrapper agreement for finite and missing-value contexts."""
    path = Path(model_path)
    if not path.is_dir():
        raise ValueError(f"TTM-R1 requires a local checkpoint directory: {path}")

    model = load_ttm_r1_model(str(path))
    core = TTMR1Core(model).eval()
    adapter = TTMR1HostAdapter(core.contract)
    core_inputs: dict[str, tuple[torch.Tensor]] = {}
    core_outputs: dict[str, torch.Tensor] = {}
    host_parity: dict[str, dict[str, float | list[int] | str]] = {}

    with torch.no_grad():
        for name, context in _reference_contexts().items():
            prepared = adapter.prepare(context)
            reference_output = TTMR1Core._extract_forecast(
                model(past_values=prepared.past_values, return_dict=True)
            )
            core_output = core(prepared.past_values)
            reference_forecast = prepared.restore(reference_output)
            split_forecast = prepared.restore(core_output)
            if not bool(torch.isfinite(reference_forecast).all()):
                raise ValueError(f"TTM-R1 reference output contains non-finite values for {name}")
            if not bool(torch.isfinite(split_forecast).all()):
                raise ValueError(f"TTM-R1 split output contains non-finite values for {name}")
            torch.testing.assert_close(split_forecast, reference_forecast, rtol=1e-5, atol=1e-6)
            delta = (split_forecast - reference_forecast).abs()
            host_parity[name] = {
                "shape": list(split_forecast.shape),
                "dtype": str(split_forecast.dtype).removeprefix("torch."),
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
            }
            core_inputs[name] = (prepared.past_values,)
            core_outputs[name] = core_output

    return TTMR1Preflight(
        core=core,
        contract=core.contract,
        core_inputs=core_inputs,
        core_outputs=core_outputs,
        host_parity=host_parity,
    )

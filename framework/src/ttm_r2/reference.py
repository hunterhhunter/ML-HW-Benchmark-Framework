"""CPU public/split preflight for the fixed R2 main checkpoint."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from ttm_r1.core import StaticTTMR1Scaler

from .contracts import TTMR2Contract
from .core import TTMR2Core, load_ttm_r2_model
from .host_adapter import TTMR2HostAdapter


@dataclass(frozen=True)
class TTMR2Preflight:
    core: TTMR2Core
    contract: TTMR2Contract
    core_inputs: dict[str, tuple[torch.Tensor, ...]]
    core_outputs: dict[str, torch.Tensor]
    host_parity: dict[str, dict[str, float | list[int] | str]]


def _contexts() -> dict[str, torch.Tensor]:
    finite = torch.linspace(-4.0, 7.0, steps=512, dtype=torch.float32).reshape(1, 512, 1)
    with_nan = finite.clone()
    with_nan[:, 17, :] = torch.nan
    with_nan[:, 289, :] = torch.nan
    return {"finite": finite, "nan": with_nan}


def run_preflight(model_path: str | Path) -> TTMR2Preflight:
    model = load_ttm_r2_model(str(model_path))
    core = TTMR2Core(deepcopy(model)).eval()
    adapter = TTMR2HostAdapter(
        core.contract,
        split_ttm_scaler=isinstance(getattr(core.model.backbone, "scaler", None), StaticTTMR1Scaler),
    )
    inputs, outputs, parity = {}, {}, {}
    with torch.inference_mode():
        for name, context in _contexts().items():
            prepared = adapter.prepare(context)
            reference = model(past_values=prepared.reference_past_values, return_dict=True).prediction_outputs
            core_output = core(prepared.past_values)
            reference_forecast = prepared.restore_reference(reference)
            split_forecast = prepared.restore(core_output)
            torch.testing.assert_close(split_forecast, reference_forecast, rtol=1e-5, atol=1e-6)
            delta = (split_forecast - reference_forecast).abs()
            parity[name] = {
                "shape": list(split_forecast.shape),
                "dtype": str(split_forecast.dtype).removeprefix("torch."),
                "max_abs_error": float(delta.max()),
                "mean_abs_error": float(delta.mean()),
            }
            inputs[name] = (prepared.past_values,)
            outputs[name] = core_output
    return TTMR2Preflight(core, core.contract, inputs, outputs, parity)

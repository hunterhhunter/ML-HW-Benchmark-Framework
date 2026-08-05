"""Public-reference parity for the static TimesFM 2.5 point core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .contracts import TimesFM25Contract
from .core import TimesFM25PointCore
from .host_adapter import TimesFM25HostAdapter
from .model import load_timesfm25_model


@dataclass(frozen=True)
class TimesFM25Preflight:
    """Deterministic CPU evidence that the public and split paths agree."""

    contract: TimesFM25Contract
    core: TimesFM25PointCore
    adapter: TimesFM25HostAdapter
    context: torch.Tensor
    public_output: torch.Tensor
    core_inputs: dict[str, tuple[torch.Tensor, ...]]
    core_outputs: dict[str, torch.Tensor]
    split_output: torch.Tensor
    host_parity: dict[str, float]


def assert_public_split_parity(public_output: torch.Tensor, split_output: torch.Tensor) -> dict[str, float]:
    """Return explicit CPU metrics and fail before an invalid graph is compiled."""
    torch.testing.assert_close(split_output, public_output, rtol=1e-5, atol=1e-5)
    delta = (split_output - public_output).abs()
    return {
        "max_abs_error": float(delta.max()),
        "mean_abs_error": float(delta.mean()),
        "rmse": float(delta.square().mean().sqrt()),
    }


def run_preflight(model_path: str | Path) -> TimesFM25Preflight:
    """Prove the fixed static core preserves the public point forecast exactly."""
    model = load_timesfm25_model(str(model_path))
    core = TimesFM25PointCore(model).eval()
    adapter = TimesFM25HostAdapter(model)
    context = torch.linspace(-4.0, 7.0, steps=1024, dtype=torch.float32).reshape(1, 1024)
    with torch.inference_mode():
        public = model(
            past_values=[context.squeeze(0)],
            forecast_context_len=1024,
            return_dict=True,
        ).mean_predictions
        prepared = adapter.prepare(context)
        normal = core(prepared.normalized_context)
        flipped = core(prepared.flipped_context)
        split = prepared.restore(normal, flipped)
    parity = assert_public_split_parity(public, split)
    return TimesFM25Preflight(
        contract=core.contract,
        core=core,
        adapter=adapter,
        context=context,
        public_output=public,
        core_inputs={"normal": (prepared.normalized_context,), "flipped": (prepared.flipped_context,)},
        core_outputs={"normal": normal, "flipped": flipped},
        split_output=split,
        host_parity=parity,
    )

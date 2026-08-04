"""CPU scaling and restoration for the fixed TTM-R1 model boundary."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import TTMR1Contract


@dataclass(frozen=True)
class PreparedTTMR1Inputs:
    """Static normalized core input and its exact inverse scale state."""

    past_values: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor

    def restore(self, normalized_forecast: torch.Tensor) -> torch.Tensor:
        """Invert CPU standard scaling after a tensor-only core execution."""
        expected = TTMR1Contract.fixed().core_output
        if not isinstance(normalized_forecast, torch.Tensor):
            raise ValueError("forecast must be a torch Tensor")
        if tuple(normalized_forecast.shape) != expected.shape:
            raise ValueError(
                "forecast shape must be "
                f"{expected.shape}, got {tuple(normalized_forecast.shape)}"
            )
        if normalized_forecast.dtype != torch.float32:
            raise ValueError("forecast must use float32")
        return normalized_forecast * self.scale + self.loc


class TTMR1HostAdapter:
    """Convert a raw context into TTM-R1's externally standardized values."""

    _EPSILON = 1e-6

    def __init__(self, contract: TTMR1Contract | None = None) -> None:
        self.contract = contract or TTMR1Contract.fixed()

    def prepare(self, context: torch.Tensor) -> PreparedTTMR1Inputs:
        """Normalize finite observations and replace missing values by their mean."""
        self._validate_context(context)
        observed = torch.isfinite(context)
        count = observed.sum(dim=1, keepdim=True)
        if not bool((count > 0).all()):
            raise ValueError("TTM-R1 context must contain at least one observed value")

        safe_values = torch.where(observed, context, torch.zeros_like(context))
        loc = safe_values.sum(dim=1, keepdim=True) / count.to(dtype=torch.float32)
        centered = torch.where(observed, safe_values - loc, torch.zeros_like(safe_values))
        variance = centered.square().sum(dim=1, keepdim=True) / count.to(dtype=torch.float32)
        scale = variance.sqrt().clamp_min(self._EPSILON)
        filled = torch.where(observed, context, loc.expand_as(context))
        past_values = (filled - loc) / scale
        if not bool(torch.isfinite(past_values).all()):
            raise ValueError("TTM-R1 normalized core input contains non-finite values")
        return PreparedTTMR1Inputs(past_values=past_values, loc=loc, scale=scale)

    def _validate_context(self, context: torch.Tensor) -> None:
        expected = self.contract.external_input
        if not isinstance(context, torch.Tensor):
            raise ValueError("context must be a torch Tensor")
        if tuple(context.shape) != expected.shape:
            raise ValueError(
                f"context shape must be {expected.shape}, got {tuple(context.shape)}"
            )
        if context.dtype != torch.float32:
            raise ValueError("context must use float32")

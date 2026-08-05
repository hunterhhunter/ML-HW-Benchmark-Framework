"""CPU-only fixed TimesFM 2.5 public-path preparation and restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import TimesFM25Contract


@dataclass(frozen=True)
class PreparedTimesFM25Inputs:
    """The two fixed core invocations and global public-path statistics."""

    normalized_context: torch.Tensor
    flipped_context: torch.Tensor
    loc: torch.Tensor
    scale: torch.Tensor
    input_was_nonnegative: torch.Tensor

    def restore(self, normal_forecast: torch.Tensor, flipped_forecast: torch.Tensor) -> torch.Tensor:
        """Reproduce public flip-invariance, global RevIN, and conditional clamping."""
        self._validate_forecast(normal_forecast)
        self._validate_forecast(flipped_forecast)
        combined = (normal_forecast - flipped_forecast) / 2
        restored = combined * self.scale + self.loc
        zero = torch.zeros(1, dtype=restored.dtype, device=restored.device)
        return torch.where(self.input_was_nonnegative, torch.maximum(restored, zero), restored)

    @staticmethod
    def _validate_forecast(value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError("TimesFM 2.5 point forecast must be a torch tensor")
        if tuple(value.shape) != (1, 128) or value.dtype != torch.float32:
            raise ValueError("TimesFM 2.5 point forecast must be float32 [1,128]")


class TimesFM25HostAdapter:
    """Move the public fixed-context global preprocessing outside the device graph."""

    def __init__(self, model: torch.nn.Module | None = None) -> None:
        self.contract = TimesFM25Contract.fixed()
        backbone = getattr(model, "model", None)
        self._tolerance = float(getattr(backbone, "tolerance", 1e-6))

    def prepare(self, context: torch.Tensor) -> PreparedTimesFM25Inputs:
        """Reproduce `TimesFm2_5ModelForPrediction` before `_decode_and_project`."""
        expected = self.contract.external_input.shape
        if not isinstance(context, torch.Tensor):
            raise ValueError("TimesFM 2.5 context must be a torch tensor")
        if tuple(context.shape) != expected or context.dtype != torch.float32:
            raise ValueError("TimesFM 2.5 context must be float32 [1,1024]")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("TimesFM 2.5 fixed context must be finite")
        loc = context.mean(dim=1, keepdim=True)
        scale = context.std(dim=1, keepdim=True)
        safe_scale = torch.where(scale < self._tolerance, torch.ones_like(scale), scale)
        normalized = (context - loc) / safe_scale
        return PreparedTimesFM25Inputs(
            normalized_context=normalized,
            flipped_context=-normalized,
            loc=loc,
            scale=scale,
            input_was_nonnegative=torch.min(context) >= 0,
        )

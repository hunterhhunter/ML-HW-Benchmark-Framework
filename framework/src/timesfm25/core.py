"""Static point-forecast core extracted from TimesFM 2.5's public path."""

from __future__ import annotations

from typing import Any

import torch

from .contracts import TimesFM25Contract


class TimesFM25PointCore(torch.nn.Module):
    """Run one normalized fixed context through TimesFM and return its median forecast."""

    def __init__(self, prediction_model: torch.nn.Module) -> None:
        super().__init__()
        self.prediction_model = prediction_model
        self.backbone = getattr(prediction_model, "model", None)
        self.output_projection_point = getattr(prediction_model, "output_projection_point", None)
        self.config = getattr(prediction_model, "config", None)
        if not isinstance(self.backbone, torch.nn.Module):
            raise ValueError("TimesFM 2.5 prediction model has no backbone module")
        if not isinstance(self.output_projection_point, torch.nn.Module):
            raise ValueError("TimesFM 2.5 prediction model has no point projection module")
        self._validate_config()
        self.contract = TimesFM25Contract.fixed()
        self.register_buffer("padding", torch.zeros((1, 1024), dtype=torch.long))

    def forward(self, normalized_context: torch.Tensor) -> torch.Tensor:
        """Return static FP32 median forecast `[1,128]` before global restoration."""
        self._validate_input(normalized_context)
        model_outputs = self.backbone(
            past_values=normalized_context,
            past_values_padding=self.padding,
        )
        hidden_states = _field(model_outputs, "last_hidden_state")
        context_mu = _field(model_outputs, "context_mu")
        context_sigma = _field(model_outputs, "context_sigma")
        if not all(isinstance(value, torch.Tensor) for value in (hidden_states, context_mu, context_sigma)):
            raise ValueError("TimesFM 2.5 backbone output is incomplete")
        point_output = self.output_projection_point(hidden_states)
        point_output = self.backbone._revin(point_output, context_mu, context_sigma, reverse=True)
        num_quantiles = len(self.config.quantiles) + 1
        point_output = point_output.reshape(1, 32, 128, num_quantiles)
        forecast = point_output[:, -1, :, int(self.config.decode_index)]
        self._validate_output(forecast)
        return forecast

    def _validate_config(self) -> None:
        expected = {
            "patch_length": 32,
            "horizon_length": 128,
            "num_hidden_layers": 20,
            "hidden_size": 1280,
            "decode_index": 5,
        }
        actual = {name: getattr(self.config, name, None) for name in expected}
        if actual != expected or len(getattr(self.config, "quantiles", ())) != 9:
            raise ValueError(f"TimesFM 2.5 point-core config is unsupported: {actual}")

    @staticmethod
    def _validate_input(value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise ValueError("TimesFM 2.5 core input must be a torch tensor")
        if tuple(value.shape) != (1, 1024) or value.dtype != torch.float32:
            raise ValueError("TimesFM 2.5 core input must be float32 [1,1024]")

    @staticmethod
    def _validate_output(value: torch.Tensor) -> None:
        if tuple(value.shape) != (1, 128) or value.dtype != torch.float32:
            raise ValueError("TimesFM 2.5 core output must be float32 [1,128]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("TimesFM 2.5 core output must be finite")


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)

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
        self.register_buffer("patch_mask", torch.zeros((1, 32, 32), dtype=torch.bool))
        self.register_buffer("position_ids", torch.arange(32, dtype=torch.long).unsqueeze(0))
        causal = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
        causal.masked_fill_(torch.triu(torch.ones((32, 32), dtype=torch.bool), diagonal=1), torch.finfo(torch.float32).min)
        self.register_buffer("causal_mask", causal)

    def forward(self, normalized_context: torch.Tensor) -> torch.Tensor:
        """Return static FP32 median forecast `[1,128]` before global restoration."""
        self._validate_input(normalized_context)
        hidden_states, context_mu, context_sigma = self._run_static_backbone(normalized_context)
        point_output = self.output_projection_point(hidden_states)
        point_output = self.backbone._revin(point_output, context_mu, context_sigma, reverse=True)
        num_quantiles = len(self.config.quantiles) + 1
        point_output = point_output.reshape(1, 32, 128, num_quantiles)
        forecast = point_output[:, -1, :, int(self.config.decode_index)]
        self._validate_output(forecast)
        return forecast

    def _run_static_backbone(self, past_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Specialize known all-valid 1024-step inputs before exporting the core."""
        patched_inputs = past_values.reshape(1, 32, 32)
        count = past_values.new_zeros(1)
        mean = past_values.new_zeros(1)
        std = past_values.new_zeros(1)
        mean_history, std_history = [], []
        for index in range(32):
            count, mean, std = self.backbone._update_running_stats(
                count, mean, std, patched_inputs[:, index, :], self.patch_mask[:, index, :]
            )
            mean_history.append(mean)
            std_history.append(std)
        context_mu = torch.stack(mean_history, dim=1)
        context_sigma = torch.stack(std_history, dim=1)
        normed = self.backbone._revin(
            patched_inputs, context_mu, context_sigma, reverse=False, mask=self.patch_mask
        )
        tokenizer_inputs = torch.cat([normed, self.patch_mask.to(dtype=normed.dtype)], dim=-1)
        hidden_states = self.backbone.input_ff_layer(tokenizer_inputs)
        position_embeddings = self.backbone.rotary_emb(hidden_states, self.position_ids)
        for layer in self.backbone.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=self.causal_mask,
                position_ids=self.position_ids,
            )
        return hidden_states, context_mu, context_sigma

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

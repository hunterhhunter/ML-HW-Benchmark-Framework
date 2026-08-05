from types import SimpleNamespace

import torch

from timesfm25.core import TimesFM25PointCore


class _Backbone(torch.nn.Module):
    def forward(self, *, past_values, past_values_padding):
        assert past_values.shape == (1, 1024)
        assert torch.equal(past_values_padding, torch.zeros((1, 1024), dtype=torch.long))
        hidden = torch.zeros((1, 32, 4), dtype=torch.float32)
        return SimpleNamespace(
            last_hidden_state=hidden,
            context_mu=torch.zeros((1, 32), dtype=torch.float32),
            context_sigma=torch.ones((1, 32), dtype=torch.float32),
        )

    @staticmethod
    def _revin(values, loc, scale, *, reverse):
        assert reverse is True
        return values * scale.unsqueeze(-1) + loc.unsqueeze(-1)


class _PredictionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.config = SimpleNamespace(
            patch_length=32,
            horizon_length=128,
            num_hidden_layers=20,
            hidden_size=1280,
            decode_index=5,
            quantiles=tuple(range(9)),
        )

        self.output_projection_point = _Projection()


class _Projection(torch.nn.Module):
    def forward(self, hidden):
        values = torch.zeros((1, 32, 1280), dtype=torch.float32)
        values[:, -1, 5::10] = 7.0
        return values


def test_point_core_selects_last_patch_median_forecast():
    output = TimesFM25PointCore(_PredictionModel())(torch.zeros((1, 1024)))

    assert output.shape == (1, 128)
    assert torch.equal(output, torch.full((1, 128), 7.0))

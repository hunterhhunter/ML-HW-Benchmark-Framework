from types import SimpleNamespace

import torch

from timesfm25.core import TimesFM25PointCore


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ff_layer = _InputLayer()
        self.rotary_emb = _Rotary()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(20)])

    @staticmethod
    def _revin(values, loc, scale, *, reverse, mask=None):
        if reverse:
            return values * scale.unsqueeze(-1) + loc.unsqueeze(-1)
        assert mask is not None
        return values

    @staticmethod
    def _update_running_stats(count, mean, std, values, mask):
        del count, mean, std, mask
        return torch.ones(1), values.mean(dim=-1), torch.ones(1)


class _InputLayer(torch.nn.Module):
    def forward(self, values):
        return torch.zeros((1, 32, 4), dtype=values.dtype)


class _Rotary(torch.nn.Module):
    def forward(self, values, positions):
        return values, positions


class _Layer(torch.nn.Module):
    def forward(self, values, **kwargs):
        del kwargs
        return values


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


def test_point_core_is_fullgraph_traceable_without_tensor_bool_checks():
    core = TimesFM25PointCore(_PredictionModel())
    compiled = torch.compile(core, backend="eager", fullgraph=True, dynamic=False)

    assert compiled(torch.zeros((1, 1024), dtype=torch.float32)).shape == (1, 128)

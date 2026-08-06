from types import SimpleNamespace

import torch

from ttm_r2.core import TTMR2Core


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SimpleNamespace(scaler=None, patching=None)

    def forward(self, *, past_values, return_dict):
        assert return_dict is True
        return SimpleNamespace(prediction_outputs=past_values[:, :96, :])


def test_r2_core_exposes_the_fixed_forecast_tensor():
    output = TTMR2Core(_Model())(torch.zeros((1, 512, 1), dtype=torch.float32))

    assert output.shape == (1, 96, 1)

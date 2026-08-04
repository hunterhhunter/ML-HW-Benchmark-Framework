from types import SimpleNamespace

import pytest
import torch

from ttm_r1.core import TTMR1Core


class _FakeTTM(torch.nn.Module):
    def forward(self, *, past_values, return_dict):
        assert return_dict is True
        forecast = past_values[:, -96:, :] + 0.25
        return SimpleNamespace(prediction_outputs=forecast)


class _BadTTM(torch.nn.Module):
    def forward(self, *, past_values, return_dict):
        return SimpleNamespace(prediction_outputs=past_values[:, :-1, :])


def test_core_unwraps_prediction_outputs_to_a_single_forecast_tensor():
    """Catches a Hugging Face output container escaping into vendor export."""
    core = TTMR1Core(_FakeTTM())

    output = core(torch.zeros((1, 512, 1), dtype=torch.float32))

    assert output.shape == (1, 96, 1)
    assert output.dtype == torch.float32
    assert torch.allclose(output, torch.full((1, 96, 1), 0.25))


def test_core_rejects_a_model_output_without_a_96_step_prediction():
    """Catches a checkpoint or library change before device compilation."""
    with pytest.raises(ValueError, match="forecast"):
        TTMR1Core(_BadTTM())(torch.zeros((1, 512, 1), dtype=torch.float32))


def test_core_rejects_a_non_contract_input_tensor():
    """Catches dynamic shape input from reaching a supposedly static artifact."""
    with pytest.raises(ValueError, match="past_values"):
        TTMR1Core(_FakeTTM())(torch.zeros((1, 511, 1), dtype=torch.float32))

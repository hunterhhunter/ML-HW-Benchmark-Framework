import pytest
import torch

from ttm_r1.host_adapter import TTMR1HostAdapter


def test_adapter_standardizes_finite_values_and_restores_original_scale():
    """Catches host processing that changes the learned core's value domain."""
    context = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)

    prepared = TTMR1HostAdapter().prepare(context)

    assert prepared.past_values.shape == (1, 512, 1)
    assert prepared.reference_past_values.shape == (1, 512, 1)
    assert prepared.past_values.dtype == torch.float32
    assert torch.allclose(prepared.past_values.mean(dim=1), torch.zeros((1, 1)))
    assert torch.allclose(
        prepared.restore(prepared.past_values[:, :96]), context[:, :96], atol=1e-4
    )


def test_adapter_replaces_nan_with_the_observed_mean_before_normalization():
    """Catches an NPU-bound core input that still contains NaN values."""
    context = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)
    context[:, 16, :] = torch.nan

    prepared = TTMR1HostAdapter().prepare(context)

    assert torch.isfinite(prepared.past_values).all()
    assert prepared.reference_past_values[0, 16, 0] == pytest.approx(0.0)


def test_adapter_emulates_ttm_internal_std_scaler_on_cpu():
    """Catches moving CA22-incompatible scaling without preserving the checkpoint math."""
    context = torch.arange(512, dtype=torch.float32).reshape(1, 512, 1)

    prepared = TTMR1HostAdapter().prepare(context)
    reference_values = prepared.reference_past_values
    loc = reference_values.mean(dim=1, keepdim=True)
    scale = ((reference_values - loc).square().mean(dim=1, keepdim=True) + 1e-5).sqrt()

    assert torch.allclose(prepared.past_values, (reference_values - loc) / scale)
    assert torch.allclose(prepared.restore(prepared.past_values[:, :96]), context[:, :96], atol=1e-4)


def test_adapter_rejects_wrong_shape_and_all_missing_context():
    """Catches invalid benchmark inputs before compiler invocation."""
    adapter = TTMR1HostAdapter()

    with pytest.raises(ValueError, match="shape"):
        adapter.prepare(torch.zeros((1, 512), dtype=torch.float32))
    with pytest.raises(ValueError, match="observed"):
        adapter.prepare(torch.full((1, 512, 1), torch.nan))

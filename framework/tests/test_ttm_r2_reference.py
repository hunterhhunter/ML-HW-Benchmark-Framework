import torch

from ttm_r2.reference import _contexts


def test_r2_preflight_defines_finite_and_missing_value_fixed_contexts():
    contexts = _contexts()

    assert set(contexts) == {"finite", "nan"}
    assert contexts["finite"].shape == (1, 512, 1)
    assert torch.isfinite(contexts["finite"]).all()
    assert torch.isnan(contexts["nan"]).sum() == 2

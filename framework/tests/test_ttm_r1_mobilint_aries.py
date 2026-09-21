from types import SimpleNamespace

import numpy as np

from ttm_r1.mobilint_aries import quantize_core_input, restore_artifact_output


def test_quantize_core_input_uses_aries_last_axis_scales_and_counts_clipping():
    """Catches feeding ARIES float input or hiding int8 saturation."""
    scale = SimpleNamespace(
        scale=0.0,
        is_uniform=False,
        scale_list=[100.0] * 64,
        zero_point=0,
        is_asymmetric=False,
        zero_points=[],
    )

    value, saturated = quantize_core_input(
        np.full((1, 512, 1), 2.0, dtype=np.float32), (1, 8, 64), scale
    )

    assert value.shape == (1, 8, 64)
    assert value.dtype == np.int8
    assert saturated == 512


def test_restore_artifact_output_transposes_aries_singleton_channel_layout():
    """Catches treating ARIES [1,1,96] output as horizon-major data."""
    restored = restore_artifact_output(
        np.arange(96, dtype=np.float32).reshape(1, 1, 96), (1, 1, 96)
    )

    assert restored.shape == (1, 96, 1)
    assert restored[0, 95, 0] == 95.0

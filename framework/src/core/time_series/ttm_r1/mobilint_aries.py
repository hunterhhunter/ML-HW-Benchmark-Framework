"""Pure ARIES MXQ ABI conversion for the fixed TTM-R1 core."""

from __future__ import annotations

from typing import Any

import numpy as np


def quantize_core_input(
    core_input: np.ndarray, artifact_shape: tuple[int, ...], scale: Any
) -> tuple[np.ndarray, int]:
    """Convert prepared FP32 core data into ARIES's discovered int8 input ABI."""
    value = np.asarray(core_input, dtype=np.float32)
    if value.shape != (1, 512, 1) or not np.isfinite(value).all():
        raise ValueError("ARIES TTM-R1 core input must be finite [1,512,1] float32")
    if int(np.prod(artifact_shape)) != 512:
        raise ValueError(f"ARIES artifact input must contain 512 values: {artifact_shape}")
    if getattr(scale, "is_asymmetric", None) or getattr(scale, "zero_point", None) != 0:
        raise ValueError("ARIES TTM-R1 input scale must be symmetric with zero point 0")
    reshaped = value.reshape(artifact_shape)
    if getattr(scale, "is_uniform", None):
        multiplier = float(getattr(scale, "scale"))
    else:
        scale_list = np.asarray(getattr(scale, "scale_list", ()), dtype=np.float32)
        if scale_list.shape != (artifact_shape[-1],):
            raise ValueError("ARIES non-uniform input scale must match the last input axis")
        multiplier = scale_list.reshape((1,) * (len(artifact_shape) - 1) + (-1,))
    unbounded = np.rint(reshaped * multiplier)
    saturated = int(np.count_nonzero((unbounded < -128) | (unbounded > 127)))
    return np.ascontiguousarray(np.clip(unbounded, -128, 127).astype(np.int8)), saturated


def restore_artifact_output(raw_output: np.ndarray, artifact_shape: tuple[int, ...]) -> np.ndarray:
    """Restore ARIES's singleton-channel output ABI to TTM `[1,96,1]`."""
    output = np.asarray(raw_output, dtype=np.float32)
    if tuple(output.shape) != artifact_shape or int(np.prod(artifact_shape)) != 96:
        raise ValueError(f"ARIES artifact output must contain 96 values: {artifact_shape}")
    if artifact_shape == (1, 1, 96):
        restored = output.transpose(0, 2, 1)
    elif artifact_shape == (1, 96, 1):
        restored = output
    else:
        raise ValueError(f"unsupported ARIES TTM-R1 output layout: {artifact_shape}")
    if not np.isfinite(restored).all():
        raise ValueError("ARIES TTM-R1 output contains non-finite values")
    return np.ascontiguousarray(restored)

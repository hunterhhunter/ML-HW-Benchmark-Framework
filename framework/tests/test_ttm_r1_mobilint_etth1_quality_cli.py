import runpy
from types import SimpleNamespace

import numpy as np


def test_aries_runner_quantizes_input_and_restores_singleton_channel_output():
    """Catches treating ARIES quantized ABI as the original FP32 core ABI."""
    module = runpy.run_path("framework/tools/ttm_r1_mobilint_etth1_quality.py", run_name="not_main")
    captured = {}

    class Model:
        def get_model_input_shape(self): return [(1, 8, 64)]
        def get_model_output_shape(self): return [(1, 1, 96)]
        def get_input_scale(self): return [SimpleNamespace(scale=0.0, is_uniform=False, scale_list=[1.0] * 64, zero_point=0, is_asymmetric=False, zero_points=[])]
        def infer_to_float(self, inputs):
            captured["input"] = inputs[0]
            return [np.zeros((1, 1, 96), dtype=np.float32)]

    output, saturated = module["build_aries_runner"](Model())(np.ones((1, 512, 1), np.float32))
    assert captured["input"].shape == (1, 8, 64)
    assert captured["input"].dtype == np.int8
    assert output.shape == (1, 96, 1)
    assert saturated == 0

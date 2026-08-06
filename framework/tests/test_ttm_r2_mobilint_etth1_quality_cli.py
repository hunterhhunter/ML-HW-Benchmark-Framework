import runpy
from types import SimpleNamespace

import numpy as np


def test_r2_aries_runner_quantizes_runtime_abi_and_restores_forecast():
    module = runpy.run_path("framework/tools/ttm_r2_mobilint_etth1_quality.py", run_name="not_main")
    captured = {}

    class Model:
        def get_model_input_shape(self): return [(1, 8, 64)]
        def get_model_output_shape(self): return [(1, 1, 96)]
        def get_input_scale(self):
            return [SimpleNamespace(scale=0.0, is_uniform=False, scale_list=[1.0] * 64,
                                    zero_point=0, is_asymmetric=False, zero_points=[])]
        def infer_to_float(self, inputs):
            captured["input"] = inputs[0]
            return [np.zeros((1, 1, 96), dtype=np.float32)]

    output, saturated = module["build_aries_runner"](Model())(np.ones((1, 512, 1), np.float32))

    assert captured["input"].shape == (1, 8, 64)
    assert captured["input"].dtype == np.int8
    assert output.shape == (1, 96, 1)
    assert saturated == 0

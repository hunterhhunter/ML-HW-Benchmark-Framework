import numpy as np
import pytest

from core.model_spec import Model_Spec, Task
from evaluators import create_evaluator
from evaluators.ttm_r2_evaluator import TTMR2Evaluator


def _label(value: float = 7.0):
    return {
        "future_values": np.full((96, 1), value, dtype=np.float32),
        "loc": np.array([[1.0]], dtype=np.float32),
        "scale": np.array([[2.0]], dtype=np.float32),
        "model_loc": np.array([[1.0]], dtype=np.float32),
        "model_scale": np.array([[2.0]], dtype=np.float32),
    }


def test_evaluator_restores_twice_scaled_forecast_on_raw_etth1_scale():
    evaluator = TTMR2Evaluator()
    evaluator.add_batch(
        {"forecast": np.ones((1, 96, 1), dtype=np.float32)},
        [_label()],
        4.0,
    )

    assert evaluator.compute() == {
        "MAE": 0.0,
        "MSE": 0.0,
        "RMSE": 0.0,
        "Average Latency (ms)": 4.0,
        "P99 Latency (ms)": 4.0,
        "Samples/s": 250.0,
        "Total Samples": 1,
    }


def test_evaluator_rejects_missing_forecast_key():
    with pytest.raises(ValueError, match="forecast"):
        TTMR2Evaluator().add_batch(
            {"output": np.ones((1, 96, 1), dtype=np.float32)},
            [_label()],
            1.0,
        )


@pytest.mark.parametrize(
    ("forecast", "message"),
    [
        (np.ones((1, 95, 1), dtype=np.float32), "shape"),
        (np.ones((1, 96, 1), dtype=np.float64), "float32"),
        (
            np.full((1, 96, 1), np.nan, dtype=np.float32),
            "finite",
        ),
        (
            np.full((1, 96, 1), np.inf, dtype=np.float32),
            "finite",
        ),
    ],
)
def test_evaluator_rejects_invalid_forecast(forecast, message):
    with pytest.raises(ValueError, match=message):
        TTMR2Evaluator().add_batch({"forecast": forecast}, [_label()], 1.0)


def test_evaluator_rejects_batch_label_count_mismatch():
    with pytest.raises(ValueError, match="label count"):
        TTMR2Evaluator().add_batch(
            {"forecast": np.ones((2, 96, 1), dtype=np.float32)},
            [_label()],
            1.0,
        )


def test_evaluator_rejects_compute_without_samples():
    with pytest.raises(ValueError, match="no samples"):
        TTMR2Evaluator().compute()


def test_factory_routes_only_ttm_r2_to_raw_scale_evaluator():
    spec = Model_Spec(
        name="ttm-r2",
        task=Task.TIME_SERIES_FORECASTING,
        input_shapes={"past_values": (1, 512, 1)},
        input_dtype={"past_values": "float32"},
        output_shapes={"forecast": (1, 96, 1)},
    )

    assert isinstance(create_evaluator(spec), TTMR2Evaluator)

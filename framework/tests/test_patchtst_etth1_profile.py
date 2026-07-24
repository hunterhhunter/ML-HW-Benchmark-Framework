from core.model_profiles import SUPPORTED_PROFILES
from core.model_spec import Task


def test_patchtst_etth1_profile_matches_published_checkpoint_contract():
    profile = SUPPORTED_PROFILES["patchtst-etth1"]

    assert profile["task"] is Task.TIME_SERIES_FORECASTING
    assert profile["input_shapes"] == {
        "past_values": (1, 512, 7),
        "past_observed_mask": (1, 512, 7),
    }
    assert profile["input_dtype"] == {
        "past_values": "float32",
        "past_observed_mask": "bool",
    }
    assert profile["output_shapes"] == {"__auto__": (1, 96, 7)}
    assert (
        profile["default_torch_model_path"]
        == "models/ibm-granite_granite-timeseries-patchtst"
    )

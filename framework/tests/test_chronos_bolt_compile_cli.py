import json

import pytest

from tools.chronos_bolt_compile import main


def test_describe_exposes_external_contract_without_vendor_sdk(capsys):
    """Catches a discovery command that needs proprietary compiler imports."""
    assert main(["--vendor", "reference", "--describe"]) == 0

    described = json.loads(capsys.readouterr().out)
    assert described["external_input"] == {
        "name": "context",
        "shape": [1, 512],
        "dtype": "float32",
    }
    assert described["external_output"] == {
        "name": "quantile_preds",
        "shape": [1, 9, 64],
        "dtype": "float32",
    }
    assert described["core_input_names"] == [
        "input_embeds",
        "attention_mask",
        "decoder_input_embeds",
    ]
    assert described["core_inputs"] == [
        {"name": "input_embeds", "shape": [1, 33, 256], "dtype": "float32"},
        {"name": "attention_mask", "shape": [1, 33], "dtype": "float32"},
        {
            "name": "decoder_input_embeds",
            "shape": [1, 1, 256],
            "dtype": "float32",
        },
    ]


def test_reference_execution_requires_a_local_model_path(tmp_path):
    """Catches an accidental Hub download during a supposedly pinned run."""
    with pytest.raises(ValueError, match="--model-path"):
        main(["--vendor", "reference", "--output-dir", str(tmp_path)])

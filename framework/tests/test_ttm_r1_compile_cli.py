import json

import pytest

from tools import ttm_r1_compile


def test_describe_reports_the_fixed_public_contract(capsys):
    """Catches a discovery command that needs a vendor SDK or checkpoint."""
    assert ttm_r1_compile.main(["--vendor", "reference", "--describe"]) == 0

    described = json.loads(capsys.readouterr().out)
    assert described["external_input"] == {
        "name": "context",
        "shape": [1, 512, 1],
        "dtype": "float32",
    }
    assert described["external_output"] == {
        "name": "forecast",
        "shape": [1, 96, 1],
        "dtype": "float32",
    }
    assert described["core_inputs"] == [
        {"name": "past_values", "shape": [1, 512, 1], "dtype": "float32"}
    ]


def test_mobilint_dispatch_requires_model_and_output_paths():
    """Catches an accidental implicit checkpoint directory in an ARIES run."""
    with pytest.raises(ValueError, match="--model-path"):
        ttm_r1_compile.main(["--vendor", "mobilint"])


def test_reference_dispatches_to_the_reference_runner(monkeypatch, tmp_path, capsys):
    """Catches command-line arguments not reaching a reproducible runner."""
    captured = {}

    def fake_run(model_path, output_dir):
        captured["model_path"] = model_path
        captured["output_dir"] = output_dir
        return output_dir / "reference-result.json"

    monkeypatch.setattr(ttm_r1_compile, "run_reference", fake_run)

    assert ttm_r1_compile.main(
        [
            "--vendor",
            "reference",
            "--model-path",
            str(tmp_path / "model"),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    ) == 0
    assert captured == {
        "model_path": tmp_path / "model",
        "output_dir": tmp_path / "results",
    }
    assert "reference-result.json" in capsys.readouterr().out

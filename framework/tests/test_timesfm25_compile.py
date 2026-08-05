from tools.timesfm25_compile import describe_contract, main


def test_describe_contract_exposes_fixed_point_forecast_abi(capsys):
    assert describe_contract()["core_output"] == {
        "name": "point_forecast",
        "shape": [1, 128],
        "dtype": "float32",
    }
    assert main(["--vendor", "rbln", "--describe"]) == 0
    assert "normalized_context" in capsys.readouterr().out

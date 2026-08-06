from tools.ttm_r2_compile import describe_contract, main


def test_r2_cli_describes_the_fixed_contract_without_loading_sdk(capsys):
    assert describe_contract()["core_output"] == {
        "name": "forecast", "shape": [1, 96, 1], "dtype": "float32"
    }
    assert main(["--vendor", "rbln", "--describe"]) == 0
    assert "past_values" in capsys.readouterr().out

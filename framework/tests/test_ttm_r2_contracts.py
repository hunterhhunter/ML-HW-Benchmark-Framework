from ttm_r2.contracts import TTMR2Contract


def test_r2_contract_is_fixed_to_one_512_step_series_and_96_step_forecast():
    contract = TTMR2Contract.fixed()

    assert contract.external_input.shape == (1, 512, 1)
    assert contract.core_inputs[0].name == "past_values"
    assert contract.core_output.shape == (1, 96, 1)

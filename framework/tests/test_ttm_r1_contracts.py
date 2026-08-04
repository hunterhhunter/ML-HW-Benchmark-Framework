import pytest

from chronos_bolt.contracts import TensorContract
from ttm_r1.contracts import TTMR1Contract


def test_fixed_contract_exposes_univariate_512_to_96_abi():
    """Catches an ABI drift that would make vendor artifacts incomparable."""
    contract = TTMR1Contract.fixed()

    assert contract.external_input == TensorContract("context", (1, 512, 1), "float32")
    assert contract.external_output == TensorContract("forecast", (1, 96, 1), "float32")
    assert contract.core_inputs == (
        TensorContract("past_values", (1, 512, 1), "float32"),
    )
    assert contract.core_output == TensorContract("forecast", (1, 96, 1), "float32")


def test_contract_rejects_non_fixed_core_input():
    """Catches a caller trying to create a dynamic or multivariate benchmark ABI."""
    with pytest.raises(ValueError, match="past_values"):
        TTMR1Contract(
            external_input=TensorContract("context", (1, 512, 1), "float32"),
            external_output=TensorContract("forecast", (1, 96, 1), "float32"),
            core_inputs=(TensorContract("past_values", (1, 511, 1), "float32"),),
            core_output=TensorContract("forecast", (1, 96, 1), "float32"),
        )

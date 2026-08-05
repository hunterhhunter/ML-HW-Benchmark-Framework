import pytest

from timesfm25.contracts import TensorContract, TimesFM25Contract


def test_fixed_contract_uses_1024_context_and_native_128_horizon():
    contract = TimesFM25Contract.fixed()

    assert contract.external_input == TensorContract("context", (1, 1024), "float32")
    assert contract.core_inputs == (
        TensorContract("normalized_context", (1, 1024), "float32"),
    )
    assert contract.core_output == TensorContract("point_forecast", (1, 128), "float32")
    assert contract.external_output == contract.core_output


def test_tensor_contract_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="positive"):
        TensorContract("context", (1, 0), "float32")

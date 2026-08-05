import pytest
import torch

from timesfm25.reference import assert_public_split_parity


def test_reference_parity_rejects_mismatched_public_and_split_forecasts():
    with pytest.raises(AssertionError):
        assert_public_split_parity(torch.zeros((1, 128)), torch.ones((1, 128)))

import torch

from timesfm25.host_adapter import PreparedTimesFM25Inputs, TimesFM25HostAdapter


def test_restore_matches_flip_invariance_and_global_denormalization():
    prepared = PreparedTimesFM25Inputs(
        normalized_context=torch.zeros((1, 1024), dtype=torch.float32),
        flipped_context=torch.zeros((1, 1024), dtype=torch.float32),
        loc=torch.tensor([[2.0]], dtype=torch.float32),
        scale=torch.tensor([[3.0]], dtype=torch.float32),
        input_was_nonnegative=torch.tensor(False),
    )

    restored = prepared.restore(
        torch.full((1, 128), 4.0), torch.full((1, 128), -2.0)
    )

    assert torch.equal(restored, torch.full((1, 128), 11.0))


def test_restore_clamps_only_nonnegative_input_series():
    adapter = TimesFM25HostAdapter()
    prepared = adapter.prepare(torch.linspace(1.0, 2.0, 1024).reshape(1, 1024))

    restored = prepared.restore(torch.full((1, 128), -20.0), torch.zeros((1, 128)))

    assert torch.equal(restored, torch.zeros((1, 128)))

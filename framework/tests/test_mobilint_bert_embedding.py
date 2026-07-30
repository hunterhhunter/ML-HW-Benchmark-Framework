import sys
from pathlib import Path

import numpy as np
import pytest
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessor.mobilint_bert_embedding import (
    MobilintBertEmbeddingTransform,
)


def _weights(width: int = 4, max_positions: int = 8):
    word_embeddings = torch.zeros((8, width), dtype=torch.float32)
    word_embeddings[2] = torch.tensor(
        [1.0, 2.0, 3.0, 4.0], dtype=torch.float32
    )
    word_embeddings[3] = torch.tensor(
        [2.0, 2.0, 2.0, 2.0], dtype=torch.float32
    )
    token_type_embeddings = torch.zeros((2, width), dtype=torch.float32)
    token_type_embeddings[1] = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )
    return {
        "word_embeddings": word_embeddings,
        "token_type_embeddings": token_type_embeddings,
        "position_embeddings": torch.zeros(
            (max_positions, width), dtype=torch.float32
        ),
        "layernorm_weight": torch.ones((width,), dtype=torch.float32),
        "layernorm_bias": torch.zeros((width,), dtype=torch.float32),
    }


def _save_weights(tmp_path, values=None):
    path = tmp_path / "weight_dict.pth"
    torch.save(_weights() if values is None else values, path)
    return path


def test_transform_trims_padding_and_builds_float32_embedding(tmp_path):
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path),
        expected_width=4,
    )

    result = transform(
        {
            "input_ids": np.array([[2, 3, 0, 0]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 0, 0]], dtype=np.int64),
        }
    )["embeddings"]

    assert result.shape == (1, 2, 4)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    expected = np.array(
        [[[-1.3416408, -0.4472136, 0.4472136, 1.3416408], [0, 0, 0, 0]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_transform_preserves_unbatched_samples_and_token_types(tmp_path):
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path),
        expected_width=4,
    )

    result = transform(
        {
            "input_ids": np.array([3, 3, 0], dtype=np.int64),
            "attention_mask": np.array([1, 1, 0], dtype=np.int64),
            "token_type_ids": np.array([0, 1, 0], dtype=np.int64),
        }
    )["embeddings"]

    assert result.shape == (2, 4)
    np.testing.assert_allclose(result[0], np.zeros(4, dtype=np.float32))
    np.testing.assert_allclose(
        result[1],
        np.array([1.7320508, -0.57735026, -0.57735026, -0.57735026]),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "word_embeddings",
        "token_type_embeddings",
        "position_embeddings",
        "layernorm_weight",
        "layernorm_bias",
    ],
)
def test_transform_rejects_missing_weight_keys(tmp_path, missing_key):
    values = _weights()
    values.pop(missing_key)

    with pytest.raises(ValueError, match=missing_key):
        MobilintBertEmbeddingTransform(
            _save_weights(tmp_path, values),
            expected_width=4,
        )


def test_transform_rejects_incompatible_embedding_width(tmp_path):
    with pytest.raises(ValueError, match="embedding width"):
        MobilintBertEmbeddingTransform(
            _save_weights(tmp_path),
            expected_width=8,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"attention_mask": np.array([[1, 1]], dtype=np.int64)},
            "same shape",
        ),
        (
            {"attention_mask": np.array([[0, 0, 0]], dtype=np.int64)},
            "valid tokens",
        ),
        (
            {"attention_mask": np.array([[1, 0, 1]], dtype=np.int64)},
            "right-padded prefix",
        ),
        (
            {
                "input_ids": np.array([[2, 3, 0], [2, 3, 0]], dtype=np.int64),
                "attention_mask": np.array(
                    [[1, 1, 0], [1, 1, 0]], dtype=np.int64
                ),
            },
            "batch size 1",
        ),
    ],
)
def test_transform_rejects_invalid_token_batches(tmp_path, overrides, message):
    inputs = {
        "input_ids": np.array([[2, 3, 0]], dtype=np.int64),
        "attention_mask": np.array([[1, 1, 0]], dtype=np.int64),
    }
    inputs.update(overrides)
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path),
        expected_width=4,
    )

    with pytest.raises(ValueError, match=message):
        transform(inputs)


def test_transform_rejects_unexpected_token_input(tmp_path):
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path),
        expected_width=4,
    )

    with pytest.raises(ValueError, match="unexpected.*position_ids"):
        transform(
            {
                "input_ids": np.array([[2]], dtype=np.int64),
                "attention_mask": np.array([[1]], dtype=np.int64),
                "position_ids": np.array([[0]], dtype=np.int64),
            }
        )


def test_transform_rejects_token_id_outside_embedding_table(tmp_path):
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path),
        expected_width=4,
    )

    with pytest.raises(ValueError, match="input_ids.*range"):
        transform(
            {
                "input_ids": np.array([[8]], dtype=np.int64),
                "attention_mask": np.array([[1]], dtype=np.int64),
            }
        )


def test_transform_rejects_sequence_longer_than_position_table(tmp_path):
    values = _weights(max_positions=2)
    transform = MobilintBertEmbeddingTransform(
        _save_weights(tmp_path, values),
        expected_width=4,
    )

    with pytest.raises(ValueError, match="position embedding capacity"):
        transform(
            {
                "input_ids": np.array([[2, 3, 2]], dtype=np.int64),
                "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
            }
        )

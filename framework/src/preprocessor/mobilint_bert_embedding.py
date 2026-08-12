"""Host BERT embedding transform for Mobilint embedding-input MXQ files."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np


class MobilintBertEmbeddingTransform:
    """Convert token arrays to the float32 boundary compiled into BERT MXQ."""

    REQUIRED_KEYS = (
        "word_embeddings",
        "token_type_embeddings",
        "position_embeddings",
        "layernorm_weight",
        "layernorm_bias",
    )
    _INPUT_KEYS = frozenset(
        {"input_ids", "attention_mask", "token_type_ids"}
    )

    def __init__(self, weights_path: str | Path, *, expected_width: int = 768):
        if (
            isinstance(expected_width, bool)
            or not isinstance(expected_width, Integral)
            or expected_width <= 0
        ):
            raise ValueError("expected embedding width must be a positive integer")
        path = Path(weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"BERT embedding weights not found: {path}")

        import torch

        document = torch.load(path, map_location="cpu", weights_only=True)
        self.expected_width = int(expected_width)
        self._weights = self._validate_weights(document)

    def _validate_weights(self, document: Any) -> dict[str, Any]:
        if not isinstance(document, Mapping):
            raise ValueError("BERT embedding weights must be a mapping")
        missing = [name for name in self.REQUIRED_KEYS if name not in document]
        if missing:
            raise ValueError(
                "BERT embedding weights missing required key: " + missing[0]
            )

        import torch

        weights = {}
        for name in self.REQUIRED_KEYS:
            value = document[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"BERT embedding weight {name} must be a tensor")
            weights[name] = (
                value.detach().cpu().to(dtype=torch.float32).contiguous()
            )

        width = self.expected_width
        for name in (
            "word_embeddings",
            "token_type_embeddings",
            "position_embeddings",
        ):
            value = weights[name]
            if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != width:
                raise ValueError(
                    f"BERT embedding width mismatch for {name}: expected {width}"
                )
        for name in ("layernorm_weight", "layernorm_bias"):
            value = weights[name]
            if value.ndim != 1 or tuple(value.shape) != (width,):
                raise ValueError(
                    f"BERT embedding width mismatch for {name}: expected {width}"
                )
        return weights

    @staticmethod
    def _integer_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim not in {1, 2}:
            raise ValueError(f"{name} must be a rank-1 or rank-2 array")
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must use an integer dtype")
        return array

    def _normalize_token_inputs(self, inputs: Any):
        if not isinstance(inputs, Mapping):
            raise TypeError("BERT token inputs must be a mapping")
        unexpected = sorted(set(inputs).difference(self._INPUT_KEYS))
        if unexpected:
            raise ValueError(
                "unexpected BERT token input: " + ", ".join(unexpected)
            )
        missing = [
            name for name in ("input_ids", "attention_mask") if name not in inputs
        ]
        if missing:
            raise ValueError("missing BERT token input: " + ", ".join(missing))

        input_ids = self._integer_array(inputs["input_ids"], "input_ids")
        attention_mask = self._integer_array(
            inputs["attention_mask"], "attention_mask"
        )
        token_type_value = inputs.get("token_type_ids")
        token_type_ids = (
            np.zeros_like(input_ids, dtype=np.int64)
            if token_type_value is None
            else self._integer_array(token_type_value, "token_type_ids")
        )
        if not (
            input_ids.shape == attention_mask.shape == token_type_ids.shape
        ):
            raise ValueError(
                "input_ids, attention_mask, and token_type_ids must have the same shape"
            )

        was_unbatched = input_ids.ndim == 1
        if was_unbatched:
            input_ids = input_ids[np.newaxis, :]
            attention_mask = attention_mask[np.newaxis, :]
            token_type_ids = token_type_ids[np.newaxis, :]
        if input_ids.shape[0] != 1:
            raise ValueError("Mobilint BERT embedding transform requires batch size 1")

        import torch

        return (
            torch.from_numpy(np.ascontiguousarray(input_ids)).to(torch.long),
            torch.from_numpy(np.ascontiguousarray(attention_mask)).to(torch.long),
            torch.from_numpy(np.ascontiguousarray(token_type_ids)).to(torch.long),
            was_unbatched,
        )

    def _valid_prefix_length(self, attention_mask) -> int:
        mask = attention_mask[0]
        if bool(((mask != 0) & (mask != 1)).any()):
            raise ValueError("attention_mask must contain only zero or one")
        valid_tokens = int(mask.sum().item())
        if valid_tokens <= 0:
            raise ValueError("attention_mask contains no valid tokens")
        if not bool((mask[:valid_tokens] == 1).all()) or not bool(
            (mask[valid_tokens:] == 0).all()
        ):
            raise ValueError("attention_mask must be a right-padded prefix")
        return valid_tokens

    @staticmethod
    def _validate_index_range(values, *, size: int, name: str) -> None:
        if bool((values < 0).any()) or bool((values >= size).any()):
            raise ValueError(f"{name} values are outside the embedding table range")

    def __call__(self, inputs: Any) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as functional

        input_ids, attention_mask, token_type_ids, was_unbatched = (
            self._normalize_token_inputs(inputs)
        )
        valid_tokens = self._valid_prefix_length(attention_mask)
        if valid_tokens > self._weights["position_embeddings"].shape[0]:
            raise ValueError("sequence exceeds position embedding capacity")

        input_ids = input_ids[:, :valid_tokens]
        token_type_ids = token_type_ids[:, :valid_tokens]
        self._validate_index_range(
            input_ids,
            size=self._weights["word_embeddings"].shape[0],
            name="input_ids",
        )
        self._validate_index_range(
            token_type_ids,
            size=self._weights["token_type_embeddings"].shape[0],
            name="token_type_ids",
        )
        positions = torch.arange(valid_tokens, dtype=torch.long).unsqueeze(0)
        embedded = (
            functional.embedding(input_ids, self._weights["word_embeddings"])
            + functional.embedding(
                token_type_ids,
                self._weights["token_type_embeddings"],
            )
            + functional.embedding(
                positions,
                self._weights["position_embeddings"],
            )
        )
        embedded = functional.layer_norm(
            embedded,
            (self.expected_width,),
            weight=self._weights["layernorm_weight"],
            bias=self._weights["layernorm_bias"],
            eps=1e-12,
        ).to(dtype=torch.float32)
        array = embedded.numpy()
        if was_unbatched:
            array = array[0]
        return {"embeddings": np.ascontiguousarray(array)}

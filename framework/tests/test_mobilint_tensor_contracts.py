import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.mobilint_tensor_contracts import build_mobilint_tensor_contract
from core.model_spec import Model_Spec, Task


@pytest.mark.parametrize(
    ("spec", "max_batch_size", "expected"),
    [
        (
            Model_Spec(
                name="bert-base-uncased",
                task=Task.NLP_CLASSIFICATION,
                input_shapes={
                    "input_ids": (1, 128),
                    "attention_mask": (1, 128),
                },
                input_dtype={
                    "input_ids": "int64",
                    "attention_mask": "int64",
                },
                output_shapes={"logits": (1, 2)},
            ),
            1,
            {
                "artifact_profile_id": "mobilint-bert-base-uncased-tensor-v1",
                "expected_input_names": ["input_ids", "attention_mask"],
                "expected_input_dtypes": ["int64", "int64"],
                "expected_unbatched_input_shapes": [[128], [128]],
                "expected_output_names": ["logits"],
                "expected_unbatched_output_shapes": [[2]],
                "max_input_batch_size": 1,
                "native_async_supported": False,
            },
        ),
        (
            Model_Spec(
                name="bert-base-uncased-squad-v1",
                task=Task.QUESTION_ANSWERING,
                input_shapes={
                    "input_ids": (1, 384),
                    "attention_mask": (1, 384),
                    "token_type_ids": (1, 384),
                },
                input_dtype={
                    "input_ids": "int64",
                    "attention_mask": "int64",
                    "token_type_ids": "int64",
                },
                output_shapes={
                    "start_logits": (1, 384),
                    "end_logits": (1, 384),
                },
            ),
            4,
            {
                "artifact_profile_id": (
                    "mobilint-bert-base-uncased-squad-v1-tensor-v1"
                ),
                "expected_input_names": [
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                ],
                "expected_input_dtypes": ["int64", "int64", "int64"],
                "expected_unbatched_input_shapes": [[384], [384], [384]],
                "expected_output_names": ["start_logits", "end_logits"],
                "expected_unbatched_output_shapes": [[384], [384]],
                "max_input_batch_size": 4,
                "native_async_supported": False,
            },
        ),
        (
            Model_Spec(
                name="patchtst-fm-r1",
                task=Task.TIME_SERIES_FORECASTING,
                input_shapes={
                    "past_values": (1, 512, 7),
                    "past_observed_mask": (1, 512, 7),
                },
                input_dtype={
                    "past_values": "float32",
                    "past_observed_mask": "bool",
                },
                output_shapes={"output": (1, 96, 7)},
            ),
            1,
            {
                "artifact_profile_id": "mobilint-patchtst-fm-r1-tensor-v1",
                "expected_input_names": [
                    "past_values",
                    "past_observed_mask",
                ],
                "expected_input_dtypes": ["float32", "bool"],
                "expected_unbatched_input_shapes": [[512, 7], [512, 7]],
                "expected_output_names": ["output"],
                "expected_unbatched_output_shapes": [[96, 7]],
                "max_input_batch_size": 1,
                "native_async_supported": False,
            },
        ),
    ],
)
def test_static_transformer_contract_preserves_order_and_strips_declared_batch(
    spec,
    max_batch_size,
    expected,
):
    contract = build_mobilint_tensor_contract(
        spec,
        max_batch_size=max_batch_size,
    )

    assert contract.runtime_contract() == expected


@pytest.mark.parametrize("max_batch_size", [True, 0, -1, 1.5, "2"])
def test_static_transformer_contract_rejects_non_positive_exact_batch_capacity(
    max_batch_size,
):
    spec = Model_Spec(
        name="bert",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"logits": (1, 2)},
    )

    with pytest.raises(ValueError, match="max_batch_size"):
        build_mobilint_tensor_contract(spec, max_batch_size=max_batch_size)


def test_static_transformer_contract_rejects_shape_without_declared_batch_axis():
    spec = Model_Spec(
        name="bert",
        task=Task.NLP_CLASSIFICATION,
        input_shapes={"input_ids": ()},
        input_dtype={"input_ids": "int64"},
        output_shapes={"logits": (1, 2)},
    )

    with pytest.raises(ValueError, match="input_ids.*batch axis"):
        build_mobilint_tensor_contract(spec, max_batch_size=1)

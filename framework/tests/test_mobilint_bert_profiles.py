import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.mobilint_bert_profiles import (
    apply_mobilint_bert_profile,
    resolve_mobilint_bert_profile,
)
from core.model_spec import Model_Spec, Task


def _token_sst2_spec() -> Model_Spec:
    return Model_Spec(
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
        model_paths={"mxq": "/models/sst2.mxq"},
    )


def _token_squad_spec() -> Model_Spec:
    return Model_Spec(
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
        model_paths={"mxq": "/models/squad1.mxq"},
    )


def test_sst2_profile_declares_embedding_mxq_boundary():
    profile = resolve_mobilint_bert_profile(
        "bert-base-uncased",
        Task.NLP_CLASSIFICATION,
    )

    assert profile is not None
    adapted = apply_mobilint_bert_profile(_token_sst2_spec(), profile)

    assert profile.profile_id == "mobilint-bert-sst2-embedding-v1"
    assert profile.embedding_width == 768
    assert profile.max_batch_size == 1
    assert profile.native_async_supported is False
    assert adapted.input_shapes == {"embeddings": (1, -1, 768)}
    assert adapted.input_dtype == {"embeddings": "float32"}
    assert adapted.output_shapes == {"logits": (1, 2)}


def test_squad_profile_binds_verified_sdk_output_order():
    profile = resolve_mobilint_bert_profile(
        "bert-base-uncased-squad-v1",
        Task.QUESTION_ANSWERING,
    )

    assert profile is not None
    adapted = apply_mobilint_bert_profile(_token_squad_spec(), profile)

    assert profile.profile_id == "mobilint-bert-squad1-embedding-v1"
    assert profile.embedding_width == 768
    assert tuple(adapted.output_shapes) == ("end_logits", "start_logits")
    assert adapted.output_shapes == {
        "end_logits": (1, -1),
        "start_logits": (1, -1),
    }


def test_profile_resolution_rejects_model_task_mismatch():
    with pytest.raises(ValueError, match="task mismatch"):
        resolve_mobilint_bert_profile(
            "bert-base-uncased",
            Task.QUESTION_ANSWERING,
        )


def test_profile_resolution_ignores_unrelated_models():
    assert (
        resolve_mobilint_bert_profile(
            "patchtst-fm-r1",
            Task.TIME_SERIES_FORECASTING,
        )
        is None
    )


def test_applying_profile_preserves_identity_paths_and_source_spec():
    source = _token_sst2_spec()
    original_inputs = dict(source.input_shapes)
    profile = resolve_mobilint_bert_profile(source.name, source.task)

    assert profile is not None
    adapted = apply_mobilint_bert_profile(source, profile)

    assert adapted is not source
    assert adapted.name == source.name
    assert adapted.task is source.task
    assert adapted.model_paths == source.model_paths
    assert source.input_shapes == original_inputs


def test_profile_tensor_mappings_are_immutable_and_applied_defensively():
    source = _token_squad_spec()
    profile = resolve_mobilint_bert_profile(source.name, source.task)

    assert profile is not None
    with pytest.raises(TypeError):
        profile.output_shapes["end_logits"] = (1, 1)

    adapted = apply_mobilint_bert_profile(source, profile)
    adapted.output_shapes["end_logits"] = (1, 1)

    assert profile.output_shapes["end_logits"] == (1, -1)

"""Runtime-facing profiles for Mobilint BERT embedding-input MXQ files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .model_spec import Model_Spec, Task


@dataclass(frozen=True)
class MobilintBertArtifactProfile:
    model_name: str
    task: Task
    profile_id: str
    input_shapes: Mapping[str, tuple[int, ...]]
    input_dtype: Mapping[str, str]
    output_shapes: Mapping[str, tuple[int, ...]]
    embedding_width: int = 768
    max_batch_size: int = 1
    native_async_supported: bool = False


MOBILINT_BERT_SST2 = MobilintBertArtifactProfile(
    model_name="bert-base-uncased",
    task=Task.NLP_CLASSIFICATION,
    profile_id="mobilint-bert-sst2-embedding-v1",
    input_shapes=MappingProxyType({"embeddings": (1, -1, 768)}),
    input_dtype=MappingProxyType({"embeddings": "float32"}),
    output_shapes=MappingProxyType({"logits": (1, 2)}),
)


MOBILINT_BERT_SQUAD1 = MobilintBertArtifactProfile(
    model_name="bert-base-uncased-squad-v1",
    task=Task.QUESTION_ANSWERING,
    profile_id="mobilint-bert-squad1-embedding-v1",
    input_shapes=MappingProxyType({"embeddings": (1, -1, 768)}),
    input_dtype=MappingProxyType({"embeddings": "float32"}),
    output_shapes=MappingProxyType(
        {
            "end_logits": (1, -1),
            "start_logits": (1, -1),
        }
    ),
)


_PROFILES = {
    profile.model_name: profile
    for profile in (MOBILINT_BERT_SST2, MOBILINT_BERT_SQUAD1)
}


def resolve_mobilint_bert_profile(
    model_name: str,
    task: Task,
) -> MobilintBertArtifactProfile | None:
    """Resolve a supported BERT MXQ profile without matching other models."""
    profile = _PROFILES.get(str(model_name).strip())
    if profile is None:
        return None
    if task is not profile.task:
        raise ValueError(
            f"Mobilint BERT task mismatch for {profile.model_name!r}: "
            f"expected {profile.task.name}, received {getattr(task, 'name', task)!r}."
        )
    return profile


def apply_mobilint_bert_profile(
    spec: Model_Spec,
    profile: MobilintBertArtifactProfile,
) -> Model_Spec:
    """Return a runtime-facing spec for an embedding-input BERT MXQ."""
    if spec.name != profile.model_name or spec.task is not profile.task:
        raise ValueError(
            "Mobilint BERT profile does not match the selected ModelSpec."
        )
    return Model_Spec(
        name=spec.name,
        task=spec.task,
        input_shapes=dict(profile.input_shapes),
        input_dtype=dict(profile.input_dtype),
        output_shapes=dict(profile.output_shapes),
        model_paths=dict(spec.model_paths),
    )

from pathlib import Path

import pytest

from core.compiled_model import CompiledModel
from core.model_spec import Model_Spec, Task


def _spec() -> Model_Spec:
    return Model_Spec(
        name="llama",
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 8)},
        input_dtype={"input_ids": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"hf_model": "furiosa-ai/Llama-3.1-8B-Instruct"},
    )


def test_compiled_model_accepts_runtime_resolved_artifact():
    compiled = CompiledModel(_spec(), "furiosa_llm", None)

    assert compiled.artifact_path is None


def test_compiled_model_still_rejects_missing_local_artifact(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="compiled artifact does not exist"):
        CompiledModel(_spec(), "furiosa_llm", tmp_path / "missing.fxb")

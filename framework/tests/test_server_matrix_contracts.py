"""Contracts needed by the server's combined quality and runtime-only matrix."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
from core.targets import get_target
from preprocessor.strategies import SQuADPreprocessStrategy


def test_llama_prompt_limit_is_a_cli_option():
    parser = benchmark_main.build_parser()
    assert any("--max-prompt-length" in action.option_strings for action in parser._actions)


def test_common_llama_preprocessor_uses_assistant_chat_template():
    class Tokenizer:
        pad_token_id = 0

        def __init__(self):
            self.messages = []

        def apply_chat_template(self, messages, **kwargs):
            self.messages.append((messages, kwargs))
            return {
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
            }

    strategy = SQuADPreprocessStrategy.__new__(SQuADPreprocessStrategy)
    strategy.tokenizer = Tokenizer()
    strategy.max_length = 480
    strategy.tokenize("Question?", "Context.")
    assert strategy.tokenizer.messages
    messages, kwargs = strategy.tokenizer.messages[-1]
    assert messages[0]["role"] == "user"
    assert kwargs["add_generation_prompt"] is True


@pytest.mark.parametrize(
    "target_id", ["furiosa-rngd-torch", "rbln-static", "mobilint-aries"]
)
def test_ttm_r2_async_offline_is_admitted_for_verified_targets(
    tmp_path, monkeypatch, target_id
):
    checkpoint = tmp_path / "ibm-granite_granite-timeseries-ttm-r2"
    checkpoint.mkdir()
    artifact = checkpoint if target_id == "furiosa-rngd-torch" else tmp_path / (
        "model.rbln" if target_id == "rbln-static" else "model.mxq"
    )
    if artifact != checkpoint:
        artifact.write_bytes(b"artifact")
    monkeypatch.setattr(benchmark_main, "validate_dataset", lambda path: "dataset-sha")
    monkeypatch.setattr(benchmark_main, "validate_checkpoint", lambda path: {})
    args = SimpleNamespace(
        batch_size=1,
        inference_mode="async_queue",
        dataset=str(tmp_path / "ETTh1.csv"),
    )
    benchmark_main.validate_ttm_r2_execution(args, get_target(target_id), artifact)

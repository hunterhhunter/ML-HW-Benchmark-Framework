"""Contracts needed by the server's combined quality and runtime-only matrix."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import main as benchmark_main
from core.targets import get_target
from dataloader.llama_loader import LlamaLoader
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


def test_llama_tokenizer_without_native_template_gets_explicit_common_template():
    class Tokenizer:
        chat_template = None
        bos_token = "<|begin_of_text|>"

        def __init__(self):
            self.template = None

        def apply_chat_template(self, messages, **kwargs):
            self.template = kwargs.get("chat_template")
            if self.template is None:
                raise ValueError("tokenizer.chat_template is not set")
            return {
                "input_ids": np.asarray([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
            }

    strategy = SQuADPreprocessStrategy.__new__(SQuADPreprocessStrategy)
    strategy.tokenizer = Tokenizer()
    strategy.max_length = 480
    strategy.tokenize("Question?", "Context.")
    assert "<|start_header_id|>" in strategy.tokenizer.template
    assert "assistant" in strategy.tokenizer.template


def test_llama_prompt_limit_uses_effective_vllm_profile_default():
    args = SimpleNamespace(
        max_new_tokens=256,
        max_model_len=None,
        max_prompt_length=None,
        backend="vllm",
    )
    assert benchmark_main._resolve_llama_prompt_length(
        args, profile={"default_max_model_len": 4096}
    ) == 3840


def test_llama_prompt_limit_uses_runtime_option_override():
    args = SimpleNamespace(
        max_new_tokens=32,
        max_model_len=None,
        max_prompt_length=480,
        backend="vllm",
    )
    with pytest.raises(ValueError, match="must not exceed max_model_len"):
        benchmark_main._resolve_llama_prompt_length(
            args,
            profile={"default_max_model_len": 4096},
            runtime_options={"max_model_len": 500},
        )


def test_chat_turn_terminator_is_an_explicit_llama_stop_token():
    class Tokenizer:
        eos_token_id = 128001
        unk_token_id = None

        def encode(self, text, add_special_tokens=False):
            return [198 if text == "\n" else 271]

        def convert_tokens_to_ids(self, token):
            assert token == "<|eot_id|>"
            return 128009

    loader = LlamaLoader.__new__(LlamaLoader)
    loader.preprocessor = SimpleNamespace(tokenizer=Tokenizer())
    assert 128009 in loader._build_stop_token_ids()


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

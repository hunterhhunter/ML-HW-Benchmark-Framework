import json
import os
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.compiled_model import CompiledModel
from core.async_inference import AsyncInferenceConfig, RunStatus
from core.inference_engine import InferenceEngine
from core.model_spec import Model_Spec, Task
from core.runtime_executor import NativeAsyncRuntimeExecutor
from runtimes.rbln_vllm_rt import RblnVllmRuntime

GIB = 1024**3


class _Completion:
    def __init__(self, token_ids):
        self.token_ids = list(token_ids)


class _RequestOutput:
    def __init__(self, token_ids, metrics=None):
        self.outputs = [_Completion(token_ids)]
        self.metrics = metrics


def _prepared_model(
    tmp_path: Path,
    model_name: str = "llama-3.2-3b",
    *,
    manifest_num_devices: int | None = None,
    manifest_batch_size: int = 1,
    manifest_max_seq_len: int = 512,
    manifest_block_size: int = 512,
    manifest_decoder_batch_sizes=(1,),
) -> Path:
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "_name_or_path": (
            "meta-llama/Llama-3.1-8B-Instruct"
            if "3.1-8b" in model_name
            else "meta-llama/Llama-3.2-3B-Instruct"
        ),
        "torch_dtype": "bfloat16",
    }
    (model_dir / "config.json").write_text(
        json.dumps(config), encoding="utf-8"
    )
    (model_dir / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    artifact_dir = model_dir / "rbln_model"
    artifact_dir.mkdir()
    (artifact_dir / "decoder.rbln").write_bytes(b"compiled")
    if manifest_num_devices is not None:
        (model_dir / "rbln-vllm-manifest.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "model_id": config["_name_or_path"],
                    "num_devices": manifest_num_devices,
                    "batch_size": manifest_batch_size,
                    "max_seq_len": manifest_max_seq_len,
                    "block_size": manifest_block_size,
                    "decoder_batch_sizes": list(
                        manifest_decoder_batch_sizes
                    ),
                }
            ),
            encoding="utf-8",
        )
    return model_dir


def _compiled(model_dir: Path, model_name: str) -> CompiledModel:
    spec = Model_Spec(
        name=model_name,
        task=Task.NLP_GENERATION,
        input_shapes={"input_ids": (1, 16), "attention_mask": (1, 16)},
        input_dtype={"input_ids": "int64", "attention_mask": "int64"},
        output_shapes={"generated_ids": (1, 4)},
        model_paths={"rbln_llm_dir": str(model_dir)},
    )
    return CompiledModel(spec, "rbln_vllm", model_dir)


def _inventory(count: int, memory_bytes: int = 16 * GIB):
    return {
        "KMD_version": "3.2.2",
        "devices": [
            {
                "npu": index,
                "name": "RBLN-CA22",
                "status": "normal",
                "memory": {"used": "0", "total": str(memory_bytes)},
            }
            for index in range(count)
        ],
        "contexts": [],
    }


def _runtime(
    *,
    num_devices=1,
    allow_single=False,
    inventory_count=None,
    **kwargs,
):
    if inventory_count is None:
        inventory_count = num_devices
    options = {
        "block_size": 512,
        "max_model_len": 512,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "num_devices": num_devices,
        "allow_unsupported_single_npu": allow_single,
        "inventory_provider": lambda: _inventory(inventory_count),
    }
    options.update(kwargs)
    return RblnVllmRuntime(**options)


def _install_fake_vllm(monkeypatch, generated_tokens=((31, 32), (41,))):
    state = {
        "llm_init": [],
        "generate": [],
        "shutdowns": 0,
        "env_at_init": [],
    }

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs):
            state["llm_init"].append(kwargs)
            state["env_at_init"].append(
                os.environ.get("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK")
            )

        def generate(self, prompts, sampling_params=None):
            state["generate"].append(
                {"prompts": prompts, "sampling_params": sampling_params}
            )
            outputs = []
            for index, _ in enumerate(prompts if isinstance(prompts, list) else [prompts]):
                tokens = generated_tokens[min(index, len(generated_tokens) - 1)]
                outputs.append(
                    _RequestOutput(
                        tokens,
                        metrics={
                            "arrival_time": 10.0,
                            "first_token_time": 10.002,
                            "last_token_time": 10.006,
                        },
                    )
                )
            return outputs

        def shutdown(self):
            state["shutdowns"] += 1

    module = types.ModuleType("vllm")
    module.LLM = LLM
    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return state


def _install_fake_async_vllm(
    monkeypatch,
    *,
    streams=None,
    failure: BaseException | None = None,
    wait_before_init: threading.Event | None = None,
    wait_before_finish: threading.Event | None = None,
):
    streams = streams or (((51,), (51, 52), (51, 52, 53)),)
    state = {
        "engine_args": [],
        "engine_init": [],
        "generate": [],
        "aborts": [],
        "shutdowns": 0,
        "env_at_init": [],
    }

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AsyncEngineArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["engine_args"].append(kwargs)

    class AsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, engine_args):
            if wait_before_init is not None:
                wait_before_init.wait()
            state["engine_init"].append(engine_args)
            state["env_at_init"].append(
                os.environ.get("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK")
            )
            return cls()

        async def generate(self, prompt, sampling_params, request_id):
            call_index = len(state["generate"])
            state["generate"].append(
                {
                    "prompt": prompt,
                    "sampling_params": sampling_params,
                    "request_id": request_id,
                }
            )
            if failure is not None:
                raise failure
            request_stream = streams[min(call_index, len(streams) - 1)]
            for token_ids in request_stream:
                yield _RequestOutput(token_ids)
            if wait_before_finish is not None:
                while not wait_before_finish.is_set():
                    await __import__("asyncio").sleep(0.005)

        async def abort(self, request_id):
            state["aborts"].append(request_id)

        def shutdown(self):
            state["shutdowns"] += 1

    module = types.ModuleType("vllm")
    module.AsyncEngineArgs = AsyncEngineArgs
    module.AsyncLLMEngine = AsyncLLMEngine
    module.SamplingParams = SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", module)
    return state


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"block_size": 0}, "block_size"),
        ({"block_size": 512, "max_model_len": 1000}, "divide"),
        ({"block_size": 512, "tensor_parallel_size": 2}, "fixed at 1"),
        ({"block_size": 512, "num_devices": True}, "num_devices"),
        ({"block_size": 512, "max_num_seqs": 0}, "max_num_seqs"),
        (
            {
                "block_size": 512,
                "max_num_seqs": 2,
                "decoder_batch_sizes": "1,3",
            },
            "decoder_batch_sizes",
        ),
    ],
)
def test_runtime_rejects_invalid_engine_options(options, message):
    with pytest.raises(ValueError, match=message):
        RblnVllmRuntime(**options)


def test_load_rejects_unopted_single_npu_llama_3_2_3b(tmp_path):
    model_dir = _prepared_model(tmp_path)
    runtime = _runtime(allow_single=False)

    with pytest.raises(ValueError, match="allow_unsupported_single_npu"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_accepts_explicit_single_npu_llama_3_2_3b_experiment(tmp_path):
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)

    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    spec = runtime.get_device_spec()
    assert spec["num_devices"] == 1
    assert spec["support_classification"] == "unsupported_single_npu_experiment"
    assert spec["available_devices"] == 1


def test_load_rejects_single_npu_llama_3_1_8b_before_sdk_import(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "vllm", None)
    model_dir = _prepared_model(tmp_path, "llama-3.1-8b")
    runtime = _runtime(allow_single=True)

    with pytest.raises(ValueError, match="cannot fit"):
        runtime.load(_compiled(model_dir, "llama-3.1-8b"))


def test_load_rejects_cli_model_alias_that_conflicts_with_artifact_identity(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "vllm", None)
    model_dir = _prepared_model(
        tmp_path,
        "llama-3.1-8b",
        manifest_num_devices=1,
    )
    runtime = _runtime(allow_single=True)

    with pytest.raises(ValueError, match="identity mismatch"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_accepts_official_eight_npu_contract(tmp_path):
    model_dir = _prepared_model(
        tmp_path, "llama-3.1-8b", manifest_num_devices=8
    )
    runtime = _runtime(num_devices=8)

    runtime.load(_compiled(model_dir, "llama-3.1-8b"))

    assert runtime.get_device_spec()["support_classification"] == "official"


def test_load_rejects_insufficient_device_inventory(tmp_path):
    model_dir = _prepared_model(tmp_path, manifest_num_devices=8)
    runtime = _runtime(num_devices=8, inventory_count=1)

    with pytest.raises(RuntimeError, match="requested 8.*found 1"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_rejects_manifest_device_count_mismatch(tmp_path):
    model_dir = _prepared_model(tmp_path, manifest_num_devices=8)
    runtime = _runtime(num_devices=1, allow_single=True)

    with pytest.raises(ValueError, match="compiled for 8.*requested 1"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


@pytest.mark.parametrize(
    ("prepared_kwargs", "runtime_kwargs", "message"),
    [
        ({"manifest_block_size": 1024}, {}, "block_size"),
        ({"manifest_max_seq_len": 8192}, {}, "max_model_len"),
        (
            {
                "manifest_batch_size": 4,
                "manifest_decoder_batch_sizes": (1, 4),
            },
            {},
            "max_num_seqs",
        ),
        (
            {
                "manifest_batch_size": 4,
                "manifest_decoder_batch_sizes": (1, 4),
            },
            {"max_num_seqs": 4, "decoder_batch_sizes": "1,2,4"},
            "decoder_batch_sizes",
        ),
    ],
)
def test_load_rejects_manifest_runtime_contract_mismatch(
    tmp_path, prepared_kwargs, runtime_kwargs, message
):
    model_dir = _prepared_model(
        tmp_path,
        manifest_num_devices=1,
        **prepared_kwargs,
    )
    runtime = _runtime(allow_single=True, **runtime_kwargs)

    with pytest.raises(ValueError, match=message):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_uses_manifest_max_seq_len_when_runtime_omits_it(tmp_path):
    model_dir = _prepared_model(
        tmp_path,
        manifest_num_devices=1,
        manifest_max_seq_len=512,
        manifest_block_size=512,
    )
    runtime = _runtime(allow_single=True, max_model_len=None)

    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    assert runtime.max_model_len == 512


@pytest.mark.parametrize("max_model_len", [None, 2048])
def test_load_rejects_single_npu_context_outside_experimental_contract(
    tmp_path, max_model_len
):
    model_dir = _prepared_model(tmp_path)
    runtime = _runtime(
        allow_single=True,
        max_model_len=max_model_len,
    )

    with pytest.raises(ValueError, match="max_model_len.*1024"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_rejects_single_npu_batch_greater_than_one_without_manifest(
    tmp_path,
):
    model_dir = _prepared_model(tmp_path)
    runtime = _runtime(
        allow_single=True,
        max_model_len=512,
        max_num_seqs=2,
    )

    with pytest.raises(ValueError, match="max_num_seqs.*exactly 1"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_rejects_unreadable_single_npu_memory_capacity(tmp_path):
    model_dir = _prepared_model(tmp_path)
    inventory = _inventory(1)
    inventory["devices"][0]["memory"]["total"] = "unknown"
    runtime = _runtime(
        allow_single=True,
        max_model_len=512,
        inventory_provider=lambda: inventory,
    )

    with pytest.raises(ValueError, match="readable device memory"):
        runtime.load(_compiled(model_dir, "llama-3.2-3b"))


def test_load_validates_artifact_without_importing_vllm(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "vllm", None)
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)

    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    assert runtime.compiled_model is not None


def test_sync_engine_scopes_device_environment_and_forwards_options(
    monkeypatch, tmp_path
):
    state = _install_fake_vllm(monkeypatch, generated_tokens=((7, 8),))
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(
        allow_single=True,
        decoder_batch_sizes="1",
        dtype="bfloat16",
        seed=17,
        trust_remote_code=True,
    )
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    monkeypatch.setenv("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", "previous")

    result = runtime.generate(
        {
            "input_ids": np.asarray([[0, 11, 12, 0]], dtype=np.int64),
            "attention_mask": np.asarray([[0, 1, 1, 0]], dtype=np.int64),
        },
        max_new_tokens=2,
        stop_token_ids=(2, 3),
    )

    assert state["env_at_init"] == ["1"]
    assert os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] == "previous"
    assert state["llm_init"] == [
        {
            "model": str(model_dir),
            "tokenizer": str(model_dir),
            "block_size": 512,
            "max_model_len": 512,
            "max_num_seqs": 1,
            "tensor_parallel_size": 1,
            "dtype": "bfloat16",
            "seed": 17,
            "trust_remote_code": True,
            "additional_config": {
                "rbln_config": {"decoder_batch_sizes": [1]}
            },
        }
    ]
    assert state["generate"][0]["prompts"] == [
        {"prompt_token_ids": [11, 12]}
    ]
    assert state["generate"][0]["sampling_params"].kwargs == {
        "max_tokens": 2,
        "temperature": 0.0,
        "stop_token_ids": [2, 3],
    }
    np.testing.assert_array_equal(result.generated_ids, [[7, 8]])
    np.testing.assert_array_equal(result.generated_lengths, [2])
    assert result.num_tokens == 2
    assert result.ttft_ms == pytest.approx(2.0)
    assert result.tpot_ms == pytest.approx(4.0)
    assert result.timing_source == "vllm_request_metrics"


def test_sync_engine_forwards_separate_tokenizer_directory(
    monkeypatch, tmp_path
):
    state = _install_fake_vllm(monkeypatch, generated_tokens=((7,),))
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    runtime = _runtime(
        allow_single=True,
        tokenizer_path=str(tokenizer_dir),
    )
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    runtime.generate({"input_ids": np.asarray([[1]], dtype=np.int64)})

    assert state["llm_init"][0]["tokenizer"] == str(
        tokenizer_dir.resolve()
    )
    runtime.unload()


def test_sync_generation_handles_left_and_right_padding_batches(
    monkeypatch, tmp_path
):
    state = _install_fake_vllm(monkeypatch)
    model_dir = _prepared_model(
        tmp_path,
        manifest_num_devices=8,
        manifest_batch_size=2,
        manifest_decoder_batch_sizes=(1, 2),
    )
    runtime = _runtime(num_devices=8, max_num_seqs=2)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    result = runtime.generate(
        {
            "input_ids": np.asarray(
                [[0, 0, 11, 12], [21, 22, 0, 0]], dtype=np.int64
            ),
            "attention_mask": np.asarray(
                [[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.int64
            ),
        },
        max_new_tokens=3,
    )

    assert state["generate"][0]["prompts"] == [
        {"prompt_token_ids": [11, 12]},
        {"prompt_token_ids": [21, 22]},
    ]
    np.testing.assert_array_equal(result.generated_ids, [[31, 32], [41, 0]])
    np.testing.assert_array_equal(result.generated_lengths, [2, 1])


def test_sync_generation_rejects_bad_inputs_before_engine_submission(
    monkeypatch, tmp_path
):
    state = _install_fake_vllm(monkeypatch)
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    with pytest.raises(ValueError, match="attention_mask shape"):
        runtime.generate(
            {
                "input_ids": np.ones((1, 4), dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
            }
        )

    assert state["llm_init"] == []


def test_unload_shuts_down_sync_engine_once(monkeypatch, tmp_path):
    state = _install_fake_vllm(monkeypatch, generated_tokens=((1,),))
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    runtime.generate({"input_ids": np.asarray([[1]], dtype=np.int64)})

    runtime.unload()
    runtime.unload()

    assert state["shutdowns"] == 1
    assert runtime.compiled_model is None


def test_native_async_stream_emits_cumulative_token_observation(
    monkeypatch, tmp_path
):
    state = _install_fake_async_vllm(monkeypatch)
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    monkeypatch.setenv("VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK", "previous")

    backend = runtime.create_native_backend(
        max_new_tokens=3,
        stop_token_ids=(2,),
    )
    event = threading.Event()
    outcomes = []
    job_id = backend.submit_async(
        {
            "input_ids": np.asarray([[0, 11, 12]], dtype=np.int64),
            "attention_mask": np.asarray([[0, 1, 1]], dtype=np.int64),
        },
        lambda outcome: (outcomes.append(outcome), event.set()),
    )

    assert event.wait(2.0)
    assert job_id.startswith("rbln-vllm-")
    assert state["env_at_init"] == ["1"]
    assert os.environ["VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK"] == "previous"
    assert state["engine_args"] == [runtime._engine_kwargs()]
    assert state["generate"][0]["prompt"] == {
        "prompt_token_ids": [11, 12]
    }
    assert state["generate"][0]["sampling_params"].kwargs == {
        "max_tokens": 3,
        "temperature": 0.0,
        "stop_token_ids": [2],
    }
    assert len(outcomes) == 1
    outcome = outcomes[0]
    np.testing.assert_array_equal(outcome.outputs["generated_ids"], [[51, 52, 53]])
    np.testing.assert_array_equal(outcome.outputs["generated_lengths"], [3])
    assert outcome.generated_tokens == 3
    assert outcome.error_type is None
    assert outcome.timing_ms["ttft_ms"] >= 0.0
    assert outcome.timing_ms["tpot_ms"] >= 0.0
    assert outcome.generation_observation.source == (
        "rbln_vllm_async_python_stream"
    )
    assert [
        item.cumulative_tokens
        for item in outcome.generation_observation.events
    ] == [1, 2, 3]

    runtime.unload()
    assert state["shutdowns"] == 1


def test_native_async_warmup_uses_the_existing_async_engine(
    monkeypatch, tmp_path
):
    state = _install_fake_async_vllm(
        monkeypatch,
        streams=(((61,),), ((62,),)),
    )
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    runtime.create_native_backend(max_new_tokens=16)

    runtime.warmup(
        {"input_ids": np.asarray([[1, 2]], dtype=np.int64)},
        num_runs=2,
    )

    assert len(state["engine_init"]) == 1
    assert len(state["generate"]) == 2
    assert [
        call["sampling_params"].kwargs["max_tokens"]
        for call in state["generate"]
    ] == [1, 1]
    runtime.unload()


def test_native_async_runner_completes_warmup_and_measurement_on_one_engine(
    monkeypatch, tmp_path
):
    state = _install_fake_async_vllm(
        monkeypatch,
        streams=(((71,),), ((72,),), ((73,),)),
    )
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    backend = runtime.create_native_backend(max_new_tokens=2)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=2.0,
    )

    class Loader:
        def __init__(self):
            self.current_idx = 0
            self.samples = [
                {
                    "input": {
                        "input_ids": np.asarray([1, 2], dtype=np.int64)
                    },
                    "label": "first",
                },
                {
                    "input": {
                        "input_ids": np.asarray([3, 4], dtype=np.int64)
                    },
                    "label": "second",
                },
            ]

        def get_metadata(self):
            return {"total_samples": 2, "is_static_batched": False}

        def load_batch(self, batch_size):
            start = self.current_idx
            end = min(start + batch_size, len(self.samples))
            self.current_idx = end
            return self.samples[start:end]

        def load_by_index(self, index):
            return self.samples[index]

    class Evaluator:
        def __init__(self):
            self.total = 0

        def add_batch(self, outputs, labels, timing_ms):
            self.total += len(labels)

        def compute(self):
            return {"Total Samples": self.total}

    result = InferenceEngine(
        Loader(),
        runtime,
        Evaluator(),
        runtime_executor=executor,
    ).run_async(
        AsyncInferenceConfig(
            queue_capacity=1,
            worker_count=1,
            max_batch_size=1,
            batch_timeout_ms=0,
            submit_timeout_sec=1.0,
            flush_timeout_sec=2.0,
            min_samples=2,
            max_samples=2,
        ),
        warmup_runs=1,
    )

    assert result.status is RunStatus.VALID
    assert result.metrics["async_completed_samples"] == 2
    assert len(state["engine_init"]) == 1
    assert len(state["generate"]) == 3
    assert executor.shutdown(timeout=1.0) is True
    runtime.unload()


def test_native_async_failure_aborts_and_callbacks_once(monkeypatch, tmp_path):
    state = _install_fake_async_vllm(
        monkeypatch, failure=RuntimeError("vendor secret detail")
    )
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    backend = runtime.create_native_backend(max_new_tokens=2)
    event = threading.Event()
    outcomes = []

    backend.submit_async(
        {"input_ids": np.asarray([[1]], dtype=np.int64)},
        lambda outcome: (outcomes.append(outcome), event.set()),
    )

    assert event.wait(2.0)
    assert len(outcomes) == 1
    assert outcomes[0].error_type == "RuntimeError"
    assert outcomes[0].error_message == "RBLN vLLM async generation failed"
    assert len(state["aborts"]) == 1
    runtime.unload()


def test_native_async_integrates_with_common_executor(monkeypatch, tmp_path):
    _install_fake_async_vllm(monkeypatch, streams=(((7,), (7, 8)),))
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    backend = runtime.create_native_backend(max_new_tokens=2)
    executor = NativeAsyncRuntimeExecutor(
        backend,
        max_inflight=1,
        completion_timeout_sec=2.0,
    )

    execution = executor.execute(
        {"input_ids": np.asarray([[1, 2]], dtype=np.int64)}
    )
    executor.acknowledge(execution)

    assert execution.error_type is None
    np.testing.assert_array_equal(execution.outputs["generated_ids"], [[7, 8]])
    assert execution.generated_tokens == 2
    assert executor.snapshot().inflight == 0
    assert executor.shutdown(timeout=1.0) is True
    runtime.unload()


def test_sync_and_async_engine_modes_are_mutually_exclusive(
    monkeypatch, tmp_path
):
    _install_fake_vllm(monkeypatch, generated_tokens=((1,),))
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    runtime.generate({"input_ids": np.asarray([[1]], dtype=np.int64)})

    with pytest.raises(RuntimeError, match="cannot be mixed"):
        runtime.create_native_backend(max_new_tokens=1)

    runtime.unload()


def test_native_async_shutdown_timeout_preserves_live_runtime(
    monkeypatch, tmp_path
):
    release = threading.Event()
    _install_fake_async_vllm(
        monkeypatch,
        streams=(((9,),),),
        wait_before_finish=release,
    )
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(allow_single=True, shutdown_timeout_sec=0.001)
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))
    backend = runtime.create_native_backend(max_new_tokens=2)
    backend.submit_async(
        {"input_ids": np.asarray([[1]], dtype=np.int64)},
        lambda outcome: None,
    )
    deadline = time.monotonic() + 1.0
    while not backend.inflight_count and time.monotonic() < deadline:
        time.sleep(0.005)

    with pytest.raises(RuntimeError, match="did not stop"):
        runtime.unload()

    assert runtime.compiled_model is not None
    assert runtime._native_backend is backend
    release.set()
    assert backend.shutdown(timeout=2.0) is True
    runtime.unload()


def test_native_async_startup_timeout_preserves_backend_until_init_finishes(
    monkeypatch, tmp_path
):
    release = threading.Event()
    state = _install_fake_async_vllm(
        monkeypatch,
        wait_before_init=release,
    )
    model_dir = _prepared_model(tmp_path, manifest_num_devices=1)
    runtime = _runtime(
        allow_single=True,
        startup_timeout_sec=0.001,
        shutdown_timeout_sec=0.001,
    )
    runtime.load(_compiled(model_dir, "llama-3.2-3b"))

    with pytest.raises(TimeoutError, match="failed to start"):
        runtime.create_native_backend(max_new_tokens=2)

    backend = runtime._native_backend
    assert backend is not None
    assert backend._thread.is_alive()

    release.set()
    deadline = time.monotonic() + 2.0
    while backend._thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.005)

    assert backend._thread.is_alive() is False
    runtime.unload()
    assert state["shutdowns"] == 1

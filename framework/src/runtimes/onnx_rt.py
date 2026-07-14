import numpy as np
from typing import Dict, Any

from utils.cuda_preload import preload_cuda_libs, check_onnxruntime_gpu

# onnxruntime import 전에 CUDA 라이브러리를 사전 로드합니다.
# 자세한 내용은 src/utils/cuda_preload.py 참조.
preload_cuda_libs()

import onnxruntime as ort

# ultralytics 등이 onnxruntime (CPU)을 설치하여 GPU 버전을 덮어쓴 경우 감지 및 복구
check_onnxruntime_gpu()

import time
from .base import Runtime
from core.generation_result import GenerationResult
from core.compiled_model import CompiledModel

class OnnxRuntime(Runtime):
    """
    ONNX Runtime 기반의 실행 엔진 래퍼.
    """
    def __init__(self, **runtime_options):
        """
        [1. Hardware Provisioning & Context Initialization]
        """
        # 실행 디바이스 환경 변수 받기 (기본값 cpu)
        self.device = runtime_options.get("device", "cpu")
        
        # ONNX Runtime의 Execution Provider 설정
        _SUPPORTED_DEVICES = {"cpu", "cuda"}
        if self.device not in _SUPPORTED_DEVICES:
            raise ValueError(
                f"지원하지 않는 device입니다: '{self.device}'. "
                f"지원 목록: {sorted(_SUPPORTED_DEVICES)}"
            )

        if self.device == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "CUDAExecutionProvider를 사용할 수 없습니다. "
                    "onnxruntime-gpu 설치 여부와 CUDA 환경을 확인하세요. "
                    f"현재 가용 Provider: {available}"
                )
            self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            self.providers = ['CPUExecutionProvider']
            
        # 런타임의 상태 변수들 초기화
        self.session = None
        self.input_names = []
        self.input_shapes = {}
        self.output_names = []
        self.compiled_model = None
        self._microbatch_notice_printed = False

    def load(self, compiled_model: CompiledModel) -> None:
        """
        [2. Artifact Deserialization & Memory Mapping]
        .onnx 파일을 computation graph로 로드.
        """
        if not self.is_compatible(compiled_model):
            raise ValueError(f"Incompatible backend: {compiled_model.backend_name}")
            
        self.compiled_model = compiled_model
        
        # ONNX Runtime은 내부적으로 mmap 최적화 및 직렬화 해제를 자체 지원.
        self.session = ort.InferenceSession(
            str(self.compiled_model.artifact_path), 
            providers=self.providers
        )
        active_providers = self.session.get_providers()
        if self.device == "cuda" and "CUDAExecutionProvider" not in active_providers:
            raise RuntimeError(
                "CUDAExecutionProvider를 요청했지만 ONNX Runtime session이 CUDA를 "
                "활성화하지 못했습니다. CPU fallback 결과를 CUDA 벤치마크로 "
                f"기록하지 않도록 중단합니다. 실제 활성 Provider: {active_providers}"
            )
        
        # 모델이 요구하는 입출력 텐서의 이름표와 shape를 추출.
        input_meta = self.session.get_inputs()
        self.input_names = [inp.name for inp in input_meta]
        self.input_shapes = {inp.name: tuple(inp.shape) for inp in input_meta}
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        [4. Kernel Dispatch & Forward Pass (Inference)]
        순수 Numpy 배열을 던져주고 결과도 Numpy로 추출.
        """
        if self.session is None:
            raise RuntimeError("ONNX Runtime session is not loaded. Call load() first.")
            
        # 모델이 필요로 하는 입력이 모두 제공됐는지 검증
        missing = [name for name in self.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing required model inputs: {missing}. Provided keys: {list(inputs.keys())}")
        ort_inputs = {name: inputs[name] for name in self.input_names}
        
        # 실제 하드웨어 커널에 연산 지시.
        # ONNX export가 batch=1로 고정된 모델은 논리 배치를 microbatch로 처리합니다.
        if self._needs_microbatch_fallback(ort_inputs):
            results = self._run_microbatched(ort_inputs)
        else:
            results = self.session.run(self.output_names, ort_inputs)
        
        # 결과 리스트를 이름표와 함께 묶어 Dict 타입의 Numpy로 반환
        return {out_name: np.array(res) for out_name, res in zip(self.output_names, results)}

    def warmup(self, inputs: Dict[str, np.ndarray], num_runs: int = 1) -> None:
        """
        [3. JIT Triggering & Cache Warming]
        실제 측정 전, Cold-start 지연 시간을 제거.
        """
        print(f"[ONNX Runtime] Warming up {num_runs} times on {self.device}...")
        # LLM 패딩 trim: NLP_GENERATION 태스크에서만 실제 토큰 길이로 슬라이싱
        # BERT 등 고정 seq_len 모델에 적용하면 shape mismatch가 발생하므로 반드시 분기
        from core.model_spec import Task
        warmup_inputs = dict(inputs)
        is_llm = (
            self.compiled_model is not None
            and self.compiled_model.spec.task == Task.NLP_GENERATION
        )
        if is_llm and "attention_mask" in inputs and "input_ids" in inputs:
            attn = np.asarray(inputs["attention_mask"])
            if attn.ndim == 1:
                real_len = int(attn.sum())
            else:
                real_len = int(np.max(attn.sum(axis=-1)))
            real_len = max(real_len, 1)
            total_len = inputs["input_ids"].shape[-1]
            if real_len < total_len:
                warmup_inputs = {
                    k: (
                        v[:, :real_len]
                        if hasattr(v, "ndim") and v.ndim >= 2
                        else v[:real_len]
                        if hasattr(v, "ndim") and v.ndim == 1
                        else v
                    )
                    for k, v in inputs.items()
                }
            # 모델이 position_ids를 요구하는 경우 warmup_inputs에 자동 생성하여 주입
            if "position_ids" in self.input_names and "position_ids" not in warmup_inputs:
                attn = warmup_inputs.get("attention_mask", inputs["attention_mask"])
                warmup_inputs["position_ids"] = np.maximum(
                    np.cumsum(attn, axis=-1) - 1, 0
                ).astype(np.int64)
        for _ in range(num_runs):
            self.run(warmup_inputs)

    def unload(self) -> None:
        """
        [5. Resource Deallocation & Teardown]
        메모리 누수 및 다른 모델 테스트 시 발생할 수 있는 VRAM OOM 에러를 방지.
        """
        self.session = None
        self.input_names = []
        self.input_shapes = {}
        self.output_names = []
        self.compiled_model = None
        self._microbatch_notice_printed = False

    def get_device_spec(self) -> Dict[str, Any]:
        """현재 런타임이 구동 중인 하드웨어 명세를 반환."""
        return {
            "backend": "onnxruntime", 
            "device": self.device, 
            "active_providers": self.providers
        }

    def supports_generate(self) -> bool:
        return True

    def supports_dynamic_batching(self) -> bool:
        return True

    def max_dynamic_batch_size(self):
        if not self.input_shapes:
            return 1
        fixed_batch_dims = [
            shape[0]
            for shape in self.input_shapes.values()
            if shape and isinstance(shape[0], int) and shape[0] > 0
        ]
        return min(fixed_batch_dims) if fixed_batch_dims else None

    def _needs_microbatch_fallback(self, inputs: Dict[str, np.ndarray]) -> bool:
        """고정 batch=1 ONNX 모델에 B>1 입력이 들어오면 microbatch fallback을 사용합니다."""
        batch_size = self._infer_batch_size(inputs)
        if batch_size <= 1:
            return False

        needs_fallback = False
        for name, value in inputs.items():
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.shape[0] != batch_size:
                continue

            expected_batch = self._expected_batch_dim(name)
            if expected_batch is None or expected_batch == batch_size:
                continue
            if expected_batch == 1:
                needs_fallback = True
                continue
            raise ValueError(
                f"ONNX 모델 입력 '{name}'은 고정 batch={expected_batch}인데 "
                f"요청 batch={batch_size}입니다. dynamic batch로 export하거나 "
                f"--batch-size {expected_batch}를 사용하세요."
            )

        return needs_fallback

    def _run_microbatched(self, inputs: Dict[str, np.ndarray]) -> list[np.ndarray]:
        """고정 batch=1 모델을 샘플별로 실행한 뒤 출력 batch를 다시 합칩니다."""
        if not self._microbatch_notice_printed:
            print(
                "[ONNX Runtime] 모델 입력 batch가 1로 고정되어 있어 "
                "logical batch를 sample-wise microbatch로 실행합니다."
            )
            self._microbatch_notice_printed = True

        batch_size = self._infer_batch_size(inputs)
        per_sample_results = []
        for sample_idx in range(batch_size):
            sample_inputs = {
                name: self._slice_microbatch_value(value, sample_idx, batch_size)
                for name, value in inputs.items()
            }
            per_sample_results.append(self.session.run(self.output_names, sample_inputs))

        return [
            self._merge_microbatch_outputs([sample[out_idx] for sample in per_sample_results])
            for out_idx in range(len(self.output_names))
        ]

    def _slice_microbatch_value(self, value: np.ndarray, sample_idx: int, batch_size: int) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == batch_size:
            return arr[sample_idx:sample_idx + 1]
        return value

    def _merge_microbatch_outputs(self, chunks: list[np.ndarray]) -> np.ndarray:
        arrays = [np.asarray(chunk) for chunk in chunks]
        if not arrays:
            return np.array([])
        if arrays[0].ndim == 0:
            return np.stack(arrays, axis=0)
        try:
            return np.concatenate(arrays, axis=0)
        except ValueError:
            return np.stack(arrays, axis=0)

    def _infer_batch_size(self, inputs: Dict[str, np.ndarray]) -> int:
        batch_dims = []
        for name, value in inputs.items():
            arr = np.asarray(value)
            expected_shape = self.input_shapes.get(name)
            if arr.ndim == 0:
                continue
            if expected_shape and len(expected_shape) != arr.ndim:
                continue
            batch_dims.append(int(arr.shape[0]))

        if not batch_dims and not self.input_shapes:
            batch_dims = [
                int(np.asarray(value).shape[0])
                for value in inputs.values()
                if np.asarray(value).ndim > 0
            ]
        return max(batch_dims) if batch_dims else 1

    def _expected_batch_dim(self, input_name: str) -> int | None:
        shape = self.input_shapes.get(input_name)
        if not shape:
            return None
        batch_dim = shape[0]
        if isinstance(batch_dim, (int, np.integer)):
            return int(batch_dim)
        return None

    def generate(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = 256,
                 stop_token_ids=None) -> GenerationResult:
        """
        Greedy autoregressive 생성 루프.
        배치 전체를 한 번의 ONNX session.run()에 넣고, 샘플별 마지막 유효 토큰 위치의
        logits에서 argmax로 다음 토큰을 결정합니다 (KV 캐시 없는 단순 구현).

        stop_token_ids: int 또는 List[int] — 해당 토큰 생성 시 즉시 중단.
                        EOS, 줄바꿈(\n) 등을 포함해 과잉 생성을 방지합니다.
        timing:
            ttft_ms: 첫 배치 forward pass 지연 시간
            tpot_ms: 이후 배치 forward pass 평균 지연 시간
            total_ms: 전체 배치 생성 시간
        """
        # stop_token_ids를 set으로 정규화
        if stop_token_ids is None:
            _stop_ids = set()
        elif isinstance(stop_token_ids, int):
            _stop_ids = {stop_token_ids}
        else:
            _stop_ids = set(stop_token_ids)

        input_ids = np.asarray(inputs["input_ids"]).copy()
        if input_ids.ndim == 1:
            input_ids = input_ids[np.newaxis, :]
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids는 (seq_len,) 또는 (batch, seq_len)이어야 합니다. "
                f"현재 shape={input_ids.shape}"
            )

        if "attention_mask" in inputs:
            attention_mask = np.asarray(inputs["attention_mask"]).copy()
            if attention_mask.ndim == 1:
                attention_mask = attention_mask[np.newaxis, :]
        else:
            attention_mask = np.ones(input_ids.shape, dtype=np.int64)

        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape={attention_mask.shape}가 "
                f"input_ids shape={input_ids.shape}와 일치하지 않습니다."
            )

        batch_size = input_ids.shape[0]

        # 배치 내 각 샘플의 실제 프롬프트 길이를 계산하고, 가장 긴 프롬프트까지만 유지합니다.
        # padding="max_length"로 들어온 경우 불필요한 logits 메모리를 크게 줄입니다.
        prompt_lengths = np.asarray(attention_mask.sum(axis=-1), dtype=np.int64)
        prompt_lengths = np.maximum(prompt_lengths, 1)
        trim_len = min(int(np.max(prompt_lengths)), input_ids.shape[1])
        input_ids = input_ids[:, :trim_len]
        attention_mask = attention_mask[:, :trim_len]
        prompt_lengths = np.minimum(prompt_lengths, trim_len)

        generated_ids = [[] for _ in range(batch_size)]
        generated_lengths = np.zeros(batch_size, dtype=np.int64)
        active = np.ones(batch_size, dtype=bool)
        last_token_indices = prompt_lengths - 1
        token_times = []
        ttft_ms = 0.0
        total_start = time.perf_counter()

        for step in range(max_new_tokens):
            if not np.any(active):
                break

            # attention_mask 누적합으로 position_ids 재계산
            position_ids = np.maximum(
                np.cumsum(attention_mask, axis=-1) - 1, 0
            ).astype(np.int64)

            t0 = time.perf_counter()
            outputs = self.run({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            })
            elapsed = (time.perf_counter() - t0) * 1000.0
            token_times.append(elapsed)
            if step == 0:
                ttft_ms = elapsed

            logits = outputs[self.output_names[0]]
            next_tokens = np.zeros(batch_size, dtype=np.int64)
            active_indices = np.flatnonzero(active)
            for idx in active_indices:
                if logits.ndim == 2:
                    next_tokens[idx] = int(np.argmax(logits[idx, :]))
                elif logits.ndim == 3:
                    last_idx = int(last_token_indices[idx])
                    next_tokens[idx] = int(np.argmax(logits[idx, last_idx, :]))
                else:
                    raise ValueError(
                        f"LLM logits는 (batch, vocab) 또는 "
                        f"(batch, seq_len, vocab)이어야 합니다. 현재 shape={logits.shape}"
                    )

            append_mask = np.zeros(batch_size, dtype=bool)
            for idx in active_indices:
                token = int(next_tokens[idx])
                if token in _stop_ids:
                    active[idx] = False
                    continue
                generated_ids[idx].append(token)
                generated_lengths[idx] += 1
                append_mask[idx] = True

            if not np.any(append_mask) and not np.any(active):
                break

            # 다음 스텝을 위해 시퀀스 확장
            next_input_col = np.zeros((batch_size, 1), dtype=input_ids.dtype)
            next_input_col[append_mask, 0] = next_tokens[append_mask].astype(input_ids.dtype, copy=False)
            input_ids = np.concatenate(
                [input_ids, next_input_col], axis=1
            )
            next_mask_col = np.zeros((batch_size, 1), dtype=attention_mask.dtype)
            next_mask_col[append_mask, 0] = 1
            attention_mask = np.concatenate(
                [attention_mask, next_mask_col], axis=1
            )
            last_token_indices[append_mask] = input_ids.shape[1] - 1

        total_ms = (time.perf_counter() - total_start) * 1000.0
        tpot_ms = float(np.mean(token_times[1:])) if len(token_times) > 1 else 0.0
        max_generated_len = int(np.max(generated_lengths)) if batch_size > 0 else 0
        padded_generated_ids = np.zeros((batch_size, max_generated_len), dtype=np.int64)
        for idx, sample_ids in enumerate(generated_ids):
            if sample_ids:
                padded_generated_ids[idx, :len(sample_ids)] = np.asarray(sample_ids, dtype=np.int64)

        if batch_size == 1:
            generated_array = padded_generated_ids[0, :generated_lengths[0]].copy()
        else:
            generated_array = padded_generated_ids

        return GenerationResult(
            generated_ids=generated_array,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            num_tokens=int(np.sum(generated_lengths)),
            timing_mode="no_kv_full_context",
            uses_kv_cache=False,
            timing_source="measured",
            generated_lengths=generated_lengths,
        )

    def is_compatible(self, compiled_model: CompiledModel) -> bool:
        """이 런타임이 실행할 수 있는 '.onnx' 확장자 모델이 맞는지 검사함."""
        backend_match = compiled_model.backend_name.startswith("onnx")
        extension_match = str(compiled_model.artifact_path).endswith(".onnx")
        return backend_match or extension_match

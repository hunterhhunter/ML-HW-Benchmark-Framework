# AI Benchmark Framework

ONNX/vLLM 백엔드에서 AI 모델의 추론 성능을 측정하는 통합 벤치마크 프레임워크입니다. 모델 이름 하나만으로 다운로드부터 추론까지 자동으로 실행되는 Zero-Config 방식을 지원하며, NPU 확장을 위해 `target_id` 기반 plugin registry를 제공합니다.

## 빠른 시작

```bash
# 환경 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Zero-Config 실행 (모델/데이터셋 자동 다운로드 포함)
python src/main.py --model resnet50 --target cpu
python src/main.py --model yolov5m --target cpu
python src/main.py --model bert-base-uncased --target cpu
python src/main.py --model llama-3.2-3b --target vllm-cuda
python src/main.py --model patchtst-fm-r1 --target cpu

# 기존 backend/device 방식도 그대로 동작합니다.
python src/main.py --model resnet50 --backend onnxruntime --device cpu

# SDK 없이 NPU plugin 구조를 확인하는 mock target
python src/main.py --model resnet50 --target vendor_mock_npu --max-steps 1 --warmup 0 --monitor
```

## 지원 모델

| 모델 이름 | 태스크 | 백엔드 | 데이터셋 |
|---|---|---|---|
| `resnet50` | 이미지 분류 | onnxruntime | ImageNet-1K |
| `yolov5m` | 객체 탐지 | onnxruntime | COCO128 |
| `bert-base-uncased` | 텍스트 분류 (SST-2) | onnxruntime | SST-2 numpy |
| `bert-base-uncased-squad-v1` | 질문 답변 (SQuAD) | onnxruntime | SQuAD numpy |
| `llama-3.1-8b` | 텍스트 생성 | vllm | SQuAD 2.0 |
| `llama-3.2-3b` | 텍스트 생성 | vllm / onnxruntime | SQuAD 2.0 |
| `patchtst-fm-r1` | 시계열 예측 | onnxruntime | ETTh1 |

## CLI 옵션

```
python src/main.py --model <name> [options]

필수:
  --model           모델 프로필 이름 (위 표 참조)

선택 (생략 시 프로필 기본값 사용):
  --target          실행 target_id. 지정 시 backend/device보다 우선
  --onnx            ONNX 모델 파일 경로
  --model-path      HuggingFace 모델 디렉토리 (vLLM 백엔드)
  --dataset         데이터셋 경로
  --backend         onnxruntime | vllm (기본: onnxruntime)
  --device          cpu | cuda (기본: cpu)
  --compile         target에 compiler가 있으면 컴파일 수행 (기본)
  --no-compile      target compiler를 건너뛰고 원본 artifact 전달
  --compile-option  벤더 compiler 옵션 key=value. 여러 번 지정 가능
  --batch-size, -b  배치 크기 (기본: 1)
  --warmup, -w      웜업 횟수 (기본: 2)
  --max-new-tokens  LLM 최대 생성 토큰 수 (기본: 256)
  --max-model-len   vLLM KV 캐시 최대 컨텍스트 길이
  --debug           샘플별 예측/정답 로그 출력
```

## 기본 Target

`target_id`는 runtime, compiler, monitor, artifact format, device selector, capability를 묶는 실행 단위입니다. CLI에서는 `--target`이 우선이며, 기존 `--backend/--device` 입력은 아래 target으로 매핑되어 하위 호환됩니다.

| target_id | Runtime | Compiler | Monitor | Artifact | 용도 |
|---|---|---|---|---|---|
| `cpu` | `onnxruntime` | - | `system` | `onnx` | CPU baseline |
| `cuda` | `onnxruntime` | - | `nvidia`, `system` | `onnx` | NVIDIA GPU ONNX 실행 |
| `vllm-cpu` | `vllm` | - | `system` | `hf_model` | CPU vLLM 생성, CPU용 vLLM backend 필요 |
| `vllm-cuda` | `vllm` | - | `nvidia`, `system` | `hf_model` | NVIDIA GPU vLLM 생성 |
| `vendor_mock_npu` | `mock_npu` | `mock_npu` | `mock_npu`, `system` | `mockbin` | SDK 없는 NPU plugin 검증 |

`vendor_mock_npu`는 실제 성능 측정용이 아니라 registry/lazy import, compiler artifact cache, monitor metric 저장 흐름을 검증하기 위한 기준 plugin입니다.

`vllm-cpu`는 일반 CUDA용 vLLM wheel에서 `device=cpu`로 전환되는 target이 아닙니다. vLLM이 CPU backend로 감지되는 build/wheel이 설치되어 있어야 하며, 그렇지 않으면 `vllm-cuda` 또는 ONNX Runtime CPU target을 사용하세요.

## 아키텍처

```
src/main.py (CLI 오케스트레이터)
      |
      v
Target Registry (src/core/targets.py)
      |
      +-- Runtime Registry   (src/runtimes/)
      +-- Compiler Registry  (src/compilers/)
      +-- Monitor Registry   (src/monitors/)
      |
      v
BenchmarkRunner (src/core/benchmarkrunner.py)
      |
      +-- DataLoader  (src/dataloader/)    ← 데이터 배치 공급
      |     |
      |     +-- Preprocessor (src/preprocessor/)  ← 모델별 전처리
      |
      +-- Runtime     (src/runtimes/)      ← 추론 실행 (ONNX / vLLM)
      |
      +-- Evaluator   (src/evaluators/)    ← 메트릭 계산
```

각 레이어는 registry 기반 팩토리 함수(`create_runtime`, `get_compiler`, `create_hw_monitor`, `create_dataloader`, `create_evaluator`)를 통해 생성됩니다. 새 모델 지원은 `src/core/model_profiles.py`에 프로필을 추가하고 각 레이어에 구현체를 추가하면 됩니다. 새 NPU 벤더 지원은 Runtime/Compiler/Collector adapter를 추가한 뒤 `src/core/targets.py`에 target 조합을 등록합니다.

## Compile-aware 실행 흐름

target에 `compiler_name`이 있으면 기본적으로 compile 단계가 먼저 실행됩니다.

1. CLI/API 요청에서 `target_id`를 해석합니다.
2. target의 compiler가 있으면 `framework/artifacts/<target_id>/` 캐시를 확인하고, 없으면 컴파일합니다.
3. compiler는 artifact path와 compile metadata를 `CompileResult`로 반환합니다.
4. runtime은 컴파일된 artifact 또는 원본 ONNX/HF artifact를 동일한 `CompiledModel` 인터페이스로 로드합니다.
5. 결과 CSV에는 `target_id`, `accelerator_vendor`, `accelerator_name`, `runtime_name`, `compiler_name`, `artifact_format`이 함께 저장됩니다.

자세한 plugin 구조는 [../docs/npu-plugin-registry.md](../docs/npu-plugin-registry.md)를 참조하세요.

## 모델/데이터셋 준비

Zero-Config 실행 시 모델과 데이터셋이 없으면 자동으로 `prepare_*.py` 스크립트가 실행됩니다. 수동으로 실행하려면:

```bash
# 모델 다운로드
python models/prepare_resnet50_kalray.py
python models/prepare_yolov5m.py
python models/prepare_bert_sst2.py
python models/prepare_llama_3_2_3b.py  # Hugging Face 토큰 필요
python models/prepare_patchtst.py

# 데이터셋 다운로드
python datasets/prepare_imagenet_1k.py
python datasets/prepare_coco128.py
python datasets/prepare_text_numpy.py  # BERT 텍스트 분류용
python datasets/prepare_squad2.py      # Llama / BERT QA용
python datasets/prepare_etth1.py       # PatchTST용
```

## 테스트

```bash
# 전체 테스트
python -m pytest tests/ -v

# 단위 테스트만 (모델 파일 불필요)
python -m pytest tests/test_factory_api.py tests/test_bert_qa_loader.py tests/test_plugin_registry.py -v

# target registry / result metadata
python -m pytest tests/test_plugin_registry.py tests/test_result_store.py -q

# 전체 ONNX 벤치마크 일괄 실행
python tests/run_all_onnx_benchmarks.py
```

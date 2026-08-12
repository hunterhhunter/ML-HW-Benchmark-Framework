# AI Benchmark Framework

ONNX/vLLM/Furiosa-LLM 백엔드와 precompiled Rebellions RBLN artifact에서 AI 모델의 추론 성능을 측정하는 통합 벤치마크 프레임워크입니다. 모델 이름 하나만으로 다운로드부터 추론까지 자동으로 실행되는 Zero-Config 방식을 지원하며, NPU 확장을 위해 `target_id` 기반 plugin registry를 제공합니다.

## 빠른 시작

```bash
# 환경 설정
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Zero-Config 실행 (모델/데이터셋 자동 다운로드 포함)
python src/main.py --model resnet50 --target cpu
python src/main.py --model yolov5m --target cpu
python src/main.py --model yolov8s-seg --target cpu --max-steps 1
python src/main.py --model yolov8s-pose --target cuda --max-steps 1
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
| `resnet50` | 이미지 분류 | onnxruntime / `rbln-static` | ImageNet-1K |
| `yolov5m` | 객체 탐지 | onnxruntime / `rbln-static` | COCO128 |
| `yolov8s-seg` | 인스턴스 분할 (Mask AP) | onnxruntime CPU/CUDA | COCO 2017 val |
| `yolov8s-pose` | 자세 추정 (OKS AP) | onnxruntime CPU/CUDA | COCO 2017 val |
| `bert-base-uncased` | 텍스트 분류 (SST-2) | onnxruntime / `mobilint-aries` / `rbln-static` | SST-2 numpy |
| `bert-base-uncased-squad-v1` | 질문 답변 (SQuAD) | onnxruntime / `mobilint-aries` / `rbln-static` | SQuAD numpy |
| `llama-3.1-8b` | 텍스트 생성 | vllm / `rbln-vllm` (ATOM 8장 공식, 1장 opt-in 용량 실험) | SQuAD 2.0 |
| `llama-3.2-3b` | 텍스트 생성 | vllm / onnxruntime / `rbln-vllm` | SQuAD 2.0 |
| `patchtst-fm-r1` | 시계열 예측 | onnxruntime / `rbln-static` | ETTh1 |

RBLN static은 이미지 분류, 객체 탐지, BERT 언어 이해
(분류·QA), 시계열 예측의 네 가지 task family를 지원한다. Llama
generation은 `rbln-static`이 아니라 내부 vLLM RBLN 엔진을 사용하는
`rbln-vllm` target으로 실행한다. Llama 모델 준비, 한 장 3B 및 8B opt-in
실험, 8장 공식 구성, sync/async 명령과 cleanup 절차는
[RBLN vLLM 실행 가이드](docs/rbln-vllm-setup.md)를 참고한다. 정적 모델은
[RBLN-CA22 운영 가이드](docs/rbln-setup.md)를 참고한다. 일곱 모델의
provenance, exact compile command, inspect, SHA256과 artifact handoff 절차는
[RBLN 컴파일 재현 가이드](docs/rbln-compilation.md)를 기준으로 한다.

## CLI 옵션

```
python src/main.py --model <name> [options]

필수:
  --model           모델 프로필 이름 (위 표 참조)

선택 (생략 시 프로필 기본값 사용):
  --target          실행 target_id. 지정 시 backend/device보다 우선
  --onnx            ONNX 모델 파일 경로
  --hef             HailoRT 실행용 HEF 파일 경로
  --artifact        target 전용 사전 컴파일 artifact 경로 (예: Mobilint .mxq, DEEPX .dxnn, Rebellions .rbln)
  --mobilint-bert-weights
                    Mobilint embedding-input BERT MXQ용 weight_dict.pth 경로
  --image-preprocess-profile
                    Mobilint raw vision artifact profile (기본: auto)
  --fxb             Furiosa RNGD의 선택적 FXB override 경로
  --model-path      HuggingFace repository ID 또는 모델 디렉터리 (`rbln-vllm`은 준비된 로컬 디렉터리 필수)
  --dataset         데이터셋 경로
  --backend         onnxruntime | iree | vllm | hailort | deepx | furiosa_llm | rbln | rbln_vllm (기본: onnxruntime)
  --device          cpu | cuda (기본: cpu)
  --compile         target에 compiler가 있으면 컴파일 수행 (기본)
  --no-compile      target compiler를 건너뛰고 원본 artifact 전달
  --compile-option  벤더 compiler 옵션 key=value. 여러 번 지정 가능
  --runtime-option  런타임 옵션 key=value. 여러 번 지정 가능
  --batch-size, -b  배치 크기 (기본: 1)
  --warmup, -w      웜업 횟수 (기본: 2)
  --max-new-tokens  LLM 최대 생성 토큰 수 (기본: 256)
  --max-model-len   vLLM KV 캐시 최대 컨텍스트 길이
  --debug           샘플별 예측/정답 로그 출력
```

## 비동기 추론 큐

기존 순차 실행인 `e2e`가 기본값이며 그대로 사용할 수 있습니다.

```bash
python src/main.py --model resnet50 --target cpu --inference-mode e2e
```

Offline형 비동기 큐는 가능한 한 빠르게 요청을 공급합니다. async에서
`--batch-size`는 동적으로 묶을 최대 batch size입니다.

```bash
python src/main.py \
  --model resnet50 \
  --target cpu \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --queue-capacity 256 \
  --worker-count 1 \
  --batch-timeout-ms 1
```

독립 요청을 동적으로 묶으려면 모델과 runtime이 dynamic batch를 지원하고,
dataloader/pipeline metadata의 `is_static_batched`가 `False`여야 합니다.
`is_static_batched=True`인 loader는 이미 batch된 단일 request 경로를 사용하므로
`--batch-size`가 1보다 커도 요청을 합치지 않으며 관측 batch가 1일 수 있습니다.

Server-like 부하는 seed 기반 요청 간격으로 target QPS를 재현합니다.

```bash
python src/main.py \
  --model resnet50 \
  --target cpu \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 100 \
  --min-duration-sec 10 \
  --min-samples 100
```

`async_queue` 결과는 MLPerf 결과가 아닙니다. MLPerf LoadGen은 신뢰성 설계의
레퍼런스로만 사용했으며, `async_queue` 모듈과 실행 경로는 LoadGen을 import하거나
사용하지 않고 SUT/QSL API, 공식 validity 규칙, 로그 호환,
submission·compliance·audit를 구현하지 않습니다. 기존
`src/adapters/loadgen_adapter.py`는 이 경로와 분리된 비활성 legacy skeleton이며
이번 구현의 통합 대상이 아닙니다.

지표 경계, 결과 파일, 유효성 판정, e2e 대비 기대 효과와 위험은
[비동기 추론 큐 측정 가이드](../docs/async-inference-queue.md)를 참고하세요.

## 기본 Target

`target_id`는 runtime, compiler, monitor, artifact format, device selector, capability를 묶는 실행 단위입니다. CLI에서는 `--target`이 우선이며, 기존 `--backend/--device` 입력은 아래 target으로 매핑되어 하위 호환됩니다.

| target_id | Runtime | Compiler | Monitor | Artifact | 용도 |
|---|---|---|---|---|---|
| `cpu` | `onnxruntime` | - | `system` | `onnx` | CPU baseline |
| `cuda` | `onnxruntime` | - | `nvidia`, `system` | `onnx` | NVIDIA GPU ONNX 실행 |
| `vllm-cpu` | `vllm` | - | `system` | `hf_model` | CPU vLLM 생성, CPU용 vLLM backend 필요 |
| `vllm-cuda` | `vllm` | - | `nvidia`, `system` | `hf_model` | NVIDIA GPU vLLM 생성 |
| `furiosa-rngd` | `furiosa_llm` | - | `system` | `fxb` | Furiosa RNGD LLM 생성 |
| `furiosa-rngd-torch` | `furiosa_torch` | - | `system` | `pytorch_model` | Furiosa RNGD BERT SST-2/SQuAD strict compile |
| `vendor_mock_npu` | `mock_npu` | `mock_npu` | `mock_npu`, `system` | `mockbin` | SDK 없는 NPU plugin 검증 |
| `hailo8` | `hailort` | - | `hailo`, `system` | `hef` | Hailo-8/8L HEF sync inference |
| `hailo10h` | `hailort` | - | `hailo`, `system` | `hef` | Hailo-10H HEF sync inference |
| `deepx` | `deepx` | `deepx` | `deepx`, `system` | `dxnn` | DEEPX DX-COM compile + DX-RT sync/native-async inference |
| `mobilint-aries` | `mobilint` | - | `mobilint`, `system` | `mxq` | ARIES용 precompiled MXQ sync/native-async inference |
| `mobilint-regulus` | `mobilint` | - | `mobilint`, `system` | `mxq` | REGULUS용 precompiled MXQ sync E2E, bundle 0·Cluster0/Core0 NPU-only 검증 |
| `rbln-static` | `rbln` | - | `rbln`, `system` | `rbln` | Rebellions CA22 precompiled static sync/native async |
| `rbln-vllm` | `rbln_vllm` | - | `rbln`, `system` | `rbln_llm_dir` | Rebellions CA22 prepared Llama directory sync/native async generation |

`vendor_mock_npu`는 실제 성능 측정용이 아니라 registry/lazy import, compiler artifact cache, monitor metric 저장 흐름을 검증하기 위한 기준 plugin입니다.

`vllm-cpu`는 일반 CUDA용 vLLM wheel에서 `device=cpu`로 전환되는 target이 아닙니다. vLLM이 CPU backend로 감지되는 build/wheel이 설치되어 있어야 하며, 그렇지 않으면 `vllm-cuda` 또는 ONNX Runtime CPU target을 사용하세요.

Hailo target은 HailoRT Python wheel과 Ubuntu package가 설치된 환경에서 `.hef` 파일을 직접 실행합니다. Hailo-8/8L은 `hailo8`, Hailo-10H는 `hailo10h` target을 사용하면 결과 CSV의 `target_id`와 `accelerator_name`이 분리되어 저장됩니다.
전력 측정이 보드/펌웨어에서 지원되지 않으면 Hailo collector는 온도만 수집하고 `hw_accel_monitor_note`에 fallback 사유를 남깁니다.

```bash
python src/main.py --model resnet50 --target hailo8 --hef /path/to/resnet50.hef --layout NHWC --monitor
python src/main.py --model resnet50 --target hailo10h --hef /path/to/resnet50_10h.hef --layout NHWC --monitor
```

Furiosa RNGD Llama는 Furiosa-LLM 2026.3.0 환경에서, BERT SST-2/SQuAD는 PyTorch 2.10.0 기반 Furiosa Torch 2026.3.0 환경에서 실행합니다. 두 환경은 분리해야 합니다. Llama의 `--model-path`에는 Furiosa 모델 repository ID 또는 로컬 repository 디렉터리를 전달하고, BERT에는 검증된 로컬 Hugging Face 모델 디렉터리를 전달합니다. 설치, 서버 없는 E2E/async 실행, 장비 검증 절차는 [../docs/furiosa-rngd-setup.md](../docs/furiosa-rngd-setup.md)를 참조하세요.

DEEPX target은 DX-COM의 `dxcom` CLI로 ONNX와 config JSON을 `.dxnn`으로 컴파일한 뒤 `dx_engine` Python package가 설치된 DX-RT 환경에서 실행합니다. `e2e`는 동기 `run()` 경로를 유지하고, `async_queue`는 warmup부터 측정까지 DX-RT callback 기반 `run_async()` 경로를 사용합니다.
DX-COM wheel은 별도로 설치해야 하며, `dxcom --version`으로 CLI가 PATH에 있는지 확인하세요.
`--monitor`를 켜면 DX-RT `DeviceStatus` API로 NPU 온도, 전압, 클럭과 CPU/RAM 지표를 함께 수집합니다.
Jetson checkout, DX-RT 3.3.2 사전 확인, ResNet50/YOLOv5M E2E·async 전체 명령과 합격 기준은 [../docs/deepx-setup.md](../docs/deepx-setup.md)를 참조하세요.

```bash
python src/main.py --model resnet50 --target deepx \
  --compile-option config_path=/path/to/resnet50_config.json \
  --layout NCHW --monitor
```

이미 컴파일된 `.dxnn` artifact를 실행할 때는 compile 단계를 건너뜁니다. 런타임 옵션으로 `device_ids=0,1`, `bound_option=NPU_ALL`, `use_ort=true`, `buffer_count=8`, `input_layout=NHWC`, `batch_mode=microbatch` 등을 지정할 수 있습니다.

```bash
python src/main.py --model resnet50 --target deepx --no-compile \
  --artifact /path/to/resnet50.dxnn --layout NCHW --monitor
```

사전컴파일 ResNet50 native async 빠른 검증 예시입니다. DX-RT native async는
job당 sample 하나만 지원하므로 `--batch-size 1`을 사용합니다.

```bash
python src/main.py --model resnet50 --target deepx --no-compile \
  --artifact models/deepx/ResNet50.dxnn \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --worker-count 4 --queue-capacity 16 \
  --min-samples 8 --max-samples 8 --warmup 2 \
  --runtime-option device_ids=0 \
  --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 \
  --save-request-trace \
  --results-path results/deepx_device_validation.csv
```

## Mobilint BERT embedding MXQ

ARIES용 SST-2와 SQuAD v1 MXQ는 token ID가 아니라 `float32 [1,L,768]`
`embeddings` 하나를 입력으로 받는다. 기존 BERT loader가 읽은 token 배열은 runtime
호출 전에 `--mobilint-bert-weights`의 embedding table과 LayerNorm을 사용해 변환된다.
따라서 보고되는 runtime latency에는 qb Runtime inference와 출력 정규화가 포함되지만
CPU embedding 생성 시간은 포함되지 않는다. 두 artifact는 batch 1, `single` Core0,
동기 E2E부터 검증한다.

```bash
python src/main.py \
  --model bert-base-uncased \
  --target mobilint-aries \
  --artifact /path/to/sst2.mxq \
  --mobilint-bert-weights /path/to/sst2/weight_dict.pth \
  --dataset datasets/sst2_numpy \
  --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 64 \
  --runtime-option core_mode=single --no-compile

python src/main.py \
  --model bert-base-uncased-squad-v1 \
  --target mobilint-aries \
  --artifact /path/to/squad1.mxq \
  --mobilint-bert-weights /path/to/squad1/weight_dict.pth \
  --dataset datasets/squad_numpy \
  --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 64 \
  --runtime-option core_mode=single --no-compile
```

SQuAD SDK 출력 순서는 `end_logits`, `start_logits`로 고정하며 결과 CSV의
`mobilint_output_order`에도 같은 순서를 기록한다. 설치 확인, 실제 서버 artifact 경로,
MXQ 검사 및 합격 기준은
[Mobilint ARIES Transformer·LLM 실행 가이드](../docs/mobilint-aries-transformers.md)를
참고한다. 이 경로는 `mobilint-aries-llm`이나 vision loader를 사용하지 않는다.

## Mobilint vision MXQ

Regulus 실행은 `mobilint-regulus`를 사용한다. adapter는 내부적으로 qbruntime을
호출하고, `mbltml`이 설치된 경우에는 이를 우선 사용하며 Regulus Yocto에서는
`/dev/regulus-npu0` kernel node를 확인한다. `force_single_npu_bundle(0)` 및 launch
후 `Cluster0/Core0` 검증이 완료된 경우에만 CSV와 async 상세에
`runtime_version`, `npu_only_verified=True`, `execution_binding`을 기록한다.
`infer_speedrun()`은 E2E 측정에 사용하지 않는다. 실제 보드 실행 명령과 결과
해석은 [Mobilint Regulus qb Runtime 실행 가이드](../docs/mobilint-regulus-runtime.md)를
따른다.

현재 raw vision 경로는 official basename 두 개만 profile로 등록합니다.

| profile ID | official basename | task | input | outputs | 기본 threshold | ARIES/REGULUS 공유 |
|---|---|---|---|---|---|---|
| `mobilint-resnet50-imagenet1k-v2` | `resnet50_IMAGENET1K_V2.mxq` | 이미지 분류 | `(1,224,224,3)` `uint8` NHWC | profile 미고정; SDK metadata 사용 | - | 예 |
| `mobilint-yolov5m-default` | `yolov5m.mxq` | COCO 객체 탐지 | `(1,640,640,3)` `uint8` NHWC | raw heads `(1,20,20,255)`, `(1,40,40,255)`, `(1,80,80,255)` | confidence `0.001`, IoU `0.65`, max detections `300` | 예 |

`auto`는 model, task, exact basename이 모두 일치해야 합니다. 알 수 없는 artifact는 generic
전처리로 fallback하지 않습니다. Model Zoo는 parity 확인에만 사용하며 production에서
import하지 않습니다. qb Runtime SDK metadata가 profile 계약과 일치해야 하고, compiler
연동/MXQ 자동 컴파일과 YOLOv5mu, P6, segmentation, pose, OBB, YOLOv8 이상은 범위 밖입니다.

다음 smoke command는 저장소 루트에서 실행하며 실제 ARIES2 성공 여부는 hardware 로그로
별도 확인해야 합니다.

```bash
python framework/src/main.py \
  --model resnet50 \
  --target mobilint-aries \
  --artifact framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq \
  --dataset framework/datasets/imagenet_1k \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --warmup 2 \
  --max-steps 10 \
  --monitor

python framework/src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact framework/models/mobilint/yolov5m/aries/yolov5m.mxq \
  --dataset /path/to/coco \
  --image-preprocess-profile auto \
  --layout NHWC \
  --no-compile \
  --runtime-option core_mode=global8 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --monitor \
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 16 \
  --worker-count 1 \
  --max-samples 10
```

Sync, monitor, native-async 인수 명령과 반환할 SDK/driver version, artifact hash,
Model Zoo 비교, telemetry 및 shutdown count 목록은
[runtimes guide](src/runtimes/README.md#aries2-vision-인수-명령)를 참고하세요.

## Rebellions RBLN static

RBLN target은 device 0 `RBLN-CA22`용으로 미리 compile된 fixed-shape
`.rbln`을 실행한다. Raw Hugging Face directory나 ONNX file을 runtime에서
자동 compile하지 않으며, single NPU와 request batch 1만 보장한다.
Offline artifact 생성과 검증은
[RBLN 컴파일 재현 가이드](docs/rbln-compilation.md)의 repository recipe를
사용한다.
SDK inspect가 BERT SQuAD의 두 output name을 생략하는 경우에는 artifact와 함께
SHA256으로 결합된 `model.rbln.json` output manifest가 필요하다. SQuAD artifact는
`input_ids`, `attention_mask`, `token_type_ids` 세 입력을 모두 `int64 (1,384)`로
compile해야 한다.

```bash
python3 -m src.main --model resnet50 --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k --batch-size 1 --monitor
```

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
      +-- Runtime     (src/runtimes/)      ← 추론 실행 (ONNX / vLLM / Furiosa-LLM / RBLN)
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
python models/prepare_yolov8_vision.py
python models/prepare_bert_sst2.py
python models/prepare_bert_squad.py
python models/prepare_llama_3_2_3b.py  # Hugging Face 토큰 필요
python models/prepare_patchtst.py

# 데이터셋 다운로드
python datasets/prepare_imagenet_1k.py
python datasets/prepare_coco128.py
python datasets/prepare_coco_vision.py
python datasets/prepare_text_numpy.py  # BERT 텍스트 분류용
python datasets/prepare_squad_numpy.py # BERT QA용; token_type_ids 포함
python datasets/prepare_squad2.py      # Llama용
python datasets/prepare_etth1.py       # PatchTST용
```

YOLOv8 vision smoke 검증은 실제 COCO 이미지 한 배치만 처리합니다.
`--max-steps`를 생략하면 COCO val2017 전체에 대해 공식 Mask AP 또는 OKS AP를
계산합니다.

```bash
# CPU/CUDA smoke (실제 ONNX + 실제 COCO 입력)
python src/main.py --model yolov8s-seg --target cpu --max-steps 1 --warmup 1
python src/main.py --model yolov8s-pose --target cuda --max-steps 1 --warmup 1

# 전체 COCO 평가
python src/main.py --model yolov8s-seg --target cpu
python src/main.py --model yolov8s-pose --target cuda
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

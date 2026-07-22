# Runtimes Package

`runtimes` 패키지는 ONNX Runtime, vLLM, IREE, 벤더 NPU SDK처럼 기술 스택과 가속 방식이 서로 다른 **하드웨어 추론 엔진**을 `BenchmarkRunner` 입장에서 투명하게 제어하도록 캡슐화합니다.

## Architecture

- **`base.py` (`Runtime`)**: 모든 외부 런타임 wrapper가 준수해야 하는 추상 인터페이스입니다.
  - `load()`: 원본 또는 컴파일된 artifact를 target 메모리에 로드
  - `warmup()`: 추론 초기 성능 왜곡을 줄이기 위한 예열
  - `run()`: 배치 단위 추론을 실행하고 Numpy dictionary 또는 generation result를 반환
  - `unload()`: 런타임 리소스 해제

- **`__init__.py` (Registry Facade)**
  - `RuntimeEntry`를 registry에 등록하고 `create_runtime(name, device, **kwargs)`로 생성합니다.
  - `get_runtime_entry(name)`로 lazy import 없이 entry metadata를 조회합니다.
  - 등록 entry는 lazy import를 사용합니다. 특정 벤더 SDK가 설치되지 않아도 프레임워크 import와 다른 target 실행은 깨지지 않습니다.
  - 기존 alias도 유지합니다. 예를 들어 `onnx`는 `onnxruntime`으로 매핑됩니다.
  - canonical name 또는 alias가 이미 다른 entry에 등록되어 있으면 등록 시점에 실패합니다.

## Built-in Runtime Registry

| name | aliases | 설명 |
|---|---|---|
| `onnxruntime` | `onnx` | ONNX Runtime backend |
| `vllm` | - | vLLM generation backend |
| `iree` | `mlir` | IREE backend placeholder |
| `mock_npu` | `vendor_mock_npu` | SDK-free NPU plugin 검증 runtime |
| `hailort` | `hailo`, `hailo8`, `hailo10h` | HailoRT HEF runtime for Hailo devices |
| `deepx` | `dxrt`, `deepx_npu` | DEEPX DXNN runtime |
| `mobilint` | `qbruntime`, `mxq` | 명시적으로 선택한 ARIES/REGULUS에서 사전 컴파일된 `.mxq`를 실행하는 공용 qb Runtime adapter |
| `mobilint_llm` | - | 로컬 Model Zoo Hugging Face 모델을 실행하는 ARIES 전용 generation runtime |

## Mobilint ARIES and REGULUS raw runtime

현재 연동 범위는 런타임, 비동기 실행, 모니터링입니다. 컴파일러 연동은 후순위이므로 대상 장치용으로 벤더가 미리 컴파일한 `.mxq`가 필요합니다. ARIES와 REGULUS는 같은 `mobilint` qb Runtime adapter를 사용하지만, 장치 패밀리를 추측하지 않고 각각 `mobilint-aries` 또는 `mobilint-regulus` target으로 명시해야 합니다.

### 지원하는 vision artifact profile

| profile ID | official basename | task | runtime input | runtime outputs | decoder 기본값 | ARIES/REGULUS 공유 |
|---|---|---|---|---|---|---|
| `mobilint-resnet50-imagenet1k-v2` | `resnet50_IMAGENET1K_V2.mxq` | ImageNet-1K 분류 | `(1,224,224,3)` `uint8` NHWC | profile에서 shape를 고정하지 않음; SDK metadata 사용 | 해당 없음 | 예 |
| `mobilint-yolov5m-default` | `yolov5m.mxq` | COCO 객체 탐지 | `(1,640,640,3)` `uint8` NHWC | `(1,20,20,255)`, `(1,40,40,255)`, `(1,80,80,255)` | confidence `0.001`, IoU `0.65`, max detections `300` | 예 |

`--image-preprocess-profile auto`는 normalized model name, task, exact artifact
basename이 모두 일치할 때만 profile을 선택합니다. 등록되지 않은 basename에는 generic
float 전처리로 fallback하지 않고 실패합니다. 이름을 바꾼 artifact를 의도적으로 사용할
때만 explicit profile ID를 지정할 수 있으며 model과 task 검증은 유지됩니다. 두 target은
동일한 profile, loader, decoder를 사용하고 target은 device family 검증만 바꿉니다.

Mobilint Model Zoo 구현과 출력은 parity oracle일 뿐 production runtime dependency가
아닙니다. 실행에는 qb Runtime이 보고하는 input/output metadata가 필요하며 profile 계약과
다르면 첫 추론 전에 실패합니다. Mobilint compiler 연동과 MXQ 자동 컴파일, YOLOv5mu,
P6, segmentation, pose, OBB 및 YOLOv8 이상 artifact는 지원 범위 밖입니다.

### ARIES2 vision 인수 명령

아래 명령은 저장소 루트에서 실행합니다. 실제 SDK/driver가 설치된 ARIES2에서 얻은 로그가
아직 없으므로 성공한 hardware acceptance로 간주하지 않습니다.

```bash
# ResNet50: 기본 warmup 2회 뒤 synchronous/e2e 10 steps와 monitor 확인
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

# YOLOv5m: synchronous/e2e 정확도 확인
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
  --warmup 2 \
  --max-steps 10

# YOLOv5m: 같은 synchronous/e2e 실행의 전력·사용률·energy 확인
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
  --warmup 2 \
  --max-steps 10 \
  --monitor

# YOLOv5m: qb Runtime native async와 정상 shutdown 확인
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

`/path/to/coco`는 `images/val2017`과 `labels/val2017`을 포함하는 COCO dataset
root로 바꿉니다. 인수 로그에는 Mobilint SDK/qbruntime, driver, firmware version,
artifact SHA-256, effective profile과 threshold를 남깁니다. ResNet은 Model Zoo와
framework의 첫 이미지 top-1/top-5, YOLO는 같은 이미지의 letterbox와 pre-NMS/final
detections 또는 COCO mAP를 비교합니다. Monitor의 power/utilization/memory/temperature,
energy/sample coverage와 native-async 종료 시 submitted/completed/failed/outstanding count도
함께 반환해야 hardware acceptance를 완료할 수 있습니다.

`qbruntime`과 `mbltml`은 해당 adapter를 실제로 로드하거나 실행할 때 지연 import됩니다. 따라서 Mobilint SDK를 기본 requirements에 추가하지 않으며, SDK가 없는 환경에서도 다른 target과 registry 조회는 동작합니다.

### 비동기 실행 구조

raw `async_queue`에는 역할이 다른 두 비동기 계층이 연결됩니다. 프레임워크의 bounded request queue가 요청 스케줄링, backpressure, 요청 identity를 소유합니다. 그 아래에서 `mobilint` native backend가 qb Runtime의 `infer_async()`를 호출하고, 반환된 Future의 완료를 프레임워크 callback 계약으로 연결합니다. Adapter가 별도의 요청 큐를 하나 더 만드는 구조는 아닙니다.

qb Runtime native async는 이 연동에서 raw CNN에만 적용되며 batch dimension은 `N=1`만 지원합니다. 실제 장치에서 queue depth와 동시성을 검증하기 전에는 target 기본값인 `activation_slots=1`과 `--worker-count 1`을 유지하는 것이 안전합니다. `infer_async()` 제출 자체가 SDK 내부 포화 상태에서 block할 수 있고 물리적 취소를 제공하지 않을 수 있으므로, 논리적 요청 timeout만으로 장치 작업이 끝났다고 간주해서는 안 됩니다. 모든 Future가 완료되어 native backend shutdown이 성공하기 전에는 모델을 dispose하지 않습니다.

### 모니터링

`--monitor`는 target에 연결된 `mobilint` collector와 `system` collector를 함께 활성화합니다. ARIES와 REGULUS 모두 mbltml에서 utilization, memory usage, temperature를 수집합니다. ARIES는 power/current/voltage도 수집하고, power 표본 사이를 사다리꼴 적분해 `hw_accel_energy_j`를 계산하며 `hw_accel_power_samples`와 `hw_accel_power_sample_coverage`를 함께 기록합니다. REGULUS는 SDK에서 지원하지 않는 전기 계측 key를 거짓 0으로 채우지 않고 결과에서 생략합니다.

Runtime과 monitor는 target에 고정된 동일한 `device_id`와 `expected_family` selector를 공유합니다. 선택한 target과 실제 장치 패밀리가 다르면 mbltml 검증 단계에서 qbruntime 모델 launch 전에 실패합니다.

### 실제 하드웨어 인수 점검

SDK-free 테스트는 adapter 계약, lazy import, queue/Future 연결과 metric 계산을 fake SDK로 검증할 뿐 실제 NPU의 성능이나 안정성을 검증하지 않습니다. ARIES 및 REGULUS가 설치된 호스트에서는 다음 항목을 별도로 확인하고 결과와 함께 기록해야 합니다.

- 사용한 Mobilint SDK, driver, firmware 버전과 장치 종류(ARIES, REGULUS PCIe 또는 REGULUS USB)를 기록합니다.
- 각 target에서 정상 장치가 선택되는지 확인하고, 의도적으로 반대 패밀리 target을 지정했을 때 모델 launch 전에 mismatch가 거부되는지 확인합니다.
- 동일한 입력에 대한 sync raw 출력과 CPU 또는 벤더 기준 출력을 비교합니다.
- raw async 부하를 포화시켜 `infer_async()` 제출 block, 요청 timeout, flush/shutdown 동작을 확인합니다.
- timeout 뒤에도 실행 중인 모델이 조기 dispose되지 않는지, dispose 직전에 outstanding qb Runtime Future가 정확히 0인지 확인합니다.
- ARIES에서 power sample 수와 coverage를 함께 검토하고, 알려진 일정 전력 또는 외부 전력계와 사다리꼴 적분 energy를 비교합니다.
- REGULUS 결과에 power/current/voltage/energy key가 존재하지 않는지 확인합니다.

## Mobilint ARIES Model Zoo LLM runtime

LLM 추론에는 Mobilint Model Zoo 형식으로 로컬에 준비한 Hugging Face 모델 디렉토리가 필요합니다. 이 경로는 `mobilint-aries-llm` target으로 명시하며 REGULUS LLM target은 제공하지 않습니다. Model Zoo, Transformers, Torch 모듈은 runtime을 실제로 로드할 때 지연 import되므로 기본 requirements에는 추가하지 않습니다.

```bash
# e2e scalar TTFT/TPOT report
python src/main.py --model llama-3.2-3b --target mobilint-aries-llm \
  --model-path /path/to/prepared-model-zoo-hf-directory \
  --inference-mode e2e --max-steps 10 --monitor

# async token-event evidence and request trace
python src/main.py --model llama-3.2-3b --target mobilint-aries-llm \
  --model-path /path/to/prepared-model-zoo-hf-directory \
  --inference-mode async_queue --batch-size 1 --worker-count 1 \
  --queue-capacity 16 --min-samples 100 --max-samples 100 \
  --save-request-trace --monitor
```

ARIES LLM runtime은 SDK native-async capability를 선언하지 않습니다. `async_queue`를 선택하면 공통 프레임워크 큐 아래에서 blocking `generate()`를 worker 하나로 실행하며 qb Runtime `infer_async()`는 사용하지 않습니다.

### LLM 토큰 지연시간의 의미

생성 producer의 Transformers streamer callback 시각과 callback별 누적 생성 토큰 수를 기록합니다. Prompt 전달 callback을 제외한 첫 생성 callback으로 TTFT를 계산하고, 두 개 이상의 토큰이 생성된 요청에서는 첫 callback부터 마지막 callback까지의 시간을 `generated_tokens - 1`로 나눠 request TPOT를 계산합니다. 한 토큰만 생성된 요청에는 TPOT 구간이 없으므로 TPOT를 보고하지 않습니다.

e2e 실행은 evaluator를 통해 scalar TTFT/TPOT만 집계합니다. Callback별 event, token ITL, grouped callback 진단을 확인하려면 `async_queue`와 `--save-request-trace`를 함께 사용해야 합니다. Async details의 `generation_stream_itl_incomplete` warning은 sidecar에 저장되고, 원시 callback 시각과 누적 토큰 수는 request trace의 각 요청 row에 저장됩니다.

Callback의 누적 토큰 수가 매번 정확히 하나씩 증가한 요청에서만 token ITL percentile을 제공합니다. 한 callback에서 여러 토큰이 함께 전달되면 존재하지 않는 토큰별 timestamp를 만들지 않습니다. 이 경우 token ITL percentile을 생략하고 `generation_stream_itl_incomplete`를 남깁니다.

실제 ARIES에서는 반환 토큰 수와 streamer의 마지막 누적 토큰 수가 일치하는지 확인해야 합니다. 한 토큰씩 전달될 때는 event 수와 토큰 수가 같고 grouped callback에서는 서로 다를 수 있습니다. Grouped callback 검증은 async details의 warning 및 token ITL 생략과 request trace의 원본 event 보존을 함께 확인해야 하며, e2e 결과만으로는 이 event-level 증거를 검증할 수 없습니다.

## 새 Runtime 추가

새 벤더 NPU runtime을 추가할 때 core 실행 코드를 수정하지 않습니다. adapter와 registry entry만 추가합니다.

1. `Runtime`을 상속한 adapter 파일을 `src/runtimes/`에 추가합니다.
2. 벤더 SDK import는 가능한 한 adapter 내부의 `load()` 또는 초기화 시점으로 미룹니다.
3. `RuntimeEntry`를 등록합니다.
4. `src/core/targets.py`에서 해당 runtime을 사용하는 `TargetSpec`을 추가합니다.
5. `core.targets.validate_registry_graph()` 또는 `framework/tests/test_plugin_registry.py`로 target graph를 검증합니다.

```python
# src/runtimes/__init__.py
register_runtime(RuntimeEntry(
    name="vendor_npu",
    module="runtimes.vendor_npu_rt",
    class_name="VendorNpuRuntime",
    aliases=("vendor-x",),
    description="Vendor NPU runtime adapter",
))
```

실제 벤더 SDK adapter는 `Runtime.load()`에서 compiler artifact 또는 원본 artifact를 target device에 올리고, `Runtime.run()`에서 프레임워크 evaluator가 이해할 수 있는 출력 형태를 반환해야 합니다.

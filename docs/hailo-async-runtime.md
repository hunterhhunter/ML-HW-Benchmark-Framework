# Hailo native async runtime 가이드

## 지원 범위

`hailort` runtime은 HailoRT `InferModel` API가 제공되는 경우 `async_queue`에서
vendor-native 비동기 추론을 사용한다. 현재 검증 대상 모델은 다음 두 프로필이다.

- `resnet50`: ImageNet 분류, dense classification output
- `yolov5m`: COCO 객체 탐지, dense 또는 ragged Hailo NMS output

동일한 adapter가 `hailo8`과 `hailo10h` target에 사용되지만 설치할 HailoRT와 HEF는
장치 세대에 맞아야 한다.

| target | 장치 | 지원 runtime 계열 | 참고 suite 조합 |
|---|---|---|---|
| `hailo8` | Hailo-8/8L | HailoRT 4.24.x | AI SW Suite 2026-06: HailoRT 4.24.0 |
| `hailo10h` | Hailo-10H | HailoRT 5.x | 2026-04 호환표: HailoRT 5.3.0 |

HailoRT 4.24 user guide는 InferModel async API를 권장 경로로 설명한다. Python
tutorial과 API reference의 핵심 규칙은 `wait_for_async_ready()`를 먼저 호출하고,
`run_async()`가 반환된 뒤에도 callback 또는 `AsyncInferJob.wait()`가 완료될 때까지
bindings와 모든 입출력 버퍼를 유지하는 것이다. callback은 짧게 실행해야 하며 async
오류가 발생하면 해당 inference pipeline은 종료된 것으로 취급해야 한다.

이 구현의 근거 문서는 사용자가 제공한 `hailort_4.24.0_user_guide (1).pdf`의
InferModel 비동기 tutorial(문서 페이지 66–68), runtime concepts(81), Python API
reference(303–305)와 `hailo_ai_sw_suite_2026-07.pdf`에 포함된 Release 2026-06
호환표 및 Hailo-8/10H 공존 절(13–15)이다. 후자의 파일명과 문서 내부 release 표기는
서로 다르므로 버전 표는 문서 내부 표기를 따른다.

Hailo-8용 HailoRT 4.x와 Hailo-10H용 HailoRT 5.x의 `libhailort`는 한 호스트에서
동시에 설치할 수 없다. 두 장치를 같은 호스트에서 사용할 때는 각 버전을 별도 Linux
container에 설치한다.

## 프레임워크 runtime 계약

Hailo native async는 현재 framework 공용 계약을 따른다.

- target의 `native_async` capability: `async_queue`에서 native executor를 선택
- `create_native_backend()`: load된 Hailo runtime을 callback backend로 제공
- `native_async_max_batch_size()`: InferModel이 받는 framework batch 상한 반환
- `native_async_max_inflight()`: SDK가 보고한 async queue 크기 반환
- `native_async_completion_timeout_sec()`: framework native completion deadline 반환
- `submit_async(inputs, callback)`: job을 제출하고 `NativeAsyncOutcome`으로 callback

`HailoRuntime`은 load된 객체에 `ConfiguredInferModel.wait_for_async_ready`와
`run_async`가 모두 있을 때만 `create_native_backend()`를 허용한다. 구형 환경에서
`InferVStreams` fallback이 선택되면 동기 `e2e`는 계속 사용할 수 있지만,
`native_async` capability를 요구하는 `async_queue`는 InferModel API 오류로 종료한다.

제출 흐름은 다음과 같다.

1. framework batch를 HEF input 이름, dtype, NHWC layout에 맞춘다.
2. bounded adapter registry에 job을 등록하고 adapter job ID를 즉시 반환한다.
3. 단일 submission worker가 각 frame의 bindings와 output buffer를 생성한다.
4. `wait_for_async_ready(timeout_ms=..., frames_count=batch_size)`로 SDK backpressure를
   확인한 뒤 `run_async(bindings, callback)`을 호출한다.
5. Hailo SDK callback은 completion record만 bounded completion worker에 넘기고 즉시
   반환한다. completion worker가 오류를 failure outcome으로 변환하거나 output을
   framework 소유 NumPy 데이터로 복사한다. ragged NMS의 자식 배열도 깊은 복사한다.
6. framework callback을 알리기 전에 adapter registry ownership을 정리하고, `unload()`는
   submission/completion worker가 모두 끝날 때까지 기다린 다음 Hailo context를 해제한다.
7. framework completion이 결과를 terminal commit한 뒤 one-shot retirement lease가
   dispatch token을 ACK하고 native permit을 정확히 한 번 반환한다. vendor job ID는
   진단에만 사용한다.

여러 framework worker가 같은 configured model을 사용하므로 SDK의 bindings 생성,
ready 확인, job 제출 구간은 submission worker로 직렬화한다. `run_async()`가 반환되면
다음 제출을 처리하므로 제출된 job들은 Hailo async queue 안에서 동시에 실행될 수 있다.
adapter registry 자체도 SDK queue 크기로 제한되어 내부 executor queue가 무한히 늘지
않는다. callback이 오기 전에는 `unload()`를 거부한다. Hailo completion 오류,
`run_async()` 실패, callback protocol 오류, output ownership 복사 실패는 fail-closed로
이후 제출을 거부한다. `wait_for_async_ready()` 실패는 job 제출 전 backpressure 실패이므로
해당 request만 실패시키고 다음 제출은 허용한다.

## runtime 옵션

| 옵션 | 기본값 | 의미 |
|---|---:|---|
| `async_timeout_ms` | `10000` | 아래 두 timeout의 공통 fallback |
| `async_ready_timeout_ms` | `async_timeout_ms` | SDK queue가 batch를 받을 수 있을 때까지 대기 |
| `async_completion_timeout_ms` | `async_timeout_ms` | ready 성공 뒤 completion에 추가로 확보할 논리 budget |
| `timeout_ms` | `1000` | 기존 동기 `run()` timeout |

Hailo가 계산하는 request deadline은 `(SDK async queue 크기 ×
async_ready_timeout_ms) + async_completion_timeout_ms`다. native executor에는 이 값과
`--flush-timeout-sec` 중 작은 값이 적용된다. submission worker가 SDK 호출을 직렬화하므로
마지막 queued job이 앞선 모든 job의 최대 ready 대기를 거친 뒤에도 자기 ready 구간과
completion budget을 확보하도록 계산한다. 입력 정규화와 framework queue 대기도 이 전체
request deadline에 포함된다.

실제 native in-flight 상한은 `--worker-count`, framework queue capacity,
HailoRT `get_async_queue_size()` 중 최솟값이다.
`--batch-size`가 2 이상이면 Hailo InferModel batch size와 framework의 동적 최대 batch
크기에 함께 적용된다. 먼저 `batch-size=1`, `worker-count=1`로 정확도를 확인하고 SDK가
보고한 queue 크기 안에서 worker 수를 늘리는 순서를 권장한다.

## 실행 예시

아래 예시는 실제 HEF와 데이터셋 경로로 바꿔 실행한다. `worker-count=2`는 예시이며
장치가 보고한 async queue 크기가 2 이상일 때만 유효하다.

### ResNet50

```bash
cd framework

.venv/bin/python src/main.py \
  --model resnet50 \
  --target hailo8 \
  --hef /path/to/hailo8_resnet50.hef \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 2 \
  --max-samples 100 \
  --min-samples 100 \
  --runtime-option async_timeout_ms=10000 \
  --monitor

.venv/bin/python src/main.py \
  --model resnet50 \
  --target hailo10h \
  --hef /path/to/hailo10h_resnet50.hef \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 2 \
  --max-samples 100 \
  --min-samples 100 \
  --runtime-option async_timeout_ms=10000 \
  --monitor
```

### YOLOv5m

```bash
cd framework

.venv/bin/python src/main.py \
  --model yolov5m \
  --target hailo8 \
  --hef /path/to/hailo8_yolov5m.hef \
  --dataset datasets/coco128 \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 2 \
  --max-samples 100 \
  --min-samples 100 \
  --runtime-option async_timeout_ms=10000 \
  --monitor

.venv/bin/python src/main.py \
  --model yolov5m \
  --target hailo10h \
  --hef /path/to/hailo10h_yolov5m.hef \
  --dataset datasets/coco128 \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 2 \
  --max-samples 100 \
  --min-samples 100 \
  --runtime-option async_timeout_ms=10000 \
  --monitor
```

YOLOv5m HEF가 NMS postprocess를 포함하면 기존 `HailoYoloNMSDecoder`가 dense
class-major tensor와 HailoRT 버전별 ragged per-class container를 모두 공통 detection
형식으로 변환한다. confidence나 좌표 규약이 HEF와 다르면 기존
`hailo_nms_conf_threshold`, `hailo_nms_box_order`, `tf_nms_format` runtime 옵션을
함께 사용한다.

## 검증 범위

SDK가 없는 CI에서는 fake PyHailoRT surface로 다음을 검증한다.

- ResNet50 batch별 bindings identity, `frames_count`, dense output과 callback timing
- YOLOv5m dense/ragged NMS output shape 및 callback 이후 buffer ownership
- Hailo completion 오류, pipeline 폐쇄, in-flight unload 차단
- nonblocking adapter submit, SDK callback offload, callback-finalization/unload race
- ready 실패 복구, run/protocol/copy 실패 fail-closed, inline/out-of-order callback
- binding/API 실패와 ready timeout의 분리, 마지막 queued job의 timeout budget
- SDK queue 크기로 제한되는 adapter submission backlog
- Hailo async adapter와 `NativeAsyncRuntimeExecutor`의 실제 callback 계약
- Hailo-8/10 target의 `native_async` capability 및 CLI executor 주입

실장 성능과 장치별 queue 크기는 HailoRT/PyHailoRT와 HEF가 설치된 각 target에서 별도로
확인해야 한다.

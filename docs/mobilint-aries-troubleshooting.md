# Mobilint ARIES 통합 및 트러블슈팅 기록

이 문서는 Mobilint ARIES를 ML-HW-Benchmark-Framework에 연결하면서 실제로
발생한 설치, 모델 준비, 전처리·후처리, 동기·비동기 실행, 모니터링 문제를
재현 가능한 형태로 기록한다. 운영자가 같은 증상을 빠르게 해결하는 것과
개발자가 어댑터의 설계 배경 및 남은 문제를 이해하는 것을 함께 목표로 한다.

## 1. 현재 범위와 검증 상태

현재 Mobilint 연동 범위는 런타임, native async bridge, 하드웨어 모니터링이다.
컴파일러 연동은 포함하지 않으므로 벤더가 미리 컴파일한 `.mxq` artifact 또는
Mobilint Model Zoo 형식의 로컬 LLM 디렉터리가 필요하다.

ARIES와 REGULUS raw 모델은 동일한 `mobilint` qb Runtime adapter를 사용하지만
장치를 자동 추측하지 않는다. 실행할 때 각각 `mobilint-aries` 또는
`mobilint-regulus` target을 명시해야 한다. 이 문서의 실장비 결과는 ARIES만
대상으로 하며 REGULUS는 아직 검증하지 않았다.

| 장치 | 모델 | 동기 E2E | 비동기 큐 | 비고 |
|---|---|---|---|---|
| ARIES | ResNet50 ImageNet1K V2 | 성공 | 3,000/3,000 valid | 전처리와 전력 측정 포함 |
| ARIES | YOLOv5m | 성공 | 128/128 valid | raw head decode/NMS 포함 |
| ARIES | BERT SST-2 | 미검증 | 미검증 | 범용 loader/evaluator만 존재 |
| ARIES | BERT SQuAD QA | 미검증 | 미검증 | 범용 loader/evaluator만 존재 |
| ARIES | PatchTST | 미검증 | 미검증 | Mobilint MXQ 계약 미확인 |
| ARIES | Llama 3.2 3B | 미검증 | 미검증 | ARIES LLM runtime은 구현됨 |
| ARIES | Llama 3.1 8B | 미검증 | 미검증 | Model Zoo 호환성과 메모리 미확인 |
| REGULUS | 전체 | 미검증 | 미검증 | 실제 장치 검증 필요 |

ResNet50과 YOLOv5m의 최종 valid 결과와 별개로 Mobilint native async adapter에는
아직 간헐적 슬롯 반환 경쟁 조건이 남아 있다. 한 번의 3,000개 성공 실행만으로
경쟁 조건이 해결됐다고 판단하지 않는다. 자세한 내용은
"10. native async 슬롯 소유권 경쟁 조건"에서 다룬다.

## 2. 검증 환경

최종 실장비 확인 환경은 다음과 같다.

| 항목 | 값 |
|---|---|
| 장치 | Mobilint ARIES, `/dev/aries0` |
| driver package | `mobilint-aries-driver 1.13` |
| kernel driver | `aries.ko`, runtime 표시 `1.13.0` |
| CLI | `mobilint-cli 1.3.2` |
| qb Runtime | `mobilint-qb-runtime 1.3.2`, Python `qbruntime v1.3.2` |
| firmware | `1.2` |
| NPU memory | 16,384 MB |
| Python environment | uv 기반 `.venv-mobilint`, Python 3.12 |

버전은 새 실험마다 다시 기록한다. 드라이버, CLI, runtime이 서로 다른 release
계열이면 CLI가 ARIES를 ARIES2로 표시하거나 driver version을 예상과 다르게
표시할 수 있다.

## 3. 설치 및 장치 인식

### 3.1 기본 점검

```bash
dpkg-query -W \
  -f='${Package}\t${Status}\t${Version}\n' \
  mobilint-aries-driver \
  mobilint-qb-runtime \
  mobilint-cli

lsmod | grep '^aries'
ls -l /dev/aries*
modinfo -n aries
command -v mobilint-cli
mobilint-cli status
```

정상 환경에서는 다음 조건을 만족해야 한다.

- 세 패키지가 `install ok installed` 상태다.
- `aries` kernel module이 로드되어 있다.
- `/dev/aries0`가 존재하고 실행 사용자에게 접근 권한이 있다.
- `mobilint-cli status`가 ARIES, firmware, memory, utilization을 표시한다.

`modinfo -F version aries` 또는 `/sys/module/aries/version`이 빈 문자열을
반환하더라도 곧바로 driver 미설치로 판단하지 않는다. 이 환경에서는 DKMS module
metadata에 version field가 없었지만 패키지 상태, module path, device node 및 CLI는
정상이었다.

### 3.2 CLI 경로가 바뀐 뒤 셸이 이전 경로를 기억하는 문제

관측된 증상:

```text
bash: /usr/local/bin/mobilint-cli: No such file or directory
```

실제 binary는 `/usr/bin/mobilint-cli`에 설치되어 있었다. 셸 command hash를
초기화한 뒤 다시 확인한다.

```bash
hash -r
type -a mobilint-cli
command -v mobilint-cli
ls -l /usr/bin/mobilint-cli /usr/local/bin/mobilint-cli 2>&1
```

### 3.3 Secure Boot와 재부팅

```bash
mokutil --sb-state 2>&1 || true
```

이 명령은 Secure Boot의 현재 상태를 조회할 뿐 비활성화하지 않는다. 이미
`lsmod`, `/dev/aries0`, `mobilint-cli status`가 정상이라면 Secure Boot를 이유로
드라이버를 다시 설치하거나 재부팅할 필요가 없다. Secure Boot가 활성화되어 있고
DKMS module 서명이 거부되는 경우에만 MOK 등록 또는 부팅 설정 변경을 검토한다.

재부팅이 어려운 서버에서는 먼저 device 사용 프로세스를 확인한다.

```bash
sudo fuser -v /dev/aries0
lsmod | grep '^aries'
```

module reload는 device 사용자가 없고 작업 중인 NPU process가 없음을 확인한 뒤에만
수행한다. 실행 중인 프로세스가 있는 상태에서 module을 강제로 제거하지 않는다.

### 3.4 uv 환경에서 Python SDK 확인

```bash
.venv-mobilint/bin/python - <<'PY'
import qbruntime
import mbltml

print("qbruntime version:", qbruntime.__version__)
print("qbruntime path:", qbruntime.__file__)
print("mbltml path:", mbltml.__file__)
PY
```

system Python이 아니라 실제 벤치마크에 사용하는 uv virtual environment에서
확인해야 한다.

## 4. Model Zoo 및 artifact 준비

### 4.1 ResNet50

검증 artifact 경로:

```text
framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq
```

Mobilint Model Zoo CLI로 artifact와 SDK의 기본 동작을 framework보다 먼저 확인할
수 있다.

```bash
TEST_IMAGE="/absolute/path/to/datasets/imagenet_1k/val/image_0001.jpg"
RESNET_MXQ="/absolute/path/to/framework/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq"

MBLT_MODEL_ZOO_VERBOSE=true \
.venv-mobilint/bin/mblt-model-zoo predict \
  --source "$TEST_IMAGE" \
  --model resnet50 \
  --model-type IMAGENET1K_V2 \
  --model-path "$RESNET_MXQ" \
  --core-mode global8 \
  --dev-no 0 \
  --output /tmp/mobilint-resnet50-result.jpg
```

관측된 정상 결과는 `uint8` 기반 모델 초기화, 추론, Top-K label 출력 및 결과 이미지
저장이었다.

### 4.2 `libGL.so.1` 누락

Model Zoo CLI가 내부에서 OpenCV를 import하면서 다음 오류가 발생할 수 있다.

```text
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

Ubuntu 계열에서는 system dependency를 설치한다.

```bash
sudo apt-get update
sudo apt-get install -y libgl1
```

이는 qbruntime 오류가 아니라 OpenCV native dependency 오류다.

### 4.3 YOLOv5m Hugging Face 파일 경로

저장소는 `mobilint/YOLOv5m`이며 root의 `yolov5m.mxq`를 직접 요청하면 404가
발생했다.

```text
RemoteEntryNotFoundError: .../mobilint/YOLOv5m/resolve/main/yolov5m.mxq
```

실제 artifact는 저장소의 `aries/yolov5m.mxq`에 있다. 파일 목록이 확실하지 않으면
저장소 snapshot 전체를 받아 구조를 보존한다.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mobilint/YOLOv5m",
    local_dir="framework/models/mobilint/yolov5m",
)
```

최종 검증 경로:

```text
framework/models/mobilint/yolov5m/aries/yolov5m.mxq
```

## 5. 데이터셋 경로 문제

### 5.1 ImageNet

Hugging Face `datasets` 최신 버전에서는 `trust_remote_code` 기반 dataset script를
지원하지 않으며, namespace가 없는 잘못된 HF URI를 처리하다 다음 오류가 발생했다.

```text
`trust_remote_code` is not supported anymore.
Invalid HF URI ... Repository id must be 'namespace/name'
```

실장비 검증에서는 로컬 ImageNet 디렉터리를 사용했다.

```text
datasets/imagenet_1k/
├── val/
└── val_labels.txt
```

`--dataset`에는 `val` 디렉터리가 아니라 `imagenet_1k` root를 전달한다.

### 5.2 COCO128

관측된 오류:

```text
FileNotFoundError: '/path/to/coco/images/train2017' 경로에 이미지가 존재하지 않습니다.
```

원인은 shell 변수에 placeholder `/path/to/coco`가 남아 있었기 때문이다. 실제
구조는 다음과 같았다.

```text
framework/datasets/coco128/
├── images/train2017/
└── labels/train2017/
```

따라서 다음처럼 dataset root를 전달한다.

```bash
--dataset /absolute/path/to/framework/datasets/coco128
```

resolver는 `images/val2017`을 우선 탐색하고, 없으면 `images/train2017`과
`labels/train2017`을 사용한다.

## 6. 모델 입출력 계약

### 6.1 ResNet50

| 항목 | 값 |
|---|---|
| artifact | `resnet50_IMAGENET1K_V2.mxq` |
| 입력 dtype | `uint8` |
| 입력 layout | NHWC |
| 입력 shape | `(1, 224, 224, 3)` |
| 출력 | class logits, runtime에서 `(1, 1, 1, 1000)`으로 관측 |

초기 generic image loader는 `float32` normalized tensor를 생성했고 다음 오류가
발생했다.

```text
Invalid input data type - Input: Float32, Supported: Uint8
QbRuntimeError: Model_DtypeMismatched occurred.
```

해결은 runtime에서 임의 cast하는 것이 아니라 artifact별 전처리 계약을 dataloader와
preprocessor에 명시하는 것이었다. `mobilint-resnet50-imagenet1k-v2` profile은
Mobilint Model Zoo와 동일한 resize/crop 및 `uint8 NHWC` 계약을 적용한다.

### 6.2 YOLOv5m

실제 MXQ metadata와 전처리 결과:

```text
Input dtype : DataType.Uint8
Input shapes: [(640, 640, 3)]
Output shapes: [(20, 20, 255), (40, 40, 255), (80, 80, 255)]
Tensor shape : (640, 640, 3)
Tensor dtype : uint8
Metadata     : {'ratio_pad': ((1.28, 1.28), (0, 80))}
```

세 출력은 완성된 detection 목록이 아니라 stride별 YOLO raw prediction head다.
각 마지막 차원 `255`는 anchor별 box, objectness, class score를 묶은 값이다. 따라서
anchor/grid decode, sigmoid, confidence filtering, letterbox 좌표 복원, NMS가 필요하다.

`mobilint-yolov5m-default` profile과 Mobilint YOLO decoder가 이 계약을 담당한다.
검증 threshold는 confidence `0.001`, IoU `0.65`, 최대 detection `300`이다.

## 7. 동기 E2E 측정 경계

동기 `e2e`는 dataloader에서 배치를 가져온 뒤 runtime 호출과 completion 처리를
순차 실행한다. 최종 품질 metric은 모든 sample이 끝난 뒤 evaluator가 계산한다.

최종 출력의 `Average Latency`, `P99 Latency`, `Samples/s` 또는 `FPS`는 runtime
호출 timing을 기반으로 한다. `Samples/s`와 `FPS`는 평균 runtime latency의 역수이며
데이터 로딩과 후처리를 포함한 실제 wall-clock 처리량이 아니다.

## 8. 비동기 큐 구조와 측정 경계

Mobilint async 실행에는 역할이 다른 계층이 연결된다.

```text
Offline producer
  -> framework bounded request queue
  -> resident worker
  -> NativeAsyncRuntimeExecutor
  -> MobilintNativeBackend
  -> qbruntime.Model.infer_async()
  -> qbruntime Future
  -> framework callback
  -> CompletionCoordinator
  -> decoder/evaluator
```

Mobilint adapter가 SDK 앞에 별도 요청 큐를 하나 더 두는 구조는 아니다. framework
queue가 요청 identity, backpressure, scheduling을 소유하고 adapter는 SDK Future를
callback 계약으로 연결한다.

요청별 async timing은 다음과 같다.

```text
issued -> enqueued -> runtime_started -> runtime_finished -> completed
       submit_wait     queue_wait          service_time      completion_overhead
```

`async_e2e_latency`는 submit wait, queue wait, service time, completion overhead를
포함한다. `completion_overhead`에는 completion queue 체류와 decoder, label 준비,
`evaluator.add_batch()`가 포함된다. 최종 `evaluator.compute()`와 결과 파일 저장은
요청 E2E에 포함되지 않는다.

따라서 YOLOv5m처럼 decode/NMS가 무거운 모델은 SDK latency가 짧아도 async E2E와
queue wait가 커질 수 있다. `conf_threshold=0.001`에서는 이미지당 평균 detection이
약 181개여서 후처리 비용이 특히 크게 관측됐다.

## 9. 비동기 메트릭 전체 rebuild 성능 문제

### 9.1 증상

수정 전 ResNet50 1,000개 실행:

```text
measurement duration       62.63 s
completed samples/s        15.97
runtime average latency     5.90 ms
async E2E P99            3123.90 ms
queue wait P99           2774.32 ms
service time P99          323.07 ms
completion overhead mean   63.58 ms
```

NPU runtime은 수 ms인데 framework 처리량은 15.97 samples/s에 불과했고, request ID가
클수록 service와 completion 시간이 증가했다. queue depth 평균도 15.42/16으로 거의
항상 가득 찼다.

### 9.2 원인

수정 전 `AsyncMetricsCollector.record_terminal()`은 terminal마다 과거 outcome 전체를
다시 계산했다.

```python
state.terminal_times[request_id] = completed_ns
_rebuild_outcome_accounting_locked(state)
```

rebuild는 accepted/rejected outcome, queue transition, inflight event를 모두 다시
순회하고 정렬했다. 일부 집합 재구성은 중첩 순회라 rebuild 한 번이 최대 `O(k^2)`,
이를 매 요청마다 실행해 전체 hot path가 최악 `O(N^3)`까지 증가했다. completion
thread가 Python lock과 GIL을 오래 점유하면서 Mobilint Future callback과 worker도
간접적으로 지연됐다.

### 9.3 해결

accepted, rejected, terminal counter와 inflight gauge를 정상 경로에서 증분
갱신하고, 전체 rebuild는 dirty recovery 또는 최종 정산에만 사용하도록 변경했다.

관련 변경:

- PR: [#32 비동기 metrics 회계를 증분 처리로 전환](https://github.com/hunterhhunter/ML-HW-Benchmark-Framework/pull/32)
- 수정 commit: `6b81b4d`
- main merge commit: `3e13400`

검증 branch가 main보다 오래된 경우 다음 코드가 남아 있는지 확인한다.

```bash
sed -n '/def record_terminal/,/def finalize/p' \
  framework/src/core/async_inference/metrics.py | sed -n '1,90p'
```

수정 전:

```python
state.terminal_times[request_id] = completed_ns
_rebuild_outcome_accounting_locked(state)
```

수정 후 정상 경로:

```python
if request_id not in state.terminal_times:
    state.terminal_times[request_id] = completed_ns
    _apply_terminal_inflight_locked(state, completed_ns)
```

별도 검증 branch에는 main 전체 merge보다 해당 commit만 가져오는 방법이 안전할 수
있다.

```bash
git fetch origin
git cherry-pick 6b81b4d
```

문서 파일의 modify/delete conflict가 발생하면 현재 branch에서 이미 삭제된 문서는
삭제 상태를 유지하고 code/test 변경만 반영한다. 이때 `git add .`로 dataset이나
결과 CSV까지 실수로 stage하지 않는다.

## 10. native async 슬롯 소유권 경쟁 조건

### 10.1 증상

메트릭 성능 수정 후 ResNet50 1,000개 실행에서 한 번 다음 결과가 발생했다.

```text
submitted=1000 accepted=1000 completed=999 failed=1
request_id=213
error_type=RuntimeError
error_message="native async submission failed"
timed_out=false
```

### 10.2 높은 신뢰도의 원인

`MobilintNativeBackend`는 SDK 동시 실행 수를 `_slots` semaphore로 제한한다.
slot은 `infer_async()` 전에 비차단 획득하지만 현재 구현에서는 Future 완료가 아니라
framework callback 반환 뒤에 해제된다.

```text
Future.get() 완료
-> callback에서 framework dispatch event 설정
-> worker/completion이 다음 dispatch permit 반환
-> 다음 요청이 backend.submit_async() 진입
-> 이전 waiter가 callback에서 아직 반환되지 않아 SDK slot 미반환
-> acquire(blocking=False) 실패
-> callback 반환 뒤에야 slot release
```

SDK 물리 작업 소유권과 framework callback/job 소유권의 종료 시점이 결합된 것이
문제다. slot은 Future가 terminal이고 출력 정규화가 끝난 직후 반환해야 한다.
`_jobs` 등록은 shutdown이 callback 종료를 기다릴 수 있도록 callback 반환까지
유지해야 한다.

권장 순서:

```text
Future.get() terminal
-> 출력 정규화
-> SDK slot exactly-once 반환
-> framework callback
-> callback 반환
-> job/input 참조 정리
```

필요한 회귀 테스트:

- activation slot 1에서 첫 callback을 block한 채 두 번째 submit이 성공하는지
- callback 예외 후 slot이 누수되지 않는지
- Future 실패 후 다음 submit이 가능한지
- waiter thread 시작 실패와 inline fallback에서도 exactly-once release인지
- callback 실행 중 shutdown은 callback이 끝날 때까지 기다리는지
- 실제 ARIES에서 1,000/3,000개를 여러 번 반복해 failed가 항상 0인지

한 번의 3,000/3,000 valid 실행은 경쟁 조건이 발생하지 않은 증거이지 코드상
경쟁 조건이 제거된 증거가 아니다.

## 11. worker count와 activation slot 불일치

관측된 오류:

```text
ValueError: worker_count=2 exceeds runtime capability 1
```

`mobilint-aries` target의 안전 기본값은 `activation_slots=1`이다. worker를 2로
설정하면서 activation slot을 명시하지 않으면 engine이 모델 실행 전에 거부한다.

ARIES와 해당 MXQ에서 SDK activation slot 2를 직접 확인한 경우에만 다음 두 옵션을
함께 사용한다.

```bash
--worker-count 2 \
--runtime-option activation_slots=2
```

안정성 우선 실행은 worker 1과 기본 slot 1을 사용한다. worker 수만 늘리거나 target
capability 검사를 우회하지 않는다.

## 12. 하드웨어 모니터링과 에너지

`--monitor`는 `mobilint`와 `system` collector를 함께 활성화한다. ARIES의
utilization, memory, temperature, power, current, voltage는 mbltml SDK에서 읽는다.
프레임워크는 power sample 사이를 사다리꼴 적분해 `hw_accel_energy_j`를 계산한다.

따라서 다음 세 값을 함께 확인해야 한다.

```text
hw_accel_energy_j
hw_accel_power_samples
hw_accel_power_sample_coverage
```

`power_sample_coverage=1.0`은 측정 경계의 전력 표본 coverage가 완전하다는 뜻이지
외부 전력계와 동일한 정확도를 보장한다는 뜻은 아니다. 샘플 수가 매우 적으면
짧은 spike가 평균과 적분에서 충분히 표현되지 않을 수 있다.

샘플당 에너지는 다음처럼 계산한다.

```text
energy_per_sample_j = hw_accel_energy_j / completed_samples
```

## 13. 전체 데이터셋 실행 명령

아래 예시는 저장소 배치가 다음과 같다고 가정한다.

```bash
REPO="/absolute/path/to/ML-HW-Benchmark-Framework"
FW="$REPO/framework"
PY="$REPO/.venv-mobilint/bin/python"

IMAGENET="$REPO/datasets/imagenet_1k"
COCO="$FW/datasets/coco128"

RESNET_MXQ="$FW/models/mobilint/resnet50/aries/resnet50_IMAGENET1K_V2.mxq"
YOLO_MXQ="$FW/models/mobilint/yolov5m/aries/yolov5m.mxq"
```

아래 비동기 명령의 worker 2/activation slot 2 설정은 최종 성능 결과를 재현하기
위한 값이다. 최초 smoke test 또는 안정성 우선 검증은 `--worker-count 1`로 바꾸고
`--runtime-option activation_slots=2`를 제거한다. 두 슬롯 실행은 10절의 경쟁 조건을
수정하고 반복 검증하기 전까지 완전히 안정적이라고 간주하지 않는다.

### 13.1 ResNet50 동기 E2E

```bash
cd "$FW"

"$PY" src/main.py \
  --model resnet50 \
  --target mobilint-aries \
  --artifact "$RESNET_MXQ" \
  --dataset "$IMAGENET" \
  --image-preprocess-profile auto \
  --layout NHWC \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --runtime-option core_mode=global8 \
  --no-compile \
  --monitor \
  --results-path "$FW/results/mobilint-aries-resnet50-e2e-full.csv"
```

`--max-steps`를 생략하면 dataloader의 전체 sample을 처리한다.

### 13.2 ResNet50 비동기 큐

```bash
IMAGENET_COUNT=$(find "$IMAGENET/val" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

"$PY" src/main.py \
  --model resnet50 \
  --target mobilint-aries \
  --artifact "$RESNET_MXQ" \
  --dataset "$IMAGENET" \
  --image-preprocess-profile auto \
  --layout NHWC \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --queue-capacity 16 \
  --worker-count 2 \
  --min-samples "$IMAGENET_COUNT" \
  --max-samples "$IMAGENET_COUNT" \
  --warmup 2 \
  --submit-timeout-sec 10 \
  --flush-timeout-sec 600 \
  --request-timeout-ms 30000 \
  --runtime-option core_mode=global8 \
  --runtime-option activation_slots=2 \
  --no-compile \
  --monitor \
  --save-request-trace \
  --results-path "$FW/results/mobilint-aries-resnet50-async-full.csv"
```

### 13.3 YOLOv5m 동기 E2E

```bash
"$PY" src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact "$YOLO_MXQ" \
  --dataset "$COCO" \
  --image-preprocess-profile auto \
  --layout NHWC \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --runtime-option core_mode=global8 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --no-compile \
  --monitor \
  --results-path "$FW/results/mobilint-aries-yolov5m-e2e-full.csv"
```

### 13.4 YOLOv5m 비동기 큐

```bash
COCO_COUNT=$(find "$COCO/images/train2017" -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | wc -l)

"$PY" src/main.py \
  --model yolov5m \
  --target mobilint-aries \
  --artifact "$YOLO_MXQ" \
  --dataset "$COCO" \
  --image-preprocess-profile auto \
  --layout NHWC \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --queue-capacity 16 \
  --worker-count 2 \
  --min-samples "$COCO_COUNT" \
  --max-samples "$COCO_COUNT" \
  --warmup 2 \
  --submit-timeout-sec 10 \
  --flush-timeout-sec 600 \
  --request-timeout-ms 30000 \
  --runtime-option core_mode=global8 \
  --runtime-option activation_slots=2 \
  --runtime-option conf_threshold=0.001 \
  --runtime-option iou_threshold=0.65 \
  --no-compile \
  --monitor \
  --save-request-trace \
  --results-path "$FW/results/mobilint-aries-yolov5m-async-full.csv"
```

## 14. 최종 실장비 결과

데이터셋은 로컬 ImageNet 3,000장과 COCO128 전체 128장이다. 공식 ImageNet val
50,000장 또는 COCO val2017 전체 결과로 해석하면 안 된다.

### 14.1 동기 E2E

| 모델 | 품질 지표 | 추론 평균 / P99 | 추론 처리율 | NPU 사용률 평균 | 평균 전력 | 샘플당 에너지 |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | Top-1 **80.80%**, Top-5 **95.03%** | **2.32 / 2.53 ms** | **430.94 samples/s** | **38.39%** | **17.24 W** | **55.81 mJ** |
| YOLOv5m | mAP@0.5 **0.7698**, 평균 탐지 **181.21개** | **7.09 / 8.45 ms** | **141.04 FPS** | **6.96%** | **15.89 W** | **1016.92 mJ** |

동기 `Samples/s`와 `FPS`는 runtime latency의 역수다.

### 14.2 비동기 큐

| 모델 | 품질 지표 | 추론 평균 / P99 | 실제 처리량 | E2E P50 / P99 | Queue wait P99 | Worker 사용률 |
|---|---|---:|---:|---:|---:|---:|
| ResNet50 | Top-1 **80.80%**, Top-5 **95.03%** | **1.95 / 3.76 ms** | **304.35 samples/s** | **55.92 / 71.54 ms** | **60.77 ms** | **88.75%** |
| YOLOv5m | mAP@0.5 **0.7698**, 평균 탐지 **181.21개** | **7.53 / 14.10 ms** | **26.30 samples/s** | **692.91 / 1125.12 ms** | **903.11 ms** | **95.98%** |

| 모델 | NPU 사용률 평균 / 최대 | 전력 평균 / 최대 | 총 에너지 | 샘플당 에너지 | 상태 |
|---|---:|---:|---:|---:|---|
| ResNet50 async | **36.86 / 39.05%** | **16.87 / 18.46 W** | **171.13 J** | **57.04 mJ** | 3,000/3,000 valid |
| YOLOv5m async | **10.49 / 66.22%** | **15.44 / 17.13 W** | **76.34 J** | **596.38 mJ** | 128/128 valid |

YOLOv5m async의 낮은 처리량과 큰 queue wait는 NPU 추론 시간만으로 설명되지 않는다.
낮은 confidence threshold에서 많은 detection을 decode/NMS하고 단일 completion
경로에서 평가하는 비용을 함께 확인해야 한다.

## 15. 결과 파일에서 오류 찾기

```bash
RUN_ID="<run-id>"
DETAIL="$FW/results/details/$RUN_ID.json"
TRACE="$FW/results/traces/$RUN_ID.jsonl"

jq '{
  invalid_reasons,
  counts,
  failure_types,
  failure_request_examples,
  lifecycle_errors,
  warnings
}' "$DETAIL"

jq -c '
  select(.status != "completed") |
  {
    request_id,
    sample_index,
    status,
    worker_id,
    error_type,
    error_message,
    timed_out
  }
' "$TRACE"
```

`async_run_status=valid`는 요청 회계, 최소 sample, timeout 등의 자체 유효성 조건을
통과했다는 의미다. 원하는 성능 수준을 보장한다는 의미는 아니다.

## 16. 남은 작업

- [ ] Mobilint native async SDK slot을 Future 물리 완료 직후 반환하고 callback/job
      소유권과 분리한다.
- [ ] callback block, callback 예외, Future 실패, inline fallback, shutdown 동시성을
      포함하는 결정적 회귀 테스트를 추가한다.
- [ ] ARIES에서 1,000/3,000개 async 실행을 여러 번 반복해 간헐 실패가 0인지 확인한다.
- [ ] YOLOv5m decoder/NMS와 completion 구간을 세분화해 queue 체류, decoder,
      evaluator 누산 시간을 별도 지표로 측정한다.
- [ ] ImageNet val 50,000장과 COCO val2017 전체로 품질·성능을 다시 측정한다.
- [ ] BERT SST-2, BERT SQuAD QA, PatchTST MXQ의 입출력 계약과 전처리를 추가한다.
- [ ] Llama 3.2 3B를 먼저 실장비 검증한 뒤 Llama 3.1 8B를 검증한다.
- [ ] REGULUS PCIe/USB에서 explicit target, runtime, async, monitor 계약을 검증한다.
- [ ] 외부 전력계와 mbltml 기반 power/energy 적분 결과를 교차 검증한다.

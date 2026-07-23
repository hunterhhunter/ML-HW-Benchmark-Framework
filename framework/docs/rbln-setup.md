# Rebellions RBLN-CA22 운영 가이드

이 문서의 모든 benchmark 명령은 repository의 `framework` 디렉터리에서
실행한다. 기본 사용법은 runtime·monitor·device 0을 함께 고정하는
`--target rbln-static`이다. 기존 backend 형식에서의 canonical 이름은
`--backend rbln`이지만, 장치와 monitor selector를 누락하지 않도록 target
형식을 권장한다.

## 1. 지원 범위

| 영역 | 현재 상태 | 계약 |
|---|---|---|
| ResNet50 | 지원 | static image classification, batch 1 |
| YOLOv5m | 지원 | static object detection, 기존 raw YOLO decoder 사용 |
| BERT base | 지원 | SST-2 classification과 SQuAD QA fixed profile |
| PatchTST FM R1 | 지원 | static time-series forecasting |
| Llama 3.2 3B / Llama 3.1 8B generation | 미지원 | 후속 in-process `rbln-vllm` target으로 계획 |
| `.rbln` compile / model download / artifact 배포 | 제외 | 이 branch는 CA22용 precompiled artifact만 실행 |
| 외부 OpenAI/HTTP serving server | 제외 | 후속 serving adapter 범위 |
| multi-NPU / tensor parallel | 제외 | device 0, `tensor_parallel_size=1`만 지원 |
| dynamic shape / shape bucketing | 제외 | inspect된 fixed positive shape만 지원 |

`async_queue --scenario server_like`는 현재 지원하는 in-process benchmark
부하 형식이다. 외부 client가 HTTP/OpenAI-compatible server에 접속하는 serving
형식과는 다르며, 그 외부 serving adapter는 후속 확장이다.

## 2. 검증된 출발 환경

최초 검증 server의 기준은 다음과 같다. Adapter가 package version을
임의로 upgrade하거나 이 버전과의 strict equality를 강제하지는 않지만,
재현 가능한 결과에는 실제 버전을 함께 기록해야 한다.

| 항목 | 검증값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| Python package | `rebel-compiler==0.11.0` |
| KMD / firmware | 3.2.2 / 3.2.2 |
| NPU | device 0, `RBLN-CA22`, `/dev/rbln0` |
| device memory | 16,877,879,296 bytes |
| PCI | `0000:ab:00.0`, NUMA node 1, 32.0 GT/s x8 |

## 3. 실행 전 점검

```bash
python3 --version
python3 -m pip show rebel-compiler
python3 -m pip list --format=freeze | grep -Ei '^(rebel|optimum-rbln|vllm-rbln|torch|transformers|tokenizers|vllm)'
cat /etc/os-release
command -v rbln-smi
rbln-smi -q
rbln-smi -j
ls -l /dev/rbln0
test -r /dev/rbln0 && test -w /dev/rbln0
```

Python API에서 device 0을 독립적으로 확인한다.

```bash
python3 - <<'PY'
import rebel

print("available", rebel.npu_is_available(0))
print("name", rebel.get_npu_name(0))
print("count", rebel.device_count())
PY
```

benchmark 시작 전에 device 0 context가 비어 있어야 한다. 아래 명령은
device 0 context를 출력하고 하나라도 있으면 non-zero로 종료한다.

```bash
rbln-smi -j | python3 -c 'import json,sys; payload=json.load(sys.stdin); contexts=[item for item in payload.get("contexts", []) if isinstance(item, dict) and str(item.get("npu")) == "0"]; print(json.dumps(contexts, indent=2)); raise SystemExit(1 if contexts else 0)'
```

KMD/FW, NPU name, memory total, status `normal`, device node 권한 중 하나라도
기대값과 다르면 artifact를 load하지 말고 server 설치 상태부터
확인한다.

## 4. Artifact 배치와 inspect

표준 배치 규칙은 repository root 기준
`framework/models/rbln/{model-name}/model.rbln`이다. Task 8에서 사용하는
정확한 다섯 경로는 다음과 같다.

```text
framework/models/rbln/resnet50/model.rbln
framework/models/rbln/yolov5m/model.rbln
framework/models/rbln/bert-base-uncased/model.rbln
framework/models/rbln/bert-base-uncased-squad-v1/model.rbln
framework/models/rbln/patchtst-fm-r1/model.rbln
```

이 경로에는 raw Hugging Face model directory, ONNX file, 다른 NPU용
artifact가 아니라 `RBLN-CA22`용으로 미리 compile된 `.rbln` file이
필요하다. Framework compile integration은 후속 작업으로 유보되었다.

각 artifact를 실행하기 전에 아래 allowlist만 inspect하여 별도로
기록한다. `ARTIFACT`를 위 다섯 file 중 하나로 바꿔 반복한다.

```bash
ARTIFACT=models/rbln/resnet50/model.rbln python3 - <<'PY'
import json
import os
from collections.abc import Mapping
from rebel import RBLNCompiledModel

metadata = RBLNCompiledModel.inspect(os.environ["ARTIFACT"])

def field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

def tensor(value):
    shape = field(value, "shape")
    return {
        "name": field(value, "name"),
        "shape": list(shape) if shape is not None else None,
        "dtype": str(field(value, "dtype")),
    }

selected = {
    "compiler_version": field(metadata, "compiler_version"),
    "npu": field(metadata, "npu"),
    "tensor_parallel_size": field(metadata, "tensor_parallel_size"),
    "uuid": field(metadata, "uuid"),
    "alloc_per_node": field(metadata, "alloc_per_node"),
    "inputs": [tensor(value) for value in (field(metadata, "inputs") or ())],
    "outputs": [tensor(value) for value in (field(metadata, "outputs") or ())],
}
print(json.dumps(selected, indent=2, default=str))
PY
```

inspect 결과는 `npu == RBLN-CA22`, 명시된 `tensor_parallel_size == 1`, 모든
dimension이 1 이상인 fixed shape여야 한다. SDK 0.11 single-device artifact는
`tensor_parallel_size`를 `null`로 생략할 수 있으며 이 경우 provenance key도
생략한다. 또한 input/output name·shape과
input dtype이 아래 model profile과 정확히 일치해야 한다. 불일치를
숨기려고 runtime에서 reshape, padding, truncation, transpose, dtype cast를
추가하지 않는다.

## 5. Model contract

| profile | input contract | output contract | dataset / 주의점 |
|---|---|---|---|
| `resnet50` | single input `float32 (1,3,224,224)` | `output float32 (1,1000)` | `datasets/imagenet_1k`; single input과 single unnamed output positional fallback 가능 |
| `yolov5m` | single input `float32 (1,3,640,640)` | raw `output float32 (1,25200,85)` | `datasets/coco128`; NMS 포함/별도 layout은 기존 decoder와 호환되지 않음 |
| `bert-base-uncased` | `input_ids int64 (1,128)`, `attention_mask int64 (1,128)` | `logits float32 (1,2)` | `datasets/sst2_numpy`; multi-input name 정확히 일치 |
| `bert-base-uncased-squad-v1` | `input_ids int64 (1,384)`, `attention_mask int64 (1,384)` | `start_logits float32 (1,384)`, `end_logits float32 (1,384)` | `datasets/squad_numpy`; input/output 순서를 name으로 검증 |
| `patchtst-fm-r1` | `past_values float32 (1,512,7)`, `past_observed_mask bool (1,512,7)` | `output float32 (1,96,7)` | `datasets/etth1/ETTh1.csv`; bool mask를 float로 cast하지 않음 |

비전 단일 input의 artifact name은 positional fallback이 가능하다. Artifact와
profile output이 각각 하나이고 artifact output name만 `null`이면 유일한 profile
output name으로 binding한다. 다중 input/output name은 profile과 일치해야 한다.
Artifact의 실제 inspect 결과가
표와 다르면 해당 artifact를 거부하고 명시적 model profile/decoder를 별도로
설계한다.

## 6. Sync E2E smoke와 full run

첫 가동 검사는 다음 명령을 그대로 사용한다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 10 \
  --monitor \
  --results-path results/rbln-resnet50-e2e.csv
```

smoke가 exit 0이고 output/evaluator/context 검증을 통과하면 `--max-steps`를
제거해 전체 dataset을 실행한다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --monitor \
  --results-path results/rbln-resnet50-e2e-full.csv
```

## 7. Async offline와 server-like

첫 async gate는 `worker_count=1`, queue 16, SDK input preparation
`async_parallel=1` 기본값으로 실행한다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 16 \
  --min-samples 100 \
  --warmup 2 \
  --flush-timeout-sec 300 \
  --save-request-trace \
  --monitor \
  --results-path results/rbln-resnet50-async-w1.csv
```

offline gate가 lifecycle invariant를 모두 만족한 뒤에만 server-like QPS를
낮은 값부터 올린다.

```bash
python3 -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 10 \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 16 \
  --min-duration-sec 30 \
  --min-samples 100 \
  --latency-slo-ms 100 \
  --warmup 2 \
  --flush-timeout-sec 300 \
  --save-request-trace \
  --monitor \
  --results-path results/rbln-resnet50-server-like-qps10.csv
```

## 8. Concurrency tuning 순서

다음 three-stage sweep을 한 번에 한 variable만 바꿔 실행한다.
각 설정은 3회 반복하고 exact-count, timeout 0, context 0인 run만
비교한다.

1. Queue 64와 `--runtime-option async_parallel=1`을 고정하고
   `--worker-count 1`, `2`, `4`, `8`을 각각 3회 실행한다.
2. 1단계의 최고 stable worker count에서만
   `--runtime-option async_parallel=1`과
   `--runtime-option async_parallel=2`를 각각 3회 비교한다.
3. 최고 stable worker/parallel pair에서 `--queue-capacity 16`, `64`,
   `256`을 각각 3회 비교한다.
4. 선택한 offline 설정을 server-like에 적용한 뒤 target QPS만
   점진적으로 높인다.

| option | 소유자 | 의미 |
|---|---|---|
| `--queue-capacity` | framework | accepted request의 bounded backlog 상한 |
| `--worker-count` | framework/executor | SDK에 동시 제출할 request 수와 executor inflight 상한 |
| `--runtime-option async_parallel=1` 또는 `2` | RBLN SDK | 하나의 `AsyncRuntime`이 input buffer를 준비하는 thread 설정 |

`worker_count` 증가는 `async_parallel=2`와 같은 의미가 아니다. SDK tuning은
반드시 `--runtime-option async_parallel=2`로만 명시하고, 최적 worker
count를 찾기 전에 동시에 바꾸지 않는다.

## 9. 지표와 monitoring 경계

`--monitor`는 device 0에 대해 exact argv `rbln-smi -b -j -d 0`을
shell 없이 호출한다. Vendor poll은 collector 내부에서 최소 1초 간격으로
throttle되며 command timeout은 2초다.

Async run은 다음 값을 함께 확인한다.

- `async_accepted_requests == async_completed_requests == async_evaluator_samples`
- `async_outstanding_requests == 0`, logical request 기준
  `async_timed_out_requests == 0`
- `async_native_inflight == 0`, `async_native_duplicate_callbacks == 0`,
  `async_native_late_callbacks == 0`, `async_native_submit_failures == 0`,
  `async_native_timeouts == 0`
- queue depth high-water가 configured capacity 이하
- throughput, p50/p95/p99 end-to-end latency, queue wait latency, service time
- NPU utilization, memory, temperature, power, energy, monitor coverage

| metric | 의미 |
|---|---|
| `async_timed_out_requests` | framework request deadline을 넘긴 logical terminal 수 |
| `async_native_inflight` | ACK·physical completion 조건이 모두 끝나지 않은 native dispatch 수 |
| `async_native_duplicate_callbacks`, `async_native_late_callbacks` | 첫 terminal 후 중복/지연 SDK callback 수 |
| `async_native_submit_failures`, `async_native_timeouts` | native submit 경계 실패와 native completion logical timeout 수 |
| `hw_accel_util` | device 0 NPU utilization percent |
| `hw_accel_mem_used_mb` | device 0 전체 사용 memory |
| `hw_accel_mem_proc_mb` | `rbln-smi` context PID가 현재 benchmark process와 일치할 때의 allocation |
| `hw_accel_temp_c` | device 0 temperature |
| `hw_accel_power_w` | device 0 whole-card power |
| `hw_accel_energy_j` | 유효한 whole-card power sample을 시간에 대해 적분한 energy |
| `hw_accel_monitor_attempts`, `hw_accel_monitor_successes`, `hw_accel_monitor_coverage` | 실제 vendor poll 시도, schema까지 유효한 poll, `successes / attempts` |

`hw_accel_monitor_coverage`는 실제 `rbln-smi` 성공 횟수를 시도 횟수로
나눈 값이다. `hw_accel_monitor_attempts >= hw_accel_monitor_successes >= 1`을
확인하고 coverage가 낮은 run은 power/temperature 비교에 사용하지 않는다.
없는 sensor는 0이 아니라 metric key 생략으로 표현된다.

`hw_accel_energy_j`는 카드 power sample을 사다리꼴로 적분한 값이다.
해당 Python process만의 에너지가 아니라 idle power와 같은 카드 전체
소비 전력을 포함한다. Warmup은 monitor 측정 구간 전에 실행되므로
measured energy와 async request counter에 포함되지 않는다.

## 10. Failure matrix

| failure | 의미 | 조치 |
|---|---|---|
| `rebel-compiler` import 실패 | optional SDK가 없거나 environment가 다름 | server의 검증된 Python에 package를 설치하고 `python3 -m pip show rebel-compiler` 재확인 |
| device 0 unavailable / NPU name mismatch | `/dev/rbln0`, driver, 권한 또는 CA22 selector 문제 | `rbln-smi -q/-j`, device node 권한, `npu_is_available(0)`, `get_npu_name(0)` 확인 |
| artifact target mismatch | `.rbln` target NPU가 detected `RBLN-CA22`가 아님 | CA22용 artifact를 재배포; runtime에서 변환하지 않음 |
| tensor parallel mismatch | 명시된 `tensor_parallel_size != 1` | single-device artifact로 교체; `null`은 SDK 0.11의 unavailable provenance로 허용하며 multi-NPU는 현재 범위 외 |
| shape/dtype/name mismatch | inspect descriptor와 model profile이 다름 | 정확한 profile용 artifact를 사용하거나 profile/decoder를 명시적으로 추가; hidden reshape/cast 금지 |
| monitor startup 실패 | `rbln-smi` 미설치, JSON/device/status 오류 | inference 전에 `rbln-smi -b -j -d 0`이 정상인지 확인; 요청한 monitor를 묵시하고 계속하지 않음 |
| request timeout / drain timeout | logical terminal은 발생했지만 SDK physical completion을 아직 증명하지 못함 | worker/parallel/QPS를 낮추고 late completion과 drain을 기다림; timeout만으로 runtime을 해제하지 않음 |
| `cleanup pending` / unload 실패 | accepted job, callback, owner loop 중 하나가 남음 | 새 run을 시작하지 말고 physical completion 후 shutdown/unload를 재시도; process 종료 후 context 0 확인 |

## 11. 종료 후 context 검증

각 sync/async run 직후 `rbln-smi -j`를 다시 저장하고 device 0의
context가 0인지 확인한다.

```bash
rbln-smi -j
rbln-smi -j | python3 -c 'import json,sys; payload=json.load(sys.stdin); contexts=[item for item in payload.get("contexts", []) if isinstance(item, dict) and str(item.get("npu")) == "0"]; print(json.dumps(contexts, indent=2)); raise SystemExit(1 if contexts else 0)'
```

성공 기준은 context list가 `[]`이고 명령 exit code가 0인 것이다. Context가
남으면 그 run은 유효한 성능 결과로 채택하지 않는다.

# Regulus MobilintAccel NPU-only 실행 가이드

현재 Mobilint Yocto 보드에는 `qbruntime` 대신 `maccel` 0.30.1이 설치되어 있다.
이 환경에서는 `regulus-maccel` target을 사용한다. 기존 `regulus` target은
qbruntime이 설치된 다른 이미지와의 호환을 위해 그대로 유지한다.

## 업그레이드가 필요 없는 이유

보드의 maccel 0.30.1에서 다음 항목을 실제로 확인했다.

- `ModelConfig.set_single_core_mode()`로 Cluster0/Core0 지정
- `ModelConfig.force_single_npu_bundle(0)` 성공 및 getter 값 0
- launch 뒤 `Model.get_target_cores()`가 Cluster0/Core0 반환
- `Model.infer()`를 통한 입력 전달과 출력 회수
- async pipeline과 `Model.infer_async()` Future 완료

따라서 현재 ResNet50 비교 측정에는 SDK 업그레이드가 필요하지 않다. Yocto Python
3.12에 Ubuntu Jammy의 Python 3.10 패키지를 섞어 설치하면 ABI가 맞지 않을 수
있으므로 사용하지 않는다. 최소 이미지에 Python `fcntl` 확장이 없는 경우에는
프레임워크가 동일한 Linux `flock(2)` 파일 잠금을 libc를 통해 사용한다.

## 측정 계약

- artifact: Regulus용 사전 컴파일 `.mxq`; ONNX→MXQ 컴파일은 범위 밖
- 모델: 1차 검증은 ResNet50만 지원
- 입력: RGB, ImageNet normalization, float32, NHWC, batch 1
- resize/crop: 짧은 변 256의 종횡비 유지 resize 후 중앙 224×224 crop
- timed sync API: `Model.infer()`
- timed async API: `Model.infer_async()`
- 제외 API: 입출력 전송을 생략하는 `Model.infer_speedrun()`

프레임워크는 추론 전에 MXQ의 입력 shape와 float32 dtype을 검사한다. bundle 0과
Cluster0/Core0의 모든 확인이 끝난 경우에만 `npu_only_verified=True`를 기록한다.

## 보드 사전 확인

```bash
python3 - <<'PY'
import importlib.metadata
import maccel

print("maccel:", importlib.metadata.version("maccel"))
print("cores:", maccel.Accelerator().get_available_cores())
PY
```

현재 확인된 보드에서는 maccel `0.30.1`,
`CoreId(cluster=Cluster.Cluster0, core=Core.Core0)`가 출력된다.

## ImageNet 데이터 구조

```text
/path/to/imagenet_1k/
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   └── ...
└── val_labels.txt
```

`val_labels.txt`의 각 줄은 `파일명 클래스인덱스` 형식이다.

```text
ILSVRC2012_val_00000001.JPEG 65
```

Hailo·DeepX와 비교할 때는 동일 이미지 목록, 클래스 인덱스, resize/crop, batch,
warmup, 측정 샘플 수를 사용한다. 보드와 호스트가 다르므로 결과는 전체 시스템
e2e 비교이며, NPU 칩 단독 비교라고 표기하지 않는다.

## 동기 e2e 실행

`framework` 디렉터리에서 실행한다.

```bash
python3 src/main.py \
  --model resnet50 \
  --target regulus-maccel \
  --artifact /path/to/resnet50.mxq \
  --dataset /path/to/imagenet_1k \
  --batch-size 1 \
  --warmup 10 \
  --max-steps 50000 \
  --results-path results/regulus_e2e.csv
```

## Native async 실행

```bash
python3 src/main.py \
  --model resnet50 \
  --target regulus-maccel \
  --artifact /path/to/resnet50.mxq \
  --dataset /path/to/imagenet_1k \
  --batch-size 1 \
  --warmup 10 \
  --inference-mode async_queue \
  --scenario offline \
  --worker-count 1 \
  --queue-capacity 256 \
  --min-samples 50000 \
  --max-samples 50000 \
  --save-request-trace \
  --results-path results/regulus_async.csv
```

maccel의 현재 모델 계약은 async batch 1이고 확인된 활성 슬롯도 1이므로
`worker-count=1`을 사용한다.

## 결과 인수 조건

CSV와 async 상세 JSON에서 다음을 확인한다.

- `target_id=regulus-maccel`
- `runtime_name=mobilint_accel`
- `runtime_version=0.30.1`
- `npu_only_verified=True`
- `execution_binding=device=0,bundle=0,core=Cluster0/Core0`
- sync latency와 throughput이 유효한 양수
- async의 `async_run_status=valid`, invalid reason 없음, outstanding request 0
- 전체 ImageNet 실행의 Top-1/Top-5가 사전에 정한 허용 범위 안에 있음

2026-08-03의 한 장 smoke 결과는 정답 65에 대해 Top-1 실패, Top-5 성공이었고,
이는 보드 제공 demo와 같은 결과다. 한 장의 정확도는 정상 동작 확인용일 뿐 모델
정확도 판정에는 사용하지 않는다.

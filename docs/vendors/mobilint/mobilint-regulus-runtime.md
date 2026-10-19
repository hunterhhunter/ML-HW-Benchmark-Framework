# Mobilint Regulus qb Runtime 실행 가이드

현재 framework에서 Regulus MXQ를 실행하는 단일 경로는
`--target mobilint-regulus`이다. target의 runtime 이름은 `mobilint`이지만,
내부 구현은 `qbruntime.Accelerator(0)`과 `qbruntime.Model`을 사용한다. 장치
family 검증은 `mbltml`이 설치된 이미지에서는 `mbltml`을 우선 사용하고, 현재
Regulus Yocto처럼 모듈이 없는 경우 `/dev/regulus-npu0` kernel node를 사용한다.

이 경로는 `.mxq` artifact를 직접 실행한다. ONNX에서 MXQ로 컴파일하는 절차와
정확도 승인 여부는 별도 문제이며, NPU-only binding 성공만으로 정확도 동등성을
뜻하지는 않는다.

현재 에지 NPU 실행 명세는 [ResNet50](../../../framework/docs/model-execution-specs/resnet50.md),
[YOLOv5m](../../../framework/docs/model-execution-specs/yolov5m.md),
[YOLOv8s-pose](../../../framework/docs/model-execution-specs/yolov8s-pose.md)에 있다.
세 모델 모두 직접 컴파일한 Regulus용 MXQ로 동기 `e2e`와 비동기
`async_queue` `quality-and-runtime` 패스를 확인했다. 모델별 입력·출력 계약과
전체 데이터 재현 명령은 해당 명세를 따른다.

## 실행 계약

- artifact: Regulus용 사전 컴파일 `.mxq`
- device: `0`
- NPU binding: `force_single_npu_bundle(0)` 성공 및 getter가 `0`
- core: `set_single_core_mode()`로 `Cluster0/Core0` 요청 후 launch 뒤 동일 core 확인
- timed API: 실제 입력 전달과 결과 회수를 포함한 `Model.infer()`
- 제외 API: 입출력 전송을 생략하는 `Model.infer_speedrun()`

binding API가 없거나 bundle/core의 확인값이 다르면 load 자체가 실패한다. 성공한
실행은 CSV와 async 상세 JSON의 다음 필드로 증적을 남긴다.

```text
runtime_version=v1.2.0            # 실제 보드 runtime 버전
npu_only_verified=True
execution_binding=npu_bundle=0; core=Cluster0/Core0
```

`runtime_version` 값은 설치된 qbruntime 버전에 따라 달라진다. 현재 검증 보드의
예시는 v1.2.0이다.

## 보드 사전 확인

framework가 있는 경로에서 실행한다. 첫 명령은 qbruntime과 NPU device 0을,
둘째 명령은 현재 Regulus Yocto의 kernel node를 확인한다. framework runtime은
load 중 이 두 조건을 다시 검증한다.

```bash
cd /root/ml-hw-benchmark/framework

python3 - <<'PY'
import qbruntime

print("qbruntime:", getattr(qbruntime, "__version__", "unknown"))
print("available_devices:", qbruntime.get_available_device_numbers())
PY

test -c /dev/regulus-npu0 && echo "regulus kernel node: OK"
```

첫 명령에서 device `0`이 없거나 둘째 명령에서 node가 없으면 벤치마크를 진행하지
않는다. `mbltml`이 설치된 별도 이미지에서는 해당 SDK의 device-family 검증이
우선 적용된다.

## 실행 예시

ResNet50의 소량 확인 예시다. 모델에 맞는 MXQ, ImageNet 이미지와 label 파일을
지정한다. 이 MXQ의 입력은 `mobilint-resnet50-imagenet1k-v2` profile의 NHWC
`uint8`, batch 1이다. 전체 데이터 실행 조건은 [ResNet50 실행 명세](../../../framework/docs/model-execution-specs/resnet50.md#재현-명령)를 따른다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export ARTIFACT=/absolute/path/to/regulus/resnet50/model.mxq
export IMAGENET_ROOT=/absolute/path/to/imagenet_1k_protocol_v2
export IMAGES="$IMAGENET_ROOT/val_protocol_v2_5000"
export LABELS="$IMAGENET_ROOT/val_protocol_v2_5000_labels.txt"
test -f "$ARTIFACT" && test -d "$IMAGES" && test -f "$LABELS"

python3 src/main.py \
  --model resnet50 \
  --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile \
  --dataset "$IMAGENET_ROOT" --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-resnet50-imagenet1k-v2 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode e2e --batch-size 1 --warmup 2 --max-steps 20 \
  --results-path results/resnet50-regulus-smoke-e2e.csv
```

완료 뒤 동기 CSV 행에서 `npu_only_verified=True`와 `execution_binding`을
확인한다. 비동기 실행에서는 CSV와 상세 JSON의 runtime diagnostics를 함께
확인한다. 값이 없거나 `False`이면 NPU-only 조건을 충족한 실행으로 해석하지
않는다.

`--monitor`는 system telemetry가 필요할 때만 추가한다. 현재 Yocto 이미지에는
Mobilint `mbltml` telemetry module이 없으므로 NPU utilization·memory·temperature
수집은 사용할 수 없으며, 이를 0으로 채워 기록하지 않는다.

## 비동기 실행

`mobilint-regulus`는 native async 경로를 제공한다. 모델별 실행 명세의
`quality-and-runtime` 명령은 품질 패스와 런타임 패스를 분리해 실행하며,
`--runtime-option activation_slots=4`와 `--async-preload-inputs`를 사용한다.
결과에서는 완료 sample 수와 `async_run_status`, NPU binding 증적을 함께
확인한다. 비동기 완료 처리량을 동기 E2E 지연 시간의 역수와 같은 지표로
해석하지 않는다. 지표 경계는 [측정 계약](../../../framework/docs/measurement-contract.md)을 따른다.

## 공정한 비교의 범위

Jetson+M.2 NPU와 Regulus SOM은 같은 host board에서 실행할 수 없다. 비교에는
동일 model weight, dataset/image list, pre/post-processing, batch, warmup,
측정 sample 수와 E2E 범위를 고정하고, 각 장치의 binding 근거를 함께 저장한다.
따라서 결과는 “동일 조건의 전체 시스템 E2E 비교”로 표기하며 NPU 칩 단독 성능
비교라고 주장하지 않는다.

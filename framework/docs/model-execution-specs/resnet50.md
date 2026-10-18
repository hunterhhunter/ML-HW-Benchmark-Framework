# ResNet50 실행 명세: 에지 NPU

`main@30c4db75`의 실행 경로를 기준으로, 확인된 에지 NPU 조합과 재현 명령을
정리한다. 측정 구간과 지표의 정의는 [측정 계약](../measurement-contract.md)을 따른다.
실행 자산의 checksum은 [별도 기록](resnet50-sha256.md)에 있다.

## 모델·데이터

- 모델: `resnet50` (`IMAGE_CLASSIFICATION`, `ImageClassificationEvaluator`)
- 입력: ImageNet-1K validation `val_protocol_v2_5000` 5,000장과
  `val_protocol_v2_5000_labels.txt`. 이미지 목록과 순서를 함께 맞춘다.

## 확인된 조합

| target / Runtime | 사용 artifact | 확인된 실행 모드 |
|---|---|---|
| `hailo8` / HailoRT 4.24.0 | 벤더 제공 `resnet_v1_50_8.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `hailo10h` / HailoRT 5.3.0 | 벤더 제공 `resnet_v1_50_10.hef` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `deepx` / DX-RT 3.4.0 | 벤더 제공 `resnet50_224x224.dxnn` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |
| `mobilint-regulus` / qbruntime 1.2.0 | 직접 컴파일 모델 MXQ `model.mxq` | 동기 `e2e`, `async_queue` 품질·런타임 패스 |

## 입력 계약

| target | 전처리·Runtime 입력 |
|---|---|
| Hailo-8/10H | RGB → 짧은 변 256 resize → 중앙 224×224 crop → NHWC `uint8`. |
| DEEPX DX-M1 | RGB → 224×224 직접 resize → HWC `uint8` list 입력. |
| Mobilint Regulus | `mobilint-resnet50-imagenet1k-v2` profile: RGB → 짧은 변 232 resize → 중앙 224×224 crop → NHWC `uint8`. `qbruntime.Model.infer()` 사용. |

이 dtype은 Runtime에 전달하는 입력 형식이다. DEEPX DXNN의 전체 tensor metadata는
보존된 실행 로그만으로 확정할 수 없으므로 artifact를 교체할 때 다시 검사한다.

## 출력 계약

분류 평가기는 Runtime이 반환한 **첫 출력**을 batch별 클래스 점수로 읽는다.
ImageNet-1K 평가에는 최종 1,000개 클래스 점수가 필요하며, 평가기는 출력의
추가 차원을 평탄화한다. 출력이 1,001개면 첫 background 열을 제외한다.
softmax 값 자체보다 클래스 점수의 순서와 정답 class index의 일치가 중요하다.

| target | Runtime 출력과 평가기 연결 |
|---|---|
| Hailo-8/10H | HEF의 출력 VStream 이름으로 dense 분류 텐서를 받는다. 프레임워크의 사용자 측 출력 형식 기본값은 `float32`이며, 실제 HEF 출력 이름·물리 dtype·shape는 HEF metadata를 확인한다. |
| DEEPX DX-M1 | DX-RT 출력을 텐서 이름의 사전으로 정규화한다. 사용한 ResNet50 DXNN 가이드에서 확인한 출력 shape은 `(1, 1000)`이다. artifact 교체 시 SDK 출력 metadata를 다시 확인한다. |
| Mobilint Regulus | `qbruntime.Model.infer()`의 출력 목록을 MXQ metadata의 출력 이름에 연결한다. 이 ResNet profile은 원시 출력 shape를 고정하지 않으므로 첫 출력이 최종 1,000개 클래스 점수로 해석되는지 확인한다. |

## 재현 명령

각 target의 SDK가 설치된 장비에서 실행한다. 전처리 캐시 충돌을 피하도록
`IMAGENET_ROOT`에는 이 5,000장 cohort만 둔다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export IMAGENET_ROOT=/absolute/path/to/imagenet_1k_protocol_v2
export IMAGES="$IMAGENET_ROOT/val_protocol_v2_5000"
export LABELS="$IMAGENET_ROOT/val_protocol_v2_5000_labels.txt"
test -d "$IMAGES" && test -f "$LABELS"
sha256sum "$LABELS"  # 별도 SHA-256 기록과 대조
```

각 명령의 artifact 변수를 해당 보드의 **실제 파일 경로**로 바꾼다. 동기 명령에서
20장 smoke를 하려면 `--max-steps 5000`을 `--max-steps 20`으로 바꾼다.

```bash
# Hailo-8
export ARTIFACT=/absolute/path/to/resnet_v1_50_8.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model resnet50 --target hailo8 --hef "$ARTIFACT" \
  --dataset "$IMAGENET_ROOT" --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option timeout_ms=10000 --results-path results/resnet50-hailo8-e2e.csv

# Hailo-10H
export ARTIFACT=/absolute/path/to/resnet_v1_50_10.hef
sha256sum "$ARTIFACT"
python3 src/main.py --model resnet50 --target hailo10h \
  --artifact "$ARTIFACT" --no-compile \
  --dataset "$IMAGENET_ROOT" --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/resnet50-hailo10h-e2e.csv

# DEEPX DX-M1
export ARTIFACT=/absolute/path/to/resnet50_224x224.dxnn
sha256sum "$ARTIFACT"
python3 src/main.py --model resnet50 --target deepx --no-compile \
  --artifact "$ARTIFACT" --dataset "$IMAGENET_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --runtime-option device_ids=0 --runtime-option bound_option=NPU_ALL \
  --runtime-option buffer_count=6 --results-path results/resnet50-deepx-e2e.csv
```

Hailo-8·Hailo-10H·DEEPX의 검증된 async 범위는 `quality-and-runtime` 두 패스,
batch 1, worker 4, offline 5,000장이다. 아래의 `TARGET`과 `ARTIFACT`를 같은
행의 target과 artifact에 맞춰 지정한다.

```bash
export TARGET=hailo8  # hailo8, hailo10h, deepx 중 하나
export ARTIFACT=/absolute/path/to/target/resnet50/artifact
sha256sum "$ARTIFACT"
python3 src/main.py --model resnet50 --target "$TARGET" \
  --artifact "$ARTIFACT" --no-compile --dataset "$IMAGENET_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --inference-mode async_queue --async-pass quality-and-runtime \
  --scenario offline --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --warmup 10 \
  --results-path "results/resnet50-${TARGET}-async.csv"
```

Regulus 보드에는 직접 컴파일한 모델 MXQ와 NPU device 0이 필요하다.

```bash
export ARTIFACT=/absolute/path/to/regulus/resnet50/model.mxq
sha256sum "$ARTIFACT"
# 동기 E2E
python3 src/main.py --model resnet50 --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$IMAGENET_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-resnet50-imagenet1k-v2 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode e2e --batch-size 1 --warmup 10 --max-steps 5000 \
  --results-path results/resnet50-regulus-e2e.csv

# async 품질·런타임
python3 src/main.py --model resnet50 --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$IMAGENET_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-resnet50-imagenet1k-v2 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode async_queue --scenario offline --async-pass quality-and-runtime \
  --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --runtime-option activation_slots=4 --warmup 10 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --flush-timeout-sec 600 --async-preload-inputs \
  --save-request-trace --results-path results/resnet50-regulus-async.csv
```

실행 절차와 SDK 점검은 [Hailo async 가이드](../../../docs/hailo-async-runtime.md),
[DeepX 가이드](../../../docs/deepx-setup.md),
[Regulus 가이드](../../../docs/mobilint-regulus-runtime.md)를 참조한다.

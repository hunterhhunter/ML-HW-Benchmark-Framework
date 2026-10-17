# ResNet50 모델별 실행 명세: 에지 NPU

이 문서는 `main@30c4db75`의 실행 경로를 기준으로 Hailo-8, Hailo-10H,
DEEPX DX-M1, Mobilint Regulus에서 **실제로 확인된 범위**를 적는다. 장치별 기록은
서로 다른 시점의 실행이며, 이 문서를 작성하면서 실장비를 재실행하지 않았다.
동기 `e2e`의 latency와 async 완료 처리량의 뜻은 [측정 계약](../measurement-contract.md)을
따른다. RBLN·ARIES 등 다른 target은 이 에지 4종 명세의 범위에 넣지 않는다.

## 모델·데이터 기준

- CLI 모델은 `resnet50`, task는 `IMAGE_CLASSIFICATION`, 평가기는
  `ImageClassificationEvaluator`다. 출력의 1,000개 class score에서 Top-1·Top-5를
  계산한다.
- 검증 기록은 ImageNet-1K validation의 `val_protocol_v2_5000` 5,000장과
  `val_protocol_v2_5000_labels.txt`를 사용한다. 당시 label 파일 SHA-256은
  [별도 기록](resnet50-sha256.md)에 있다.
  이미지 목록·순서도 별도로 맞춰야 한다. Hailo-10H의 2026-10-17 실행은 별도
  입력 복사 디렉터리를 사용했으므로 동일 label hash만으로 이미지 파일 동일성을
  증명하지 않는다.
- [공통 모델 프로필](../../src/core/model_profiles.py)의 기본 ONNX는 Kalray 경로다.
  아래 HEF·DXNN·MXQ가 그 ONNX에서 생성됐다는 뜻이 아니다.
- **Hailo-8·Hailo-10H·DEEPX는 각 벤더가 제공한 사전 컴파일 모델**(HEF·DXNN)을
  사용했다. 프레임워크에서 다시 컴파일하지 않았다. 이 artifact의 원천 checkpoint와
  컴파일 recipe는 실행 기록에 묶여 있지 않다.
- **Regulus는 연구팀이 컴파일한 MXQ**를 사용했다. TorchVision ImageNet-1K V2
  source ONNX와 ImageNet train 1,024장 calibration을 사용한 QBCompiler 1.1.2
  산출물이며, 벤더 제공 MXQ가 아니다. 출처가 다른 이 artifact들의 Top-1 수치를
  동일 가중치의 NPU 정확도 차이로 해석하지 않는다.

## 확인된 조합

artifact와 label의 SHA-256은 [ResNet50 SHA-256 기록](resnet50-sha256.md)에
분리했다. 새 장비에서 파일명뿐 아니라 checksum을 대조한다. 원시 결과 파일은
Git에 포함되지 않은 로컬 `results/` archive에 있고, 증거 목록은 후속 문서에서 정리한다.

| target / 실행 환경 | 사용 artifact | 확인된 실행 범위 |
|---|---|---|
| `hailo8` / HailoRT 4.24.0 | 벤더 제공 `resnet_v1_50_8.hef` | 동기 5,000장 `a91b3f06`: Top-1 73.78%, Top-5 90.68%. async schema v2 `4e4f2b57`: `valid`, `quality_status=passed`, 두 패스 각 5,000장. |
| `hailo10h` / HailoRT 5.3.0 | 벤더 제공 `resnet_v1_50_10.hef` | 동기 5,000장 `73865359`: Top-1 73.82%, Top-5 90.70%. async schema v2 `3418ef6d`: `valid`, `quality_status=passed`, 두 패스 각 5,000장. |
| `deepx` / DX-RT 3.4.0 | 벤더 제공 `resnet50_224x224.dxnn` | 동기 5,000장 `0747ae23`: Top-1 79.36%, Top-5 94.76%. async schema v2 `671faf17`: `valid`, `quality_status=passed`, 두 패스 각 5,000장. |
| `mobilint-regulus` / qbruntime 1.2.0 | 연구팀 컴파일 `model.mxq` | async schema v2 runtime-only `946a58d1`: 5,000건 완료, `valid`, `quality_status=skipped`, `comparison_eligible=False`, NPU bundle 0/Core0 확인. 프레임워크 정확도 패스는 미실행. |

Hailo-8의 2026-09-27 Top-1 28.08% 결과는 다른 이미지 cohort와 전처리 캐시
파일명이 충돌한 실행이다. 교정된 `a91b3f06`을 사용한다. Hailo-10H의 위 10월
실행은 `main@30c4db75`에 시간 marker 계측 패치를 추가한 보드에서 수행됐다.
Hailo-8·DeepX 동기 기록은 `main@6868f6ba`, Regulus runtime-only 기록은
`main@5b286a24` 기반이다. 이 세 commit은 모두 현재 `main`의 조상이다.
Regulus MXQ의 별도 vendor-native E2E 실행에는 5,000장 Top-1 77.42%가 있지만,
그 값은 프레임워크 run `946a58d1`의 품질 평가 결과가 아니다.
DEEPX async `671faf17`의 실행 파일명은 `deepx-official-smoke-resnet50.dxnn`이며,
동기 실행의 `resnet50_224x224.dxnn`과 기록된 SHA-256이 같다. 두 파일명을
벤더가 각각 별도로 제공한 모델이라고 해석하지 않는다.

확인에 사용한 로컬 archive 경로는 다음과 같다. 이 디렉터리들은 Git 배포물에
포함되지 않으므로 새 checkout에서는 별도로 전달받아야 한다.

- Hailo-8: `results/2026-09-28-hailo8-worker4-rerun/results.csv`,
  `results/2026-10-16-hailo8-ina3221-runtime-return-w4/resnet50/results.csv`
- Hailo-10H: `results/2026-10-17-hailo10h-ina3221-5000/e2e/resnet50/results.csv`,
  `results/2026-10-17-hailo10h-ina3221-5000/async/resnet50/results.csv`
- DEEPX: `results/2026-09-28-edge-npu-5000/deepx/results.csv`,
  `results/2026-10-16-deepx-ina3221-runtime-return-w4/resnet50/results.csv`
- Regulus: `results/2026-10-01-regulus-vg-e2e-async-runtime-v2-preloaded-5000/resnet50/results.csv`

## artifact별 입출력·처리 계약

| target | 입력과 전처리 | 출력·Runtime 경계 |
|---|---|---|
| Hailo-8/10H | RGB, 짧은 변 256 resize → 중앙 224×224 crop. `HailoImageClassificationLoader`는 raw pixel을 만들고 Hailo runtime이 HEF 입력에 맞춰 NHWC `uint8`로 변환한다. HEF input은 `resnet_v1_50/input_layer1` `(224,224,3)`로 관측됐다. | HEF output `resnet_v1_50/softmax1` `(1000,)`. HailoRT `InferModel`의 반환 결과를 평가기로 전달한다. HEF 세대별 파일을 교차 사용하지 않는다. |
| DEEPX DX-M1 | 선택된 실행 로그에서 raw `NHWC/UINT8` 경로를 확인했다. `DeepXImageClassificationLoader`는 RGB를 224×224로 직접 resize하고 정규화하지 않는다. Runtime에는 batch 축을 제거한 HWC `uint8`을 list 입력으로 보낸다. | DX-RT 출력의 1,000개 class score를 평가한다. 일반 ONNX 경로의 NCHW `float32` 전처리를 이 DXNN에 적용하면 안 된다. |
| Mobilint Regulus | `mobilint-resnet50-imagenet1k-v2` profile: RGB, 짧은 변 232 resize → 중앙 224×224 crop, raw `uint8` NHWC `(1,224,224,3)`. | `qbruntime.Model.infer()`의 실제 출력 반환이 완료 경계다. `infer_speedrun()`은 사용하지 않는다. 내부 연산별 정밀도는 미확인이다. |

정확도 패스를 수행한 경로에서는 분류 평가기가 출력 tensor를 `(batch, 1000)`으로
정리한다. Regulus의 위 runtime-only 패스에서는 정확도 평가기를 실행하지 않았다.
위의 dtype은 **host-visible 입력 계약**이며 컴파일된 내부 연산 정밀도를 뜻하지 않는다.
DeepX 가이드의 `input.1`·`PRE_IM2COL` 예시는 별도 `ResNet50.dxnn`의 metadata다.
위 `resnet50_224x224.dxnn`의 실제 tensor 이름·rmap 전체는 보존된 실행 로그만으로
확정할 수 없으므로 새 artifact를 받을 때 DXNN metadata를 다시 검사한다.
전처리 구현은 [Hailo loader](../../src/dataloader/hailo_image_classification_loader.py),
[DeepX loader](../../src/dataloader/deepx_image_classification_loader.py),
[Mobilint profile](../../src/dataloader/mobilint_vision_profiles.py)에 있다.

## 재현 명령

각 target의 SDK가 설치된 장비에서 실행한다. `IMAGENET_ROOT`는 이 5,000장 cohort만
사용하는 디렉터리로 지정한다. 현재 이미지 캐시 키는 파일명과 전처리 설정에
의존하므로 다른 cohort의 같은 파일명을 한 cache에 섞으면 정확도가 달라질 수 있다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
export IMAGENET_ROOT=/absolute/path/to/imagenet_1k_protocol_v2
export IMAGES="$IMAGENET_ROOT/val_protocol_v2_5000"
export LABELS="$IMAGENET_ROOT/val_protocol_v2_5000_labels.txt"
test -d "$IMAGES" && test -f "$LABELS"
sha256sum "$LABELS"  # 별도 SHA-256 기록과 대조
```

각 명령의 artifact 변수를 해당 보드의 **실제 파일 경로**로 바꾸고, 먼저 20장
smoke를 하려면 `--max-steps 5000`만 `--max-steps 20`으로 바꿔 실행한다.

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
행의 target과 [별도 SHA-256 기록](resnet50-sha256.md)에 맞춰 지정한다.

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

Regulus에서 검증된 **프레임워크 범위는 정확도를 생략한 async 실행**이다. 보드가
Regulus용 MXQ와 NPU device 0을 제공해야 하며 Aries MXQ로 바꿔 실행하면 안 된다.

```bash
export ARTIFACT=/absolute/path/to/regulus/resnet50/model.mxq
sha256sum "$ARTIFACT"
python3 src/main.py --model resnet50 --target mobilint-regulus \
  --artifact "$ARTIFACT" --no-compile --dataset "$IMAGENET_ROOT" \
  --image-dir "$IMAGES" --label-dir "$LABELS" \
  --image-preprocess-profile mobilint-resnet50-imagenet1k-v2 \
  --image-preprocess-mode raw --layout NHWC \
  --inference-mode async_queue --scenario offline --async-pass runtime-only \
  --batch-size 1 --worker-count 4 --queue-capacity 256 \
  --runtime-option activation_slots=4 --warmup 10 \
  --max-samples 5000 --min-samples 5000 --min-duration-sec 0 \
  --schedule-seed 0 --flush-timeout-sec 600 --async-preload-inputs \
  --save-request-trace --results-path results/resnet50-regulus-runtime-only.csv
```

결과 CSV에서 `run_id`, `target_id`, `inference_mode`, 처리 sample 수와 Top-1·Top-5
또는 `quality_status`를 확인한다. async는 상세 JSON까지 열어 submitted·accepted·
completed·runtime completed가 모두 5,000이고 실패·거절·timeout·outstanding이 0인지
확인한다. Regulus는 `npu_only_verified=True`와
`execution_binding=npu_bundle=0; core=Cluster0/Core0`도 확인한다.

```bash
export RESULT_CSV=results/resnet50-hailo8-e2e.csv  # 실행한 target의 경로로 변경
python3 - "$RESULT_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert rows, "result CSV has no rows"
print(f"rows: {len(rows)} (showing the latest row)")
row = rows[-1]
for key in (
    "run_id", "target_id", "inference_mode", "Total Samples",
    "Top-1 Accuracy", "Top-5 Accuracy", "async_metric_schema_version",
    "async_run_status", "quality_status", "comparison_eligible",
    "async_submitted_requests", "async_accepted_requests",
    "async_completed_samples", "async_runtime_completed_samples",
    "quality_evaluator_samples", "async_failed_requests",
    "async_rejected_requests", "async_timed_out_requests",
    "async_outstanding_requests", "npu_only_verified", "execution_binding",
    "details_path",
):
    if row.get(key):
        print(f"{key}: {row[key]}")
if row.get("inference_mode") == "async_queue":
    # details_path는 CSV 폴더의 상위 디렉터리를 기준으로 기록된다.
    details_file = Path(sys.argv[1]).resolve().parent.parent / row["details_path"]
    details = json.loads(details_file.read_text(encoding="utf-8"))
    expected = int(row["async_submitted_requests"])
    counts = details["counts"]
    assert details["status"] == "valid"
    assert all(counts[k] == expected for k in (
        "submitted", "accepted", "completed", "completed_samples",
        "runtime_completed_requests", "runtime_completed_samples",
    ))
    assert counts["outstanding"] == 0
    assert all(row[k] == "0" for k in (
        "async_failed_requests", "async_rejected_requests",
        "async_timed_out_requests", "async_outstanding_requests",
    ))
    print(f"details: {details_file}")
    print(f"quality_phase: {details['quality_phase']['status']}")
    print("async lifecycle: PASS")
PY
```

실행 절차와 SDK 점검은 [Hailo async 가이드](../../../docs/hailo-async-runtime.md),
[DeepX 가이드](../../../docs/deepx-setup.md),
[Regulus 가이드](../../../docs/mobilint-regulus-runtime.md)를 참조한다.

# Mobilint qbcompiler 1.2 멀티 모델 컴파일 실험 실행서

이 문서는 BERT SST-2, BERT SQuAD v1, PatchTST ETTh1, ResNet50과 YOLOv5m을
`qbcompiler==1.2.0`으로 컴파일한 과정과 결과를 한 형식으로 남기는 기준 문서다.
먼저 모델 하나를 컴파일하고, `MXQ_COMPILE=pass`인 attempt만 ARIES에서 검사한 뒤,
마지막에 framework 품질 실험을 연결한다. 자동화 테스트 통과는 실제 compiler 또는
ARIES 성공으로 세지 않는다.

## 1. 실행 경계와 고정 환경

컴파일 호스트와 ARIES 호스트의 역할은 분리한다.

| 구분 | 역할 | 고정 조건 |
|---|---|---|
| compiler host | source·calibration 준비, MBLT/MXQ 생성 | Ubuntu 22.04, Intel x86-64, CPython 3.10 |
| ARIES host | MXQ load, metadata·입출력 계약, 1회 동기 추론 검사 | ARIES, `qbruntime` `v1.3.2` |
| framework E2E | task 품질과 성능 측정 | ARIES 검사를 통과한 같은 MXQ와 입력 경계 사용 |

Docker와 compiler host의 ARIES 장치·driver는 필요하지 않다. compiler wheel은
라이선스가 있는 벤더 배포물이므로 저장소에 넣지 않는다.

| 항목 | 고정값 |
|---|---|
| wheel | `qbcompiler-1.2.0-py3-none-any.whl` |
| wheel SHA256 | `28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b` |
| target | `aries-rb` |
| Python packages | `torch==2.7.1`, `torchvision==0.22.1`, `numpy==1.26.0`, `tensorflow==2.17.0`, `onnx==1.16.2`, `onnxruntime==1.19.2`, `opencv-python==4.11.0.86`, `transformers==4.57.1`, `datasets==3.6.0` |

공통 runner는 OS, architecture, Python, wheel 파일명과 SHA256을 먼저 검사한다. 전용
venv에 `pip`이 없으면 `ensurepip`을 적용하고 위 버전을 설치한 뒤 `pip check`와
`qbcompiler.mblt_compile`/`mxq_compile` signature 검사를 수행한다. 현재 shell의
가상환경은 꺼도 되고 켜 둬도 되지만 runner에는 CPython 3.10 경로를 명시한다.

```bash
REPO="$HOME/ML-HW-Benchmark-Framework"
RUNNER="$REPO/framework/scripts/run_mobilint_compile_experiment.sh"
WHEEL="$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl"
PY310="$(command -v python3.10)"
COMPILER_VENV="$REPO/.venv-qbcompiler-1.2-py310"
OUTPUT_ROOT="$REPO/mobilint-compile-attempts"

cd "$REPO" || exit 1
test -x "$PY310"
test -s "$WHEEL"
sha256sum "$WHEEL"
bash "$RUNNER" --help
```

처음 dependency와 Hugging Face source를 받을 때만 network가 필요하다. runner는
token이나 임의의 환경 변수를 결과에 복사하지 않는다.

## 2. 모델과 데이터 준비

| runner model/variant | source와 고정 조건 | calibration 입력 |
|---|---|---|
| `bert-sst2/default` | `textattack/bert-base-uncased-SST-2`; `glue/sst2` validation | 최대 길이 128의 host embedding 32개 |
| `bert-squad1/default` | `csarron/bert-base-uncased-squad-v1`; `squad` validation | 최대 길이 384의 host embedding 32개 |
| `patchtst-etth1/stock` | `ibm-granite/granite-timeseries-patchtst`; 요청 revision을 실제 40자 commit SHA로 해소 | ETTh1 validation window 32개 |
| `patchtst-etth1/compat-static-patchifier` | stock manifest의 정확한 resolved SHA; stock lowering 실패의 자식 attempt만 허용 | stock과 같은 ETTh1 계약 |
| `resnet50/default` | `torchvision.models.resnet50:IMAGENET1K_V2`, `torchvision==0.22.1` | ImageNet validation RGB image 32개 |
| `yolov5m/default` | `ultralytics/yolov5@86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`; 파일명이 정확히 `yolov5m.pt`인 non-empty weight | COCO RGB image 32개 |

서버의 실제 경로를 먼저 고정한다.

```bash
ETTH1="$REPO/framework/datasets/etth1/ETTh1.csv"
IMAGENET_VAL="$REPO/datasets/imagenet_1k/val"
COCO_IMAGES="$REPO/framework/datasets/coco128/images/train2017"
YOLOV5_ROOT="$HOME/mobillint/yolov5"
YOLOV5_WEIGHTS="$HOME/mobillint/yolov5m.pt"

test -s "$ETTH1"
test -d "$IMAGENET_VAL"
test -d "$COCO_IMAGES"
test -s "$YOLOV5_WEIGHTS"
test "$(basename "$YOLOV5_WEIGHTS")" = yolov5m.pt
git -C "$YOLOV5_ROOT" rev-parse HEAD
git -C "$YOLOV5_ROOT" diff --exit-code -- \
  models/experimental.py models/yolo.py
```

YOLOv5 checkout이 없다면 별도 디렉터리에 clone하고 exact commit으로 이동한다.

```bash
git clone https://github.com/ultralytics/yolov5.git "$YOLOV5_ROOT"
git -C "$YOLOV5_ROOT" checkout --detach \
  86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51
```

weight는 사용자가 준비한 공식 `yolov5m.pt`를 사용한다. runner가 크기와 SHA256,
checkpoint의 `depth_multiple=0.67`, `width_multiple=0.75` 및 가능한 경우
`yaml_file=yolov5m.yaml`을 기록·검사한다. 확인되지 않은 weight hash를 문서에
추정해서 넣지 않는다.

## 3. 입출력 ABI와 compiler option

| 모델 | compiler 입력 | ARIES runtime 입력 | 출력 순서 | core mode |
|---|---|---|---|---|
| BERT SST-2 | token 3종 `int64 [1,L]` | `embeddings float32 [1,L,768]` | `logits float32 [1,1,2]` | `single` |
| BERT SQuAD v1 | token 3종 `int64 [1,L]` | `embeddings float32 [1,L,768]` | `end_logits`, `start_logits`; 각각 `float32 [1,L,1]` | `single` |
| PatchTST | `past_values float32 [1,512,7]`, `past_observed_mask bool [1,512,7]` | compiler와 같음 | `prediction_outputs float32 [1,96,7]` | `global8` |
| ResNet50 | `input_np float32 [1,224,224,3]`, 범위 `[0,1]` | `input_np uint8 [1,224,224,3]` | `logits float32 [1,1000]` | `global8` |
| YOLOv5m | `input_np float32 [1,640,640,3]`, 범위 `[0,1]` | `input_np uint8 [1,640,640,3]` | stride 32 `[1,20,20,255]`, stride 16 `[1,40,40,255]`, stride 8 `[1,80,80,255]` | `global8` |

qbruntime은 batch/variant singleton 축을 추가해 표시할 수 있다. 엄격한 verifier는
허용한 경계 singleton만 정규화하며 tensor 순서를 바꾸지 않는다. SQuAD source model의
반환 순서 `start_logits,end_logits`와 ARIES positional 순서
`end_logits,start_logits`는 다르므로 임의로 교환하면 안 된다.

BERT MXQ compiler 호출의 `inference_scheme`은 기존 recipe의 `all`이며, ARIES에서는
명시적인 cluster 0/core 0 `single`로 연다. 나머지 세 recipe는 `global8`이다. vision
recipe는 `Uint8InputConfig(apply=True, inputs=["input_np"],
division_factor=255.0)`를 사용한다. 모든 MXQ는 다음 calibration 설정을 공유한다.

```text
method=1, output=0, mode=1
MaxPercentile(percentile=0.999, topk_ratio=0.01)
```

ResNet50 preset은 `classification_torchvision`, YOLOv5m preset은 `yolo_640`이고
YOLO decode는 compiler artifact에 포함하지 않는다. `.mblt`와 `.mxq`는 동일 source와
feed에서 별도 API를 호출한 독립 산출물이다.

## 4. 먼저 개별 컴파일 실행

각 명령은 새 attempt를 만든다. `tee` 뒤에는 반드시 runner의 종료 코드를
`PIPESTATUS[0]`으로 받는다. 아래 여섯 명령을 한꺼번에 실행하지 말고, 우선 필요한
모델 하나만 실행해 결과를 확인한다.

### 4.1 BERT SST-2

```bash
LOG="$REPO/mobilint-compile-experiment-bert-sst2.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model bert-sst2 --variant default --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

### 4.2 BERT SQuAD v1

```bash
LOG="$REPO/mobilint-compile-experiment-bert-squad1.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model bert-squad1 --variant default --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

### 4.3 PatchTST stock

`main`은 준비 단계에서 실제 commit SHA로 해소되고 manifest에 저장된다.

```bash
LOG="$REPO/mobilint-compile-experiment-patchtst-stock.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model patchtst-etth1 --variant stock --dataset "$ETTH1" \
  --model-revision main --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

compat variant는 일반적인 두 번째 recipe가 아니다. stock attempt가
`MBLT_COMPILE` 또는 `MXQ_COMPILE`의 실제 lowering 오류로 실패했을 때만 사용한다.
dataset·사용자 입력·dependency·용량 오류에는 compat를 적용하지 않는다. 실패한 stock
attempt의 `source-manifest.json`에서 resolved SHA를 읽고 그 attempt를 parent로 둔다.

```bash
STOCK_ATTEMPT="<stock ATTEMPT_ROOT>"
test "$(jq -r '.model' "$STOCK_ATTEMPT/result.json")" = patchtst-etth1
test "$(jq -r '.variant' "$STOCK_ATTEMPT/result.json")" = stock
case "$(jq -r '.failed_at' "$STOCK_ATTEMPT/result.json")" in
  MBLT_COMPILE|MXQ_COMPILE) ;;
  *) echo "compat 대상 lowering 실패가 아닙니다" >&2; exit 1 ;;
esac
PATCHTST_SHA="$(jq -r '.resolved_revision' \
  "$STOCK_ATTEMPT/source-manifest.json")"
[[ "$PATCHTST_SHA" =~ ^[0-9a-f]{40}$ ]]
```

### 4.4 PatchTST compat-static-patchifier

```bash
LOG="$REPO/mobilint-compile-experiment-patchtst-compat.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model patchtst-etth1 --variant compat-static-patchifier \
  --dataset "$ETTH1" --model-revision "$PATCHTST_SHA" \
  --parent-attempt "$STOCK_ATTEMPT" --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

compat는 patchification만 정적으로 바꾸고 boolean mask를 values dtype으로 cast한다.
source smoke가 stock과 `rtol=1e-5`, `atol=1e-6`로 같은 출력을 내지 않으면 컴파일하지
않는다.

### 4.5 ResNet50

```bash
LOG="$REPO/mobilint-compile-experiment-resnet50.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model resnet50 --variant default --dataset "$IMAGENET_VAL" \
  --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

### 4.6 YOLOv5m

```bash
LOG="$REPO/mobilint-compile-experiment-yolov5m.log"
set +e
bash "$RUNNER" --wheel "$WHEEL" --python "$PY310" --venv "$COMPILER_VENV" \
  --model yolov5m --variant default --dataset "$COCO_IMAGES" \
  --yolov5-root "$YOLOV5_ROOT" --weights "$YOLOV5_WEIGHTS" \
  --output-root "$OUTPUT_ROOT" \
  |& tee "$LOG"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "RUNNER_EXIT_CODE=$RUNNER_EXIT_CODE"
```

## 5. attempt와 calibration 증거

runner는 다음 단계를 각기 새 child process로 실행하고 첫 실패에서 멈춘다.

| 단계 | 의미 |
|---|---|
| `SOURCE_PREPARE` | source identity, dataset와 calibration 입력 준비 |
| `SOURCE_SMOKE` | qbcompiler import 전 CPU source graph·출력 검사 |
| `CALIBRATION_PREPARE` | manifest/report와 non-empty calibration 묶음 검사 |
| `MBLT_COMPILE` | `.mblt` 생성 및 크기·SHA256 기록 |
| `MXQ_COMPILE` | `.mxq` 생성 및 크기·SHA256 기록 |
| `ARIES_LOAD` | qbruntime 1.3.2에서 config/model launch·dispose 검사 |
| `CONTRACT_CHECK` | SDK metadata와 저장 입력·출력 shape/dtype/order 검사 |
| `TASK_SMOKE` | 저장 입력 하나로 동기 `infer()`하고 finite 출력 검사 |

각 stage의 `status`는 `not_run`, `pass`, `fail` 중 하나다. `compile_status`는 MBLT와
MXQ가 모두 pass일 때만 pass다. `runtime_status`, `contract_status`, `quality_status`는
서로 독립적으로 갱신된다. runner는 attempt를 만든 뒤 종료하는 모든 경로에서 다음 두
줄을 출력한다.

```text
ATTEMPT_ROOT=/absolute/path/to/attempt
EXPERIMENT_EXIT_CODE=0
```

shell pipeline의 `RUNNER_EXIT_CODE`, 출력된 `EXPERIMENT_EXIT_CODE`, `result.json`의
stage 결과가 일치하는지 확인한다.

```bash
ATTEMPT_ROOT="<runner가 출력한 절대 경로>"
jq '{attempt_id,model,variant,failed_at,compile_status,runtime_status,
     contract_status,quality_status,stages,artifacts,metadata}' \
  "$ATTEMPT_ROOT/result.json"
sed -n '1,240p' "$ATTEMPT_ROOT/compile.log"
find "$ATTEMPT_ROOT" -type f \( -name '*.mblt' -o -name '*.mxq' \) \
  -exec sha256sum {} +
```

calibration은 정렬된 전체 후보에서 처음과 끝을 포함해 32개를 균등 선택한다.

| 모델 | 순서와 provenance |
|---|---|
| BERT | `calibration_data/000.npy`~`031.npy`; manifest의 dataset index, sequence length, path, size, SHA256; embedding weight path·size·SHA256 |
| PatchTST | `past_values`, `past_observed_mask` 순서; sample별 두 path·size·SHA256; ETTh1 파일 SHA256, validation 경계 `[8640,11520]`, context 512, prediction 96, stride 12, normalization 통계 |
| ResNet50 | 정렬된 ImageNet image index와 source SHA256; `calibration/NNN.npy` path·size·SHA256; raw RGB resize short side 232, center crop 224, uint8 NHWC |
| YOLOv5m | 정렬된 COCO image index와 source SHA256; `calibration/NNN.npy` path·size·SHA256; RGB letterbox 640, pad 114, uint8 NHWC; source Git blob과 weight 크기·SHA256 |

실제 배열을 이동하거나 다시 만들었다면 같은 attempt로 이어가지 않는다. `result.json`,
source manifest와 compile report가 가리키는 path·size·SHA256가 한 묶음이어야 한다.

## 6. MXQ pass 뒤 ARIES 엄격 검사

`compile_status=pass`, `MXQ_COMPILE=pass`인 attempt만 ARIES로 옮긴다. attempt 전체를
경로 구조 그대로 전송해야 저장 입력과 hash 검사가 가능하다. ARIES host에서 저장소와
qbruntime 환경을 지정한다.

```bash
REPO="$HOME/ML-HW-Benchmark-Framework"
FW="$REPO/framework"
PY="$REPO/.venv-mobilint/bin/python"
ATTEMPT_ROOT="<ARIES host의 attempt 절대 경로>"
MXQ="$(jq -r '.artifacts[] | select(.path | endswith(".mxq")) | .path' \
  "$ATTEMPT_ROOT/result.json")"
MXQ="$ATTEMPT_ROOT/$MXQ"

test -s "$MXQ"
"$PY" - <<'PY'
import qbruntime
assert qbruntime.__version__ in {"1.3.2", "v1.3.2"}
print(qbruntime.__version__, qbruntime.__file__)
PY
mobilint-cli status
sha256sum "$MXQ"
```

검사는 CLI `--core-mode`로 덮어쓰지 않는다. BERT attempt는 명시적인 cluster 0/core 0
`single`, 나머지는 recipe에 기록된 `global8`을 사용한다. SDK metadata의 구·신 API
차이를 처리하되 shape/dtype/order는 위 ABI와 정확히 맞아야 한다.

```bash
mkdir -p "$ATTEMPT_ROOT/logs"
set +e
PYTHONPATH="$FW:$FW/src" "$PY" -m \
  tools.mobilint_compile_recipes.runtime_verify \
  --attempt-root "$ATTEMPT_ROOT" --artifact "$MXQ" \
  2>&1 | tee "$ATTEMPT_ROOT/logs/aries-runtime.log"
ARIES_EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "ARIES_EXIT_CODE=$ARIES_EXIT_CODE"
mobilint-cli status
```

실패하면 같은 attempt를 다시 실행해 증거를 덮지 않는다. 먼저 결과와 device/kernel
로그를 보존하고 새 child attempt를 만든다.

```bash
jq '{runtime_status,contract_status,failed_at,runtime_verification,
     ARIES_LOAD:.stages.ARIES_LOAD,
     CONTRACT_CHECK:.stages.CONTRACT_CHECK,
     TASK_SMOKE:.stages.TASK_SMOKE}' "$ATTEMPT_ROOT/result.json"
journalctl -k --since '-10 minutes' --no-pager | tail -n 160
mobilint-cli status
```

qbruntime extension의 segmentation fault도 `journalctl -k`의 process와 shared-object
위치를 포함해 실패로 기록한다. 모델은 성공·실패와 관계없이 `dispose()`되어야 하며,
검사 뒤 `mobilint-cli status`에 잔류 process가 없어야 한다.

## 7. 품질 실험 연결

`runtime_status=pass`와 `contract_status=pass`를 확인한 뒤에만 framework E2E를
실행한다. BERT·PatchTST 명령은
[ARIES Transformer 실행 가이드](mobilint-aries-transformers.md), ResNet50·YOLOv5m
명령은 [ARIES 트러블슈팅 기록](mobilint-aries-troubleshooting.md)의 동기 E2E 절을
사용하되 artifact만 현재 attempt의 MXQ로 바꾼다. BERT는 같은 attempt 안의
`weight_dict.pth`도 함께 사용한다.

CSV와 로그를 attempt 안에 복사한 다음 품질 상태를 한 번만 기록한다.

```bash
QUALITY_CSV="<framework가 생성한 CSV 절대 경로>"
QUALITY_LOG="<framework E2E 전체 로그 절대 경로>"
mkdir -p "$ATTEMPT_ROOT/quality"
cp "$QUALITY_CSV" "$ATTEMPT_ROOT/quality/result.csv"

PYTHONPATH="$FW:$FW/src" "$PY" -m tools.mobilint_compile_recipes.attempt \
  quality --attempt-root "$ATTEMPT_ROOT" \
  --result-csv "$ATTEMPT_ROOT/quality/result.csv"
```

E2E process가 nonzero라면 non-empty 로그와 실제 종료 코드만 기록한다. 성공 CSV로
대체하거나 `TASK_SMOKE` 결과를 바꾸지 않는다.

```bash
cp "$QUALITY_LOG" "$ATTEMPT_ROOT/quality/e2e-failure.log"
PYTHONPATH="$FW:$FW/src" "$PY" -m tools.mobilint_compile_recipes.attempt \
  quality-failure --attempt-root "$ATTEMPT_ROOT" \
  --exit-code "<nonzero E2E exit code>" \
  --log "$ATTEMPT_ROOT/quality/e2e-failure.log"
```

## 8. 재시도와 legacy BERT 원칙

attempt는 불변이다. 성공·실패 어느 쪽도 같은 경로를 덮어쓰지 않으며 runner가 UTC
timestamp와 PID로 새 root를 만든다. 실패한 root를 삭제하거나 `result.json`을 직접
고쳐 재사용하지 않는다. 원인과 parent attempt를 기록하고 새 명령을 실행한다.

과거 BERT task root는 `bert_bridge`로 compiler 결과만 가져올 수 있다. 그러나 당시
manifest에는 각 calibration 파일의 compile-time path·size·SHA256가 없으므로 엄격한
ARIES verifier의 입력 provenance 조건을 충족하지 못한다. 기존 hash를 추정하거나
manifest를 사후 생성·이식하지 않는다. strict 검증에는 이전 attempt를 parent로 둔 새
child attempt에서 calibration을 다시 준비하고 BERT를 다시 컴파일해야 한다. bridge는
compiler stage만 매핑하며 runtime·contract·quality 상태를 성공으로 만들지 않는다.

## 9. 관측 결과 ledger

아래 `prior/legacy` 행은 이 공통 runner와 strict attempt schema가 생기기 전에 얻은
기록이다. 새 strict BERT 재컴파일과 PatchTST·ResNet50·YOLOv5m qbcompiler 실험은 아직
실행하지 않았으며 성공으로 표시하지 않는다.

| 구분 | 모델/variant | compile | strict ARIES | quality | 근거 |
|---|---|---|---|---|---|
| prior/legacy | BERT SST-2/default | pass | legacy pass | legacy pass | MXQ `0ce6d9d2d7ba7637c98e1fffd9f1098aac86fa0c90db061eb21a561856d5a1f4`; weight `495f46d460820a3f80300e53a5569678474819c31d36d205e03d8a6f2639d80d` |
| prior/legacy | BERT SQuAD v1/default | pass | legacy pass | legacy pass | MXQ `5d1ff5a263a15b49e62a4d14fdfbfd9e261a7b114f65a2541c6d0d3bf54d03a2`; weight `a8bd92f3879929e481b097f682fbc9244c2ddb002b286d4a91499c756003a223` |
| new strict attempt | BERT SST-2/default | `not_run` | `not_run` | `not_run` | fresh reprepare/recompile required |
| new strict attempt | BERT SQuAD v1/default | `not_run` | `not_run` | `not_run` | fresh reprepare/recompile required |
| new attempt | PatchTST ETTh1/stock | `not_run` | `not_run` | `not_run` | compiler server 실행 전 |
| conditional retry | PatchTST ETTh1/compat-static-patchifier | `not_run` | `not_run` | `not_run` | stock lowering 실패 때만 실행 |
| new attempt | ResNet50/default | `not_run` | `not_run` | `not_run` | compiler server 실행 전 |
| new attempt | YOLOv5m/default | `not_run` | `not_run` | `not_run` | compiler server 실행 전 |

legacy 64-sample 기록은 서로 다른 검증 경계를 섞지 않고 따로 읽는다.

| 모델 | 실행 경계 | 64-sample 지표 |
|---|---|---|
| BERT SST-2 | 이전 standalone verifier | 59/64, accuracy `0.921875`, mean inference `4.8368 ms` |
| BERT SQuAD v1 | 이전 standalone 정규화 문자열 평가 | EM `0.828125`, F1 `0.886318`, mean inference `20.9896 ms` |
| BERT SST-2 | 이전 framework E2E | accuracy `93.75`, Average Latency `5.0860 ms`, P99 `6.1769 ms`, Samples/s `196.6193` |
| BERT SQuAD v1 | 이전 framework E2E token-coordinate 평가 | EM `68.75`, F1 `78.8566`, Average Latency `18.2216 ms`, P99 `22.6088 ms`, Samples/s/QPS `54.8800` |

standalone 문자열 지표와 framework token-coordinate 지표는 정의가 다르므로 수치를
직접 비교하지 않는다. 이전 framework E2E run ID는 SST-2 `f350cfc9`, SQuAD v1
`52ee87b4`였고 당시 ARIES 환경은 qbruntime `v1.3.2`, driver `1.13.0`이었다. 새
attempt 결과가 생기면 `result.json`의 artifact hash, stage exit code와 환경을 함께
기록한다.

별도로, 기존 벤더 MXQ의 native async slot lifecycle은 ARIES에서 성공했다. 이는 새
qbcompiler recipe의 성공이 아니다.

| 기존 vendor artifact | SHA256 | 관측 |
|---|---|---|
| ResNet50 `resnet50_IMAGENET1K_V2.mxq` | `5212979749555738439bab7c851edd8e6d2f0cf76da580ce15cf2d74b70a1c49` | global8 `infer_async` 1회와 callback-blocked 2-submit slot lifecycle pass |
| YOLOv5m `yolov5m.mxq` | `fabda3ea34a4422afa864fe9c191395dfbe0bc17d341e6fdfdbf840c7e45c257` | global8, 세 raw head의 callback-blocked 2-submit slot lifecycle pass |

새로 만든 ResNet50·YOLOv5m의 byte hash가 vendor artifact와 같을 필요는 없다. 대신
source, calibration, ABI, ARIES 1회 추론과 task 품질을 각각 검증한다.

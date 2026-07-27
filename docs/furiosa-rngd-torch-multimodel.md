# Furiosa RNGD tensor model 실행 가이드

이 문서는 Llama 이외의 고정 shape 모델을 ML-HW-Benchmark-Framework의 공통
동기(`e2e`)·비동기(`async_queue`) 파이프라인에서 RNGD로 실행하는 절차를
정리한다.

## 실행 경로와 지원 판정

두 Furiosa target은 목적과 비동기 구현이 다르다.

| Target | 모델 | 실행 API | 비동기 방식 |
|---|---|---|---|
| `furiosa-rngd` | Llama 등 생성 모델 | `furiosa_llm.LLM`, FXB | Furiosa native async |
| `furiosa-rngd-torch` | 비전·BERT·PatchTST | `furiosa.torch` + `torch.compile` | 프레임워크 queue + blocking worker 1개 |

`furiosa-rngd-torch`는 `fullgraph=True`, `dynamic=False`,
`eager_fallback=False`, batch 1로만 실행한다. 컴파일되지 않은 연산을 CPU에서
조용히 실행하지 않는다. 따라서 프레임워크 연결 테스트가 통과하더라도 첫 RNGD
호출에서 compiler가 실패하면 해당 SDK 버전에서는 **실장비 미지원**으로 판정한다.

현재 상태는 다음과 같다.

| 모델 프로필 | 로컬 모델 소스 | 프레임워크 통합 | RNGD 2026.3 실장비 |
|---|---|---|---|
| `resnet50` | Kalray ONNX → PyTorch | 완료 | compiler 내부 panic 재현, 차단됨 |
| `yolov5m` | Ultralytics YOLOv5u-medium `yolov5mu.pt` | 완료 | 검증 필요 |
| `bert-base-uncased` | HF PyTorch SST-2 | 완료 | 검증 필요 |
| `bert-base-uncased-squad-v1` | HF PyTorch SQuAD v1 | 완료 | 검증 필요 |
| `patchtst-fm-r1` | IBM TSFM custom PyTorch | 완료 | 검증 필요, 선택 의존성 있음 |
| `patchtst-etth1` | Transformers PatchTST | 완료 | 검증 필요 |

ResNet50 실패는 입력 연결 실패가 아니다. 정확한 Kalray ONNX의 CPU parity를
확인한 뒤 full model과 단일 `Conv2d(512, 512, 3)` 모두에서
`align_up_required`, `EinsumByDpe`, `Option::unwrap` compiler panic을 재현했다.
같은 로그가 나오면 전처리나 프레임워크를 바꾸지 말고 Furiosa compiler 이슈로
보존한다.

## 1. 서버에서 코드와 환경 준비

기존 checkout이 dirty이면 그 자리에서 branch를 바꾸지 말고 새 worktree를 만든다.

```bash
cd ~/ML-HW-Benchmark-Framework

git status --short
git fetch origin feat/furiosa-rngd-multimodel
VERIFY_SUFFIX="$(date +%Y%m%d-%H%M%S)"
VERIFY_DIR="$HOME/ML-HW-Benchmark-Framework-furiosa-multimodel-${VERIFY_SUFFIX}"
git worktree add \
  --detach \
  "$VERIFY_DIR" \
  origin/feat/furiosa-rngd-multimodel

cd "$VERIFY_DIR"
export ROOT="$PWD"
git branch --show-current
git rev-parse --short HEAD
git status --short
```

PR이 main에 merge된 뒤에는 clean main worktree에서 다음처럼 갱신하면 된다.

```bash
cd ~/ML-HW-Benchmark-Framework-main
git switch main
git pull --ff-only origin main
```

Furiosa-LLM 환경과 Torch 환경은 분리한다.

```bash
cd "$ROOT"

uv venv --python 3.12 .venv-furiosa-torch
uv pip install \
  --python .venv-furiosa-torch/bin/python \
  -r framework/requirements-furiosa-torch.txt

.venv-furiosa-torch/bin/python - <<'PY'
from importlib.metadata import version
import furiosa.torch

for package in (
    "furiosa-torch",
    "torch",
    "transformers",
    "numpy",
    "onnx",
    "onnx2torch",
    "ultralytics",
):
    print(f"{package}: {version(package)}")
print("furiosa.torch import: OK")
PY

furiosa-smi info
furiosa-smi status
furiosa-smi ps
```

검증 서버 기준 핵심 버전은 `furiosa-torch 2026.3.0`, `torch 2.10.0`,
Python 3.12이다. 설치된 SDK 조합이 다르면 결과에 버전을 함께 남긴다.

## 2. 모델 다운로드

아래 명령은 repository root에서 실행한다.

```bash
cd "$ROOT"

ROOT="$PWD"
PY="$ROOT/.venv-furiosa-torch/bin/python"
HF="$ROOT/.venv-furiosa-torch/bin/hf"

# ResNet50: 정확한 Kalray ONNX 소스
cd "$ROOT/framework"
"$PY" models/prepare_resnet50_kalray.py --format onnx --output models
cd "$ROOT"

# YOLOv5m: .pt를 모델 디렉터리에 직접 받는다.
mkdir -p framework/models/yolov5m
cd framework/models/yolov5m
"$PY" -c 'from ultralytics import YOLO; YOLO("yolov5mu.pt")'
cd "$ROOT"

# BERT SST-2
"$HF" download textattack/bert-base-uncased-SST-2 \
  --local-dir framework/models/textattack_bert-base-uncased-SST-2

# BERT SQuAD v1
"$HF" download csarron/bert-base-uncased-squad-v1 \
  --local-dir framework/models/csarron_bert-base-uncased-squad-v1

# IBM custom PatchTST-FM-r1
"$HF" download ibm-research/patchtst-fm-r1 \
  --local-dir framework/models/ibm-research_patchtst-fm-r1

# 표준 Transformers PatchTST
"$HF" download ibm-granite/granite-timeseries-patchtst \
  --local-dir framework/models/ibm-granite_granite-timeseries-patchtst
```

`patchtst-fm-r1`만 IBM custom class가 필요하다. `granite-tsfm 0.3.6`은 공식
메타데이터에서 `transformers<5`, `scikit-learn<1.8`, `pandas>=2.3.3`,
`deprecated`, `filelock>=3.20.3`, `einops>=0.7`을 선언한다. 이 중 Furiosa와
충돌하지 않는 의존성은 `requirements-furiosa-torch.txt`에 고정했다. Furiosa
2026.3 서버가 Transformers 5.x를 사용한다면 기본 환경을 다운그레이드하지 말고
granite-tsfm 코드만 설치한 뒤 즉시 전체 import를 검증한다.

```bash
cd "$ROOT"
uv pip install \
  --python "$PY" \
  --no-deps \
  granite-tsfm==0.3.6

"$PY" - <<'PY'
import deprecated
import einops
import filelock
import pandas
import sklearn
from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction
print("PatchTST-FM overlay imports: OK")
PY
```

이 import가 실패하면 Transformers를 임의로 downgrade하지 않는다. 그 환경에서는
우선 `patchtst-etth1`만 검증하고, `patchtst-fm-r1`은 별도 호환성 이슈로 기록한다.
또한 `ibm-research/patchtst-fm-r1`은 모델 카드상 비상업적 연구용 라이선스이므로
사용 범위를 확인한다.

## 3. 데이터셋 준비

```bash
cd "$ROOT/framework"

# ImageNet validation 3,000건
"$PY" datasets/prepare_imagenet_1k.py

# COCO128
"$PY" datasets/prepare_coco128.py

# GLUE SST-2 validation, 고정 길이 128
"$PY" datasets/prepare_text_numpy.py \
  --model-id textattack/bert-base-uncased-SST-2 \
  --dataset-name glue \
  --dataset-config sst2 \
  --split validation \
  --seq-len 128 \
  --output-dir datasets/sst2_numpy

# SQuAD v1 validation, 고정 길이 384
"$PY" datasets/prepare_squad_numpy.py \
  --model-id csarron/bert-base-uncased-squad-v1 \
  --seq-len 384 \
  --output-dir datasets/squad_numpy

# ETTh1
"$PY" datasets/prepare_etth1.py --output-dir datasets/etth1
```

## 4. 프레임워크 연결 smoke test

이 테스트는 SDK가 없는 개발 호스트에서도 registry, strict runtime 계약, 모델
wrapper, 공통 pipeline routing을 검증한다. 실장비 컴파일 성공을 뜻하지는 않는다.

```bash
cd "$ROOT/framework"

"$PY" -m pytest \
  tests/test_furiosa_torch_environment_contract.py \
  tests/test_furiosa_torch_models.py \
  tests/test_furiosa_torch_runtime.py \
  tests/test_furiosa_torch_integration.py \
  -q

# 실제 checkpoint가 프레임워크의 정적 YOLO 출력 계약과 맞는지 CPU에서 확인한다.
"$PY" - <<'PY'
import torch
from runtimes.furiosa_torch_models import get_torch_model_adapter

model = get_torch_model_adapter("yolov5m").loader(
    "models/yolov5m/yolov5mu.pt"
)
with torch.inference_mode():
    output = model(torch.zeros(1, 3, 640, 640))
assert tuple(output.shape) == (1, 84, 8400), tuple(output.shape)
print("YOLOv5u medium CPU output contract: OK", tuple(output.shape))
PY
```

## 5. 모델별 e2e와 async_queue 실행

다음 shell 함수를 한 번 등록한다. 첫 호출에는 컴파일이 포함될 수 있으므로 smoke는
1건부터 시작한다. `timeout`의 강제 종료 여유는 Rust compiler thread가 `Ctrl-C`에
즉시 반응하지 않는 경우를 위한 것이다.

```bash
cd "$ROOT/framework"
mkdir -p results

run_furiosa_e2e() {
  local model="$1"
  local model_path="$2"
  local dataset="$3"
  local steps="${4:-1}"

  timeout --signal=INT --kill-after=30s 45m \
    "$PY" src/main.py \
      --model "$model" \
      --target furiosa-rngd-torch \
      --model-path "$model_path" \
      --dataset "$dataset" \
      --inference-mode e2e \
      --batch-size 1 \
      --warmup 1 \
      --max-steps "$steps" \
      --results-path "results/furiosa-${model}-e2e.csv"
}

run_furiosa_async() {
  local model="$1"
  local model_path="$2"
  local dataset="$3"
  local samples="${4:-10}"

  timeout --signal=INT --kill-after=30s 45m \
    "$PY" src/main.py \
      --model "$model" \
      --target furiosa-rngd-torch \
      --model-path "$model_path" \
      --dataset "$dataset" \
      --inference-mode async_queue \
      --scenario offline \
      --batch-size 1 \
      --warmup 1 \
      --max-samples "$samples" \
      --min-samples "$samples" \
      --queue-capacity 32 \
      --worker-count 1 \
      --save-request-trace \
      --results-path "results/furiosa-${model}-async.csv"
}
```

모델별 명령은 다음과 같다.

```bash
# ResNet50
run_furiosa_e2e \
  resnet50 \
  models/Kalray_resnet50/resnet50-v1-7s.onnx \
  datasets/imagenet_1k \
  1
run_furiosa_async \
  resnet50 \
  models/Kalray_resnet50/resnet50-v1-7s.onnx \
  datasets/imagenet_1k \
  10

# YOLOv5m
run_furiosa_e2e \
  yolov5m \
  models/yolov5m/yolov5mu.pt \
  datasets/coco128 \
  1
run_furiosa_async \
  yolov5m \
  models/yolov5m/yolov5mu.pt \
  datasets/coco128 \
  10

# BERT SST-2
run_furiosa_e2e \
  bert-base-uncased \
  models/textattack_bert-base-uncased-SST-2 \
  datasets/sst2_numpy \
  1
run_furiosa_async \
  bert-base-uncased \
  models/textattack_bert-base-uncased-SST-2 \
  datasets/sst2_numpy \
  10

# BERT SQuAD v1
run_furiosa_e2e \
  bert-base-uncased-squad-v1 \
  models/csarron_bert-base-uncased-squad-v1 \
  datasets/squad_numpy \
  1
run_furiosa_async \
  bert-base-uncased-squad-v1 \
  models/csarron_bert-base-uncased-squad-v1 \
  datasets/squad_numpy \
  10

# IBM custom PatchTST-FM-r1
run_furiosa_e2e \
  patchtst-fm-r1 \
  models/ibm-research_patchtst-fm-r1 \
  datasets/etth1/ETTh1.csv \
  1
run_furiosa_async \
  patchtst-fm-r1 \
  models/ibm-research_patchtst-fm-r1 \
  datasets/etth1/ETTh1.csv \
  10

# 표준 Transformers PatchTST
run_furiosa_e2e \
  patchtst-etth1 \
  models/ibm-granite_granite-timeseries-patchtst \
  datasets/etth1/ETTh1.csv \
  1
run_furiosa_async \
  patchtst-etth1 \
  models/ibm-granite_granite-timeseries-patchtst \
  datasets/etth1/ETTh1.csv \
  10
```

smoke가 성공한 모델만 샘플 수를 늘린다. e2e는 마지막 인자를 `1000`, async는
마지막 인자를 `1000`으로 바꾸면 된다. `--worker-count`는 1을 유지한다. queue
capacity는 대기열 크기이며 RNGD에서 동시에 실행되는 model instance 수가 아니다.

## 6. 실행 중 상태와 결과 판정

다른 터미널에서 다음을 확인한다.

```bash
watch -n 1 furiosa-smi status
watch -n 1 furiosa-smi ps
```

모델이 메모리에 올라갔는데 utilization이 0%인 구간은 host graph capture 또는
compiler가 실행 중일 수 있다. 프로세스와 compiler 로그를 함께 본다. 자세한 Rust
오류를 남기려면 단일 smoke 명령 앞에 다음 환경변수를 붙이고 전체 로그를 저장한다.

```bash
RUST_BACKTRACE=full run_furiosa_e2e \
  resnet50 \
  models/Kalray_resnet50/resnet50-v1-7s.onnx \
  datasets/imagenet_1k \
  1 2>&1 | tee /tmp/furiosa-resnet50.log
```

성공 기준은 다음을 모두 만족하는 것이다.

- 첫 RNGD 호출의 graph compile과 model load 완료
- `eager fallback is not allowed` 또는 compiler panic 없음
- 요청 수만큼 추론 완료
- 결과 CSV 저장
- async의 `async_run_status=valid`, failed/rejected/timed-out request가 모두 0
- 종료 뒤 `furiosa-smi ps`에서 프로세스와 메모리 해제

두 PatchTST profile은 같은 ETTh1 원본을 쓰지만 raw/normalized 캐시 경로를
분리한다. 따라서 어느 모델을 먼저 실행해도 다른 profile의 전처리 캐시를 재사용하지
않는다.

`async_queue`의 worker utilization은 프레임워크 worker 사용률이다. Furiosa native
engine 동시성이나 RNGD core utilization을 뜻하지 않는다. 전력·에너지 수치도 현재
system monitor target에는 자동 저장되지 않으므로 별도 SMI sampling을 결과와 함께
기록해야 한다.

## 참고 자료

- [FuriosaAI RNGD 개요](https://developer.furiosa.ai/latest/en/overview/rngd.html)
- [Furiosa SDK 2026.3 release notes](https://developer.furiosa.ai/latest/en/whatsnew/release-2026.3.0.html)
- [IBM PatchTST-FM-r1](https://huggingface.co/ibm-research/patchtst-fm-r1)
- [IBM Granite PatchTST](https://huggingface.co/ibm-granite/granite-timeseries-patchtst)
- [granite-tsfm 0.3.6](https://pypi.org/project/granite-tsfm/0.3.6/)

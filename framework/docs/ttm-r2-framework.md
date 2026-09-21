# TTM-R2 3종 NPU 실행 및 최종 판정

이 문서는 IBM Granite TTM-R2를 동일한 ETTh1 `OT` 240-window 계약으로
Furiosa RNGD, Rebellions CA22, Mobilint ARIES에서 실행하고 최종 성공 여부를
판정하는 절차다. 세 실행 모두 `src/main.py`가 조립하는 공통
`BenchmarkRunner`와 ResultStore를 사용한다.

## 고정 계약

| 항목 | 값 |
|---|---|
| 모델 | `ttm-r2` |
| 입력 / 출력 | FP32 `[1,512,1]` / FP32 `[1,96,1]` |
| 데이터 | ETTh1 `OT`, test split, stride 1, 240 windows |
| contract ID | `ttm-r2-etth1-ot-512-96-v1` |
| `config.json` SHA-256 | `5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5` |
| `model.safetensors` SHA-256 | `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710` |
| `ETTh1.csv` SHA-256 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| RBLN artifact SHA-256 | `4159ce147a9d91524117b39eba67a706df1c5c987e3317a999e3992c2d3bf172` |
| ARIES artifact SHA-256 | `208958c81f1c62ad47557158b788f7f758ac8ef634032286e7f304dac752e217` |

Furiosa는 고정 체크포인트를 런타임에서 strict compile한다. RBLN과 ARIES는
위 해시의 기존 precompiled artifact를 사용하며 `main.py`에서 다시 컴파일하지
않는다. ARIES 내부 artifact ABI(INT8 `[1,8,64]` → FP32 `[1,1,96]`)는 전용
어댑터가 공통 의미 ABI로 변환한다.

## 서버 경로와 사전 확인

아래 예시는 현재 서버에서 검증된 자산 위치를 사용한다. `TTM_PR`만 PR 브랜치를
checkout한 디렉터리에 맞춘다.

```bash
export TTM_PR="$HOME/ML-HW-Benchmark-Framework-ttm-r2-integration"
export TTM_MODEL="$HOME/ML-HW-Benchmark-Framework-furiosa-compile-repro/framework/models/ibm-granite_granite-timeseries-ttm-r2"
export TTM_DATASET="$HOME/ML-HW-Benchmark-Framework-furiosa-compile-repro/framework/datasets/etth1/ETTh1.csv"
export TTM_RBLN="$HOME/ML-HW-Benchmark-Framework-rbln/framework/results/ttm-r2/rbln-rerun-20260806T021145Z/ttm-r2-core.rbln"
export TTM_ARIES="$HOME/ttm-r2-aries-20260806T023135Z/ttm-r2-core.mxq"
export TTM_RESULTS="$(mktemp -d "$HOME/ttm-r2-framework-final-XXXXXX")"

test -f "$TTM_PR/framework/src/main.py"
test -x "$HOME/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python"
test -x "$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"
test -x "$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python"

sha256sum "$TTM_MODEL/config.json" \
  "$TTM_MODEL/model.safetensors" \
  "$TTM_DATASET" \
  "$TTM_RBLN" \
  "$TTM_ARIES"
```

출력 해시는 위 표와 모두 일치해야 한다.

## 1. Furiosa RNGD

```bash
"$HOME/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python" \
  "$TTM_PR/framework/src/main.py" \
  --model ttm-r2 \
  --target furiosa-rngd-torch \
  --model-path "$TTM_MODEL" \
  --dataset "$TTM_DATASET" \
  --batch-size 1 \
  --inference-mode e2e \
  --compile \
  --warmup 2 \
  --results-path "$TTM_RESULTS/furiosa.csv"
```

이 경로는 로컬 체크포인트에서 Furiosa Torch strict compile을 수행한 뒤 RNGD에서
240개 window를 실행한다.

## 2. Rebellions CA22

```bash
"$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python" \
  "$TTM_PR/framework/src/main.py" \
  --model ttm-r2 \
  --target rbln-static \
  --artifact "$TTM_RBLN" \
  --dataset "$TTM_DATASET" \
  --batch-size 1 \
  --inference-mode e2e \
  --warmup 2 \
  --results-path "$TTM_RESULTS/rbln.csv"
```

## 3. Mobilint ARIES

```bash
"$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python" \
  "$TTM_PR/framework/src/main.py" \
  --model ttm-r2 \
  --target mobilint-aries \
  --artifact "$TTM_ARIES" \
  --dataset "$TTM_DATASET" \
  --batch-size 1 \
  --inference-mode e2e \
  --warmup 2 \
  --results-path "$TTM_RESULTS/mobilint.csv"
```

최종 ARIES 행에는 `mobilint_saturation_elements=0`과
`mobilint_saturation_total=122880`이 기록되어야 한다. warmup에서 발생한
양자화는 이 측정 합계에 포함되지 않는다.

## 4. 3종 최종 판정

```bash
"$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python" \
  "$TTM_PR/framework/tools/validate_ttm_r2_matrix.py" \
  --furiosa-csv "$TTM_RESULTS/furiosa.csv" \
  --rbln-csv "$TTM_RESULTS/rbln.csv" \
  --mobilint-csv "$TTM_RESULTS/mobilint.csv"
```

판정기는 각 CSV에서 해당 target의 최신 `ttm-r2` 행을 선택한다. 세 행 모두
full scope, 240 samples, 고정 데이터셋 해시, 유한 MAE/RMSE와 과거 실장 실행값
허용오차(`rtol=1e-3`, `atol=1e-4`)를 만족해야 한다. RBLN/ARIES artifact
해시도 위 표와 일치해야 하고 ARIES saturation은 0이어야 한다. 모두 통과하면
한 줄 JSON의 `status`가 `pass`이고 종료 코드는 0이다.

과거 CPU reference는 MAE `1.76834237575531`, RMSE
`2.106243848800659`이다. 이 값은 동일 workload 비교용 기존 증거이며 각 NPU
실행 안에서 CPU 모델을 다시 돌리지 않는다. strict tensor parity도 별도
진단이다. 과거 Furiosa/RBLN strict parity 실패를 숨기거나 새로 만들지 않으며,
task-level MAE/RMSE와 operational success 판정에 섞지 않는다.

`--max-steps`를 지정한 실행은 smoke로 기록되며 최종 3종 성공으로 인정되지
않는다. CSV, 모델, 데이터셋, compiled artifact와 prediction bundle은 Git에
추가하지 않는다.

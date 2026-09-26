# TTM-R2 3종 NPU 실행 및 최종 판정

이 문서는 IBM Granite TTM-R2를 동일한 ETTh1 `OT` 240-window 계약으로
Furiosa RNGD, Rebellions CA22, Mobilint ARIES에서 실행하고 최종 성공 여부를
판정하는 절차다. 세 실행 모두 `src/main.py`가 조립하는 공통
`BenchmarkRunner`와 ResultStore를 사용한다.

다른 모델을 포함한 전체 검증 단계와 cross-NPU 비교 경계는
[Cross-NPU Transformer·시계열 모델 검증 현황](cross-npu-model-validation.md)을
함께 따른다.

## 고정 계약

서버 검증 source revision과 현재 framework 이식 commit의 대응 관계는
[중앙 검증 현황의 TTM-R2 절](cross-npu-model-validation.md#4-ttm-r2-공통-계약과-결과)에
기록한다. 두 계열은 직접 ancestor 관계가 아니므로 source branch commit을 현재
브랜치 이력으로 표현하지 않는다.

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

아래 예시는 현재 서버에서 검증된 자산 위치를 사용한다. `TTM_PR`만
`feat/power_measurment`를 checkout한 worktree에 맞춘다.

```bash
export TTM_PR="$HOME/ML-HW-Benchmark-Framework-power-measurment"
export TTM_MODEL="$HOME/ML-HW-Benchmark-Framework-furiosa-compile-repro/framework/models/ibm-granite_granite-timeseries-ttm-r2"
export TTM_DATASET="$HOME/ML-HW-Benchmark-Framework-furiosa-compile-repro/framework/datasets/etth1/ETTh1.csv"
export TTM_RBLN="$HOME/ML-HW-Benchmark-Framework-rbln/framework/results/ttm-r2/rbln-rerun-20260806T021145Z/ttm-r2-core.rbln"
export TTM_ARIES="$HOME/ttm-r2-aries-20260806T023135Z/ttm-r2-core.mxq"
export TTM_RESULTS="$(mktemp -d "$HOME/ttm-r2-framework-final-XXXXXX")"

test -f "$TTM_PR/framework/src/main.py"
test -x "$HOME/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python"
test -x "$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"
test -x "$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python"

# 기존 Furiosa Torch 환경을 재사용할 때 원시 전력 API를 명시적으로 준비한다.
"$HOME/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python" \
  -m pip install --no-deps furiosa-smi-py==2026.1.2
"$HOME/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python" \
  -m pip check

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
  --power-trace \
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
  --power-trace \
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
  --power-trace \
  --results-path "$TTM_RESULTS/mobilint.csv"
```

최종 ARIES 행에는 `mobilint_saturation_elements=0`과
`mobilint_saturation_total=122880`이 기록되어야 한다. warmup에서 발생한
양자화는 이 측정 합계에 포함되지 않는다.

## 원시 전력 trace 확인

세 실행은 워밍업 뒤 3초 `baseline`과 240-window `inference`의 원시 W를
0.2초 주기로 기록한다. 결과 행의 `power_trace_path`는 모두
`power/<run_id>.power.csv` 형식이어야 하며, 파일 SHA-256과
`power_trace_sha256`이 일치해야 한다. Furiosa는
`furiosa-smi-py`/`device`, RBLN은 `rbln-smi-json`/`whole_card`, ARIES는
`mbltml`/`device_total` 범위다.

```bash
"$HOME/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python" - \
  "$TTM_RESULTS/furiosa.csv" \
  "$TTM_RESULTS/rbln.csv" \
  "$TTM_RESULTS/mobilint.csv" <<'PY'
import csv
import hashlib
import sys
from pathlib import Path

for result_name in sys.argv[1:]:
    result_path = Path(result_name)
    rows = list(csv.DictReader(result_path.open()))
    row = rows[-1]
    assert row["power_trace_status"] in {"complete", "partial"}
    trace = result_path.parent / row["power_trace_path"]
    assert trace.is_file(), trace
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == row["power_trace_sha256"]
    samples = list(csv.DictReader(trace.open()))
    assert len(samples) == int(row["power_trace_sample_count"])
    assert {sample["phase"] for sample in samples} == {"baseline", "inference"}
    assert all(sample["run_id"] == row["run_id"] for sample in samples)
    print(result_path.name, row["power_monitor_source"], row["power_scope"], len(samples))
PY
```

`partial`은 파일은 보존됐지만 `read_error`, `unavailable` 또는 `overrun` 표본이
하나 이상 있다는 뜻이므로 `sample_status`와 `error_code`를 확인해야 한다.
프레임워크는 총에너지, idle 차감 또는 평균 `J/inference`를 계산하지 않는다.
세 장치의 측정 범위가 다르므로 에너지 분석과 범위 정규화는 이 원시 CSV를 입력으로
별도 수행한다.

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

### 서버 보존 결과

| 실행 | MAE | RMSE | CPU 대비 MAE | CPU 대비 RMSE | 별도 진단 |
|---|---:|---:|---:|---:|---|
| CPU reference | 1.7683423758 | 2.1062438488 | - | - | 고정 reference |
| Furiosa RNGD | 1.7680668831 | 2.1058883667 | -0.015579% | -0.016878% | strict tensor `parity_failed` |
| Rebellions CA22 | 1.7690539360 | 2.1070954799 | +0.040239% | +0.040434% | strict tensor `parity_failed` |
| Mobilint ARIES | 1.8834694624 | 2.2656056881 | +6.510452% | +7.566163% | unsaturated, 0/122,880 saturated elements |

세 NPU 실행은 모두 exit code 0과 task-quality 계산을 완료했다. Furiosa와
RBLN의 strict parity 실패는 tensor-level 진단이며 위 MAE/RMSE 결과를 실패로
바꾸지 않는다. ARIES 결과를 CPU와 “동등”하다고 표현하려면 별도의 사전 품질
허용 기준이 필요하다.

전체 결과를 얻을 당시의 Furiosa 환경과 별개로, Torch 2.10.0+cpu,
Furiosa Torch 2026.3.0, Transformers 4.57.6, NumPy 2.5.1 조합에서도
`--max-steps 1` strict compile·실행 smoke가 성공했다. 이는 4.57.6 호환성
증거일 뿐 240-window full 결과를 대체하지 않는다.

`--max-steps`를 지정한 실행은 smoke로 기록되며 최종 3종 성공으로 인정되지
않는다. CSV, 모델, 데이터셋, compiled artifact와 prediction bundle은 Git에
추가하지 않는다.

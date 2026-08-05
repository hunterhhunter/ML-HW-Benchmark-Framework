# Granite TTM-R1: CA22, RNGD, ARIES strict compilation

이 runbook은 사전학습 checkpoint
`ibm-granite/granite-timeseries-ttm-r1`를 단변량 고정 계약으로 검증한다.

```text
context  float32 [1,512,1]
forecast float32 [1,96,1]
```

TTM-R1은 입력 데이터를 외부에서 standard scaling 하도록 설계되어 있다. CA22가
checkpoint 내부의 standard-scaler graph를 지원하지 않으므로, 외부 scaling뿐 아니라
그 내부 scaler의 동일한 mean/variance(`+1e-5`) 계산·NaN 대체·두 단계 복원까지 CPU
host adapter가 수행한다. NPU에는 scaler-free FP32 `past_values [1,512,1]`와
TinyTimeMixer prediction core만 들어간다. 각 command는 finite/NaN CPU preflight에서
원본 checkpoint와 scaler-free core를 직접 비교하며, 불일치 시 compiler를 호출하지
않는다.

모든 실행은 **새 output directory**를 사용한다. 결과 파일의 `device_verified`만
실제 device 실행과 공통 numeric gate(`rtol=1e-3`, `atol=1e-3`) 통과를 뜻한다.
`compile_failed` 또는 `parity_failed`는 실패 증거이며 fallback 또는 tolerance
완화의 근거가 아니다.

TTM-R1 checkpoint의 공식 구현은 IBM `granite-tsfm`에도 들어 있다. 현재 CA22
환경의 `transformers 5.8.1`은 `TinyTimeMixerForPrediction`을 최상위로 export하지
않으므로, vendor SDK의 Torch/Transformers 버전을 바꾸지 않도록 아래 checkpoint
절차에서 IBM 패키지를 의존성 없이 먼저 추가한다. 모델 로더는 Transformers 구현이
있으면 그것을 우선 쓰고, 없으면 이 IBM 구현을 사용한다. import 단계에서 별도의
누락 모듈 오류가 나면 SDK 환경 전체를 업그레이드하지 말고, 해당 traceback을
보존한다.

## 1. Checkpoint acquisition

아래는 한 번만 실행한다. 이후 `--model-path`는 로컬 checkpoint만 읽으며 자동
download를 하지 않는다.

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework-rbln/framework
PY=/home/etri_ecas/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python
MODEL=/home/etri_ecas/ML-HW-Benchmark-Framework/framework/models/ibm-granite_granite-timeseries-ttm-r1

uv pip install --python "$PY" --no-deps "granite-tsfm==0.2.27"
"$PY" -c "from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction; print(TinyTimeMixerForPrediction.__module__)"
"$PY" tools/acquire_ttm_r1.py --output-dir "$MODEL"
"$PY" tools/ttm_r1_compile.py --vendor reference --describe
OUT=results/ttm-r1/reference-$(date -u +%Y%m%dT%H%M%SZ)
"$PY" tools/ttm_r1_compile.py --vendor reference --model-path "$MODEL" --output-dir "$OUT"
```

`$MODEL/ttm-r1-manifest.json`에는 config와 weight의 SHA-256이 기록된다. CPU
reference가 실패하면 아래 vendor command를 실행하지 말고 result와 traceback을
보존한다.

## 2. Rebellions CA22

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework-rbln/framework
RBLN_PY=/home/etri_ecas/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python
MODEL=/home/etri_ecas/ML-HW-Benchmark-Framework/framework/models/ibm-granite_granite-timeseries-ttm-r1
OUT=results/ttm-r1/rbln-$(date -u +%Y%m%dT%H%M%SZ)

"$RBLN_PY" -m pytest tests/test_ttm_r1_contracts.py tests/test_ttm_r1_host_adapter.py tests/test_ttm_r1_core.py tests/test_ttm_r1_reference.py tests/test_ttm_r1_compile_cli.py tests/test_ttm_r1_rbln.py -q
mkdir -p results/ttm-r1
rbln-smi -b -j -d 0 > "${OUT}-before.json"
"$RBLN_PY" tools/ttm_r1_compile.py --vendor rbln --model-path "$MODEL" --output-dir "$OUT"
rbln-smi -b -j -d 0 > "${OUT}-after.json"
```

성공 시 `$OUT/ttm-r1-core.rbln`과 `$OUT/rbln-result.json`이 생성된다. JSON에는
artifact SHA-256, CA22 inspect ABI, finite/NaN core parity가 포함된다.

`DEVICE_GRAPH_CONVERSION`처럼 compiler가 연산명을 내보내지 않는 실패는 새
directory에서 stage bisection을 실행한다. 이는 benchmark artifact가 아니라
scaler, static patchify, encoder, decoder, head, restore를 각각 compile하여 첫
실패 경계를 `rbln-bisect-result.json`에 남기는 진단 명령이다.

```bash
OUT=results/ttm-r1/rbln-bisect-$(date -u +%Y%m%dT%H%M%SZ)
"$RBLN_PY" tools/ttm_r1_rbln_bisect.py --model-path "$MODEL" --output-dir "$OUT"
"$RBLN_PY" -m json.tool "$OUT/rbln-bisect-result.json"
```

## 3. Furiosa RNGD

Furiosa는 portable artifact 대신 strict first call을 검증한다. `fullgraph=True`,
`dynamic=False`, `eager_fallback=False`가 고정되어 있다.

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework-furiosa-compile-repro/framework
FURIOSA_PY=/home/etri_ecas/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python
MODEL=/home/etri_ecas/ML-HW-Benchmark-Framework/framework/models/ibm-granite_granite-timeseries-ttm-r1
OUT=results/ttm-r1/furiosa-$(date -u +%Y%m%dT%H%M%SZ)

"$FURIOSA_PY" -m pytest tests/test_ttm_r1_contracts.py tests/test_ttm_r1_host_adapter.py tests/test_ttm_r1_core.py tests/test_ttm_r1_reference.py tests/test_ttm_r1_compile_cli.py tests/test_ttm_r1_furiosa.py -q
mkdir -p results/ttm-r1
furiosa-smi > "${OUT}-before.txt"
"$FURIOSA_PY" tools/ttm_r1_compile.py --vendor furiosa --model-path "$MODEL" --output-dir "$OUT"
furiosa-smi > "${OUT}-after.txt"
```

성공 조건은 `$OUT/furiosa-result.json`의 `status: device_verified`다. Furiosa
compiler exception도 동일 output directory의 JSON에 보존된다.

## 4. Mobilint ARIES

Mobilint는 static ONNX export를 ONNX Runtime CPU에서 먼저 실행한 뒤,
`qbcompiler.mblt_compile_V2(..., target_device="aries-rb")`로 compile하고
`qbruntime.Model.infer_to_float()`로 ARIES device 0에서 실행한다. `qbcompiler`,
`onnxruntime`, `onnx`가 모두 선택된 virtual environment에 있어야 한다.

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework-chronos-bolt/framework
MBLT_PY=/home/etri_ecas/ML-HW-Benchmark-Framework/.venv-mobilint/bin/python
MODEL=/home/etri_ecas/ML-HW-Benchmark-Framework/framework/models/ibm-granite_granite-timeseries-ttm-r1
OUT=results/ttm-r1/mobilint-$(date -u +%Y%m%dT%H%M%SZ)

"$MBLT_PY" -m pytest tests/test_ttm_r1_contracts.py tests/test_ttm_r1_host_adapter.py tests/test_ttm_r1_core.py tests/test_ttm_r1_reference.py tests/test_ttm_r1_compile_cli.py tests/test_ttm_r1_mobilint.py -q
mkdir -p results/ttm-r1
mobilint-cli
"$MBLT_PY" tools/ttm_r1_compile.py --vendor mobilint --model-path "$MODEL" --output-dir "$OUT"
```

성공 시 `$OUT/ttm-r1-core.onnx`, `$OUT/ttm-r1-core.mblt`,
`$OUT/mobilint-result.json`이 생성된다. ONNX Runtime CPU parity와 finite/NaN
ARIES core parity는 결과 JSON에서 별도로 확인한다.

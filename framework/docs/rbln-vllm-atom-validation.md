# Rebellions ATOM Llama vLLM 검증 보고서

## 1. 검증 범위

이 보고서는 RBLN-CA22 한 장에서 Llama 3.2 3B와 Llama 3.1 8B를 준비하고
실행하는 재현 절차를 기록한다. 실제 지표 표는 제공된 물리 검증 증거가 있는
Llama 3.1 8B만을 대상으로 한다. 동기 run `a3168997`와 비동기 run
`9dd3bf7a`는 각각 정상 종료했고, 종료 뒤 `rbln-smi -j`의 `contexts`가 빈
배열임을 확인했다.

두 모델의 한 장 구성은 모두
`support_classification=unsupported_single_npu_experiment`인 비공식 실험이다.
Rebellions 공식 지원 구성은 각 모델의 8-NPU 구성이며, 이 결과가 그 지원
범위를 대체하거나 확대하지 않는다. 8B smoke는 생성 토큰을 하나만 사용했으므로
capacity, engine lifecycle, context cleanup 확인용이지 TPOT 또는 모델 품질
검증용이 아니다.

## 2. 서버와 Python 환경

- 장치: RBLN-CA22 1장, 읽을 수 있는 NPU 메모리 16,096 MiB
- KMD/firmware: 3.2.2
- Python: 3.10.12
- 패키지: `rebel-compiler==0.11.0`, `optimum-rbln==0.11.0.post1`,
  `vllm-rbln==0.11.0`, `torch==2.11.0+cpu`, `transformers==5.8.1`,
  `tokenizers==0.22.1`

이 서버는 hybrid 환경이다. 전역 user-site의 `rebel`을 사용하고 프로젝트
venv는 나머지 Python 패키지를 제공한다. 따라서 이 서버에서는
`PYTHONNOUSERSITE=1`을 설정하면 안 된다. 실행 전에 다음으로 interpreter,
패키지 origin, 단일 장치 상태를 함께 확인한다. 이 문서의 모든 명령은 아래
변수 정의를 전제로 한다.

```bash
export RBLN_FW_ROOT="$HOME/ML-HW-Benchmark-Framework-rbln-vllm"
export RBLN_VLLM_PY="$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"
export RBLN_DATASET="$RBLN_FW_ROOT/framework/datasets/squad2/val.json"
export RBLN_LLAMA32_DIR="$HOME/rebelion/rbln-model-zoo/custom/framework-contracts/llama-3.2-3b-npu1-seq512"
export RBLN_LLAMA31_DIR="$HOME/rebelion/rbln-model-zoo/custom/framework-contracts/llama-3.1-8b-npu1-seq512"
```

```bash
"$RBLN_VLLM_PY" - <<'PY'
import importlib.metadata as md
import sys

import rebel

print("python:", sys.executable)
print("rebel:", rebel.__file__)
for name in (
    "rebel-compiler",
    "optimum-rbln",
    "vllm-rbln",
    "torch",
    "transformers",
    "tokenizers",
    "vllm",
):
    distribution = md.distribution(name)
    print(f"{name}: {distribution.version} ({distribution.locate_file('')})")
print("NPU:", rebel.npu_is_available(0), rebel.get_npu_name(0))
assert rebel.npu_is_available(0)
assert rebel.device_count() == 1
PY
```

## 3. 모델 준비 및 컴파일 계약

artifact 디렉터리는 tokenizer/config, `.rbln` 파일, 그리고
`rbln-vllm-manifest.json`을 한 묶음으로 유지한다. 두 모델의 one-card
manifest는 1 device, sequence/block 512, batch 1, decoder batch 1 계약을
runtime과 정확히 맞춰야 하며 명시 opt-in이 필요하다. 8B one-card 실험은
추가로 NPU readable memory 15 GiB 이상을 전제로 한다.

### Llama 3.2 3B 한 장 compile

```bash
cd "$RBLN_FW_ROOT/framework"
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.2-3b \
  --output-dir "$RBLN_LLAMA32_DIR" \
  --num-devices 1 \
  --max-seq-len 512 \
  --block-size 512 \
  --batch-size 1 \
  --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

### Llama 3.1 8B 한 장 compile

```bash
cd "$RBLN_FW_ROOT/framework"
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.1-8b \
  --output-dir "$RBLN_LLAMA31_DIR" \
  --num-devices 1 \
  --max-seq-len 512 \
  --block-size 512 \
  --batch-size 1 \
  --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

compile 전에 `rbln-smi -j`의 `contexts`가 비어 있는지와 8B의 경우 15 GiB
이상의 readable memory를 확인한다. compile/load 전 artifact 일부만 복사하지
말고 manifest를 포함한 완전한 준비 디렉터리를 사용한다.

## 4. 실행 순서

먼저 정확한 SQuAD 파일이 존재하고 양수 개의 QA가 있는지 확인한 뒤, 실행할
모델의 manifest 검사, 동기 smoke, context cleanup, 비동기 smoke, 다시
context cleanup 순으로 실행한다. 아래 3B와 8B 명령은 재현 절차이며, 제공된
측정 지표와 run ID는 뒤의 8B 표에만 한정한다.

```bash
test -f "$RBLN_DATASET"
"$RBLN_VLLM_PY" - "$RBLN_DATASET" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
qa_count = sum(
    len(paragraph.get("qas", []))
    for article in payload.get("data", [])
    for paragraph in article.get("paragraphs", [])
)
print("SQuAD QA count:", qa_count)
assert qa_count > 0, "SQuAD dataset must contain at least one QA"
PY
```

### Llama 3.2 3B 한 장 실행 순서

```bash
test -f "$RBLN_LLAMA32_DIR/config.json"
test -f "$RBLN_LLAMA32_DIR/rbln-vllm-manifest.json"
find "$RBLN_LLAMA32_DIR" -type f -name '*.rbln' -ls
"$RBLN_VLLM_PY" -m json.tool "$RBLN_LLAMA32_DIR/rbln-vllm-manifest.json"
```

동기 E2E smoke:

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLAMA32_DIR" \
  --tokenizer-path "$RBLN_LLAMA32_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode e2e \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 16 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option allow_unsupported_single_npu=true \
  --warmup 1 \
  --max-steps 1 \
  --monitor \
  --results-path results/rbln-llama32-3b-npu1-e2e-smoke.csv

rbln-smi -j
```

`contexts: []`를 확인한 경우에만 다음 비동기 offline smoke를 실행한다.
`decoder_batch_sizes=1,`의 마지막 쉼표는 공용 parser가 정수로 coerce하지
않도록 해 list 계약을 유지한다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLAMA32_DIR" \
  --tokenizer-path "$RBLN_LLAMA32_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 16 \
  --worker-count 1 \
  --queue-capacity 1 \
  --min-samples 4 \
  --max-samples 4 \
  --warmup 1 \
  --flush-timeout-sec 600 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option decoder_batch_sizes=1, \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --debug \
  --results-path results/rbln-llama32-3b-npu1-async-smoke.csv

rbln-smi -j
```

### Llama 3.1 8B 한 장 실행 순서

```bash

test -f "$RBLN_LLAMA31_DIR/config.json"
test -f "$RBLN_LLAMA31_DIR/rbln-vllm-manifest.json"
find "$RBLN_LLAMA31_DIR" -type f -name '*.rbln' -ls
"$RBLN_VLLM_PY" -m json.tool "$RBLN_LLAMA31_DIR/rbln-vllm-manifest.json"
```

동기 E2E smoke는 다음과 같다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.1-8b \
  --target rbln-vllm \
  --model-path "$RBLN_LLAMA31_DIR" \
  --tokenizer-path "$RBLN_LLAMA31_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode e2e \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 1 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option allow_unsupported_single_npu=true \
  --warmup 1 \
  --max-steps 1 \
  --monitor \
  --debug \
  --results-path results/rbln-llama31-8b-npu1-e2e-smoke.csv

rbln-smi -j
```

동기 종료 후 `contexts: []`를 확인한 경우에만 다음 비동기 offline smoke를
실행한다. 공용 runtime-option parser가 `decoder_batch_sizes=1`을 정수로
coerce하므로, list 계약으로 전달하려면 마지막 쉼표를 포함한
`decoder_batch_sizes=1,` 문법이 필요하다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.1-8b \
  --target rbln-vllm \
  --model-path "$RBLN_LLAMA31_DIR" \
  --tokenizer-path "$RBLN_LLAMA31_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 1 \
  --worker-count 1 \
  --queue-capacity 1 \
  --min-samples 4 \
  --max-samples 4 \
  --warmup 1 \
  --flush-timeout-sec 600 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option decoder_batch_sizes=1, \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --debug \
  --results-path results/rbln-llama31-8b-npu1-async-smoke.csv

rbln-smi -j
```

## 5. 실제 검증 결과

| Mode | Run ID | Samples | Generated tokens | Engine latency/TTFT | Throughput | NPU memory peak | Process RAM peak | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| sync E2E | `a3168997` | 1 | 1 | 203.8591 ms | 4.9053 tokens/s | 14,630 MiB | 21,063.23 MiB | exit 0, contexts empty |
| async offline | `9dd3bf7a` | 4 | 4 | avg TTFT 202.7706 ms | 4.8801 tokens/s | 14,630 MiB | 21,260.79 MiB | valid, exit 0, contexts empty |

비동기 run은 4/4 요청을 완료했다. request E2E p99는 604.8718 ms, queue wait
p99는 205.7612 ms, service p99는 214.3395 ms였다. failed, rejected,
timed-out을 포함한 failure counter는 모두 0이었고 monitor coverage는 1.0,
native async counter도 모두 0이었다. 이 결과는 한 장 artifact의 engine
allocation과 lifecycle이 동기·비동기 경로에서 동작했음을 보이지만, 한 토큰
생성으로는 TPOT/ITL을 계산할 수 없고 모델 품질도 판정하지 않는다.

## 6. 트러블슈팅 기록

| 원인 | 증거 | 해결 |
|---|---|---|
| Portal/wheel 401 | Rebellions wheel index 인증 없이 SDK wheel 요청이 거절됐다. | 권한 있는 Rebellions index에 인증한다. credential은 저장소에 commit하지 않는다. |
| uv venv에 `rebel` 없음 | project venv만으로 import하면 `rebel`을 찾지 못했다. | 검증한 hybrid user-site 구성을 유지하거나 권한 있는 호환 wheel을 설치하고, package origin을 명시적으로 확인한다. |
| `vllm-rbln` uv 의존성 해석 | RBLN/vLLM 전용 index 조합에서 해석 충돌이 났다. | 호환되는 RBLN/vLLM index와 검증된 index strategy를 사용한 뒤 설치 버전을 확인한다. |
| 단일 NPU 8B 초기 거부 | opt-in 또는 one-card shape/batch 제약이 충족되지 않았다. | opt-in, manifest, context 512, batch 1, decoder batch 1, readable memory 15 GiB 이상을 함께 충족한다. |
| manifest 없음 또는 artifact 계약 불일치 | 일부 artifact만 있거나 compile/runtime device·shape·batch가 달랐다. | 완전한 준비 모델 디렉터리를 보존하고 compile/runtime 계약을 정확히 일치시킨다. |
| SQuAD path 불일치 | 현재 worktree script는 파일을 만들었지만 loader는 이전 worktree path를 계속 사용했다. | `--dataset`을 정확한 생성 파일로 지정하고 양수 QA 수를 assert하며 새 code guard에 의존한다. |
| `num_samples=0` false success | run `4930bef2`는 NPU memory 0 MiB, P14, util 0, EngineCore 생성 없음이었다. | 이 run을 invalid로 표시하고 하드웨어 검증 결과에서 제외한다. |
| TPOT/ITL이 0 또는 `None` | 한 generated token에는 inter-token interval이 없다. | TPOT benchmark는 multi-token run으로 수행한다. |
| 짧은 run의 util 0 | 약 204 ms burst는 monitor sampling 사이에 끝날 수 있다. | NPU memory, P-state, power, engine log, exit status, context cleanup을 acceptance evidence로 함께 확인한다. |

## 7. 운영 및 병합 판정

운영 runbook의 한 장 Llama 3.2 3B 및 Llama 3.1 8B compile, sync E2E, async
offline smoke 상태는 통과다. 단, 이 문서에 보존한 run ID와 측정 지표는 8B
physical evidence뿐이며, 두 모델의 결과 분류는 계속
`unsupported_single_npu_experiment`로 공식 8-NPU 지원 구성과 구분한다. 운영
성능 또는 TPOT 비교를 주장하려면 multi-token, 장시간, 충분한 monitor sampling을
갖춘 별도 측정이 필요하다. 원본 artifact, weights, tokenizers, datasets,
caches, CSV results, traces, logs와 credential은 이 문서 변경에 포함하지 않는다.

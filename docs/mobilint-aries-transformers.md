# Mobilint ARIES Transformer·LLM 실행 가이드

이 문서는 Mobilint ARIES에서 다음 모델을 실행하는 절차를 정리한다.

- BERT SST-2 문장 분류
- BERT SQuAD v1 질의응답
- PatchTST ETTh1 시계열 예측
- Llama 3.1 8B Instruct
- Llama 3.2 3B Instruct

컴파일러 연동은 범위 밖이다. BERT와 PatchTST는 아래 텐서 계약으로 별도
컴파일한 `.mxq`가 필요하다. 공개 Mobilint Model Zoo에는 이 세 benchmark와 정확히
일치하는 task-specific MXQ가 확인되지 않았으므로 base BERT나 다른 PatchTST MXQ를
대신 사용하면 안 된다. Llama는 Mobilint의 공식 Model Zoo Hugging Face repository를
전체 snapshot으로 내려받아 사용한다.

## 1. 실행 환경과 장치 확인

명령은 저장소 root에서 실행한다. 경로는 서버 환경에 맞게 한 번만 지정한다.

```bash
REPO="$(pwd)"
FW="$REPO/framework"
PY="$REPO/.venv-mobilint/bin/python"

test -x "$PY"
```

OS package, kernel module, device node와 CLI를 확인한다.

```bash
dpkg-query -W \
  -f='${Package}\t${Status}\t${Version}\n' \
  mobilint-aries-driver mobilint-qb-runtime mobilint-cli

lsmod | grep '^aries'
ls -l /dev/aries*
modinfo -n aries
command -v mobilint-cli
mobilint-cli status
```

실제 benchmark Python 환경에서도 SDK와 Model Zoo import를 확인한다.

```bash
"$PY" - <<'PY'
import qbruntime
import mbltml

print("qbruntime:", qbruntime.__version__, qbruntime.__file__)
print("mbltml:", mbltml.__file__)

try:
    import mblt_model_zoo
    import transformers
except ImportError as exc:
    print("LLM optional dependency unavailable:", exc)
else:
    print("mblt_model_zoo:", mblt_model_zoo.__file__)
    print("transformers:", transformers.__version__)
PY
```

BERT/PatchTST raw MXQ에는 `qbruntime`과 `mbltml`이 필요하다. Llama에는 벤더가
배포한 Transformers 지원 Model Zoo package도 필요하다. SDK package는 일반 PyPI
package로 임의 대체하지 말고 현재 driver/runtime release에 맞는 Mobilint 설치
문서를 따른다. 기존 설치 이력과 재부팅이 어려운 환경의 점검 방법은
[ARIES 트러블슈팅 기록](mobilint-aries-troubleshooting.md)을 참고한다.

데이터 준비와 Llama 다운로드에 필요한 Python package는 uv 환경에 설치한다.

```bash
uv pip install --python "$PY" huggingface-hub datasets transformers
```

## 2. 배치 크기의 의미

`--batch-size`는 한 번의 framework runtime 호출에 실제로 묶는 sample 수다.
Mobilint Llama repository 이름의 `Batch16`과 `Batch32`는 반드시 16개 또는 32개를
넣으라는 뜻이 아니라 컴파일된 최대 용량이다.

| artifact | `config.json` 용량 | 허용하는 실제 `--batch-size` |
|---|---:|---:|
| standard | 1 | 1 |
| Batch16 | 16 | 1~16 |
| Batch32 | 32 | 1~32 |

예를 들어 Batch16 artifact에 `--batch-size 4`를 사용하는 것은 정상이다. 반대로
standard artifact에 `--batch-size 2`, Batch16에 `--batch-size 32`를 지정하면
runtime이 실행 전에 거부한다. Batch16/32 실행은 prompt를 먼저 한 그룹으로 만든 뒤
한 번의 blocking `generate()`로 처리하는 grouped generation이며 continuous batching은
아니다.

BERT와 PatchTST의 권장 최초 검증값은 `--batch-size 1`이다. 더 큰 실제 batch는 해당
MXQ가 그 batch를 받도록 컴파일됐음을 확인한 경우에만 사용한다.

## 3. BERT·PatchTST MXQ 계약과 검사

MXQ는 다음 순서와 shape/dtype으로 컴파일돼야 한다. qb Runtime v1.3은 portable한
tensor-name metadata를 제공하지 않으므로 input/output 순서는 컴파일 때 고정한 순서와
아래 ModelSpec 순서가 같아야 한다.

| 모델 | 입력 순서 | 입력 dtype 및 sample shape | 출력 순서 및 sample shape |
|---|---|---|---|
| BERT SST-2 | `input_ids`, `attention_mask` | `int64 (128)`, `int64 (128)` | `logits (2)` |
| BERT SQuAD | `input_ids`, `attention_mask`, `token_type_ids` | 각각 `int64 (384)` | `start_logits (384)`, `end_logits (384)` |
| PatchTST ETTh1 | `past_values`, `past_observed_mask` | `float32 (512,7)`, `bool (512,7)` | 첫 출력 `(96,7)` |

권장 artifact 위치를 만든 뒤 외부에서 받은 MXQ를 배치한다.

```bash
mkdir -p \
  "$FW/models/mobilint/bert-base-uncased/aries" \
  "$FW/models/mobilint/bert-base-uncased-squad-v1/aries" \
  "$FW/models/mobilint/patchtst-etth1/aries"

BERT_SST2_MXQ="$FW/models/mobilint/bert-base-uncased/aries/bert_sst2.mxq"
BERT_QA_MXQ="$FW/models/mobilint/bert-base-uncased-squad-v1/aries/squad.mxq"
PATCHTST_MXQ="$FW/models/mobilint/patchtst-etth1/aries/patchtst_etth1.mxq"
```

NPU launch 없이 MXQ metadata를 읽어 계약을 먼저 확인한다. 이 도구도 model object는
항상 dispose한다.

```bash
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$BERT_SST2_MXQ" --core-mode global8
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$BERT_QA_MXQ" --core-mode global8
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$PATCHTST_MXQ" --core-mode global8
```

shape 표시에 leading `1`이 추가될 수 있다. 예를 들어 `(128)`과 `(1,128)`은 같은
sample tensor의 SDK 표현 차이로 허용한다. 입력 개수, dtype, 나머지 shape 또는 출력
개수/shape가 다르면 benchmark를 시작하지 않는다. SQuAD의 두 output은 shape가 같아
metadata만으로 서로의 의미를 판별할 수 없으므로 컴파일 output 순서가 특히 중요하다.

## 4. 데이터셋 준비

### 4.1 SST-2 validation 전체

```bash
"$PY" "$FW/datasets/prepare_text_numpy.py" \
  --model-id textattack/bert-base-uncased-SST-2 \
  --dataset-name glue \
  --dataset-config sst2 \
  --split validation \
  --seq-len 128 \
  --output-dir "$FW/datasets/sst2_numpy"

SST2="$FW/datasets/sst2_numpy"
```

### 4.2 SQuAD v1 validation 전체

```bash
"$PY" "$FW/datasets/prepare_squad_numpy.py" \
  --model-id csarron/bert-base-uncased-squad-v1 \
  --dataset-name rajpurkar/squad \
  --split validation \
  --seq-len 384 \
  --output-dir "$FW/datasets/squad_numpy"

SQUAD_NUMPY="$FW/datasets/squad_numpy"
```

### 4.3 ETTh1

```bash
"$PY" "$FW/datasets/prepare_etth1.py" \
  --output-dir "$FW/datasets/etth1"

ETTH1="$FW/datasets/etth1/ETTh1.csv"
```

### 4.4 Llama 평가용 SQuAD v2

`prepare_squad2.py`는 현재 working directory 아래 `datasets/squad2`에 기록하므로
framework directory에서 실행한다.

```bash
(
  cd "$FW"
  "$PY" datasets/prepare_squad2.py
)

SQUAD2="$FW/datasets/squad2/val.json"
```

## 5. Static Transformer 실행

세 모델 모두 framework의 기존 loader, evaluator와 decoder를 그대로 사용한다.
동기 `e2e`에서 `--max-steps`를 생략하면 데이터셋 전체를 처리한다.

### 5.1 BERT SST-2

```bash
# 동기 E2E 전체
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased \
  --target mobilint-aries \
  --artifact "$BERT_SST2_MXQ" \
  --dataset "$SST2" \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor \
  --results-path "$FW/results/mobilint-aries-bert-sst2-e2e.csv"

# framework async queue 전체
SST2_COUNT=$("$PY" -c 'import numpy as n,sys; print(len(n.load(sys.argv[1], mmap_mode="r")))' "$SST2/labels.npy")
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased \
  --target mobilint-aries \
  --artifact "$BERT_SST2_MXQ" \
  --dataset "$SST2" \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --queue-capacity 16 --worker-count 1 \
  --min-samples "$SST2_COUNT" --max-samples "$SST2_COUNT" \
  --warmup 2 --flush-timeout-sec 600 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-bert-sst2-async.csv"
```

### 5.2 BERT SQuAD QA

```bash
# 동기 E2E 전체
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased-squad-v1 \
  --target mobilint-aries \
  --artifact "$BERT_QA_MXQ" \
  --dataset "$SQUAD_NUMPY" \
  --inference-mode e2e \
  --batch-size 1 --warmup 2 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor \
  --results-path "$FW/results/mobilint-aries-bert-squad-e2e.csv"

# framework async queue 전체
SQUAD_COUNT=$("$PY" -c 'import numpy as n,sys; print(len(n.load(sys.argv[1], mmap_mode="r")))' "$SQUAD_NUMPY/start_positions.npy")
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased-squad-v1 \
  --target mobilint-aries \
  --artifact "$BERT_QA_MXQ" \
  --dataset "$SQUAD_NUMPY" \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --queue-capacity 16 --worker-count 1 \
  --min-samples "$SQUAD_COUNT" --max-samples "$SQUAD_COUNT" \
  --warmup 2 --flush-timeout-sec 600 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-bert-squad-async.csv"
```

### 5.3 PatchTST ETTh1

```bash
# 동기 E2E 전체
"$PY" "$FW/src/main.py" \
  --model patchtst-etth1 \
  --target mobilint-aries \
  --artifact "$PATCHTST_MXQ" \
  --dataset "$ETTH1" \
  --inference-mode e2e \
  --batch-size 1 --warmup 2 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor \
  --results-path "$FW/results/mobilint-aries-patchtst-etth1-e2e.csv"

# framework async queue 전체: finite loader가 소진될 때까지 처리
"$PY" "$FW/src/main.py" \
  --model patchtst-etth1 \
  --target mobilint-aries \
  --artifact "$PATCHTST_MXQ" \
  --dataset "$ETTH1" \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --queue-capacity 16 --worker-count 1 \
  --min-samples 1 --warmup 2 --flush-timeout-sec 600 \
  --runtime-option core_mode=global8 \
  --no-compile --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-patchtst-etth1-async.csv"
```

qb Runtime v1.3 native `infer_async()`는 CNN, `N=1` 용도이며 LLM/RNN/LSTM과
CPU-offload 모델은 지원 대상이 아니다. 이 연동은 BERT와 PatchTST의
`async_queue`를 framework blocking executor로 실행한다. 따라서 위 명령은 SDK native
async라고 해석하면 안 되며, 한 모델 instance를 안전하게 소유하도록
`--worker-count 1`을 유지한다.

## 6. 공식 Mobilint Llama 다운로드

다음 script는 선택한 repository의 일부 파일만 추측하지 않고 snapshot 전체를 안정된
경로에 받는다.

```bash
# 먼저 standard artifact로 smoke test
"$PY" "$FW/models/prepare_mobilint_llm.py" \
  --model llama-3.1-8b --batch-capacity 1
"$PY" "$FW/models/prepare_mobilint_llm.py" \
  --model llama-3.2-3b --batch-capacity 1

LLAMA31="$FW/models/mobilint/llama-3.1-8b/standard"
LLAMA32="$FW/models/mobilint/llama-3.2-3b/standard"
```

필요한 grouped batch artifact만 추가로 받는다.

```bash
"$PY" "$FW/models/prepare_mobilint_llm.py" \
  --model llama-3.1-8b --batch-capacity 16
"$PY" "$FW/models/prepare_mobilint_llm.py" \
  --model llama-3.2-3b --batch-capacity 32

LLAMA31_B16="$FW/models/mobilint/llama-3.1-8b/batch16"
LLAMA32_B32="$FW/models/mobilint/llama-3.2-3b/batch32"
```

다운로드 후 용량을 확인한다.

```bash
for MODEL_DIR in "$LLAMA31" "$LLAMA32" "$LLAMA31_B16" "$LLAMA32_B32"; do
  test -d "$MODEL_DIR" || continue
  printf '%s: ' "$MODEL_DIR"
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_batch_size"])' \
    "$MODEL_DIR/config.json"
done
```

## 7. Llama 실행

Llama runtime은 Model Zoo `AutoModelForCausalLM.generate()`를 사용한다. 동기 E2E와
비동기 queue가 동일한 blocking generate를 호출하며, `async_queue`는 request
backpressure·회계·trace를 제공할 뿐 qb Runtime `infer_async()`를 호출하지 않는다.

먼저 각 모델 10개 sample을 batch 1로 확인한다.

```bash
for MODEL_AND_PATH in \
  "llama-3.1-8b:$LLAMA31" \
  "llama-3.2-3b:$LLAMA32"
do
  MODEL=${MODEL_AND_PATH%%:*}
  MODEL_PATH=${MODEL_AND_PATH#*:}
  "$PY" "$FW/src/main.py" \
    --model "$MODEL" \
    --target mobilint-aries-llm \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --dataset "$SQUAD2" \
    --inference-mode e2e \
    --batch-size 1 --max-new-tokens 64 \
    --warmup 1 --max-steps 10 \
    --monitor \
    --results-path "$FW/results/mobilint-aries-${MODEL}-e2e-smoke.csv"
done
```

전체 SQuAD v2 동기 실행은 위 명령에서 `--max-steps 10`만 제거한다. 생성량에 따라
매우 오래 걸릴 수 있으므로 먼저 10개, 100개, 전체 순으로 늘린다.

표준 artifact의 async queue smoke와 전체 실행 형식은 다음과 같다.

```bash
# 100-request smoke
"$PY" "$FW/src/main.py" \
  --model llama-3.2-3b \
  --target mobilint-aries-llm \
  --model-path "$LLAMA32" --tokenizer-path "$LLAMA32" \
  --dataset "$SQUAD2" \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --queue-capacity 16 --worker-count 1 \
  --min-samples 100 --max-samples 100 \
  --max-new-tokens 64 --warmup 1 --flush-timeout-sec 3600 \
  --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-llama-3.2-3b-async.csv"
```

전체 async 실행에서는 정확한 QA 수를 구해 두 sample 제한에 함께 사용한다.

```bash
SQUAD2_COUNT=$(jq '[.data[].paragraphs[].qas[]] | length' "$SQUAD2")

"$PY" "$FW/src/main.py" \
  --model llama-3.2-3b \
  --target mobilint-aries-llm \
  --model-path "$LLAMA32" --tokenizer-path "$LLAMA32" \
  --dataset "$SQUAD2" \
  --inference-mode async_queue --scenario offline \
  --batch-size 1 --queue-capacity 16 --worker-count 1 \
  --min-samples "$SQUAD2_COUNT" --max-samples "$SQUAD2_COUNT" \
  --max-new-tokens 64 --warmup 1 --flush-timeout-sec 86400 \
  --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-llama-3.2-3b-async-full.csv"
```

Llama 3.1 8B도 `--model llama-3.1-8b`, `--model-path "$LLAMA31"`로 동일하게
실행한다. 전체 생성 benchmark는 매우 오래 걸릴 수 있으므로 shell `timeout`을 쓸 때는
정상 완료 시간을 충분히 확보하고, 강제 종료 결과를 valid 성능 결과로 사용하지 않는다.

Batch16/32 artifact에서 실제 grouped batch를 확인할 때도 worker는 하나다.

```bash
# Batch16 capacity 중 실제 batch 4만 사용
"$PY" "$FW/src/main.py" \
  --model llama-3.1-8b \
  --target mobilint-aries-llm \
  --model-path "$LLAMA31_B16" --tokenizer-path "$LLAMA31_B16" \
  --dataset "$SQUAD2" \
  --inference-mode async_queue --scenario offline \
  --batch-size 4 --batch-timeout-ms 5 \
  --queue-capacity 16 --worker-count 1 \
  --min-samples 100 --max-samples 100 \
  --max-new-tokens 64 --warmup 1 --flush-timeout-sec 3600 \
  --monitor --save-request-trace \
  --results-path "$FW/results/mobilint-aries-llama-3.1-8b-b16-actual4.csv"
```

Grouped callback은 한 decoding step에서 여러 row의 token을 함께 전달할 수 있다.
그 경우 framework는 존재하지 않는 sample별 token timestamp를 만들지 않는다.
aggregate event를 개별 request trace row에도 복제하지 않으며, details에
`generation_timing_batch_ambiguous`와 `generation_stream_itl_incomplete` warning을
남기고 request별 TTFT/TPOT 및 token ITL percentile을 생략한다. Group 전체 scalar
timing은 request별 timing으로 해석하면 안 된다. 정확한 request TTFT/TPOT/ITL 검증은
`--batch-size 1`로 실행한다.

## 8. 결과 판정

각 실행에서 CSV와 `results/details/<RUN_ID>.json`을 함께 확인한다.

```bash
RUN_ID="<printed-run-id>"
jq '{invalid_reasons, counts, failure_types, warnings, timing_ms}' \
  "$FW/results/details/$RUN_ID.json"
```

최소 합격 조건은 다음과 같다.

- `async_run_status=valid`, `async_failed_requests=0`,
  `async_outstanding_requests=0`이다.
- completed sample 수가 요청한 전체 sample 수와 같다.
- BERT/PatchTST의 `mobilint_artifact_profile_id`가 선택한 model profile과 일치한다.
- 품질 지표를 CPU/원본 모델 baseline과 비교해 output 순서 또는 전처리 불일치를
  배제한다.
- `--monitor` 결과의 NPU utilization, memory, temperature, power sample coverage를
  확인한다. 전력/energy는 mbltml 측정값이며 framework는 표본을 적분한다.
- Llama batch 1 trace에서 TTFT/TPOT source가
  `mobilint_transformers_streamer`인지 확인한다.

이 저장소의 fake-SDK 테스트는 계약, cleanup, batch 경계와 async routing만 검증한다.
BERT/PatchTST MXQ의 실제 수치 정확성, Llama repository와 설치된 SDK의 호환성,
Batch16/32 메모리 사용량 및 ARIES 성능은 위 명령으로 실장비에서 확인해야 한다.

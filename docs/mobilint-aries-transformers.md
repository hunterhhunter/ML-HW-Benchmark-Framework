# Mobilint ARIES Transformer·LLM 실행 가이드

이 문서는 Mobilint ARIES에서 다음 모델을 실행하는 절차를 정리한다.

직접 컴파일한 artifact의 공통 attempt 기록, 엄격한 ARIES 1회 추론 검사와 결과 승격
순서는 [Mobilint qbcompiler 실험 실행서](mobilint-compilation-experiments.md)를 따른다.
아래 내용은 framework E2E와 Transformer·LLM 운용 절차에 집중한다.

- BERT SST-2 문장 분류
- BERT SQuAD v1 질의응답
- PatchTST ETTh1 시계열 예측
- Llama 3.1 8B Instruct
- Llama 3.2 3B Instruct

framework가 실행 중 MXQ를 컴파일하지는 않는다. BERT와 PatchTST에는 아래 계약으로
미리 컴파일한 `.mxq`가 필요하다. 특히 공개 base BERT MXQ는 masked-LM용이므로 SST-2나
SQuAD artifact 대신 사용할 수 없다. 이 문서의 BERT 경로는 qbcompiler 1.2로 별도
컴파일해 ARIES에서 검증한 task-specific MXQ와 embedding weight를 사용한다. Llama는
Mobilint 공식 Model Zoo Hugging Face repository의 전체 snapshot을 사용한다.
BERT 두 artifact를 qbcompiler 1.2로 다시 만드는 절차는
[Mobilint BERT 컴파일 재현 가이드](mobilint-bert-compilation.md)에 별도로 정리했다.
컴파일 호스트에는 ARIES 장치나 qb Runtime이 필요하지 않다.

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
import torch

print("qbruntime:", qbruntime.__version__, qbruntime.__file__)
print("mbltml:", mbltml.__file__)
print("torch:", torch.__version__)

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

BERT embedding 생성에는 PyTorch도 필요하고, BERT/PatchTST MXQ 실행에는
`qbruntime`과 `mbltml`이 필요하다. Llama에는 벤더가 배포한 Transformers 지원 Model
Zoo package도 필요하다. SDK package는 일반 PyPI package로 임의 대체하지 말고 현재
driver/runtime release에 맞는 Mobilint 설치 문서를 따른다. 기존 설치 이력과 재부팅이
어려운 환경의 점검 방법은
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

이 문서의 두 BERT embedding MXQ는 `--batch-size 1`만 지원한다. PatchTST의 권장 최초
검증값도 1이며, 더 큰 실제 batch는 해당 MXQ가 그 batch를 받도록 컴파일됐음을 확인한
경우에만 사용한다.

## 3. BERT·PatchTST MXQ 계약과 검사

BERT SST-2/SQuAD v1의 모델·calibration 준비, exact compiler option과 산출물 구조는
[BERT 컴파일 재현 가이드](mobilint-bert-compilation.md)를 따른다. 이 절의 명령은
컴파일이 끝난 artifact를 ARIES에서 검사하고 benchmark에 연결하는 단계다.

MXQ는 다음 순서와 shape/dtype으로 컴파일돼야 한다. qb Runtime v1.3은 portable한
tensor-name metadata를 제공하지 않으므로 input/output 순서는 컴파일 때 고정한 순서와
아래 ModelSpec 순서가 같아야 한다.

| 모델 | 입력 순서 | 입력 dtype 및 sample shape | 출력 순서 및 sample shape |
|---|---|---|---|
| BERT SST-2 | `embeddings` | `float32 (L,768)` | `logits (2)` |
| BERT SQuAD | `embeddings` | `float32 (L,768)` | `end_logits (L)`, `start_logits (L)` |
| PatchTST ETTh1 | `past_values`, `past_observed_mask` | `float32 (512,7)`, `bool (512,7)` | 첫 출력 `(96,7)` |

BERT의 `input_ids`, `attention_mask`, 선택적 `token_type_ids`는 여전히 기존 numpy
loader가 읽는다. loader에 주입된 host transform이 유효 토큰 prefix만 남기고
`word + token_type + position embedding` 및 LayerNorm을 계산한 뒤 MXQ에는
`embeddings` 하나만 전달한다. SQuAD 출력 순서는 ARIES 실측 결과인
`end_logits`, `start_logits`이며 Hugging Face 모델의 속성 나열 순서와 반대다.

컴파일 작업 디렉터리와 artifact를 지정한다. 날짜가 다른 디렉터리를 사용했다면
`WORK`만 수정한다.

```bash
WORK="$REPO/.mobilint-bert-tasks-20260730-105143"
BERT_SST2_MXQ="$WORK/artifacts/sst2/mxq/sst2.mxq"
BERT_SST2_WEIGHTS="$WORK/artifacts/sst2/weights/weight_dict.pth"
BERT_QA_MXQ="$WORK/artifacts/squad1/mxq/squad1.mxq"
BERT_QA_WEIGHTS="$WORK/artifacts/squad1/weights/weight_dict.pth"
PATCHTST_MXQ="$FW/models/mobilint/patchtst-etth1/aries/patchtst_etth1.mxq"

test -s "$BERT_SST2_MXQ"
test -s "$BERT_SST2_WEIGHTS"
test -s "$BERT_QA_MXQ"
test -s "$BERT_QA_WEIGHTS"
sha256sum "$BERT_SST2_MXQ" "$BERT_SST2_WEIGHTS" \
  "$BERT_QA_MXQ" "$BERT_QA_WEIGHTS"
```

NPU launch 없이 MXQ metadata를 읽어 계약을 먼저 확인한다. 이 도구도 model object는
항상 dispose한다.

```bash
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$BERT_SST2_MXQ" --core-mode single
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$BERT_QA_MXQ" --core-mode single
"$PY" "$FW/tools/inspect_mobilint_mxq.py" "$PATCHTST_MXQ" --core-mode global8
```

BERT inspect 결과는 입력 `Float32 [1,-1,768]` 하나여야 한다. SST-2 출력은
`[1,1,2]`, SQuAD 출력 두 개는 각각 `[1,-1,1]`처럼 singleton 차원을 포함할 수 있다.
runtime은 이를 evaluator용 `[1,2]`, `[1,L]`로 정규화한다. 입력 개수, dtype, embedding
width, 출력 개수 또는 논리 shape가 다르면 benchmark를 시작하지 않는다. SQuAD의 두
output은 shape가 같아 metadata만으로 의미를 판별할 수 없으므로 컴파일 output 순서가
특히 중요하다.

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

세 모델 모두 framework의 기존 loader와 evaluator를 그대로 사용한다. 다음 BERT
명령은 먼저 64 sample만 검증한다. `--max-steps`를 생략하면 데이터셋 전체를 처리한다.

### 5.1 BERT SST-2

```bash
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased \
  --target mobilint-aries \
  --artifact "$BERT_SST2_MXQ" \
  --mobilint-bert-weights "$BERT_SST2_WEIGHTS" \
  --dataset "$SST2" \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-steps 64 \
  --runtime-option core_mode=single \
  --no-compile --monitor \
  --results-path "$FW/results/mobilint-aries-bert-sst2-e2e-64.csv"
```

이전에 같은 artifact를 직접 검증했을 때 첫 64개 중 59개가 정답이었다. framework의
`accuracy`는 백분율이므로 같은 데이터 순서라면 약 `92.1875`가 기준이다.

### 5.2 BERT SQuAD QA

```bash
"$PY" "$FW/src/main.py" \
  --model bert-base-uncased-squad-v1 \
  --target mobilint-aries \
  --artifact "$BERT_QA_MXQ" \
  --mobilint-bert-weights "$BERT_QA_WEIGHTS" \
  --dataset "$SQUAD_NUMPY" \
  --inference-mode e2e \
  --batch-size 1 --warmup 2 \
  --max-steps 64 \
  --runtime-option core_mode=single \
  --no-compile --monitor \
  --results-path "$FW/results/mobilint-aries-bert-squad-e2e-64.csv"
```

framework QA evaluator의 `exact_match`와 `f1`은 저장된 start/end token 좌표를 직접
비교한다. 별도 검증 script가 계산했던 정규화 문자열 EM/F1과 같은 지표가 아니므로
숫자를 그대로 비교하지 않는다. 두 지표가 퇴화하지 않는지와 출력 순서가
`end_logits,start_logits`인지 함께 확인한다.

두 BERT 명령의 `Average Latency (ms)`, `P99 Latency (ms)`, `Samples/s`는
`MobilintRuntime.run()` 구간을 측정한다. 여기에는 입력 계약 검사, qb Runtime inference,
singleton output 정규화와 이름 결합이 포함되지만 loader가 먼저 수행하는 CPU
token-to-embedding 계산은 포함되지 않는다. 전체 command wall time에는 데이터 로드와
embedding 준비 시간도 당연히 포함된다.

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
CPU-offload 모델은 지원 대상이 아니다. 이 BERT profile은
`native_async_supported=False`이므로 처음에는 위 `e2e` 명령으로 인수한다. 나중에
framework `async_queue`를 선택해도 SDK native async가 아니라 blocking executor로
실행된다.

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
- SQuAD CSV의 `mobilint_output_order`와 async details의
  `runtime_device_spec.expected_output_names`가 모두 `end_logits,start_logits` 순서다.
- 품질 지표를 CPU/원본 모델 baseline과 비교해 output 순서 또는 전처리 불일치를
  배제한다.
- `--monitor` 결과의 NPU utilization, memory, temperature, power sample coverage를
  확인한다. 전력/energy는 mbltml 측정값이며 framework는 표본을 적분한다.
- Llama batch 1 trace에서 TTFT/TPOT source가
  `mobilint_transformers_streamer`인지 확인한다.

이 저장소의 fake-SDK 테스트는 계약, cleanup, batch 경계와 async routing만 검증한다.
BERT/PatchTST MXQ의 실제 수치 정확성, Llama repository와 설치된 SDK의 호환성,
Batch16/32 메모리 사용량 및 ARIES 성능은 위 명령으로 실장비에서 확인해야 한다.

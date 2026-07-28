# RNGD 논문용 생성 지연 벤치마크 프로토콜

## OpenAI-compatible server 프로토콜

이 문서는 Furiosa RNGD의 생성 지연을 논문에 보고할 때 사용하는 고정 프로토콜입니다. 대상은 Furiosa-LLM 2026.3.0의 OpenAI-compatible server와 `vllm bench serve` 0.16.0이며, 임베디드 `AsyncLLMEngine` 결과와 HTTP serving 결과를 같은 실험으로 섞지 않습니다.

공식 API 근거는 다음 두 문서로 고정합니다.

- [Furiosa OpenAI-Compatible Server 2026.3.0](https://developer.furiosa.ai/latest/en/furiosa_llm/furiosa-llm-serve.html)
- [vLLM 0.16.0 serve benchmark](https://docs.vllm.ai/en/v0.16.0/api/vllm/benchmarks/serve/)

### 측정 경계

서버 벤치마크는 서로 다른 두 관측 계층을 동시에 저장합니다.

| 네임스페이스 | 관측 위치 | 지연 정의 | 백분위 계산 |
| --- | --- | --- | --- |
| `server_client_*` | 부하 생성기에서 받은 streaming 응답 | TTFT, 요청별 평균 TPOT, 개별 streaming ITL, E2EL | 상세 표본에서 `numpy.percentile(method="linear")`로 재계산 |
| `server_vendor_*` | Furiosa server의 `/metrics` | 서버가 계측한 TTFT, ITL, E2EL | 전후 Prometheus histogram delta의 bucket에서 선형 보간 |

`server_client_itl_p*`가 토큰 사이의 클라이언트 관측 지연 분포입니다. `server_client_tpot_p*`는 요청별 평균이므로 토큰별 분포가 아닙니다. 두 계층은 네트워크·HTTP serialization·stream buffering과 histogram bucket 오차 때문에 값이 다를 수 있습니다. 논문 표에서도 두 값을 별도 열로 유지하며 평균하거나 합치지 않습니다.

모든 분포는 p50, p85, p90, p95, p99를 저장합니다. 단위는 최종 결과에서 밀리초입니다.

### 고정 환경

서버와 부하 생성기는 가능하면 별도 호스트를 사용합니다. 같은 호스트를 쓰면 CPU affinity, NIC, NUMA 배치와 백그라운드 프로세스를 모든 실행에서 동일하게 고정하고 그 사실을 논문에 밝힙니다.

Furiosa 서버 환경은 [RNGD runtime 설정 문서](furiosa-rngd-setup.md)의 `.venv-rngd`를 사용합니다. 부하 생성기는 Furiosa server 프로세스와 분리된 vLLM 0.16.0 전용 환경을 사용합니다.

```bash
cd framework
uv venv .venv-vllm --python 3.12
uv pip install --python .venv-vllm/bin/python vllm==0.16.0
.venv-vllm/bin/vllm --version
```

측정 전에 다음 증거를 실행별 보관 디렉터리에 남깁니다.

```bash
python --version
python -m pip show furiosa-llm
furiosa-smi info
fxb show /absolute/path/to/model.fxb
sha256sum /absolute/path/to/model.fxb
.venv-vllm/bin/vllm --version
.venv-vllm/bin/python -m pip freeze
```

최소 기록 항목은 호스트 CPU·메모리·커널, RNGD 수와 topology, driver/Furiosa-LLM/FXB/vLLM 버전, 모델과 tokenizer revision, FXB SHA-256, server 실행 인자입니다. 전력 모드, clock 정책, 냉각 조건도 고정합니다.

### 서버 준비

측정 전용 서버를 시작합니다. 아래의 artifact 경로와 server option은 실험 전체에서 고정하고 실제 명령을 로그로 남깁니다.

```bash
source framework/.venv-rngd/bin/activate
furiosa-llm serve /absolute/path/to/model.fxb --host 0.0.0.0 --port 8000
```

공식 문서에 따라 identity와 metrics endpoint를 확인합니다.

```bash
curl --fail --silent --show-error http://SERVER:8000/version
curl --fail --silent --show-error http://SERVER:8000/v1/models
curl --fail --silent --show-error http://SERVER:8000/metrics
```

`/v1/models`가 반환한 정확한 model ID를 도구의 `--model`에 사용합니다. `/metrics`에서 대상 series의 정확한 `model_name`, `engine` label을 확인해 각각 `--metrics-label`로 전달합니다. 서버는 측정 동안 다른 요청을 받지 않아야 합니다. 그렇지 않으면 전후 histogram delta가 벤치마크 요청만을 나타내지 않습니다.

기본 논문 결과는 steady-state 측정입니다. 최대 input/output 길이의 비기록 warmup을 먼저 수행하고 JIT·메모리 할당이 안정된 뒤 시작합니다. prefix cache와 compilation cache 정책은 고정해 기록합니다. 실험 셀 사이의 cache 영향을 제거해야 한다면 서버 재시작과 동일 warmup을 매 반복마다 적용합니다.

### 단일 실행

framework 환경에는 `httpx`, `numpy`, `prometheus-client`와 결과 저장 의존성이 있어야 합니다. `vllm` 실행 파일만 전용 부하 생성기 환경에서 가져옵니다.

```bash
cd framework
PATH="$PWD/.venv-vllm/bin:$PATH" \
PYTHONPATH=src .venv/bin/python tools/rngd_server_benchmark.py \
  --base-url http://SERVER:8000 \
  --model MODEL_ID_FROM_V1_MODELS \
  --input-tokens 512 \
  --output-tokens 128 \
  --num-prompts 1000 \
  --max-concurrency 16 \
  --request-rate inf \
  --seed 20260721 \
  --metrics-label model_name=MODEL_NAME_FROM_METRICS \
  --metrics-label engine=ENGINE_FROM_METRICS \
  --result-dir results/rngd-server/raw \
  --results-path results/rngd-server/benchmark_results.csv
```

도구는 shell을 거치지 않는 고정 argv로 `vllm bench serve`를 호출합니다. endpoint는 `/v1/completions`, temperature는 0, EOS는 무시하며 요청 output 길이를 고정합니다. 오케스트레이터가 readiness를 직접 확인하므로 vLLM의 별도 probe 요청은 `--ready-check-timeout-sec 0`으로 끕니다. `--request-rate inf`는 offline-like 공급이고 숫자 QPS는 server-like 도착률입니다.

실행 순서는 다음과 같습니다.

1. `/version`, `/v1/models`, `/metrics` 전 스냅샷을 수집합니다.
2. vLLM 0.16.0 detailed result를 생성합니다.
3. `/metrics`, `/version`, `/v1/models` 후 스냅샷을 수집합니다.
4. 클라이언트 raw 표본과 vendor histogram delta를 별도로 정규화합니다.
5. 교차 검증을 통과한 경우에만 CSV row를 `valid`로 저장합니다.

raw vLLM JSON과 전후 Prometheus text는 `--result-dir`에 남고, 각 SHA-256과 정규화 근거는 results sidecar에 남습니다. CSV와 sidecar뿐 아니라 `--result-dir` 전체도 함께 보관해야 합니다.

### 실험 행렬

논문에 사용할 기본 행렬은 다음과 같습니다.

- input tokens: 128, 512, 2048
- output tokens: 32, 128, 256
- max concurrency: 1, 4, 16, 32
- arrival: `inf`와 사전에 고정한 finite QPS 집합
- independent repetitions: 셀마다 5회 이상
- request count: 반복마다 1,000개 이상

finite QPS 집합은 결과를 본 뒤 바꾸지 않습니다. 별도 pilot에서 해당 모델·길이의 포화 throughput을 찾고, 예를 들어 그 값의 25%, 50%, 75%, 90%를 절대 QPS로 고정합니다. pilot 결과는 본 측정에서 제외하고 선택 절차와 최종 QPS를 공개합니다.

행렬 순서는 seed로 무작위화하고 순서와 seed를 보관합니다. 모델·FXB·server option이 다른 결과는 같은 모집단으로 합치지 않습니다. Llama 3.1 8B와 Llama 3.2 3B는 각각 독립된 표와 반복 집합을 사용합니다.

### 유효성 판정

다음 조건 중 하나라도 발생하면 row는 `invalid`이며 논문 집계에 포함하지 않습니다.

- vLLM 명령, HTTP endpoint, parsing 또는 artifact 저장 실패
- 클라이언트 실패 요청이 1개 이상 존재
- 상세 배열의 요청 수·성공 수·output token 수가 aggregate와 불일치
- vLLM aggregate percentile과 raw 표본 재계산 값이 불일치
- raw result의 backend, model ID, `num_prompts`가 실행 계약과 불일치
- 측정 전후 server version 또는 `/v1/models`가 변경
- vendor 성공 요청 수나 generation token 수가 클라이언트 결과와 불일치
- vendor ITL observation 수가 성공 요청의 `sum(output_len - 1)`과 불일치
- Prometheus counter reset, bucket schema 변경, label scope 모호성

실패 row와 실패 sidecar는 삭제하지 않습니다. exclusion 사유별 개수를 결과와 함께 보고하고, 설정을 바꿔 재실행한 경우 새 run ID로 보존합니다.

### 논문 보고

각 셀마다 최소한 다음을 보고합니다.

- 성공/실패 요청 수, output tokens/s
- client TTFT/TPOT/ITL/E2EL p50·p85·p90·p95·p99
- vendor TTFT/ITL/E2EL p50·p85·p90·p95·p99
- 반복별 값과 반복 간 변동성
- 유효 반복 수, 제외 반복 수와 사유

client percentile은 raw sample 기반이지만 vendor percentile은 Prometheus bucket 보간값입니다. vendor p99는 bucket 폭보다 정밀하게 해석하지 않습니다. p99 신뢰구간을 제시할 때는 client raw 표본에 대한 non-parametric bootstrap과 반복 간 변동을 함께 공개하고 bootstrap seed도 기록합니다. 서로 다른 반복의 raw 요청을 단순히 합쳐 하나의 큰 표본처럼 보고하지 않습니다.

실장비 완료 gate는 두 대상 모델의 전체 행렬 실행, raw artifact 보존 확인, invalid 사유 검토입니다. fake SDK와 단위 테스트 통과는 코드 완료 조건일 뿐 하드웨어 결과의 유효성을 대신하지 않습니다.

## 임베디드 네이티브 `async_queue` 프로토콜

이 문서는 Furiosa RNGD에서 얻은 생성 지연 결과를 논문에 사용할 때의 실행·검증 기준을 고정한다. 집계 CSV만으로는 결과를 채택하지 않는다. 반드시 async details JSON과 request trace JSONL을 함께 보존하고, raw monotonic timestamp에서 통계를 다시 계산할 수 있어야 한다.

### 측정값 정의

- Request TTFT: framework가 요청을 발행한 `issued_ns`부터 Furiosa Python async stream에서 처음으로 non-empty output을 관측한 시각까지다. 스케줄러·queue 대기와 backend submit 대기를 포함한다.
- Backend TTFT: `backend_submitted_ns`부터 첫 non-empty output 관측까지다.
- Request mean TPOT: `(마지막 output 관측 시각 - 첫 output 관측 시각) / (생성 토큰 수 - 1)`이다. 생성 토큰이 하나 이하이면 기록하지 않는다.
- Stream-event ITL: 인접한 single-token Python stream event 사이의 시간이다. 모든 event의 누적 토큰 수가 정확히 1씩 증가하고 마지막 누적값이 최종 생성 토큰 수와 같은 요청만 포함한다.

Furiosa stream event 하나에서 누적 토큰 수가 둘 이상 증가했다면 그 사이 토큰의 실제 도착 시각은 관측되지 않은 것이다. 이 요청은 request TTFT와 request mean TPOT에는 포함하지만 stream-event ITL에는 일부 구간도 넣지 않는다. `stream_event_itl_coverage`가 1.0이 아니면 ITL percentile을 논문 표에 싣지 않는다.

모든 percentile은 `numpy.percentile(method="linear")`로 계산한다. p50, p85, p90, p95, p99와 표본 수를 함께 보고한다.

### 실험 행렬

아래 축의 조합을 서로 다른 workload cell로 취급한다. cell 사이의 요청을 합쳐 percentile을 계산하지 않는다.

- 모델 및 정확한 FXB artifact
- input token length bucket
- output token length bucket 또는 고정 `max_new_tokens`
- offline / server-like scenario
- server-like target QPS
- framework worker count와 queue capacity
- RNGD device 및 tensor/data parallel topology

Server-like 실험은 낮은 부하부터 포화점을 넘는 부하까지 여러 target QPS로 실행한다. 각 QPS에서 achieved throughput, request TTFT, request mean TPOT, 오류·timeout 비율을 함께 기록한다. Offline 결과와 server-like 결과를 같은 분포로 합치지 않는다.

### 반복 및 채택 기준

1. runtime load 후 논문 측정에서 제외되는 warmup을 수행한다.
2. workload cell마다 successful request 1,000개 이상을 수집한다.
3. 동일 cell을 독립적으로 5회 이상 실행한다.
4. 대표값은 run별 지표의 median으로 보고하고, run-to-run 범위 또는 IQR도 함께 보고한다.
5. 한 run 안의 요청 표본을 독립 run처럼 취급하지 않는다.
6. `--save-request-trace`를 항상 사용한다.
7. trace drop, request failure, timeout, counter/timing invariant 실패가 하나라도 있으면 해당 run을 폐기한다.
8. stream-event ITL은 coverage 100%인 run만 채택한다. coverage가 낮은 run에서도 request TTFT와 request mean TPOT은 별도로 사용할 수 있지만, 관측 한계를 명시한다.

### 실행 전 기록

다음을 실험 노트와 결과 artifact에 보존한다.

- framework Git commit과 dirty 여부
- Python 및 `furiosa-llm` 버전
- driver/firmware 버전과 `furiosa-smi info`
- NPU 수, device mapping, data/pipeline parallel 설정
- HF model 경로·revision과 tokenizer revision
- FXB 파일 SHA-256 및 `fxb show` 결과
- 필요하면 `fxb check` 결과
- OS, CPU, RAM 및 전원·클럭 정책
- input/output token policy
- temperature, EOS/stop policy, `max_new_tokens`, seed
- scenario, target QPS, worker count, queue capacity

예시 사전 점검:

```bash
furiosa-smi info
fxb show /absolute/path/model.fxb
fxb check /absolute/path/model.fxb
sha256sum /absolute/path/model.fxb
git rev-parse HEAD
git status --short
```

### 실행 예시

Offline 예시:

```bash
cd framework
PYTHONPATH=src .venv/bin/python src/main.py \
  --model llama-3.2-3b \
  --target furiosa-rngd \
  --backend furiosa_llm \
  --model-path /absolute/path/hf-model \
  --fxb /absolute/path/model.fxb \
  --dataset /absolute/path/dataset \
  --inference-mode async_queue \
  --scenario offline \
  --worker-count 16 \
  --queue-capacity 64 \
  --min-samples 1000 \
  --max-new-tokens 128 \
  --schedule-seed 1 \
  --save-request-trace
```

Server-like 예시는 `--scenario server_like --target-qps <QPS>`를 추가한다. QPS만 바꾸는 경우에도 각각 독립 run으로 저장한다.

### 사후 검증

각 run에 대해 다음을 확인한다.

- CSV, details JSON, trace JSONL의 run ID가 같다.
- `async_outstanding_requests == 0`이다.
- `invalid_reasons`가 비어 있다.
- `request_trace_dropped:*` warning이 없다.
- trace의 successful generation row 수가 집계의 `async_generation_observed_requests`와 같다.
- raw event에서 다시 계산한 TTFT/TPOT/ITL percentile이 details JSON과 일치한다.
- `generation_stream_itl_incomplete` warning이 있으면 stream-event ITL percentile을 사용하지 않는다.
- SDK version이 `null`이거나 Git dirty 여부를 알 수 없는 run은 환경을 보완해 다시 실행한다.

### 논문 표기

표와 본문에는 하드웨어명만 쓰지 말고 SDK/driver/FXB hash, 모델, token-length 조건, scenario, offered QPS, achieved throughput, 표본 수, 반복 횟수를 함께 적는다. Python stream에서 관측한 시각은 NPU 내부 cycle-level timestamp가 아니므로 지표 이름을 `Python stream-observed ITL`처럼 명시한다. 내부 토큰 실행 시간을 직접 측정한 값으로 표현하지 않는다.

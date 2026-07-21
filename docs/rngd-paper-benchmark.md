# RNGD 논문용 생성 지연 벤치마크 프로토콜

이 문서는 Furiosa RNGD의 생성 지연을 논문에 보고할 때 사용하는 고정 프로토콜입니다. 대상은 Furiosa-LLM 2026.3.0의 OpenAI-compatible server와 `vllm bench serve` 0.16.0이며, 임베디드 `AsyncLLMEngine` 결과와 HTTP serving 결과를 같은 실험으로 섞지 않습니다.

공식 API 근거는 다음 두 문서로 고정합니다.

- [Furiosa OpenAI-Compatible Server 2026.3.0](https://developer.furiosa.ai/latest/en/furiosa_llm/furiosa-llm-serve.html)
- [vLLM 0.16.0 serve benchmark](https://docs.vllm.ai/en/v0.16.0/api/vllm/benchmarks/serve/)

## 측정 경계

서버 벤치마크는 서로 다른 두 관측 계층을 동시에 저장합니다.

| 네임스페이스 | 관측 위치 | 지연 정의 | 백분위 계산 |
| --- | --- | --- | --- |
| `server_client_*` | 부하 생성기에서 받은 streaming 응답 | TTFT, 요청별 평균 TPOT, 개별 streaming ITL, E2EL | 상세 표본에서 `numpy.percentile(method="linear")`로 재계산 |
| `server_vendor_*` | Furiosa server의 `/metrics` | 서버가 계측한 TTFT, ITL, E2EL | 전후 Prometheus histogram delta의 bucket에서 선형 보간 |

`server_client_itl_p*`가 토큰 사이의 클라이언트 관측 지연 분포입니다. `server_client_tpot_p*`는 요청별 평균이므로 토큰별 분포가 아닙니다. 두 계층은 네트워크·HTTP serialization·stream buffering과 histogram bucket 오차 때문에 값이 다를 수 있습니다. 논문 표에서도 두 값을 별도 열로 유지하며 평균하거나 합치지 않습니다.

모든 분포는 p50, p85, p90, p95, p99를 저장합니다. 단위는 최종 결과에서 밀리초입니다.

## 고정 환경

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

## 서버 준비

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

## 단일 실행

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

도구는 shell을 거치지 않는 고정 argv로 `vllm bench serve`를 호출합니다. endpoint는 `/v1/completions`, temperature는 0, EOS는 무시하며 요청 output 길이를 고정합니다. `--request-rate inf`는 offline-like 공급이고 숫자 QPS는 server-like 도착률입니다.

실행 순서는 다음과 같습니다.

1. `/version`, `/v1/models`, `/metrics` 전 스냅샷을 수집합니다.
2. vLLM 0.16.0 detailed result를 생성합니다.
3. `/metrics`, `/version`, `/v1/models` 후 스냅샷을 수집합니다.
4. 클라이언트 raw 표본과 vendor histogram delta를 별도로 정규화합니다.
5. 교차 검증을 통과한 경우에만 CSV row를 `valid`로 저장합니다.

raw vLLM JSON과 전후 Prometheus text는 `--result-dir`에 남고, 각 SHA-256과 정규화 근거는 results sidecar에 남습니다. CSV와 sidecar뿐 아니라 `--result-dir` 전체도 함께 보관해야 합니다.

## 실험 행렬

논문에 사용할 기본 행렬은 다음과 같습니다.

- input tokens: 128, 512, 2048
- output tokens: 32, 128, 256
- max concurrency: 1, 4, 16, 32
- arrival: `inf`와 사전에 고정한 finite QPS 집합
- independent repetitions: 셀마다 5회 이상
- request count: 반복마다 1,000개 이상

finite QPS 집합은 결과를 본 뒤 바꾸지 않습니다. 별도 pilot에서 해당 모델·길이의 포화 throughput을 찾고, 예를 들어 그 값의 25%, 50%, 75%, 90%를 절대 QPS로 고정합니다. pilot 결과는 본 측정에서 제외하고 선택 절차와 최종 QPS를 공개합니다.

행렬 순서는 seed로 무작위화하고 순서와 seed를 보관합니다. 모델·FXB·server option이 다른 결과는 같은 모집단으로 합치지 않습니다. Llama 3.1 8B와 Llama 3.2 3B는 각각 독립된 표와 반복 집합을 사용합니다.

## 유효성 판정

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

## 논문 보고

각 셀마다 최소한 다음을 보고합니다.

- 성공/실패 요청 수, output tokens/s
- client TTFT/TPOT/ITL/E2EL p50·p85·p90·p95·p99
- vendor TTFT/ITL/E2EL p50·p85·p90·p95·p99
- 반복별 값과 반복 간 변동성
- 유효 반복 수, 제외 반복 수와 사유

client percentile은 raw sample 기반이지만 vendor percentile은 Prometheus bucket 보간값입니다. vendor p99는 bucket 폭보다 정밀하게 해석하지 않습니다. p99 신뢰구간을 제시할 때는 client raw 표본에 대한 non-parametric bootstrap과 반복 간 변동을 함께 공개하고 bootstrap seed도 기록합니다. 서로 다른 반복의 raw 요청을 단순히 합쳐 하나의 큰 표본처럼 보고하지 않습니다.

실장비 완료 gate는 두 대상 모델의 전체 행렬 실행, raw artifact 보존 확인, invalid 사유 검토입니다. fake SDK와 단위 테스트 통과는 코드 완료 조건일 뿐 하드웨어 결과의 유효성을 대신하지 않습니다.

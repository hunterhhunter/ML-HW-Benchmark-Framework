# RNGD 논문용 생성 지연 벤치마크 프로토콜

이 문서는 Furiosa RNGD에서 얻은 생성 지연 결과를 논문에 사용할 때의 실행·검증 기준을 고정한다. 집계 CSV만으로는 결과를 채택하지 않는다. 반드시 async details JSON과 request trace JSONL을 함께 보존하고, raw monotonic timestamp에서 통계를 다시 계산할 수 있어야 한다.

## 측정값 정의

- Request TTFT: framework가 요청을 발행한 `issued_ns`부터 Furiosa Python async stream에서 처음으로 non-empty output을 관측한 시각까지다. 스케줄러·queue 대기와 backend submit 대기를 포함한다.
- Backend TTFT: `backend_submitted_ns`부터 첫 non-empty output 관측까지다.
- Request mean TPOT: `(마지막 output 관측 시각 - 첫 output 관측 시각) / (생성 토큰 수 - 1)`이다. 생성 토큰이 하나 이하이면 기록하지 않는다.
- Stream-event ITL: 인접한 single-token Python stream event 사이의 시간이다. 모든 event의 누적 토큰 수가 정확히 1씩 증가하고 마지막 누적값이 최종 생성 토큰 수와 같은 요청만 포함한다.

Furiosa stream event 하나에서 누적 토큰 수가 둘 이상 증가했다면 그 사이 토큰의 실제 도착 시각은 관측되지 않은 것이다. 이 요청은 request TTFT와 request mean TPOT에는 포함하지만 stream-event ITL에는 일부 구간도 넣지 않는다. `stream_event_itl_coverage`가 1.0이 아니면 ITL percentile을 논문 표에 싣지 않는다.

모든 percentile은 `numpy.percentile(method="linear")`로 계산한다. p50, p85, p90, p95, p99와 표본 수를 함께 보고한다.

## 실험 행렬

아래 축의 조합을 서로 다른 workload cell로 취급한다. cell 사이의 요청을 합쳐 percentile을 계산하지 않는다.

- 모델 및 정확한 FXB artifact
- input token length bucket
- output token length bucket 또는 고정 `max_new_tokens`
- offline / server-like scenario
- server-like target QPS
- framework worker count와 queue capacity
- RNGD device 및 tensor/data parallel topology

Server-like 실험은 낮은 부하부터 포화점을 넘는 부하까지 여러 target QPS로 실행한다. 각 QPS에서 achieved throughput, request TTFT, request mean TPOT, 오류·timeout 비율을 함께 기록한다. Offline 결과와 server-like 결과를 같은 분포로 합치지 않는다.

## 반복 및 채택 기준

1. runtime load 후 논문 측정에서 제외되는 warmup을 수행한다.
2. workload cell마다 successful request 1,000개 이상을 수집한다.
3. 동일 cell을 독립적으로 5회 이상 실행한다.
4. 대표값은 run별 지표의 median으로 보고하고, run-to-run 범위 또는 IQR도 함께 보고한다.
5. 한 run 안의 요청 표본을 독립 run처럼 취급하지 않는다.
6. `--save-request-trace`를 항상 사용한다.
7. trace drop, request failure, timeout, counter/timing invariant 실패가 하나라도 있으면 해당 run을 폐기한다.
8. stream-event ITL은 coverage 100%인 run만 채택한다. coverage가 낮은 run에서도 request TTFT와 request mean TPOT은 별도로 사용할 수 있지만, 관측 한계를 명시한다.

## 실행 전 기록

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

## 실행 예시

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

## 사후 검증

각 run에 대해 다음을 확인한다.

- CSV, details JSON, trace JSONL의 run ID가 같다.
- `async_outstanding_requests == 0`이다.
- `invalid_reasons`가 비어 있다.
- `request_trace_dropped:*` warning이 없다.
- trace의 successful generation row 수가 집계의 `async_generation_observed_requests`와 같다.
- raw event에서 다시 계산한 TTFT/TPOT/ITL percentile이 details JSON과 일치한다.
- `generation_stream_itl_incomplete` warning이 있으면 stream-event ITL percentile을 사용하지 않는다.
- SDK version이 `null`이거나 Git dirty 여부를 알 수 없는 run은 환경을 보완해 다시 실행한다.

## 논문 표기

표와 본문에는 하드웨어명만 쓰지 말고 SDK/driver/FXB hash, 모델, token-length 조건, scenario, offered QPS, achieved throughput, 표본 수, 반복 횟수를 함께 적는다. Python stream에서 관측한 시각은 NPU 내부 cycle-level timestamp가 아니므로 지표 이름을 `Python stream-observed ITL`처럼 명시한다. 내부 토큰 실행 시간을 직접 측정한 값으로 표현하지 않는다.

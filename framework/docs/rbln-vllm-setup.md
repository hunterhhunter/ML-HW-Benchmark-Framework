# Rebellions ATOM vLLM 런타임 설치·실행 가이드

이 문서는 Rebellions ATOM(`RBLN-CA22`) 서버에서 Llama 3.2 3B와
Llama 3.1 8B를 프레임워크의 내부 Python 엔진으로 실행하는 절차를
정리한다. `rbln-vllm` target은 외부 HTTP 서버를 띄우지 않고, 준비된
Optimum RBLN 모델 디렉터리를 같은 프로세스의 vLLM RBLN 엔진에 넘긴다.

모델 다운로드와 RBLN 컴파일은 벤치마크 실행과 분리한다. 벤치마크
프로세스는 이미 준비된 로컬 디렉터리만 읽기 때문에 실행 중 Hub 상태나
컴파일 시간에 영향을 받지 않는다.

## 1. 지원 범위와 한 장 구성

| 모델 | ATOM 수 | 분류 | 권장 계약 |
|---|---:|---|---|
| Llama 3.2 3B Instruct | 8 | Rebellions 공식 지원 | `max_seq_len=4096`, `block_size=4096` |
| Llama 3.2 3B Instruct | 1 | 비공식 실험 | `max_seq_len=512`, 최대 1024, batch 1 |
| Llama 3.1 8B Instruct | 8 | Rebellions 공식 지원 | `max_seq_len=131072`, `block_size=16384` |
| Llama 3.1 8B Instruct | 1 | 실행 차단 | BF16 가중치만 약 16 GB라 15.7 GiB 카드에 런타임·KV cache까지 수용 불가 |

따라서 현재 한 장 서버에서는 Llama 3.2 3B만 실험할 수 있다. 이 경로는
공식 지원 조합이 아니므로 컴파일 성공을 보장하지 않으며, 성능 결과에는
`support_classification=unsupported_single_npu_experiment`가 기록된다.
Llama 3.1 8B 한 장 실행은 늦은 OOM 대신 모델 로드 전에 명시적으로
거부한다.

공식 지원 표와 버전별 요구 사항은 아래 문서를 기준으로 확인한다.

- <https://docs.rbln.ai/latest/software/optimum/index.html>
- <https://docs.rbln.ai/latest/software/model_serving/vllm_support/index.html>
- <https://docs.rbln.ai/latest/software/model_serving/vllm_support/configuration/configuration-guide.html>
- <https://docs.rbln.ai/latest/software/model_serving/vllm_support/tutorial/vllm_llama3.1-8B_flash_attention.html>

## 2. 현재 ATOM 서버에서 바로 시작

현재 서버에서 이미 검증한 구성은 다음과 같다.

- 프레임워크 저장소: `~/ML-HW-Benchmark-Framework-rbln`
- 실행 Python: `~/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python`
- 준비된 모델: `~/rebelion/rbln-model-zoo/custom/framework-contracts/llama-3.2-3b-npu1-seq512`
- Python 3.10.12, `rebel-compiler==0.11.0`,
  `optimum-rbln==0.11.0.post1`, `vllm-rbln==0.11.0`
- RBLN-CA22 한 장, driver/firmware 3.2.2

다른 가속기 실험 worktree를 건드리지 않고 원격 feature 브랜치를 별도
worktree로 연다.

```bash
cd ~/ML-HW-Benchmark-Framework-rbln
git status --short
git fetch origin

test ! -e "$HOME/ML-HW-Benchmark-Framework-rbln-vllm"
git worktree add --detach \
  "$HOME/ML-HW-Benchmark-Framework-rbln-vllm" \
  origin/feat/rbln-vllm

export RBLN_FW_ROOT="$HOME/ML-HW-Benchmark-Framework-rbln-vllm"
export RBLN_RUN_PY="$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"
export RBLN_VLLM_PY="$RBLN_RUN_PY"
export RBLN_LLM_DIR="$HOME/rebelion/rbln-model-zoo/custom/framework-contracts/llama-3.2-3b-npu1-seq512"
export RBLN_DATASET="$HOME/ML-HW-Benchmark-Framework-rbln/framework/datasets/squad2/val.json"
```

이 서버의 `rebel` 모듈은
`~/.local/lib/python3.10/site-packages`에 있고 `.venv-rbln`이 user-site를
포함하는 hybrid 환경이다. 따라서 실행할 때 `PYTHONNOUSERSITE=1`을
설정하지 않는다. 아래 preflight로 정확한 interpreter와 패키지 위치를
고정해서 확인한다.

```bash
"$RBLN_RUN_PY" - <<'PY'
import importlib.metadata as md
import sys

import rebel
from optimum.rbln import RBLNLlamaForCausalLM

print("python:", sys.executable)
print("rebel:", rebel.__file__)
for name in ("rebel-compiler", "optimum-rbln", "vllm-rbln", "vllm"):
    print(name, md.version(name))
print("NPU:", rebel.npu_is_available(0), rebel.get_npu_name(0))
assert rebel.npu_is_available(0)
assert rebel.device_count() == 1
PY

test -f "$RBLN_LLM_DIR/config.json"
test -f "$RBLN_LLM_DIR/rbln-vllm-manifest.json"
test -f "$RBLN_LLM_DIR/prefill.rbln"
test -f "$RBLN_LLM_DIR/decoder_batch_1.rbln"
test -f "$RBLN_DATASET"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_RUN_PY" -m src.main --help >/dev/null
rbln-smi -j
```

실행 전 `contexts`는 빈 배열이어야 한다. 이 preflight가 통과하면 모델을
다시 다운로드하거나 7.6 GB artifact를 다시 컴파일할 필요가 없다.

## 3. 새 환경을 만드는 경우

다른 가속기 실험과 작업 파일이 섞이지 않도록 별도 worktree를 권장한다.
이미 해당 worktree가 있다면 그 디렉터리로 이동하면 된다.

```bash
cd ~/ML-HW-Benchmark-Framework
git fetch origin
git worktree add --detach \
  ../ML-HW-Benchmark-Framework-rbln-vllm \
  origin/feat/rbln-vllm
cd ../ML-HW-Benchmark-Framework-rbln-vllm

export RBLN_FW_ROOT="$PWD"
export RBLN_VLLM_PY="$RBLN_FW_ROOT/.venv-rbln-vllm/bin/python"
```

서버와 장치를 먼저 확인한다.

```bash
cat /etc/os-release
rbln-smi -q
rbln-smi -j
ls -l /dev/rbln*
```

한 장 서버에서는 JSON의 `devices` 길이가 1이고 `status`가 `normal`이어야
한다. 실행 전 다른 프로세스의 `contexts`가 있으면 같은 카드에서 얻은
성능 결과가 아니므로 종료하거나 별도 시간에 실행한다.

## 4. uv 가상환경과 RBLN 패키지

드라이버와 `/usr/bin/rbln-smi`가 전역 설치되어 있어도 Python SDK는
가상환경에 별도로 설치해야 한다. RBLN SDK 0.11 계열은 이 서버에서
검증한 Python 3.10 환경을 사용한다.

```bash
cd "$RBLN_FW_ROOT"
uv venv --python /usr/bin/python3.10 .venv-rbln-vllm

uv pip install \
  --python "$RBLN_VLLM_PY" \
  --extra-index-url https://pypi.rbln.ai/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://wheels.vllm.ai/0.22.0/cpu \
  rebel-compiler==0.11.0 \
  optimum-rbln==0.11.0.post1 \
  vllm-rbln==0.11.0 \
  datasets \
  huggingface_hub \
  transformers
```

`401 Unauthorized`가 나오면 일반 PyPI 계정으로 해결하지 않는다.
Rebellions Portal 계정에 패키지 저장소 권한이 있어야 하며, 조직에서
제공한 pip 인증 설정을 적용해야 한다. 비밀번호나 토큰을 명령줄 URL,
쉘 히스토리, 저장소 파일에 넣지 않는다. 권한이 없다면 Portal 관리자나
`client_support@rebellions.ai`에 SDK 저장소 접근 권한을 요청한다.

설치 결과와 장치 접근을 확인한다.

```bash
uv pip list --python "$RBLN_VLLM_PY" \
  | grep -Ei 'rebel|optimum-rbln|vllm-rbln|vllm|torch|transformers'

"$RBLN_VLLM_PY" - <<'PY'
import rebel

print("available:", rebel.npu_is_available(0))
print("name:", rebel.get_npu_name(0))
print("count:", rebel.device_count())
PY
```

`vllm-rbln` 의존성 해석에서 PyTorch/vLLM 전용 index의 `setuptools`만
선택해 충돌하면 `uv`가 안내하는 대로 신뢰 가능한 index에 한해서
`--index-strategy unsafe-best-match`를 추가한다. 현재 서버의 검증된
`.venv-rbln`에는 다시 설치하지 않는다.

## 5. Hugging Face 접근과 데이터셋

Meta Llama 모델은 Hugging Face에서 라이선스 승인 후 접근 토큰이 필요하다.
서버에서 한 번 로그인하되 토큰을 저장소에 기록하지 않는다.

```bash
"$RBLN_FW_ROOT/.venv-rbln-vllm/bin/hf" auth login
```

이미 `framework/datasets/squad2/val.json`이 있으면 준비 과정을 건너뛴다.
없으면 framework 디렉터리에서 자동 준비 경로를 한 번 실행하거나 현재
프로젝트의 SQuAD 준비 스크립트를 사용한다.

```bash
cd "$RBLN_FW_ROOT/framework"
test -f datasets/squad2/val.json
export RBLN_DATASET="$RBLN_FW_ROOT/framework/datasets/squad2/val.json"
```

## 6. 모델 다운로드·RBLN 준비

준비 도구는 모델 다운로드, Optimum RBLN 컴파일, tokenizer/config 저장,
파일별 SHA256과 실행 계약을 담은 `rbln-vllm-manifest.json` 생성을 한 번에
처리한다. 기존 출력 디렉터리는 덮어쓰지 않는다.

### 한 장: Llama 3.2 3B 실험

먼저 512 토큰 계약으로 시작한다. 컴파일도 한 장 구성이므로
`--allow-unsupported-single-npu`가 반드시 필요하다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.2-3b \
  --output-dir models/rbln-vllm/llama-3.2-3b-npu1-seq512 \
  --num-devices 1 \
  --max-seq-len 512 \
  --block-size 512 \
  --batch-size 1 \
  --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

512가 성공한 뒤에만 1024를 별도 디렉터리로 시험한다. 한 장 실험의
프레임워크 상한은 1024이며 공식 지원을 뜻하지 않는다.

### 여덟 장: Llama 3.2 3B 공식 구성

```bash
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.2-3b \
  --output-dir models/rbln-vllm/llama-3.2-3b-npu8-seq4096 \
  --num-devices 8 \
  --max-seq-len 4096 \
  --block-size 4096 \
  --batch-size 1 \
  --decoder-batch-sizes 1
```

### 여덟 장: Llama 3.1 8B 공식 구성

```bash
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.1-8b \
  --output-dir models/rbln-vllm/llama-3.1-8b-npu8-seq131072 \
  --num-devices 8 \
  --max-seq-len 131072 \
  --block-size 16384 \
  --batch-size 1 \
  --decoder-batch-sizes 1
```

8장 공식 구성에서 실제 continuous batching을 4개까지 사용하려면 준비할 때
`--batch-size 4 --decoder-batch-sizes 1,2,4`로 별도 artifact를 만들고,
실행할 때도 `max_num_seqs=4`, `decoder_batch_sizes=1,2,4`로 동일하게 맞춘다.
컴파일 batch 1 artifact를 실행 옵션만 4로 바꾸는 것은 허용하지 않는다.

산출물을 검사한다.

```bash
export RBLN_LLM_DIR="$RBLN_FW_ROOT/framework/models/rbln-vllm/llama-3.2-3b-npu1-seq512"

test -f "$RBLN_LLM_DIR/config.json"
test -f "$RBLN_LLM_DIR/rbln-vllm-manifest.json"
find "$RBLN_LLM_DIR" -type f -name '*.rbln' -ls
"$RBLN_VLLM_PY" -m json.tool \
  "$RBLN_LLM_DIR/rbln-vllm-manifest.json"
```

manifest의 `num_devices`, `max_seq_len`, `block_size`, `batch_size`,
`decoder_batch_sizes`는 실행 옵션과 같아야 한다. 프레임워크는 장치·shape·
batch 계약 불일치와 손상된 기본 구조를 엔진 생성 전에 거부한다.

## 7. 동기 E2E smoke

한 장 Llama 3.2 3B의 기본 smoke 명령이다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLM_DIR" \
  --tokenizer-path "$RBLN_LLM_DIR" \
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
```

공식 여덟 장 구성은 모델 디렉터리, `--max-model-len`, `block_size`,
`num_devices=8`을 manifest와 맞추고
`allow_unsupported_single_npu` 옵션을 제거한다.

## 8. 비동기 offline

프레임워크 큐가 입장 제어, 제한 시간, 요청 추적과 통계를 담당하고,
vLLM RBLN의 내부 `AsyncLLMEngine`이 continuous batching과 토큰 스트림을
담당한다. `--worker-count`는 프레임워크 요청 제출 동시성이며 NPU 엔진을
여러 개 만드는 값이 아니다.

첫 실행은 큐와 worker를 모두 1로 고정한 4-request smoke다.

```bash
"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLM_DIR" \
  --tokenizer-path "$RBLN_LLM_DIR" \
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
  --runtime-option decoder_batch_sizes=1 \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --debug \
  --results-path results/rbln-llama32-3b-npu1-async-smoke.csv
```

smoke와 context 정리를 확인한 뒤에만 100-request 측정을 실행한다.

```bash
"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLM_DIR" \
  --tokenizer-path "$RBLN_LLM_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 32 \
  --worker-count 4 \
  --queue-capacity 16 \
  --min-samples 100 \
  --max-samples 100 \
  --warmup 2 \
  --flush-timeout-sec 600 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option decoder_batch_sizes=1 \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --results-path results/rbln-llama32-3b-npu1-async-offline.csv
```

한 장에서 메모리 부족이나 초기화 실패가 나면
`worker-count=1`로 낮춰 smoke를 먼저 통과시킨다. 한 장 실험 artifact는
컴파일 batch가 1이므로 `max_num_seqs=1`, `decoder_batch_sizes=1`은 바꾸지
않는다. 측정 조건을 바꾸면 결과 파일도 분리한다.

## 9. 비동기 server-like

낮은 QPS부터 시작해서 timeout/rejection 없이 안정적인 구간을 찾는다.

```bash
"$RBLN_VLLM_PY" -m src.main \
  --model llama-3.2-3b \
  --target rbln-vllm \
  --model-path "$RBLN_LLM_DIR" \
  --tokenizer-path "$RBLN_LLM_DIR" \
  --dataset "$RBLN_DATASET" \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 1 \
  --batch-size 1 \
  --max-model-len 512 \
  --max-new-tokens 32 \
  --worker-count 4 \
  --queue-capacity 16 \
  --min-samples 100 \
  --min-duration-sec 60 \
  --max-samples 100 \
  --warmup 2 \
  --schedule-seed 23 \
  --latency-slo-ms 5000 \
  --flush-timeout-sec 600 \
  --runtime-option block_size=512 \
  --runtime-option num_devices=1 \
  --runtime-option max_num_seqs=1 \
  --runtime-option decoder_batch_sizes=1 \
  --runtime-option allow_unsupported_single_npu=true \
  --save-request-trace \
  --monitor \
  --results-path results/rbln-llama32-3b-npu1-async-server-qps1.csv
```

## 10. 결과 판정과 모니터링

LLM에서는 다음 값을 함께 본다.

- `Samples/s`: 프레임워크 샘플 처리율
- `async_completed_tokens_per_sec`: 완료 토큰 처리율
- `async_generation_request_ttft_p50/p95/p99_ms`: 첫 토큰 지연
- `async_generation_request_mean_tpot_p50/p95/p99_ms`: 토큰 간 평균 시간
- `async_e2e_latency_p50/p95/p99_ms`: 큐 대기 포함 요청 지연
- `async_queue_wait_p99_ms`, `async_service_time_p99_ms`: 병목 분리
- failed/rejected/timed-out/outstanding: 모두 0이어야 valid 후보
- `hw_accel_util_*`, `power_w_*`, `energy_j`, `monitor_coverage`: NPU 상태

측정 시간이 너무 짧으면 `rbln-smi` 샘플 수가 1~2개라 utilization과 전력이
대표성을 갖지 못한다. full run은 최소 수십 초 이상 실행하고
`hw_accel_monitor_coverage=1.0` 및 충분한 `hw_accel_power_samples`를 확인한다.
현재 collector는 다중 NPU 공식 구성에서도 선택한 대표 장치의 지표를
수집하므로 카드별 전력 합계가 필요하면 별도의 외부 수집을 병행한다.

## 11. 종료와 context 정리

각 실행 뒤 엔진이 unload되고 모든 context가 사라져야 한다.

```bash
rbln-smi -j

rbln-smi -j | "$RBLN_VLLM_PY" -c '
import json, sys
payload = json.load(sys.stdin)
contexts = payload.get("contexts", [])
print("contexts:", contexts)
raise SystemExit(0 if not contexts else 1)
'
```

`contexts`가 남으면 즉시 다음 실험을 시작하지 않는다. JSON의 PID를
`ps -fp <PID>`로 확인하고, 본인이 시작한 벤치마크 프로세스가 실제로 종료된
뒤 다시 검사한다. 다른 사용자의 프로세스를 종료하지 않는다.

## 12. 자주 발생하는 문제

| 증상 | 원인·조치 |
|---|---|
| `401 Unauthorized` | Portal 저장소 권한/조직 pip 인증 확인. 자격 증명을 저장소에 넣지 말 것 |
| gated model 401/403 | Meta 라이선스 승인과 Hugging Face 토큰 확인 |
| `No module named rebel/optimum/vllm` | 전역 드라이버와 uv 가상환경 패키지는 별개. `--python "$RBLN_VLLM_PY"`로 재확인 |
| `block_size must divide max_model_len` | 준비 manifest와 실행의 두 값을 동일 계약으로 맞춤 |
| `compiled for N devices but runtime requested M` | 다른 장치 수로 만든 디렉터리 재사용 금지 |
| 3B 한 장 opt-in 오류 | 준비와 실행 양쪽에 single-NPU 실험 opt-in 필요 |
| 8B 한 장 거부 | 정상 동작. 여덟 장 공식 구성 사용 |
| async 지연이 크고 NPU 사용률이 낮음 | queue wait와 service time을 분리하고 max_num_seqs/worker/QPS를 단계적으로 조정 |
| 종료 뒤 context 잔류 | 해당 PID 확인 후 엔진 종료 완료를 기다리고 다음 실험 중지 |
| async 엔진 시작이 10분 초과 | `startup_timeout_sec`를 늘리기 전에 PID, 메모리, context와 모델 경로를 확인. 실패 시 runtime이 소유권을 보존하고 unload를 재시도함 |

## 13. 서버 검증 순서

처음에는 아래 순서를 바꾸지 않는다.

1. `rbln-smi -j`에서 장치 `normal`, `contexts: []` 확인
2. 관련 Python 회귀 테스트 실행
3. 동기 E2E 1 sample smoke
4. 종료 뒤 `contexts: []` 확인
5. 비동기 offline 4 samples, worker 1, queue 1 smoke
6. 종료 뒤 `contexts: []` 확인
7. 비동기 offline 100 samples, worker 4, queue 16
8. server-like QPS 1부터 단계적으로 증가

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_RUN_PY" -m pytest -q \
  tests/test_rbln_vllm_runtime.py \
  tests/test_prepare_rbln_vllm_model.py \
  tests/test_plugin_registry.py \
  tests/test_main_paths.py \
  tests/test_async_cli.py
```

다른 터미널에서는 `watch -n 1 rbln-smi`로 context, memory, utilization,
전력과 온도를 본다. 벤치마크 결과는 성공 메시지만으로 판정하지 않고
failed/rejected/timed-out/outstanding가 모두 0인지, TTFT/TPOT와 처리율이
기록됐는지, monitor coverage가 1.0인지 함께 확인한다.

## 14. 현재 구현 경계

현재 target은 내부 Python 런타임과 프레임워크의 동기 E2E 및 비동기 큐만
지원한다. OpenAI 호환 HTTP endpoint, 다중 클라이언트 네트워크 부하,
외부 vLLM 서버 프로세스 관리는 후속 확장 범위다. 그때도 모델 준비
manifest와 결과 지표 계약은 그대로 재사용한다.

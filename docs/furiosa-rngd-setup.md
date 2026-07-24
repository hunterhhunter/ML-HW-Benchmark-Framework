# Furiosa RNGD runtime

이 문서는 Furiosa SDK 2026.3.0에서 Hugging Face 모델 ID, 로컬 모델 디렉터리 또는 사전 컴파일된 `.fxb`를 사용해 Llama 3.1/3.2 생성 벤치마크를 실행하는 절차를 설명합니다. 임베디드 E2E·비동기 추론은 Furiosa-LLM Python API를 직접 호출하므로 `furiosa-llm serve`가 필요하지 않습니다. OpenAI-compatible server 측정은 별도 서버를 실행하고 [RNGD 논문용 생성 지연 프로토콜](rngd-paper-benchmark.md)을 따릅니다. 이 브랜치는 BERT, FXB 컴파일과 Furiosa SMI collector를 포함하지 않습니다.

설치·빌드·실행 중 오류가 발생하면 [Furiosa RNGD 트러블슈팅 Runbook과 개발자 분석](furiosa-rngd-troubleshooting.md)에서 오류 문자열별 원인, 확인 명령, 해결 절차와 현재 SDK 한계를 확인하세요.

## 전용 Python 환경

Furiosa-LLM 2026.3.0은 PyTorch 2.5.1 환경을 요구합니다. 기본 `framework/requirements.txt`는 PyTorch 2.10.0/CUDA 패키지를 고정하므로 RNGD 환경에 설치하지 마세요.

```bash
cd framework
uv venv .venv-rngd --python 3.12
source .venv-rngd/bin/activate
python -m pip install --upgrade pip setuptools wheel uv
uv pip install --upgrade --torch-backend=auto furiosa-llm==2026.3.0

# 프레임워크 실행과 fake-SDK 테스트에 필요한 비벤더 패키지
uv pip install numpy datasets pandas onnx psutil pyyaml pytest
uv pip check
```

호스트 드라이버와 prerequisite 설치는 [Furiosa-LLM Quick Start](https://developer.furiosa.ai/latest/en/get_started/furiosa_llm.html)를 따릅니다. `furiosa_llm`은 import 시 native runtime을 초기화할 수 있으므로 adapter는 `load()` 또는 native backend 생성 시점까지 vendor import를 지연합니다.

## 모델 artifact 선택

권장 경로는 `--model-path`에 `furiosa-ai/Llama-3.1-8B-Instruct` 같은 Hugging Face repository ID를 전달하고 `--fxb`를 생략하는 것입니다. 이때 Furiosa-LLM SDK가 모델 ID로부터 호환되는 artifact를 해석해 로딩합니다. 처음 실행할 때는 모델 다운로드를 위한 네트워크와 충분한 캐시 공간이 필요할 수 있습니다.

미리 받은 Hugging Face 모델 repository 디렉터리도 `--model-path`에 전달할 수 있습니다. 디렉터리 안에 `artifact.json`, `binary_bundle.zip`, safetensors 등이 있고 독립된 `.fxb` 파일이 보이지 않더라도 repository root 자체가 SDK의 모델 입력입니다.

특정 FXB를 고정해 재현해야 할 때만 `--fxb`를 함께 지정합니다. `--artifact`는 기존 호환을 위한 같은 의미의 alias입니다.

```bash
furiosa-smi info

# 명시적 FXB override를 사용할 때만 확인
fxb show /absolute/path/to/model.fxb

# cache에 등록한 FXB의 모델 fingerprint 호환성을 추가로 확인할 때
fxb add /absolute/path/to/model.fxb
fxb check <hugging-face-repo-id>
```

`fxb check`는 cache의 bundle과 Hugging Face repository ID의 config를 비교하는 명령입니다. 명시적 FXB를 쓰지 않는 기본 경로에서는 실행할 필요가 없습니다. 로컬 디렉터리와 FXB만 보유한 경우에는 원본 repository ID가 있을 때 이 검사를 수행하고, 없으면 `fxb show` 결과와 실제 `LLM(..., fxb=...)` load 검증을 사용합니다.

FXB fingerprint와 실행 중인 FuriosaIR revision이 맞아야 합니다. 자세한 형식과 명령은 [Furiosa FXB 문서](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)를 참고하세요.

## E2E 실행

```bash
cd framework
source .venv-rngd/bin/activate

python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --dataset datasets/squad2/val.json \
  --inference-mode e2e \
  --warmup 2 \
  --max-new-tokens 32 \
  --max-steps 2
```

이 명령은 공통 `BenchmarkRunner` 파이프라인 안에서 `LLM.generate()`를 직접 호출합니다. 별도 `furiosa-llm serve` 프로세스를 실행하지 마세요. 이미 로컬에 받은 repository를 사용할 때는 `--model-path`만 해당 디렉터리로 바꿉니다.

명시적 FXB override가 필요한 경우 다음 두 인자를 함께 사용합니다.

```bash
--model-path /absolute/path/to/Llama-3.1-8B-Instruct \
--fxb /absolute/path/to/llama-3.1-8b.fxb
```

Llama 3.2 3B는 `--model llama-3.2-3b`와 그 모델에 맞는 repository ID 또는 로컬 디렉터리를 사용합니다. `--artifact /path/model.fxb`도 명시적 FXB fallback으로 동작하지만 새 명령은 `--fxb`를 권장합니다.

동기 `LLM.generate()`는 전체 wall-clock latency만 제공합니다. 이 경로의 TTFT/TPOT은 `None`이며 프레임워크가 total latency를 임의로 나눠 추정하지 않습니다.

## Native async 실행

Offline은 요청을 가능한 빠르게 공급하고, server-like는 설정한 QPS로 요청을 생성합니다.

```bash
# Offline
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --dataset /absolute/path/to/squad2/val.json \
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 32 \
  --worker-count 8 \
  --max-samples 100

# Server-like
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --dataset /absolute/path/to/squad2/val.json \
  --inference-mode async_queue \
  --scenario server_like \
  --target-qps 10 \
  --queue-capacity 32 \
  --worker-count 8 \
  --min-duration-sec 30 \
  --min-samples 100
```

async 모드에서는 framework가 생성 요청을 다시 동적 배칭하지 않습니다. worker마다 한 요청을 `AsyncLLMEngine`에 제출하고 Furiosa continuous batching에 맡깁니다. `max_inflight`는 `min(worker_count, queue_capacity)`로 제한됩니다. TTFT는 첫 non-empty stream output, TPOT은 첫 token부터 final output까지의 평균 간격으로 기록됩니다.

## Runtime options

`--runtime-option key=value`로 다음 옵션만 전달할 수 있습니다.

- `devices`
- `data_parallel_size`
- `pipeline_parallel_size`
- `max_io_memory_mb`
- `seed`
- `cache_dir`
- `npu_queue_limit`
- `max_processing_samples`
- `spare_blocks_ratio`

예:

```bash
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --runtime-option devices=npu:0 \
  --runtime-option npu_queue_limit=4 \
  --runtime-option max_processing_samples=256
```

실장비 완료 판정에서는 두 Llama 프로필에 대해 E2E, Offline async, Server-like async를 각각 실행하고 결과의 실패 요청 수, outstanding 요청 수, TTFT/TPOT 표본 수를 함께 확인합니다.

## OpenAI-compatible server 측정

HTTP serving 결과는 위의 임베디드 `AsyncLLMEngine` 결과와 다른 실험입니다. `furiosa-llm serve`의 `/v1/completions`를 vLLM 0.16.0 부하 생성기로 호출하고, client streaming ITL과 server Prometheus ITL을 별도 네임스페이스로 저장합니다. 실행 환경, 부하 행렬, 유효성 gate와 보고 방식은 [논문용 프로토콜](rngd-paper-benchmark.md)을 참고하세요.

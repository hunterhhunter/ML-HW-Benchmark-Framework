# Furiosa RNGD runtime

이 문서는 Furiosa SDK 2026.3.0과 사전 컴파일된 `.fxb`를 사용해 Llama 3.1/3.2 생성 벤치마크를 실행하는 절차를 설명합니다. 이 브랜치는 BERT, FXB 컴파일과 Furiosa SMI collector를 포함하지 않습니다. OpenAI-compatible server 측정은 [RNGD 논문용 생성 지연 프로토콜](rngd-paper-benchmark.md)을 따릅니다.

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

## 모델과 FXB 확인

`--model-path`에는 Hugging Face 형식의 로컬 모델·토크나이저 디렉터리를, `--fxb`에는 명시적인 FXB 파일을 전달합니다. 두 경로를 섞지 않습니다.

```bash
furiosa-smi info
fxb show /absolute/path/to/model.fxb

# cache에 등록한 FXB의 모델 fingerprint 호환성을 추가로 확인할 때
fxb add /absolute/path/to/model.fxb
fxb check <hugging-face-repo-id>
```

`fxb check`는 cache의 bundle과 Hugging Face repository ID의 config를 비교하는 명령입니다. 로컬 디렉터리만 보유한 경우에는 원본 repository ID가 있을 때 이 검사를 수행하고, 없으면 `fxb show` 결과와 실제 `LLM(..., fxb=...)` load 검증을 사용합니다.

FXB fingerprint와 실행 중인 FuriosaIR revision이 맞아야 합니다. 자세한 형식과 명령은 [Furiosa FXB 문서](https://developer.furiosa.ai/latest/en/furiosa_llm/fxb.html)를 참고하세요.

## E2E 실행

```bash
cd framework
source .venv-rngd/bin/activate

python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path /absolute/path/to/Llama-3.1-8B \
  --fxb /absolute/path/to/llama-3.1-8b.fxb \
  --dataset /absolute/path/to/squad2/val.json \
  --inference-mode e2e \
  --max-new-tokens 128
```

Llama 3.2 3B는 `--model llama-3.2-3b`와 그 모델에 맞는 model/FXB 경로를 사용합니다. `--artifact /path/model.fxb`도 fallback으로 동작하지만 새 명령은 `--fxb`를 권장합니다.

동기 `LLM.generate()`는 전체 wall-clock latency만 제공합니다. 이 경로의 TTFT/TPOT은 `None`이며 프레임워크가 total latency를 임의로 나눠 추정하지 않습니다.

## Native async 실행

Offline은 요청을 가능한 빠르게 공급하고, server-like는 설정한 QPS로 요청을 생성합니다.

```bash
# Offline
python src/main.py \
  --model llama-3.2-3b \
  --target furiosa-rngd \
  --model-path /absolute/path/to/Llama-3.2-3B \
  --fxb /absolute/path/to/llama-3.2-3b.fxb \
  --dataset /absolute/path/to/squad2/val.json \
  --inference-mode async_queue \
  --scenario offline \
  --queue-capacity 32 \
  --worker-count 8 \
  --max-samples 100

# Server-like
python src/main.py \
  --model llama-3.2-3b \
  --target furiosa-rngd \
  --model-path /absolute/path/to/Llama-3.2-3B \
  --fxb /absolute/path/to/llama-3.2-3b.fxb \
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
  --model-path /models/Llama-3.1-8B \
  --fxb /models/llama-3.1-8b.fxb \
  --runtime-option devices=npu:0 \
  --runtime-option npu_queue_limit=4 \
  --runtime-option max_processing_samples=256
```

실장비 완료 판정에서는 두 Llama 프로필에 대해 E2E, Offline async, Server-like async를 각각 실행하고 결과의 실패 요청 수, outstanding 요청 수, TTFT/TPOT 표본 수를 함께 확인합니다.

## OpenAI-compatible server 측정

HTTP serving 결과는 위의 임베디드 `AsyncLLMEngine` 결과와 다른 실험입니다. `furiosa-llm serve`의 `/v1/completions`를 vLLM 0.16.0 부하 생성기로 호출하고, client streaming ITL과 server Prometheus ITL을 별도 네임스페이스로 저장합니다. 실행 환경, 부하 행렬, 유효성 gate와 보고 방식은 [논문용 프로토콜](rngd-paper-benchmark.md)을 참고하세요.

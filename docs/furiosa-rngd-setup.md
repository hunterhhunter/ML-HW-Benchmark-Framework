# Furiosa RNGD runtime

이 문서는 Furiosa SDK 2026.3.0에서 Llama 3.1/3.2 생성과 BERT SST-2/SQuAD 추론 벤치마크를 실행하는 절차를 설명합니다. Llama는 Furiosa-LLM, BERT는 Furiosa Torch를 사용하므로 Python 환경을 분리해야 합니다. 임베디드 E2E·비동기 추론은 Python API를 직접 호출하므로 `furiosa-llm serve`가 필요하지 않습니다. OpenAI-compatible server 측정은 별도 서버를 실행하고 [RNGD 논문용 생성 지연 프로토콜](rngd-paper-benchmark.md)을 따릅니다. Llama 3.2 3B FXB 컴파일 절차는 포함하지만 Furiosa SMI collector 구현은 포함하지 않습니다.

설치·빌드·실행 중 오류가 발생하면 [Furiosa RNGD 트러블슈팅 Runbook과 개발자 분석](furiosa-rngd-troubleshooting.md)에서 오류 문자열별 원인, 확인 명령, 해결 절차와 현재 SDK 한계를 확인하세요. ResNet50, YOLOv5m, PatchTST의 strict 컴파일 실패를 다시 확인하려면 [모델 컴파일 실패 재현 기록](furiosa-rngd-compilation-troubleshooting.md)을 사용하세요.

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

### BERT 전용 Furiosa Torch 환경

Furiosa Torch 2026.3.0은 PyTorch 2.10.0을 사용하므로 위 Furiosa-LLM 환경에 같이 설치하지 않습니다.

```bash
cd framework
uv venv .venv-furiosa-torch --python 3.12
uv pip install \
  --python .venv-furiosa-torch/bin/python \
  -r requirements-furiosa-torch.txt
```

지원 범위는 RNGD 서버에서 검증한 다음 두 로컬 Hugging Face 모델 디렉터리입니다.

- `bert-base-uncased`: `models/textattack_bert-base-uncased-SST-2`
- `bert-base-uncased-squad-v1`: `models/csarron_bert-base-uncased-squad-v1`

ResNet50, YOLOv5m, PatchTST는 현재 strict RNGD 컴파일 검증을 통과하지 않았으므로 `furiosa-rngd-torch` 어댑터에 등록하지 않습니다. 세 모델의 CPU 성공 경계, graph 정규화, 최종 오류와 재현 CLI는 [Furiosa RNGD 모델 컴파일 실패 재현 기록](furiosa-rngd-compilation-troubleshooting.md)에 정리되어 있습니다.

## 컴파일과 artifact 재사용

RNGD에서 사용한 모델이 모두 같은 방식으로 준비된 것은 아닙니다. Llama 3.2 3B는 `fxb build`로 실행 bundle을 사전 컴파일했고, BERT 두 모델은 첫 추론 시 `torch.compile`이 정적 graph를 컴파일했습니다. Llama 3.1 8B는 Furiosa가 배포한 사전 컴파일 artifact를 사용했습니다.

| 모델 | 방식 | 결과 | 영구 artifact |
|---|---|---|---|
| Llama 3.2 3B Instruct | `fxb build` 사전 컴파일 | 성공 | `.fxb` |
| BERT SST-2 | 첫 추론 `torch.compile` | 성공 | 프레임워크가 별도 FXB를 만들지 않음 |
| BERT SQuAD v1 | 첫 추론 `torch.compile` | 성공 | 프레임워크가 별도 FXB를 만들지 않음 |
| Llama 3.1 8B Instruct | Furiosa 배포 artifact 로드 | 직접 컴파일하지 않음 | 모델 저장소의 배포 artifact |
| ResNet50 | strict `torch.compile` | 실패 | 없음 |
| YOLOv5m | strict `torch.compile` | 실패 | 없음 |
| PatchTST-FM-r1 | strict `torch.compile` | 실패 | 없음 |

### Llama 3.2 3B FXB 사전 컴파일

서버에서 성공한 조건은 Furiosa SDK 2026.3.0, tensor parallel 8, 최적화 수준 O0, 최대 model length 4096입니다. SDK 2026.3.0은 원본 `config.json`의 중복 `head_dim` 필드를 override key로 처리하다가 `Invalid config override key: head_dim`으로 중단했습니다. `head_dim=128`이 `hidden_size / num_attention_heads`와 같은지 확인한 뒤 빌드용 config에서만 제거했습니다.

```bash
cd ~/ML-HW-Benchmark-Framework

FURIOSA_BIN="$PWD/.venv-furiosa/bin"
BUILD_MODEL_DIR="$PWD/framework/models/llama-3.2-3b-instruct-config-test"

mkdir -p "$BUILD_MODEL_DIR"
"$FURIOSA_BIN/hf" download \
  meta-llama/Llama-3.2-3B-Instruct \
  config.json \
  --local-dir "$BUILD_MODEL_DIR"

cp -n \
  "$BUILD_MODEL_DIR/config.json" \
  "$BUILD_MODEL_DIR/config.original.json"

BUILD_MODEL_DIR="$BUILD_MODEL_DIR" "$FURIOSA_BIN/python" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["BUILD_MODEL_DIR"]) / "config.json"
config = json.loads(path.read_text())
explicit = config.get("head_dim")
derived = config["hidden_size"] // config["num_attention_heads"]
if explicit is not None and explicit != derived:
    raise RuntimeError(
        f"head_dim is not redundant: explicit={explicit}, derived={derived}"
    )
config.pop("head_dim", None)
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"head_dim override removed or already absent: previous={explicit}")
PY
```

먼저 `--dry-run`으로 architecture, bucket, TP 구성을 확인합니다. 이 단계는 kernel을 만들지 않으므로 컴파일 성공으로 판정하면 안 됩니다.

```bash
FXB_OUT="$PWD/framework/models/llama-3.2-3b-real-tw1024-$(date +%s).fxb"

"$FURIOSA_BIN/fxb" build \
  "$BUILD_MODEL_DIR" \
  "$FXB_OUT" \
  -tp 8 \
  -O O0 \
  --max-model-len 4096 \
  --dry-run

time "$FURIOSA_BIN/fxb" build \
  "$BUILD_MODEL_DIR" \
  "$FXB_OUT" \
  -tp 8 \
  -O O0 \
  --max-model-len 4096

"$FURIOSA_BIN/fxb" show "$FXB_OUT"
printf 'FXB_OUT=%s\n' "$FXB_OUT"
```

실제 성공 로그에서는 `first_tokenwise`, `mid_tokenwise`, `last_tokenwise_with_lm_head`, `full_attention`의 9개 bucket kernel이 모두 성공했고 마지막에 `Artifact Build Completed`가 출력됐습니다. 총 컴파일 시간은 약 15분 21초였습니다. 당시 생성된 파일은 다음과 같습니다.

```text
framework/models/llama-3.2-3b-real-tw1024-1784794954.fxb
```

파일명의 마지막 숫자는 실행 시각에 따른 값이므로 새 빌드에서는 달라집니다. 모든 kernel이 성공하고 `Artifact Build Completed`가 출력돼야 FXB 컴파일 성공입니다. `Broken pipe`, `tcc subprocess failed`, 일부 kernel만 성공한 로그는 실패입니다.

생성된 FXB는 원본 weight와 함께 프레임워크에 전달합니다.

```bash
cd ~/ML-HW-Benchmark-Framework/framework

../.venv-furiosa/bin/python src/main.py \
  --model llama-3.2-3b \
  --target furiosa-rngd \
  --model-path models/meta-llama_Llama-3.2-3B-Instruct-furiosa \
  --fxb models/llama-3.2-3b-real-tw1024-1784794954.fxb \
  --dataset datasets/squad2/val.json \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 2 \
  --max-new-tokens 64 \
  --max-steps 1
```

이때 FXB는 kernel bundle이고 모델 weight를 대신하지 않습니다. `--model-path`의 safetensors index에는 `lm_head.weight`가 실제 파일에 연결돼 있어야 합니다. 원본 tied-weight index에 `lm_head.weight`가 없던 디렉터리는 `param 'lm_head.weight' not in safetensors index`로 로딩에 실패했으며, 검증에서는 `model-lm-head.safetensors`를 포함한 `meta-llama_Llama-3.2-3B-Instruct-furiosa` 디렉터리를 사용했습니다.

### BERT 첫 호출 JIT 컴파일

BERT SST-2와 SQuAD v1은 FXB를 미리 만들지 않습니다. `FuriosaTorchRuntime.load()`가 다음 strict backend를 준비하고, 첫 warmup 또는 inference에서 실제 graph 컴파일과 RNGD 로딩이 시작됩니다.

```python
backend = furiosa.torch.backend.with_config(
    CompilerConfig(tactic_hint=TacticHintConfig.Default),
    eager_fallback=False,
)
compiled = torch.compile(
    model,
    backend=backend,
    fullgraph=True,
    dynamic=False,
)
```

따라서 로그가 warmup 직후 오랫동안 멈춘 것처럼 보여도 `furiosa-smi status`의 메모리와 프로세스를 함께 확인해야 합니다. `torch.compile()` 호출이 반환됐다는 사실만으로는 성공이 아니며, strict compile을 마친 첫 inference 결과가 반환돼야 성공입니다.

고정 입력 계약은 다음과 같습니다.

| 모델 | 입력 | dtype | 출력 |
|---|---|---|---|
| BERT SST-2 | `input_ids`, `attention_mask`: 각각 `(1, 128)` | `int64` | `logits`: `(1, 2)` |
| BERT SQuAD v1 | `input_ids`, `attention_mask`, `token_type_ids`: 각각 `(1, 384)` | `int64` | `start_logits`, `end_logits`: 각각 `(1, 384)` |

이 경로는 프레임워크 디렉터리에 재사용 가능한 FXB를 출력하지 않습니다. 같은 프로세스에서는 컴파일된 callable을 계속 사용하지만, 프로세스를 다시 시작하면 SDK cache 상태에 따라 컴파일이 다시 실행될 수 있습니다. 첫 실행 시간은 steady-state inference latency에 포함하지 말고 warmup으로 분리합니다.

### Llama 3.1 8B와 실패 모델의 구분

Llama 3.1 8B는 `furiosa-ai/Llama-3.1-8B-Instruct` 저장소의 `artifact.json`과 `binary_bundle.zip`을 Furiosa-LLM이 로드한 것입니다. 이 검증 과정에서는 Llama 3.1 8B를 `fxb build`로 직접 컴파일하지 않았습니다.

다음 세 모델은 CPU forward 또는 부분 graph 진단에 성공한 경우가 있지만, strict 전체 모델 컴파일은 성공하지 않았습니다.

```text
ResNet50: furiosa.UnsupportedOpError; compiler panic에 다음 오류가 포함됨
  align_up_required (true) != false (false)
  EinsumByDpe should be given only a single pass

YOLOv5m: CPU forward는 통과했지만 strict 전체 모델 RNGD 컴파일은 통과하지 못함

PatchTST-FM-r1: CPU forward는 통과했지만 strict 전체 모델 RNGD 컴파일은 통과하지 못함
  isolated transpose -> clone -> view graph는 통과했지만 전체 모델 지원 근거가 아님
```

작은 Conv2d나 isolated transpose graph의 성공은 compiler 경로 진단에는 유용하지만, 전체 모델이 컴파일·로딩·첫 inference까지 완료됐다는 의미는 아닙니다. 따라서 이 세 모델은 현재 `furiosa-rngd-torch` 지원 목록에서 제외합니다.

## BERT E2E 및 비동기 실행

Furiosa Torch 런타임은 `eager_fallback=False`, `fullgraph=True`, `dynamic=False`로 컴파일합니다. 배치와 worker는 모두 1로 고정하며, 비동기 모드는 네이티브 async가 아니라 프레임워크 blocking worker 큐를 사용합니다.

```bash
cd framework
PY=.venv-furiosa-torch/bin/python

# SST-2 E2E
"$PY" src/main.py \
  --model bert-base-uncased \
  --target furiosa-rngd-torch \
  --model-path models/textattack_bert-base-uncased-SST-2 \
  --dataset datasets/sst2_numpy \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 1 \
  --max-steps 1000

# SST-2 async queue
"$PY" src/main.py \
  --model bert-base-uncased \
  --target furiosa-rngd-torch \
  --model-path models/textattack_bert-base-uncased-SST-2 \
  --dataset datasets/sst2_numpy \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 1 \
  --warmup 1 \
  --max-samples 1000 \
  --min-samples 1000

# SQuAD v1 E2E
"$PY" src/main.py \
  --model bert-base-uncased-squad-v1 \
  --target furiosa-rngd-torch \
  --model-path models/csarron_bert-base-uncased-squad-v1 \
  --dataset datasets/squad_numpy \
  --inference-mode e2e \
  --batch-size 1 \
  --warmup 1 \
  --max-steps 1000

# SQuAD v1 async queue
"$PY" src/main.py \
  --model bert-base-uncased-squad-v1 \
  --target furiosa-rngd-torch \
  --model-path models/csarron_bert-base-uncased-squad-v1 \
  --dataset datasets/squad_numpy \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 1 \
  --warmup 1 \
  --max-samples 1000 \
  --min-samples 1000
```

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

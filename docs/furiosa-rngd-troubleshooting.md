# Furiosa RNGD 트러블슈팅 Runbook과 개발자 분석

이 문서는 Furiosa RNGD 서버에서 ML-HW-Benchmark-Framework의 LLM·비전 추론을 준비하고 검증하면서 실제로 확인한 장애를 정리한다. 앞부분은 오류 문자열로 원인을 찾아 실행을 복구하는 운영 Runbook이고, 뒷부분은 프레임워크 경계와 후속 개선점을 설명하는 개발자 분석이다.

정상 설치·실행 절차는 [Furiosa RNGD runtime](furiosa-rngd-setup.md), OpenAI-compatible serving 측정은 [RNGD 논문용 생성 지연 벤치마크 프로토콜](rngd-paper-benchmark.md)을 먼저 참고한다.

## 문서 사용법

- **검증 완료**: 서버 로그와 완료된 RNGD 벤치마크로 확인했다.
- **추정**: 로그로 가능성을 좁혔지만 통제된 대조 실험이 더 필요하다.
- **미해결**: 재현했지만 우회 또는 수정이 검증되지 않았다.
- **현재 SDK 한계**: 프레임워크 외부의 compiler/runtime에서 중단됐다.

Unit test 통과와 실장비 검증은 별개다. 이 문서에서 모델 지원 완료는 RNGD 모델 로드, 실제 추론, 결과 저장까지 확인했을 때만 사용한다.

## 검증 환경

아래 값은 이번 검증 서버에서 관찰한 환경이며 모든 서버의 보편적인 요구사항을 뜻하지 않는다.

| 항목 | 확인값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-124-generic |
| CPU | Intel Xeon Silver 4514Y, 64 logical CPUs |
| Memory | 251 GiB |
| RNGD | PCI ID `1ed2:0001`, `npu0`, 47.50 GiB |
| Furiosa driver | 2026.3.0 |
| Firmware | 1.11.0, `cfd5306` |
| Furiosa-LLM | 2026.3.0 |
| Furiosa Torch | 2026.3.0 |
| Python | Furiosa virtual environment: 3.12.13; host initially: 3.10.12 |

검증 호스트에는 실행 중인 커널과 일치하는 headers 및 `linux-modules-extra`가 있었고 Secure Boot는 비활성화되어 있었다.

## 모델 상태

| 모델 | 실행 경로 | 상태 | 근거 |
|---|---|---|---|
| Llama 3.1 8B | Furiosa-LLM legacy artifact/repository ID | **검증 완료** | SQuAD2 E2E 1,000건 및 native async 1,000건 완료 |
| Llama 3.2 3B | local HF weights + custom FXB | **조건부 검증 완료** | E2E/native async 완료. Exact registry entry 없이 nearest preset으로 fallback |
| ResNet50 | ONNX → PyTorch → Furiosa Torch | **현재 SDK 한계** | CPU parity 성공 후 Conv2d compiler 내부 panic |
| YOLOv5m | adapter/준비 코드 | **미검증** | RNGD full-graph 실행 증거 없음 |
| BERT SST-2/SQuAD | adapter/준비 코드 | **미검증** | RNGD full-graph 실행 증거 없음 |
| PatchTST | adapter/준비 코드 | **미검증** | RNGD full-graph 실행 증거 없음 |

Llama 3.2 3B가 실행됐다는 사실과 Furiosa SDK에 exact production preset으로 공식 지원된다는 주장은 다르다. 빌드 로그에 nearest model registry fallback 경고가 있었으므로 이 문서는 조건부 검증으로 기록한다.

## 5분 초기 점검

### 호스트와 장치

```bash
date -Is
cat /etc/os-release
uname -r
lscpu | sed -n '1,30p'
free -h
df -h

lspci -Dnnk | grep -A3 -iE 'FuriosaAI|1ed2:'
lsmod | grep -i furiosa
furiosa-smi info
furiosa-smi status
furiosa-smi ps
dkms status 2>/dev/null | grep -i furiosa || true
```

장치 인식 성공의 핵심 증거는 다음 세 가지다.

```text
Processing accelerators: FuriosaAI RNGD [1ed2:0001]
Kernel driver in use: furiosa_rngd
furiosa-smi info에 rngd / npu0 행 출력
```

`furiosa-smi info`의 전력은 조회 시점의 장치 값이다. 기존 벤치마크 CSV에 자동 저장된 평균 전력이나 실행 에너지를 뜻하지 않는다.

### 실행 환경과 저장소

```bash
command -v furiosa-smi || true
command -v furiosa-llm || true
command -v fxb || true

python --version
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in ("furiosa-llm", "furiosa-torch", "torch", "transformers", "numpy"):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY

git branch --show-current
git rev-parse --short HEAD
git status --short
```

벤더별 가상환경을 사용한다면 `python` 대신 해당 환경의 절대 실행 경로로 다시 확인한다. Git branch, commit, dirty 상태는 결과와 함께 보존한다.

## 오류 문자열 빠른 색인

| 오류 또는 증상 | 바로 갈 항목 |
|---|---|
| `unknown variant Primitive` | [Llama 3.1 artifact schema 불일치](#err-primitive) |
| `Invalid config override key: head_dim` | [Llama 3.2 head_dim](#err-head-dim) |
| `failed printing to stderr: Broken pipe` | [TCC Broken pipe](#err-broken-pipe) |
| `param 'lm_head.weight' not in safetensors index` | [Tied embedding과 lm_head](#err-lm-head) |
| `No module named 'cv2'` | [벤더 전처리 eager import](#err-cv2) |
| `worker_count=8 exceeds runtime capability 1` | [Worker capability](#err-worker-capability) |
| `NativeAsyncBackpressureTimeout` | [Native async backpressure](#err-backpressure) |
| 디버그 로그 뒤에서 멈춘 것처럼 보임 | [Async 진행 상태 확인](#err-async-progress) |
| `eager fallback is not allowed` | [Furiosa Torch 컴파일 실패](#err-furiosa-torch) |
| `align_up_required (true) != false (false)` | [Furiosa Torch 컴파일 실패](#err-furiosa-torch) |
| `EinsumByDpe should be given only a single pass` | [Furiosa Torch 컴파일 실패](#err-furiosa-torch) |

## 운영 Runbook

### 드라이버와 장치 인식

#### DKMS 오류가 있지만 RNGD는 정상 인식되는 경우

**상태: 검증 완료**

다음 오류는 설치되어 있던 다른 벤더의 손상된 DKMS 항목에서 발생할 수 있다.

```text
Error! Could not locate dkms.conf file.
/var/lib/dkms/rebellions-dkms/.../source/dkms.conf does not exist.
```

이 메시지만으로 Furiosa driver 실패를 판정하지 않는다. 실제 검증 서버에서는 위 오류와 함께 다음 상태가 확인됐다.

```text
furiosa-driver-rngd/2026.3.0, 6.8.0-124-generic, x86_64: installed
furiosa_rngd ...
Kernel driver in use: furiosa_rngd
```

다음 순서로 Furiosa 항목을 따로 확인한다.

```bash
uname -r
test -d "/usr/src/linux-headers-$(uname -r)" && echo 'kernel headers: OK'

dkms status 2>/dev/null | grep -i furiosa || true
lsmod | grep -i furiosa
lspci -Dnnk | grep -A3 -iE 'FuriosaAI|1ed2:'
furiosa-smi info
```

패키지를 설치했지만 module이 아직 올라오지 않았고 NPU를 사용하는 프로세스가 없다면 다음처럼 module load를 시도할 수 있다.

```bash
furiosa-smi ps
sudo modprobe furiosa_rngd
```

`modprobe`가 성공하고 `lspci -k`에 driver가 표시되면 즉시 재부팅하지 않고도 점검을 계속할 수 있다. 반대로 running kernel과 빌드된 module이 다르거나 module 교체가 안전하지 않다면 유지보수 시간에 재부팅한다. 사용 중인 RNGD module을 벤치마크 도중 강제로 unload하지 않는다.

### Git worktree와 가상환경 분리

#### Dirty 검증 브랜치에서 main을 직접 pull하지 않는다

**상태: 검증 완료**

서버의 기존 checkout에는 데이터셋 준비 변경과 결과 CSV가 남아 있었다. 이 상태에서 main으로 checkout하거나 무조건 pull하면 사용자 변경과 검증 결과를 섞을 수 있다. 코드와 큰 asset의 위치를 분리한다.

```bash
SOURCE_REPO=/absolute/path/ML-HW-Benchmark-Framework
MAIN_WORKTREE=/absolute/path/ML-HW-Benchmark-Framework-main

FRAMEWORK="$MAIN_WORKTREE/framework"
PY="$SOURCE_REPO/.venv-furiosa/bin/python"
DATASET="$SOURCE_REPO/framework/datasets/squad2/val.json"
MODEL_ROOT="$SOURCE_REPO/framework/models"

git -C "$SOURCE_REPO" branch --show-current
git -C "$SOURCE_REPO" status --short
git -C "$MAIN_WORKTREE" branch --show-current
git -C "$MAIN_WORKTREE" status --short
```

최신 main 코드는 clean worktree에서 실행하고 model, dataset, virtual environment는 원래 checkout의 절대경로로 참조할 수 있다. 현재 작업 디렉터리와 실제 `--model-path`는 독립적이다. 오류 로그의 절대 모델 경로를 먼저 확인한다.

벤더별 SDK 의존성이 다르므로 다음처럼 환경을 분리한다.

```text
.venv-furiosa        Furiosa-LLM 생성 모델
.venv-furiosa-torch  Furiosa Torch 비전 모델
.venv-rebellions     Rebellions runtime
.venv-mobilint       Mobilint runtime
```

한 벤더 환경에 다른 벤더 SDK 전체를 설치하는 방식은 피한다. 프레임워크 공통 의존성도 각 환경에서 필요한 최소 집합만 설치한다.

<a id="err-cv2"></a>

#### `No module named 'cv2'`가 Mobilint 선택 전에 발생

**상태: 원인 확인, 코드 개선 필요**

```text
main.py
→ dataloader
→ preprocess_strategies
→ preprocessor/__init__.py
→ mobilint_vision.py
→ import cv2 실패
```

Furiosa target을 선택했는데도 이 오류가 발생한 이유는 Mobilint 추론이 선택돼서가 아니다. 특정 검증 브랜치의 package `__init__`가 vendor 전처리기를 eager import하면서 Furiosa 실행 전에 OpenCV를 요구했다.

확인은 traceback의 import 순서와 현재 branch로 한다.

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
```

운영 우회는 해당 vendor import가 없는 검증 대상 main 코드에서 실행하는 것이다. 장기 수정은 vendor 전처리기 등록과 import를 lazy/optional하게 바꾸는 것이다. Furiosa LLM 실행만을 위해 관련 없는 OpenCV와 Mobilint SDK를 무조건 설치하면 환경 격리의 의미가 사라진다.

### 모델 디렉터리, legacy artifact, FXB

세 종류의 입력을 구분한다.

| 입력 | 대표 파일 | 역할 |
|---|---|---|
| Hugging Face 모델 디렉터리 | `config.json`, tokenizer, `*.safetensors` | 모델 구조·tokenizer·weights |
| Legacy Furiosa artifact | `artifact.json`, `binary_bundle.zip`, compiled parameter artifact | 이전 Furiosa-LLM 형식의 모델·실행 artifact 묶음 |
| FXB | `*.fxb` | 모델 fingerprint와 RNGD compiled kernels를 담은 실행 bundle |

`fxb download`는 대상 Hugging Face repository에 실제 `.fxb`가 있을 때만 성공한다.

```text
error: repo '...' contains no .fxb file; is this a Furiosa FXB repo?
```

이 오류가 나오면 같은 명령을 반복하지 않는다. Repository의 파일 목록과 tag를 확인하고, legacy artifact repository인지 별도 FXB repository인지 구분한다.

```bash
fxb cache ls
fxb show /absolute/path/to/model.fxb
```

현재 source tree의 Furiosa CLI 계약은 로컬 Hugging Face 디렉터리와 명시적인 `.fxb`를 요구한다. 반면 실장비에서 검증된 이전 경로는 Furiosa repository ID를 받아 호환 legacy revision을 선택했다. 두 경로의 성공을 서로 바꿔 말하지 않는다. 특히 shell 변수가 비어 있는데 다음과 같이 실행하지 않는다.

```text
--fxb ""
```

### Llama 3.1 8B

<a id="err-primitive"></a>

#### `unknown variant Primitive`

**상태: legacy 경로 해결, 현재 explicit-FXB 계약은 별도 검증 필요**

증상은 artifact deserialization 단계에서 다음처럼 발생했다.

```text
unknown variant `Primitive`, expected one of `Decompose`, `Kernelized`,
`PreLower`, `PostLower`, `PreCommandGen`, `CommandGraph`, `TiledGraph`, `Lir`
```

이 오류는 tokenizer, SQuAD 데이터, RNGD driver 또는 현재 작업 디렉터리 문제가 아니다. Traceback에는 실제로 읽은 절대 artifact 경로가 표시됐고 NPU 실행 전 compiler semantic 역직렬화에서 중단됐다. 설치된 Furiosa-LLM 2026.3.0보다 새로운 schema의 legacy artifact snapshot을 로드한 것이 직접 원인이었다.

Artifact metadata를 먼저 기록한다.

```bash
MODEL_DIR=/absolute/path/to/furiosa-artifact

MODEL_DIR="$MODEL_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads(
    (Path(os.environ["MODEL_DIR"]) / "artifact.json").read_text()
).get("metadata", {})

for key in (
    "artifact_id",
    "name",
    "timestamp",
    "furiosa_llm_version",
    "furiosa_compiler_version",
    "includes_composable_ir",
):
    print(f"{key}: {metadata.get(key)}")
PY
```

실장비 legacy 경로에서 검증된 입력은 다음 repository ID였다.

```text
furiosa-ai/Llama-3.1-8B-Instruct
```

이 경로에서는 Furiosa runtime이 설치 버전에 맞는 repository revision을 선택했다. 로컬로 고정해야 한다면 기존 최신 snapshot을 덮어쓰지 말고 호환 tag를 새 디렉터리에 받은 뒤 1건 smoke test로 검증한다.

```bash
FURIOSA_BIN=/absolute/path/to/.venv-furiosa/bin
MODEL31_COMPAT=/absolute/path/to/models/Llama-3.1-8B-Instruct-v2026.3

"$FURIOSA_BIN/hf" download \
  furiosa-ai/Llama-3.1-8B-Instruct \
  --revision v2026.3 \
  --local-dir "$MODEL31_COMPAT"
```

이 수동 tag 경로는 artifact metadata와 실제 load 성공을 다시 확인해야 한다. 현재 main의 local-model + explicit-FXB 계약을 검증하려면 호환 Llama 3.1 FXB를 별도로 확보하거나 빌드해야 한다. Legacy repository ID 성공만으로 current main의 FXB 경로까지 검증됐다고 기록하지 않는다.

### Llama 3.2 3B FXB 빌드와 로딩

#### Exact model registry가 없는 fallback

**상태: 조건부 검증 완료**

Dry run에서 실제 모델 크기는 `hidden_size=3072`, `intermediate_size=8192`였지만 exact registry entry가 없어 `4096/14336` nearest preset을 선택했다는 경고가 출력됐다. 이후 kernel generator 이름에도 Llama 3.1 8B가 나타났다.

이 경고가 있어도 custom FXB는 컴파일·로드·추론까지 완료했다. 그러나 official exact preset 지원이나 production 최적화와 동일하다고 표현하지 않는다. 성능 차이는 모델 크기만이 아니라 fallback preset, bucket, scheduler와 함께 해석한다.

<a id="err-head-dim"></a>

#### `Invalid config override key: head_dim`

**상태: 해결**

Llama 3.2 config에는 `head_dim=128`이 명시되어 있었고, Furiosa 2026.3 config override 경로는 이 key를 허용하지 않았다. 이 값이 실제로 중복인지 확인한 뒤 config 사본에서만 제거했다.

```bash
TEST_MODEL_DIR=/absolute/path/to/llama-3.2-3b-config-test

TEST_MODEL_DIR="$TEST_MODEL_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["TEST_MODEL_DIR"]) / "config.json"
config = json.loads(path.read_text())
explicit = config.get("head_dim")
derived = config["hidden_size"] // config["num_attention_heads"]

print("explicit head_dim:", explicit)
print("derived head_dim:", derived)
if explicit != derived:
    raise RuntimeError(
        f"head_dim is not redundant: explicit={explicit}, derived={derived}"
    )

backup = path.with_name("config.original.json")
if not backup.exists():
    backup.write_text(path.read_text())
config.pop("head_dim")
path.write_text(json.dumps(config, indent=2) + "\n")
print("removed redundant head_dim from config copy")
PY
```

성공 기준은 같은 config 사본을 사용한 `fxb build ... --dry-run`이 bucket summary를 출력하고 `Invalid config override key` 없이 끝나는 것이다.

#### Final EDF 단계의 AArch64 cross compiler 누락

**상태: 해결**

Kernel code 생성 이후 최종 EDF 빌드에서 `aarch64-linux-gnu-gcc`가 필요했다. Host의 `gcc`, `ld`, `objcopy` 존재 여부만 확인해서는 충분하지 않았다.

```bash
for command in aarch64-linux-gnu-gcc aarch64-linux-gnu-ld aarch64-linux-gnu-objcopy; do
  command -v "$command" || echo "MISSING: $command"
done
```

Ubuntu 검증 서버에서는 승인된 system package workflow로 다음 패키지를 설치해 해결했다.

```bash
sudo apt update
sudo apt install -y gcc-aarch64-linux-gnu
```

<a id="err-broken-pipe"></a>

#### `Compilation failed: failed printing to stderr: Broken pipe`

**상태: 원인 추적 후 해결**

`Broken pipe`는 가장 먼저 발생한 원인이 아니라 TCC 하위 프로세스가 실패한 뒤 stderr 전달 과정에서 보인 2차 오류였다. 모델 config를 다시 바꾸기 전에 실행된 프로세스와 최초 실패를 확인했다.

```bash
TRACE_LOG=/tmp/fxb-build.trace

strace -f -e trace=process,file -o "$TRACE_LOG" \
  /absolute/path/to/fxb build \
  /absolute/path/to/model \
  /absolute/path/to/output.fxb \
  -tp 8 -O O0 --max-model-len 4096 --concurrency 1

grep -n 'execve(' "$TRACE_LOG" | tail -n 50
grep -n 'aarch64-linux-gnu' "$TRACE_LOG" | tail -n 50
```

Cross compiler를 설치한 뒤에도 즉시 같은 짧은 실패가 반복되면 이전 실패가 compiler cache에 남았는지 확인한다. Cache 전체를 지우지 말고 실제 실패 key에 해당하는 단일 entry를 먼저 찾는다. 이동 전에는 반드시 대상이 한 entry인지 출력한다.

```bash
CACHE_ENTRY=/absolute/path/to/one-resolved-negative-cache-entry
test -e "$CACHE_ENTRY"
printf 'cache entry: %s\n' "$CACHE_ENTRY"

CACHE_BACKUP="${CACHE_ENTRY}.backup-$(date +%Y%m%d-%H%M%S)"
mv -- "$CACHE_ENTRY" "$CACHE_BACKUP"
printf 'recoverable backup: %s\n' "$CACHE_BACKUP"
```

`CACHE_ENTRY`가 구체적인 단일 경로로 확인되지 않았다면 이동하지 않는다. 검증 빌드는 `--concurrency 1`과 `--build-report`로 최초 실패가 보이게 실행한다.

```bash
FURIOSA_BIN=/absolute/path/to/.venv-furiosa/bin
MODEL32_BUILD=/absolute/path/to/llama-3.2-3b-build-source
FXB32=/absolute/path/to/llama-3.2-3b.fxb

"$FURIOSA_BIN/fxb" build \
  "$MODEL32_BUILD" \
  "$FXB32" \
  -tp 8 \
  -O O0 \
  --max-model-len 4096 \
  --concurrency 1 \
  --build-report
```

검증 결과는 9 kernels succeeded, 0 failed, 총 약 15분 21초였다.

<a id="err-lm-head"></a>

#### `param 'lm_head.weight' not in safetensors index`

**상태: 해결**

Llama 3.2 config는 `tie_word_embeddings=true`였고 원본 safetensors index에는 `model.embed_tokens.weight`만 있으며 별도 `lm_head.weight`가 없었다. Custom FXB의 fallback runtime 경로는 명시적인 `lm_head.weight` 이름을 요구했다.

```bash
MODEL_SRC=/absolute/path/to/meta-llama_Llama-3.2-3B-Instruct

MODEL_SRC="$MODEL_SRC" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["MODEL_SRC"])
config = json.loads((root / "config.json").read_text())
index = json.loads((root / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]

print("tie_word_embeddings:", config.get("tie_word_embeddings"))
print("embed_tokens:", weight_map.get("model.embed_tokens.weight"))
print("lm_head:", weight_map.get("lm_head.weight"))
PY
```

원본은 변경하지 않고 derived directory를 만들었다. 다른 파일은 원본을 참조하고, tied embedding tensor를 동일 값의 별도 shard로 저장한 뒤 복사한 index에 `lm_head.weight` mapping을 추가한다.

```bash
MODEL_SRC=/absolute/path/to/meta-llama_Llama-3.2-3B-Instruct
MODEL_DST=/absolute/path/to/meta-llama_Llama-3.2-3B-Instruct-furiosa

MODEL_SRC="$MODEL_SRC" MODEL_DST="$MODEL_DST" python3 - <<'PY'
import json
import os
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

src = Path(os.environ["MODEL_SRC"]).resolve()
dst = Path(os.environ["MODEL_DST"]).resolve()
if dst.exists():
    raise RuntimeError(f"destination already exists: {dst}")
dst.mkdir(parents=True)
(dst / "original").symlink_to(src, target_is_directory=True)

index_path = src / "model.safetensors.index.json"
index = json.loads(index_path.read_text())
weight_map = index["weight_map"]
source_shard = weight_map.get("model.embed_tokens.weight")
if source_shard is None:
    raise RuntimeError("model.embed_tokens.weight is missing")
if weight_map.get("lm_head.weight") is not None:
    raise RuntimeError("lm_head.weight already exists")

for path in src.iterdir():
    if path.name == "model.safetensors.index.json":
        continue
    (dst / path.name).symlink_to(path)

with safe_open(src / source_shard, framework="pt", device="cpu") as handle:
    lm_head = handle.get_tensor("model.embed_tokens.weight").clone()

new_shard = "model-lm-head.safetensors"
save_file({"lm_head.weight": lm_head}, dst / new_shard)
weight_map["lm_head.weight"] = new_shard

metadata = index.setdefault("metadata", {})
metadata["total_size"] = int(metadata.get("total_size", 0)) + (
    lm_head.numel() * lm_head.element_size()
)
(dst / "model.safetensors.index.original.json").write_text(
    index_path.read_text()
)
(dst / "model.safetensors.index.json").write_text(
    json.dumps(index, indent=2) + "\n"
)
print("created derived model:", dst)
PY
```

이 우회는 tensor 값을 바꾸지 않지만 약 0.75 GiB의 중복 weight storage를 추가한다. 성공 기준은 새 index가 `lm_head.weight → model-lm-head.safetensors`를 가리키고 Furiosa runtime이 parameter load를 통과하는 것이다.

### 프레임워크 LLM 입력 계약

#### Pre-tokenized 입력과 `attention_mask`

**상태: 해결**

프레임워크 Llama loader는 `input_ids`와 `attention_mask`를 함께 생성한다. 초기 Furiosa adapter는 pre-tokenized `BatchEncoding` 경계에서 mask 의미를 보존하지 못해 padded prompt를 올바르게 구성하지 못했다.

현재 adapter는 다음을 수행한다.

1. `input_ids`를 2차원 배열로 정규화한다.
2. `attention_mask`가 있으면 shape가 정확히 같은지 검증한다.
3. Mask가 참인 token만 남겨 prompt token ID 목록을 만든다.
4. Trim된 ID로 Furiosa pre-tokenized request를 생성한다.

관련 경계는 `framework/src/runtimes/furiosa_llm_rt.py`의 `_trim_prompt_tokens()`이며, shape mismatch와 padding trim은 `framework/tests/test_furiosa_llm_runtime.py`로 회귀 검증한다. `attention_mask`가 없을 때 모든 ID를 실제 prompt로 간주하는 fallback과, mask가 있을 때 padding을 제거하는 경로를 구분한다.

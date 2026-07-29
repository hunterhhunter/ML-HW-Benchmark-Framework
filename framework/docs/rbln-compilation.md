# Rebellions RBLN-CA22 모델 컴파일 재현 가이드

이 문서는 `RBLN-CA22`용 artifact를 벤치마크 실행과 분리해서 만들고,
검사하고, 프레임워크에 전달하는 기준 절차다. 대상은 ResNet50, YOLOv5m,
BERT SST-2, BERT SQuAD, PatchTST ETTh1, Llama 3.2 3B, Llama 3.1 8B의
일곱 모델이다.

정적 모델은 저장소의 다섯 recipe module을 호출한다. Llama는 기존
`tools/prepare_rbln_vllm_model.py`를 호출한다. 이 문서는 recipe의 Python
구현을 복사한 별도 compiler가 아니며, `rbln-static` runtime도 실행 중에
자동 컴파일하지 않는다.

## 1. 범위와 안전 규칙

- 모든 recipe는 device target `RBLN-CA22`, fixed shape, request batch 1을
  전제로 한다.
- `.rbln` 정적 artifact와 Llama 준비 디렉터리는 기존 경로를 덮어쓰지 않는다.
  재컴파일할 때는 새 출력 경로를 사용한다.
- 모델 weight, `.rbln`, tokenizer, dataset, cache, 결과 CSV, trace, 로그와
  인증 정보는 Git에 추가하지 않는다.
- compile과 benchmark는 서로 다른 Python 환경일 수 있다. 실행한 interpreter,
  package version과 package origin을 결과와 함께 남긴다.
- Hugging Face gated model과 Rebellions package index 접근 권한은 운영자가
  서버에서 설정한다. 토큰이나 비밀번호를 repository, pip URL, shell script,
  terminal 캡처에 기록하지 않는다.
- 한 장 ATOM에서 두 Llama를 준비하는 경로는
  `unsupported_single_npu_experiment`다. 한 장 실행 성공 이력은 Rebellions의
  공식 지원 범위를 바꾸지 않는다.

Runtime, E2E/async, monitor 계약은 [RBLN-CA22 운영 가이드](rbln-setup.md)와
[RBLN vLLM 실행 가이드](rbln-vllm-setup.md)를 따른다. 실제 오류와 검증값은
[트러블슈팅 기록](rbln-troubleshooting.md), Llama 물리 run ID는
[단일 ATOM 검증 보고서](rbln-vllm-atom-validation.md)에 있다.

## 2. 경로와 환경 고정

검증 서버의 디렉터리 배치를 따르는 예다. 네 변수는 shell마다 다시
확인한다.

```bash
export RBLN_FW_ROOT="$HOME/ML-HW-Benchmark-Framework-rbln-vllm"
export RBLN_ZOO_ROOT="$HOME/rebelion/rbln-model-zoo"
export RBLN_BUILD_PY="$RBLN_ZOO_ROOT/.venv-rbln-zoo/bin/python"
export RBLN_VLLM_PY="$HOME/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python"

export RBLN_RUN_PY="$RBLN_VLLM_PY"
export RBLN_BUILD_ROOT="$RBLN_ZOO_ROOT/custom/framework-contracts"
```

`RBLN_BUILD_PY`는 정적 Torch/Hugging Face 모델 컴파일에, `RBLN_VLLM_PY`는
Optimum/vLLM Llama 준비와 Llama benchmark에 사용한다. 검증 서버처럼
`rebel`만 Python 3.10 user-site에 있는 hybrid 환경에서는
`PYTHONNOUSERSITE=1`을 설정하지 않는다. 새 환경은 권한 있는 RBLN wheel로
구성하고, 무관한 Python ABI의 site-packages를 복사하지 않는다.

### 2.1 OS, 장치, Python package preflight

```bash
cat /etc/os-release
"$RBLN_BUILD_PY" --version
"$RBLN_VLLM_PY" --version
command -v rbln-smi
rbln-smi -q
rbln-smi -j
ls -l /dev/rbln0
```

두 interpreter의 package origin을 따로 기록한다.

```bash
for py in "$RBLN_BUILD_PY" "$RBLN_VLLM_PY"; do
  "$py" - <<'PY'
import importlib.metadata as md
import importlib.util
import sys

print("python:", sys.executable)
for package in (
    "rebel-compiler",
    "optimum-rbln",
    "vllm-rbln",
    "vllm",
    "torch",
    "transformers",
    "tokenizers",
):
    try:
        dist = md.distribution(package)
    except md.PackageNotFoundError:
        print(package, "NOT INSTALLED")
    else:
        print(package, dist.version, dist.locate_file(""))
spec = importlib.util.find_spec("rebel")
print("rebel module:", spec.origin if spec else "NOT FOUND")
PY
done
```

Compiler와 장치 API를 확인한다. 기준 서버에서 검증한 조합은 Ubuntu 22.04.5,
Python 3.10.12, `rebel-compiler==0.11.0`, KMD/firmware 3.2.2였다. 이 값은
재현 기준이지 recipe가 임의로 upgrade하거나 strict equality를 강제할 값은 아니다.

```bash
"$RBLN_BUILD_PY" - <<'PY'
import rebel

print("available:", rebel.npu_is_available(0))
print("name:", rebel.get_npu_name(0))
print("count:", rebel.device_count())
print("RBLNCompiledModel:", hasattr(rebel, "RBLNCompiledModel"))
print("Runtime:", hasattr(rebel, "Runtime"))
print("AsyncRuntime:", hasattr(rebel, "AsyncRuntime"))
assert rebel.npu_is_available(0)
assert rebel.get_npu_name(0) == "RBLN-CA22"
PY
```

## 3. Model Zoo와 모델 접근 준비

Model Zoo가 없다면 별도 디렉터리에 clone한다. 기존 clone이 있으면 remote와
현재 commit을 기록하고 재사용한다.

```bash
test ! -e "$RBLN_ZOO_ROOT"
git clone --recursive \
  https://github.com/rebellions-sw/rbln-model-zoo.git \
  "$RBLN_ZOO_ROOT"

git -C "$RBLN_ZOO_ROOT" remote -v
git -C "$RBLN_ZOO_ROOT" rev-parse HEAD
```

정적 recipe 자체는 framework repository에 있으므로 모든 module command는
`$RBLN_FW_ROOT/framework`에서 실행한다. Model Zoo 아래
`custom/framework-contracts`는 Git에 넣지 않는 build output 영역으로만 쓴다.

Hugging Face 공개 모델은 anonymous rate limit에 걸릴 수 있다. Meta Llama와
ImageNet 같은 gated resource는 해당 약관 승인과 Hub 로그인이 먼저다.

```bash
"$(dirname "$RBLN_VLLM_PY")/hf" auth login
```

Rebellions package index의 `401 Unauthorized`는 일반 PyPI 로그인이 아니라
Portal 조직 권한 또는 조직이 배포한 pip 인증 설정 문제다. 권한이 없으면
Portal 관리자나 Rebellions 지원 창구에 요청한다. credential을 URL에 직접
붙여 해결하지 않는다.

## 4. Recipe 공통 사용법과 artifact handoff

다섯 정적 module은 optional SDK를 import하지 않고 `--help`와 `--describe`를
실행할 수 있다. 먼저 interface와 JSON ABI를 확인한다.

```bash
cd "$RBLN_FW_ROOT/framework"

for module in \
  tools.rbln_compile_recipes.resnet50.compile \
  tools.rbln_compile_recipes.yolov5m.compile \
  tools.rbln_compile_recipes.bert_sst2.compile \
  tools.rbln_compile_recipes.bert_squad.compile \
  tools.rbln_compile_recipes.patchtst_etth1.compile
do
  "$RBLN_BUILD_PY" -m "$module" --help >/dev/null
  "$RBLN_BUILD_PY" -m "$module" --describe \
    | "$RBLN_BUILD_PY" -m json.tool
done
```

`--output`은 명시적인 새 `.rbln` 파일이어야 한다. 기존 파일, 확장자가 다른
파일, batch 1이 아닌 계약, CA22가 아닌 artifact, inspect ABI 불일치는 recipe가
거부한다. 성공 시 recipe는 output path, compiler version, byte size, SHA256과
계약 JSON을 출력한다.

### 4.1 독립 inspect와 hash 보존

다음 함수는 SDK 0.11의 mapping/attribute 두 inspect 표현을 모두 처리한다.
각 모델 절에서 source artifact에 실행한다.

```bash
inspect_rbln() {
  ARTIFACT="$1" "$RBLN_BUILD_PY" - <<'PY'
import json
import os
from collections.abc import Mapping
from rebel import RBLNCompiledModel

metadata = RBLNCompiledModel.inspect(os.environ["ARTIFACT"])

def field(value, name):
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)

def tensor(value):
    return {
        "name": field(value, "name"),
        "shape": list(field(value, "shape")),
        "dtype": str(field(value, "dtype")),
    }

print(json.dumps({
    "compiler_version": field(metadata, "compiler_version"),
    "npu": field(metadata, "npu"),
    "tensor_parallel_size": field(metadata, "tensor_parallel_size"),
    "uuid": field(metadata, "uuid"),
    "inputs": [tensor(item) for item in field(metadata, "inputs")],
    "outputs": [tensor(item) for item in field(metadata, "outputs")],
}, indent=2, default=str))
PY
}

copy_verified() {
  (
    if [ "$#" -ne 2 ]; then
      printf 'usage: copy_verified SOURCE DESTINATION\n' >&2
      exit 2
    fi
    source_path="$1"
    destination_path="$2"
    temp_path=""

    cleanup_copy_verified() {
      if [ -n "$temp_path" ] && [ -e "$temp_path" ]; then
        rm -f -- "$temp_path"
      fi
    }
    trap cleanup_copy_verified EXIT
    trap 'exit 129' HUP
    trap 'exit 130' INT
    trap 'exit 143' TERM

    if [ ! -s "$source_path" ]; then
      printf 'source is missing or empty: %s\n' "$source_path" >&2
      exit 1
    fi
    if [ -e "$destination_path" ]; then
      printf 'destination already exists: %s\n' "$destination_path" >&2
      exit 1
    fi
    destination_dir="$(dirname -- "$destination_path")" || exit 1
    destination_name="$(basename -- "$destination_path")" || exit 1
    if ! mkdir -p -- "$destination_dir"; then
      printf 'could not create destination directory: %s\n' \
        "$destination_dir" >&2
      exit 1
    fi
    temp_path="$(
      mktemp --tmpdir="$destination_dir" \
        ".${destination_name}.copy.XXXXXXXX"
    )" || exit 1
    if ! cp -- "$source_path" "$temp_path"; then
      printf 'artifact temporary copy failed: %s\n' "$temp_path" >&2
      exit 1
    fi
    if [ ! -s "$temp_path" ]; then
      printf 'artifact temporary copy is empty: %s\n' "$temp_path" >&2
      exit 1
    fi
    source_sha="$(sha256sum "$source_path" | awk '{print $1}')" || exit 1
    temp_sha="$(sha256sum "$temp_path" | awk '{print $1}')" || exit 1
    if [ "$source_sha" != "$temp_sha" ]; then
      printf 'artifact SHA256 mismatch before publish\n' >&2
      exit 1
    fi

    # temp와 destination은 같은 filesystem에 있다. link(2)는 destination이
    # 이미 있으면 EEXIST로 실패하므로 preflight 이후 race도 덮어쓰지 않는다.
    if ! ln -- "$temp_path" "$destination_path"; then
      printf 'destination appeared before atomic publish: %s\n' \
        "$destination_path" >&2
      exit 1
    fi
    printf 'source      %s\n' "$source_sha"
    printf 'destination %s\n' "$temp_sha"
    exit 0
  )
}
```

Source와 destination hash가 같아야 handoff가 끝난다. 기존 framework artifact를
교체해야 할 때도 위 함수로 덮어쓰지 않는다. 새 이름으로 검증한 뒤 운영자의
명시적인 배포 절차에서 교체한다.

### 4.2 다섯 정적 ABI

| Framework profile | Model provenance | Input ABI | Output ABI |
|---|---|---|---|
| `resnet50` | TorchVision ResNet50 `IMAGENET1K_V2` | `input_np float32 (1,3,224,224)` | `output float32 (1,1000)`; 단일 unnamed 허용 |
| `yolov5m` | Ultralytics YOLOv5m, source commit `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51` | `input_np float32 (1,3,640,640)` | raw head `output float32 (1,25200,85)`; 단일 unnamed 허용 |
| `bert-base-uncased` | `textattack/bert-base-uncased-SST-2` | `input_ids`, `attention_mask`: 각각 `int64 (1,128)` | `logits float32 (1,2)` |
| `bert-base-uncased-squad-v1` | `csarron/bert-base-uncased-squad-v1` | `input_ids`, `attention_mask`, `token_type_ids`: 각각 `int64 (1,384)` | 위치 0 start, 위치 1 end, 각각 `float32 (1,384)`; unnamed 허용 |
| `patchtst-fm-r1` | `ibm-granite/granite-timeseries-patchtst` | `past_values float32 (1,512,7)`, `past_observed_mask bool (1,512,7)` | `prediction_outputs float32 (1,96,7)`; unnamed 허용 |

## 5. ResNet50

TorchVision이 `IMAGENET1K_V2` weight를 받아 fixed NCHW artifact를 만든다.

```bash
export RBLN_RESNET_BUILD="$RBLN_BUILD_ROOT/resnet50-ca22-b1"
export RBLN_RESNET_SOURCE="$RBLN_RESNET_BUILD/resnet50.rbln"
test ! -e "$RBLN_RESNET_SOURCE"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.resnet50.compile --describe \
  | "$RBLN_BUILD_PY" -m json.tool
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.resnet50.compile \
  --output "$RBLN_RESNET_SOURCE"

inspect_rbln "$RBLN_RESNET_SOURCE"
sha256sum "$RBLN_RESNET_SOURCE"
copy_verified "$RBLN_RESNET_SOURCE" \
  "$RBLN_FW_ROOT/framework/models/rbln/resnet50/model.rbln"
```

Dataset은 `"$RBLN_BUILD_PY" datasets/prepare_imagenet_1k.py`로 준비하거나 이미
검증한 `datasets/imagenet_1k`를 읽기 전용으로 연결한다. Sync smoke는
[운영 가이드 6절](rbln-setup.md#6-sync-e2e-smoke와-full-run)의 10-sample
명령을 사용한다.

컴파일 성공 뒤 `cp: cannot stat`이 나오면 compiler 문제가 아니라 현재
디렉터리와 output path가 다른 것이다. 위 절대 변수로 source를 확인한다.
10-sample smoke의 utilization 0은 짧은 poll 구간일 수 있으므로, 성능 판정은
3,000-sample full run과 monitor sample 수를 함께 본다.

## 6. YOLOv5m raw head

Recipe는 NMS/AutoShape를 포함하지 않는 raw prediction head만 만든다. Framework의
기존 YOLO decoder가 기대하는 `(1,25200,85)`와 다르면 사용하지 않는다.

Model Zoo submodule과 weight를 먼저 준비한다. Weight는 검증한 Ultralytics
YOLOv5m checkpoint여야 하며 출처와 SHA256을 별도 기록한다.

```bash
cd "$RBLN_ZOO_ROOT"
git submodule update --init --recursive -- \
  pytorch/vision/detection/yolov5/yolov5

export RBLN_YOLO_ROOT="$RBLN_ZOO_ROOT/pytorch/vision/detection/yolov5/yolov5"
export RBLN_YOLO_WEIGHTS="$RBLN_ZOO_ROOT/pytorch/vision/detection/yolov5/yolov5m.pt"
git -C "$RBLN_YOLO_ROOT" checkout --detach \
  86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51
test "$(git -C "$RBLN_YOLO_ROOT" rev-parse HEAD)" = \
  86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51
if [ ! -e "$RBLN_YOLO_WEIGHTS" ]; then
  curl --fail --location \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m.pt \
    --output "$RBLN_YOLO_WEIGHTS"
fi
test -s "$RBLN_YOLO_WEIGHTS"
sha256sum "$RBLN_YOLO_WEIGHTS"
```

```bash
export RBLN_YOLO_BUILD="$RBLN_BUILD_ROOT/yolov5m-ca22-b1"
export RBLN_YOLO_SOURCE="$RBLN_YOLO_BUILD/yolov5m.rbln"
test ! -e "$RBLN_YOLO_SOURCE"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.yolov5m.compile --describe \
  | "$RBLN_BUILD_PY" -m json.tool
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.yolov5m.compile \
  --yolov5-root "$RBLN_YOLO_ROOT" \
  --weights "$RBLN_YOLO_WEIGHTS" \
  --output "$RBLN_YOLO_SOURCE"

inspect_rbln "$RBLN_YOLO_SOURCE"
sha256sum "$RBLN_YOLO_SOURCE"
copy_verified "$RBLN_YOLO_SOURCE" \
  "$RBLN_FW_ROOT/framework/models/rbln/yolov5m/model.rbln"
```

Dataset은 `"$RBLN_BUILD_PY" datasets/prepare_coco128.py`로 준비한다. Sync smoke는
`--model yolov5m --target rbln-static --artifact models/rbln/yolov5m/model.rbln
--dataset datasets/coco128 --batch-size 1 --warmup 2 --max-steps 10 --monitor`로
실행하고, 공통 판정은 [운영 가이드 6절](rbln-setup.md#6-sync-e2e-smoke와-full-run)을
따른다.

Source submodule이 비었으면 `git submodule update`를 다시 실행한다. Commit이
다르면 recipe가 checkout 명령을 포함한 오류로 중단한다. Weight가 없거나 0 byte여도
compile 전에 중단한다.

## 7. BERT SST-2

기본 checkpoint는 `textattack/bert-base-uncased-SST-2`다. 다른
`--model-id`를 지정할 수 있지만 output class 수와 ABI가 동일한지는 recipe의
inspect gate가 확인해야 한다.

```bash
export RBLN_BERT_SST2_BUILD="$RBLN_BUILD_ROOT/bert-sst2-ca22-b1-seq128"
export RBLN_BERT_SST2_SOURCE="$RBLN_BERT_SST2_BUILD/bert-sst2.rbln"
test ! -e "$RBLN_BERT_SST2_SOURCE"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.bert_sst2.compile --describe \
  | "$RBLN_BUILD_PY" -m json.tool
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.bert_sst2.compile \
  --model-id textattack/bert-base-uncased-SST-2 \
  --output "$RBLN_BERT_SST2_SOURCE"

inspect_rbln "$RBLN_BERT_SST2_SOURCE"
sha256sum "$RBLN_BERT_SST2_SOURCE"
copy_verified "$RBLN_BERT_SST2_SOURCE" \
  "$RBLN_FW_ROOT/framework/models/rbln/bert-base-uncased/model.rbln"
```

SST-2 NumPy 데이터는 namespaced dataset ID를 사용해 준비한다.

```bash
"$RBLN_BUILD_PY" datasets/prepare_text_numpy.py \
  --model-id textattack/bert-base-uncased-SST-2 \
  --seq-len 128 \
  --dataset-name nyu-mll/glue \
  --dataset-config sst2 \
  --split validation \
  --output-dir datasets/sst2_numpy
```

Sync smoke는 `--model bert-base-uncased --target rbln-static --artifact
models/rbln/bert-base-uncased/model.rbln --dataset datasets/sst2_numpy
--batch-size 1 --warmup 2 --max-steps 10 --monitor`로 실행한다. 일반 검증 순서는
[운영 가이드](rbln-setup.md#6-sync-e2e-smoke와-full-run)를 따른다.

`glue`가 `Invalid HF URI`로 실패한 검증 환경에서는 `nyu-mll/glue`로 해결했다.
이는 모델 compiler 오류가 아니다. 생성된 `input_ids.npy`,
`attention_mask.npy`가 `(872,128)` int64이고 label 수가 872인지 확인한다.

## 8. BERT SQuAD 3-input

기본 checkpoint는 `csarron/bert-base-uncased-squad-v1`다. 이 recipe는
`token_type_ids`까지 포함한 세 입력과 tuple 순서
`output[0]=start_logits`, `output[1]=end_logits`를 만든다.

```bash
export RBLN_BERT_SQUAD_BUILD="$RBLN_BUILD_ROOT/bert-squad-ca22-b1-seq384"
export RBLN_BERT_SQUAD_SOURCE="$RBLN_BERT_SQUAD_BUILD/bert-squad.rbln"
test ! -e "$RBLN_BERT_SQUAD_SOURCE"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.bert_squad.compile --describe \
  | "$RBLN_BUILD_PY" -m json.tool
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.bert_squad.compile \
  --model-id csarron/bert-base-uncased-squad-v1 \
  --output "$RBLN_BERT_SQUAD_SOURCE"

inspect_rbln "$RBLN_BERT_SQUAD_SOURCE"
sha256sum "$RBLN_BERT_SQUAD_SOURCE"
copy_verified "$RBLN_BERT_SQUAD_SOURCE" \
  "$RBLN_FW_ROOT/framework/models/rbln/bert-base-uncased-squad-v1/model.rbln"
```

SDK 0.11은 두 output name을 모두 `null`로 저장할 수 있다. 같은 shape 두 개만
보고 의미를 정하면 안 된다. 최종 배포 byte에 대해
[실제 CPU/NPU mapping script와 sidecar 생성 절차](rbln-setup.md#51-bert-squad-unnamed-output-검증과-sidecar)를
실행한다. 과거 물리 검증에서는 direct assignment MAE 합이 0.667688로 swapped
0.981693보다 작았고, context start/end argmax가 CPU와 NPU 모두 11/15였으며
decode된 답도 `neural processing unit inference performance`로 같았다.

새 artifact의 합격 gate는 다음 증거를 함께 요구한다.

1. 세 input의 이름·순서·shape·dtype과 두 output shape를 inspect한다.
2. Direct mapping과 swapped mapping의 MAE/RMSE/correlation을 모두 기록한다.
3. 실제 context token만 대상으로 start/end argmax와 best span을 비교한다.
4. CPU/NPU가 같은 answer text를 선택하는지 확인한다.
5. real `token_type_ids`와 all-zero 입력의 NPU output 차이를 확인한다.

ATOM compiled precision에서는 전체 384 logit의 strict
`rtol=atol=1e-3` all-element allclose가 실패했지만 올바른 span을 선택한 이력이
있다. 따라서 strict allclose 하나만을 output 순서나 semantic parity의 유일한
판정으로 사용하지 않는다. 수치 차이와 아직 남은 task-level 위험은
[트러블슈팅 7.2와 7.4](rbln-troubleshooting.md#72-두-output-이름이-null)에
기록되어 있다.

Mapping을 통과한 최종 artifact에만 `model.rbln.json`을 만들며, artifact가
바뀌면 mapping과 sidecar를 모두 다시 만든다. Source에서 만든 sidecar를 hash
확인 없이 destination에 복사하지 않는다.

```bash
"$RBLN_BUILD_PY" datasets/prepare_squad_numpy.py \
  --model-id csarron/bert-base-uncased-squad-v1 \
  --seq-len 384 \
  --dataset-name rajpurkar/squad \
  --split validation \
  --output-dir datasets/squad_numpy
```

Sync smoke 명령과 sidecar binding 확인은
[운영 가이드 6절](rbln-setup.md#6-sync-e2e-smoke와-full-run)에 있다. 현재 기록은
단일 샘플 semantic mapping을 증명하지만 context-masked evaluator와 전체 SQuAD
task-level validation은 별도 acceptance gate로 남아 있다.

## 9. PatchTST ETTh1

기본 checkpoint는 `ibm-granite/granite-timeseries-patchtst`다. Recipe는
원본 PatchTST patchifier의 unsupported `aten::unfold`를 fixed 42-patch
stack으로 교체하고 CPU equivalence를 확인한다. 외부 ABI의 bool mask는 유지하되
모델 계산 전에 float32로 변환하고 bool/float CPU equivalence도 확인한다.

```bash
export RBLN_PATCHTST_BUILD="$RBLN_BUILD_ROOT/patchtst-etth1-ca22-b1"
export RBLN_PATCHTST_SOURCE="$RBLN_PATCHTST_BUILD/patchtst-etth1.rbln"
test ! -e "$RBLN_PATCHTST_SOURCE"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.patchtst_etth1.compile --describe \
  | "$RBLN_BUILD_PY" -m json.tool
"$RBLN_BUILD_PY" -m tools.rbln_compile_recipes.patchtst_etth1.compile \
  --model-id ibm-granite/granite-timeseries-patchtst \
  --output "$RBLN_PATCHTST_SOURCE"

inspect_rbln "$RBLN_PATCHTST_SOURCE"
sha256sum "$RBLN_PATCHTST_SOURCE"
copy_verified "$RBLN_PATCHTST_SOURCE" \
  "$RBLN_FW_ROOT/framework/models/rbln/patchtst-fm-r1/model.rbln"
```

Dataset은 `"$RBLN_BUILD_PY" datasets/prepare_etth1.py --output-dir
datasets/etth1`로 준비한다. Sync smoke는 `--model patchtst-fm-r1 --target
rbln-static --artifact models/rbln/patchtst-fm-r1/model.rbln --dataset
datasets/etth1/ETTh1.csv --batch-size 1 --warmup 2 --max-steps 10 --monitor`로
실행한다.

첫 compile의 `aten::unfold` 미지원과 static rewrite 뒤 bool `clamp_min`
lowering 오류는 현재 recipe의 static patchifier와 내부 mask cast로 해결한다.
Recipe가 출력하는 두 CPU equivalence gate 또는 trace의 `aten::unfold` 제거가
실패하면 artifact를 배포하지 않는다.

## 10. Llama 3.2 3B 한 장 준비

모델 ID는 `meta-llama/Llama-3.2-3B-Instruct`다. Output directory 전체가 하나의
artifact이며 기존 디렉터리를 덮어쓰지 않는다.

```bash
export RBLN_LLAMA32_DIR="$RBLN_BUILD_ROOT/llama-3.2-3b-npu1-seq512"
test ! -e "$RBLN_LLAMA32_DIR"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.2-3b --output-dir "$RBLN_LLAMA32_DIR" \
  --num-devices 1 --max-seq-len 512 --block-size 512 \
  --batch-size 1 --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

준비 디렉터리에는 tokenizer/config, `rbln-vllm-manifest.json`, prefill과 decoder
`.rbln`을 함께 둔다. 일부 파일만 framework 쪽으로 복사하지 않는다.

```bash
test -f "$RBLN_LLAMA32_DIR/config.json"
test -f "$RBLN_LLAMA32_DIR/tokenizer_config.json"
test -f "$RBLN_LLAMA32_DIR/rbln-vllm-manifest.json"
test -f "$RBLN_LLAMA32_DIR/prefill.rbln"
test -f "$RBLN_LLAMA32_DIR/decoder_batch_1.rbln"
find "$RBLN_LLAMA32_DIR" -type f -name '*.rbln' -ls
"$RBLN_VLLM_PY" -m json.tool "$RBLN_LLAMA32_DIR/rbln-vllm-manifest.json"
sha256sum "$RBLN_LLAMA32_DIR"/*.rbln
```

검증 서버에서 한 번 관찰한 3B 파일 크기는 `prefill.rbln`
7,238,844,846 bytes, `decoder_batch_1.rbln` 806,195,660 bytes였다. 이는 해당
SDK/model/contract build의 historical evidence일 뿐 새 빌드의 보편적인 예상 크기나
합격 조건이 아니다. Manifest의 file hash와 실제 파일을 검증한다.

물리 실행 증거는 sync run `b7808504`, async run `a307b84f`이며 자세한 판정은
[검증 보고서](rbln-vllm-atom-validation.md#llama-32-3b-물리-검증-증거)에 있다.

## 11. Llama 3.1 8B 한 장 준비

모델 ID는 `meta-llama/Llama-3.1-8B-Instruct`다. 한 장 ATOM에서 성공한 이력이
있어도 공식 지원 구성이 아니다. 충분한 host RAM과 저장 공간, 빈 NPU context를
확인하고 `tmux`에서 실행한다.

```bash
export RBLN_LLAMA31_DIR="$RBLN_BUILD_ROOT/llama-3.1-8b-npu1-seq512"
df -BG "$RBLN_ZOO_ROOT"
free -h
rbln-smi -j
test ! -e "$RBLN_LLAMA31_DIR"

cd "$RBLN_FW_ROOT/framework"
"$RBLN_VLLM_PY" tools/prepare_rbln_vllm_model.py \
  --model llama-3.1-8b --output-dir "$RBLN_LLAMA31_DIR" \
  --num-devices 1 --max-seq-len 512 --block-size 512 \
  --batch-size 1 --decoder-batch-sizes 1 \
  --allow-unsupported-single-npu
```

3B와 마찬가지로 tokenizer/config, manifest, prefill, decoder artifact를 한
디렉터리로 유지하고 manifest의 file별 SHA256을 확인한다.

```bash
test -f "$RBLN_LLAMA31_DIR/config.json"
test -f "$RBLN_LLAMA31_DIR/tokenizer_config.json"
test -f "$RBLN_LLAMA31_DIR/rbln-vllm-manifest.json"
test -f "$RBLN_LLAMA31_DIR/prefill.rbln"
test -f "$RBLN_LLAMA31_DIR/decoder_batch_1.rbln"
find "$RBLN_LLAMA31_DIR" -type f -name '*.rbln' -ls
"$RBLN_VLLM_PY" -m json.tool "$RBLN_LLAMA31_DIR/rbln-vllm-manifest.json"
sha256sum "$RBLN_LLAMA31_DIR"/*.rbln
```

물리 실행 증거는 sync run `a3168997`, async run `9dd3bf7a`다. 두 run은
engine allocation과 cleanup을 증명하지만 8B가 생성 토큰 하나만 사용했으므로
TPOT이나 모델 품질 증거가 아니다. 자세한 값은
[검증 보고서](rbln-vllm-atom-validation.md#llama-31-8b-물리-검증-증거)를 본다.

공식 8-NPU artifact를 만들 때는 [vLLM 가이드의 공식 구성](rbln-vllm-setup.md#여덟-장-llama-32-3b-공식-구성)과
별도 output directory를 사용하고 single-NPU opt-in을 제거한다. 한 장 directory를
장치 수만 바꿔 재사용하지 않는다.

## 12. 서버 acceptance gate

Compile 성공 메시지만으로 benchmark 지원을 승인하지 않는다. 모델마다 다음 순서를
통과하고 원본 terminal output, package versions, contract JSON, hash, run ID와
post-run device JSON을 보존한다.

1. `rbln-smi -j`에서 device 0이 `normal`이고 시작 context가 `[]`인지 확인한다.
2. 정적 모델은 recipe `--describe`, compile report, 독립 inspect와 source/destination
   SHA256을 확인한다. Llama는 manifest 계약과 모든 file hash를 확인한다.
3. 관련 unit/regression test를 실행한다.
4. Batch 1 sync E2E 1~10 sample smoke를 실행하고 evaluator sample 수가 양수인지
   확인한다. `num_samples=0`인 exit 0은 실패다.
5. Runtime unload 뒤 `contexts: []`를 확인한다.
6. Async offline은 worker 1과 작은 queue로 시작한다. accepted/completed/evaluator가
   같고 failed/rejected/timed-out/outstanding 및 native error counter가 모두 0인지
   확인한다.
7. 다시 `contexts: []`를 확인한 뒤에만 full/offline/server-like 측정으로 간다.
8. 성능 결과에는 latency p50/p95/p99, throughput, utilization, memory, power,
   energy와 monitor coverage를 함께 남긴다. 짧은 smoke의 util 0만으로 CPU
   fallback을 단정하지 않는다.

```bash
cd "$RBLN_FW_ROOT/framework"
"$RBLN_RUN_PY" -m pytest -q \
  tests/test_rbln_compile_recipes.py \
  tests/test_prepare_rbln_vllm_model.py \
  tests/test_rbln_runtime.py \
  tests/test_rbln_vllm_runtime.py \
  tests/test_main_paths.py
```

정적 모델의 구체적인 sync/async 명령은 [RBLN 운영 가이드](rbln-setup.md),
Llama sync/async 명령은 [vLLM 가이드](rbln-vllm-setup.md#7-동기-e2e-smoke)를
사용한다. Compile contract와 runtime option의 device 수, sequence length,
block size, batch 및 decoder batch를 정확히 일치시킨다.

### 12.1 종료 후 context 0

모든 compile/load/sync/async 시도 뒤에 실행한다.

```bash
rbln-smi -j
rbln-smi -j | "$RBLN_RUN_PY" -c '
import json, sys
payload = json.load(sys.stdin)
contexts = payload.get("contexts", [])
print("contexts:", contexts)
raise SystemExit(0 if not contexts else 1)
'
```

성공 기준은 출력이 `contexts: []`이고 exit code가 0인 것이다. 남은 PID가 본인이
시작한 프로세스인지 확인하고 정상 종료를 기다린다. 다른 사용자의 프로세스를
종료하지 않는다. Context가 남은 run은 성능 결과로 채택하지 않는다.

## 13. 관찰된 문제와 복구 기준

| 증상 | 원인 | 복구 |
|---|---|---|
| `401 Unauthorized` | Rebellions package index 권한 또는 조직 인증 없음 | Portal 권한/조직 pip 설정을 확인한다. credential을 저장소나 URL에 넣지 않는다. |
| `No module named rebel`인데 `rbln-smi`는 정상 | Driver와 Python package는 별개이고 Python ABI/site가 다름 | 각 interpreter의 package origin을 확인하고 검증된 Python 3.10 env 또는 권한 있는 wheel을 사용한다. |
| uv가 `setuptools==77.0.3`을 찾지 못함 | 여러 package index의 first-index 해석 충돌 | 신뢰 가능한 index만 둔 뒤 필요할 때 uv 안내의 `--index-strategy unsafe-best-match`를 사용하고 최종 lock/version을 기록한다. |
| `.rbln`은 생성됐는데 `cp`가 실패 | Compile directory와 현재 shell directory가 다름 | recipe에 절대 `--output`을 주고 source의 `test -s`, `find`, SHA256을 확인한다. |
| inspect에서 `dict` attribute 오류 | SDK 경로에 따라 inspect가 mapping 또는 object를 반환 | 4.1절 `field()` 방식으로 읽는다. |
| Sync runtime constructor `TypeError` | SDK 0.11 timeout pybind는 float가 아니라 int 초를 요구 | runtime timeout을 `60` 같은 정수로 전달한다. |
| dataset 준비에서 `No module named datasets` | Runtime env와 data/build env가 분리됨 | 준비 script를 `RBLN_BUILD_PY`로 실행하거나 검증 dataset을 안전하게 연결한다. |
| SST-2 `Invalid HF URI ... glue` | dataset repository namespace 해석 문제 | `nyu-mll/glue`를 사용한다. |
| YOLO source/weight/revision 오류 | Submodule 미초기화, 잘못된 checkout 또는 weight | submodule을 초기화하고 정확한 commit, non-empty weight와 weight hash를 확인한다. |
| PatchTST `aten::unfold` 또는 bool clamp lowering 오류 | SDK 0.11 frontend 미지원 연산 | static patchifier와 bool-to-float 내부 cast/equivalence gate가 있는 현재 recipe를 사용한다. |
| SQuAD strict logit allclose 실패 | Compiled precision과 padding logit 차이가 큼 | Direct/swapped quantitative evidence, context argmax/best span, answer text와 token-type sensitivity를 함께 판정한다. |
| Llama `decoder_batch_sizes` type 오류 | Runtime option parser가 단일 숫자를 int로 변환 | Compile tool에는 `--decoder-batch-sizes 1`, runtime에는 가이드대로 `decoder_batch_sizes=1,`를 사용한다. |
| Llama `num_samples=0`인데 exit 0 | 다른 worktree의 dataset path를 사용해 loader가 비었음 | 정확한 `--dataset` 파일과 양수 QA count를 실행 전에 assert한다. |
| 8B compile 후 engine allocation 실패 | Compile artifact 존재와 한 장 runtime 수용은 별도 gate | `compiled_but_single_npu_runtime_capacity_failed`로 기록하고 async를 실행하지 않으며 로그와 context를 보존한다. |
| 짧은 run에서 util 0 | Monitor polling 사이에 burst가 끝남 | Full run, memory/P-state/power/engine log와 monitor sample 수를 함께 확인한다. |

실패한 output directory나 artifact를 즉시 삭제하거나 같은 경로에 덮어쓰지 않는다.
Compiler log, partial file 목록, host OOM 여부와 `rbln-smi -j`를 먼저 저장하고,
원인이 확인된 뒤 새 output path로 재시도한다.

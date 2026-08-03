# Furiosa RNGD 모델 컴파일 실패 재현 기록

이 문서는 Furiosa Torch 2026.3.0에서 ResNet50, YOLOv5m,
PatchTST-FM-r1의 CPU 추론은 성공했지만 strict RNGD 전체 모델 컴파일이 끝나지
않았던 과정을 재현하고 판정하는 기준을 정리합니다. 성공한 BERT와 Llama 경로는
[Furiosa RNGD runtime](furiosa-rngd-setup.md), 일반적인 설치·LLM·비동기 문제는
[Furiosa RNGD 트러블슈팅 Runbook](furiosa-rngd-troubleshooting.md)을 참고하세요.

## 결론

2026년 7월 RNGD 서버와 Furiosa SDK 2026.3.0에서 확인한 결과는 다음과 같습니다.

| 모델 | CPU 첫 추론 | graph 정규화 | strict RNGD 첫 호출 | 판정 |
|---|---:|---|---|---|
| ResNet50 ImageNet V2 | 성공 | Conv-BN fusion 및 CPU parity 성공 | compiler 내부 panic | **실패 재현 완료, 현재 미지원** |
| YOLOv5m (`yolov5mu.pt`) | 성공, `(1,84,8400)` | `YOLO.fuse()`로 mutable BatchNorm 제거 | tactic solver 내부 panic | **실패 재현 완료, 현재 미지원** |
| PatchTST-FM-r1 | 성공, `(1,96,7)` | logger capture 제거; 최소 layout graph는 성공 | full-model `transpose/reshape` decomposition 실패 | **실패 재현 완료, 현재 미지원** |
| BERT SST-2 | 성공 | eager attention graph | strict compile 및 첫 추론 성공 | **지원 검증 완료** |
| BERT SQuAD v1 | 성공 | eager attention graph | strict compile 및 첫 추론 성공 | **지원 검증 완료** |

여기서 실패 재현 완료는 모델 지원을 뜻하지 않습니다. CPU output 반환, strict
compile setup, RNGD 첫 output 반환을 별도 단계로 기록했다는 뜻입니다.

## 성공 판정 기준

재현 도구는 모든 모델에 다음 조건을 고정합니다.

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

`torch.compile()`이 callable을 반환한 시점에는 실제 compiler가 아직 시작되지 않을
수 있습니다. `compiled(*inputs)`의 첫 호출이 output을 반환하고 shape·finite 검사를
통과해야 성공입니다. CPU 또는 eager fallback, 일부 graph만의 성공, compiler
메모리 할당만으로는 성공으로 판정하지 않습니다.

## 당시 검증 환경

아래 값은 과거 오류를 확인한 서버의 기록입니다. 다른 SDK 버전의 결과를 이 표로
일반화하면 안 됩니다.

| 항목 | 확인값 |
|---|---|
| OS / kernel | Ubuntu 22.04.5 LTS / 6.8.0-124-generic |
| 장치 | RNGD `npu0`, PCI ID `1ed2:0001`, 47.50 GiB |
| Driver / firmware | 2026.3.0 / 1.11.0 (`cfd5306`) |
| Python | 3.12.13 |
| Furiosa Torch | 2026.3.0 |
| PyTorch | 2.10.0+cpu |
| Transformers / NumPy | 5.1.0 / 2.5.1 |
| Ultralytics | 8.3.216 |
| 모델 입력 device | `furiosa:0` |

이 문서를 작성한 개발 호스트에는 RNGD와 Furiosa SDK가 없습니다. 따라서 재현
도구의 stage, subprocess, 오류 분류, 로그·JSON 생성은 로컬 자동 테스트로
검증했고, 새 하드웨어 로그는 아래 명령을 RNGD 연결 서버에서 실행해 생성해야 합니다.

## 재현 준비

### 브랜치와 장치

```bash
cd ~/ML-HW-Benchmark-Framework

git branch --show-current
git rev-parse --short HEAD
git status --short

furiosa-smi info
furiosa-smi status
```

`furiosa-smi info`에 `rngd`, `npu0`가 나타나야 합니다. 다른 프로세스가 장치
메모리를 점유하고 있으면 종료하거나 결과에 점유 상태를 함께 남깁니다.

### Python 환경

```bash
cd ~/ML-HW-Benchmark-Framework/framework

PY=../.venv-furiosa-torch/bin/python

"$PY" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

for package in (
    "furiosa-torch",
    "torch",
    "torchvision",
    "transformers",
    "ultralytics",
    "granite-tsfm",
):
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")
PY
```

기본 Furiosa Torch 환경은 다음 파일로 설치합니다.

```bash
uv pip install \
  --python "$PY" \
  -r requirements-furiosa-torch.txt

uv pip install \
  --python "$PY" \
  torchvision==0.25.0 \
  ultralytics==8.3.216
```

PatchTST-FM-r1에는 `granite-tsfm==0.3.6`이 추가로 필요합니다. 이 package는 당시
`transformers<5`를 요구했고, `--no-deps`로 설치한 뒤 Transformers 5.1.0에서 import하면
`transformers.utils.download_url` 부재로 실패할 수 있었습니다. 공유 BERT 환경의
Transformers를 즉시 downgrade하지 말고 PatchTST 전용 복제 환경에서 import smoke를
먼저 통과시켜야 합니다.

```bash
uv pip install \
  --python "$PY" \
  --no-deps \
  granite-tsfm==0.3.6

"$PY" - <<'PY'
from tsfm_public.models.patchtst_fm import PatchTSTFMForPrediction

print("PatchTST-FM import: PASS", PatchTSTFMForPrediction)
PY
```

위 smoke가 실패하면 아직 compiler 재현 단계가 아니라 Python dependency 단계입니다.
그 상태의 결과를 PatchTST RNGD compiler 실패로 분류하면 안 됩니다.

### 모델 파일

ResNet50은 실행 시 TorchVision `IMAGENET1K_V2` weights를 사용하므로 처음 실행하는
서버에는 다운로드 네트워크 또는 `TORCH_HOME` cache가 필요합니다. YOLOv5m과
PatchTST는 다음 경로를 사용합니다.

```text
framework/models/yolov5m/yolov5mu.pt
framework/models/ibm-research_patchtst-fm-r1/
```

필요하면 Hugging Face CLI로 PatchTST 모델을 준비합니다.

```bash
cd ~/ML-HW-Benchmark-Framework/framework

HF=../.venv-furiosa-torch/bin/hf

"$HF" download \
  ibm-research/patchtst-fm-r1 \
  --local-dir models/ibm-research_patchtst-fm-r1
```

## 재현 명령

각 case는 자식 프로세스에서 실행됩니다. Rust panic이나 compiler shutdown이 다음
모델 실행을 오염시키지 않게 하기 위해서입니다. SDK 2026.3.0의 과거 실패를 다시
만나는 경우 정상적으로 exit code 1이 반환됩니다.

### ResNet50

```bash
cd ~/ML-HW-Benchmark-Framework/framework

PY=../.venv-furiosa-torch/bin/python

timeout --signal=INT --kill-after=30s 45m \
  env PYTHONUNBUFFERED=1 RUST_BACKTRACE=full \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case resnet50
```

### YOLOv5m

```bash
cd ~/ML-HW-Benchmark-Framework/framework

PY=../.venv-furiosa-torch/bin/python

timeout --signal=INT --kill-after=30s 45m \
  env PYTHONUNBUFFERED=1 RUST_BACKTRACE=full \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case yolov5m \
  --yolov5-path models/yolov5m/yolov5mu.pt
```

### PatchTST-FM-r1

```bash
cd ~/ML-HW-Benchmark-Framework/framework

PY=../.venv-furiosa-torch/bin/python

timeout --signal=INT --kill-after=30s 45m \
  env PYTHONUNBUFFERED=1 RUST_BACKTRACE=full \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case patchtst \
  --patchtst-path models/ibm-research_patchtst-fm-r1
```

### 세 모델 연속 실행

```bash
cd ~/ML-HW-Benchmark-Framework/framework

PY=../.venv-furiosa-torch/bin/python

timeout --signal=INT --kill-after=30s 120m \
  env PYTHONUNBUFFERED=1 RUST_BACKTRACE=full \
  "$PY" tools/reproduce_furiosa_compile_failures.py \
  --case all \
  --yolov5-path models/yolov5m/yolov5mu.pt \
  --patchtst-path models/ibm-research_patchtst-fm-r1
```

## 결과 파일 읽기

기본 출력은 gitignore된 `framework/results/furiosa-compile-repro/` 아래에 생성됩니다.

```text
<timestamp>-<case>.log
<timestamp>-<case>.child.json
<timestamp>-<case>.json
```

- `.log`: child의 Python traceback, Furiosa/Rust stderr, 진행 stage를 합친 원본입니다.
- `.child.json`: Python/package/SMI 환경과 stage별 성공·실패입니다.
- `.json`: child exit code, 원본 log 경로, 알려진 오류 서명 매칭 결과입니다.

다음 명령으로 최근 결과를 확인할 수 있습니다.

```bash
cd ~/ML-HW-Benchmark-Framework/framework

find results/furiosa-compile-repro \
  -maxdepth 1 \
  -type f \
  -name '*.json' \
  -print

grep -R -n -E \
  'align_up_required|EinsumByDpe|empty transition cost table|Cannot view a tensor|result=passed' \
  results/furiosa-compile-repro
```

`matched_known_signature`가 채워져 있으면 과거 장애와 같은 오류 문자열을 만난
것입니다. 값이 없다고 성공은 아닙니다. `status`, `exit_code`, `stages`, 전체 log를
함께 확인해야 합니다.

## 모델별 실패 경계

### ResNet50: 모델 변환을 넘어선 compiler panic

초기 Kalray ONNX 경로에서는 `onnx2torch`가 ONNX `Flatten` version 1을 변환하지
못했습니다. ONNX opset을 올리고 의미가 같은 flatten boundary를 명시한 뒤 ONNX
Runtime과 PyTorch CPU parity가 통과했습니다. 별도로 TorchVision ImageNet V2
ResNet50을 사용하고 Conv-BN을 fusion했을 때도 CPU Top-1과 output parity가
통과했습니다.

그 다음 strict 첫 호출에서 다음 compiler 내부 오류가 관측됐습니다.

```text
called `Result::unwrap()` on an `Err` value:
align_up_required (true) != false (false)

EinsumByDpe should be given only a single pass:
first_chunk (VeMultiPass([0, 1])) != VeMultiPass([0])

called `Option::unwrap()` on a `None` value

furiosa.UnsupportedOpError: failed to compile the graph
```

`Conv2d(512, 512, 3, padding=1)` 최소 graph에서도 FP32/BF16,
contiguous/channels-last, bias 유무, Default/vision tactic을 바꿔 재현했습니다.
따라서 특정 Kalray ONNX의 `Flatten` 문제나 Python output wrapper 하나로 원인을
축소할 수 없습니다. 확인된 경계는 RNGD tactic/shape compiler 내부입니다.

### YOLOv5m: frontend BatchNorm 해결 뒤 tactic solver panic

원본 Ultralytics graph는 CPU에서 `(1,84,8400)` output을 반환했지만 decomposition
단계에서 mutable BatchNorm을 거부했습니다.

```text
mutable op violation at iteration 0
offenders=[aten._native_batch_norm_legit, ...]
```

공식 `YOLO.fuse()`를 적용해 Conv-BN을 결합한 뒤 이 오류는 사라졌습니다. CPU raw
prediction도 유지됐지만 strict RNGD 첫 호출은 다음 위치에서 중단됐습니다.

```text
tactic-solver/src/find_tactics.rs:514:13
EdgeIndex(162) has empty transition cost table

furiosa.UnsupportedOpError: failed to compile the graph
```

따라서 fusion 전 오류는 frontend graph normalization 문제였고, fusion 후 최종
오류는 Furiosa tactic solver 단계입니다. fusion 성공만으로 모델 지원 완료로 바꿀
수 없습니다.

### PatchTST-FM-r1: full-model layout normalization 미완료

CPU 첫 호출은 FP32 `(1,512,7)` values와 bool mask를 받아 `(1,96,7)` output을
반환했습니다. 처음에는 모델 내부 logger와 CPU autocast metadata가 graph에 잡혀
다음 오류가 나타났습니다.

```text
Tensor device mismatch! Expected: furiosa:0, Got: cpu
```

PatchTST logger만 Dynamo ignore 집합에 등록하고 capture 경계를 좁힌 뒤, 최종 실패
위치는 attention의 다음 코드로 이동했습니다.

```python
x = x.transpose(1, 2).reshape(B, N, C)
```

실제 오류는 다음과 같습니다.

```text
ValueError: Cannot view a tensor with shape
torch.Size([7, 512, 16, 64])
and strides (524288, 64, 32768, 1)
as a tensor with shape (7, 512, 1024)!
```

같은 shape에 `transpose -> torch.clone(memory_format=torch.contiguous_format) -> view`
를 적용한 최소 graph는 strict RNGD에서 성공했습니다. 이는 명시적 복사로 layout을
정규화할 수 있다는 진단 증거일 뿐입니다. 설치된 TSFM 전체 모델을 수정해 다시
compile하고 accuracy를 검증한 결과가 없으므로 PatchTST는 여전히 미지원입니다.

## 왜 BERT는 성공했는가

BERT SST-2와 SQuAD v1도 attention의 non-contiguous reshape 문제를 처음 만났습니다.
하지만 Hugging Face loader에서 `attn_implementation="eager"`를 선택해 explicit
attention graph로 바꾼 뒤 strict compile과 첫 추론, 프레임워크 E2E까지 완료했습니다.
이는 모든 Transformer가 지원된다는 뜻이 아니라, 해당 BERT graph에는 검증된
모델별 정규화 경로가 있다는 뜻입니다.

## SDK 업그레이드 후 재검증

Furiosa Torch, PyTorch, firmware 중 하나라도 바뀌면 과거 오류와 동일하다고
가정하지 않습니다.

1. `furiosa-smi info`, package version, git commit을 새 JSON과 함께 보관합니다.
2. 세 case를 개별 프로세스로 실행합니다.
3. 오류가 달라지면 새 log를 보존하고 stage가 frontend인지 compiler인지 다시
   분류합니다.
4. 첫 호출이 성공하면 CPU/RNGD output parity와 두 번째 warm inference를 확인합니다.
5. 그 다음에만 프레임워크 E2E smoke와 전체 데이터 정확도를 실행합니다.
6. E2E 결과 저장까지 완료되기 전에는 `furiosa-rngd-torch` 지원 registry에 모델을
   추가하지 않습니다.

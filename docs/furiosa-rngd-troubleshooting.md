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


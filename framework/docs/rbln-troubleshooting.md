# Rebellions RBLN-CA22 트러블슈팅

이 문서는 `rbln-static` 통합 과정에서 실제로 관찰한 실패와 검증된 복구
절차를 기록한다. 정상 설치·실행 계약은 [RBLN 운영 가이드](rbln-setup.md)를
먼저 확인한다.

문제 하나를 해결할 때마다 artifact, dataset, runtime, evaluator, monitor 경계를
따로 확인한다. 여러 설정을 동시에 바꾸면 어떤 변경이 문제를 해결했는지 증명할
수 없으므로 한 번에 하나의 변수만 바꾼다.

## 1. 기준 환경과 빠른 판정

최초 검증 서버의 기준은 다음과 같다.

| 항목 | 검증값 |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12 |
| SDK | `rebel-compiler==0.11.0` |
| NPU | device 0, `RBLN-CA22`, `/dev/rbln0` |
| KMD/FW | 3.2.2 / 3.2.2 |
| Device memory | 16,877,879,296 bytes |
| PCI | `0000:ab:00.0`, NUMA node 1, 32.0 GT/s x8 |

문제를 재현하기 전에 환경과 장치 상태를 함께 저장한다.

```bash
python3 --version
python3 -m pip show rebel-compiler
cat /etc/os-release
rbln-smi -q
rbln-smi -j
```

SDK Python API도 별도로 확인한다.

```bash
python3 - <<'PY'
import rebel

print("available:", rebel.npu_is_available(0))
print("name:", rebel.get_npu_name(0))
print("count:", rebel.device_count())
print("RBLNCompiledModel:", hasattr(rebel, "RBLNCompiledModel"))
print("Runtime:", hasattr(rebel, "Runtime"))
print("AsyncRuntime:", hasattr(rebel, "AsyncRuntime"))
PY
```

정상 기준은 device 0이 available이고 이름이 `RBLN-CA22`이며 device count가
1인 것이다. Benchmark 시작 전과 종료 후에는 device 0 context가 없어야 한다.
다음 명령은 context를 출력하고 하나라도 있으면 non-zero로 종료한다.

```bash
rbln-smi -j | python3 -c 'import json,sys; payload=json.load(sys.stdin); contexts=[item for item in payload.get("contexts", []) if isinstance(item, dict) and str(item.get("npu")) == "0"]; print(json.dumps(contexts, indent=2)); raise SystemExit(1 if contexts else 0)'
```

`contexts: []`가 아니면 이전 process의 물리적 completion과 unload가 끝나지 않은
상태다. 이 상태에서 다음 모델을 시작하지 않는다.

## 2. 공통 환경·브랜치·artifact 문제

### 2.1 uv 환경에서 `pip` 또는 `rebel`을 찾지 못함

**증상**

```text
Python 3.12.13
.../.venv-rebelion/bin/python3: No module named pip
ModuleNotFoundError: No module named 'rebel'
```

반면 `/usr/bin/python3.10`에서는 다음 경로의 SDK를 찾았다.

```text
~/.local/lib/python3.10/site-packages/rebel/__init__.py
rebel-compiler==0.11.0
```

**원인**

NPU driver 설치와 Python package 설치는 별개다. `rbln-smi`가 동작해도 현재 uv
환경의 Python ABI와 site-packages에 `rebel`이 존재한다는 뜻은 아니다. 검증 서버는
Python 3.10 user site에 SDK가 있었지만 처음 만든 uv 환경은 Python 3.12였다.

**해결**

Model Zoo compile 환경과 framework 실행 환경을 분리하고 각각 사용할 Python을
명시한다.

```bash
export RBLN_FW_ROOT="$HOME/ML-HW-Benchmark-Framework-rbln"
export RBLN_ZOO_ROOT="$HOME/rebelion/rbln-model-zoo"
export RBLN_RUN_PY="$RBLN_FW_ROOT/.venv-rbln/bin/python"
export RBLN_BUILD_PY="$RBLN_ZOO_ROOT/.venv-rbln-zoo/bin/python"
```

**검증**

```bash
for py in "$RBLN_RUN_PY" "$RBLN_BUILD_PY" /usr/bin/python3.10; do
  "$py" - <<'PY'
import importlib.util
import sys

print("python:", sys.executable)
spec = importlib.util.find_spec("rebel")
print("rebel:", spec.origin if spec else "NOT FOUND")
PY
done
```

Compile은 `RBLN_BUILD_PY`, benchmark는 `RBLN_RUN_PY`로만 실행한다. 하나의
environment를 억지로 공용화하지 않는다.

**상태:** 해결됨. Python 3.10에서 device API와 sync/async runtime 존재를 확인했다.

### 2.2 RBLN 전용 remote branch와 worktree 생성 실패

**증상**

```text
fatal: invalid reference: origin/feat/rbln-runtime-monitor
fatal: Cannot setup tracking information; starting point
'origin/feat/rbln-runtime-monitor' is not a branch.
```

**원인**

첫 오류는 remote ref를 fetch하지 않은 상태에서 switch한 경우다. 두 번째 오류는
one-off refspec으로 ref는 만들었지만 `--track`이 기대하는 remote tracking 설정과
일치하지 않은 경우다. 동시에 기존 checkout에는 다른 NPU 실험 변경이 있었다.

**해결**

먼저 현재 worktree를 확인하고 이미 RBLN worktree가 있으면 재사용한다.

```bash
git worktree list
git fetch origin \
  '+refs/heads/feat/rbln-runtime-monitor:refs/remotes/origin/feat/rbln-runtime-monitor'
```

새 worktree가 필요하면 `--track` 없이 fetched ref에서 local branch를 만들고, 최초
push 때 upstream을 설정한다.

```bash
git worktree add \
  -b feat/rbln-runtime-monitor \
  ../ML-HW-Benchmark-Framework-rbln \
  refs/remotes/origin/feat/rbln-runtime-monitor

git -C ../ML-HW-Benchmark-Framework-rbln \
  push -u origin feat/rbln-runtime-monitor
```

**검증**

```bash
git -C ../ML-HW-Benchmark-Framework-rbln branch --show-current
git -C ../ML-HW-Benchmark-Framework-rbln status --short
```

다른 가속기 실험의 dirty worktree를 정리하거나 덮어쓰지 않는다.

**상태:** 해결됨. RBLN 변경은 `feat/rbln-runtime-monitor`에서 격리했다.

### 2.3 compile은 성공했지만 `.rbln` 복사 경로가 틀림

**증상**

```text
cp: cannot stat 'resnet50.rbln': No such file or directory
```

**원인**

Compiler는 성공했지만 artifact가 생성된 Model Zoo 하위 디렉터리와 `cp`를 실행한
현재 디렉터리가 달랐다. Compiler 성공 로그만으로 현재 디렉터리에 파일이 있다고
가정하면 안 된다.

**해결 및 검증**

```bash
find "$RBLN_ZOO_ROOT" -type f -name 'resnet50.rbln' -ls

RBLN_SOURCE_ARTIFACT="$(
  find "$RBLN_ZOO_ROOT" -type f -name 'resnet50.rbln' -print -quit
)"
test -s "$RBLN_SOURCE_ARTIFACT"

mkdir -p "$RBLN_FW_ROOT/framework/models/rbln/resnet50"
cp "$RBLN_SOURCE_ARTIFACT" \
  "$RBLN_FW_ROOT/framework/models/rbln/resnet50/model.rbln"

sha256sum \
  "$RBLN_FW_ROOT/framework/models/rbln/resnet50/model.rbln"
```

복사 후에는 반드시 최종 framework artifact를 inspect한다. Source artifact inspect만
저장하고 다른 파일을 배포하는 실수를 막기 위해서다.

**상태:** 해결됨.

### 2.4 artifact 환경변수가 비어 `LOADING_FILE_NOT_FOUND` 발생

**증상**

진단 script가 `artifact:` 뒤에 아무 경로도 출력하지 않고 다음 오류로 끝났다.

```text
RuntimeError: LOADING_FILE_NOT_FOUND: The specified file does not exist.
```

**원인**

다른 터미널에서 설정한 `RBLN_SQUAD_ARTIFACT`는 현재 shell로 전달되지 않았다.
NPU runtime이나 artifact 자체의 오류가 아니다.

**해결 및 검증**

```bash
export RBLN_SQUAD_ARTIFACT="$RBLN_FW_ROOT/framework/models/rbln/bert-base-uncased-squad-v1/model.rbln"

printf 'ARTIFACT=<%s>\n' "$RBLN_SQUAD_ARTIFACT"
test -s "$RBLN_SQUAD_ARTIFACT" && echo "ARTIFACT PRECHECK: PASS"
sha256sum "$RBLN_SQUAD_ARTIFACT"
```

Artifact를 사용하는 heredoc 앞에서도 값을 다시 전달한다.

```bash
ARTIFACT="$RBLN_SQUAD_ARTIFACT" "$RBLN_BUILD_PY" - <<'PY'
import os
from pathlib import Path

artifact = Path(os.environ["ARTIFACT"])
print(artifact.resolve())
assert artifact.is_file()
PY
```

이 실패에서는 runtime 생성 전에 중단되었으므로 종료 후 `contexts: []`가 정상이다.

**상태:** 해결됨.

### 2.5 SDK 0.11 sync runtime이 float timeout을 거부함

**증상**

```text
TypeError: create_sync_runtime(): incompatible function arguments
Invoked with: ..., 60.0, False, {}
```

**원인**

`rebel-compiler==0.11.0`의 pybind constructor는 timeout에 C++ signed `int`로
변환 가능한 Python `int`를 요구한다. `60.0`은 값이 정수처럼 보여도 float이므로
거부된다.

**해결**

Adapter의 `runtime_timeout_sec`를 `[1, 2_147_483_647]` 범위의 Python 정수로
제한하고 SDK constructor에도 정수로 전달한다.

```bash
--runtime-option runtime_timeout_sec=60
```

`runtime_timeout_sec=60.0`과 `runtime_timeout_sec=1.5`는 사용하지 않는다.
`shutdown_timeout_sec`는 host drain/join 제한이므로 같은 제약을 적용하지 않는다.

**검증:** 같은 ResNet50 artifact가 warmup과 sync E2E를 완료했다.

**상태:** 해결됨.

### 2.6 `RBLNCompiledModel.inspect()` 반환 형식 차이

**증상**

```text
AttributeError: 'dict' object has no attribute 'npu'
```

**원인**

SDK 버전과 API 경로에 따라 inspect 결과와 tensor descriptor가 mapping이거나 attribute
object일 수 있다.

**해결**

```python
from collections.abc import Mapping

def field(value, name):
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)
```

`npu`, `compiler_version`, `inputs`, `outputs`와 tensor의 `name`, `shape`, `dtype`를
모두 이 helper로 읽는다.

**검증:** BERT SQuAD 최종 artifact에서 3개 input과 2개 unnamed output을 출력했다.

**상태:** 해결됨.

### 2.7 데이터셋 준비 환경과 기존 데이터 재사용

**증상**

ImageNet 자동 준비가 framework runtime environment에서 실행되며 다음 오류가 났다.

```text
ModuleNotFoundError: No module named 'datasets'
```

**원인**

Runtime environment에는 benchmark 실행에 필요한 package만 있었고 Hugging Face
Datasets는 build environment에만 있었다. 다른 worktree에는 이미 검증된 ImageNet
데이터가 존재했다.

**해결**

준비 script는 `RBLN_BUILD_PY`로 실행하거나 기존 dataset을 RBLN worktree에서
참조한다. 원본 dataset을 중복 다운로드하거나 이동하지 않는다.

```bash
test -s "$HOME/ML-HW-Benchmark-Framework/datasets/imagenet_1k/val_labels.txt"
ln -s "$HOME/ML-HW-Benchmark-Framework/datasets/imagenet_1k" \
  "$RBLN_FW_ROOT/framework/datasets/imagenet_1k"
```

이미 대상 경로가 있으면 symlink를 덮어쓰지 말고 내용과 소유권을 먼저 확인한다.

**상태:** 해결됨.

## 3. ResNet50

### 3.1 짧은 smoke에서 NPU utilization이 0으로 기록됨

**증상**

10-sample sync smoke는 정상 추론했지만 `hw_accel_util_avg`와 max가 0이었다.
Monitor attempt/success는 각각 2였고 coverage는 1.0이었다.

**원인**

실행 시간이 monitor의 실효 vendor poll 간격보다 짧았다. Poll 자체는 성공했지만
두 시점이 idle 또는 짧은 실행의 경계에 걸렸다. 이는 추론이 CPU에서 수행됐다는
증거가 아니다.

**해결 및 검증**

Monitor 경로 검증은 smoke의 coverage로 하고 utilization/power 비교는 더 긴 full
run으로 한다. 3,000-sample sync run에서는 다음 결과가 확인됐다.

| Metric | Value |
|---|---:|
| Top-1 / Top-5 | 80.7333 / 95.0333 |
| Average / P99 latency | 0.7161 / 0.8135 ms |
| Throughput | 1396.5449 samples/s |
| NPU util average / max | 4.35 / 4.90% |
| Monitor coverage | 1.0 |

**상태:** 해결됨. 짧은 run의 utilization 0은 sampling 한계로 분류한다.

### 3.2 최초 async full이 measurement에서 멈춘 것처럼 보임

**증상**

```text
[AsyncDebug] phase=measurement event=start
```

이후 최종 metric이 나오지 않았고 `rbln-smi`에는 66 MiB context가 `idle`로 남았다.
초기 진단은 다른 현재 디렉터리에서 실행되어 trace 임시 파일도 찾지 못했다.

**원인 판정**

당시 로그만으로 SDK callback, producer, trace 경로 중 어느 경계가 원인인지 확정하지
못했다. `idle`은 context가 살아 있다는 뜻이지 logical request가 모두 완료됐다는
뜻은 아니다. 이 사례를 SDK deadlock으로 단정하지 않는다.

**해결 및 검증**

Framework 디렉터리에서 worker와 queue를 고정하고 debug 및 trace를 켠 채 재실행했다.

```bash
cd "$RBLN_FW_ROOT/framework"

"$RBLN_RUN_PY" -m src.main \
  --model resnet50 \
  --target rbln-static \
  --artifact models/rbln/resnet50/model.rbln \
  --dataset datasets/imagenet_1k \
  --inference-mode async_queue \
  --scenario offline \
  --batch-size 1 \
  --worker-count 1 \
  --queue-capacity 16 \
  --min-samples 3000 \
  --max-samples 3000 \
  --warmup 2 \
  --flush-timeout-sec 300 \
  --save-request-trace \
  --monitor \
  --debug \
  --results-path results/rbln-resnet50-async-offline-w1-full.csv
```

재실행 결과 3,000개 request가 모두 완료됐다.

| Metric | Value |
|---|---:|
| Async completed samples | 3000 |
| Async throughput | 500.1366 samples/s |
| E2E P50 / P99 | 35.1874 / 39.5664 ms |
| Queue wait P99 | 35.9588 ms |
| Worker utilization | 70.50% |
| Run status | `valid` |

Logical/native failure, timeout, duplicate/late callback, outstanding, inflight가 모두 0이었다.

**상태:** 해결됨. 최초 불완전 run은 보존하되 최종 3,000-sample async full은 통과했다.

## 4. YOLOv5m

### 4.1 Model Zoo submodule 누락

**증상**

YOLOv5 compile 경로가 존재하지만 내부 Ultralytics source가 비어 있었다.

**원인**

Model Zoo의 YOLOv5 구현은 Git submodule이다. Repository clone만으로 submodule
내용이 준비되지 않는다.

**해결 및 검증**

```bash
cd "$RBLN_ZOO_ROOT"
git submodule update --init --recursive -- \
  pytorch/vision/detection/yolov5/yolov5

git -C pytorch/vision/detection/yolov5/yolov5 \
  rev-parse HEAD
```

검증 서버에서는 submodule commit `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`을
checkout했다.

**상태:** 해결됨.

### 4.2 Offline E2E latency가 model latency보다 큼

**증상**

128-request async full에서 model 평균 latency는 6.9404 ms, P99는 7.5183 ms였지만
E2E P99는 197.2488 ms였다.

**원인**

Offline producer가 bounded queue를 가능한 빠르게 채우므로 request가 service 전에
기다린다. Queue wait P99가 174.9483 ms로 E2E 지연의 대부분을 차지했다.

**검증 결과**

| Metric | Value |
|---|---:|
| Samples | 128 |
| mAP@0.5 | 0.5903 |
| Average detections | 6.0938 |
| Async throughput | 89.2293 samples/s |
| E2E P50 / P99 | 189.1151 / 197.2488 ms |
| Queue wait P99 | 174.9483 ms |
| NPU util average / max | 25.8 / 38.7% |
| Run status | `valid` |

수락·완료·평가 sample이 모두 128이고 logical/native failure가 모두 0이다.

**상태:** 해결됨. 이 값은 offline saturation 결과이며 server-like latency와 직접
비교하지 않는다.

## 5. BERT SST-2

### 5.1 Hugging Face dataset URI 파싱 실패

**증상**

```text
Invalid HF URI 'hf://datasets/glue@.../.huggingface.yaml'.
Repository id must be 'namespace/name', got 'glue'.
```

**원인**

검증 환경의 Hugging Face Hub/Datasets 조합은 dataset repository ID에 명시적인
namespace를 요구했다. 모델이나 RBLN compile 오류가 아니다.

**해결**

동일 오류가 재현되면 `glue` 대신 namespaced repository를 사용한다.

```bash
"$RBLN_BUILD_PY" \
  "$RBLN_FW_ROOT/framework/datasets/prepare_text_numpy.py" \
  --model-id textattack/bert-base-uncased-SST-2 \
  --seq-len 128 \
  --dataset-name nyu-mll/glue \
  --dataset-config sst2 \
  --split validation \
  --output-dir "$RBLN_FW_ROOT/framework/datasets/sst2_numpy"
```

**검증**

`input_ids.npy`와 `attention_mask.npy`는 `(872,128)` int64이고 labels 수가 872인지
확인한다. Artifact contract는 두 input 모두 `(1,128)` int64이다.

**상태:** 해결됨.

### 5.2 Sync/async 품질과 lifecycle 검증

872-sample sync full 결과는 다음과 같다.

| Metric | Value |
|---|---:|
| Accuracy | 79.5872 |
| Average / P99 latency | 1.5951 / 1.5996 ms |
| Throughput | 626.9312 samples/s |
| Monitor coverage | 1.0 |

같은 전체 validation의 async worker 1 결과는 다음과 같다.

| Metric | Value |
|---|---:|
| Completed samples | 872 |
| Async throughput | 357.3785 samples/s |
| E2E P50 / P99 | 49.6675 / 53.6269 ms |
| Queue wait P99 | 47.8658 ms |
| Worker utilization | 93.74% |
| Run status | `valid` |

Accuracy는 sync와 동일했고 실패·거절·timeout·outstanding 및 모든 native error
counter가 0이었다.

**상태:** sync E2E와 async offline full 통과.

## 6. PatchTST ETTh1

### 6.1 `aten::unfold` 미지원

**증상**

```text
NotImplementedError: The following operators are not implemented:
['aten::unfold']
```

**원인**

Transformers PatchTST의 patchification이 `aten::unfold`를 생성했고 SDK 0.11의
PyTorch/Relay 변환기가 이 operator를 지원하지 않았다.

**해결**

고정 `context_length=512`, `patch_length=12`, `patch_stride=12` 계약을 이용해
동일한 patch를 정적 slice/reshape 경로로 생성하는 wrapper를 사용했다. 이 경로는
42개 patch와 `sequence_start=8`을 생성한다.

**검증**

Compile 전에 다음을 모두 확인했다.

```text
patch_length: 12
patch_stride: 12
num_patches: 42
sequence_start: 8
output: (1, 96, 7)
aten::unfold present: False
CPU equivalence: PASS
```

**상태:** 해결됨. Dynamic patch shape 지원으로 일반화하지 않고 이 fixed contract에만
적용한다.

### 6.2 bool mask의 Relay `clamp_min` 변환 실패

**증상**

정적 patch wrapper 이후 다음 오류가 발생했다.

```text
ValueError: Invalid integer data type 'b'.
```

Stack은 Relay PyTorch frontend의 bool `clamp_min` 변환에서 `np.iinfo(bool)`을
시도했다.

**원인**

PatchTST 내부 mask 계산에 bool tensor가 전달되어 SDK 0.11 converter의 dtype 처리
한계를 만났다.

**해결**

외부 artifact input contract는 bool로 유지하고 모델 내부 계산 직전에 mask를
`past_values.dtype`으로 변환한다.

```python
def forward(self, past_values, past_observed_mask):
    observed_mask = past_observed_mask.to(dtype=past_values.dtype)
    outputs = self.model(
        past_values=past_values,
        past_observed_mask=observed_mask,
        return_dict=True,
    )
    return outputs.prediction_outputs
```

변경 전에 같은 0/1 mask의 bool과 float32 CPU output이 정확히 같은지 검증했다.

```python
torch.testing.assert_close(
    float_mask_output,
    bool_mask_output,
    rtol=0,
    atol=0,
)
```

**상태:** 해결됨. 의미 동등성 검증 없이 임의 dtype cast를 추가하지 않는다.

### 6.3 Async full 검증

| Metric | Value |
|---|---:|
| Windows | 240 |
| MAE / RMSE | 0.4242 / 0.6217 |
| Average / P99 latency | 1.4640 / 2.2520 ms |
| Async throughput | 337.0399 samples/s |
| E2E P50 / P99 | 52.0644 / 55.2329 ms |
| Queue wait P99 | 50.4033 ms |
| NPU util average / max | 2.35 / 4.70% |
| Run status | `valid` |

수락·완료·평가 sample이 모두 240이고 logical/native failure가 모두 0이다.

**상태:** sync E2E와 async offline full 통과.

## 7. BERT SQuAD

### 7.1 기존 2-input artifact가 profile과 맞지 않음

**증상**

처음 artifact는 `input_ids`와 `attention_mask`만 받았다. BERT question answering의
질문과 context segment를 구분하는 `token_type_ids`가 artifact contract에 없었다.

**원인**

Wrapper와 framework profile/dataset이 서로 다른 입력 계약을 사용했다.

**해결**

최종 고정 contract는 다음과 같다.

| Input | Shape | Dtype |
|---|---|---|
| `input_ids` | `(1,384)` | int64 |
| `attention_mask` | `(1,384)` | int64 |
| `token_type_ids` | `(1,384)` | int64 |

Wrapper output은 tuple 순서로 start와 end를 반환한다.

```python
def forward(self, input_ids, attention_mask, token_type_ids):
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    return outputs.start_logits, outputs.end_logits
```

최종 source와 framework artifact SHA256은 모두 다음 값이었다.

```text
caada10a3e055df43b24ac388e8fccb5b71fc8fe4a1c08c51dca91922a600b33
```

**상태:** 3-input compile, copy, inspect까지 해결됨.

### 7.2 두 output 이름이 `null`

**증상**

SDK inspect 결과는 float32 `(1,384)` output 두 개를 반환하지만 이름은 모두
`null`이었다. Shape가 같으므로 shape만으로 의미를 결정할 수 없다.

**원인**

SDK 0.11 artifact가 tuple output의 Python 의미 이름을 보존하지 않았다.

**해결 및 검증**

CPU/NPU comparison에서 keyword PyTorch 호출과 inspect-order NumPy positional 호출이
같은 결과를 냈다. Direct mapping의 MAE 합이 swapped mapping보다 작고, 두 runtime
호출 모두 다음 위치를 선택했다.

```text
output[0] = start_logits
output[1] = end_logits
CPU context argmax: 11 15
NPU context argmax: 11 15
CPU answer: neural processing unit inference performance
NPU answer: neural processing unit inference performance
```

따라서 최종 artifact 옆에 SHA-bound sidecar를 생성해야 한다.

```json
{
  "schema_version": 1,
  "artifact_sha256": "caada10a3e055df43b24ac388e8fccb5b71fc8fe4a1c08c51dca91922a600b33",
  "output_names": ["start_logits", "end_logits"]
}
```

Sidecar는 출력 위치와 이름을 결합할 뿐 numerical accuracy를 인증하지 않는다.
Artifact가 바뀌면 CPU/NPU mapping 검증과 sidecar hash 생성을 모두 다시 수행한다.

**상태:** 출력 순서 검증과 sidecar 계약은 해결됨. 배포 환경에서 실제
`model.rbln.json` 존재와 hash 일치는 E2E 전에 다시 확인해야 한다.

### 7.3 `token_type_ids`가 실제 NPU graph에서 사용되는지 확인

**증상**

3-input inspect만으로 세 번째 tensor가 graph 계산에 실제로 반영되는지는 증명되지
않았다.

**검증**

Real token type과 all-zero token type의 output 차이를 CPU와 NPU에서 비교했다.

| Head | CPU context sensitivity MAE | NPU context sensitivity MAE |
|---|---:|---:|
| start | 6.771585 | 7.372314 |
| end | 5.934141 | 6.885633 |

NPU output은 CPU zero-token-type 결과보다 CPU real-token-type 결과에 훨씬 가까웠다.
따라서 세 번째 input이 무시되거나 상수화됐다는 가설은 기각됐다.

**상태:** 해결됨. `token_type_ids` 사용 확인.

### 7.4 CPU/NPU strict logit equality 실패

**증상**

`rtol=1e-3`, `atol=1e-3` allclose가 실패했다. Output 순서와 최종 answer는 같지만
context와 padding logits가 수치적으로 일치하지 않았다.

| Head | Context MAE | Context correlation | All-position max error |
|---|---:|---:|---:|
| start | 1.655135 | 0.734280 | 11.599066 |
| end | 0.952200 | 0.940383 | 12.541334 |

Padding 영역 correlation은 start 0.021215, end 0.141009로 낮았다.

**현재 판정**

- Input 순서 문제 아님: keyword와 positional runtime 결과가 동일했다.
- Output 순서 문제 아님: direct mapping과 span이 일치했다.
- `token_type_ids` 무시 문제 아님: real/zero sensitivity가 확인됐다.
- Compiler precision 또는 graph lowering 차이는 아직 task-level 전체 데이터로 판정하지
  않았다.

현재 `BertQAEvaluator`는 전체 384 logits에 argmax를 적용하며 persisted context mask를
사용하지 않는다. Padding logits의 큰 차이가 final metric을 오염할 수 있으므로 다음을
완료하기 전 SQuAD benchmark를 accepted로 표시하지 않는다.

1. Dataset에 answer 후보가 되는 context 위치 mask를 보존한다.
2. Evaluator가 question, special, padding 위치를 제외하고 span을 계산한다.
3. 같은 preprocessing과 evaluator로 CPU/NPU task-level 결과를 비교한다.
4. Sync smoke/full, async offline, 종료 후 `contexts: []`를 확인한다.

**상태:** 열린 이슈. 단일 샘플 semantic parity는 통과했지만 strict numerical parity와
전체 SQuAD E2E는 미승인이다.

## 8. 비동기 큐와 monitoring 해석

### 8.1 세 latency 경계를 구분할 것

- `Average/P99 Latency`: evaluator에 전달된 모델별 추론 timing record다.
- `async_service_time_*`: worker가 runtime service를 수행한 구간이다.
- `async_e2e_latency_*`: request 발행부터 completion까지이며 queue wait를 포함한다.

Offline scenario는 producer가 queue를 가능한 빨리 채우므로 E2E latency가 service
latency보다 커지는 것이 정상이다. 검증된 네 모델의 E2E P99 중 약 89~91%가 queue
wait였다. 이 수치를 interactive serving latency로 해석하지 않는다. Serving 특성은
`server_like`에서 target QPS를 명시해 별도로 측정한다.

### 8.2 검증된 async offline full 비교

| Model | Samples | Async samples/s | E2E P50/P99 ms | Queue P99 ms | NPU util avg/max | Memory MB | Power avg/max W | Energy J |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv5m | 128 | 89.2293 | 189.1151 / 197.2488 | 174.9483 | 25.8 / 38.7 | 82 | 18.85 / 18.89 | 29.0225 |
| BERT SST-2 | 872 | 357.3785 | 49.6675 / 53.6269 | 47.8658 | 30.52 / 49.0 | 180 | 36.99 / 47.77 | 78.8359 |
| PatchTST ETTh1 | 240 | 337.0399 | 52.0644 / 55.2329 | 50.4033 | 2.35 / 4.7 | 16 | 18.93 / 18.93 | 15.6191 |
| ResNet50 | 3000 | 500.1366 | 35.1874 / 39.5664 | 35.9588 | 19.6 / 24.3 | 66 | 38.91 / 41.65 | 238.2348 |

`hw_accel_energy_j`는 benchmark process만의 에너지가 아니라 idle power를 포함한
카드 전체 전력 적분값이다. 서로 다른 실행 시간의 total energy만 비교하지 말고
samples, duration, power sample 수와 함께 해석한다.

### 8.3 Async 성공 조건

CSV, details JSON과 console에서 다음 invariant를 모두 확인한다.

```text
async_accepted_requests == async_completed_requests == async_evaluator_samples
async_failed_requests == async_rejected_requests == async_timed_out_requests == 0
async_outstanding_requests == async_native_inflight == 0
async_native_duplicate_callbacks == async_native_late_callbacks == 0
async_native_submit_failures == async_native_timeouts == 0
hw_accel_monitor_coverage == 1.0
async_run_status == valid
```

이 조건은 logical/native accounting을 검증한다. Python process가 종료된 뒤 실제 device
context가 해제됐는지는 별도의 `rbln-smi -j`에서 `contexts: []`로 증명한다.

### 8.4 Monitor sample이 적을 때

10~100 sample smoke는 1~3개의 power sample만 남을 수 있다. Coverage 1.0은 시도한
poll이 모두 성공했다는 뜻이지 충분한 시간 해상도를 뜻하지 않는다. Utilization,
temperature, power, energy를 모델 간 비교할 때는 다음을 같이 저장한다.

- `hw_accel_monitor_attempts`
- `hw_accel_monitor_successes`
- `hw_accel_monitor_coverage`
- `hw_accel_power_samples`
- 전체 실행 시간과 sample 수

## 9. 최종 검증 상태

| Model | Sync E2E | Async offline full | Remaining work |
|---|---|---|---|
| ResNet50 | 통과 | 통과, 3,000 requests | Server-like와 concurrency sweep |
| YOLOv5m | 통과 | 통과, 128 requests | Server-like와 concurrency sweep |
| BERT SST-2 | 통과 | 통과, 872 requests | Server-like와 concurrency sweep |
| PatchTST ETTh1 | 통과 | 통과, 240 windows | Server-like와 concurrency sweep |
| BERT SQuAD | 미승인 | 미실행 | Context-masked evaluator와 task-level validation |
| Llama 3.1 8B / 3.2 3B | Static 범위 밖 | Static 범위 밖 | 후속 in-process `rbln-vllm` target |

초기 실패가 있었더라도 최종 valid run이 있으면 둘 다 보존한다. 실패 로그를 삭제하거나
최종 성공으로 원인을 소급해서 단정하지 않는다.

## 10. 재실행 체크리스트

모델마다 다음 순서로 실행한다. Single CA22에서 여러 모델을 동시에 실행하지 않는다.

1. 올바른 Python environment와 SDK import를 확인한다.
2. `rbln-smi -j`에서 시작 context가 0인지 확인한다.
3. 최종 framework artifact를 inspect하고 name, shape, dtype, NPU target을 저장한다.
4. Artifact SHA256과 sidecar가 필요한 모델의 sidecar hash를 확인한다.
5. Dataset file 수, tensor shape/dtype과 profile contract를 확인한다.
6. Sync smoke를 실행해 runtime allocation과 evaluator 연결을 확인한다.
7. Sync full을 실행해 품질과 장시간 monitoring을 확인한다.
8. Async offline을 `worker_count=1`, queue 16에서 실행한다.
9. CSV, details JSON, request trace를 저장한다.
10. Exact accounting과 logical/native zero-error invariant를 확인한다.
11. Process 종료 후 `rbln-smi -j`의 `contexts: []`를 확인한다.
12. 위 gate를 통과한 모델만 server-like와 worker/parallel sweep으로 이동한다.

문제가 재발하면 실행 명령, console 전체 로그, CSV, details JSON, trace JSONL,
실행 전후 `rbln-smi -j`, artifact inspect와 SHA256을 하나의 진단 묶음으로 보관한다.

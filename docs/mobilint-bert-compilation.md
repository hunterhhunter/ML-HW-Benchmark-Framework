# Mobilint BERT SST-2·SQuAD v1 컴파일 재현 가이드

이 문서는 Mobilint `qbcompiler==1.2.0`으로 framework용 BERT SST-2와
SQuAD v1 MBLT/MXQ를 다시 만드는 절차를 설명한다. 컴파일 호스트만 대상으로 하며
ARIES에서 artifact를 실행하거나 정확도를 측정하지 않는다.

컴파일에는 Docker가 필요하지 않다. ARIES 장치가 필요하지 않으며 Mobilint driver,
qb Runtime과 `/dev/aries*`도 설치할 필요가 없다. compiler host와 실제 benchmark를
수행하는 ARIES server는 달라도 된다.

## 1. 지원 환경과 준비물

검증한 환경은 다음과 같다.

| 항목 | 고정값 |
|---|---|
| OS | Ubuntu 22.04 |
| CPU architecture | x86-64 (`uname -m`: `x86_64`) |
| Python | CPython 3.10 |
| compiler | `qbcompiler-1.2.0-py3-none-any.whl` |
| compile target | `aries-rb` |

Mobilint 배포 페이지에서 Ubuntu 22.04/x86-64/Python 3.10용 wheel을 받아 사용자가
compiler host의 `~/Downloads` 등에 둔다. wheel은 라이선스가 있는 vendor 배포물이므로
이 저장소에는 포함하지 않는다.

Ubuntu package가 아직 없다면 다음처럼 Python 3.10 venv 지원을 설치한다.

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv
```

package dependency, Hugging Face 모델과 calibration dataset을 처음 받을 때는 외부
network가 필요하다. Hugging Face 인증이 필요한 환경에서는 사용자가 정상적인 cache와
인증을 미리 설정한다. 스크립트는 token 값을 읽거나 출력하지 않는다.

wheel을 확인한다.

```bash
WHEEL="$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl"

test -s "$WHEEL"
sha256sum "$WHEEL"
```

검증에 사용한 wheel SHA256은 다음 값이다.

```text
28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b
```

파일명이나 checksum이 다르면 one-shot script가 dependency를 설치하기 전에
중단한다. 같은 version 이름으로 다시 배포된 다른 wheel을 사용해야 한다면 변경 이유와
새 checksum을 검토한 뒤 저장소의 guard를 명시적으로 갱신한다.

## 2. 한 명령으로 두 task 컴파일

저장소 root에서 실행한다. 현재 shell에서 다른 venv가 활성화돼 있어도 스크립트는
`--python`과 자체 compiler venv의 interpreter를 명시적으로 사용한다.

```bash
cd ~/ML-HW-Benchmark-Framework

bash framework/scripts/compile_mobilint_bert.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --task all \
  --output-root "$PWD/mobilint-bert-artifacts" \
  |& tee "$PWD/mobilint-bert-compile.log"
```

`--task all`은 `sst2`와 `squad1` 두 task를 순서대로 실행한다는 뜻이다. 모든
Mobilint hardware용 binary를 한꺼번에 만든다는 뜻이 아니다. 두 task의 hardware
target은 qbcompiler 1.2에서 ARIES를 나타내는 `aries-rb` 하나로 고정돼 있다.

기본 venv는 저장소의 `.venv-qbcompiler-1.2-py310`이다. 다른 위치를 사용하려면
`--venv /path/to/venv`를 추가한다. 스크립트는 다음 순서로 동작한다.

1. Ubuntu 22.04, x86_64, CPython 3.10 검사
2. wheel 파일명과 SHA256 검사
3. compiler 전용 venv 생성 또는 기존 venv 검사
4. `pip`이 없으면 `ensurepip` 적용
5. 검증한 dependency 설치와 `pip check`
6. `onnxruntime`, `qbcompiler` import 및 compiler signature 확인
7. task별 모델·dataset·embedding weight·calibration data 준비
8. task별 MBLT와 MXQ 컴파일
9. artifact 크기와 SHA256 출력

다음 primary dependency를 고정한다.

```text
torch==2.7.1
torchvision==0.22.1
numpy==1.26.0
tensorflow==2.17.0
onnx==1.16.2
onnxruntime==1.19.2
opencv-python==4.11.0.86
transformers==4.57.1
datasets==3.6.0
qbcompiler==1.2.0  # 전달한 wheel
```

PyTorch와 TorchVision은 실제 성공 환경과 같은 `cu128` wheel index에서 설치하지만
compiler는 CPU에서 실행된다. NVIDIA GPU나 CUDA device는 요구하지 않는다.

한 task만 만들려면 새 output root와 task를 지정한다.

```bash
bash framework/scripts/compile_mobilint_bert.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --task sst2 \
  --output-root "$PWD/mobilint-bert-sst2-artifacts"
```

문서에서 사용하는 `mobilint-bert-artifacts*`와
`mobilint-bert-*-artifacts*` 경로는 `.gitignore`에 포함돼 있다. 저장소 안에서 임의의
다른 `--output-root`를 지정하면 Git에서 자동으로 숨겨지지 않으므로, 실행 후
`git status --short`로 대용량 artifact가 staging 대상이 아닌지 확인한다. 가능하면
문서의 기본 이름을 사용하거나 저장소 밖 경로를 지정한다.

## 3. 모델, 데이터와 calibration 선택

| task | source model | calibration dataset | 최대 길이 | sample 수 |
|---|---|---|---:|---:|
| SST-2 | `textattack/bert-base-uncased-SST-2` | `glue/sst2` validation | 128 | 32 |
| SQuAD v1 | `csarron/bert-base-uncased-squad-v1` | `squad` validation | 384 | 32 |

각 validation split의 처음부터 끝까지 `numpy.linspace`로 32개 index를 균등하게
선택한다. random seed에는 의존하지 않는다. 실제 index, token 길이, 모델과 dataset
식별자는 task별 `calibration_manifest.json`에 기록된다.

fine-tuned BERT에서 다음 embedding parameter를 `weight_dict.pth`로 저장한다.

```text
word_embeddings
token_type_embeddings
position_embeddings
layernorm_weight
layernorm_bias
```

calibration `.npy`는 위 weight로 계산한 contiguous
`float32 [1, valid_sequence_length, 768]` embedding이다. framework가 benchmark에서
사용하는 `MobilintBertEmbeddingTransform`으로 생성하므로 compiler calibration과
실행 시 host preprocessing이 같은 경계를 사용한다.

## 4. 실제 compiler 호출

source graph에는 다음 세 token tensor를 제공하고 sequence dimension을 dynamic으로
표시한다. attention mask에는 qbcompiler의 `padding_mask` semantic을 설정한다.

```text
input_ids       int64 [1,L]
attention_mask  int64 [1,L]
token_type_ids  int64 [1,L]
```

SQuAD의 Hugging Face forward가 사용하는 `Tensor.split(int)`는 qbcompiler 1.2에서
지원되지 않았다. recipe는 같은 BERT/QA head를 공유하면서 마지막 channel을 indexing해
`start_logits`, `end_logits`를 반환하는 동등한 wrapper를 사용한다. 각 compiler 호출
전에 모델을 새로 load하고 원본 모델과 두 tensor가 같은지 CPU에서 확인한다.

MBLT 호출의 핵심 option은 다음과 같다.

```python
mblt_compile(
    model=model,
    mblt_save_path=str(mblt_path),
    target_device="aries-rb",
    backend="torch",
    feed_dict=feed_dict,
    cpu_offload=True,
)
```

MXQ는 다음 calibration 설정으로 별도 호출한다.

```python
mxq_compile(
    model=model,
    target_device="aries-rb",
    save_path=str(mxq_path),
    calib_data_path=str(calibration_dir),
    backend="torch",
    feed_dict=feed_dict,
    inference_scheme="all",
    calibration_config=CalibrationConfig(
        method=1,
        output=0,
        mode=1,
        max_percentile=CalibrationConfig.MaxPercentile(
            percentile=0.999,
            topk_ratio=0.01,
        ),
    ),
)
```

`.mblt`는 `.mxq`의 입력 파일이 아니다. 두 파일은 같은 source model과 feed contract에서
각 compiler API를 독립적으로 호출한 결과다. one-shot script는 실제 검증 때처럼
MBLT와 MXQ를 별도 Python process에서 실행해 각 단계가 새 모델로 시작하게 한다.
Python entrypoint의 `--stage all`도 단계 사이에 모델을 다시 load한다.

## 5. Python entrypoint를 직접 실행하는 방법

dependency와 venv를 이미 준비했다면 shell bootstrap 없이 단계를 나눠 실행할 수 있다.

```bash
REPO="$HOME/ML-HW-Benchmark-Framework"
PY="$REPO/.venv-qbcompiler-1.2-py310/bin/python"
ARTIFACT_ROOT="$REPO/mobilint-bert-artifacts"

cd "$REPO"
export PYTHONPATH="$REPO/framework:$REPO/framework/src"

"$PY" -m tools.mobilint_bert_compile.prepare \
  --task sst2 \
  --output-root "$ARTIFACT_ROOT"

"$PY" -m tools.mobilint_bert_compile.compile \
  --task sst2 \
  --stage mblt \
  --artifact-root "$ARTIFACT_ROOT"

"$PY" -m tools.mobilint_bert_compile.compile \
  --task sst2 \
  --stage mxq \
  --artifact-root "$ARTIFACT_ROOT"
```

컴파일을 시작하지 않고 고정 contract만 보려면 다음을 실행한다. 이 경로는
qbcompiler를 import하지 않는다.

```bash
"$PY" -m tools.mobilint_bert_compile.prepare --task squad1 --describe
"$PY" -m tools.mobilint_bert_compile.compile --task squad1 --describe
```

## 6. 산출물과 입출력 계약

정상 완료 시 다음 구조가 생성된다.

```text
mobilint-bert-artifacts/
├── compile-environment.json
├── sst2/
│   ├── calibration_data/000.npy ... 031.npy
│   ├── calibration_manifest.json
│   ├── weights/weight_dict.pth
│   ├── mblt/sst2.mblt
│   ├── mxq/sst2.mxq
│   └── compile-report.json
└── squad1/
    ├── calibration_data/000.npy ... 031.npy
    ├── calibration_manifest.json
    ├── weights/weight_dict.pth
    ├── mblt/squad1.mblt
    ├── mxq/squad1.mxq
    └── compile-report.json
```

실제 ARIES/qb Runtime v1.3.2에서 확인했던 MXQ 경계는 다음과 같다.

| task | MXQ 입력 | source model 출력 | ARIES 실측 runtime 출력 |
|---|---|---|---|
| SST-2 | `Float32 [1,-1,768]` | `logits` | `logits [1,1,2]` |
| SQuAD v1 | `Float32 [1,-1,768]` | `start_logits`, `end_logits` | `end_logits`, `start_logits`; 각각 `[1,-1,1]` |

SQuAD의 두 runtime output은 이름 metadata가 없고 shape도 같다. compiler source의 반환
순서와 ARIES에서 관측한 positional 순서가 반대이므로 `compile-report.json`은 두 순서를
별도 필드로 기록한다. compiler-only host에서는 이 runtime 순서를 다시 확인할 수 없다.
새 compiler/SDK로 만든 artifact는 ARIES에서 별도로 계약을 검사해야 한다.

framework benchmark에 필요한 파일은 task별 `.mxq`와 `weight_dict.pth`다. `.mblt`와
calibration data는 실행 입력이 아니라 컴파일 재현·진단 자료다.

```bash
SST2_MXQ="$ARTIFACT_ROOT/sst2/mxq/sst2.mxq"
SST2_WEIGHTS="$ARTIFACT_ROOT/sst2/weights/weight_dict.pth"
SQUAD1_MXQ="$ARTIFACT_ROOT/squad1/mxq/squad1.mxq"
SQUAD1_WEIGHTS="$ARTIFACT_ROOT/squad1/weights/weight_dict.pth"

sha256sum \
  "$SST2_MXQ" "$SST2_WEIGHTS" \
  "$SQUAD1_MXQ" "$SQUAD1_WEIGHTS"
```

ARIES로 전달한 뒤 실행하는 방법은
[Mobilint ARIES Transformer·LLM 실행 가이드](mobilint-aries-transformers.md)를
따른다.

## 7. 재실행과 오류 처리

recipe는 기존 task 디렉터리나 `.mblt`/`.mxq`를 덮어쓰지 않는다. 실패한 compiler
artifact도 자동 삭제하지 않으므로 로그와 부분 결과를 보존해 진단할 수 있다. 다시
시작할 때는 기존 결과를 수동으로 덮어쓰지 말고 새 output root를 지정한다.

```bash
--output-root "$PWD/mobilint-bert-artifacts-retry-$(date +%Y%m%d-%H%M%S)"
```

자주 확인할 오류는 다음과 같다.

- `No module named pip`: script가 새 venv에는 `ensurepip`을 적용한다. 기존 venv가
  손상됐다면 새 `--venv` 경로를 사용한다.
- `Please install onnxruntime >= 1.19.2`: compiler venv에서 script를 처음부터 실행해
  exact `onnxruntime==1.19.2` 설치와 import 검사를 통과시킨다.
- wheel SHA256 mismatch: 다른 wheel로 계속하지 말고 배포 파일과 대상 OS/Python을
  다시 확인한다.
- task output already exists: 새 output root를 사용한다.
- calibration file set mismatch: manifest와 `000.npy`~`031.npy`가 한 실행에서 생성된
  동일한 묶음인지 확인한다.
- compile 명령 이후 terminal 종료: `|& tee`로 남긴 log 마지막 오류와 host resource를
  확인한다. 성공 표시는 `COMPILATION_COMPLETE=<path>`다.

컴파일 성공은 non-empty `.mblt`/`.mxq`와 report/hash가 만들어졌다는 뜻이다. ARIES
실행 성공이나 task 정확도까지 의미하지 않는다.

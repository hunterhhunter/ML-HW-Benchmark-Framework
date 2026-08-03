# Mobilint BERT MXQ 컴파일 재현 설계

## 목표

qbcompiler 1.2로 검증했던 BERT SST-2와 SQuAD v1 artifact를 저장소의
스크립트와 문서만으로 다시 만들 수 있게 한다. 지원 환경은 Mobilint가 배포한
wheel의 조건과 실제 성공 환경에 맞춰 Ubuntu 22.04, Intel/AMD x86-64, Python
3.10으로 고정한다.

이 기능은 컴파일 호스트 전용이다. ARIES 장치, `qbruntime`, driver,
`mobilint-cli` 또는 실제 추론을 요구하지 않는다. benchmark runtime과 자동 컴파일
경로도 변경하지 않는다.

## 범위

재현 경로는 다음 결과를 만든다.

- fine-tuned Hugging Face 모델과 validation dataset 다운로드
- task별 embedding weight인 `weight_dict.pth` 추출
- validation split에서 균등 간격으로 선택한 32개 calibration embedding 생성
- `aries-rb` 대상 `.mblt`와 `.mxq` 생성
- 모델, dataset, calibration index/길이, package version, compiler option을 담은
  manifest 기록
- 모든 최종 artifact의 파일 크기와 SHA256 출력

다음 작업은 포함하지 않는다.

- ARIES에서 MXQ를 load하거나 inference하는 작업
- `qbruntime` 계약 검사 또는 정확도 측정
- benchmark CLI 실행
- vendor wheel, Hugging Face cache, dataset, weight, `.mblt`, `.mxq`의 Git 저장
- PatchTST 및 BERT 이외 모델 컴파일

## 사용자 인터페이스

저장소 root에서 다음 한 명령으로 두 task를 준비하고 컴파일한다.

```bash
bash framework/scripts/compile_mobilint_bert.sh \
  --wheel "$HOME/Downloads/qbcompiler-1.2.0-py3-none-any.whl" \
  --python "$(command -v python3.10)" \
  --task all \
  --output-root "$PWD/mobilint-bert-artifacts"
```

`--task`는 `sst2`, `squad1`, `all`을 받는다. 여기서 `all`은 두 BERT task를
뜻하며 모든 Mobilint hardware를 대상으로 컴파일한다는 의미가 아니다. hardware
target은 두 task 모두 `aries-rb` 하나로 고정한다.

스크립트는 명시적인 wheel 경로와 새 output root를 요구한다. `--python`은 생략하면
`python3.10`을 찾고, `--venv`는 생략하면 저장소 root의
`.venv-qbcompiler-1.2-py310`을 사용한다. 기존 task artifact가 있는 디렉터리는
덮어쓰지 않는다. 재실행하려면 새 output root를 지정해야 한다.

고급 사용자와 단위 테스트를 위해 Python entrypoint도 제공한다.

```bash
PYTHONPATH="framework:framework/src" python -m tools.mobilint_bert_compile.prepare \
  --task sst2 --output-root ./mobilint-bert-artifacts

PYTHONPATH="framework:framework/src" python -m tools.mobilint_bert_compile.compile \
  --task sst2 --stage all --artifact-root ./mobilint-bert-artifacts
```

## 파일 구조

```text
framework/
├── scripts/compile_mobilint_bert.sh
├── tools/mobilint_bert_compile/
│   ├── __init__.py
│   ├── common.py
│   ├── prepare.py
│   └── compile.py
└── tests/test_mobilint_bert_compile.py
docs/
└── mobilint-bert-compilation.md
```

`common.py`는 task 규격, embedding weight 추출, qbcompiler 1.2가 처리할 수 있는
SQuAD wrapper와 JSON-safe contract를 소유한다. `prepare.py`는 모델·dataset 준비와
calibration/평가 입력 생성을 담당한다. `compile.py`는 compiler API 호출과 artifact
보고만 담당한다. shell script는 host 검사, 전용 가상환경 준비, 고정 dependency 설치와
두 Python entrypoint의 순서 제어만 담당한다.

## 고정 환경

shell script는 다음 조건을 시작 전에 확인한다.

- `/etc/os-release`의 Ubuntu version이 `22.04`
- `uname -m`이 `x86_64`
- 선택한 interpreter가 CPython 3.10
- wheel 파일명이 `qbcompiler-1.2.0-py3-none-any.whl`
- wheel SHA256이 검증 당시 값
  `28f276baef1bff86ed313cb819b53d8abb684a7555cf4c81c459edc09abf1b4b`

기본 compiler environment는 저장소 root의
`.venv-qbcompiler-1.2-py310`이다. 이미 호환되는 환경이면 재사용하고, 없으면
`python3.10 -m venv`로 만든다. `pip`가 빠진 venv에는 `ensurepip`을 먼저 적용한다.

검증 때 사용한 package를 그대로 고정한다.

- `qbcompiler==1.2.0` — 사용자가 전달한 wheel
- `torch==2.7.1`
- `torchvision==0.22.1`
- `numpy==1.26.0`
- `tensorflow==2.17.0`
- `onnx==1.16.2`
- `onnxruntime==1.19.2`
- `opencv-python==4.11.0.86`
- `transformers==4.57.1`
- `datasets==3.6.0`

PyTorch 두 package는 실제 성공 환경과 동일하게 PyTorch의 `cu128` index에서
설치한다. 컴파일 과정 자체는 CPU에서 수행하며 NVIDIA GPU나 CUDA device를 요구하지
않는다. index까지 고정하는 이유는 같은 version 문자열의 다른 wheel 조합이 섞이는
것을 막기 위해서다.

`onnxruntime>=1.19.2`는 qbcompiler import 자체에 필요하므로 설치 후 별도 import
검사를 수행한다. 설치가 끝나면 Python, OS, architecture, package version과
`qbcompiler.mblt_compile`/`mxq_compile` signature를 로그에 기록한다.

## Task와 데이터 규격

| task | Hugging Face model | dataset | 최대 길이 | source output |
|---|---|---|---:|---|
| SST-2 | `textattack/bert-base-uncased-SST-2` | `glue/sst2`, validation | 128 | `logits` |
| SQuAD v1 | `csarron/bert-base-uncased-squad-v1` | `squad`, validation | 384 | `start_logits`, `end_logits` |

calibration sample은 validation split 전체 범위에서 `numpy.linspace`로 32개 index를
결정한다. 임의 seed에 의존하지 않으며 선택한 index와 실제 token 길이를 manifest에
기록한다. 각 sample은 batch 1로 tokenize하고 attention mask의 유효 prefix까지만
남긴다.

준비 단계는 fine-tuned 모델의 BERT embedding에서 다음 다섯 tensor를 복사한다.

- `word_embeddings`
- `token_type_embeddings`
- `position_embeddings`
- `layernorm_weight`
- `layernorm_bias`

calibration `.npy`는 이 weight를 사용해 `word + token_type + position` embedding과
LayerNorm(`eps=1e-12`)을 계산한 contiguous `float32 [1,L,768]` 값이다. 중복 구현을
피하기 위해 현재 benchmark가 사용하는 `MobilintBertEmbeddingTransform`으로 이 값을
만든다. 따라서 컴파일 calibration 경계와 실행 시 host transform 경계가 같은 코드로
유지된다.

## Compiler graph와 option

source model은 token 입력 세 개를 받는다.

```text
input_ids       int tensor [1,L]
attention_mask  int tensor [1,L]
token_type_ids  int tensor [1,L]
```

SST-2 tokenizer가 `token_type_ids`를 생략하면 zero tensor를 만든다. 세 입력의 sequence
dimension은 qbcompiler의 `wrap_tensor`로 dynamic 처리하고 attention mask에는
`padding_mask` semantic을 지정한다.

SQuAD 모델의 기본 forward는 qbcompiler 1.2가 지원하지 않는
`Tensor.split(int)` 경로를 사용하므로, 동일한 `bert`와 `qa_outputs` module을 공유하는
wrapper가 마지막 logits의 두 channel을 indexing한다. compile 전 CPU 비교 테스트로
wrapper의 `start_logits`와 `end_logits`가 원본 Hugging Face 출력과 bitwise 동일한지
확인한다.

MBLT 생성 호출은 다음으로 고정한다.

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

MXQ 생성 호출은 다음으로 고정한다.

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

`.mblt`는 `.mxq`의 입력 파일이 아니다. 검증 때 실행한 graph compile 결과를 별도로
남기는 artifact이며, MXQ 단계는 같은 source model과 feed dictionary에서 독립적으로
시작한다. 기본 `--stage all`은 원래 성공 절차를 재현하기 위해 두 호출을 순서대로
실행한다.

`aries-rb`는 qbcompiler 1.2의 ARIES용 compile target 이름이다. 하나의 호출이 다른
Mobilint hardware까지 모두 컴파일하는 옵션이 아니다.

## 산출물과 계약

```text
<output-root>/
├── compile-environment.json
├── sst2/
│   ├── calibration_data/*.npy
│   ├── calibration_manifest.json
│   ├── weights/weight_dict.pth
│   ├── mblt/sst2.mblt
│   ├── mxq/sst2.mxq
│   └── compile-report.json
└── squad1/
    └── ...
```

실제 ARIES에서 검증한 MXQ runtime 경계는 다음과 같다.

| task | MXQ input | MXQ runtime output |
|---|---|---|
| SST-2 | `Float32 [1,-1,768]` 하나 | `[1,1,2]` 하나, 의미는 `logits` |
| SQuAD v1 | `Float32 [1,-1,768]` 하나 | `[1,-1,1]` 두 개, 실측 순서는 `end_logits`, `start_logits` |

SQuAD source model의 반환 순서는 `start_logits`, `end_logits`지만 qbruntime의 unnamed
output 실측 순서는 그 반대다. compiler-only 호스트에서는 qbruntime metadata나 의미
순서를 재검증할 수 없으므로, compile report는 source output 순서와 기존 ARIES에서
검증된 runtime output 순서를 별도 필드로 기록한다. 새 compiler/SDK에서 이 동작이
바뀌었는지 확인하는 일은 이 스크립트의 성공 판정에 포함하지 않는다.

## 안전성과 실패 처리

- host, Python 또는 wheel checksum이 기준과 다르면 dependency 설치 전에 중단한다.
- output root 자체는 만들 수 있지만 task 디렉터리에 기존 calibration 또는 artifact가
  있으면 중단한다.
- 각 compiler 호출 전에 CPU smoke forward와 output name/shape를 검사한다.
- calibration 파일이 정확히 manifest 개수만큼 없으면 MXQ compile을 시작하지 않는다.
- compiler가 반환해도 최종 파일이 없거나 빈 파일이면 실패한다.
- 성공한 artifact마다 SHA256과 byte size를 report에 기록한다.
- `set -euo pipefail`을 사용하되, 실패 원인이 보이도록 현재 단계와 로그 경로를 출력한다.
- wheel, access token 또는 Hugging Face cache 경로의 내용을 로그에 노출하지 않는다.

부분 성공 시 이미 생성된 큰 artifact를 자동 삭제하지 않는다. 사용자가 진단할 수 있게
그대로 남기되, 같은 output root 재사용은 거부한다.

## 테스트 전략

일반 개발 환경에는 qbcompiler가 없어도 다음 검증이 가능해야 한다.

- task spec과 JSON description이 고정돼 있는지
- `--describe`/`--help`가 qbcompiler, datasets, transformers를 import하지 않는지
- output overwrite guard와 calibration count guard
- embedding weight 추출 key/shape
- 기존 runtime embedding transform으로 생성한 calibration dtype/shape
- SQuAD wrapper 출력과 원본 Hugging Face QA 출력의 동일성
- fake qbcompiler를 주입했을 때 `aries-rb`, dynamic feed, calibration option과 output
  경로가 정확히 전달되는지
- shell `bash -n`과 문서에 실제 entrypoint가 연결돼 있는지

실제 qbcompiler wheel로 `.mblt`/`.mxq`를 만드는 작업은 Ubuntu 22.04 compiler
호스트에서 수행하는 수동 통합 검증이다. 이 컴파일 전용 기능의 PR에서는 ARIES runtime
성공을 새로 주장하지 않고, 기존에 성공한 runtime 결과와 새 artifact를 구분해 기록한다.

## 문서 연결

`docs/mobilint-bert-compilation.md`를 canonical compile runbook으로 사용한다. 기존
`docs/mobilint-aries-transformers.md`의 artifact 준비 부분은 긴 명령을 중복하지 않고 이
runbook을 링크한다. runbook은 다음을 명시한다.

- compiler host와 ARIES runtime host는 같을 필요가 없음
- wheel은 사용자가 Mobilint에서 받아 경로로 전달해야 함
- network는 Hugging Face model/dataset 및 Python dependency 다운로드에 필요함
- `all`은 두 task, `aries-rb`는 단일 hardware target이라는 점
- 생성한 `.mxq`와 `weight_dict.pth`만 기존 benchmark 명령에 전달한다는 점
- source output 순서와 ARIES에서 관측한 SQuAD runtime output 순서의 차이

# RBLN 모델 컴파일 레시피 설계

## 목표

실제 RBLN-CA22 서버에서 컴파일하고 실행한 일곱 모델의 준비 과정을 저장소
안에서 재현할 수 있게 만든다. 정적 모델 다섯 개에는 독립 실행 가능한 컴파일
스크립트를 추가하고, Llama 두 개에는 이미 검증된
`tools/prepare_rbln_vllm_model.py`를 단일 진입점으로 유지한다. 통합 문서는 환경
준비, 모델 다운로드, 컴파일, artifact 검사, framework 배치, 종료 후 context
검사까지 같은 순서로 설명한다.

대상 모델은 다음과 같다.

| 경로 | 모델 |
|---|---|
| `rbln-static` | ResNet50, YOLOv5m, BERT SST-2, BERT SQuAD, PatchTST ETTh1 |
| `rbln-vllm` | Llama 3.2 3B Instruct, Llama 3.1 8B Instruct |

## 비목표

- `rbln-static` target에 자동 컴파일러를 연결하지 않는다.
- `.rbln`, Hugging Face weight, tokenizer, dataset 또는 credential을 Git에 넣지
  않는다.
- RBLN SDK가 없는 개발 환경에서 실제 컴파일을 흉내 내지 않는다.
- 기존 model profile, runtime, evaluator, async queue 또는 monitor 동작을 바꾸지
  않는다.
- 단일 ATOM Llama 결과를 Rebellions 공식 지원으로 분류하지 않는다.
- 검증하지 않은 동적 shape, batch 또는 다른 compiler/firmware 버전으로 계약을
  일반화하지 않는다.

## 검토한 대안

### 선택: 모델별 독립 스크립트와 통합 문서

정적 모델별 `compile.py`를 분리하고 공통 안전 검사만 작은 Python package로
공유한다. 각 스크립트가 하나의 model ID, wrapper, fixed-shape ABI와 output을
소유하므로 특정 모델의 의존성이나 converter 우회가 다른 모델에 전파되지 않는다.

### 제외: 모든 모델을 처리하는 단일 CLI

명령은 짧지만 TorchVision, YOLO submodule, Transformers QA/classification,
PatchTST 우회 코드가 한 모듈에 결합된다. 또한 framework의 target compiler가
자동으로 지원되는 것처럼 오해하기 쉽다.

### 제외: 문서 안에만 전체 Python 소스 삽입

기록은 남지만 복사 과정에서 들여쓰기·파일명·수정 이력이 갈라질 수 있다. 문서는
실행 명령과 설명을 담당하고, 실제 소스는 테스트 가능한 저장소 파일로 유지한다.

## 파일 구조

```text
framework/
  tools/
    __init__.py
    rbln_compile_recipes/
      __init__.py
      common.py
      resnet50/{__init__.py,compile.py}
      yolov5m/{__init__.py,compile.py}
      bert_sst2/{__init__.py,compile.py}
      bert_squad/{__init__.py,compile.py}
      patchtst_etth1/{__init__.py,compile.py}
  tests/
    test_rbln_compile_recipes.py
  docs/
    rbln-compilation.md
    rbln-setup.md
    rbln-vllm-setup.md
```

스크립트는 repository의 `framework` 디렉터리에서 module 형태로 실행한다.

```bash
python -m tools.rbln_compile_recipes.resnet50.compile --output /path/model.rbln
```

`tools/__init__.py` 추가는 기존 standalone 도구의 직접 실행을 변경하지 않는다.

## 공통 CLI와 안전 계약

각 모듈은 표준 라이브러리만 import한 상태에서 `--help`와 `--describe`를 처리한다.
`rebel`, Torch, Transformers, TorchVision 같은 build dependency는 실제 컴파일을
시작한 뒤 lazy import한다. 따라서 일반 개발·CI 환경에서도 명령 형식과 정적
계약을 검증할 수 있다.

공통 옵션은 다음으로 제한한다.

- `--output`: 생성할 `.rbln` 파일의 명시적 경로
- `--describe`: SDK import 없이 model ID와 입출력 계약을 JSON으로 출력

`--describe`는 output 경로 없이 단독 실행할 수 있다. 실제 컴파일에서는
`--output`이 필수다.

모델별로 꼭 필요한 옵션만 추가한다. 예를 들어 YOLOv5m은 local YOLOv5 checkout과
weight 경로를, 필요한 경우 Hugging Face 모델은 `--model-id`를 받는다. 출력 파일이
이미 존재하면 덮어쓰지 않고 실패한다. 상위 디렉터리는 생성할 수 있지만 광범위한
삭제나 기존 artifact 교체는 수행하지 않는다.

컴파일 성공 뒤에는 같은 process에서 다음을 검사한다.

1. 출력 파일이 존재하고 크기가 0보다 큰지 확인한다.
2. `RBLNCompiledModel.inspect()`를 mapping/object 양쪽 형식에 대응해 읽는다.
3. target NPU, compiler version, input name·shape·dtype와 output shape·dtype를
   recipe 계약과 비교한다.
4. artifact 크기와 SHA256을 출력한다.
5. SDK가 output 이름을 보존하지 못하는 허용 사례는 명시적으로 표시하되 임의의
   이름을 artifact metadata에 있다고 주장하지 않는다.

## 모델별 고정 계약

| Recipe | 원본 | 입력 | 출력 | 특수 처리 |
|---|---|---|---|---|
| ResNet50 | TorchVision ResNet50 ImageNet weights | `float32 (1,3,224,224)` | `float32 (1,1000)` | 평가 모드와 고정 NCHW 입력 |
| YOLOv5m | 검증한 YOLOv5 submodule과 YOLOv5m weight | `float32 (1,3,640,640)` | raw `float32 (1,25200,85)` | submodule/weight 존재와 source revision 기록 |
| BERT SST-2 | `textattack/bert-base-uncased-SST-2` | 두 `int64 (1,128)` | `float32 (1,2)` | `input_ids`, `attention_mask` 순서 고정 |
| BERT SQuAD | `csarron/bert-base-uncased-squad-v1` | 세 `int64 (1,384)` | 두 `float32 (1,384)` | `token_type_ids` 포함, tuple 순서 start/end 고정 |
| PatchTST ETTh1 | `ibm-granite/granite-timeseries-patchtst` | values `float32 (1,512,7)`, mask `bool (1,512,7)` | `float32 (1,96,7)` | 정적 patchification, `aten::unfold` 제거, bool/float mask CPU 동등성 검사 |
| Llama 3.2 3B | `meta-llama/Llama-3.2-3B-Instruct` | prepared directory 계약 | prefill/decoder `.rbln` | 기존 prepare tool, 단일 ATOM seq/block 512 opt-in |
| Llama 3.1 8B | `meta-llama/Llama-3.1-8B-Instruct` | prepared directory 계약 | prefill/decoder `.rbln` | 기존 prepare tool, 단일 ATOM seq/block 512 opt-in과 15 GiB gate |

Recipe가 사용하는 정확한 weight enum, YOLO checkout revision과 wrapper 구현은 실제
서버 로그 및 기존 Model Zoo 계약과 대조해 코드와 통합 문서에 함께 고정한다.

## 모델별 오류 처리

### YOLOv5m

submodule directory, import 가능한 YOLO source, weight 파일 중 하나라도 없으면
compiler import 전에 구체적인 경로와 함께 실패한다. 검증 서버에서 사용한 submodule
revision `86fd1ab270cb2f7e53ee7412cd4a0650bf4bcc51`을 문서에 기록하되 다른 revision을
조용히 같은 provenance로 취급하지 않는다.

### PatchTST

기본 Transformers graph의 `aten::unfold`는 SDK 0.11에서 지원되지 않는다. Recipe는
context 512, patch length/stride 12, 42 patches, sequence start 8인 정적 wrapper만
제공한다. 컴파일 전에 원본 CPU output과 정적 wrapper output, bool mask와 float mask
output을 검증한다. 이 검증이 실패하면 artifact를 만들지 않는다.

### BERT SQuAD

세 번째 `token_type_ids` 입력을 생략하지 않는다. SDK 0.11이 두 output 이름을
`null`로 inspect하므로 recipe는 tuple 위치만 고정하고, 배포용
`model.rbln.json`은 별도의 실제 CPU/NPU mapping 검증 뒤 최종 artifact SHA256으로
만든다. Compile recipe 자체가 수치 정확도를 인증하거나 sidecar를 추측해 생성하지
않는다.

### Llama

새 정적 recipe를 만들지 않는다. 기존 prepare tool의 manifest, 파일별 SHA256,
output-directory overwrite 방지, 모델 identity와 single-NPU opt-in 계약을 그대로
사용한다. 통합 문서는 3B/8B의 정확한 호출과 산출물 검사를 한곳에서 연결한다.

## 통합 문서 구성

`framework/docs/rbln-compilation.md`를 컴파일의 canonical entrypoint로 추가한다.

1. 검증 환경과 package origin 확인
2. RBLN Model Zoo clone/submodule 및 Hugging Face 인증
3. build/run Python 분리와 공통 환경 변수
4. 정적 다섯 모델의 recipe 호출
5. Llama 두 모델의 기존 prepare tool 호출
6. 모델별 inspect 계약, SHA256과 artifact 배치 경로
7. 컴파일 실패를 보존하고 재시도하는 절차
8. CPU/NPU mapping 또는 의미 동등성이 필요한 모델의 추가 gate
9. framework smoke로 넘어가기 전 `rbln-smi -j` 확인
10. 실제 관찰한 오류와 `rbln-troubleshooting.md` 연결

`rbln-setup.md`의 “compile 제외” 표현은 runtime이 자동 컴파일하지 않는다는 뜻으로
명확히 바꾸고 새 문서로 연결한다. `rbln-vllm-setup.md`도 모델 준비 상세의 canonical
링크를 새 문서에 추가하되 기존 실행 runbook과 물리 검증 증거는 유지한다.

## 테스트와 검증

SDK 없는 자동 테스트는 다음을 확인한다.

- 다섯 recipe의 `--help`가 heavy dependency 없이 성공한다.
- `--describe`가 model ID, input/output 계약과 특수 처리 분류를 안정적인 JSON으로
  반환한다.
- 기존 출력 파일을 거부하는 공통 보호 로직이 동작한다.
- 문서에 나오는 module path와 실제 recipe module이 일치한다.
- model contract가 기존 `model_profiles.py`의 정적 profile과 모순되지 않는다.
- 기존 Llama prepare tool의 parser와 single-NPU 계약 테스트가 계속 통과한다.

문서 검증은 Markdown link, 명령에 사용된 file/module path, placeholder와 conflict
marker 부재를 검사한다. 전체 framework 테스트를 마지막 회귀 gate로 실행한다.

실장비에서는 각 recipe에 대해 compile exit 0, inspect 일치, SHA256 기록,
framework artifact 복사 후 동일 hash, sync smoke, 종료 후 `contexts: []` 순으로
확인한다. 이 저장소 변경은 기존 실장비 결과를 보존하지만 새 스크립트를 현재
로컬 NPU 없는 환경에서 컴파일 성공했다고 주장하지 않는다.

## 배포와 이력

스크립트, 통합 문서와 테스트만 Git에 포함한다. 생성된 artifact는 기존 표준 경로인
`framework/models/rbln/<profile>/model.rbln`에 서버 운영자가 배치하며 Git에는
추가하지 않는다. PR에는 각 recipe의 provenance, 아직 재실행하지 않은 실장비 gate,
그리고 `rbln-static` 자동 컴파일이 여전히 범위 밖임을 명시한다.

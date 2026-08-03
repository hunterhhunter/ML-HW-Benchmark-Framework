# Furiosa RNGD 컴파일 실패 재현 설계

## 목적

Furiosa Torch 2026.3.0에서 전체 모델 컴파일에 실패했던 ResNet50,
YOLOv5m, PatchTST-FM-r1을 같은 strict 조건으로 다시 실행할 수 있게 만들고,
CPU 성공과 RNGD 컴파일 실패 사이의 정확한 경계를 문서화한다. 과거 로그의 오류
문자열만 보관하는 데 그치지 않고, 서버에서 같은 판정 절차와 결과 형식을 반복할 수
있는 재현 도구를 제공한다.

## 기준 상태

- BERT SST-2와 BERT SQuAD v1은 strict Furiosa Torch 첫 호출 컴파일 및 추론에
  성공했으며 지원 모델로 유지한다.
- Llama 3.1 8B와 Llama 3.2 3B는 Furiosa-LLM 경로에서 성공했으므로 이번
  Furiosa Torch 실패 재현 대상이 아니다.
- ResNet50, YOLOv5m, PatchTST-FM-r1은 CPU forward에는 성공했지만 strict 전체
  모델 RNGD 컴파일과 첫 추론까지 완료하지 못했다.
- 기존 문서는 ResNet50 compiler panic만 상세히 기록하고 YOLOv5m과 PatchTST의
  최종 오류 서명은 `미검증` 또는 일반적인 `strict compile 실패`로만 남겨 현재
  증거와 일치하지 않는다.

## 접근 방식

최신 `origin/main`에서 독립 브랜치를 만들고 다음 두 산출물을 추가한다.

1. `framework/tools/reproduce_furiosa_compile_failures.py`
   - 모델별 CPU forward와 strict RNGD 첫 호출을 단계별로 실행한다.
   - 모델 로더와 Furiosa SDK import를 해당 case 실행 시점까지 지연한다.
   - `--case resnet50`, `--case yolov5m`, `--case patchtst`, `--case all`을
     지원한다.
   - 실행 환경, 단계별 상태, 예외 type과 정규화된 오류 서명을 JSON으로 저장한다.
2. `docs/furiosa-rngd-compilation-troubleshooting.md`
   - 서버 환경, 모델 경로, 재현 명령, 성공 판정 조건, 실제 오류, 우회 시도와 현재
     지원 상태를 한 문서에서 연결한다.

기존 `docs/furiosa-rngd-setup.md`와
`docs/furiosa-rngd-troubleshooting.md`는 새 문서로 연결하고, 실패 모델을
`미검증`이 아니라 `실패 재현 완료`로 분류한다.

## 재현 도구 계약

### 공통 strict 조건

모든 RNGD case는 다음 조건을 바꾸지 않는다.

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

`torch.compile()` callable 생성만으로 성공 처리하지 않는다. 첫 입력을 호출해
컴파일, 장치 로딩, 출력 반환까지 끝나야 `rngd_first_inference=passed`로 판정한다.

### 단계

각 case는 가능한 범위에서 다음 순서로 실행한다.

1. 환경과 장치 확인
2. 모델 로딩
3. CPU 첫 추론 및 출력 shape/finite 검사
4. 모델별 graph 정규화
5. RNGD strict compile 및 첫 추론
6. 결과 JSON 저장

단계 실패는 이후 단계를 실행하지 않으며 프로세스 exit code는 실패로 반환한다.
`all`은 compiler panic이 다른 case 실행을 손상시킬 수 있으므로 각 case를 별도
subprocess로 격리한다.

### 모델별 입력과 정규화

| case | 모델과 입력 | 정규화 | 통과 조건 |
|---|---|---|---|
| `resnet50` | TorchVision ImageNet V2, `(1,3,224,224)` FP32 | eval Conv-BN fusion | CPU와 RNGD output `(1,1000)` |
| `yolov5m` | `yolov5mu.pt`, `(1,3,640,640)` FP32 | Ultralytics `YOLO.fuse()` | CPU와 RNGD output `(1,84,8400)` |
| `patchtst` | PatchTST-FM-r1, values `(1,512,7)` FP32와 bool mask | 모델 logger capture 억제, 원본 full-model graph 유지 | CPU와 RNGD output `(1,96,7)` |

PatchTST의 `transpose -> clone -> view` 최소 graph 성공은 별도 진단 명령으로
문서화하지만, 전체 모델 loader를 몰래 patch하지 않는다. 최소 graph 성공과 전체
모델 지원 여부를 혼동하지 않기 위해서다.

## 결과 형식

기본 로그 디렉터리는 `framework/results/furiosa-compile-repro/`로 하되 git에
커밋하지 않는다. JSON은 다음 정보를 가진다.

- 실행 시각과 case 이름
- Python, PyTorch, Furiosa Torch, driver/firmware 정보
- 모델 경로와 입력 shape/dtype
- 단계별 `passed`, `failed`, `skipped`
- 예외 class와 첫 번째 안정적인 오류 서명
- 전체 traceback을 기록한 text log 경로

재현 도구는 오류를 성공으로 바꾸는 도구가 아니다. 과거와 같은 알려진 오류를
만나도 exit code 1을 반환하되 JSON의 `matched_known_signature`로 과거 재현 여부를
표현한다.

## 기록할 오류 서명

### ResNet50

- `align_up_required (true) != false (false)`
- `EinsumByDpe should be given only a single pass`
- `called Option::unwrap() on a None value`
- 최종 Python 예외 `furiosa.UnsupportedOpError: failed to compile the graph`

Kalray ONNX 경로에서 발생했던 `Flatten` version 1 변환 실패는 frontend 모델 변환
문제로 분리한다. TorchVision ImageNet V2 모델과 Conv-BN fusion에서도 strict compile이
실패했으므로 특정 ONNX 파일만의 문제로 결론 내리지 않는다.

### YOLOv5m

- fusion 전: `mutable op violation`과 `aten._native_batch_norm_legit`
- `YOLO.fuse()` 후: `EdgeIndex(162) has empty transition cost table`
- 최종 Python 예외 `furiosa.UnsupportedOpError: failed to compile the graph`

fusion은 frontend mutable BatchNorm 문제를 제거했지만 tactic solver 내부 panic까지
해결하지 못했다.

### PatchTST-FM-r1

- 초기 capture: `Tensor device mismatch! Expected: furiosa:0, Got: cpu`
- 최종 full model: `Cannot view a tensor with shape torch.Size([7, 512, 16, 64])`
  및 `(7, 512, 1024)` reshape 실패
- 오류 위치: PatchTST attention의 `x.transpose(1, 2).reshape(B, N, C)`

`transpose -> torch.clone(memory_format=torch.contiguous_format) -> view` 최소 graph는
strict RNGD에서 통과했다. 따라서 현재 증거는 backend kernel panic이 아니라
PatchTST full-model frontend/layout 정규화 미완료로 분류한다.

## 안전 장치

- `eager_fallback=True`나 CPU fallback을 성공으로 허용하지 않는다.
- 모델 가중치, FXB, 데이터셋, 생성된 결과 로그를 커밋하지 않는다.
- 실제 RNGD가 없거나 필수 package가 없으면 명시적인 prerequisite 오류를 내고
  compiler 실패로 잘못 분류하지 않는다.
- exception message 전체에 모델 입력이나 민감한 경로가 포함될 수 있으므로 JSON에는
  알려진 서명 또는 정규화된 첫 줄만 저장한다. 전체 traceback은 사용자가 지정한
  로컬 text log에만 둔다.
- 한 case의 Rust panic 또는 timeout이 다른 case의 결과를 오염시키지 않도록
  `all` 실행을 subprocess로 격리한다.

## 검증

- Furiosa SDK 없이도 CLI help, case 선택, 경로 검증, known-signature 분류와 JSON
  직렬화를 단위 테스트한다.
- fake model/backend로 CPU 단계와 strict first-call 단계의 순서를 검증한다.
- 기존 Furiosa BERT, runtime, environment 테스트를 실행해 지원 경로 회귀가 없는지
  확인한다. 현재 샌드박스에서 cross-thread asyncio wakeup이 차단되는
  `test_furiosa_native_backend.py`는 환경 제한으로 별도 표기한다.
- RNGD 서버에서는 세 case를 각각 실행해 JSON, text log, `furiosa-smi info`를 함께
  보관하고 문서의 기대 오류 서명과 비교한다.

## 범위 밖

- 실패 모델을 `furiosa-rngd-torch` 지원 registry에 다시 등록하는 작업
- 벤더 SDK 또는 설치된 PatchTST package를 직접 수정하는 작업
- compiler panic을 우회하기 위한 eager fallback
- 모델 정확도 및 성능 벤치마크
- Furiosa SDK 업그레이드 후의 지원 여부 보장

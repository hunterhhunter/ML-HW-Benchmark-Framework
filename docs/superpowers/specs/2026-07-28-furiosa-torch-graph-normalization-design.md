# Furiosa Torch 모델 그래프 정규화 설계

## 목적

Furiosa RNGD의 strict 실행 계약인 `eager_fallback=False`, `fullgraph=True`,
`dynamic=False`를 유지하면서 YOLOv5m, BERT SST-2, BERT QA,
PatchTST-FM-R1이 모델별 프런트엔드 장애를 통과하도록 한다. CPU나 eager 경로로
연산을 우회하지 않는다.

## 확인된 원인

- BERT SST-2와 BERT QA는 기본 SDPA attention 출력의 non-contiguous
  `reshape`가 Furiosa decomposition에서 잘못된 `view`로 처리되어 실패한다.
- YOLOv5m은 추론 그래프에 남아 있는 BatchNorm 상태 변경 연산
  `aten._native_batch_norm_legit` 때문에 decomposition이 실패한다.
- PatchTST-FM-R1은 모델 `forward()` 내부의 `logger.info()` 호출을
  TorchDynamo가 fullgraph로 캡처하지 못해 Furiosa backend 진입 전에 실패한다.

## 변경 설계

### BERT SST-2와 BERT QA

두 Hugging Face 모델 로더에 `attn_implementation="eager"`를 명시한다. 이는
attention을 CPU에서 실행한다는 뜻이 아니라, SDPA 대신 명시적인 PyTorch 연산
그래프를 생성해 전체 그래프를 RNGD compiler에 넘긴다는 뜻이다.

### YOLOv5m

Ultralytics의 공식 `YOLO.fuse()`를 모델 로딩 직후 호출해 Conv2d와 BatchNorm2d를
동일한 추론 연산의 fused Conv2d로 바꾼다. wrapper의 입력과 raw detection 출력
계약 `(1, 84, 8400)`은 유지한다.

### PatchTST-FM-R1

`tsfm_public.models.patchtst_fm.modeling_patchtst_fm`의 logger 인스턴스에 한해
`logger.info`를 `torch._dynamo.config.ignore_logger_methods`에 등록한다. 로그
side effect만 제거하며 모델 계산과 출력을 변경하지 않는다.

## 오류 처리

- 필요한 모델 API나 PyTorch Dynamo 설정이 없으면 조용히 fallback하지 않고
  로딩 단계에서 설명 가능한 오류를 낸다.
- 정규화 후 Furiosa compiler가 연산을 지원하지 않으면 해당 모델을
  `compiler-blocked`로 분류한다. `eager_fallback=True`로 숨기지 않는다.

## 검증

1. 단위 테스트에서 BERT 두 로더의 eager attention 선택을 확인한다.
2. 단위 테스트에서 YOLO 공식 fusion 호출 및 기존 raw 출력 계약을 확인한다.
3. 단위 테스트에서 PatchTST logger만 Dynamo ignore 집합에 등록되는지 확인한다.
4. 기존 Furiosa Torch 관련 테스트 전체를 실행한다.
5. RNGD 서버에서 모델별 CPU 출력 shape/finite 검사를 다시 수행한다.
6. 모델별 `e2e --max-steps 1`을 독립 프로세스로 실행한다.
7. E2E 성공 모델만 single-worker `async_queue`로 확장한다.

RNGD 서버 검증 전까지는 컴파일 성공을 주장하지 않는다.

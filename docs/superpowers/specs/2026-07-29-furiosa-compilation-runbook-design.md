# Furiosa RNGD 컴파일 Runbook 문서 설계

## 목적

Furiosa RNGD에서 수행한 컴파일 작업을 재현 가능한 형태로 남기고, 사전 컴파일 FXB와 첫 호출 JIT 컴파일을 혼동하지 않도록 한다. 실제 성공한 모델과 컴파일 단계에서 실패한 모델을 명확히 분리한다.

## 문서 위치

새 독립 문서를 만들지 않고 `docs/furiosa-rngd-setup.md`에 `컴파일과 artifact 재사용` 절을 추가한다. 설치, 컴파일, 실행 절차를 한 문서에서 순서대로 확인할 수 있기 때문이다.

## 포함 범위

1. 모델별 컴파일 상태 표
   - Llama 3.2 3B: `fxb build` 성공
   - BERT SST-2/SQuAD v1: 첫 추론의 strict `torch.compile` 성공
   - Llama 3.1 8B: 직접 컴파일하지 않고 Furiosa 배포 artifact 사용
   - ResNet50/YOLOv5m/PatchTST: 전체 모델 strict 컴파일 실패
2. Llama 3.2 3B FXB 빌드
   - SDK 2026.3.0, TP=8, O0, max model length 4096
   - 원본 config의 중복 `head_dim` 제거본을 별도 디렉터리에 준비
   - dry-run과 실제 build 명령을 구분
   - 성공한 9개 kernel과 약 15분 21초의 관측 결과 기록
   - 생성 artifact 예시 경로와 `fxb show` 확인 방법
3. BERT 첫 호출 JIT 컴파일
   - `eager_fallback=False`, `fullgraph=True`, `dynamic=False`
   - compile 호출 자체가 아니라 첫 warmup/inference에서 실제 컴파일이 시작됨
   - SST-2 `(1,128)`, SQuAD `(1,384)` 고정 계약
   - 프로세스 메모리 캐시로 취급하고 영구 FXB와 구분
4. 실패 모델 기록
   - 모델별 마지막 확인 단계와 대표 오류 범주만 기록
   - 성공 명령이나 지원 모델로 오해할 표현은 사용하지 않음
   - 작은 isolated graph 성공은 전체 모델 지원으로 판정하지 않음

## 안전 장치와 판정 기준

- `--dry-run` 성공은 bucket/config 검증일 뿐 컴파일 성공으로 기록하지 않는다.
- 모든 kernel이 성공하고 `Artifact Build Completed`가 출력되어야 FXB 성공으로 판정한다.
- BERT는 strict compile 뒤 첫 inference 결과가 반환되어야 성공으로 판정한다.
- 사전 배포 artifact를 로드한 경우에는 직접 컴파일 성공으로 기록하지 않는다.
- 정확한 hash를 보관하지 않은 artifact 경로는 예시 경로로 표시하고, 재현 시 `fxb show`와 SDK 버전을 다시 기록하도록 안내한다.

## 검증

- 문서 코드 블록의 shell 문법을 `bash -n`으로 검사한다.
- 문서에 모델별 상태, `fxb build`, strict `torch.compile`, 실패 모델과 artifact 재사용 절이 모두 존재하는지 검색한다.
- 링크와 경로가 현재 저장소 구조를 가리키는지 확인한다.

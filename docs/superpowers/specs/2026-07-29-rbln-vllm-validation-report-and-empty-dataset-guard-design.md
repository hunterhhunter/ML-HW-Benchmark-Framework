# RBLN vLLM ATOM 검증 보고서와 빈 데이터셋 방어 설계

## 목적

PR #40의 Rebellions vLLM 통합을 Main에 병합하기 전에 다음 두 결과를
완성한다.

1. 실제 단일 RBLN-CA22에서 수행한 Llama 3.2 3B 및 Llama 3.1 8B의
   컴파일, 동기 E2E, 비동기 큐 검증 과정을 재현 가능한 문서로 남긴다.
2. 명시한 데이터셋 경로가 없거나 로더가 샘플을 하나도 만들지 못했을 때
   벤치마크가 성공 CSV를 저장하지 못하도록 실행 전에 차단한다.

이번 변경은 기존 정적 RBLN 모델의 실행 계약을 바꾸지 않는다. 정적
ResNet50, YOLOv5m, BERT, PatchTST 결과는 기존 RBLN 문서를 링크하고 이
보고서에 중복 수록하지 않는다.

## 확인된 문제

사용자가 이전 worktree의 존재하지 않는 SQuAD 경로를 `--dataset`으로
지정했다. 자동 준비 로직은 해당 경로가 없음을 감지했지만, 준비 스크립트는
현재 worktree의 기본 위치에 SQuAD를 다운로드했다. 이후 `args.dataset`은
처음 지정한 경로를 계속 가리켰다.

Llama 로더는 파일이 없다는 경고만 출력하고 `total_samples=0`인 상태로
생성되었다. 동기 벤치마크는 추론을 한 번도 실행하지 않았지만 정확도 0,
샘플 0인 결과를 저장하고 종료 코드 0을 반환했다. NPU 메모리도 할당되지
않았으므로 이 실행은 하드웨어 검증 결과로 사용할 수 없다.

## 설계 선택

### 선택: 자동 준비 후조건과 로더 샘플 수를 모두 검증

두 경계에서 방어한다.

- 자동 준비 경계: 준비 스크립트가 성공한 뒤 사용자가 요청한 정확한
  `args.dataset` 경로가 실제로 생성됐는지 확인한다.
- 로더 경계: `total_samples`를 정수로 제공하는 로더에서 값이 0이면 런타임
  생성과 벤치마크 실행 전에 명확한 오류를 발생시킨다.

첫 검사는 잘못된 출력 위치를 원인에 가까운 지점에서 설명한다. 두 번째
검사는 파일은 있지만 파싱 결과가 비었거나 필터링으로 샘플이 사라진 경우를
막는다. `total_samples`를 제공하지 않는 스트리밍·사용자 정의 로더에는 새
제약을 적용하지 않는다.

### 검토했지만 선택하지 않은 대안

- Llama 로더만 수정: 현재 문제는 자동 준비 계약도 깨졌기 때문에 원인을
  늦게 발견하고 다른 로더의 빈 데이터 문제를 막지 못한다.
- 문서에 사전 확인 명령만 추가: 운영자의 실수를 줄일 수 있지만 샘플 0을
  성공으로 기록하는 프레임워크 동작은 그대로 남는다.

## 코드 변경

### 자동 준비 후조건

`framework/src/main.py`의 데이터셋 자동 준비 경로는 준비 스크립트 종료 후
처음 요청한 `dataset_path`가 존재하는지 재검사한다. 존재하지 않으면 다음
정보를 포함한 오류를 발생시킨다.

- 요청한 데이터셋 경로
- 실행한 준비 스크립트
- 스크립트가 요청 경로를 생성하지 않았다는 사실

사용자가 명시한 다른 worktree 경로를 현재 worktree의 기본 경로로
조용히 바꾸지 않는다. 경로를 자동 추측하거나 파일을 복사하지 않는다.

### 빈 로더 방어

데이터로더 생성 직후, 런타임 생성 전에 작은 검증 함수를 호출한다.

- `total_samples` 속성이 정확한 정수이고 0이면 `ValueError`를 발생시킨다.
- 양수이면 통과한다.
- 속성이 없거나 `None`이면 기존 동작을 유지한다.
- bool은 정수로 취급하지 않는다.

오류 메시지에는 모델 이름, 태스크와 데이터셋 경로를 포함해 재현 가능한
진단을 제공한다. 이 오류가 발생하면 엔진 생성, warmup, 모니터링 측정 및
결과 저장이 시작되지 않아야 한다.

## 테스트 설계

`framework/tests/test_main_paths.py`에 다음 회귀 테스트를 추가한다.

1. 준비 스크립트가 성공 반환해도 명시 데이터셋 경로가 생기지 않으면
   자동 준비가 실패한다.
2. 준비 스크립트가 정확한 경로를 생성하면 기존 자동 준비가 통과한다.
3. `total_samples=0`인 로더는 실행 전에 거부된다.
4. `total_samples>0`, 속성 없음, `None`인 로더는 기존처럼 통과한다.
5. 빈 Llama 경로가 성공 결과 저장으로 이어지지 않는 CLI 또는 조립 경로를
   검증한다.

최종 검증은 RBLN vLLM, 결과 저장, Main 경로 및 정적 RBLN 집중 회귀 묶음과
전체 `framework/tests`를 모두 실행한다.

## 검증 보고서

새 문서 `framework/docs/rbln-vllm-atom-validation.md`를 추가한다.

문서는 다음을 포함한다.

- 검증 장치: RBLN-CA22 한 장, 16,096 MiB, KMD/firmware 3.2.2
- Python 3.10.12와 검증된 RBLN/Optimum/vLLM 패키지 조합
- hybrid virtual environment 구성과 user-site의 `rebel` 사용 이유
- Llama 3.2 3B 및 Llama 3.1 8B 단일 NPU 컴파일 명령
- manifest 계약: 1 device, sequence/block 512, batch/decoder batch 1,
  `unsupported_single_npu_experiment`
- Hugging Face gated 모델 인증, Rebellions wheel 인증 및 uv index 해석 문제
- manifest 없는 artifact 거부와 컴파일·실행 계약 일치 방법
- SQuAD worktree 경로 불일치와 0-sample 오검증의 원인·해결
- 실제 동기·비동기 실행 결과, run ID, 메모리, 지연, 처리량 및 종료 상태
- `rbln-smi -j`의 `contexts: []` 확인 절차
- 단일 카드 경로가 Rebellions 공식 지원 구성이 아니라는 명시

기록할 실제 8B 결과는 다음과 같다.

- 동기 run ID `a3168997`: 1 sample, 1 generated token, 203.8591 ms,
  4.9053 tokens/s, NPU memory peak 14,630 MiB, process RAM peak
  21,063.23 MiB, monitor coverage 1.0, exit 0, contexts empty.
- 비동기 run ID `9dd3bf7a`: 4/4 completed, 0 failed/rejected/timed out,
  engine TTFT average 202.7706 ms, completed tokens 4.8801/s, request E2E
  p99 604.8718 ms, service p99 214.3395 ms, NPU memory peak 14,630 MiB,
  native async error counters 0, status valid, exit 0, contexts empty.

토큰을 하나만 생성했으므로 TPOT/ITL이 0 또는 `None`인 것은 정상이며 품질
점수는 모델 품질 비교 자료로 사용하지 않는다고 기록한다.

## 기존 문서와 PR 갱신

`framework/docs/rbln-vllm-setup.md`의 하드웨어 상태를 다음과 같이 바꾼다.

- 3B뿐 아니라 8B도 단일 ATOM에서 compile, sync, async smoke 성공
- 상세 결과는 새 검증 보고서로 링크
- 지속 성능 비교에는 multi-token·장시간 측정이 추가로 필요

PR #40 본문도 같은 상태로 갱신하되 단일 카드 실험 분류는 유지한다. 코드와
문서 변경을 push한 뒤에도 사용자가 Main 병합을 명시하기 전에는 PR을 직접
병합하지 않는다.

## 커밋하지 않는 자료

- `.rbln` artifact
- 모델 weight와 tokenizer 대용량 파일
- 데이터셋과 전처리 캐시
- 원본 결과 CSV, trace 및 서버 로그
- 인증 토큰과 portal 자격 증명

문서에는 재현 명령, run ID와 검증에 필요한 요약 지표만 기록한다.

## 완료 기준

- 빈 데이터셋이 성공 결과로 저장되지 않는다.
- 자동 준비 경로 불일치가 정확한 요청 경로를 포함한 오류로 종료된다.
- 기존 정상 데이터셋 및 기존 모델 테스트가 회귀하지 않는다.
- 단일 ATOM 3B/8B 컴파일과 실행 과정이 새 문서만으로 재현 가능하다.
- 실제 8B 동기·비동기 결과와 비공식 지원 경계가 문서와 PR에 일치한다.
- 전체 테스트와 `git diff --check`가 통과한다.

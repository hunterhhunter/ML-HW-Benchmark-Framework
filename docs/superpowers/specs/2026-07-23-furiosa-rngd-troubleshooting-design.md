# Furiosa RNGD 트러블슈팅 문서 설계

## 목적

Furiosa RNGD 서버에서 `ML-HW-Benchmark-Framework`로 모델을 준비하고 추론·벤치마크를 실행하면서 실제로 확인한 장애와 해결 절차를 한 문서에 정리한다. 문서의 앞부분은 서버 작업자가 오류 메시지로 빠르게 대응할 수 있는 Runbook으로, 뒷부분은 프레임워크 개발자가 원인과 개선 지점을 이해할 수 있는 분석 자료로 사용한다.

최종 문서는 `docs/furiosa-rngd-troubleshooting.md`에 작성하고 기존 정상 실행 절차인 `docs/furiosa-rngd-setup.md`, 논문용 serving 절차인 `docs/rngd-paper-benchmark.md`와 상호 링크한다.

## 대상 독자

- RNGD가 장착된 Ubuntu 서버를 인수해 드라이버부터 확인해야 하는 작업자
- Llama 3.1 8B와 Llama 3.2 3B의 E2E·native async 벤치마크를 재현하는 작업자
- Furiosa runtime, model/FXB 입력 계약, async 실행기, monitor plugin을 유지보수하는 프레임워크 개발자
- Furiosa Torch 비전 모델 지원 여부와 현재 SDK 한계를 판단해야 하는 개발자

## 범위

문서에는 다음 내용을 포함한다.

1. 검증 서버의 OS, 커널, RNGD, driver, firmware, Python 및 Furiosa SDK 버전
2. PCIe 인식부터 `furiosa-smi`까지의 초기 점검 절차
3. dirty worktree를 보존하면서 최신 main 코드와 기존 모델·데이터·가상환경을 분리해 사용하는 방법
4. 벤더별 Python 가상환경 분리와 optional dependency의 eager import 문제
5. Hugging Face 모델 디렉터리, legacy artifact, FXB의 차이와 프레임워크 입력 계약
6. Llama 3.1 artifact schema 불일치와 `Primitive` 역직렬화 오류
7. Llama 3.2 3B FXB 빌드와 실행 과정에서 발생한 `head_dim`, cross compiler, negative compiler cache, `lm_head.weight` 문제
8. `attention_mask` 누락으로 인한 pre-tokenized `BatchEncoding` 실패
9. native async worker capability, backpressure timeout, 진행 로그 부재, single-stream과 동시성 결과 해석
10. `furiosa-smi`를 이용한 상태 확인과 현재 프레임워크의 RNGD 전력 collector 부재
11. ResNet50 ONNX 변환 성공 이후 Furiosa Torch compiler 내부 panic으로 중단된 상태
12. 검증된 Llama 실행 명령과 성공 판정 기준

YOLOv5m, BERT, PatchTST처럼 adapter 또는 준비 코드만 있고 RNGD 실추론이 끝나지 않은 모델은 지원 완료로 표현하지 않는다. 기존 사용자 결과 파일이나 모델 파일을 문서 작업 중 변경하지 않는다.

## 문서 구성

### 1부: 운영 Runbook

1. 적용 환경과 모델 지원 상태표
2. 5분 초기 점검
3. 오류 메시지 빠른 색인
4. 드라이버·장치 인식
5. Git·worktree·가상환경
6. 모델·artifact·FXB
7. Llama 3.1 문제 해결
8. Llama 3.2 빌드·로딩 문제 해결
9. 프레임워크 import·입력 문제 해결
10. async 멈춤·timeout·worker 문제 해결
11. SMI 상태·전력 확인
12. 검증된 실행 명령과 성공 기준

### 2부: 개발자 분석

1. 프레임워크 공통 추론 파이프라인과 Furiosa runtime 경계
2. 모델 경로와 compiled artifact의 입력 계약
3. native async 실행과 지표 의미
4. Furiosa monitor plugin의 현재 공백
5. Furiosa Torch 비전 컴파일 실패 분석
6. 후속 개선 과제와 완료 조건

## 장애 항목 형식

모든 장애 항목은 가능한 한 다음 형식을 사용한다.

1. **증상**: 실제 오류의 핵심 한두 줄과 발생 단계
2. **원인**: 확인된 직접 원인과 상위 원인
3. **확인**: 원인을 다른 장애와 구분하는 읽기 전용 명령
4. **해결 또는 우회**: 재현 가능한 명령과 적용 범위
5. **성공 기준**: 기대 로그, 결과 컬럼 또는 종료 상태
6. **상태**: 해결, 우회, 미해결, 현재 SDK 한계 중 하나
7. **개발자 메모**: 장기 수정이 필요한 코드 경계

오류 전문은 반복하지 않고 검색 가능한 핵심 문자열만 인용한다. 모델 경로와 사용자 이름은 변수 또는 예시 경로로 표현한다.

## 사실성과 상태 표기

- 서버 출력이나 성공한 벤치마크로 확인한 내용은 **검증 완료**로 표기한다.
- 로그에 근거하지만 별도 대조 실험이 필요한 설명은 **추정**으로 표기한다.
- compiler panic처럼 우회가 검증되지 않은 항목은 **미해결** 또는 **현재 SDK 한계**로 표기한다.
- unit test 통과와 실제 RNGD 실행 성공을 구분한다.
- TDP 150 W를 실측 전력처럼 사용하지 않는다. 전력이 수집되지 않은 기존 실행은 `미수집`으로 기록한다.
- Llama 3.2 3B의 정확한 registry 지원은 fallback 경고가 존재하므로 공식 지원으로 단정하지 않는다.

## 명령 작성 원칙

- 기본 실행 위치, Python, dataset, model, FXB를 셸 변수로 먼저 정의한다.
- 최신 main 코드는 별도 worktree에서 실행하되 모델·데이터·가상환경은 기존 저장소의 절대경로로 참조할 수 있음을 설명한다.
- dirty worktree에서 `checkout`, `reset`, 무조건적인 `pull`을 지시하지 않는다.
- 긴 벤치마크 전 `max-steps=1` 또는 `max-samples=1` smoke test를 둔다.
- async 비교에서는 `queue-capacity=1`, `worker-count=1` single-stream과 동시성 실험을 구분한다.
- `max-new-tokens`는 고정 출력 길이가 아니라 상한이라는 점을 명시하고, 모델 비교에는 samples/s와 함께 tokens/s·TTFT·TPOT을 사용한다.

## 검증 방법

문서 작성 후 다음을 확인한다.

1. 문서 내부의 로컬 Markdown 링크가 실제 파일을 가리키는지 확인한다.
2. 명령 옵션이 현재 `src/main.py`와 `furiosa_llm_rt.py`의 계약과 일치하는지 대조한다.
3. 모델별 상태표가 실제 결과와 모순되지 않는지 확인한다.
4. 미완성 표식, 임시 문구와 개인 환경에만 유효한 경로가 남지 않았는지 검색한다.
5. 새 문서만 변경되었는지 확인하고 기존 사용자 변경을 커밋에 포함하지 않는다.

문서 작성 자체는 RNGD 하드웨어 재실행을 요구하지 않는다. 이미 확보한 서버 로그와 CSV를 근거로 작성하며, 하드웨어에서 다시 확인하지 않은 절차는 검증 완료로 승격하지 않는다.

## 산출물

- 설계 명세: `docs/superpowers/specs/2026-07-23-furiosa-rngd-troubleshooting-design.md`
- 최종 Runbook 및 개발자 분석: `docs/furiosa-rngd-troubleshooting.md`
- 필요하면 기존 `docs/furiosa-rngd-setup.md`에 최종 문서 링크 한 줄을 추가하되, 별도 변경으로 명확히 기록한다.

# Mobilint 아티팩트 생성·입수 이력 문서 설계

> 이 설계는 실제 다중 모델 컴파일 실험까지 PR 범위가 확대되면서
> `2026-08-03-mobilint-multi-model-compilation-experiment-design.md`로 대체됐다.

## 목적

저장소에서 사용하는 Mobilint 모델을 한곳에서 찾을 수 있게 하고, 각 아티팩트가
직접 컴파일된 것인지, 공식 사전 컴파일 배포본인지, 컴파일 과정이 아직 확인되지
않은 기존 파일인지를 명확히 구분한다. 확인하지 않은 과정을 재현 가능한 컴파일
절차로 표현하지 않는다.

## 범위

이번 변경은 전체 현황과 검증된 경로를 설명하는 문서를 추가한다. 다음 모델군을
포함한다.

- BERT SST-2, BERT SQuAD v1
- PatchTST ETTh1
- ResNet50 ImageNet1K V2
- YOLOv5m
- Llama 3.1 8B, Llama 3.2 3B 및 지원되는 배치 변형

PatchTST, ResNet50, YOLOv5m용 새 qbcompiler recipe는 이번 문서 변경에 포함하지
않는다. 해당 recipe는 Ubuntu 22.04 컴파일 서버에서 실제 컴파일과 ARIES 로딩을
검증한 뒤 별도 변경으로 추가한다.

## 접근 방식

### 채택: 단일 현황 문서와 검증된 상세 문서 연결

`docs/mobilint-artifacts.md`를 정본 인덱스로 추가한다. 모델별 행에는 다음 정보를
기록한다.

- 모델 및 벤치마크 작업
- 원본 모델 또는 배포 저장소
- 아티팩트 생성 방식
- 현재 재현 수준
- 입력·출력 계약
- 컴파일 또는 다운로드 상세 문서
- ARIES 검증 상태

직접 컴파일, 공식 사전 컴파일 배포, 출처 미확인 기존 아티팩트를 서로 다른 상태로
표시한다. BERT 행은 `docs/mobilint-bert-compilation.md`에 연결하고, 실행 계약은
기존 ARIES 문서에 연결한다.

### 검토했지만 채택하지 않은 방식

1. 모든 내용을 BERT 컴파일 문서에 추가: 문서 이름과 책임이 맞지 않고 공식 배포
   LLM까지 컴파일 대상으로 오해하게 한다.
2. 검증되지 않은 PatchTST·비전 컴파일 명령을 추정해 추가: 재현성을 높이는 대신
   잘못된 명령과 아티팩트 계약을 고정할 위험이 있다.

## 모델별 사실 상태

| 모델군 | 생성·입수 방식 | 이번 문서의 상태 |
|---|---|---|
| BERT SST-2 / SQuAD v1 | qbcompiler 1.2로 직접 컴파일 | 재현 가능 |
| PatchTST ETTh1 | 기존 MXQ와 런타임 계약은 있으나 Mobilint 컴파일 이력 미확인 | 컴파일 recipe 검증 필요 |
| ResNet50 / YOLOv5m | 기존 MXQ 다운로드·ARIES 추론 이력은 있으나 원본 컴파일 이력 미확인 | 외부 아티팩트, 컴파일 recipe 검증 필요 |
| Llama 3.1 / 3.2 | Mobilint Model Zoo의 사전 컴파일 배포본 사용 | 다운로드·배치 준비 재현 가능, 로컬 컴파일 대상 아님 |

## 문서 구조

`docs/mobilint-artifacts.md`는 다음 순서로 구성한다.

1. 상태 용어 정의
2. 전체 모델 현황표
3. 직접 컴파일 모델
4. 기존 MXQ 및 컴파일 recipe 미확인 모델
5. 공식 사전 컴파일 모델
6. 새 컴파일 recipe를 추가할 때 충족할 검증 기준

기존 `docs/mobilint-aries-transformers.md`와
`docs/mobilint-aries-troubleshooting.md`에는 정본 현황 문서 링크만 추가한다. 실행법을
중복 복사하지 않는다.

## 검증 기준

문서가 저장소의 실제 코드 및 설정과 일치하는지 다음 항목으로 검사한다.

- BERT 컴파일 script 및 상세 문서 링크가 존재한다.
- PatchTST, ResNet50, YOLOv5m을 직접 컴파일 완료로 표시하지 않는다.
- Llama 모델은 `framework/models/prepare_mobilint_llm.py`의 공식 저장소 ID와
  일치한다.
- 비전 입력·출력 계약은 Mobilint profile 및 기존 ARIES 문서와 일치한다.
- PatchTST 계약은 기존 ARIES 실행 문서와 일치한다.
- Markdown 링크와 코드 경로가 모두 유효하다.

## 후속 작업

1. PatchTST를 qbcompiler 1.2 환경에서 컴파일하고 ARIES에서 계약과 결과를 검증한다.
2. 성공한 명령만 저장소 recipe와 상세 문서로 추가한다.
3. ResNet50과 YOLOv5m도 동일한 순서로 검증한다.
4. 검증이 끝난 모델의 현황 상태를 `재현 가능`으로 갱신한다.

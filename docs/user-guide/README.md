# 벤치마크 프레임워크 사용자 설명서

`main@30c4db75`의 CLI와 `feat/web-cli-parity` 통합 후의 웹 화면을 기준으로,
환경 준비부터 결과 확인까지 순서대로 안내한다. 모델별 세부 명령과 벤더 SDK 설치 절차는
각 장의 연결 문서를 따른다.

| 순서 | 문서 | 내용 |
|---|---|---|
| 01 | [환경 구축 및 설치](01-installation.md) | Python·웹 환경과 벤더 SDK 준비 |
| 02 | [NPU·모델·데이터셋 설정](02-configuration.md) | target과 실행 자산 선택 |
| 03 | [벤치마크 실행](03-execution.md) | 소량 확인, 본 실행, 웹 화면 |
| 04 | [결과 확인 및 해석](04-results.md) | CSV, 상태, 지표의 측정 경계 |
| 05 | [오류 대응](05-troubleshooting.md) | 준비·실행·결과 오류 확인 |
| 06 | [실험 재현](06-reproduction.md) | 같은 조건으로 다시 실행하는 절차 |

결과를 비교하거나 인용하기 전에는 [측정 계약](../../framework/docs/measurement-contract.md)을
확인한다. 지원 목록에 있는 target과 모델의 모든 조합이 검증된 것은 아니다.

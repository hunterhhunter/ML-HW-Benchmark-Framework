# 벤더별 문서

장치별 설치·실행 절차와 컴파일·문제 해결 기록을 모았다. SDK 버전과 검증 시점은
각 문서에 적힌 조건을 확인한다. 공통 실행 방법은 [사용자 설명서](../user-guide/README.md),
확인된 에지 조합은 [모델별 실행 명세](../../framework/docs/model-execution-specs/resnet50.md),
Runtime 구현 위치는 [개발자 벤더 구현 지도](../developer-guide/10-vendor-implementation-map.md)를 따른다.

## Hailo

- 실행·비동기: [Hailo-8/10H Runtime](hailo/hailo-async-runtime.md)

## DEEPX

- 설치·실행: [DX-COM·DX-RT 가이드](deepx/deepx-setup.md)

## Mobilint

- 실행: [Regulus Runtime](mobilint/mobilint-regulus-runtime.md) · [ARIES Transformer·LLM](mobilint/mobilint-aries-transformers.md)
- 컴파일: [BERT 재현 가이드](mobilint/mobilint-bert-compilation.md) · [qbcompiler 실험 기록](mobilint/mobilint-compilation-experiments.md)
- 문제 해결: [ARIES 기록](mobilint/mobilint-aries-troubleshooting.md)
- 검증 기록: [Regulus 정확도](mobilint/regulus-runtime.md)
- 과거 경로 안내: [maccel 명칭](mobilint/mobilint-accel-runtime.md)

## Furiosa

- 설치·실행: [RNGD Runtime](furiosa/furiosa-rngd-setup.md)
- 문제 해결: [RNGD Runbook](furiosa/furiosa-rngd-troubleshooting.md)
- 컴파일 실패 재현: [strict compile 기록](furiosa/furiosa-rngd-compilation-troubleshooting.md)

## Rebellions

- 설치·실행: [RBLN 정적 모델](rebellions/rbln-setup.md) · [ATOM vLLM](rebellions/rbln-vllm-setup.md)
- 컴파일: [RBLN 모델 재현 가이드](rebellions/rbln-compilation.md)
- 문제 해결: [RBLN-CA22 기록](rebellions/rbln-troubleshooting.md)
- 검증 기록: [ATOM Llama vLLM](rebellions/rbln-vllm-atom-validation.md)

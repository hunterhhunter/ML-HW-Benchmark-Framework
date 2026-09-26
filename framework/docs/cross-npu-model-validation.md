# Cross-NPU Transformer·시계열 모델 검증 현황

이 문서는 2026-09-26까지 Furiosa RNGD, Rebellions CA22, Mobilint ARIES에서
확인한 Llama 3.1 8B, Llama 3.2 3B, BERT SQuAD v1, IBM Granite TTM-R2의
검증 상태를 한곳에서 조회하기 위한 중앙 ledger다. 상세 컴파일 및 실행 절차는
벤더별 문서에 유지하고, 이 문서에는 공통 판정 기준, 비교 경계, 증거 수준과
아직 확보하지 못한 항목만 기록한다.

이 표의 목적은 지원 여부를 한 단어로 축약하는 것이 아니다. artifact가 존재한다는
사실, 컴파일, 실장비 실행, 태스크 품질, full benchmark와 전력 측정을 서로 다른
게이트로 판정한다.

## 1. 상태 용어

| 상태 | 의미 |
|---|---|
| `artifact_verified` | artifact가 존재하고 요구되는 경우 SHA-256 또는 manifest를 확인했다. |
| `compiled` | 해당 모델 전체 graph 또는 벤더 artifact 컴파일이 완료됐다. dry-run이나 부분 graph 성공은 제외한다. |
| `device_smoke_passed` | 실제 NPU에서 하나 이상의 유효한 출력이 반환되고 프로세스가 정상 종료됐다. |
| `task_quality_measured` | 고정 evaluator로 태스크 품질을 계산했다. sample 수와 metric 정의를 함께 읽어야 한다. |
| `full_benchmark_passed` | 사전에 정한 전체 sample 수, 결과 저장, 오류 회계와 종료 조건을 모두 만족했다. |
| `conditional` | 실행은 확인했으나 공식 지원 구성, production preset 또는 동일한 측정 계약이 아니다. |
| `legacy_evidence` | 현재 strict attempt schema 또는 현재 `main.py` 이전에 얻은 결과다. 현재 결과처럼 승격하지 않는다. |
| `blocked` | 다음 게이트로 진행하는 데 필요한 코드, sidecar, evaluator 또는 artifact가 없다. |
| `evidence_pending` | 성공 실행이 보고됐지만 이 중앙 ledger에 run ID, CSV, 로그, hash가 아직 묶이지 않았다. |

`parity_failed`는 태스크 품질 실패와 동의어가 아니다. 같은 입력에 대한 CPU와 NPU
tensor가 지정된 `rtol`/`atol` 안에서 일치하지 않았다는 진단 결과다. 운영 성공과
태스크 품질은 각각 별도로 판정한다.

## 2. 모델 × NPU 현황

| 모델 | Furiosa RNGD | Rebellions CA22 | Mobilint ARIES |
|---|---|---|---|
| Llama 3.1 8B | `full_benchmark_passed`: SQuAD2 E2E 1,000건과 native async 1,000건 | `device_smoke_passed`, `conditional`: 비공식 one-card 구성, sync 1건·async 4건, 1 token lifecycle 검증 | `evidence_pending`: 실행 경로와 서버 성공 보고는 있으나 중앙 ledger용 run ID·결과 bundle 미수집 |
| Llama 3.2 3B | `full_benchmark_passed`, `conditional`: E2E/native async 완료, custom FXB가 nearest preset fallback 사용 | `device_smoke_passed`, `conditional`: 비공식 one-card 구성, sync 1건·async 4건 | `evidence_pending`: 실행 경로와 서버 성공 보고는 있으나 중앙 ledger용 run ID·결과 bundle 미수집 |
| BERT SQuAD v1 | `compiled`, `device_smoke_passed`, `task_quality_measured`: 1-sample smoke만 완료 | `artifact_verified`, `compiled`, `device_smoke_passed`, `blocked`: 단일 sample semantic mapping은 확인했으나 전체 evaluator 미승인 | `legacy_evidence`, `task_quality_measured`: 64-sample 기록 존재. 현재 `main.py` 성공 결과 bundle은 `evidence_pending` |
| TTM-R2 | `full_benchmark_passed`, `parity_failed`: ETTh1 240 windows | `artifact_verified`, `full_benchmark_passed`, `parity_failed`: ETTh1 240 windows | `artifact_verified`, `full_benchmark_passed`: ETTh1 240 windows, saturation 0 |

따라서 “네 모델이 세 NPU에서 모두 실행됐다”는 운영 요약은 사용할 수 있지만,
“네 모델 모두 세 NPU에서 동일한 범위의 full 품질·성능 검증을 마쳤다”거나
“동일 정밀도와 동일 공식 지원 구성으로 비교했다”고 표현하면 안 된다.

## 3. 검증 환경 snapshot

가상환경은 벤더 SDK마다 분리한다. 동일 Python package 집합은 공정성 조건이 아니며,
각 벤더가 요구하는 SDK를 사용하되 모델 의미와 측정 경계를 고정한다.

| 경로 | 확인된 핵심 환경 | 적용 범위와 주의사항 |
|---|---|---|
| Furiosa-LLM | Python 3.12.13, `furiosa-llm==2026.3.0`, PyTorch 2.5.1 계열 | Llama 3.1/3.2. BERT·TTM-R2용 Furiosa Torch 환경과 분리한다. |
| Furiosa Torch | Python 3.12.13, `torch==2.10.0+cpu`, `furiosa-torch==2026.3.0`, `transformers==4.57.6`, `numpy==2.5.1` | 이 조합에서 BERT SQuAD v1 1-sample과 TTM-R2 1-step strict smoke가 성공했다. |
| Rebellions vLLM | Python 3.10.12, `rebel-compiler==0.11.0`, `optimum-rbln==0.11.0.post1`, `vllm-rbln==0.11.0`, `torch==2.11.0+cpu`, `transformers==5.8.1` | Llama one-card 실험. 공식 지원 구성은 두 모델 모두 8 NPU다. |
| Rebellions static | 서버의 `.venv-rbln` | BERT와 TTM-R2 precompiled `.rbln` 실행. 이 중앙 snapshot에는 전체 package freeze가 아직 없다. |
| Mobilint | 서버의 `.venv-mobilint` | BERT, TTM-R2, Llama 실행. BERT legacy 기록은 qbruntime 1.3.2, driver 1.13.0이다. 현재 전체 package freeze는 미수집이다. |

현재 저장소의 `framework/requirements-furiosa-torch.txt`와 환경 계약 테스트는
`transformers==5.1.0`을 고정한다. 반면 최근 서버 실측에서 BERT SQuAD v1과
TTM-R2를 함께 통과한 환경은 4.57.6이다. 이 불일치는 문서에서 숨기지 않으며,
requirements와 테스트 pin 변경은 별도 환경 계약 변경으로 수행한다.

## 4. TTM-R2 공통 계약과 결과

TTM-R2는 현재 네 모델 가운데 세 NPU의 full task-quality 계약이 가장 완전하게
고정된 항목이다.

서버에서 검증한 원 구현은 다음 milestone로 발전했다. 이 commit들은 현재 통합
브랜치의 직접 ancestor가 아니라, 이식 근거가 된 `feat/chronos-bolt-cross-vendor`
계열의 Git 객체다.

| revision | 역할 |
|---|---|
| `6343924` | 고정 TTM-R2 checkpoint 취득 |
| `1146996` | CPU 전처리/후처리와 NPU core 분리 |
| `df16ace` | 벤더별 strict compile dispatch |
| `cca3303` | Furiosa ETTh1 품질 측정 |
| `e50a642` | Rebellions ETTh1 품질 측정 |
| `d75e50d` | Mobilint ARIES calibration과 품질 측정 |
| `eeda7bf` | lowered ARIES ABI 기준 calibration 수정 |

현재 `feat/ttm-r2-framework-integration`에는 이 동작을 기존 framework 구조로 옮긴
다음 commit들이 있다.

| revision | 통합 내용 |
|---|---|
| `9b06356` | 검증된 TTM-R2 numerical core 이식 |
| `8079abb` | canonical profile 등록 |
| `1bcbd39` | canonical ETTh1 loader 추가 |
| `a187079` | raw-scale evaluator 추가 |
| `8373032` | Furiosa Torch runtime 연결 |
| `e31a32e` | ARIES MXQ semantic tensor adapter 추가 |
| `c296e65` | `main.py` 통합 |
| `6404759` | 3-target acceptance flow 문서화 |
| `7882f8d` | acceptance provenance 강화 |
| `1b8e1cb` | ARIES logical float input metadata 허용 |
| `3752406` | Furiosa profile에서 ONNX optional 처리 |
| `e4a0048` | task별 pipeline dependency lazy-load |

| 항목 | 고정값 |
|---|---|
| checkpoint | `ibm-granite/granite-timeseries-ttm-r2` 로컬 snapshot |
| 입력 / 출력 | FP32 `[1,512,1]` / FP32 `[1,96,1]` |
| 데이터 | ETTh1 `OT`, test split, stride 1, 240 windows |
| contract ID | `ttm-r2-etth1-ot-512-96-v1` |
| `config.json` SHA-256 | `5e2367547c103e92cb8ebc63cbd5ad4d7bf83facad5a2a2ec261fa770ed659d5` |
| `model.safetensors` SHA-256 | `a706726a7eb01bbcb42994b7dcb3c06ea9557898dbae8d480eb04fe8ccb89710` |
| `ETTh1.csv` SHA-256 | `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066` |
| RBLN artifact SHA-256 | `4159ce147a9d91524117b39eba67a706df1c5c987e3317a999e3992c2d3bf172` |
| ARIES artifact SHA-256 | `208958c81f1c62ad47557158b788f7f758ac8ef634032286e7f304dac752e217` |

| 실행 | MAE | RMSE | CPU 대비 MAE | CPU 대비 RMSE | 별도 진단 |
|---|---:|---:|---:|---:|---|
| CPU reference | 1.7683423758 | 2.1062438488 | - | - | 고정 reference |
| Furiosa RNGD | 1.7680668831 | 2.1058883667 | -0.015579% | -0.016878% | strict tensor `parity_failed` |
| Rebellions CA22 | 1.7690539360 | 2.1070954799 | +0.040239% | +0.040434% | strict tensor `parity_failed` |
| Mobilint ARIES | 1.8834694624 | 2.2656056881 | +6.510452% | +7.566163% | quantization unsaturated, 0/122,880 saturated elements |

세 실행은 모두 exit code 0과 task-quality 계산을 완료했다. Furiosa와 RBLN의
strict parity 실패를 숨기지 않되, 이를 MAE/RMSE 평가 실패로 바꾸어 적지 않는다.
Mobilint의 차이는 INT8 artifact와 boundary adapter를 포함한 결과이며 별도 품질
허용 기준을 정의하기 전에는 “동등”이라고 표현하지 않는다.

위 full 결과와 별개로 Furiosa `transformers==4.57.6` 환경에서는 `--max-steps 1`
strict compile·실행 smoke가 성공했다. 이는 4.57.6 호환성 증거이며 240-window full
결과를 새로 측정한 것으로 간주하지 않는다.

상세 실행과 판정 명령은 [TTM-R2 3종 NPU 실행 및 최종 판정](ttm-r2-framework.md)을
따른다.

## 5. BERT SQuAD v1

### Furiosa RNGD

2026-09-26 서버 실행에서 다음 조합을 확인했다.

| 항목 | 값 |
|---|---|
| target | `furiosa-rngd-torch` |
| 환경 | Torch 2.10.0+cpu, Furiosa Torch 2026.3.0, Transformers 4.57.6, NumPy 2.5.1 |
| sample | 1 |
| exact match / F1 | 100 / 100 |
| average / P99 | 14.3294 / 14.3294 ms |
| QPS | 69.7865 |
| run ID | `c5113b3b` |
| 결과 경로 | `/home/etri_ecas/bert-squad-v1-furiosa-tf4576-mMB1Ba/furiosa-squad-v1-smoke.csv` |

이 결과는 strict 첫 호출, RNGD 출력, evaluator 연결과 ResultStore를 확인한
1-sample operational smoke다. 전체 SQuAD 품질이나 안정적인 latency 분포를
증명하지 않는다. `bert.pooler.dense.*`의 `UNEXPECTED` load report는 QA head가
pooler를 사용하지 않아 발생한 비치명적 경고였다.

같은 BERT graph는 Transformers 5.1.0 및 5.14.1 환경에서
`eager fallback is not allowed`와 `furiosa.UnsupportedOpError: failed to compile the graph`
로 strict compile에 실패했다. 이 실패는 데이터셋이나 RNGD 미인식이 아니라 package
조합별 graph/compiler 호환성 기록으로 보존한다.

### Rebellions CA22

최종 3-input artifact SHA-256은
`caada10a3e055df43b24ac388e8fccb5b71fc8fe4a1c08c51dca91922a600b33`이다.
단일 sample에서 `output[0]=start_logits`, `output[1]=end_logits`와 동일 answer span을
확인했다. 그러나 배포 경로의 `model.rbln.json` sidecar가 없던 실행이 있었고,
현재 evaluator는 question, special, padding 위치를 제외하는 persisted context mask를
사용하지 않는다. strict logit parity와 전체 task-level 평가는 미승인 상태다.
상세 수치와 남은 evaluator 조건은
[Rebellions 트러블슈팅](rbln-troubleshooting.md#7-bert-squad)을 따른다.

### Mobilint ARIES

legacy artifact SHA-256은
`5d1ff5a263a15b49e62a4d14fdfbfd9e261a7b114f65a2541c6d0d3bf54d03a2`이며,
64-sample standalone 문자열 평가에서 EM 0.828125, F1 0.886318, 평균 runtime
20.9896 ms를 기록했다. 이전 framework token-coordinate 평가와 metric 정의가
다르므로 두 값을 직접 비교하지 않는다. 현재 `main.py` 실행 성공 보고는 새 CSV,
run ID, package snapshot을 모은 뒤 strict/current evidence로 승격한다.
legacy attempt와 metric 정의는
[Mobilint 컴파일 실험 기록](../../docs/mobilint-compilation-experiments.md#9-관측-결과-ledger)을
따른다.

## 6. Llama 3.1 8B와 Llama 3.2 3B

### Furiosa RNGD

- Llama 3.1 8B는 Furiosa 배포 legacy artifact/repository 경로에서 SQuAD2 E2E
  1,000건과 native async 1,000건을 완료했다.
- Llama 3.2 3B는 local weights와 custom FXB로 E2E/native async를 완료했다.
  다만 exact registry entry가 없어 nearest preset fallback을 사용했으므로
  `conditional`로 유지한다.

세부 결과는 [Furiosa RNGD 트러블슈팅](../../docs/furiosa-rngd-troubleshooting.md)을
따른다.

### Rebellions CA22

두 모델 모두 공식 8-NPU 구성이 아니라 한 장 CA22의
`unsupported_single_npu_experiment`다.

| 모델 | sync run | async run | 검증 범위 |
|---|---|---|---|
| Llama 3.2 3B | `b7808504`, 1 sample, 2 tokens | `a307b84f`, 4 samples, 37 tokens | engine 실행과 async 회계. 공식 지원 구성이나 full 품질 검증 아님 |
| Llama 3.1 8B | `a3168997`, 1 sample, 1 token | `9dd3bf7a`, 4 samples, 4 tokens | capacity, lifecycle, context cleanup. 1 token이므로 TPOT·품질 판정 불가 |

세부 artifact와 runtime 계약은
[Rebellions ATOM Llama vLLM 검증 보고서](rbln-vllm-atom-validation.md)를 따른다.

### Mobilint ARIES

두 모델의 동기 및 async 실행 명령과 artifact 준비 계약은
[Mobilint ARIES Transformer·LLM 실행 가이드](../../docs/mobilint-aries-transformers.md)에
있다. 현재 중앙 ledger에는 성공 실행의 run ID, 전체 sample 수, tokenizer/checkpoint
hash와 결과 CSV가 아직 없다. 이를 확보하기 전에는 `evidence_pending`으로 유지한다.

## 7. 공정한 비교를 위한 고정 계약

벤더마다 compiler, runtime ABI, 양자화와 host adapter가 다르므로 동일 binary나 동일
Python 환경을 요구하지 않는다. 대신 다음 의미 경계를 고정해야 한다.

1. 모델 repository와 revision, tokenizer, checkpoint hash를 고정한다.
2. 데이터셋 file hash, split, sample 수와 순서를 고정한다.
3. 입력 길이, batch size, warmup과 worker 수를 고정한다.
4. LLM은 prompt template, decoding, EOS/stop과 `max_new_tokens`를 고정한다.
5. 동일 evaluator와 metric 정의를 사용한다.
6. host 전처리·후처리와 CPU subgraph를 공개한다.
7. precision, quantization, NPU 수, tensor parallel과 sequence limit를 공개한다.
8. 성능 비교의 주 경계는 공통 `System E2E`로 두고, vendor runtime-call은 진단값으로
   분리한다.

예를 들어 Mobilint BERT는 host에서 embedding을 계산한 뒤 ARIES artifact에 전달할 수
있다. 이 host 구간이 runtime-call에 포함되지 않으면 다른 벤더의 runtime-call과 직접
비교하지 않는다. 반대로 System E2E에는 공통 runner가 관측한 데이터 준비, adapter,
runtime과 evaluator 경계가 모두 반영되어야 한다.

같은 서버에 세 NPU가 장착됐다는 사실은 host 조건 통제에 유리하지만, 서로 다른
artifact precision, NPU 수, offload 경계와 vendor runtime 차이를 제거하지는 않는다.
따라서 이 결과는 고정 workload에서의 vendor-native deployment 비교이지, 동일한
저수준 연산 조건의 순수 NPU-core 순위표가 아니다.

## 8. 전력 상태

전력 수집기는 구현 중이다. 공통 측정 경계가 검증되기 전까지 cross-NPU 표에는
`unavailable`을 사용하며 0 W 또는 추정값을 채우지 않는다. 최종 전력 기록에는 최소한
다음 항목이 필요하다.

- telemetry source와 측정 domain: chip, board/card 또는 system
- sample timestamp와 interval
- idle baseline과 안정화 구간
- measurement 시작·종료 경계
- 성공 sample 수와 monitor coverage
- 평균·최대 전력, energy J와 idle-subtracted energy
- System E2E sample 구간과의 정렬 방식

서로 다른 telemetry domain을 같은 `power_w` 열에 넣어 직접 비교하지 않는다.

## 9. 남은 증거 수집 순서

1. Furiosa 4.57.6 환경에서 TTM-R2 240-window full CSV를 새로 저장한다.
2. Furiosa BERT SQuAD v1을 전체 또는 사전 고정한 sample 수로 실행한다.
3. Mobilint BERT SQuAD 현재 `main.py` CSV, run ID, artifact hash와 package freeze를 묶는다.
4. Rebellions BERT sidecar를 복구하고 context-masked evaluator로 전체 품질을 재검증한다.
5. Mobilint Llama 3.1/3.2의 run ID, artifact/tokenizer hash와 smoke/full 범위를 수집한다.
6. Llama 세 벤더의 prompt·sample order·generation parameter가 같은지 manifest로 고정한다.
7. 공통 전력 collector의 domain과 System E2E 정렬을 검증한 뒤에만 전력 비교표를 연다.

새 결과가 생기면 기존 행을 덮어쓰지 않고 날짜, commit, 환경 snapshot과 run ID를 함께
추가한다. 실패 attempt도 삭제하지 않으며 성공한 후속 실행과 별도로 보존한다.

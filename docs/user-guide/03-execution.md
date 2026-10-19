# 03. 벤치마크 실행

[목차](README.md) · [이전: NPU·모델·데이터셋 설정](02-configuration.md) · [다음: 결과 확인 및 해석](04-results.md)

## CLI: 소량 확인

선택한 장비에서 준비, Runtime load, 추론 호출이 끝나는지 먼저 확인한다.
다음은 CPU에서 ResNet50 한 step을 실행하는 예시다. 모델과 ImageNet 데이터가
없으면 자동 준비가 시작될 수 있다. 이 실행의 정확도·처리량은 최종 결과로
사용하지 않는다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/framework
source .venv/bin/activate
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 0 --max-steps 1 \
  --results-path results/user-guide-smoke.csv
```

## 실행 모드 선택

| 목적 | CLI 옵션 | 확인할 결과 |
|---|---|---|
| 순차 실행과 기본 동작 확인 | `--inference-mode e2e` | 명령 종료 상태와 저장된 CSV 행 |
| 품질 평가와 비동기 완료 처리량 확인 | `--inference-mode async_queue --async-pass quality-and-runtime` | 품질·성능 패스와 완료·실패 수 |
| 품질 평가를 생략한 런타임 부하 진단 | `--inference-mode async_queue --async-pass runtime-only` | 성능 패스만 확인. `quality_status=skipped`가 예상 상태 |

`e2e`가 기본 모드이며, `quality-and-runtime`은 비동기 기본 패스다.
`async_queue`는 해당 target이 지원할 때 사용한다. 모드별 시간 측정 구간은
[측정 계약](../../framework/docs/measurement-contract.md)에 있다.

## CLI: 본 실행

모델·NPU 조합은 [설정 문서](02-configuration.md)에 연결된 실행 명세의
모델·데이터·artifact 설정과 표본 수, warmup 조건을 따른다. 다음 명령은 준비된
CPU 프로필의 전체 데이터를 실행하는 기본 예시다.

```bash
python src/main.py --model resnet50 --target cpu \
  --inference-mode e2e --warmup 2 \
  --results-path results/resnet50-cpu-e2e.csv
```

동기 `e2e`의 `--max-steps`는 반복 step 상한이다. `async_queue`에서는
`--max-steps` 대신 `--max-samples`를 사용한다.
worker 수는 target이 지원하는 범위에서 정한다. 소량 확인과 본 실행의
CSV 경로를 분리한다. 비동기 부하 옵션과 trace는
[비동기 추론 큐 가이드](../../framework/docs/async-inference-queue.md)를 따른다.

## 웹 화면

[설치 절차](01-installation.md)를 마친 뒤, NPU가 연결된 **한 호스트**에서
백엔드를 worker 1개로 실행한다. 벤더별 Python 환경은 서버의
`WEB_RUNNERS_FILE`에 등록한다. 예시는
[`runner_profiles.example.json`](../../backend/config/runner_profiles.example.json)에 있다.

```bash
cd /path/to/ML-HW-Benchmark-Framework/backend
../framework/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

```bash
cd /path/to/ML-HW-Benchmark-Framework/frontend
npm ci
npm run dev
```

`http://localhost:5173`에서 다음 순서로 실행한다.

![웹의 벤치마크 실행 설정 화면: 모델, 타깃, 실행 모드와 자산 경로 입력](images/web-run.png)

위 화면은 ResNet50과 CPU를 선택한 사용 예시다. 화면을 캡처할 때 추론은 실행하지 않았다.

1. **실행**에서 모델·target과 필요한 자산 경로를 선택한다. 경로는 브라우저
   컴퓨터가 아닌 백엔드 호스트 기준이다. 빈 입력은 CLI 기본값을 유지한다.
2. `e2e` 또는 `async_queue`를 고른다. 비동기는 `quality-and-runtime`과
   `runtime-only`, `offline`과 `server_like` 시나리오, worker·queue·표본 수·
   timeout·trace 옵션을 지정할 수 있다. `server_like`에는 target QPS가 필요하다.
3. **설정 검토**에서 Python 환경, 결과 CSV, 최종 CLI 명령, 자산 경로 오류를
   확인한다. 검토에서 오류가 없으면 **실행 시작** 버튼이 활성화된다.
   장치·SDK·품질의 실제 검증은 실행 결과로 확인한다.
4. **실행 시작**을 누르고 **작업**에서 `queued`/`running`/종료 상태와 로그를
   확인한다. 필요하면 **중단**한다. `RUN_ID_RESERVED`와 `RUN_ID`, CSV 행
   존재 여부는 별도로 표시된다.

### 실행 탭 입력 항목

먼저 [조합별 실행 명세](02-configuration.md#조합별-설정과-재현-명령)에서 모델·target과
필요한 artifact·데이터를 확인한다. 아래 항목은 **필요할 때만** 채운다. 빈 칸과
`CLI 기본값`은 해당 CLI 옵션을 전달하지 않는다. 경로는 백엔드 호스트의 경로다.
자산·결과 CSV의 상대 경로는 프레임워크 디렉터리, 이미지·라벨 하위 경로는
데이터셋 경로를 기준으로 해석된다. 다른 호스트의 파일은 입력할 수 없다.
`모델 경로 또는 Hugging Face ID`의 ID만 파일 경로가 아니다.

| 기본 선택 | 입력할 값 |
|---|---|
| **모델** | 실행할 등록 모델 프로필을 고른다. 예: `resnet50`. |
| **타깃** | 실제 연결된 장비의 등록 target을 고른다. 예: CPU 확인은 `cpu`. 등록은 장치 동작 검증을 뜻하지 않는다. |
| **실행 모드** | 순차 실행은 `동기 e2e`, 비동기 완료 처리량은 target이 지원할 때 `비동기 async_queue`를 고른다. |
| **Backend / Device** | 타깃을 `Backend/Device 직접 지정`으로 비웠을 때만 표시된다. 사용할 Runtime 이름과 장치 식별자를 입력한다. 등록 target이 있으면 그것을 선택하는 편이 설정을 재현하기 쉽다. |

**모델·데이터 자산**에서는 선택한 모델·target 명세에 나온 경로만 입력한다.
사전 컴파일 artifact를 사용할 때는 해당 명세의 `컴파일 안 함` 조건도 확인한다.

| 항목 | 입력할 값 |
|---|---|
| **ONNX 경로** | ONNX Runtime에서 사용할 `.onnx` 파일. |
| **HEF 경로** | Hailo Runtime에서 사용할 `.hef` 파일. |
| **컴파일 artifact 경로** | target에 맞는 사전 컴파일 파일. 예: `.dxnn`, `.mxq`, `.rbln`; 일부 Hailo 명세는 `.hef`를 이 칸에 지정한다. |
| **Mobilint BERT weights 경로** | embedding 입력 BERT MXQ에 필요한 호스트 `weight_dict.pth` 파일. |
| **Furiosa FXB 경로** | Furiosa RNGD의 `.fxb` 파일을 명시적으로 지정할 때 사용한다. 해당 경로는 `컴파일 artifact 경로`로도 전달할 수 있다. |
| **모델 경로 또는 Hugging Face ID** | 모델이 요구하는 로컬 파일·디렉터리 또는 지원되는 Hugging Face 모델 ID. |
| **모델 경로 종류** | 바로 위 항목을 입력하면 나타난다. 자동 판별이 맞지 않으면 `호스트 파일/디렉터리` 또는 `Hugging Face ID`를 지정한다. |
| **Tokenizer 경로** | NLP 모델이 사용하는 토크나이저 디렉터리. |
| **데이터셋 경로** | 평가 데이터의 최상위 디렉터리 또는 해당 loader가 받는 CSV·파일 경로. |
| **이미지 하위 폴더 / 라벨 하위 폴더** | 데이터셋의 자동 탐색 대신 이미지 폴더와 라벨 폴더·파일을 지정할 때 입력한다. |

**전처리** 값은 artifact의 입력 계약과 맞춰야 한다. 확실하지 않으면 기본값을
유지하고 모델별 실행 명세를 확인한다.

| 항목 | 입력할 값 |
|---|---|
| **텐서 Layout** | 입력 텐서가 요구하는 `NCHW` 또는 `NHWC`. 빈 값은 target별 CLI 처리를 유지한다. |
| **전처리 dtype** | `auto`, 정규화된 입력의 `normalized`, 0~255 픽셀 입력의 `raw` 중 artifact 계약에 맞는 값. |
| **전처리 profile** | Mobilint raw vision artifact처럼 별도 전처리 profile이 지정된 경우 그 이름. |
| **이미지 resize** | 객체 탐지 입력에 필요한 `direct` 또는 `letterbox`. `auto`는 CLI의 모델·target 처리를 따른다. |

**기본·고급 실행 옵션**은 실행 조건을 명시하거나 특정 Runtime 기능이 필요할
때 사용한다. 숫자는 단위를 확인해 입력한다.

| 항목 | 입력할 값 |
|---|---|
| **Batch size** | 한 번에 처리할 sample 수인 양의 정수. 비동기에서는 동적 batch의 상한이며, 모델·Runtime 지원 범위를 따른다. |
| **Warmup 횟수** | 측정 전 준비 실행 횟수인 0 이상의 정수. |
| **Max steps** | 동기 `e2e`에서만 사용하는 반복 step 상한인 양의 정수. 소량 확인에는 `1`을 쓸 수 있다. |
| **최대 생성 token** | 텍스트 생성 모델의 최대 새 token 수인 양의 정수. |
| **최대 모델 길이** | vLLM의 최대 문맥 길이인 양의 정수. |
| **GPU 메모리 사용률** | vLLM GPU 메모리 비율 `0`~`1`. 예: `0.7`. |
| **vLLM eager 실행** | CUDA graph 캡처를 비활성화해야 할 때 체크한다. |
| **컴파일** | `CLI 기본값`은 target의 기본 흐름을 사용한다. 사전 컴파일 artifact를 그대로 쓸 때는 명세가 허용하는 경우 `컴파일 안 함`을 고른다. |
| **컴파일 옵션 / Runtime 옵션** | 해당 벤더 명세의 `key=value`를 **한 줄에 하나씩** 입력한다. 예: `timeout_ms=10000`. |
| **하드웨어 모니터링** | 지원되는 모니터로 실행 중 하드웨어 지표를 기록할 때 체크한다. |
| **모니터링 간격(초)** | 모니터링을 켰을 때만 표시된다. 양수 초 단위로 입력한다. |
| **Debug 로그** | 실행 로그를 더 볼 때 체크한다. 동기 실행과 비동기 실행에서 출력 내용이 다르다. |
| **웹 작업 timeout(초)** | 웹 작업 전체의 제한 시간. `10`~`86400`초; 빈 값은 서버 기본값 `1800`초다. |

`async_queue`를 선택하면 **비동기 실행 옵션**이 추가된다. 이 모드에서
`Max steps`는 사라지고 `최대 sample 수`를 사용한다. 옵션의 의미와 유효성
판정은 [비동기 추론 큐 가이드](../../framework/docs/async-inference-queue.md)에 있다.

| 항목 | 입력할 값 |
|---|---|
| **부하 시나리오** | 최대한 빨리 공급하려면 `offline`, 목표 요청률에 맞춰 발행하려면 `server_like`. 빈 값은 `offline`. |
| **비동기 패스** | 품질과 성능을 모두 실행하려면 `품질·런타임`; 품질을 생략한 진단은 `런타임만`. 빈 값은 품질·런타임이다. |
| **목표 QPS** | `server_like`를 고르면 필수로 나타난다. 초당 발행할 목표 요청 수인 양수. |
| **Queue 용량** | 대기 요청 수의 상한인 양의 정수. Batch size 이상으로 둔다. |
| **Worker 수** | 동시 실행 worker 수인 양의 정수. target의 허용 동시성 이하로 둔다. |
| **Batch timeout(ms)** | 동적 batch가 채워지기를 기다리는 최대 시간. 0 이상, 밀리초 단위. |
| **Submit timeout(초)** | queue 공간을 기다리는 제한 시간. 양수 초 단위. |
| **Flush timeout(초)** | 남은 요청의 완료·종료를 기다리는 제한 시간. 양수 초 단위. |
| **Request timeout(ms)** | 개별 요청 제한 시간. 입력한다면 양수 밀리초; 비워 두면 CLI 기본값인 제한 없음. |
| **최소 sample 수** | 실행 유효성에 필요한 최소 완료 sample 수인 양의 정수. |
| **최소 측정 시간(초)** | 실행 유효성에 필요한 최소 측정 시간. 0 이상, 초 단위. |
| **최대 sample 수** | 사용할 고유 데이터 sample의 상한인 양의 정수. 소량 확인에는 최소 sample 수보다 작아지지 않게 함께 조정한다. |
| **Schedule seed** | `server_like` 요청 발행 간격을 재현할 때 쓰는 정수 seed. |
| **Latency SLO(ms)** | 지연 목표를 평가할 때 쓰는 양수 밀리초. |
| **Request trace 저장** | 요청별 시각·상태를 담은 trace 파일이 필요할 때 체크한다. |
| **입력 사전 적재** | 지원되는 정적 NumPy 입력을 측정 전에 메모리 매핑 방식으로 준비할 때 체크한다. |

**결과·실행 환경**에서 저장 위치와 서버 Python 환경을 정한다.

| 항목 | 입력할 값 |
|---|---|
| **기존 결과 CSV source** | 이미 등록된 CSV에 행을 추가할 때 고른다. 기본 항목은 CLI 기본 결과 CSV를 사용한다. |
| **새 결과 CSV 경로** | 별도 CSV에 저장하려면 `results/user-guide-smoke.csv`처럼 허용된 `WEB_RESULTS_ROOTS` 안의 `.csv` 경로를 입력한다. 기존 source 선택과 동시에 쓰지 않는다. |
| **Python 실행 환경** | 벤더 SDK가 설치된 등록 runner를 고른다. `서버 자동 선택`은 모델·target에 맞는 runner를 서버가 고르게 한다. |

값을 바꾸면 이전 **설정 검토** 결과가 지워진다. 다시 검토해 오류가 없는지
확인한 뒤 **실행 시작**을 누른다. **초기화**는 입력을 기본 선택으로 되돌린다.
CPU 소량 확인은 위 화면처럼 `resnet50`·`cpu`·`동기 e2e`를 선택하고
**Warmup 횟수** `0`, **Max steps** `1`, **새 결과 CSV 경로**
`results/user-guide-smoke.csv`를 입력하면 된다. 기본 모델·데이터가 없으면
자동 준비가 시작될 수 있다.

실행 시작 후 창을 새로고침해도 작업 목록을 다시 조회할 수 있다. 별도 CLI
실행으로 만든 CSV는 결과 화면에서 경로를 등록한다. 허용 결과 root는
`WEB_RESULTS_ROOTS`, 자산 root는 `WEB_ASSET_ROOTS`로 설정한다.
웹 설정은 [프론트엔드 README](../../frontend/README.md)를 참고한다.

실행이 끝나면 [결과 확인 및 해석](04-results.md)으로 이동한다.

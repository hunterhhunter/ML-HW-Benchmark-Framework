# 06. 함수별 구현 참조

[목차](README.md) · [공용 계약](02-component-contracts.md) · [NPU 어댑터 확장](04-npu-adapter-extension.md)

새 모델·장치용 클래스를 작성할 때 각 함수의 호출자, 반환값, 구현 범위를 확인한다.
아래의 필수는 추상 메서드라 subclass 생성에 필요하다는 뜻이다.
시그니처의 최종 기준은 링크된 코드다.

## DataLoader와 Preprocessor

일반 sample은 `{"input": ndarray | {input_name: ndarray}, "label": ...}`이고,
좌표 복원 등이 필요하면 `preprocess_context`를 추가한다.
[DataLoader](../../framework/src/dataloader/base.py)의 메서드:

| 함수 | 호출자와 구현 범위 |
|---|---|
| `__init__(model_spec, **kwargs)` | **필수; factory 호출.** 데이터 경로·split·전처리 설정을 확정하고 sample·정답 순서를 준비한다. |
| `preprocess(raw_input)` | **필수; Loader 내부 사용.** resize/crop/letterbox, 정규화, dtype, layout, 값 범위를 실제 artifact 계약에 맞춘다. |
| `load_single()` | **필수.** 주 벤치마크 루프에서는 직접 호출하지 않는다. 한 sample을 같은 입력·정답 규칙으로 반환한다. |
| `load_batch(batch_size)` | **필수; 동기 loop·warmup 호출.** 일반 Loader는 sample 딕셔너리 목록을, 정적 batch Loader는 이미 묶인 딕셔너리를 반환한다. 소진 시 빈 값을 반환한다. |
| `load_by_index(index)` | **필수; 비동기 producer 호출.** 0 기반 절대 index의 sample 하나를 반환한다. 범위 밖을 거부하고 호출 순서와 무관하게 같은 sample을 가리킨다. stub으로 남길 수 없다. |
| `get_metadata()` | **필수; Pipeline·비동기 producer·main 호출.** `total_samples`를 실제 개수와 맞는 양의 정수로 둔다. 필요하면 `is_static_batched`, `stop_token_ids`, `runtime_options`를 반환한다. |
| `get_labels()` | **필수.** 주 실행 루프에서는 직접 호출하지 않는다. 반환한 정답의 index·class ID 기준을 sample의 `label`과 일치시킨다. |

`InferencePipeline.collate_batch()`는 일반 sample을 `np.stack`하므로 sample마다
입력 shape가 맞아야 한다. `prepare_runtime_input()`은 단일 배열을 모델 입력
이름으로 감싼다. `prepare_eval_labels()`는 전처리 context가 있으면 label과
함께 evaluator에 넘긴다. 정적 batch Loader는 `is_static_batched=True`를
선언한다. 동기 순차 접근과 비동기 절대 index 접근에서 입력·정답이 같아야 한다.

[BasePreprocessor](../../framework/src/preprocessor/base.py)는
`preprocess(raw_input)`만 추상 메서드다. 상속 helper인
`load_or_preprocess()`와 `load_or_preprocess_npz()`는 각각 `.npy` 배열과
`.npz` 배열 딕셔너리를 캐시한다. 캐시 key에 색상 순서, resize, 정규화,
layout, 모델·artifact profile처럼 출력을 바꾸는 조건을 반영한다.
Loader 안에서 전처리한다면 별도 Preprocessor subclass는 필요하지 않다.

## Compiler와 artifact

[Compiler](../../framework/src/compilers/base.py)는 framework가 컴파일을
호출하는 target에만 추가한다. 사전 컴파일 artifact만 실행하면
`TargetSpec.compiler_name=None`으로 둔다.

| 함수 | 호출자와 구현 범위 |
|---|---|
| `__init__(**compile_options)` | **필수; registry factory 호출.** target 옵션을 받고 저장한다. SDK 의존성은 다른 target의 registry 조회를 막지 않게 한다. |
| `get_artifact_name(model_spec)` | **필수.** 예상 파일명을 돌려준다. 원천·옵션·calibration이 다른 결과가 같은 cache key로 충돌하지 않게 한다. |
| `compile(model_spec, output_dir)` | **필수; main 호출.** `model_spec.model_paths`에서 원천 모델을 선택해 변환하고, 생성 완료된 artifact의 경로를 반환한다. `str`도 허용되지만 `CompileResult(artifact_path, metadata)`를 권장한다. |
| `is_cached(model_spec, output_dir)` | **상속 가능.** 기본 구현은 파일 존재만 검사한다. 실제 동등성은 adapter가 별도 검사한다. |
| `get_compile_config()` | **상속 가능.** 기본 구현은 compile 옵션을 반환한다. 추가 provenance가 필요하면 확장한다. |

`CompiledModel`은 지정 artifact 경로의 존재만 확인한다. 부분 생성·잘못된 cache
hit는 Compiler가 막고, 장치 세대·물리 I/O ABI·SDK 호환성은 Runtime의
`load()`에서도 확인한다.

## Runtime

[Runtime](../../framework/src/runtimes/base.py)의 아래 일곱 메서드는 모두
추상 메서드다. [create_runtime()](../../framework/src/runtimes/__init__.py)은
`device=...`와 병합된 Runtime 옵션을 생성자에 전달한다.

| 함수 | 호출 경계와 구현 범위 |
|---|---|
| `__init__(**runtime_options)` | **필수; factory 생성.** `device`와 SDK 옵션을 받는다. 자원 획득·SDK import를 `load()`로 미룰 수 있다. 유효하지 않은 옵션을 조용히 무시하지 않는다. |
| `is_compatible(compiled_model)` | **필수; main은 직접 호출하지 않음.** artifact·장치 적합 여부를 bool로 답한다. `load()`에서 이를 호출하거나 같은 검증을 수행한다. registry graph 검증은 이를 대신하지 않는다. |
| `load(compiled_model)` | **필수; main에서 호출.** artifact·sidecar, SDK/driver, 실제 입출력 ABI를 확인하고 모델·buffer를 준비한다. `compiled_model`을 보관해 Pipeline이 `spec.input_shapes`에서 입력 이름을 읽게 한다. 실패 시 획득한 자원을 정리한다. |
| `warmup(inputs, num_runs)` | **필수; Engine이 측정 전에 호출.** 본 실행과 같은 입력·장치 설정으로 지정 횟수 준비한다. 결과 지표에 합산하지 않는다. |
| `run(inputs)` | **필수; blocking executor 호출.** 입력 이름·shape·dtype·layout·batch를 검사하고 SDK 실행 완료 후 `dict[str, np.ndarray]`를 반환한다. 출력은 다음 호출 전에 덮어쓰이지 않는 host-visible 값이어야 한다. 생성 전용 Runtime도 메서드 자체는 구현하고, 사용하지 않는다면 명시적으로 거부한다. |
| `get_device_spec()` | **필수; main이 결과 metadata에 사용.** 실제 선택 장치·backend·벤더 등 확인 가능한 정보를 딕셔너리로 반환한다. 요청 문자열만으로 실제 장치 선택을 증명하지 않는다. |
| `unload()` | **필수; 종료·오류 정리.** 모델·buffer·SDK context를 해제한다. 비동기 물리 작업이 남았으면 성공한 해제처럼 처리하지 않는다. |

생성형 모델은 `supports_generate()=True`와
`generate(inputs, max_new_tokens, stop_token_ids) -> GenerationResult`를 함께
구현한다. `generated_ids`·`num_tokens`·`total_ms`는 실제 실행과 맞추고,
관측하지 못한 TTFT·TPOT는 `None`으로 둔다.
`supports_dynamic_batching()`, `max_dynamic_batch_size()`,
`supports_batch_generation()`, `max_concurrent_workers()`,
`supports_streaming_generate()`는 확인한 안전 범위에서만 override한다.
선언한 batch·worker 상한을 넘으면 비동기 엔진이 거부할 수 있다.

## Decoder와 Evaluator

전체 task를 아우르는 Decoder ABC는 없다. 객체 탐지용
[DetectionDecoder](../../framework/src/decoders/object_detection.py)는
`decode(outputs)`가 필수이고 `result_metadata()`는 선택적이다. Pose·
segmentation Decoder도 completion 경로가 호출할 `decode(outputs)`를
제공해야 한다.

| 함수 | 구현 범위 |
|---|---|
| `decode(outputs)` | Runtime의 실제 출력 이름·shape·dtype·배치와 그래프 내부 후처리 여부를 검사한다. 해당 task의 evaluator 입력으로 변환하고 지원하지 않는 ABI는 거부한다. 탐지는 빈 결과도 안정된 `detections` 배열 형식으로 돌려준다. |
| `result_metadata()` | threshold, 좌표계, decoder·ABI 식별자처럼 결과 해석에 필요한 안정된 설정을 반환한다. main이 저장 metadata에 병합한다. |

[Evaluator](../../framework/src/evaluators/base.py)는 다음 모두가 추상
메서드다. 현재 벤치마크의 주 경로는 `add_batch() → compute()`다.

| 함수 | 구현 범위 |
|---|---|
| `__init__(**eval_options)` | 정답 파일·지표 옵션을 검증하고 batch 집계 상태를 초기화한다. |
| `add_batch(outputs, labels, timing_ms)` | 출력과 같은 sample의 정답을 대조하고 필요한 통계만 누적한다. `timing_ms`는 정적 추론의 float 또는 생성 경로의 timing 딕셔너리일 수 있다. |
| `compute()` | 지금까지 누적한 표본의 지표 이름→값을 반환한다. 분모와 빈 입력 처리 방식을 정한다. |
| `evaluate(result)` | **필수이지만 주 loop에서는 미호출.** 레거시 `InferenceResult`를 받아 같은 지표 의미로 계산한다. |
| `is_applicable(device_spec, model_spec)` | 적용 가능성을 bool로 답한다. 현재 factory의 자동 선택을 대신하지 않는다. |
| `get_metric_names()` | 반환 가능한 지표 이름 목록을 제공한다. 이름·단위를 저장 스키마 및 측정 계약과 맞춘다. |

새 클래스를 만든 것만으로 선택되지 않는다.
[create_decoder()](../../framework/src/decoders/__init__.py)와
[create_evaluator()](../../framework/src/evaluators/__init__.py)의
task·backend·모델·annotation·출력 ABI 분기를 연결하고, main에서 필요한
옵션이 factory까지 전달되는지 확인한다. `runtime-only` pass는 둘 다
생성하지 않으며 품질을 계산하지 않는다.

## Collector와 native async

[Collector](../../framework/src/monitors/base.py)의 `start()`,
`collect()`, `stop()`은 필수다.

| 함수 | 구현 범위 |
|---|---|
| `is_available()` | **선택적 override.** factory가 호출하므로 실제 SDK·sensor·선택 장치의 수집 가능 여부를 확인한다. 기본 구현은 `True`다. |
| `start()` | 측정용 handle을 열고 선택적으로 시작 경계 sample을 반환한다. warmup은 모니터링 구간에 넣지 않는다. |
| `collect()` | 현재 관측값의 `dict[str, float | None]`를 반환한다. 읽지 못한 값은 `None` 또는 생략으로 나타낸다. |
| `stop()` | 수집 자원을 닫고 선택적으로 종료 경계 sample을 반환한다. 시작 후 오류가 나도 정리 가능한 상태를 유지한다. |
| `get_summary_metrics()` | **선택적 override.** 평균·최대만으로 표현할 수 없는 energy, coverage 등의 집계를 반환한다. |
| `get_static_info()` | **ABC 밖의 선택적 메서드.** 구현하면 `HWMonitor.summary()`가 장치 식별 정보를 반영한다. |

`create_hw_monitor()`는 `is_available()=False`이거나 생성 예외가 난
collector를 건너뛴다. 따라서 `--monitor`만으로 특정 NPU 지표 수집을
보장하지 않는다. 전력은 값과 함께 chip·card·system 중 실제 범위와
sample coverage를 확인한다.

`native_async` target은 Runtime ABC 밖의
`create_native_backend()`와 `native_async_max_batch_size()`를
제공해야 한다. 전자는 `submit_async(inputs, callback)`를 가진 backend를
돌려준다. 후자가 양의 정수 batch 상한을 돌려주면 native executor를 쓰고,
`None`이면 현재 모델에서는 blocking executor 경로로 진행한다.
[NativeAsyncRuntimeExecutor](../../framework/src/core/runtime_executor.py)가
backend를 호출하고 callback의 `NativeAsyncOutcome(outputs, timing_ms, ...)`을
`RuntimeExecution`으로 바꾼다.

| 지점 | 추가 구현·검증 |
|---|---|
| `submit_async(inputs, callback)` | SDK job을 제출하고 빠르게 반환한다. 입력·출력 buffer와 SDK job의 소유권을 물리 완료까지 유지한다. 반환한 vendor job ID는 진단용이다. |
| `callback(NativeAsyncOutcome)` | 완료된 출력 또는 오류를 한 작업에 대응시켜 전달한다. SDK 소유 buffer는 재사용 전에 framework가 읽을 수 있는 값으로 확보한다. 중복·늦은 callback도 안전하게 다룬다. |
| `native_async_max_inflight()` · `native_async_completion_timeout_sec()` | **선택적 Runtime hook.** 실제 SDK 한계와 논리 deadline을 반환한다. 물리 취소가 불가능하면 timeout 뒤에도 job·buffer를 보유한다. |
| Runtime `unload()` | executor의 `shutdown()`은 framework dispatch의 완료·ACK를 기다린다. 벤더 backend의 drain·shutdown과 SDK 모델 해제는 Runtime 쪽에서 별도로 완료한다. |

비동기 worker 허용량, callback 완료, ACK, 물리 완료는 서로 다른 사건이다.
`native_async` 선언 전에 성공·제출 예외·callback 오류·중복·timeout 후 늦은
완료·종료 시 잔여 작업을 각각 검증한다. 자세한 lifecycle은
[NPU 어댑터 확장](04-npu-adapter-extension.md#5-native-async-추가)을 따른다.

## 등록 후 실제로 연결할 곳

| 새로 만든 것 | 연결·확인 위치 |
|---|---|
| 모델·task | `core/model_profiles.py`의 profile, 필요하면 `Task`와 dataset resolver |
| Loader·Preprocessor | `dataloader/__init__.py`의 factory, `main.py`의 `loader_kwargs` |
| Runtime·Compiler·Collector | 각 패키지의 registry entry, `core/targets.py`의 `TargetSpec` |
| Decoder·Evaluator | 각 factory의 선택·거부 분기, `main.py`의 전달 옵션 |
| 실행 옵션 | CLI parser·조립부, 웹에서 지원할 경우 backend 요청 schema·CLI 명령 변환·frontend |
| 새 결과 필드 | `result_store.py`와 결과 API, 이름·단위·측정 구간 문서 |

구현 후에는 [변경 검증 절차](05-verification.md)에서 구조·계약·소량
실행·실장비 검증을 단계별로 확인한다. registry 조회나 mock 실행 성공은
실제 artifact의 품질과 장치 안정성을 보장하지 않는다.

# 02. 공용 구성 요소 계약

[목차](README.md) · [이전: 전체 구조](01-architecture.md) · [다음: 모델·태스크 확장](03-model-and-task-extension.md)

이 장은 `framework/src`의 **현재 호출 계약**을 설명한다. 추상 클래스의 메서드를
구현하는 것과 `main.py`에서 그 구현을 선택할 수 있게 연결하는 것은 별개 작업이다.
아래의 파일 링크가 시그니처의 기준이며, 모델·벤더별 입력값은 실행 명세를 따른다.
각 메서드의 필수 여부·호출자·반환 및 검증 조건은
[함수별 구현 참조](06-implementation-reference.md)에 정리했다.

## 책임을 나누는 기준

| 구성 요소 | 책임 | 다음 계층에 전달하는 것 |
|---|---|---|
| Model profile / `Model_Spec` | 모델 식별자, task, 기본 자산, 기대 I/O | 모델 스펙과 준비 경로 |
| `TargetSpec` | Runtime·Compiler·Monitor, 장치, artifact 형식, capability 조합 | 실행 target과 기본 옵션 |
| DataLoader / Preprocessor | 데이터·정답 읽기, 모델 입력 생성, sample 순서·메타데이터 | `input`, `label`, 선택적 `preprocess_context` |
| Compiler | 원천 모델에서 장치 artifact 생성 | artifact 경로와 compile metadata |
| Runtime | artifact 로드, 장치 호출, host-visible 출력 회수, 자원 해제 | 출력 이름 → tensor의 매핑 |
| Decoder | Runtime 출력의 의미 해석, evaluator 입력 형태로 변환 | task별 정규화 출력 |
| Evaluator | 정답 대조, 품질·Runtime timing 통계 | 최종 지표 딕셔너리 |
| Collector | 측정 구간의 관측 가능 telemetry 수집 | 수집값·측정 범위·요약 |
| Engine / ResultStore | warmup·실행·완료·저장 lifecycle | CSV, 비동기 details·trace |

입력 layout 변환이 필요한 곳은 Loader·전처리 또는 Runtime의 **전송 경계**다.
Runtime은 SDK가 실제 반환한 출력을 숨기지 말고, 이름·배치·자료형을 확인 가능한
형태로 노출한다. 모델 의미를 해석하는 sigmoid/DFL/anchor/NMS/좌표 변환은
해당 artifact의 그래프 내부에 이미 있는지 확인하고, 필요한 host 작업을 decoder
또는 evaluator의 기존 책임에 맞춰 연결한다. 같은 모델명도 artifact마다 이 경계가
다를 수 있다.

## 모델과 target: 등록 정보는 실행 보증이 아니다

[`Model_Spec`](../../framework/src/core/model_spec.py)은 `name`, `Task`,
`input_shapes`, `input_dtype`, `output_shapes`, `model_paths`를 가진 불변
dataclass다. [`SUPPORTED_PROFILES`](../../framework/src/core/model_profiles.py)은
기본 모델·데이터 경로와 준비 스크립트도 보관한다. `create_model_spec()`은
`sniff_onnx=True`일 때만 ONNX I/O 이름을 읽어 `__auto__`를 바인딩한다.
HEF·DXNN·MXQ 등의 경로는 profile의 기본 이름을 쓸 수 있으므로 그 이름이나
shape를 실제 artifact ABI로 가정하면 안 된다.

[`TargetSpec`](../../framework/src/core/targets.py)은 `target_id`, `runtime_name`,
`device`, 선택적 `compiler_name`, `monitor_names`, `artifact_format`,
`runtime_options`·`compiler_options`·`monitor_options`, `capabilities`를
묶는다. `resolve_target()`은 명시한 `--target`을 우선한다.
`RuntimeEntry`·`CompilerEntry`·`CollectorEntry`는 각 패키지의 `__init__.py`에
등록하며, registry 조회는 벤더 SDK 모듈을 즉시 import하지 않는다.
`validate_registry_graph()`는 이름·형식 연결을 검사하지만 artifact 호환성,
장치 존재, 품질, callback 안전성을 대신 검증하지 않는다.

## DataLoader와 Preprocessor

[`DataLoader`](../../framework/src/dataloader/base.py)의 필수 메서드는
`__init__(model_spec, **kwargs)`, `load_single()`, `load_batch(batch_size)`,
`load_by_index(index)`, `get_labels()`, `get_metadata()`,
`preprocess(raw_input)`다. 다음을 구현 책임으로 본다.

- `load_batch()`는 유효한 batch를 공급하고 소진 시 빈 값을 반환한다.
  `InferenceEngine`은 빈 값으로 반복을 끝낸다. batch 안의 sample 순서와
  `load_by_index()`의 절대 index 순서는 일치해야 한다.
- 일반 sample은 `{"input": ndarray 또는 입력명→ndarray, "label": ...}`
  구조다. 좌표 복원 등이 필요하면 sample별 `preprocess_context`를 제공한다.
  `InferencePipeline`이 이를 label에 붙여 evaluator에 넘긴다.
- `get_metadata()`는 최소한 총 표본 수 등 호출자가 필요한 값을 제공한다.
  `is_static_batched`, `stop_token_ids`, `runtime_options`는 해당 경로에서만
  의미가 있는 선택적 값이다. 정적 batch loader는 `is_static_batched=True`를
  선언하고 이미 batch 형태로 반환한다.
- 전처리 캐시를 쓴다면 입력 파일뿐 아니라 resize·정규화·layout·프로필처럼
  **결과를 바꾸는 조건**을 cache key에 반영해야 한다. 캐시 존재가 올바른
  artifact 입력이라는 증거는 아니다.

[`BasePreprocessor`](../../framework/src/preprocessor/base.py)의 필수 메서드는
`preprocess(raw_input)`다. `load_or_preprocess()`는 단일 배열 `.npy`,
`load_or_preprocess_npz()`는 여러 배열 `.npz`의 선택적 캐시 helper다.
전처리가 Loader 내부에서 구현된 경우도 있으므로 모든 모델에 별도
`BasePreprocessor` subclass를 추가할 필요는 없다.

선택 함수 [`create_dataloader()`](../../framework/src/dataloader/__init__.py)는
task만 보지 않는다. `backend`, `target_id`, 모델명에 따라 Hailo·DEEPX·Mobilint·
Regulus·TTM-R2 전용 loader로 분기한다. 새 경로에 필요한 옵션은
`main.py`에서 실제로 factory에 전달되는지도 확인한다.

## Compiler와 CompiledModel

[`Compiler`](../../framework/src/compilers/base.py)의 현재 필수 계약은
`__init__(**compile_options)`,
`compile(model_spec: Model_Spec, output_dir: str) -> str | CompileResult`,
`get_artifact_name(model_spec) -> str`이다. 새 adapter는
`CompileResult(artifact_path, metadata)`를 반환하면 경로와 compiler 정보가
함께 전달된다. 기존 문자열 반환도 `normalize_compile_result()`가 받는다.
`is_cached()`는 기본 파일 존재 검사 helper일 뿐, 컴파일 조건의 동등성을
자동 증명하지 않는다.

`main.py`가 target의 `compiler_name`으로 compiler를 골라 호출하고,
`CompiledModel(spec, backend_name, artifact_path)`을 만든다.
[`CompiledModel`](../../framework/src/core/compiled_model.py)은 주어진 artifact
경로의 존재를 검사한다. Runtime의 `load()`는 **장치 호환성, 입력·출력 ABI,
필요한 sidecar와 SDK 상태**를 추가로 검사해야 한다. `--no-compile`은
컴파일을 생략한다는 뜻이며, native 형식 target에는 해당 형식의 사전 컴파일
artifact가 필요하다.

## Runtime과 RuntimeExecutor

[`Runtime`](../../framework/src/runtimes/base.py)의 필수 메서드:

| 메서드 | 구현해야 할 범위 |
|---|---|
| `__init__(**runtime_options)` | 선택 장치와 SDK 옵션 보관. 생성 시 무거운 SDK import·장치 점유는 필요 시 지연 |
| `is_compatible(compiled_model)` | artifact 형식·장치·필수 조건의 사전 호환성 검사 |
| `load(compiled_model)` | artifact 검증 후 장치 모델·buffer 준비, `compiled_model` 보관 |
| `run(inputs)` | 입력명→배열을 SDK에 전달하고 **완료된** host-visible 출력 딕셔너리 반환 |
| `warmup(inputs, num_runs)` | 같은 입력 계약으로 준비 호출. 측정 결과에 합산하지 않음 |
| `get_device_spec()` | 실제 선택 장치와 런타임 진단 정보 반환 |
| `unload()` | 완료된 작업의 자원 해제. 미완료 SDK 작업을 무시하고 해제하면 안 됨 |

정적 모델은 `run()`, 생성 모델은 `supports_generate()=True`와
`generate(...)->GenerationResult`를 사용한다. `GenerationResult`는 생성 ID와
전체 생성 시간, 이용 가능한 경우 TTFT·TPOT 및 timing source를 담는다.
측정하지 않은 token 시각을 추정값으로 채우지 않는다. `max_concurrent_workers()`,
dynamic batching, streaming, native async 등의 hook은 **선택적 capability**다.
메서드가 존재한다는 사실만으로 해당 target이 실제 병렬 실행을 검증했다는
뜻은 아니다.

[`RuntimeExecutor`](../../framework/src/core/runtime_executor.py)는 framework
내부 호출·완료 경계다. 기본 `BlockingRuntimeExecutor`가 `run()`/`generate()`를
호출한다. 벤더 adapter가 일반적으로 이 클래스를 직접 상속할 필요는 없다.
`native_async` target은 Runtime의 hook이 만든 backend를
`NativeAsyncRuntimeExecutor`에 연결한다. 그 lifecycle 계약은
[NPU 어댑터 확장](04-npu-adapter-extension.md#5-native-async-추가)에서 다룬다.

## Decoder와 Evaluator

`create_decoder()`는 필요 없으면 `None`을 반환한다.
[`DetectionDecoder`](../../framework/src/decoders/object_detection.py)은
객체 탐지 decoder의 ABC로 `decode(outputs)`를 요구한다. Pose·segmentation
decoder에는 전체 task 공통 ABC가 없지만 completion 경로는 동일하게
`decoder.decode(outputs)`를 호출한다. 새 decoder는 Runtime 출력의 실제
이름·shape·단위·내장 후처리를 확인하고 evaluator가 받는 형태로 변환한다.
예를 들어 탐지 evaluator의 `detections`는
`[batch 내부 image index, class id, confidence, x1, y1, x2, y2]` 행이다.
Pose는 `detections`와 `keypoints`를 함께 사용한다. 각 모델의 정확한 형식은
[모델별 출력 계약](../../framework/docs/model-execution-specs/yolov8s-pose.md)을
참조한다.

[`Evaluator`](../../framework/src/evaluators/base.py)는
`__init__(**eval_options)`, `add_batch(outputs, labels, timing_ms)`,
`compute()`, `evaluate(result)`, `is_applicable()`, `get_metric_names()`를
추상 메서드로 정의한다. **실제 벤치마크의 기본 경로는 batch마다
`add_batch()`, 완료 후 `compute()`**다. `evaluate()`는 레거시·배치 호환
인터페이스다. evaluator는 정답과 출력의 의미를 검증하고 지표를 계산한다.
Runtime의 raw head 자체를 정답과 직접 비교해서는 안 된다.
`create_evaluator()`는 task 외에 모델명·backend·annotation 존재 여부에
따라 TTM-R2 전용 또는 `LatencyOnlyEvaluator`를 선택할 수 있다.
`LatencyOnly`는 품질 평가를 수행했다는 뜻이 아니다.

## Monitor와 ResultStore

[`Collector`](../../framework/src/monitors/base.py)은 `start()`, `collect()`,
`stop()`을 구현하고 `is_available()`·`get_summary_metrics()`를 필요에
따라 확장한다. `HWMonitor`가 background polling과 lifecycle을 관리한다.
수집 불가 값은 `None` 또는 미수집 상태로 남기고 0으로 대체하지 않는다.
전력 값에는 chip·card·system 중 실제 측정 지점을 붙여야 한다.
샘플링 시작은 warmup 후이며 세부 시간 경계는 측정 계약을 따른다.

[`result_store.py`](../../framework/src/core/result_store.py)는 CSV 스키마,
`RUN_ID`, 비동기 reservation·details·trace의 연결을 관리한다.
`RUN_ID` 저장 성공, `async_run_status` 실행 유효성,
`quality_status` 평가 수행 상태, `comparison_eligible` 비교 후보 상태는
각기 다른 판정이다. 추가 metric·metadata는 이름·단위·측정 경계를 정한 뒤
저장하고 [측정 계약](../../framework/docs/measurement-contract.md)도 갱신한다.

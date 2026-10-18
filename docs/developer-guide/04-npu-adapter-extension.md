# 04. NPU 어댑터 확장

[목차](README.md) · [이전: 모델·태스크 확장](03-model-and-task-extension.md) · [다음: 변경 검증](05-verification.md)

새 장치를 추가할 때 **Runtime → 선택적 Compiler/Collector → registry entry →
`TargetSpec` → 모델·artifact별 입출력 연결 → 검증** 순서로 진행한다.
같은 SDK의 새 장치라면 Runtime을 복제하기 전에 기존 adapter의 장치 선택과
artifact 호환 검사를 확장할 수 있는지 확인한다.

## 1. Runtime adapter

[`Runtime`](../../framework/src/runtimes/base.py)을 구현하고
`framework/src/runtimes/<vendor>_rt.py`에 둔다. 생성자는
`create_runtime(name, device=..., **runtime_options)`가 호출하므로 `device`와
target 기본 옵션을 수용해야 한다.

| 단계 | 구현·검증할 내용 |
|---|---|
| 생성·`is_compatible()` | 장치 선택과 옵션, artifact 형식·대상 장치 조건 |
| `load(CompiledModel)` | 실제 파일·필요한 sidecar와 SDK/firmware 확인, 모델·입출력 metadata 읽기, buffer 준비 |
| `run(inputs)` | 입력명·dtype·layout·batch 검증, SDK submit/완료 대기, 반환 전 host-visible 출력 확보 |
| `warmup()` | 같은 API·입력 계약으로 지정 횟수만큼 호출 |
| `get_device_spec()` | 실제 선택한 장치·runtime 버전·binding 진단 |
| `unload()` | 완료된 작업의 모델·buffer·context 해제, 미완료 작업이 있으면 안전성 확인 |

`CompiledModel.artifact_path`의 파일 존재 검사는 공통 계층이 하지만,
HEF·DXNN·MXQ 등의 **장치 세대, 실제 I/O ABI, graph 내부 NMS/CPU offload,
SDK 호환성**은 adapter가 확인해야 한다. `run()`은 완료 전에 반환하거나
SDK 소유 output buffer를 호출자에게 그대로 넘겨 다음 추론에서 덮어쓰게
해서는 안 된다. 정적 추론의 저장 latency는 framework의 `run()` 호출
전후이므로, adapter 안의 host 변환·전송·완료 대기·출력 회수가 포함될 수 있다.

실제 구현 예시는 [`MockNpuRuntime`](../../framework/src/runtimes/mock_npu_rt.py),
장치별 adapter는 `framework/src/runtimes/`에 있다. SDK import는 해당 target을
선택하지 않은 실행을 막지 않도록 필요한 시점으로 미룬다.

구현 후 [`runtimes/__init__.py`](../../framework/src/runtimes/__init__.py)에
`RuntimeEntry(name, module, class_name, aliases, description)`를 등록한다.
`unsupported_reason`가 있으면 등록 목록에는 나타나지만 Runtime 로드는
거부된다. alias 충돌은 registry 등록 시 오류다.

## 2. Compiler와 native artifact

컴파일을 framework 호출에서 지원한다면
[`Compiler`](../../framework/src/compilers/base.py)을 구현하고
`compilers/__init__.py`에 `CompilerEntry`로 등록한다.
현재 호출 시그니처는 `compile(model_spec, output_dir)`이며 결과는
`str` 또는 `CompileResult(artifact_path, metadata)`다.
`get_artifact_name(model_spec)`은 파일명·캐시 검사에 쓰인다.

Compiler는 원천 모델·설정·calibration과 target에 맞는 artifact를 만들고,
반환 전 생성 성공을 확인한다. `metadata`에 compiler·artifact 형식, 버전,
사용한 옵션과 검증 가능한 provenance를 남긴다. 기본 `is_cached()`는 예상
파일의 존재만 본다. 옵션·원천 모델·calibration이 달라진 경우 같은 이름의
파일을 재사용하지 않도록 adapter가 cache 정책을 정해야 한다.

미리 컴파일한 artifact만 받는 target은 `compiler_name=None`으로 두고
`--artifact` 또는 해당 전용 옵션으로 전달한다. Compiler가 등록됐더라도
`--no-compile`을 선택하면 그 target의 native 형식에 맞는 artifact를
제공해야 한다. `Model_Spec`의 ONNX 출력 이름이 native artifact ABI와
같다고 가정하지 않는다.

## 3. Collector와 전력 범위

telemetry가 있으면 [`Collector`](../../framework/src/monitors/base.py)을
구현하고 `monitors/__init__.py`에 `CollectorEntry`를 등록한다.
`start()`는 수집 자원을 초기화하고 선택적 시작 경계 sample을,
`collect()`는 현재 측정값을, `stop()`은 정리와 선택적 종료 경계 sample을
반환한다. `get_summary_metrics()`는 collector 전용 요약이 필요한 경우에만
구현한다. `HWMonitor`가 주기적 호출과 종료를 맡는다.

장치 API가 제공하지 않는 지표는 `None` 또는 미수집 상태로 남긴다. 특히
`power_w`는 chip, board/card, system 중 실제 계측 지점을 기록한다.
공통 모니터 이름 `system`을 함께 쓰더라도 그것만으로 NPU 전력이 측정되는
것은 아니다. [측정 계약](../../framework/docs/measurement-contract.md)의
전력 경계를 따른다.

## 4. TargetSpec과 조립 옵션

[`core/targets.py`](../../framework/src/core/targets.py)에 `TargetSpec`을
등록한다. 최소한 `target_id`, `runtime_name`, `device`,
`artifact_format`, `monitor_names`, 실제 지원 capability를 정한다.
Compiler가 있으면 `compiler_name`, 필요하면 `runtime_options`·
`compiler_options`·`monitor_options`를 설정한다. `target_id`는 CLI·API에
노출되는 이름이다.

```python
# 구조 예시: 실제 이름과 옵션은 구현한 adapter에 맞춰 결정한다.
register_target(TargetSpec(
    target_id="vendor_npu",
    label="Vendor NPU",
    runtime_name="vendor_npu",
    device="npu0",
    compiler_name=None,                # precompiled artifact만 실행할 경우
    monitor_names=("system",),
    artifact_format="vendorbin",
    capabilities=("sync", "npu", "local"),
))
```

이 코드는 registry 연결만 보여준다. 벤더 Runtime과 실제 artifact, 입력 Loader,
출력 Decoder가 준비되지 않았다면 벤치마크 실행은 성립하지 않는다.
`validate_registry_graph()`는 SDK를 import하지 않고 registry 참조·형식을
점검한다. `strict=True`에서는 등록됐지만 `unsupported_reason`이 있는
entry 같은 warning도 `ok=False`로 본다.

`main.py`는 target 기본 Runtime 옵션, Loader metadata의 `runtime_options`,
CLI `--runtime-option`을 병합하고 일부 장치 옵션은 고정·검증한다.
새 SDK가 추가 정보를 요구하면 `main.py`에서 옵션을 받아 적절한 Loader·Runtime·
Decoder에 전달해야 한다. 새 `TargetSpec` 하나로 모델별 전처리·후처리가
자동으로 맞춰지는 구조는 아니다.

## 5. native async 추가

`native_async` capability는 **callback 기반 SDK job을 안전하게 인수할 때만**
선언한다. `main.py`는 이 capability를 보면 Runtime의
`create_native_backend()`와 `native_async_max_batch_size()`를 호출하고,
후자가 양의 정수를 반환할 때 `NativeAsyncRuntimeExecutor`를 만든다.
`native_async_max_inflight()`와 `native_async_completion_timeout_sec()`는
선택적 한계다. 실제 동시 호출 가능 수는 Runtime의
`max_concurrent_workers()`와 장치 SDK의 안전 조건도 확인해야 한다.

native backend의 `submit_async(inputs, callback)`은 작업을 제출하고
빠르게 반환한다. callback은 `NativeAsyncOutcome`의 출력 또는 오류와
필요한 timing을 전달한다. 구현 시 다음 순서를 검사한다.

1. 제출 전에 입력·출력 buffer와 job state를 보유하고, SDK가 요구하는
   queue slot·bindings의 소유권을 명시한다.
2. callback 또는 물리적 완료 증거가 오기 전에는 buffer를 재사용하거나
   Runtime을 unload하지 않는다. 결과를 framework에 전달할 때 출력의
   소유권을 안전하게 확보한다.
3. 중복 callback, submit 예외, callback 오류, timeout, 늦은 callback을
   실패·진단 상태로 처리한다. 논리적 timeout은 SDK 작업 취소를 뜻하지 않는다.
4. framework의 terminal 처리와 `acknowledge()` 후에도 물리 작업이 남아
   있으면 slot을 반환하지 않는다. `shutdown()`은 남은 작업이 0임을 확인해야
   안전하게 성공한다.
5. SDK queue 크기, batch 상한, worker 수, timeout을 각각 검사한다.
   worker 수 증가만으로 NPU 병렬 실행을 주장하지 않는다.

상세 executor 동작은
[`runtime_executor.py`](../../framework/src/core/runtime_executor.py)와
[비동기 추론 가이드](../async-inference-queue.md)를 따른다. callback adapter는
공통 `_AsyncRunController` 같은 private 클래스에 직접 결합하지 않는다.

## 6. 모델과 결과 연결

새 target의 Runtime 출력은 [모델·태스크 확장](03-model-and-task-extension.md)의
Loader·Decoder·Evaluator 연결까지 검증한다. 같은 모델을 다른 벤더
artifact로 실행할 때는 원천 모델·전처리·출력 의미를 비교하고,
artifact 내부에서 처리된 단계와 host에서 처리한 단계를 구분한다.
결과 CSV에 `target_id`, runtime·compiler·artifact 정보가 저장되는지 보고,
실제 장치와 artifact 신원은 별도 기록과 연결한다.

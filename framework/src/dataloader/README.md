# DataLoader 패키지

데이터와 정답을 같은 sample 순서로 공급하고, 모델·artifact에 맞는 입력을
준비한다. 전체 실행 흐름과 확장 순서는 [개발자 설명서](../../../docs/developer-guide/README.md)의
[공용 계약](../../../docs/developer-guide/02-component-contracts.md#dataloader와-preprocessor)과
[모델·태스크 확장](../../../docs/developer-guide/03-model-and-task-extension.md#2-입력-계약과-loader-연결)을
따른다.

## 현재 호출 계약

[`DataLoader`](base.py)는 `__init__(model_spec, **kwargs)`,
`load_single()`, `load_batch(batch_size)`, `load_by_index(index)`,
`get_labels()`, `get_metadata()`, `preprocess(raw_input)`를 추상 메서드로
정의한다. 일반 sample은 `input`과 `label`을 가지며, 좌표 복원에는
`preprocess_context`를 추가할 수 있다. `load_batch()`의 소진 상태는 빈
값으로 나타낸다. `InferencePipeline`은 일반 sample 목록을 stack하지만
metadata의 `is_static_batched=True`인 loader는 이미 batch로 반환한다.

`get_metadata()`의 `total_samples`, 선택적 `runtime_options`와
`stop_token_ids`는 실행 조립에 사용된다. loader가 전처리 캐시를 사용하면
입력·전처리 설정이 달라진 경우 다른 캐시를 선택해야 한다.

## 선택과 확장

[`create_dataloader()`](__init__.py)는 `Model_Spec.task`뿐 아니라
`backend`, `target_id`, 모델명에 따라 전용 loader를 선택한다.
예를 들어 Hailo·DEEPX·Mobilint vision, Regulus Pose, TTM-R2에는 별도
경로가 있다. 실제 선택 kwargs는 [`main.py`](../main.py)의
`loader_kwargs` 조립부에서 전달한다.

새 모델이 기존 입력·정답 계약을 그대로 쓰면 Loader를 재사용한다.
전처리·입력 ABI 또는 정답 형식이 달라지면 구현과 factory 선택 조건,
`main.py`의 옵션 전달을 함께 수정한다. 새 task에는
[`Task`](../core/model_spec.py), 데이터 경로 resolver, Decoder·Evaluator
연결도 필요할 수 있다.

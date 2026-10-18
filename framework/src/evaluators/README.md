# Evaluator 패키지

Evaluator는 Decoder가 넘긴 출력을 정답과 대조하고 지표를 계산한다.
모델·벤더별 raw 출력은 [Decoder](../decoders/__init__.py)에서 해석한다.
전체 계약과 확장 절차는 [개발자 설명서](../../../docs/developer-guide/README.md)의
[공용 계약](../../../docs/developer-guide/02-component-contracts.md#decoder와-evaluator)과
[모델·태스크 확장](../../../docs/developer-guide/03-model-and-task-extension.md#4-evaluator와-새-task)을
따른다.

## 현재 호출 계약

[`Evaluator`](base.py)의 추상 메서드는 `__init__()`,
`add_batch(outputs, labels, timing_ms)`, `compute()`,
`evaluate(result)`, `is_applicable()`, `get_metric_names()`다.
실제 벤치마크는 각 batch마다 `add_batch()`를 부르고 마지막에
`compute()`를 한 번 호출한다. `evaluate()`는 레거시·배치 호환 경로다.
`add_batch()`에서 큰 tensor 전체를 오래 보관하기보다 필요한 집계값을
축적한다.

[`create_evaluator()`](__init__.py)는 task뿐 아니라 모델명, backend,
annotation 유무를 보고 구현을 고를 수 있다. TTM-R2는 전용 evaluator를
사용한다. 일부 segmentation·pose 조합에서 annotation이 없으면
`LatencyOnlyEvaluator`를 선택하므로 품질 평가가 수행된 것으로 표시하면
안 된다.

## 구현과 검증

새 evaluator는 입력 tensor의 의미·좌표계와 label 형식을 명시하고,
지표 이름·단위·표본 수를 정의한다. 여러 batch를 거친 `compute()`,
빈 입력, 잘못된 출력 ABI, 호환 `evaluate()` 경로를 검증한다.
`quality_status=passed`는 집계 성공 상태이며 정확도 목표 달성이 아니다.
결과의 시간 경계와 비교 조건은 [측정 계약](../../docs/measurement-contract.md)을
따른다.

# Mobilint PatchTST clamp lowering 보완 설계

## 배경

qbcompiler 1.2 compiler host에서 PatchTST ETTh1을 실제 실행한 결과, stock attempt는
`Tensor.clamp_min`이 boolean denominator에 적용된 FX node에서 `MBLT_COMPILE`에
실패했다. 기존 `compat-static-patchifier` attempt는 `Tensor.unfold`를 고정
slice/stack으로 바꾸고 mask를 values dtype으로 변환하는 데 성공했지만, 같은
`Tensor.clamp_min` node가 float32로 남아 다시 `MBLT_COMPILE`에 실패했다. 따라서
원인은 mask dtype이 아니라 qbcompiler 1.2 Torch parser가 `clamp_min` method를
지원하지 않는 것이다.

두 attempt는 다음 사실을 보존하며 수정하거나 덮어쓰지 않는다.

- stock: source·calibration pass, boolean `clamp_min`에서 MBLT fail
- compat revision 1: CPU equivalence pass, static patchification과 mask cast 적용,
  float32 `clamp_min`에서 MBLT fail
- 두 attempt 모두 MBLT/MXQ artifact 없음, MXQ와 ARIES stage는 `not_run`

## 벤치마크 유효성 원칙

이 보완은 framework의 benchmark model, loader, evaluator, runtime 또는 외부 ABI를
변경하지 않는다. exact Hugging Face commit과 model weight도 변경하지 않는다. 오직
compiler 전용 compat wrapper에서 다음 수학적 항등 변환을 적용한다.

```text
denominator.clamp_min(1.0)
    -> denominator.clamp(min=1.0)
```

두 표현은 모든 finite denominator에 대해 같은 함수를 계산한다. qbcompiler 1.2 wheel의
Torch parser는 `Tensor.clamp`를 명시적으로 처리하지만 `Tensor.clamp_min`은 처리하지
않는다. 따라서 이것은 model weight·architecture·task 의미를 바꾸는 우회가 아니라
compiler가 지원하는 동일 연산 표기로의 lowering이다.

그럼에도 생성 artifact를 stock이라고 부르지 않는다. 결과와 문서는 항상
`compat-static-patchifier` variant 및 compat recipe revision을 표시하고, stock과 이전
compat 실패를 함께 제시한다. 다른 backend와 비교할 때도 이 provenance를 숨기지
않는다.

## 선택한 구현

현재 compat wrapper의 세 rewrite를 한 compiler-only module 경계에 모은다.

1. 외부 `past_observed_mask bool [1,512,7]` ABI를 유지한다.
2. wrapper 내부에서 mask를 `past_values.dtype`으로 변환한다. bool의 0/1 값은
   float32의 0.0/1.0으로 정확히 표현된다.
3. checkpoint의 `model.patchifier`만 고정 42-patch slice/stack 구현으로 교체한다.
4. checkpoint의 std scaler만 동일 수식의 compat scaler로 교체하고 denominator에
   `clamp(min=1.0)`을 사용한다. epsilon, mean, variance, loc, scale 계산 순서와 반환
   구조는 stock 구현을 유지한다.

기존 variant 이름은 유지하되 manifest와 compile report에 다음을 기록한다.

- `compat_recipe_revision`
- 적용한 rewrite 목록
- compiler recipe source file의 SHA256
- exact source model commit SHA
- stock parent attempt identity

새 실행은 기존 attempt를 재사용하지 않고 새로운 immutable child attempt를 만든다.

## 동등성 및 실패 정책

compiler를 import하기 전에 exact checkpoint의 stock wrapper와 compat wrapper를 같은
입력으로 비교한다. ETTh1 all-observed calibration sample뿐 아니라 단위 테스트에서
다음 mask를 검사한다.

- 모두 observed
- 일부 channel/time point가 missing인 sparse mask
- channel 하나가 전부 missing인 zero-denominator mask

각 경우에 output shape `float32 [1,96,7]`, finite 값, `rtol=1e-5`, `atol=1e-6`
동등성을 요구한다. FX graph 검사에서는 compat graph에 `unfold`와 `clamp_min`이 없고
지원 대상인 `clamp`가 존재해야 한다. 외부 placeholder의 mask dtype은 계속 bool이어야
한다.

어느 검사든 실패하면 compiler를 호출하지 않고 새 attempt의 해당 stage를 fail로
기록한다. MBLT가 다시 실패하면 추가 즉석 graph rewrite를 만들지 않고 실제 오류를
기록한 뒤 PatchTST를 qbcompiler 1.2 실패로 판정한다. MBLT와 MXQ가 모두 artifact
path·size·SHA256과 함께 pass인 경우에만 ARIES global8 strict verifier로 넘어간다.

## 검토한 대안

- `torch.maximum(denominator, torch.ones_like(denominator))`: parser가 지원하지만
  node와 constant 생성이 늘어난다. 직접 지원되는 `Tensor.clamp`보다 불필요하게
  복잡해 선택하지 않는다.
- host에서 미리 normalization: NPU artifact의 입력 의미를 바꾸고 benchmark 경계를
  흐리므로 사용하지 않는다.
- stock 실패를 숨기고 compat를 기본 모델로 승격: provenance를 왜곡하므로 금지한다.

## 완료 조건

1. 새 RED 테스트가 현재 compat graph의 `clamp_min` 잔존을 재현한다.
2. sparse와 zero-denominator mask를 포함한 stock/compat CPU 동등성이 통과한다.
3. compat FX graph에서 `unfold`와 `clamp_min`이 제거되고 `clamp`가 확인된다.
4. manifest와 compile report가 recipe revision, rewrite 목록, recipe SHA를 기록한다.
5. 관련 PatchTST·runner·attempt 회귀 테스트와 shell syntax 검사가 통과한다.
6. 서버에서는 새 immutable attempt로만 재실행하며 실제 결과를 PR ledger에 기록한다.

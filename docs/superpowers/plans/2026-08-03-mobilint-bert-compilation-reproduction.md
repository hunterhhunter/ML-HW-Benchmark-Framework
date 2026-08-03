# Mobilint BERT MXQ 컴파일 재현 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before claiming success.

**목표:** Ubuntu 22.04 x86-64 Python 3.10 compiler host에서 Mobilint
qbcompiler 1.2 wheel을 사용해 BERT SST-2와 SQuAD v1의 calibration data,
embedding weight, MBLT와 MXQ를 재현하는 저장소 소유 스크립트와 한국어 runbook을
추가한다.

**구조:** benchmark runtime에는 컴파일을 넣지 않는다. 표준 라이브러리 중심의 task
contract와 lazy import를 사용하는 Python recipe를 `framework/tools`에 두고, 기존
`MobilintBertEmbeddingTransform`으로 calibration 경계를 공유한다. 한 개의 shell
entrypoint가 host/wheel 검증, 전용 venv, 고정 dependency 설치와 두 task 실행을
조정한다. 실제 vendor compiler 없이도 contract와 API wiring을 단위 테스트한다.

**기술:** Bash, Python 3.10, argparse, dataclasses, NumPy, PyTorch, Transformers,
Datasets, qbcompiler 1.2, pytest, Markdown.

## 공통 제약

- `qbruntime`, ARIES device, driver 또는 `mobilint-cli`를 compiler script에서 사용하지
  않는다.
- compile target은 `aries-rb`로 고정하고 사용자 입력으로 다른 target을 받지 않는다.
- shell의 `--task all`은 `sst2`와 `squad1`만 뜻한다.
- 기존 task output 또는 artifact를 덮어쓰지 않는다.
- wheel, model/cache, calibration, weight, MBLT, MXQ, log를 Git에 추가하지 않는다.
- `--help`와 `--describe`는 qbcompiler, datasets, transformers를 import하지 않아도
  동작해야 한다.
- SQuAD source 출력 `start_logits,end_logits`와 검증된 qbruntime positional 출력
  `end_logits,start_logits`를 별도 contract 필드로 유지한다.
- 실제 compiler 성공은 Ubuntu 22.04 compiler host에서만 주장한다.

---

## 작업 1: Task contract와 compiler-safe model helper

**파일:**

- 생성: `framework/tools/mobilint_bert_compile/__init__.py`
- 생성: `framework/tools/mobilint_bert_compile/common.py`
- 생성: `framework/tests/test_mobilint_bert_compile.py`

**인터페이스:**

- `TaskSpec`
- `get_task_spec(task: str) -> TaskSpec`
- `contract_to_dict(spec: TaskSpec) -> dict[str, object]`
- `extract_embedding_weights(model: object) -> dict[str, Tensor]`
- `make_compiler_model(task: str, model: object) -> object`
- `sha256_file(path: Path) -> str`

- [ ] `TaskSpec`이 model/dataset/max length, source outputs, runtime outputs,
  dynamic embedding ABI와 `aries-rb` target을 정확히 직렬화하는 실패 테스트를 쓴다.
- [ ] unknown task를 거부하고 task mapping이 immutable한지 테스트한다.
- [ ] 작은 `BertConfig` 모델에서 다섯 embedding weight key와 shape를 검증하는 테스트를
  쓴다.
- [ ] 작은 QA 모델에서 compiler wrapper의 두 logits가 원본 출력과 bitwise 동일한지
  테스트한다.
- [ ] 테스트가 import/구현 부재로 실패하는지 확인한다.
- [ ] optional package import를 함수 내부로 지연한 최소 구현을 추가한다.
- [ ] focused test를 실행한다.

```bash
cd framework
python -m pytest -q tests/test_mobilint_bert_compile.py
```

---

## 작업 2: 재현 가능한 calibration과 embedding weight 준비

**파일:**

- 생성: `framework/tools/mobilint_bert_compile/prepare.py`
- 수정: `framework/tests/test_mobilint_bert_compile.py`

**인터페이스:**

- `select_calibration_indices(dataset_size: int, count: int = 32) -> tuple[int, ...]`
- `prepare_task(task: str, output_root: Path) -> dict[str, object]`
- module CLI: `python -m tools.mobilint_bert_compile.prepare`

- [ ] 32개 index가 validation split의 처음과 끝을 포함하고 결정론적인지 실패 테스트를
  쓴다.
- [ ] dataset보다 sample 수가 작거나 task output 디렉터리가 이미 있으면 준비를
  거부하는 테스트를 쓴다.
- [ ] fake tokenizer/model/dataset과 실제 runtime embedding transform을 사용해 생성된
  `.npy`가 contiguous `float32 [1,L,width]`인지 테스트한다.
- [ ] manifest에 model/dataset, 선택 index, token length, weight path, expected compile
  contract가 들어가는지 테스트한다.
- [ ] `--describe` subprocess가 network/vendor package 없이 JSON contract를 출력하는지
  테스트한다.
- [ ] 테스트 실패를 확인한 뒤 lazy loading으로 prepare 구현을 추가한다.
- [ ] `compile-environment.json`에는 Python/platform과 설치된 primary package version을
  기록하되 secret이나 cache 내용을 넣지 않는다.
- [ ] focused test를 실행한다.

---

## 작업 3: MBLT/MXQ compiler API recipe와 artifact 보고

**파일:**

- 생성: `framework/tools/mobilint_bert_compile/compile.py`
- 수정: `framework/tests/test_mobilint_bert_compile.py`

**인터페이스:**

- `build_feed_dict(inputs: Mapping[str, Tensor]) -> dict[str, object]`
- `validate_calibration_set(task_root: Path, manifest: Mapping) -> list[Path]`
- `run_mblt_compile(...) -> Path`
- `run_mxq_compile(...) -> Path`
- module CLI with `--task`, `--stage {mblt,mxq,all}`, `--artifact-root`,
  `--describe`

- [ ] calibration file count/name과 manifest가 다르면 compile 전에 거부하는 테스트를
  쓴다.
- [ ] existing `.mblt`/`.mxq`와 zero-byte compiler 결과를 거부하는 테스트를 쓴다.
- [ ] fake `wrap_tensor`/`set_attention_mask`로 세 token input의 sequence dimension이
  dynamic이고 padding semantic이 지정되는지 테스트한다.
- [ ] fake compiler API로 MBLT 호출의 `target_device="aries-rb"`, `backend="torch"`,
  `cpu_offload=True`를 검증한다.
- [ ] fake compiler API로 MXQ 호출의 calibration directory, `inference_scheme="all"`,
  method/output/mode와 percentile/top-k 설정을 검증한다.
- [ ] 테스트 실패를 확인한 뒤 정확한 qbcompiler 1.2 API 호출을 구현한다.
- [ ] CPU smoke forward로 task별 source output 이름과 논리 shape를 확인한 후에만
  compiler를 호출한다.
- [ ] 각 성공 artifact의 size/SHA256와 source/runtime output order를
  `compile-report.json`에 병합 기록한다.
- [ ] focused test를 실행한다.

---

## 작업 4: Compiler-only one-shot shell script

**파일:**

- 생성: `framework/scripts/compile_mobilint_bert.sh`
- 수정: `framework/tests/test_mobilint_bert_compile.py`

**CLI:**

```bash
bash framework/scripts/compile_mobilint_bert.sh \
  --wheel ~/Downloads/qbcompiler-1.2.0-py3-none-any.whl \
  --python "$(command -v python3.10)" \
  --task all \
  --output-root "$PWD/mobilint-bert-artifacts"
```

- [ ] `bash -n`, `--help`, unknown/missing argument를 검사하는 테스트를 쓴다.
- [ ] script text가 Ubuntu 22.04, x86_64, CPython 3.10과 exact wheel checksum guard를
  포함하는지 테스트한다.
- [ ] script text가 `qbruntime`, `mobilint-cli`, `/dev/aries`를 사용하지 않는지
  테스트한다.
- [ ] 실패를 확인한 뒤 `set -euo pipefail` one-shot script를 구현한다.
- [ ] pip 없는 venv에는 `ensurepip`을 사용하고 기존 venv의 Python version을 다시
  검사한다.
- [ ] wheel과 primary dependency version을 설치한 뒤 qbcompiler/onnxruntime import와
  compiler signature를 출력한다.
- [ ] task마다 prepare 후 compile `--stage all`을 호출하고 종료 시 artifact 경로,
  크기, SHA256를 요약한다.
- [ ] shell/static focused test를 실행한다.

---

## 작업 5: 정식 한국어 runbook과 기존 문서 연결

**파일:**

- 생성: `docs/mobilint-bert-compilation.md`
- 수정: `docs/mobilint-aries-transformers.md`
- 수정: `framework/tests/test_mobilint_bert_compile.py`

- [ ] canonical runbook에 prerequisites, wheel 배치, 전체 명령, task별 명령, 산출물
  구조, contract, hash 확인, 재실행/실패 처리와 benchmark handoff를 작성한다.
- [ ] Docker와 ARIES device가 필요하지 않으며 network는 dependency/model/dataset
  다운로드에 필요하다고 명시한다.
- [ ] `aries-rb`는 하나의 ARIES compile target이고 `all`은 두 task일 뿐임을 명시한다.
- [ ] `.mblt`가 `.mxq` 입력이 아니라 별도 compile 결과임을 명시한다.
- [ ] SQuAD source output과 검증된 runtime output 순서 차이를 표로 남긴다.
- [ ] 기존 ARIES transformer 문서의 BERT artifact 준비 구간에서 새 runbook을
  상대 링크한다.
- [ ] 테스트에서 문서의 실제 script/module 경로가 모두 존재하는지 확인한다.

---

## 작업 6: 전체 검증과 PR 갱신

- [ ] syntax/format 검사:

```bash
bash -n framework/scripts/compile_mobilint_bert.sh
python -m py_compile \
  framework/tools/mobilint_bert_compile/common.py \
  framework/tools/mobilint_bert_compile/prepare.py \
  framework/tools/mobilint_bert_compile/compile.py
git diff --check
```

- [ ] focused test:

```bash
cd framework
python -m pytest -q tests/test_mobilint_bert_compile.py
```

- [ ] 기존 Mobilint BERT 회귀 테스트:

```bash
cd framework
python -m pytest -q \
  tests/test_mobilint_bert_compile.py \
  tests/test_mobilint_bert_embedding.py \
  tests/test_mobilint_bert_profiles.py \
  tests/test_inspect_mobilint_mxq.py \
  tests/test_mobilint_runtime.py \
  tests/test_main_paths.py
```

- [ ] 전체 test suite를 실행하고 기존 Furiosa timing failure가 재현되면 이번 변경과
  무관한 known failure로 정확히 기록한다.
- [ ] `superpowers:verification-before-completion` 절차로 최신 출력만 근거로 결과를
  정리한다.
- [ ] 생성 코드와 문서를 커밋하고 remote branch에 push한다.
- [ ] PR #47의 한국어 본문에 컴파일 명령, 입출력 계약, 문서 링크와 compiler-only
  검증 범위를 추가한다.

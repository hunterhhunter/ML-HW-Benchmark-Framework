# Furiosa Hub Runtime Input Contract Design

## Goal

Allow the existing `furiosa-rngd` framework pipeline to load an official
Furiosa Hugging Face model reference directly, without starting
`furiosa-llm serve` and without requiring an explicit `.fxb` file.

The intended command is:

```bash
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path furiosa-ai/Llama-3.1-8B-Instruct \
  --dataset datasets/squad2/val.json \
  --inference-mode e2e
```

## Scope

This change is limited to the Furiosa model-loading input contract.

- `--model-path` accepts either a Hugging Face repository ID or an existing
  local model/artifact directory.
- `--fxb` remains supported but becomes optional.
- `--artifact` remains a backward-compatible alias for an explicit `.fxb`.
- When no FXB is supplied, `FuriosaLlmRuntime` calls `LLM(model_reference, ...)`
  and lets Furiosa-LLM select the artifact revision compatible with the
  installed SDK.
- When an FXB is supplied, the existing
  `LLM(model_reference, fxb=fxb_path, ...)` behavior is preserved.
- E2E generation, native async generation, prompt preprocessing, evaluation,
  metrics, and result persistence are unchanged.

The OpenAI-compatible HTTP server and external-server benchmark are outside
this change.

## Data Contract

`ModelSpec.model_paths["hf_model"]` remains the source of truth for the model
reference. It stores the original string unchanged so both repository IDs and
local directories can pass through to Furiosa-LLM and Transformers.

`CompiledModel.artifact_path` becomes optional. A non-null value continues to
mean an existing local compiled artifact and is validated exactly as before.
A null value means the selected runtime resolves its executable artifact from
the model reference. The Furiosa target uses this state when `--fxb` is absent;
all existing local-artifact runtimes continue to provide a path.

## CLI Validation

For `furiosa_llm`:

1. The task must be `NLP_GENERATION`.
2. `--model-path` must be a non-empty string.
3. If the model reference resolves to an existing filesystem entry, it must be
   a directory.
4. A non-existing string is passed to Furiosa-LLM as a possible Hub repository
   ID; repository existence and revision errors remain SDK-owned.
5. If `--fxb` or its `--artifact` alias is present, it must be an existing
   `.fxb` file.
6. If no tokenizer override is present, `--tokenizer-path` defaults to the
   original model reference string.

## Runtime Loading

`FuriosaLlmRuntime.load()` always passes the existing device, parallelism,
memory, cache, seed, and scheduler options. It adds the `fxb` keyword only when
`CompiledModel.artifact_path` is not null.

This distinction matters because omitting `fxb` activates Furiosa-LLM's own
Hub revision selection and artifact discovery, while passing `fxb=None` or a
fabricated filesystem path would misrepresent the caller's intent.

## Error Handling

- Local model references that already exist as regular files fail during CLI
  validation.
- Explicit FXB errors fail before vendor SDK initialization.
- Hub authentication, missing repository, missing compatible artifact, and SDK
  version mismatch errors propagate from Furiosa-LLM with their original cause.
- Runtime unloading remains idempotent and always calls `LLM.shutdown()` after
  successful initialization.

## Compatibility

The following existing invocation remains valid and behaviorally unchanged:

```bash
python src/main.py \
  --model llama-3.1-8b \
  --target furiosa-rngd \
  --model-path /models/Llama-3.1-8B \
  --fxb /models/llama-3.1-8b.fxb \
  --dataset datasets/squad2/val.json
```

No ONNX Runtime, vLLM, HailoRT, DEEPX, compiler, evaluator, or result schema
contract changes are included.

## Tests

Tests will prove:

- Hub model references pass CLI validation without an FXB.
- Local model directories pass without an FXB.
- Existing explicit-FXB and `--artifact` alias behavior remains intact.
- Invalid local files and invalid explicit FXBs still fail early.
- A null compiled artifact is accepted as a runtime-resolved artifact.
- Furiosa runtime loading omits the `fxb` keyword for automatic resolution.
- Furiosa runtime loading still passes an explicit FXB unchanged.
- Compatibility checks handle a null artifact without dereferencing it.
- The existing Furiosa registry, synchronous runtime, and native async tests
  remain green.

Hardware execution is verified separately on the RNGD host with a short E2E
run, because the development environment uses a fake Furiosa SDK.

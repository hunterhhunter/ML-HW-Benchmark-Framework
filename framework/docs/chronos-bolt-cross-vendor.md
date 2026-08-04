# Chronos-Bolt Tiny: RBLN and Furiosa execution

This workflow validates the exact fixed external ABI `(1,512)` FP32 context to
`(1,9,64)` FP32 quantiles. The loaded Tiny checkpoint has `d_model=256`, 32
data patches, and a learned REG token, so the compiled Transformer-core ABI is:

| Core input | Shape | Type |
| --- | --- | --- |
| `input_embeds` | `(1,33,256)` | FP32 |
| `attention_mask` | `(1,33)` | FP32 |
| `decoder_input_embeds` | `(1,1,256)` | FP32 |

The command always runs the CPU preflight first. It rejects a device compile if
either finite or NaN CPU parity against the original model fails. The artifact
or first device output is then compared with the FP32 CPU core at `rtol=1e-3`,
`atol=1e-3`; an exit status of zero and a `device_verified` JSON result are the
acceptance gate.

## Rebellions CA22

Run from the checkout that contains the Chronos-Bolt files. Choose a new output
directory for every attempt: the command deliberately refuses to overwrite a
prior `.rbln` or result JSON.

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework-rbln/framework
RBLN_PY=/home/etri_ecas/ML-HW-Benchmark-Framework-rbln/.venv-rbln/bin/python
MODEL=/home/etri_ecas/ML-HW-Benchmark-Framework/framework/models/amazon_chronos-bolt-tiny
OUT=results/chronos-bolt/rbln-$(date -u +%Y%m%dT%H%M%SZ)

uv pip install --python "$RBLN_PY" 'chronos-forecasting==2.3.1'
"$RBLN_PY" -m pytest tests/test_chronos_bolt_contracts.py tests/test_chronos_bolt_host_adapter.py tests/test_chronos_bolt_core.py tests/test_chronos_bolt_compile_cli.py tests/test_chronos_bolt_rbln.py -q
rbln-smi -b -j -d 0 > "${OUT}-before.json"
"$RBLN_PY" tools/chronos_bolt_compile.py --vendor rbln --model-path "$MODEL" --output-dir "$OUT"
rbln-smi -b -j -d 0 > "${OUT}-after.json"
```

Success produces `chronos-bolt-tiny-core.rbln` and `rbln-result.json` under
`$OUT`. The result includes its SHA-256, inspected CA22 input/output ABI, CPU
preflight evidence, and first-device-core error.

## Furiosa RNGD

The Furiosa command compiles only through `torch.compile` with
`fullgraph=True`, `dynamic=False`, and `eager_fallback=False`. It does not
create a portable FXB artifact; its success evidence is the strict first RNGD
call and `furiosa-result.json`.

```bash
cd /home/etri_ecas/ML-HW-Benchmark-Framework/framework
FURIOSA_PY=/home/etri_ecas/ML-HW-Benchmark-Framework/.venv-furiosa-torch/bin/python
MODEL=../framework/models/amazon_chronos-bolt-tiny
OUT=results/chronos-bolt/furiosa-$(date -u +%Y%m%dT%H%M%SZ)

uv pip install --python "$FURIOSA_PY" 'chronos-forecasting==2.3.1'
"$FURIOSA_PY" -m pytest tests/test_chronos_bolt_contracts.py tests/test_chronos_bolt_host_adapter.py tests/test_chronos_bolt_core.py tests/test_chronos_bolt_compile_cli.py tests/test_chronos_bolt_furiosa.py -q
furiosa-smi > "${OUT}-before.txt"
"$FURIOSA_PY" tools/chronos_bolt_compile.py --vendor furiosa --model-path "$MODEL" --output-dir "$OUT"
furiosa-smi > "${OUT}-after.txt"
```

If the model snapshot is absent on the Furiosa host, download it into the
selected local path before running the command:

```bash
"$FURIOSA_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('amazon/chronos-bolt-tiny', local_dir='$MODEL')"
```

## Mobilint ARIES

Mobilint compilation is intentionally not part of these server commands. Run
the ARIES adapter locally only after its `qbcompiler` wheel is available; the
same `(1,33,256)`, `(1,33)`, `(1,1,256)` core ABI is reserved for that adapter.

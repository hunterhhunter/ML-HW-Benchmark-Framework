# Server-NPU power tracing on current main

## Decision and scope

Forward-port the existing server-NPU raw power-trace implementation onto the
current `origin/main` without merging the older measurement branch wholesale.
The measurement defaults approved for this port are a 100 ms *target* polling
interval, a 60 s pre-inference baseline in the same run, and 2,785 ETTh1
TTM-R2 windows with stride 1. The baseline retains the old implementation's
**model-loaded, pre-inference** meaning: this change does not move collection
before model loading. A model-unloaded idle phase, post-inference idle phase,
new power-analysis formula, and hardware reruns are separate work.

The implementation source is the measurement code at
`70d2f98bc3f3a905bfc4fc08978abe380d90ca27`, plus its three known local
changes: six target intervals from 0.2 to 0.1 s, the monitor's default trace
baseline from 3 to 60 s, and `TTM_R2_EXPECTED_WINDOWS` from 240 to 2,785.
The destination baseline is `origin/main` at
`dbf558f53dcede6881579f9fd0534f13bf52174e`. These SHAs identify the
reviewed snapshots, not a requirement to reset either checkout. The older
source has no uncommitted `main.py` change. Existing current-main features,
including `--async-preload-inputs`, must remain intact.

## Integration strategy

Port the trace as a small set of contracts and integration points:

1. Add the source trace schema/writer and vendor power readers needed for
   Furiosa RNGD, Rebellions CA22, and Mobilint ARIES. Integrate with the
   run-artifact reservation already present on current `main`; adapt current
   collectors only where necessary and retain summary-monitor behavior and
   target metadata.
2. Add the source `--power-trace` CLI flag and monitor setup to current
   `main.py`. `--monitor` remains an independent summary option: requesting a
   raw trace must create a monitor even without `--monitor`.
3. Reserve one `run_id` before collection, write the trace under the result
   directory, and link its path, SHA-256, sample count, source, scope, and
   status to that run's CSV and sidecar. Retain the current result schema and
   async runtime-only semantics; do not accidentally reintroduce old
   quality/evaluator work into `--async-pass runtime-only`.
4. Reconcile the source E2E and async lifecycle with current `main.py` rather
   than replacing the whole runner. Start collection after model loading and
   warmup, keep it active across the benchmark's measured inference phase,
   and always close/flush it on both normal and exceptional paths. For async,
   keep existing request-completion, outstanding-request, and result-commit
   invariants; the power polling thread must not change the Runtime-call
   timing boundary.
5. Set the six relevant server target raw-power polling intervals to 0.1 s
   and the trace baseline to 60 s. The ordinary summary monitor interval is
   separate. Set TTM-R2's expected-window contract to 2,785 only alongside
   a stride-1 ETTh1 loader check; fail rather than silently accepting 240
   windows or a different dataset. Non-TTM models retain their sample limits.

This is a selective forward port, not a branch merge. Changes to unrelated
edge targets, vendor runtime APIs, precision, model artifacts, prompt
templates, or quality scoring are out of scope.

## Trace and result contract

The raw CSV keeps schema version 1.0 and the existing fields:
`run_id`, `sample_index`, `phase`, `target_id`, `collector`,
`monitor_source`, `device_id`, `power_scope`, scheduled and observed elapsed
milliseconds, query latency, watts, `sample_status`, and `error_code`.
Phases remain `baseline` and `inference`; there is no invented `post_idle`
phase. The 60 s baseline begins after runtime/model setup and, where the
runner warms up, after warmup. This is not an unloaded-device idle wattage.

The collector reports its actual physical scope rather than a normalized
``NPU-core power`` claim: Furiosa `device`, RBLN `whole_card`, and Mobilint
`device_total` in the validated server setup. The 100 ms interval is a
schedule target, not proof that the vendor sensor refreshes every 100 ms.
Keep actual timestamps, repeated readings, and unsuccessful sample rows.
An `overrun`/`deadline_missed` row is missing power data, not zero watts; a
run with such rows remains `partial`, even if its inference succeeded. Do not
promote `partial` to `complete` by a missing-sample percentage threshold.

When power tracing is disabled, benchmark behavior and prior result fields
must remain compatible; no trace is generated. When requested, missing
telemetry or writer errors must be explicit (`unavailable`, `partial`, or
`failed` as appropriate), not silently reported as complete. A completed
trace's SHA-256 and sample count must agree with its saved CSV, and all trace
rows must use the reserved result `run_id`.

## Verification and acceptance

First run focused unit tests for trace serialization/status, each vendor
reader, monitor lifecycle (60 s simulated with injected clock/wait rather
than a real minute), artifact reservation/result linkage, and E2E/async
cleanup. Add regression coverage for trace-off behavior and preservation of
current-main async options. Verify `--help` exposes `--power-trace` and
`--async-pass runtime-only`; use CPU/mock paths to exercise one E2E and one
offline runtime-only result without an NPU. Run the affected framework test
suite with the repository's test environment and `git diff --check`.

The port is code-complete only after those tests pass. Real server readiness
requires a separate hardware smoke test for all three vendors, checking
baseline and inference rows, power source/scope, `run_id`, SHA-256, sample
counts, completion counters, and `partial` handling. It must not be described
as physically validated merely because unit tests pass. Record source and
destination revisions and preserve any observed sensor overruns in the
results. No prior run's energy numbers are automatically transferable to the
new code.

## Review risks

- A 60 s baseline adds one minute to every traced case and is not a global
  idle measurement. Changing its location would alter the measurement
  definition and needs its own design and rerun.
- 100 ms polling can overrun under load; status and timestamps must make
  that visible without changing the inference success decision.
- Updating the TTM-R2 expected count changes the accepted workload. A
  2,785-window claim requires the stride-1 ETTh1 preparation to be present
  and independently checked.
- The source and destination `main.py` differ substantially. Copying the
  source file wholesale could delete newer async/input-loading behavior;
  reconcile only the trace-specific paths and test both modes.

# Framework-Wide Raw Power Trace Design

## Purpose

Add opt-in raw power sampling to the existing hardware-monitoring path for
every benchmark model that runs through the common synchronous or asynchronous
inference runners. The first supported accelerator targets are Furiosa RNGD,
Rebellions RBLN, and Mobilint ARIES.

This feature records instantaneous power readings in watts during a fixed idle
baseline and the measured inference loop. It does not calculate energy,
idle-subtracted power, or joules per inference. Those calculations belong to a
separate, future analysis workflow.

TTM-R2 is the first physical acceptance workload because its 240-window runs
have already been exercised on all three targets. No power-trace component may
import a TTM-R2 module, assume 240 samples, or branch on a model name.

## Scope

The implementation must:

- add an explicit `--power-trace` CLI flag;
- work without `--monitor` and compose correctly with `--monitor`;
- cover the measured portions of both `e2e` and `async_queue` modes;
- record a three-second post-warmup idle baseline;
- record raw instantaneous power throughout the measured inference phase;
- publish one atomic CSV sidecar per run;
- link the sidecar to the normal result row through run metadata;
- preserve inference and task-quality results when power telemetry is
  unavailable or fails;
- remove energy integration from vendor collectors.

The implementation must not:

- calculate total energy or any other joule value;
- subtract idle power;
- calculate `J/inference`;
- create an energy-analysis command or analysis CSV;
- treat missing power as zero or estimate it from voltage;
- claim that unlike vendor telemetry scopes are physically equivalent;
- parse a human-oriented Furiosa table as the production data source.

The initial physical target scope is Furiosa RNGD, Rebellions, and Mobilint
ARIES. The common collector interface permits additional targets later, but
Hailo, DeepX, Mobilint REGULUS, NVIDIA, and other targets are not added to the
new raw trace contract in this change.

## Existing Structure and Required Refactoring

`HWMonitor` already owns collector lifecycle and background polling.
`BenchmarkRunner` starts it after warmup and stops it after the synchronous
measured loop. The asynchronous runner has corresponding guarded start, stop,
and summary callbacks. Target definitions already select vendor collectors
through `monitor_names` and `monitor_options`.

The current Rebellions and Mobilint collectors also integrate power into
energy independently. That is the wrong ownership boundary for raw trace
collection and duplicates numerical behavior across vendors. This change
removes the following collector responsibilities:

- previous-power and previous-time state used only for integration;
- trapezoidal integration;
- collector-owned `hw_accel_energy_j` output;
- stop-boundary reads whose only purpose is completing an energy summary.

Collectors remain responsible for device selection, SDK lifecycle, raw value
validation, source metadata, and instantaneous telemetry. `HWMonitor` remains
the sole lifecycle owner and gains optional raw power-trace orchestration.

Legacy `--monitor` summaries for CPU, RAM, utilization, temperature, and
instantaneous power averages remain available. Running only `--power-trace`
does not add average or maximum power values to the result row. Energy summary
fields are removed in all modes.

## User-Facing CLI Contract

The new flag is:

```text
--power-trace
```

Behavior by flag combination is:

| `--monitor` | `--power-trace` | Behavior |
|---|---|---|
| absent | absent | Existing unmonitored benchmark behavior |
| present | absent | Existing hardware summaries, except removed energy fields |
| absent | present | Raw power CSV only; no hardware summary merge |
| present | present | Existing summaries plus the raw power CSV |

The baseline duration is fixed at three seconds for the initial contract. It
is not exposed as another CLI option. Target-specific power sampling periods
are configuration owned by the target and collector, not model profiles.

`--monitor-interval` continues to control ordinary monitoring only. It does
not silently change the power sampling period.

## Collector Power Contract

Extend the monitor collector abstraction with an optional instantaneous-power
capability. A power-capable collector returns a structured reading containing:

- `power_w`: a finite, non-negative float, or no value on failure;
- `status`: `ok`, `unavailable`, or `read_error`;
- `source`: stable SDK or tool identifier;
- `power_scope`: the scope documented by that source;
- `device_id`: the exact selected accelerator;
- `error_code`: a bounded, sanitized diagnostic when no value is returned.

`HWMonitor`, not the collector, measures query start and finish times. This
keeps timestamps and query-latency semantics identical across vendors.

Ordinary `collect()` remains available for the existing summary monitor.
Power-capable collectors add a narrow `collect_power()` path so high-frequency
power sampling does not repeatedly query temperature, memory, utilization,
current, or voltage. The collector serializes ordinary and power calls against
the same SDK session; the framework must never issue concurrent calls through
one vendor collector.

An unavailable capability is data, not an inference exception. A target with
no usable direct power source produces `power_trace_status=unavailable` and no
fabricated samples.

## Vendor Implementations

### Mobilint ARIES

Use the existing `mbltml` device session and
`mbltmlGetTotalPower(device_id)`. The source identifier is `mbltml`, and the
scope is recorded as `device_total`, matching the vendor API's total-power
boundary without relabeling it as process or NPU-core power.

`collect_power()` invokes only the total-power API. Existing current and
voltage collection remains part of ordinary `--monitor` behavior and is not
used to derive watts. Mobilint REGULUS remains unavailable for raw power until
a power API is verified on that exact platform.

The initial ARIES sampling-period candidate is 50 ms. Real-device acceptance
may increase this target option if the installed SDK's latency or sensor update
rate cannot support it without perturbing inference.

### Rebellions

The currently verified source is the JSON output of exact argv
`rbln-smi -b -j -d <device_id>`, with `card_power` interpreted by the existing
strict parser. The source identifier is `rbln-smi-json`, and the scope is
`whole_card`.

Remove the collector's energy state and retain its device/schema validation.
The existing subprocess source is expensive and is currently throttled to at
least one second, so the initial period remains 1,000 ms. Server probing must
check for a supported lower-overhead library or persistent telemetry interface.
If none exists, the CLI period is reduced only when measured query latency and
sensor update rate justify it.

### Furiosa RNGD

Add a `FuriosaCollector` and register it on both Furiosa RNGD targets. The
production collector must use the installed official Python SMI binding rather
than parse `furiosa-smi info` table text.

Before fixing the adapter call, a read-only server probe records:

- installed SMI Python package and version;
- device enumeration result and selected device identifier;
- the exact callable that returns power;
- returned type and unit;
- 100-call query-latency distribution;
- observed sensor update cadence.

The implementation then binds explicitly to that verified API and fails
closed as `unavailable` when the package or capability is absent. It must not
guess return types, units, or method names through broad reflection. The
initial sampling-period candidate after API verification is 50 ms. The final
`power_scope` value is taken from the verified API documentation and server
behavior, not inferred from the CLI display.

## Sampling Scheduler

`HWMonitor` owns one scheduler for each vendor collector session. When both
ordinary monitoring and power tracing are active, it serializes their due
operations so two threads never enter the same SDK session concurrently.
Power sampling is scheduled first when both operations are due.

Scheduling uses monotonic deadlines:

```text
next_deadline = previous_deadline + configured_period
```

It does not sleep for a full period after each query because that would add
query time to the interval and drift. When a query overruns one or more
deadlines, the scheduler does not execute a burst of catch-up calls. It records
an `overrun` attempt and advances to the next future deadline.

Each attempted read records its scheduled elapsed time. `HWMonitor` records
monotonic query start and finish times and assigns the observation time to
their midpoint. This preserves actual timing and makes later analysis robust
to jitter without doing analysis in the benchmark process.

Ten milliseconds is not a common default. A target's accepted period must be
no shorter than both twice its measured query-latency p99 and its observed
sensor update period. In five alternating trace-off and trace-on smoke pairs,
the trace-on median measured-loop latency must remain within 2% of trace-off
and no individual run may fail because of telemetry before a shorter period is
accepted. These checks calibrate configuration; they do not run automatically
before every benchmark.

## Measurement Lifecycle

### Synchronous `e2e`

1. Load the runtime and record any existing load-time monitor state.
2. Run the configured warmup with no power trace active.
3. Start collectors and the trace writer in `baseline` phase.
4. Record the three-second idle baseline.
5. Request one boundary sample and switch to `inference` immediately before
   `InferenceEngine.run_e2e()`.
6. Keep tracing for the entire existing measured loop.
7. Request one final boundary sample immediately after the loop returns or
   raises.
8. Stop collectors and finalize the trace without replacing the primary
   inference exception.

### Asynchronous `async_queue`

The same phases apply, but the inference transition occurs after async warmup
and immediately before the measured producer is allowed to submit its first
logical request. The inference phase ends only after accepted work reaches its
existing terminal completion and outstanding-request shutdown boundary. It
must not stop merely because request submission has ended.

The trace covers the existing measured loop, including intervals between
accelerator calls. It is therefore a device-power trace over the framework's
measured inference phase, not an isolated NPU-kernel trace.

## Raw CSV Contract

One row represents one scheduled power-read attempt, including unsuccessful
attempts. The schema is:

```csv
schema_version,run_id,sample_index,phase,target_id,collector,monitor_source,device_id,power_scope,scheduled_elapsed_ms,observed_elapsed_ms,query_latency_ms,power_w,sample_status,error_code
```

Field rules are:

- `schema_version` is `1.0`;
- `run_id` is the reserved framework run identifier;
- `sample_index` starts at zero and increases by one for every attempted row;
- `phase` is `baseline` or `inference`;
- `scheduled_elapsed_ms` is relative to trace start;
- `observed_elapsed_ms` is the query midpoint relative to trace start;
- `query_latency_ms` is query finish minus query start;
- `power_w` is the unmodified finite watt value for `ok`, otherwise empty;
- `sample_status` is `ok`, `unavailable`, `read_error`, or `overrun`;
- `error_code` is empty for `ok` and otherwise contains a sanitized bounded
  diagnostic, never an unrestricted exception or SDK output dump.

The CSV contains no energy, averages, idle subtraction, inference count, or
quality metric.

## Artifact Reservation and Atomic Publication

Generalize the existing run-artifact reservation so `e2e` can reserve a
`run_id` before measurement when `--power-trace` is active. Add the following
reserved path:

```text
<results-root>/power/<run_id>.power.csv
```

The writer creates a same-directory hidden temporary file with exclusive
creation. It writes the header immediately, buffers rows during sampling, and
flushes at phase transitions. On stop it flushes, fsyncs, closes, and publishes
the final filename with the repository's existing no-overwrite artifact
semantics. It then computes SHA-256 from the published file.

No partially written file may appear at the final path. If a run terminates or
publication fails, a recoverable partial artifact may be retained under a
`.partial.csv` diagnostic name, but it is never reported as a complete trace.
Power publication failure does not delete or overwrite a previously published
artifact.

An ordinary `e2e` run without `--power-trace` keeps its existing result-store
path and run-ID behavior. Async reservation behavior remains compatible with
its existing details and request-trace artifacts.

## Result Linkage and Status

The normal benchmark result stores provenance only:

- `power_trace_status`;
- `power_trace_path`;
- `power_trace_sha256`;
- `power_trace_sample_count`;
- `power_monitor_source`;
- `power_scope`.

It does not store power averages or energy values on behalf of
`--power-trace`. Paths are relative to the results root and must resolve to the
artifact reserved for the same `run_id`. `power_trace_sample_count` is the
number of data rows in the published CSV, including unsuccessful attempts.
For `disabled`, `unavailable`, and `failed`, the final trace path, SHA-256, and
sample count are empty because no complete final artifact was published. A
diagnostic `.partial.csv` is not placed in `power_trace_path`.

Collection status has lifecycle meaning only:

| Status | Meaning |
|---|---|
| `disabled` | `--power-trace` was not requested |
| `complete` | final CSV published and every data row has `sample_status=ok` |
| `partial` | final CSV published, but one or more reads failed or overran |
| `unavailable` | selected target had no verified usable power capability |
| `failed` | trace startup, writing, cleanup, or publication failed |

These statuses do not judge whether a later energy analysis is statistically
valid.

## Failure and Cleanup Semantics

Power telemetry is auxiliary to inference correctness. Telemetry failure must
not erase a successful inference or task-quality result. `HWMonitor` converts
ordinary power-read exceptions to bounded diagnostic rows and continues
sampling. Writer or collector-lifecycle failures change trace status to
`failed` while the runner preserves the primary benchmark outcome.

If inference itself raises, trace stop and collector cleanup still run. A
secondary trace exception is attached as a diagnostic and never replaces the
primary inference exception. If cleanup ownership is uncertain, the existing
transactional monitor rules retain it for explicit retry rather than silently
dropping the session.

When `--monitor` and `--power-trace` are combined, non-power monitor failures
retain their existing behavior. Only the new power-trace path gains the
non-fatal result-preservation policy described here.

## Verification Strategy

### SDK-free tests

- collector power-capability defaults and unavailable behavior;
- finite, non-negative raw watt validation without unit conversion guesses;
- timestamp midpoint and query-latency calculation with a fake clock;
- monotonic deadline scheduling, jitter, missed slots, and no catch-up burst;
- baseline-to-inference phase transition and boundary attempts;
- synchronous start/stop placement around the existing measured loop;
- async start before first measured submission and stop after terminal
  completion;
- `--power-trace` alone and with `--monitor`;
- collector-call serialization when both modes are active;
- raw CSV escaping, failed-read rows, schema ordering, and sample indices;
- atomic no-overwrite publication, fsync failure, partial artifacts, and
  SHA-256 linkage;
- e2e and async run reservations sharing the same artifact authority;
- inference success preservation after collector or writer failure;
- removal of Rebellions and Mobilint energy integration and energy-summary
  fields;
- no model-name or TTM-R2 dependency in monitor modules.

### Real-device acceptance

For each of Furiosa RNGD, Rebellions, and Mobilint ARIES:

1. record tool/package, driver, firmware, device identity, source, and scope;
2. execute 100 idle power queries and record latency and value-change cadence;
3. select or revise the target sampling period from that evidence;
4. run five alternating trace-off and trace-on smoke pairs and require the
   trace-on median measured-loop latency to remain within 2% of trace-off;
5. run the canonical TTM-R2 240-window benchmark with `--power-trace`;
6. confirm both phases, monotonic timing, raw W values, final SHA-256, and
   result linkage;
7. run one existing non-TTM workload supported by that target to demonstrate
   that trace collection is model-independent.

Real-device acceptance validates collection and provenance only. It does not
calculate or approve an energy metric.

## Documentation Changes

Update the framework CLI and vendor setup documents to state:

- `--power-trace` is opt-in and separate from `--monitor`;
- warmup is excluded and the three-second baseline is explicit;
- the CSV stores raw W samples and query timing only;
- each vendor's `power_scope` differs and must be retained in later analysis;
- energy and `J/inference` are not benchmark outputs;
- unavailable and failed telemetry are never replaced with zero.

Existing Rebellions and Mobilint documentation that describes
`hw_accel_energy_j` must be removed or replaced with the raw trace contract.

## Completion Criteria

The implementation is complete when:

1. all SDK-free power-trace tests pass;
2. existing monitor behavior remains compatible apart from the explicitly
   removed energy fields;
3. both inference modes publish correctly linked raw trace artifacts;
4. trace failure cannot convert a successful benchmark into an inference or
   quality failure;
5. no benchmark path calculates joules or `J/inference`;
6. server probes fix the explicit Furiosa API binding and confirm sampling
   configuration for all three initial targets;
7. TTM-R2 and one non-TTM workload per available target produce physically
   collected raw power traces.

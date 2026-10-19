import { expect, test } from 'vitest'
import { metricDescription, metricGroup, numericMetric, recordedStatus } from './webMetrics'

const assert = { equal: (actual: unknown, expected: unknown) => expect(actual).toBe(expected) }

test('metric labels keep recorded measurement boundaries', () => {
  assert.equal(metricDescription('Average Latency (ms)', 'e2e').boundary, 'runtime_call')
  assert.equal(metricDescription('async_runtime_completed_samples_per_sec', 'async_queue').boundary,
    'framework_runtime_call_return')
})

test('selected metric reads only its exact stored key', () => {
  const metrics = { 'Average Latency (ms)': 10, 'Samples/s': 12 }
  assert.equal(numericMetric(metrics, 'Samples/s'), 12)
  assert.equal(numericMetric(metrics, 'FPS'), null)
  assert.equal(numericMetric({ FPS: 'NaN' }, 'FPS'), null)
})

test('runtime-only status stays separate from quality values', () => {
  const row = { metrics: { quality_status: 'skipped', comparison_eligible: 'False' }, metadata: { async_run_status: 'valid' } }
  assert.equal(recordedStatus(row, 'quality_status'), 'skipped')
  assert.equal(recordedStatus(row, 'async_run_status'), 'valid')
  assert.equal(metricGroup('quality_status'), 'status')
  assert.equal(metricGroup('async_runtime_completed_samples_per_sec'), 'timing')
  assert.equal(metricGroup('hw_power_w'), 'hardware')
  assert.equal(metricGroup('COCO AP50'), 'quality')
  assert.equal(metricGroup('Top-5 Accuracy'), 'quality')
})

test('selected edge power metrics show their measured boundary and system input scope', () => {
  assert.equal(metricDescription('throughput_samples_per_sec', 'async_queue').boundary,
    'framework_runtime_call_return')
  assert.equal(metricDescription('throughput_samples_per_sec', 'e2e').boundary,
    'system_e2e_after_warmup_including_evaluator_compute')
  assert.equal(metricDescription('incremental_j_per_sample', 'e2e').unit, 'J/sample')
  assert.equal(metricDescription('incremental_j_per_sample', 'e2e').boundary,
    'jetson_dc_input · system_e2e_after_warmup_including_evaluator_compute')
  assert.equal(metricGroup('incremental_j_per_sample'), 'hardware')
})

test('async latency diagnostics stay in timing with distinct boundaries', () => {
  assert.equal(metricGroup('async_queue_wait_p99_ms'), 'timing')
  assert.equal(metricDescription('async_queue_wait_p99_ms', 'async_queue').boundary,
    'enqueued_to_runtime_started')
  assert.equal(metricDescription('async_queue_wait_p99_ms', 'async_queue').unit, 'ms')
  assert.equal(metricDescription('async_service_time_p99_ms', 'async_queue').boundary,
    'worker_executor_execute')
})

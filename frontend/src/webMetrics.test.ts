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
})

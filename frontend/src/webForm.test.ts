import { expect, test } from 'vitest'
import { buildBenchmarkRequest } from './webForm'

const assert = { deepEqual: (actual: unknown, expected: unknown) => expect(actual).toEqual(expected),
  throws: (fn: () => unknown, pattern: RegExp) => expect(fn).toThrow(pattern) }

test('omitted form fields preserve CLI defaults including Hailo layout', () => {
  assert.deepEqual(buildBenchmarkRequest({
    model: 'yolov8s-pose', target_id: 'hailo8', layout: '',
    monitor: false, compile: 'auto', batch_size: '', inference_mode: 'e2e',
  }), { model: 'yolov8s-pose', target_id: 'hailo8' })
})

test('async request forwards typed values and drops stale e2e values', () => {
  assert.deepEqual(buildBenchmarkRequest({
    model: 'resnet50', target_id: 'cpu', inference_mode: 'async_queue',
    scenario: 'server_like', target_qps: '5.5', worker_count: '2',
    max_steps: '9', compile: 'off', monitor: true,
    runtime_options: 'output_format_type=uint8\nfoo=bar',
  }), {
    model: 'resnet50', target_id: 'cpu', inference_mode: 'async_queue',
    scenario: 'server_like', target_qps: 5.5, worker_count: 2,
    compile: false, monitor: true,
    runtime_options: { output_format_type: 'uint8', foo: 'bar' },
  })
})

test('e2e drops hidden async fields and rejects malformed numeric input', () => {
  assert.deepEqual(buildBenchmarkRequest({
    model: 'resnet50', target_id: 'cpu', inference_mode: 'e2e',
    worker_count: '4', target_qps: '8', max_steps: '2',
  }), { model: 'resnet50', target_id: 'cpu', max_steps: 2 })
  assert.throws(() => buildBenchmarkRequest({model: 'resnet50', batch_size: '2x'}), /batch_size/)
})

test('monitor interval is omitted when monitoring is off', () => {
  assert.deepEqual(buildBenchmarkRequest({ model: 'resnet50', target_id: 'cpu', monitor: false,
    monitor_interval: '0.25' }), { model: 'resnet50', target_id: 'cpu' })
})

test('offline async request drops a hidden stale target QPS', () => {
  assert.deepEqual(buildBenchmarkRequest({ model: 'resnet50', inference_mode: 'async_queue',
    scenario: 'offline', target_qps: '42' }),
  { model: 'resnet50', inference_mode: 'async_queue', scenario: 'offline' })
})

test('model path kind is omitted when the model path is cleared', () => {
  assert.deepEqual(buildBenchmarkRequest({ model: 'resnet50', model_path_kind: 'local' }),
    { model: 'resnet50' })
})

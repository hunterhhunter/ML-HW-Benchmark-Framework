export interface MetricDescription {
  label: string
  unit: string
  boundary: string
}

const known: Record<string, MetricDescription> = {
  'Average Latency (ms)': { label: '평균 Runtime 지연', unit: 'ms', boundary: 'runtime_call' },
  'P99 Latency (ms)': { label: 'P99 Runtime 지연', unit: 'ms', boundary: 'runtime_call' },
  'Samples/s': { label: '기록된 Samples/s', unit: 'samples/s', boundary: 'evaluator_defined' },
  FPS: { label: '기록된 FPS', unit: 'FPS', boundary: 'evaluator_defined' },
  'Top-1 Accuracy': { label: 'Top-1 Accuracy', unit: '%', boundary: 'quality_evaluation' },
  'mAP@0.5': { label: 'mAP@0.5', unit: '', boundary: 'quality_evaluation' },
  'Mask mAP': { label: 'Mask mAP', unit: '', boundary: 'quality_evaluation' },
  'OKS mAP': { label: 'OKS mAP', unit: '', boundary: 'quality_evaluation' },
  MAE: { label: 'MAE', unit: '', boundary: 'quality_evaluation' },
  MSE: { label: 'MSE', unit: '', boundary: 'quality_evaluation' },
  RMSE: { label: 'RMSE', unit: '', boundary: 'quality_evaluation' },
  async_runtime_completed_samples_per_sec: {
    label: '비동기 완료 처리량', unit: 'samples/s', boundary: 'framework_runtime_call_return',
  },
  'Avg TTFT (ms)': { label: '평균 TTFT', unit: 'ms', boundary: 'generation_timing_source 확인' },
  'P99 TTFT (ms)': { label: 'P99 TTFT', unit: 'ms', boundary: 'generation_timing_source 확인' },
  'Avg TPOT (ms)': { label: '평균 TPOT', unit: 'ms', boundary: 'generation_timing_source 확인' },
  'P99 TPOT (ms)': { label: 'P99 TPOT', unit: 'ms', boundary: 'generation_timing_source 확인' },
  'Avg TTFT estimate (ms)': { label: '평균 TTFT 추정', unit: 'ms', boundary: 'estimated_from_total' },
  'P99 TTFT estimate (ms)': { label: 'P99 TTFT 추정', unit: 'ms', boundary: 'estimated_from_total' },
  'Avg TPOT estimate (ms)': { label: '평균 TPOT 추정', unit: 'ms', boundary: 'estimated_from_total' },
  'P99 TPOT estimate (ms)': { label: 'P99 TPOT 추정', unit: 'ms', boundary: 'estimated_from_total' },
}

export function metricDescription(key: string, mode: string): MetricDescription {
  if (known[key]) return known[key]
  if (key.startsWith('reported_')) {
    return { label: `${key.slice('reported_'.length)} (보고서)`, unit: key.endsWith(' %') ? '%' : '', boundary: 'report_summary' }
  }
  const edgeBoundary = mode === 'async_queue' ? 'framework_runtime_call_return'
    : mode === 'e2e' ? 'system_e2e_after_warmup_including_evaluator_compute' : '미확인'
  if (key === 'throughput_samples_per_sec') {
    return { label: '선택 회차 처리량', unit: 'samples/s', boundary: edgeBoundary }
  }
  if (key === 'incremental_j_per_sample') {
    return { label: '시스템 증가 에너지', unit: 'J/sample', boundary: `jetson_dc_input · ${edgeBoundary}` }
  }
  if (['idle_median_w', 'inference_mean_w', 'incremental_mean_w'].includes(key)) {
    return { label: key, unit: 'W', boundary: `jetson_dc_input · ${edgeBoundary}` }
  }
  if (mode === 'async_queue' && key.startsWith('async_runtime_ready_latency_')) {
    return { label: key, unit: 'ms', boundary: 'request_issued_to_runtime_return' }
  }
  if (mode === 'async_queue' && key.startsWith('async_service_time_')) {
    return { label: key, unit: 'ms', boundary: 'worker_executor_execute' }
  }
  if (mode === 'async_queue' && key.startsWith('async_queue_wait_')) {
    return { label: key, unit: 'ms', boundary: 'enqueued_to_runtime_started' }
  }
  return { label: key, unit: '', boundary: '미확인' }
}

export function numericMetric(metrics: Record<string, unknown>, key: string): number | null {
  const value = metrics[key]
  if (typeof value === 'number') return Number.isFinite(value) ? value : null
  if (typeof value !== 'string' || !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim())) {
    return null
  }
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

const statusKeys = new Set([
  'async_run_status', 'async_invalid_reasons', 'quality_status', 'quality_error',
  'comparison_eligible', 'power_trace_status', 'power_scope',
  'async_metric_schema_version', 'async_measurement_boundary', 'generation_timing_source',
  'power_evidence_status', 'measurement_boundary',
])
const qualityKeys = new Set([
  'Top-1 Accuracy', 'accuracy', 'mAP@0.5', 'Mask mAP', 'Mask AP50', 'OKS mAP',
  'OKS AP50', 'Exact Match', 'F1 Score', 'exact_match', 'f1', 'MAE', 'MSE', 'RMSE',
])
const timingKeys = new Set([
  'Average Latency (ms)', 'P50 Latency (ms)', 'P95 Latency (ms)',
  'P99 Latency (ms)', 'Samples/s', 'FPS', 'TTFT (ms)', 'TPOT (ms)',
  'async_runtime_completed_samples_per_sec',
  'throughput_samples_per_sec',
  'Avg TTFT (ms)', 'P99 TTFT (ms)', 'Avg TPOT (ms)', 'P99 TPOT (ms)',
  'Avg TTFT estimate (ms)', 'P99 TTFT estimate (ms)',
  'Avg TPOT estimate (ms)', 'P99 TPOT estimate (ms)',
  'Avg Decode Step (no KV cache) (ms)', 'P99 Decode Step (no KV cache) (ms)',
])

export type MetricGroup = 'status' | 'quality' | 'timing' | 'hardware' | 'raw'

export function metricGroup(key: string): MetricGroup {
  if (key.startsWith('reported_') && key !== 'reported_quality_secondary_metric') return 'quality'
  if (statusKeys.has(key)) return 'status'
  if (qualityKeys.has(key)) return 'quality'
  if (timingKeys.has(key) || key.startsWith('async_runtime_') || key.startsWith('async_service_time_') || key.startsWith('async_queue_wait_')) return 'timing'
  if (key.startsWith('hw_') || ['idle_median_w', 'inference_mean_w', 'incremental_mean_w', 'incremental_j_per_sample'].includes(key)) return 'hardware'
  return 'raw'
}

export function recordedStatus(row: {
  metrics: Record<string, unknown>; metadata: Record<string, string>
}, key: string): string | null {
  const value = row.metadata[key] || row.metrics[key]
  return value === undefined || value === null || value === '' ? null : String(value)
}
